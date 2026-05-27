"""Test the driver/thermometer distinction across all labeled features.

For each labeled monosemantic feature, we:
  1. Map its auto-interp label to a set of "predicted concept tokens"
     (e.g., a "Newlines" feature should produce newline-containing tokens).
  2. Compute its logit weight for those tokens (cheap, no model needed).
  3. Run residual-stream steering at alpha = peak_activation.
  4. Count predicted-concept tokens in baseline vs steered output.
  5. Classify as driver / thermometer / ambiguous based on the delta.

Aggregate result: population-level rates of drivers vs thermometers, plus
correlation between logit weight and steering effect. This upgrades the
n=4 anecdote from 19_causal_intervention.py to n=26 population claim.

Usage:
    python 22_thermometer_at_scale.py
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


# --- Label-to-token mapping ---
# Each entry: detect this label keyword, then check whether a token matches
# the concept. Token strings come from model.tokenizer.decode([tid]).
LABEL_CATEGORIES = [
    {
        "name": "newline",
        "label_keywords": ["newline"],
        "token_match": lambda s: "\n" in s,
    },
    {
        "name": "punctuation_formatting",
        "label_keywords": ["punctuation", "formatting"],
        "token_match": lambda s: bool(re.fullmatch(r"[\s\.,;:!?\-()\[\]{}\"'\\/]+", s)) and len(s) > 0,
    },
    {
        "name": "file_path",
        "label_keywords": ["file path", "directory", "path"],
        "token_match": lambda s: ("/" in s) or ("\\" in s) or
                                  any(s.rstrip().endswith(ext) for ext in
                                      [".py", ".js", ".html", ".css", ".cpp", ".java", ".md", ".txt"]),
    },
    {
        "name": "french",
        "label_keywords": ["french"],
        "token_match": lambda s: any(c in s.lower() for c in "àâäçéèêëîïôöùûüÿ") or
                                  any(w in s.lower() for w in [" la ", " le ", " du ", " rue ", " des "]),
    },
    {
        "name": "math_notation",
        "label_keywords": ["math", "equation", "mathematical", "exponent"],
        "token_match": lambda s: any(c in s for c in "=+-*/^∑∏∫π√") and len(s) <= 5,
    },
    {
        "name": "citation_latex",
        "label_keywords": ["bibtex", "latex", "citation"],
        "token_match": lambda s: any(t in s for t in ["\\cite", "\\ref", "\\bibitem", "@", "et al"]),
    },
    {
        "name": "logical_operators",
        "label_keywords": ["logical operator", "negation"],
        "token_match": lambda s: s.strip() in ["!", "not", "&&", "||", "!=", "==", "&", "|"],
    },
    {
        "name": "code_keyword",
        "label_keywords": ["def keyword", "function definition", "code definition", "class"],
        "token_match": lambda s: s.strip() in ["def", "class", "function", "return", "import", "from"],
    },
    {
        "name": "decimal_numerical",
        "label_keywords": ["decimal", "numerical"],
        "token_match": lambda s: bool(re.fullmatch(r"[\d\.\-]+", s.strip())) and len(s.strip()) > 0,
    },
    {
        "name": "subword_bpe",
        "label_keywords": ["subword", "bpe", "continuation"],
        "token_match": lambda s: len(s) >= 2 and not s.startswith(" ") and s[0].islower(),
    },
    {
        "name": "abbreviation_acronym",
        "label_keywords": ["acronym", "abbreviation"],
        "token_match": lambda s: bool(re.fullmatch(r"[A-Z]{2,5}\.?", s.strip())),
    },
]


def classify_label(label: str):
    """Return the category dict whose label_keywords match this label, or None."""
    label_lower = label.lower()
    for cat in LABEL_CATEGORIES:
        if any(kw in label_lower for kw in cat["label_keywords"]):
            return cat
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--max-new-tokens", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha-mult", type=float, default=1.0)
    p.add_argument("--n-prompts", type=int, default=3, help="How many neutral prompts per feature")
    p.add_argument("--driver-threshold", type=float, default=3.0,
                   help="Δ predicted-concept tokens above this = driver")
    p.add_argument("--thermometer-threshold", type=float, default=0.5,
                   help="Δ below this = thermometer")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


NEUTRAL_PROMPTS = [
    "The recipe for chocolate cake is",
    "My favorite color is",
    "Yesterday I went to the park and saw",
]


def make_hook(decoder_col, alpha):
    delta = (alpha * decoder_col).to(torch.float32)
    def hook(activation, hook):
        return activation + delta.to(activation.dtype)
    return hook


@torch.no_grad()
def generate_with_hook(model, prompt, hook_fn, hook_name, max_new, temp):
    tokens = model.to_tokens(prompt)
    if hook_fn is None:
        out = model.generate(tokens, max_new_tokens=max_new, temperature=temp,
                             do_sample=True, verbose=False)
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(tokens, max_new_tokens=max_new, temperature=temp,
                                 do_sample=True, verbose=False)
    return model.to_string(out[0])


def count_concept_tokens(text: str, token_match_fn, tokenizer):
    """Tokenize text, count how many tokens match the concept."""
    # Simple version: just check the decoded text token-by-token.
    # For speed, tokenize and decode each token id.
    ids = tokenizer.encode(text)
    count = 0
    for tid in ids:
        s = tokenizer.decode([tid])
        if token_match_fn(s):
            count += 1
    return count


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    sae = TopKSAE(d_model, ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    # --- Load model ---
    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"loading {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    W_U = model.W_U.detach()

    # --- Load catalog and classify features by label ---
    catalog = json.loads(Path(args.catalog).read_text())
    classified = []
    uncategorized = []
    for fid_str, info in catalog.items():
        if "error" in info:
            continue
        if info.get("classification") != "MONOSEMANTIC":
            continue
        cat = classify_label(info.get("label", ""))
        if cat is None:
            uncategorized.append((int(fid_str), info))
        else:
            classified.append((int(fid_str), info, cat))

    print(f"\nclassified {len(classified)} features into categories:")
    by_cat = {}
    for fid, info, cat in classified:
        by_cat.setdefault(cat["name"], []).append((fid, info["label"]))
    for cn, fs in by_cat.items():
        print(f"  {cn}: {len(fs)} features")
        for fid, lbl in fs[:3]:
            print(f"    f{fid}  {lbl[:50]}")
    print(f"\nuncategorized: {len(uncategorized)} features (skipped)")

    # --- Pre-generate baselines for each prompt (saves time) ---
    prompts = NEUTRAL_PROMPTS[: args.n_prompts]
    print(f"\ngenerating baselines for {len(prompts)} prompts...")
    baselines = {}
    for p in prompts:
        torch.manual_seed(args.seed)
        baselines[p] = generate_with_hook(model, p, None, hook_name,
                                          args.max_new_tokens, args.temperature)
        print(f"  {p!r} -> {baselines[p][:60]!r}")

    # --- Run steering for each classified feature ---
    results = []
    t0 = time.time()
    for i, (fid, info, cat) in enumerate(classified):
        peak = info["peak_activation"]
        alpha = args.alpha_mult * peak
        decoder_col = sae.W_dec[:, fid].detach().clone()

        # Logit weight for predicted-concept tokens
        logit_w = (decoder_col @ W_U).cpu().numpy()
        median_w = float(np.median(logit_w))
        vocab_size = model.cfg.d_vocab
        concept_token_ids = []
        for tid in range(vocab_size):
            s = model.tokenizer.decode([tid])
            if cat["token_match"](s):
                concept_token_ids.append(tid)
        if not concept_token_ids:
            max_concept_w = 0.0
            sum_concept_w = 0.0
        else:
            concept_weights = logit_w[concept_token_ids] - median_w
            max_concept_w = float(concept_weights.max())
            sum_concept_w = float(concept_weights.sum())

        # Steering on each prompt
        deltas = []
        steered_outputs = []
        for prompt in prompts:
            torch.manual_seed(args.seed)
            steered = generate_with_hook(model, prompt, make_hook(decoder_col, alpha),
                                          hook_name, args.max_new_tokens, args.temperature)
            steered_outputs.append(steered)
            base_count = count_concept_tokens(baselines[prompt], cat["token_match"],
                                              model.tokenizer)
            steer_count = count_concept_tokens(steered, cat["token_match"], model.tokenizer)
            deltas.append(steer_count - base_count)

        mean_delta = float(np.mean(deltas))

        # Classify driver / thermometer / ambiguous
        if mean_delta >= args.driver_threshold:
            verdict = "driver"
        elif mean_delta <= args.thermometer_threshold:
            verdict = "thermometer"
        else:
            verdict = "ambiguous"

        results.append({
            "feature_id": fid,
            "label": info["label"],
            "category": cat["name"],
            "peak_activation": peak,
            "alpha": alpha,
            "n_concept_tokens_in_vocab": len(concept_token_ids),
            "max_logit_weight_concept": max_concept_w,
            "sum_logit_weight_concept": sum_concept_w,
            "mean_delta_concept_tokens": mean_delta,
            "deltas_per_prompt": deltas,
            "verdict": verdict,
        })

        print(f"  [{i+1:>2}/{len(classified)}] f{fid:>5}  [{cat['name']:>20}]  "
              f"max_w={max_concept_w:.2f}  Δ={mean_delta:>+5.1f}  -> {verdict.upper()}")

    elapsed = time.time() - t0
    print(f"\nfeature pass complete in {elapsed:.1f}s")

    # --- Aggregate summary ---
    print(f"\n{'=' * 70}")
    print("POPULATION-LEVEL RESULTS")
    print(f"{'=' * 70}")
    by_verdict = {"driver": 0, "thermometer": 0, "ambiguous": 0}
    for r in results:
        by_verdict[r["verdict"]] += 1
    total = len(results)
    print(f"  Total classified: {total}")
    for v, n in by_verdict.items():
        pct = 100 * n / total if total > 0 else 0
        print(f"    {v:>15}: {n:>3} ({pct:>5.1f}%)")

    # Correlation: logit weight vs delta
    if len(results) >= 3:
        logw = np.array([r["max_logit_weight_concept"] for r in results])
        delta = np.array([r["mean_delta_concept_tokens"] for r in results])
        if logw.std() > 0 and delta.std() > 0:
            corr = float(np.corrcoef(logw, delta)[0, 1])
            print(f"\n  Pearson(max_logit_weight, Δconcept_tokens) = {corr:.3f}")
            print(f"  (higher correlation = logit weight predicts driver behavior)")

    # Per-category breakdown
    print(f"\n{'=' * 70}")
    print("BY CATEGORY")
    print(f"{'=' * 70}")
    cats_seen = sorted({r["category"] for r in results})
    for cn in cats_seen:
        rs = [r for r in results if r["category"] == cn]
        d = sum(1 for r in rs if r["verdict"] == "driver")
        t = sum(1 for r in rs if r["verdict"] == "thermometer")
        a = sum(1 for r in rs if r["verdict"] == "ambiguous")
        print(f"  {cn:>22}: {len(rs):>3} features  drivers={d}  thermo={t}  amb={a}")

    # Save
    out_path = args.out or str(Path(args.ckpt).with_suffix(".thermometer_at_scale.json"))
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "alpha_mult": args.alpha_mult,
        "driver_threshold": args.driver_threshold,
        "thermometer_threshold": args.thermometer_threshold,
        "n_prompts": len(prompts),
        "n_features_classified": len(results),
        "n_features_uncategorized": len(uncategorized),
        "by_verdict": by_verdict,
        "summary_corr_logitweight_delta": float(corr) if len(results) >= 3 else None,
        "results": results,
        "uncategorized_features": [{"feature_id": fid, "label": info["label"]}
                                    for fid, info in uncategorized],
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
