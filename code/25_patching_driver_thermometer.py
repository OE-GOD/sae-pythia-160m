"""Activation-patching version of the driver/thermometer classifier.

Replaces the synthetic steering intervention (alpha * decoder_column) from
scripts 22/24 with a more rigorous activation-patching protocol following
the IOI-style methodology (Wang et al., Nanda 2023).

Procedure for each labeled feature:
  1. Find a "clean" context where the feature fires hard, using the existing
     activation cache.
  2. Pick a baseline "corrupted" prompt where the feature is silent.
  3. Run the model on the clean context, save the residual stream at the
     final token position of layer 6.
  4. Run the model on the corrupted prompt, but patch in the clean residual
     at the final token position.
  5. Compare patched logits to corrupted logits over predicted-concept tokens.
  6. Classify driver / thermometer / ambiguous.

Compared to script 22 (steering with alpha * decoder_column), this:
  - Uses real residual-state activations (not synthetic injections)
  - Stays in-distribution
  - Provides a cleaner causal interpretation
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


# Reuse the same label-category mapping as script 22 for consistency.
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


CORRUPTED_PROMPTS = [
    "The recipe for chocolate cake is",
    "My favorite color is",
    "Yesterday I went to the park and saw",
]


def classify_label(label: str):
    label_lower = label.lower()
    for cat in LABEL_CATEGORIES:
        if any(kw in label_lower for kw in cat["label_keywords"]):
            return cat
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--tokens", type=str, default="data/token_stream.npy")
    p.add_argument("--context_window", type=int, default=30,
                   help="How many tokens of context to take ending at the firing position")
    p.add_argument("--driver_threshold", type=float, default=0.5,
                   help="Mean concept logit-diff above this => driver")
    p.add_argument("--thermometer_threshold", type=float, default=0.05,
                   help="Mean concept logit-diff below this => thermometer")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_top_firing_position(
    sae: TopKSAE,  # noqa: F821
    feature_id: int,
    acts: np.ndarray,
    device: str,
    sample_size: int = 50000,
) -> int:
    """Find the token position (index into the activation cache) where this
    feature fires most strongly. Returns the global index.
    """
    # Sample to keep things tractable
    n_total = acts.shape[0]
    if n_total > sample_size:
        sample_idx = np.random.choice(n_total, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n_total)

    sample_acts = torch.from_numpy(acts[sample_idx].astype(np.float32)).to(device)
    with torch.no_grad():
        f = sae.encode(sample_acts)  # (sample_size, n_features)
        feature_activations = f[:, feature_id].cpu().numpy()

    best_local = int(np.argmax(feature_activations))
    best_activation = float(feature_activations[best_local])
    best_global = int(sample_idx[best_local])
    return best_global, best_activation


def get_context_around_position(
    token_stream: np.ndarray,
    position: int,
    window: int,
) -> np.ndarray:
    """Return the `window` tokens ending at position (inclusive)."""
    start = max(0, position - window + 1)
    return token_stream[start:position + 1]


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    sae = TopKSAE(d_model, ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"loading {model_name}, hook={hook_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # --- Load activations and token stream ---
    print(f"loading activations from {args.acts}...")
    acts = np.load(args.acts)
    print(f"  shape: {acts.shape}")
    print(f"loading token stream from {args.tokens}...")
    token_stream = np.load(args.tokens)
    print(f"  shape: {token_stream.shape}")
    assert acts.shape[0] == token_stream.shape[0], \
        f"act/token length mismatch: {acts.shape[0]} vs {token_stream.shape[0]}"

    # --- Load catalog ---
    catalog = json.loads(Path(args.catalog).read_text())
    classified = []
    for fid_str, info in catalog.items():
        if "error" in info:
            continue
        if info.get("classification") != "MONOSEMANTIC":
            continue
        cat = classify_label(info.get("label", ""))
        if cat is None:
            continue
        classified.append((int(fid_str), info, cat))
    print(f"\n{len(classified)} categorizable monosemantic features\n")

    # --- For each feature, run the patching experiment ---
    results = []
    t0 = time.time()

    for i, (fid, info, cat) in enumerate(classified):
        label = info["label"]
        print(f"\n[{i+1:>2}/{len(classified)}] f{fid}  [{cat['name']}]  {label[:50]}")

        # 1. Find top firing position
        best_pos, best_act = find_top_firing_position(sae, fid, acts, device)
        print(f"  top firing: position {best_pos}, activation {best_act:.3f}")

        # 2. Build the clean context (window of tokens ending at firing position)
        clean_tokens = get_context_around_position(token_stream, best_pos, args.context_window)
        clean_tokens_tensor = torch.tensor([clean_tokens.tolist()], dtype=torch.long, device=device)
        clean_text = model.tokenizer.decode(clean_tokens.tolist())
        print(f"  clean context (last 50 chars): ...{clean_text[-50:]!r}")

        # 3. Build pre-computed concept-token IDs for this feature's category
        vocab_size = model.cfg.d_vocab
        concept_token_ids = []
        for tid in range(vocab_size):
            s = model.tokenizer.decode([tid])
            if cat["token_match"](s):
                concept_token_ids.append(tid)
        if not concept_token_ids:
            print(f"  WARN: no concept tokens for category {cat['name']}; skipping")
            continue

        # 4. Run clean, save resid at final position
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_tokens_tensor, names_filter=[hook_name]
            )
        clean_resid_at_last = clean_cache[hook_name][:, -1, :].clone()  # (1, d_model)

        # 5. For each corrupted prompt, run baseline + patched, compute concept logit diff
        per_prompt_results = []
        for prompt in CORRUPTED_PROMPTS:
            corrupted_tokens = model.to_tokens(prompt).to(device)

            with torch.no_grad():
                corrupted_logits = model(corrupted_tokens)
                corrupted_concept_logits = corrupted_logits[0, -1, concept_token_ids].mean().item()

            def patching_hook(activation, hook):
                activation[:, -1, :] = clean_resid_at_last
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, patching_hook)],
                )
                patched_concept_logits = patched_logits[0, -1, concept_token_ids].mean().item()

            logit_diff = patched_concept_logits - corrupted_concept_logits
            per_prompt_results.append({
                "prompt": prompt,
                "corrupted_mean_concept_logit": corrupted_concept_logits,
                "patched_mean_concept_logit": patched_concept_logits,
                "logit_diff": logit_diff,
            })

        mean_logit_diff = float(np.mean([r["logit_diff"] for r in per_prompt_results]))

        # 6. Classify
        if mean_logit_diff >= args.driver_threshold:
            verdict = "driver"
        elif mean_logit_diff <= args.thermometer_threshold:
            verdict = "thermometer"
        else:
            verdict = "ambiguous"

        print(f"  mean concept-logit diff (patched - corrupted): {mean_logit_diff:+.3f}  -> {verdict.upper()}")

        results.append({
            "feature_id": fid,
            "label": label,
            "category": cat["name"],
            "top_firing_position": best_pos,
            "top_firing_activation": best_act,
            "n_concept_tokens": len(concept_token_ids),
            "per_prompt_results": per_prompt_results,
            "mean_logit_diff": mean_logit_diff,
            "verdict": verdict,
        })

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # --- Compare to script 22 (steering) verdicts ---
    print(f"\n{'=' * 70}")
    print("PATCHING vs STEERING COMPARISON")
    print(f"{'=' * 70}")
    steering_path = Path("checkpoints/sae_layer6_topk64_full.thermometer_at_scale.json")
    if steering_path.exists():
        steering_data = json.loads(steering_path.read_text())
        steering_verdicts = {r["feature_id"]: r["verdict"] for r in steering_data["results"]}
        print(f"  {'fid':>6} {'label':<40} {'steering':>12} {'patching':>12} {'agree':>6}")
        agree = 0
        for r in results:
            sv = steering_verdicts.get(r["feature_id"], "?")
            pv = r["verdict"]
            mark = "✓" if sv == pv else "✗"
            if sv == pv:
                agree += 1
            print(f"  f{r['feature_id']:>5} {r['label'][:40]:<40} {sv:>12} {pv:>12} {mark:>6}")
        print(f"\n  Agreement: {agree}/{len(results)} ({100*agree/len(results):.1f}%)")
    else:
        print("  (steering results not found for comparison)")

    # --- Population-level summary ---
    print(f"\n{'=' * 70}")
    print("PATCHING-BASED CLASSIFICATION SUMMARY")
    print(f"{'=' * 70}")
    by_verdict = {"driver": 0, "thermometer": 0, "ambiguous": 0}
    for r in results:
        by_verdict[r["verdict"]] += 1
    n = len(results)
    for v, c in by_verdict.items():
        print(f"  {v:>12}: {c:>3} ({100*c/n:.1f}%)")

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".patching_driver_thermometer.json"))
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "context_window": args.context_window,
        "driver_threshold": args.driver_threshold,
        "thermometer_threshold": args.thermometer_threshold,
        "n_features": n,
        "by_verdict": by_verdict,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
