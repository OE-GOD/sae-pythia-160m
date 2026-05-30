"""SAE-feature-only patching for driver/thermometer classification.

Whole-residual patching (script 25) over-attributes effects to a feature
because it transplants the entire residual stream — including all co-firing
features. This script isolates the intervention to ONE feature.

Method:
  For each labeled feature X:
    1. Find clean position where feature X fires hard.
    2. Compute clean residual at that position; encode through SAE to get
       feature X's activation in the clean context: f_X^clean.
    3. For each corrupted prompt, get the corrupted residual at the final
       position; encode through SAE to get f_X^corrupted.
    4. Compute the intervention as a delta on the residual:
         delta = (f_X^clean - f_X^corrupted) * W_dec[:, X]
    5. Patch: replace corrupted residual at final position with
       corrupted_resid + delta.
    6. Measure mean logit diff for concept tokens between patched and
       corrupted runs.

This is the principled middle ground between:
  - Synthetic steering (script 22): alpha is arbitrary, may be too aggressive
  - Whole-residual patching (script 25): contaminated by other co-firing features

Three-way comparison (steering vs whole-residual patching vs SAE-feature
patching) is the methodological contribution.
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


LABEL_CATEGORIES = [
    {"name": "newline", "label_keywords": ["newline"],
     "token_match": lambda s: "\n" in s},
    {"name": "punctuation_formatting", "label_keywords": ["punctuation", "formatting"],
     "token_match": lambda s: bool(re.fullmatch(r"[\s\.,;:!?\-()\[\]{}\"'\\/]+", s)) and len(s) > 0},
    {"name": "file_path", "label_keywords": ["file path", "directory", "path"],
     "token_match": lambda s: ("/" in s) or ("\\" in s) or
                              any(s.rstrip().endswith(ext) for ext in
                                  [".py", ".js", ".html", ".css", ".cpp", ".java", ".md", ".txt"])},
    {"name": "french", "label_keywords": ["french"],
     "token_match": lambda s: any(c in s.lower() for c in "àâäçéèêëîïôöùûüÿ") or
                              any(w in s.lower() for w in [" la ", " le ", " du ", " rue ", " des "])},
    {"name": "math_notation", "label_keywords": ["math", "equation", "mathematical", "exponent"],
     "token_match": lambda s: any(c in s for c in "=+-*/^∑∏∫π√") and len(s) <= 5},
    {"name": "citation_latex", "label_keywords": ["bibtex", "latex", "citation"],
     "token_match": lambda s: any(t in s for t in ["\\cite", "\\ref", "\\bibitem", "@", "et al"])},
    {"name": "logical_operators", "label_keywords": ["logical operator", "negation"],
     "token_match": lambda s: s.strip() in ["!", "not", "&&", "||", "!=", "==", "&", "|"]},
    {"name": "code_keyword", "label_keywords": ["def keyword", "function definition", "code definition", "class"],
     "token_match": lambda s: s.strip() in ["def", "class", "function", "return", "import", "from"]},
    {"name": "decimal_numerical", "label_keywords": ["decimal", "numerical"],
     "token_match": lambda s: bool(re.fullmatch(r"[\d\.\-]+", s.strip())) and len(s.strip()) > 0},
    {"name": "subword_bpe", "label_keywords": ["subword", "bpe", "continuation"],
     "token_match": lambda s: len(s) >= 2 and not s.startswith(" ") and s[0].islower()},
    {"name": "abbreviation_acronym", "label_keywords": ["acronym", "abbreviation"],
     "token_match": lambda s: bool(re.fullmatch(r"[A-Z]{2,5}\.?", s.strip()))},
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
    p.add_argument("--context_window", type=int, default=30)
    p.add_argument("--driver_threshold", type=float, default=0.5)
    p.add_argument("--thermometer_threshold", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_top_firing_positions(sae, feature_id, acts, device, top_k=10,
                                sample_size=100000, batch_size=8192):
    """Find the top-k positions where this feature fires hardest.

    Encodes in batches to avoid OOM on MPS.
    Returns list of (global_position, activation) sorted by activation descending.
    """
    n_total = acts.shape[0]
    if n_total > sample_size:
        sample_idx = np.random.choice(n_total, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n_total)

    feature_activations = np.zeros(len(sample_idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sample_idx), batch_size):
            batch = sample_idx[start:start + batch_size]
            batch_acts = torch.from_numpy(acts[batch].astype(np.float32)).to(device)
            f = sae.encode(batch_acts)
            feature_activations[start:start + len(batch)] = f[:, feature_id].cpu().numpy()
            del batch_acts, f
            if device == "mps":
                torch.mps.empty_cache()

    top_local_idx = np.argsort(-feature_activations)[:top_k]
    out = []
    for lidx in top_local_idx:
        out.append((int(sample_idx[lidx]), float(feature_activations[lidx])))
    return out


def get_context_around_position(token_stream, position, window):
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

    # --- Load data ---
    print(f"loading activations from {args.acts}...")
    acts = np.load(args.acts)
    print(f"  shape: {acts.shape}")
    print(f"loading tokens from {args.tokens}...")
    token_stream = np.load(args.tokens)

    # --- Load catalog ---
    catalog = json.loads(Path(args.catalog).read_text())
    classified = []
    for fid_str, info in catalog.items():
        if "error" in info or info.get("classification") != "MONOSEMANTIC":
            continue
        cat = classify_label(info.get("label", ""))
        if cat is None:
            continue
        classified.append((int(fid_str), info, cat))
    print(f"\n{len(classified)} categorizable monosemantic features\n")

    # --- Main loop ---
    results = []
    t0 = time.time()

    for i, (fid, info, cat) in enumerate(classified):
        label = info["label"]
        print(f"\n[{i+1:>2}/{len(classified)}] f{fid}  [{cat['name']}]  {label[:50]}")

        # 1. Find top-10 firing positions; try each until one re-fires.
        candidates = find_top_firing_positions(sae, fid, acts, device, top_k=10)
        f_clean = 0.0
        best_pos = None
        best_act = None
        clean_resid = None
        for cand_pos, cand_act in candidates:
            cand_tokens = get_context_around_position(token_stream, cand_pos, args.context_window)
            cand_tokens_tensor = torch.tensor([cand_tokens.tolist()], dtype=torch.long, device=device)
            with torch.no_grad():
                _, cand_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
            cand_resid = cand_cache[hook_name][:, -1, :]
            with torch.no_grad():
                cand_features = sae.encode(cand_resid)
            cand_f = cand_features[0, fid].item()
            if cand_f > 0.5:  # threshold for "actually firing"
                f_clean = cand_f
                best_pos = cand_pos
                best_act = cand_act
                clean_resid = cand_resid
                break
        if clean_resid is None:
            print(f"  WARN: no candidate position re-fired (tried {len(candidates)}); skipping")
            continue
        print(f"  re-fired at pos {best_pos}; original activation {best_act:.3f}; "
              f"re-encoded f_X={f_clean:.3f}")

        # 4. Concept tokens
        vocab_size = model.cfg.d_vocab
        concept_token_ids = []
        for tid in range(vocab_size):
            s = model.tokenizer.decode([tid])
            if cat["token_match"](s):
                concept_token_ids.append(tid)
        if not concept_token_ids:
            print(f"  WARN: no concept tokens; skipping")
            continue

        # 5. Decoder column for feature X
        decoder_col_X = sae.W_dec[:, fid].detach()  # (d_model,)

        # 6. For each corrupted prompt: get corrupted resid, compute f_X^corrupted,
        #    then patch with delta = (f_X^clean - f_X^corrupted) * decoder_col_X
        per_prompt_results = []
        for prompt in CORRUPTED_PROMPTS:
            corrupted_tokens = model.to_tokens(prompt).to(device)
            with torch.no_grad():
                corrupted_logits = model(corrupted_tokens)
                _, corrupted_cache = model.run_with_cache(
                    corrupted_tokens, names_filter=[hook_name]
                )
            corrupted_resid_at_last = corrupted_cache[hook_name][:, -1, :]
            with torch.no_grad():
                corrupted_features = sae.encode(corrupted_resid_at_last)
            f_corrupted = corrupted_features[0, fid].item()

            delta_magnitude = f_clean - f_corrupted
            corrupted_concept_logits = corrupted_logits[0, -1, concept_token_ids].mean().item()

            def patching_hook(activation, hook):
                activation[:, -1, :] = activation[:, -1, :] + delta_magnitude * decoder_col_X
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, patching_hook)],
                )
                patched_concept_logits = patched_logits[0, -1, concept_token_ids].mean().item()

            per_prompt_results.append({
                "prompt": prompt,
                "f_corrupted": f_corrupted,
                "delta_magnitude": delta_magnitude,
                "corrupted_mean_concept_logit": corrupted_concept_logits,
                "patched_mean_concept_logit": patched_concept_logits,
                "logit_diff": patched_concept_logits - corrupted_concept_logits,
            })

        mean_logit_diff = float(np.mean([r["logit_diff"] for r in per_prompt_results]))
        mean_delta = float(np.mean([r["delta_magnitude"] for r in per_prompt_results]))

        if mean_logit_diff >= args.driver_threshold:
            verdict = "driver"
        elif mean_logit_diff <= args.thermometer_threshold:
            verdict = "thermometer"
        else:
            verdict = "ambiguous"

        print(f"  delta magnitude (mean): {mean_delta:+.3f}")
        print(f"  mean concept-logit diff: {mean_logit_diff:+.3f}  -> {verdict.upper()}")

        results.append({
            "feature_id": fid,
            "label": label,
            "category": cat["name"],
            "f_clean": f_clean,
            "top_firing_activation": best_act,
            "n_concept_tokens": len(concept_token_ids),
            "per_prompt_results": per_prompt_results,
            "mean_delta": mean_delta,
            "mean_logit_diff": mean_logit_diff,
            "verdict": verdict,
        })

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # --- Three-way comparison ---
    print(f"\n{'=' * 80}")
    print("THREE-WAY COMPARISON: STEERING vs WHOLE-RESID PATCHING vs SAE-FEATURE PATCHING")
    print(f"{'=' * 80}")
    steering_path = Path("checkpoints/sae_layer6_topk64_full.thermometer_at_scale.json")
    whole_patch_path = Path("checkpoints/sae_layer6_topk64_full.patching_driver_thermometer.json")
    steering_verdicts = {}
    whole_verdicts = {}
    if steering_path.exists():
        d = json.loads(steering_path.read_text())
        steering_verdicts = {r["feature_id"]: r["verdict"] for r in d["results"]}
    if whole_patch_path.exists():
        d = json.loads(whole_patch_path.read_text())
        whole_verdicts = {r["feature_id"]: r["verdict"] for r in d["results"]}

    print(f"\n  {'fid':>6} {'label':<35} {'steering':>11} {'whole-patch':>13} {'feat-patch':>12}")
    print(f"  {'-'*6:>6} {'-'*35:<35} {'-'*11:>11} {'-'*13:>13} {'-'*12:>12}")
    for r in results:
        fid = r["feature_id"]
        s = steering_verdicts.get(fid, "?")
        w = whole_verdicts.get(fid, "?")
        f = r["verdict"]
        print(f"  f{fid:>5} {r['label'][:35]:<35} {s:>11} {w:>13} {f:>12}")

    # Rates
    n = len(results)
    f_drivers = sum(1 for r in results if r["verdict"] == "driver")
    f_therms = sum(1 for r in results if r["verdict"] == "thermometer")
    f_amb = sum(1 for r in results if r["verdict"] == "ambiguous")

    print(f"\n{'=' * 80}")
    print(f"DRIVER RATES (n={n})")
    print(f"{'=' * 80}")
    if steering_verdicts:
        s_drivers = sum(1 for r in results if steering_verdicts.get(r["feature_id"]) == "driver")
        print(f"  Steering:                   {s_drivers:>3}/{n} ({100*s_drivers/n:.1f}%)")
    if whole_verdicts:
        w_drivers = sum(1 for r in results if whole_verdicts.get(r["feature_id"]) == "driver")
        print(f"  Whole-residual patching:    {w_drivers:>3}/{n} ({100*w_drivers/n:.1f}%)")
    print(f"  SAE-feature patching (new): {f_drivers:>3}/{n} ({100*f_drivers/n:.1f}%)")

    print(f"\n  SAE-feature patching breakdown:")
    print(f"    driver:      {f_drivers}")
    print(f"    thermometer: {f_therms}")
    print(f"    ambiguous:   {f_amb}")

    # Save
    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_patching.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "sae_feature_patching",
        "driver_threshold": args.driver_threshold,
        "thermometer_threshold": args.thermometer_threshold,
        "n_features": n,
        "by_verdict": {"driver": f_drivers, "thermometer": f_therms, "ambiguous": f_amb},
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
