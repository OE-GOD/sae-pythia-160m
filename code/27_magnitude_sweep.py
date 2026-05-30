"""Magnitude sweep for SAE feature interventions.

For each feature that survived re-firing (script 26), sweep through multiple
intervention magnitudes (relative to the feature's natural f_clean) and
measure the resulting concept-logit-diff.

This answers: at what magnitude does each feature become a "driver"?

Three possible patterns per feature:
  - Sharp threshold: no effect until magnitude crosses some point, then driver
  - Linear ramp: concept-logit-diff scales linearly with magnitude
  - Flat: no effect at any magnitude (true thermometer)

The shape of the curve is itself the publishable finding.
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

MAGNITUDE_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]


def classify_label(label):
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
    p.add_argument("--context_window", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_top_firing_positions(sae, feature_id, acts, device,
                                top_k=10, sample_size=100000, batch_size=8192):
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
    return [(int(sample_idx[lidx]), float(feature_activations[lidx])) for lidx in top_local_idx]


def get_context_around_position(token_stream, position, window):
    start = max(0, position - window + 1)
    return token_stream[start:position + 1]


def classify_curve(magnitudes, logit_diffs, sharp_thresh=0.5, flat_thresh=0.1):
    """Classify the shape of the magnitude → logit_diff curve."""
    arr = np.array(logit_diffs)
    if np.all(np.abs(arr) < flat_thresh):
        return "flat (true thermometer)"
    # Check for sharp threshold: most action between two consecutive magnitudes
    diffs = np.diff(arr)
    if np.max(diffs) > 0.5 * (arr.max() - arr.min()) and arr.max() > sharp_thresh:
        return "sharp threshold (binary driver)"
    # Otherwise linear-ish
    if arr.max() > sharp_thresh:
        return "linear ramp (graded driver)"
    return "weak / borderline"


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

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

    print(f"loading activations from {args.acts}...")
    acts = np.load(args.acts)
    print(f"loading tokens from {args.tokens}...")
    token_stream = np.load(args.tokens)

    catalog = json.loads(Path(args.catalog).read_text())
    classified = []
    for fid_str, info in catalog.items():
        if "error" in info or info.get("classification") != "MONOSEMANTIC":
            continue
        cat = classify_label(info.get("label", ""))
        if cat is None:
            continue
        classified.append((int(fid_str), info, cat))
    print(f"\n{len(classified)} categorizable monosemantic features")
    print(f"magnitude multipliers to sweep: {MAGNITUDE_MULTIPLIERS}\n")

    results = []
    t0 = time.time()

    for i, (fid, info, cat) in enumerate(classified):
        label = info["label"]
        print(f"\n[{i+1:>2}/{len(classified)}] f{fid}  [{cat['name']}]  {label[:50]}")

        # Find a context where the feature re-fires
        candidates = find_top_firing_positions(sae, fid, acts, device, top_k=10)
        f_clean = 0.0
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
            if cand_f > 0.5:
                f_clean = cand_f
                break
        if f_clean < 0.5:
            print(f"  WARN: no candidate re-fired; skipping")
            continue
        print(f"  natural f_clean: {f_clean:.3f}")

        # Concept tokens
        vocab_size = model.cfg.d_vocab
        concept_token_ids = []
        for tid in range(vocab_size):
            s = model.tokenizer.decode([tid])
            if cat["token_match"](s):
                concept_token_ids.append(tid)
        if not concept_token_ids:
            print(f"  WARN: no concept tokens; skipping")
            continue

        decoder_col_X = sae.W_dec[:, fid].detach()

        # For each magnitude, average across prompts
        magnitude_results = []
        for mult in MAGNITUDE_MULTIPLIERS:
            intervention_magnitude = mult * f_clean
            prompt_diffs = []
            for prompt in CORRUPTED_PROMPTS:
                corrupted_tokens = model.to_tokens(prompt).to(device)
                with torch.no_grad():
                    corrupted_logits = model(corrupted_tokens)
                corrupted_concept = corrupted_logits[0, -1, concept_token_ids].mean().item()

                def patching_hook(activation, hook):
                    activation[:, -1, :] = activation[:, -1, :] + intervention_magnitude * decoder_col_X
                    return activation

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens, fwd_hooks=[(hook_name, patching_hook)]
                    )
                patched_concept = patched_logits[0, -1, concept_token_ids].mean().item()
                prompt_diffs.append(patched_concept - corrupted_concept)
            mean_diff = float(np.mean(prompt_diffs))
            magnitude_results.append({
                "multiplier": mult,
                "absolute_magnitude": intervention_magnitude,
                "mean_concept_logit_diff": mean_diff,
            })
            print(f"  mult {mult:>4.1f}x  (magnitude {intervention_magnitude:>6.2f})  "
                  f"logit_diff = {mean_diff:+.3f}")

        magnitudes = [r["multiplier"] for r in magnitude_results]
        logit_diffs = [r["mean_concept_logit_diff"] for r in magnitude_results]
        curve_shape = classify_curve(magnitudes, logit_diffs)
        print(f"  curve shape: {curve_shape}")

        results.append({
            "feature_id": fid,
            "label": label,
            "category": cat["name"],
            "f_clean": f_clean,
            "magnitude_sweep": magnitude_results,
            "curve_shape": curve_shape,
        })

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"CURVE-SHAPE CLASSIFICATION (n={len(results)})")
    print(f"{'=' * 70}")
    by_shape = {}
    for r in results:
        by_shape[r["curve_shape"]] = by_shape.get(r["curve_shape"], 0) + 1
    for shape, count in by_shape.items():
        print(f"  {shape:<35}: {count} features")

    # Per-feature table
    print(f"\n{'=' * 90}")
    print("PER-FEATURE MAGNITUDE SWEEP — concept-logit-diff at each multiplier")
    print(f"{'=' * 90}")
    header = "  fid    " + "  ".join(f"{m:>5.1f}x" for m in MAGNITUDE_MULTIPLIERS) + "   shape"
    print(header)
    for r in results:
        diffs = "  ".join(f"{m['mean_concept_logit_diff']:>+6.2f}" for m in r["magnitude_sweep"])
        print(f"  f{r['feature_id']:>5}  {diffs}  {r['curve_shape']}")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".magnitude_sweep.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "magnitude_multipliers": MAGNITUDE_MULTIPLIERS,
        "n_features_tested": len(results),
        "curve_shape_counts": by_shape,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
