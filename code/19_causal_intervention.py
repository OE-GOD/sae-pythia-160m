"""Causally validate SAE features by forcing them on during generation.

Standard mech-interp Test 3:
  1. Pick a feature with a hypothesized label (e.g., "file paths").
  2. Pick a neutral prompt (e.g., "The recipe for chocolate cake is").
  3. Run baseline generation — model continues normally.
  4. Run intervened generation — add alpha * decoder_column[feature_i] to the
     residual stream at the SAE's layer during every forward pass.
  5. If the intervened output reflects the feature's hypothesized concept, the
     feature is causally meaningful. If output is unchanged, the feature is a
     "thermometer" (correlational only).

This promotes features from correlationally interpretable (which top-activating
examples already show) to causally validated (the gold standard).

Usage:
    python 19_causal_intervention.py
    python 19_causal_intervention.py --features 2255,6630,12520 --alpha-mults 1,3,10
"""
import argparse
import json
import os
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


# A handful of neutral prompts that should NOT naturally elicit the features we test.
# If feature steering works, the model should drift toward the feature's concept
# even from these unrelated starting points.
NEUTRAL_PROMPTS = [
    "The recipe for chocolate cake is",
    "Yesterday I went to the park and saw",
    "My favorite color is",
    "The weather today is",
    "Once upon a time there was",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json",
                   help="JSON of labeled features from auto-interp")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature IDs to test (default: pick top monosemantic)")
    p.add_argument("--n-features", type=int, default=5,
                   help="How many features to test if --features not given")
    p.add_argument("--alpha-mults", type=str, default="1,3,10",
                   help="Comma-separated multipliers of peak_activation to use as alpha")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def select_features(catalog, n):
    """Pick the top N features that are monosemantic with high/medium confidence."""
    candidates = []
    for fid_str, info in catalog.items():
        if info.get("classification") != "MONOSEMANTIC":
            continue
        if info.get("confidence") not in ("high", "medium"):
            continue
        candidates.append(info)
    # Sort by peak_activation descending (strongest features first)
    candidates.sort(key=lambda x: -x["peak_activation"])
    return candidates[:n]


def make_steering_hook(decoder_col: torch.Tensor, alpha: float):
    """Returns a hook function that adds alpha * decoder_col to the residual stream."""
    delta = (alpha * decoder_col).to(torch.float32)  # (d_model,)

    def hook(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        return activation + delta.to(activation.dtype)

    return hook


@torch.no_grad()
def generate_with_hook(model, prompt: str, hook_fn, hook_name: str,
                       max_new_tokens: int, temperature: float):
    """Generate text from prompt with the given hook installed at hook_name."""
    tokens = model.to_tokens(prompt)
    if hook_fn is None:
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            verbose=False,
        )
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                verbose=False,
            )
    return model.to_string(out[0])


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    k_active = ckpt["k"]
    sae = TopKSAE(d_model, n_features, k=k_active).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    print(f"  d_model={d_model}  n_features={n_features}  k={k_active}")

    # --- Load model ---
    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"\nloading {model_name}, hook={hook_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # --- Load feature catalog and pick features ---
    catalog = json.loads(Path(args.catalog).read_text())
    if args.features:
        chosen_ids = [int(x) for x in args.features.split(",")]
        chosen = [catalog[str(fid)] for fid in chosen_ids if str(fid) in catalog]
    else:
        chosen = select_features(catalog, args.n_features)

    if not chosen:
        raise SystemExit("No features selected — check --features or catalog.")

    print(f"\ntesting {len(chosen)} features:")
    for info in chosen:
        print(f"  f{info['feature_id']:>6}  peak={info['peak_activation']:>7.2f}  "
              f"[{info['confidence']:>6}]  {info['label']}")

    alpha_mults = [float(x) for x in args.alpha_mults.split(",")]
    print(f"alpha multipliers (× peak activation): {alpha_mults}")

    # --- Run interventions ---
    results = []
    t0 = time.time()
    for info in chosen:
        fid = info["feature_id"]
        peak = info["peak_activation"]
        label = info["label"]
        print(f"\n{'=' * 70}")
        print(f"FEATURE f{fid}  —  {label}  (peak activation = {peak:.2f})")
        print(f"{'=' * 70}")

        decoder_col = sae.W_dec[:, fid].detach().clone()  # (d_model,)

        feature_results = {
            "feature_id": fid,
            "label": label,
            "classification": info.get("classification"),
            "confidence": info.get("confidence"),
            "peak_activation": peak,
            "evidence_note": info.get("evidence_note"),
            "decoder_norm": float(decoder_col.norm().item()),
            "generations": [],
        }

        for prompt in NEUTRAL_PROMPTS:
            # Baseline
            torch.manual_seed(args.seed)
            baseline = generate_with_hook(
                model, prompt, None, hook_name,
                args.max_new_tokens, args.temperature,
            )
            print(f"\n  prompt:    {prompt!r}")
            print(f"  baseline:  {baseline!r}")

            for mult in alpha_mults:
                alpha = mult * peak
                hook_fn = make_steering_hook(decoder_col, alpha)
                torch.manual_seed(args.seed)
                steered = generate_with_hook(
                    model, prompt, hook_fn, hook_name,
                    args.max_new_tokens, args.temperature,
                )
                print(f"  α×{mult:>4.1f}:    {steered!r}")
                feature_results["generations"].append({
                    "prompt": prompt,
                    "alpha_mult": mult,
                    "alpha_value": alpha,
                    "baseline": baseline,
                    "steered": steered,
                })

        results.append(feature_results)

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".causal_intervention.json"))
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "model": model_name,
        "hook": hook_name,
        "layer": layer,
        "alpha_mults": alpha_mults,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "n_features_tested": len(chosen),
        "results": results,
    }, indent=2))

    print(f"\n{'=' * 70}")
    print(f"saved {out_path}")
    print(f"total time: {time.time() - t0:.1f}s")
    print(f"\nReview the generations manually. For each feature:")
    print(f"  - If steered outputs reflect the feature's label → causally validated")
    print(f"  - If steered outputs unchanged → 'thermometer' feature (correlational only)")
    print(f"  - If steered outputs are garbage → alpha too high, lower the multiplier")


if __name__ == "__main__":
    main()
