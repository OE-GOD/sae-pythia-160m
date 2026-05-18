"""Compare SAEs trained at multiple widths (4k / 16k / 64k).

Produces:
  1. Pareto table: width vs (MSE, var_explained, L0, dead_features)
  2. Feature-splitting analysis: pick a feature in 16k SAE, find its closest
     match in 4k SAE (likely 1) and in 64k SAE (likely multiple — "splits")
  3. Save results to data/width_comparison.json

Usage:
    python 08_width_comparison.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-4k", type=str, default="checkpoints/sae_layer6_topk64_w4k.pt")
    p.add_argument("--ckpt-16k", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--ckpt-64k", type=str, default="checkpoints/sae_layer6_topk64_w64k.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json",
                   help="labeled features in 16k SAE (from auto-interp)")
    p.add_argument("--split-thresh", type=float, default=0.7,
                   help="cosine similarity threshold for counting as a 'split'")
    p.add_argument("--demo-n", type=int, default=10,
                   help="number of labeled features to show splitting demo for")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="data/width_comparison.json")
    return p.parse_args()


def load_sae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    k = ckpt["k"]
    sae = TopKSAE(d_model, n_features, k=k).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae, ckpt


def decoder_unit_norm(W_dec: torch.Tensor) -> torch.Tensor:
    """Normalize decoder columns to unit length. Returns shape (d_model, n_features)."""
    return W_dec / W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    # ----------------------------------------------------------------------
    # Load all three SAEs and their summary metrics
    # ----------------------------------------------------------------------
    print("loading three SAEs...")
    sae_4k, _ = load_sae(args.ckpt_4k, device)
    sae_16k, _ = load_sae(args.ckpt_16k, device)
    sae_64k, _ = load_sae(args.ckpt_64k, device)
    print(f"  4k:  d_model={sae_4k.d_model}  n_features={sae_4k.n_features}  k={sae_4k.k}")
    print(f"  16k: d_model={sae_16k.d_model} n_features={sae_16k.n_features} k={sae_16k.k}")
    print(f"  64k: d_model={sae_64k.d_model} n_features={sae_64k.n_features} k={sae_64k.k}")

    # Load eval metrics if present
    pareto = {}
    for name, ckpt_path in [("4k", args.ckpt_4k), ("16k", args.ckpt_16k), ("64k", args.ckpt_64k)]:
        eval_path = Path(ckpt_path).with_suffix(".eval.json")
        if eval_path.exists():
            pareto[name] = json.loads(eval_path.read_text())
        else:
            pareto[name] = {"warning": f"no eval file at {eval_path}"}

    # ----------------------------------------------------------------------
    # Pareto table
    # ----------------------------------------------------------------------
    print("\n=== PARETO TABLE ===")
    print(f"{'width':<6} {'n_feat':<8} {'MSE':<10} {'var_exp':<10} {'L0':<8} {'dead %':<8} {'alive':<8}")
    for name, info in pareto.items():
        if "warning" in info:
            print(f"  {name:<6}  {info['warning']}")
            continue
        n_feat = sae_4k.n_features if name == "4k" else (sae_16k.n_features if name == "16k" else sae_64k.n_features)
        print(f"  {name:<6} {n_feat:<8} {info['mse']:<10.3f} {info['var_explained']:<10.4f} "
              f"{info['L0_mean']:<8.1f} {info['pct_dead']:<8.2f} {info['n_alive']:<8}")

    # ----------------------------------------------------------------------
    # Feature splitting analysis
    # ----------------------------------------------------------------------
    print("\n=== FEATURE SPLITTING ===")
    print("For each labeled 16k feature, find:")
    print("  - 1 closest match in 4k (cos sim)")
    print(f"  - N matches in 64k with cos sim > {args.split_thresh} (the 'splits')")
    print()

    # Decoder columns as unit-norm vectors (d_model, n_features)
    W_4k = decoder_unit_norm(sae_4k.W_dec.data)
    W_16k = decoder_unit_norm(sae_16k.W_dec.data)
    W_64k = decoder_unit_norm(sae_64k.W_dec.data)

    # Load labeled features
    catalog_path = Path(args.catalog)
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        labeled_fids = [int(k) for k in catalog.keys() if "error" not in catalog[k]]
        print(f"  loaded {len(labeled_fids)} labeled features from {catalog_path}")
    else:
        # Fall back to peak-active features
        ckpt_16k = torch.load(args.ckpt_16k, map_location="cpu", weights_only=False)
        peak_idx = ckpt_16k["model_state"]["W_dec"].abs().sum(0).argsort(descending=True)[:50].tolist()
        labeled_fids = peak_idx
        catalog = {str(fid): {"label": "(unlabeled)"} for fid in labeled_fids}
        print(f"  no catalog; using top-50 peak features")

    demo_fids = labeled_fids[: args.demo_n]
    print(f"\nShowing splits for {len(demo_fids)} labeled features:\n")

    splits = {}
    for fid in demo_fids:
        v_16k = W_16k[:, fid]  # (d_model,)
        sims_4k = v_16k @ W_4k  # (4096,)
        sims_64k = v_16k @ W_64k  # (65536,)

        # Closest in 4k
        best_4k_idx = int(sims_4k.argmax().item())
        best_4k_sim = float(sims_4k.max().item())

        # All matches in 64k above threshold
        match_64k_mask = sims_64k > args.split_thresh
        match_64k_idx = torch.nonzero(match_64k_mask).flatten().tolist()
        match_64k_sims = sims_64k[match_64k_mask].tolist()
        # Sort by similarity descending
        match_64k = sorted(zip(match_64k_idx, match_64k_sims), key=lambda x: -x[1])

        label = catalog[str(fid)].get("label", "?") if str(fid) in catalog else "?"
        splits[fid] = {
            "label_16k": label,
            "best_4k_idx": best_4k_idx,
            "best_4k_sim": best_4k_sim,
            "n_64k_splits": len(match_64k),
            "top_5_64k_splits": match_64k[:5],
        }

        print(f"  f{fid} ({label[:50]})")
        print(f"    4k:  best match f{best_4k_idx} cos={best_4k_sim:.3f}")
        print(f"    64k: {len(match_64k)} features above cos>{args.split_thresh}")
        if match_64k:
            top5 = [f"f{i}(cos={s:.2f})" for i, s in match_64k[:5]]
            print(f"         top: {', '.join(top5)}")

    # ----------------------------------------------------------------------
    # Aggregate splitting statistics
    # ----------------------------------------------------------------------
    print("\n=== SPLIT-COUNT DISTRIBUTION OVER {} LABELED FEATURES ===".format(len(demo_fids)))
    split_counts = [splits[fid]["n_64k_splits"] for fid in demo_fids]
    if split_counts:
        print(f"  mean splits per 16k feature in 64k SAE: {np.mean(split_counts):.2f}")
        print(f"  median: {np.median(split_counts):.0f}")
        print(f"  max:    {np.max(split_counts)}")
        print(f"  features with 1 split (no splitting):    {sum(1 for c in split_counts if c == 1)}")
        print(f"  features with 2-3 splits:                {sum(1 for c in split_counts if 2 <= c <= 3)}")
        print(f"  features with 4+ splits:                 {sum(1 for c in split_counts if c >= 4)}")

    # Save
    results = {
        "pareto": pareto,
        "split_threshold": args.split_thresh,
        "demo_splits": {str(k): v for k, v in splits.items()},
        "split_count_stats": {
            "mean": float(np.mean(split_counts)) if split_counts else None,
            "median": float(np.median(split_counts)) if split_counts else None,
            "max": int(np.max(split_counts)) if split_counts else None,
            "n_evaluated": len(demo_fids),
        },
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
