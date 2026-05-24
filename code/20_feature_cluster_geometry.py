"""Test whether a group of SAE features are pieces of one underlying direction.

Motivation: Build #4 showed that "unstable" newline-variant features have huge
CE-ablation impact (+13.82 nats/token on active tokens), while supposedly
"stable" features have negligible impact (+0.003). Hypothesis: the SAE has
chopped one important underlying direction (newline-ness) into N nearly-parallel
features. Test by computing pairwise cosine similarity between the decoder
columns of the suspected cluster.

If pairwise cosine similarities are mostly high (> 0.5), the features are
pieces of one direction. If they're mostly low, they're independent features
that happen to share a label.

Usage:
    # Auto-find all "newline" features by label substring
    python 20_feature_cluster_geometry.py --label-contains newline

    # Or specify exact feature IDs
    python 20_feature_cluster_geometry.py --features 2255,7448,12117
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
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--label-contains", type=str, default=None,
                   help="Pick features whose label contains this substring (case-insensitive)")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature IDs (overrides --label-contains)")
    p.add_argument("--compare-random", type=int, default=20,
                   help="Sample N random features as a baseline distribution")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def select_features(catalog, label_substr, feature_ids):
    if feature_ids:
        ids = [int(x) for x in feature_ids.split(",")]
        return [(fid, catalog.get(str(fid), {}).get("label", "?")) for fid in ids]
    chosen = []
    for fid_str, info in catalog.items():
        label = info.get("label", "")
        if label_substr.lower() in label.lower():
            chosen.append((int(fid_str), label))
    return chosen


def cosine_matrix(vectors):
    """Compute pairwise cosine similarity. vectors shape (n, d). Returns (n, n)."""
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-9)
    unit = vectors / norms
    return (unit @ unit.T).cpu().numpy()


def summarize_matrix(mat, label):
    n = mat.shape[0]
    off_diag = mat[~np.eye(n, dtype=bool)]
    print(f"\n  {label}:  n={n}  pairs={off_diag.size}")
    print(f"    mean   cos = {off_diag.mean():.4f}")
    print(f"    median cos = {np.median(off_diag):.4f}")
    print(f"    min    cos = {off_diag.min():.4f}")
    print(f"    max    cos = {off_diag.max():.4f}")
    print(f"    fraction > 0.5 = {(off_diag > 0.5).mean():.2%}")
    print(f"    fraction > 0.8 = {(off_diag > 0.8).mean():.2%}")
    print(f"    fraction > 0.9 = {(off_diag > 0.9).mean():.2%}")
    return {
        "n": int(n),
        "n_pairs": int(off_diag.size),
        "mean": float(off_diag.mean()),
        "median": float(np.median(off_diag)),
        "min": float(off_diag.min()),
        "max": float(off_diag.max()),
        "frac_gt_0_5": float((off_diag > 0.5).mean()),
        "frac_gt_0_8": float((off_diag > 0.8).mean()),
        "frac_gt_0_9": float((off_diag > 0.9).mean()),
    }


def print_matrix(mat, feature_ids, labels, max_cells=20):
    n = mat.shape[0]
    if n > max_cells:
        print(f"\n  (matrix too large to print, showing first {max_cells}×{max_cells})")
        n = max_cells
    print(f"\n  cosine similarity matrix:")
    header = "        " + "  ".join(f"f{fid:>5}" for fid in feature_ids[:n])
    print(header)
    for i in range(n):
        row = "  f{:>5}  ".format(feature_ids[i]) + "  ".join(
            f"{mat[i,j]:>6.3f}" for j in range(n)
        )
        print(row)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    k_active = ckpt["k"]
    sae = TopKSAE(d_model, n_features, k=k_active).to(args.device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    print(f"  d_model={d_model}  n_features={n_features}")

    # --- Pick the cluster ---
    catalog = json.loads(Path(args.catalog).read_text())
    if not args.features and not args.label_contains:
        raise SystemExit("Pass --features or --label-contains")
    cluster = select_features(catalog, args.label_contains, args.features)
    if not cluster:
        raise SystemExit(f"No features matched (label='{args.label_contains}')")

    cluster_ids = [fid for fid, _ in cluster]
    cluster_labels = [lbl for _, lbl in cluster]
    print(f"\nselected cluster: {len(cluster_ids)} features")
    for fid, lbl in cluster:
        print(f"  f{fid:>5}  {lbl}")

    # --- Extract decoder columns ---
    # W_dec shape: (d_model, n_features). Column i is the direction of feature i.
    W_dec = sae.W_dec.detach()
    cluster_dirs = W_dec[:, cluster_ids].T  # (n_cluster, d_model)
    print(f"\ndecoder column norms (should all be ~1.0 after training normalization):")
    norms = cluster_dirs.norm(dim=1).cpu().numpy()
    print(f"  min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")

    # --- Cluster cosine matrix ---
    cluster_cos = cosine_matrix(cluster_dirs)
    cluster_stats = summarize_matrix(cluster_cos, "CLUSTER")
    print_matrix(cluster_cos, cluster_ids, cluster_labels)

    # --- Random baseline ---
    if args.compare_random > 0:
        # Sample N random features that are NOT in the cluster
        all_ids = set(range(n_features))
        for fid in cluster_ids:
            all_ids.discard(fid)
        random_ids = np.random.choice(sorted(all_ids), size=min(args.compare_random, len(all_ids)),
                                      replace=False).tolist()
        random_dirs = W_dec[:, random_ids].T
        random_cos = cosine_matrix(random_dirs)
        random_stats = summarize_matrix(random_cos, "RANDOM BASELINE")
    else:
        random_stats = None
        random_ids = []

    # --- Cross-cluster baseline: each cluster feature vs each random feature ---
    if random_ids:
        cross = (cluster_dirs / cluster_dirs.norm(dim=1, keepdim=True).clamp(min=1e-9)) @ \
                (W_dec[:, random_ids] / W_dec[:, random_ids].norm(dim=0, keepdim=True).clamp(min=1e-9))
        cross = cross.cpu().numpy().flatten()
        print(f"\n  CROSS (cluster ↔ random):  pairs={cross.size}")
        print(f"    mean   cos = {cross.mean():.4f}")
        print(f"    median cos = {np.median(cross):.4f}")
        cross_stats = {
            "n_pairs": int(cross.size),
            "mean": float(cross.mean()),
            "median": float(np.median(cross)),
        }
    else:
        cross_stats = None

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    mean_cluster = cluster_stats["mean"]
    mean_random = random_stats["mean"] if random_stats else float("nan")
    if mean_cluster > 0.5:
        verdict = "STRONG CLUSTER — these features are nearly parallel."
        verdict += "\n  Strong evidence the SAE chopped one underlying direction into pieces."
    elif mean_cluster > 0.2:
        verdict = "WEAK CLUSTER — features are correlated but not parallel."
        verdict += "\n  The 'cluster' likely contains both a shared direction and independent variation."
    else:
        verdict = "NO CLUSTER — features are essentially orthogonal despite shared label."
        verdict += "\n  The label is misleading; these are independent features the auto-interp grouped together."
    print(verdict)
    print(f"\n  cluster mean cos:  {mean_cluster:.4f}")
    if random_stats:
        print(f"  random mean cos:   {mean_random:.4f}")
        print(f"  ratio:             {mean_cluster / max(abs(mean_random), 1e-6):.2f}x")
    print(f"{'=' * 70}")

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".cluster_geometry.json"))
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "label_contains": args.label_contains,
        "feature_ids": cluster_ids,
        "feature_labels": cluster_labels,
        "decoder_norms": norms.tolist(),
        "cluster_cosine_matrix": cluster_cos.tolist(),
        "cluster_stats": cluster_stats,
        "random_baseline_stats": random_stats,
        "cross_cluster_random_stats": cross_stats,
        "verdict": verdict,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
