"""Measure the effective rank of a feature cluster's decoder subspace via SVD.

If the cluster of newline features really lives in a low-dimensional manifold,
their decoder columns should span only a few directions in residual space. SVD
gives a clean quantitative answer:

  - Stack N decoder columns as rows of a matrix M (shape N x d_model).
  - Compute singular values s_1 >= s_2 >= ... >= s_N.
  - If a few s_i dominate, the cluster has low effective rank (a manifold).
  - If all s_i are similar, the cluster spans all N dimensions (no manifold structure).

Effective rank metrics:
  - r_95: number of singular values needed to explain 95% of squared variance
  - r_99: same threshold at 99%
  - "spectral entropy" or "participation ratio" as continuous measures

Compared against a random baseline: N random unit vectors in d_model dimensions
will have full rank N with roughly uniform singular values.
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
    p.add_argument("--label-contains", type=str, default="newline")
    p.add_argument("--n-random", type=int, default=16,
                   help="size of random baseline cluster (should match real cluster)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def effective_rank_metrics(singular_values: np.ndarray) -> dict:
    """Compute multiple effective-rank measures from singular values."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 1e-12]
    var = s ** 2
    total = var.sum()
    cum_frac = np.cumsum(var) / total

    # r_X: how many sing values to capture X% of variance
    r_50 = int(np.searchsorted(cum_frac, 0.50) + 1)
    r_75 = int(np.searchsorted(cum_frac, 0.75) + 1)
    r_90 = int(np.searchsorted(cum_frac, 0.90) + 1)
    r_95 = int(np.searchsorted(cum_frac, 0.95) + 1)
    r_99 = int(np.searchsorted(cum_frac, 0.99) + 1)

    # Spectral entropy (in bits)
    p = var / total
    entropy_bits = float(-(p * np.log2(p + 1e-15)).sum())

    # Participation ratio: (sum s^2)^2 / sum s^4 — continuous "effective dim"
    pr = float((s ** 2).sum() ** 2 / ((s ** 4).sum() + 1e-15))

    return {
        "r_50": r_50,
        "r_75": r_75,
        "r_90": r_90,
        "r_95": r_95,
        "r_99": r_99,
        "spectral_entropy_bits": entropy_bits,
        "participation_ratio": pr,
        "n_total": len(s),
    }


def main():
    args = parse_args()
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sae = TopKSAE(ckpt["d_model"], ckpt["n_features"], k=ckpt["k"]).to(args.device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    print(f"  d_model={d_model}  n_features={n_features}")

    # --- Find cluster ---
    catalog = json.loads(Path(args.catalog).read_text())
    cluster = [(int(fid_str), info.get("label", "?"))
               for fid_str, info in catalog.items()
               if args.label_contains.lower() in info.get("label", "").lower()]
    cluster_ids = [fid for fid, _ in cluster]
    print(f"\nfound {len(cluster_ids)} features matching '{args.label_contains}'")
    for fid, lbl in cluster:
        print(f"  f{fid:>5}  {lbl}")

    if len(cluster_ids) < 2:
        raise SystemExit("Need at least 2 features in cluster.")

    # --- SVD on the cluster ---
    # W_dec shape (d_model, n_features). Decoder columns are unit-norm by training.
    W = sae.W_dec.data
    cluster_cols = W[:, cluster_ids]  # (d_model, n_cluster)
    cluster_norms = cluster_cols.norm(dim=0)
    print(f"\ncolumn norms (should all be ~1): min={cluster_norms.min():.4f}  max={cluster_norms.max():.4f}")

    # Run SVD on (n_cluster, d_model) — features as rows
    M = cluster_cols.T.cpu().numpy().astype(np.float64)  # (n_cluster, d_model)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"\nSVD on M of shape {M.shape}:")
    print(f"  number of singular values: {len(S)}")
    print(f"  largest:  {S[0]:.4f}")
    print(f"  smallest: {S[-1]:.4f}")
    print(f"  condition number: {S[0] / max(S[-1], 1e-12):.2f}")

    # --- Effective rank metrics for the cluster ---
    cluster_metrics = effective_rank_metrics(S)
    var_explained = (S ** 2).cumsum() / (S ** 2).sum()

    print(f"\n--- CLUSTER singular value spectrum ---")
    print(f"  i  | singular  |  s_i^2  | cum %")
    print(f"  ---+-----------+----------+------")
    for i, (sv, ve) in enumerate(zip(S, var_explained)):
        print(f"  {i+1:>2} | {sv:>8.4f}  | {sv**2:>7.4f} | {ve*100:>5.1f}%")
    print(f"\nCluster effective rank metrics:")
    for k, v in cluster_metrics.items():
        print(f"  {k}: {v}")

    # --- Random baseline ---
    print(f"\ncomputing random baseline ({args.n_random} random unit vectors)...")
    np.random.seed(0)
    R = np.random.randn(args.n_random, d_model)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    _, S_rand, _ = np.linalg.svd(R, full_matrices=False)
    random_metrics = effective_rank_metrics(S_rand)

    print(f"\n--- RANDOM BASELINE singular value spectrum ---")
    var_explained_rand = (S_rand ** 2).cumsum() / (S_rand ** 2).sum()
    print(f"  i  | singular  |  s_i^2  | cum %")
    print(f"  ---+-----------+----------+------")
    for i, (sv, ve) in enumerate(zip(S_rand, var_explained_rand)):
        print(f"  {i+1:>2} | {sv:>8.4f}  | {sv**2:>7.4f} | {ve*100:>5.1f}%")
    print(f"\nRandom baseline effective rank metrics:")
    for k, v in random_metrics.items():
        print(f"  {k}: {v}")

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print(f"  n_cluster = {len(cluster_ids)}")
    print()
    print(f"  Cluster:  r_90 = {cluster_metrics['r_90']}, r_95 = {cluster_metrics['r_95']}, "
          f"participation_ratio = {cluster_metrics['participation_ratio']:.2f}")
    print(f"  Random:   r_90 = {random_metrics['r_90']}, r_95 = {random_metrics['r_95']}, "
          f"participation_ratio = {random_metrics['participation_ratio']:.2f}")
    print()
    cluster_pr = cluster_metrics["participation_ratio"]
    random_pr = random_metrics["participation_ratio"]
    pr_ratio = cluster_pr / max(random_pr, 1e-9)
    print(f"  Participation-ratio difference: cluster {cluster_pr:.2f} vs random {random_pr:.2f}")
    print(f"  Cluster effective dim is {pr_ratio:.2f}x random's.")
    print()
    if cluster_pr < 0.5 * random_pr:
        verdict = "STRONG MANIFOLD — cluster lives in much smaller subspace than random vectors."
        verdict += f"\n  Effective dimensionality ~{cluster_pr:.1f}, vs {len(cluster_ids)} features available."
    elif cluster_pr < 0.8 * random_pr:
        verdict = "MODERATE MANIFOLD — cluster has some compression but isn't a tight manifold."
    else:
        verdict = "NO MANIFOLD — cluster spans about as many dimensions as random vectors."
        verdict += "\n  The features don't share a low-dimensional subspace; they're nearly independent directions."
    print(verdict)
    print(f"{'=' * 70}")

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".manifold_rank.json"))
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "label_contains": args.label_contains,
        "feature_ids": cluster_ids,
        "feature_labels": [lbl for _, lbl in cluster],
        "cluster_singular_values": S.tolist(),
        "cluster_var_explained_cumulative": var_explained.tolist(),
        "cluster_metrics": cluster_metrics,
        "random_singular_values": S_rand.tolist(),
        "random_metrics": random_metrics,
        "participation_ratio_ratio": float(pr_ratio),
        "verdict": verdict,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
