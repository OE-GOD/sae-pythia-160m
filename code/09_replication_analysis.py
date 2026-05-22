"""T4 — Replication study across SAE training seeds.

Train SAEs at the same hyperparameters but with different random seeds. Match
features cross-seed via cosine similarity of decoder columns. Report:

  - How many features have a cross-seed cosine match above each threshold?
  - For our 26 labeled features (data/feature_catalog.json), which replicate?
  - Combine with T1's width-invariant features to produce a
    "stable across width AND seed" subset — the strongest "real feature"
    proxy in our setup.

Output:
  data/replication_analysis.json

Usage:
    python 09_replication_analysis.py
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
    p.add_argument("--ckpt-a", type=str, default="checkpoints/sae_layer6_topk64_full.pt",
                   help="Reference SAE (the W2 run)")
    p.add_argument("--ckpt-b", type=str, default="checkpoints/sae_layer6_topk64_seed42.pt",
                   help="Second-seed SAE")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json",
                   help="Auto-interp catalog of labeled features (T5)")
    p.add_argument("--width-comparison", type=str, default="data/width_comparison.json",
                   help="T1 output — for the joint width+seed analysis")
    p.add_argument("--thresholds", type=str, default="0.5,0.7,0.9,0.95,0.99")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="data/replication_analysis.json")
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


def unit_norm_columns(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"loading SAE A (reference): {args.ckpt_a}")
    sae_a, ckpt_a = load_sae(args.ckpt_a, device)
    print(f"loading SAE B (seed=42):   {args.ckpt_b}")
    sae_b, ckpt_b = load_sae(args.ckpt_b, device)

    assert sae_a.d_model == sae_b.d_model, "d_model mismatch"
    assert sae_a.n_features == sae_b.n_features, "n_features mismatch"
    assert sae_a.k == sae_b.k, "k mismatch"

    W_a = unit_norm_columns(sae_a.W_dec.data)  # (d_model, n_features)
    W_b = unit_norm_columns(sae_b.W_dec.data)

    # ----------------------------------------------------------------------
    # For every feature in A, find its best cosine match in B
    # ----------------------------------------------------------------------
    print("\ncomputing all-pairs cos sim (decoder columns)...")
    # W_a.T @ W_b: (n_features_a, n_features_b)
    sims = W_a.T @ W_b
    best_b_sim, best_b_idx = sims.max(dim=1)
    best_b_sim = best_b_sim.cpu().numpy()
    best_b_idx = best_b_idx.cpu().numpy()

    n_features = sae_a.n_features

    # Distribution of best-match cos sim across all features (alive ones)
    # We define alive as features that have any meaningful decoder column;
    # check the original ckpt for is-alive info via the eval file.
    eval_a_path = Path(args.ckpt_a).with_suffix(".eval.json")
    eval_b_path = Path(args.ckpt_b).with_suffix(".eval.json")
    if eval_a_path.exists():
        eval_a = json.loads(eval_a_path.read_text())
    else:
        eval_a = {}
    if eval_b_path.exists():
        eval_b = json.loads(eval_b_path.read_text())
    else:
        eval_b = {}

    print(f"\n=== SAE eval summaries ===")
    for name, ev in [("A", eval_a), ("B", eval_b)]:
        if ev:
            print(f"  {name}: var_explained={ev.get('var_explained', 'n/a'):.4f}  "
                  f"L0={ev.get('L0_mean', 'n/a'):.1f}  "
                  f"dead={ev.get('pct_dead', 'n/a'):.2f}%")

    # ----------------------------------------------------------------------
    # Histogram of best-match similarities
    # ----------------------------------------------------------------------
    print(f"\n=== CROSS-SEED COSINE SIMILARITY (all {n_features} features) ===")
    print(f"  for each feature in SAE-A, best match in SAE-B's decoder columns:")
    print(f"  threshold   |  count above threshold  |  fraction")
    print(f"  ----------- + ------------------------ + ----------")
    threshold_counts = {}
    for t in thresholds:
        n_above = int((best_b_sim > t).sum())
        frac = n_above / n_features
        print(f"  cos > {t:.2f}  | {n_above:>7,} / {n_features:>7,}    | {frac:.4f}")
        threshold_counts[t] = {"count": n_above, "fraction": frac}

    # ----------------------------------------------------------------------
    # Replication of labeled features (T5)
    # ----------------------------------------------------------------------
    catalog_path = Path(args.catalog)
    labeled_results = {}
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        labeled_fids = [int(k) for k in catalog.keys() if "error" not in catalog[k]]
        print(f"\n=== LABELED-FEATURE REPLICATION (n={len(labeled_fids)}) ===")
        print(f"  for each auto-labeled feature, its best cos-sim match in SAE-B:")
        print(f"  {'fid':<7} {'label':<55} {'best B':<8} {'cos':<6}")
        for fid in sorted(labeled_fids, key=lambda i: -best_b_sim[i]):
            label = catalog[str(fid)].get("label", "?")[:55]
            b_idx = int(best_b_idx[fid])
            sim = float(best_b_sim[fid])
            mark = "★" if sim > 0.9 else ("·" if sim > 0.7 else " ")
            print(f"  f{fid:<6} {label:<55} f{b_idx:<7} {sim:.3f} {mark}")
            labeled_results[fid] = {
                "label": catalog[str(fid)].get("label"),
                "best_b_idx": b_idx,
                "best_b_cos_sim": sim,
                "replicates_at_0.7": sim > 0.7,
                "replicates_at_0.9": sim > 0.9,
            }
        n_rep_0_9 = sum(1 for r in labeled_results.values() if r["replicates_at_0.9"])
        n_rep_0_7 = sum(1 for r in labeled_results.values() if r["replicates_at_0.7"])
        print(f"\n  features replicating at cos > 0.9: {n_rep_0_9}/{len(labeled_fids)}")
        print(f"  features replicating at cos > 0.7: {n_rep_0_7}/{len(labeled_fids)}")
    else:
        print(f"\n(no labeled catalog at {catalog_path}; skipping)")

    # ----------------------------------------------------------------------
    # Joint analysis: stable across BOTH width and seed
    # ----------------------------------------------------------------------
    width_path = Path(args.width_comparison)
    joint = {}
    if width_path.exists() and labeled_fids:
        width = json.loads(width_path.read_text())
        demo_splits = width.get("demo_splits", {})
        print(f"\n=== JOINT WIDTH + SEED STABILITY ===")
        print(f"  for the {len(demo_splits)} features analyzed in T1, are they ALSO seed-stable?")
        print(f"  {'fid':<7} {'4k cos':<8} {'64k cos':<9} {'seed42 cos':<11} {'verdict'}")

        stable_count = 0
        for fid_str, w in demo_splits.items():
            fid = int(fid_str)
            cos_4k = w["best_4k_sim"]
            top5_64k = w.get("top_5_64k_splits", [])
            cos_64k = top5_64k[0][1] if top5_64k else 0.0
            cos_seed = float(best_b_sim[fid]) if fid < n_features else float("nan")
            # Stable if high cos at ALL three (4k, 64k, seed42)
            stable_4k = cos_4k > 0.7
            stable_64k = cos_64k > 0.7
            stable_seed = cos_seed > 0.7
            n_stable = sum([stable_4k, stable_64k, stable_seed])
            if n_stable == 3:
                verdict = "★★★ stable across width + seed"
                stable_count += 1
            elif n_stable == 2:
                verdict = "★★ partial stability"
            elif n_stable == 1:
                verdict = "★ minimal stability"
            else:
                verdict = "unstable"
            print(f"  f{fid:<6} {cos_4k:<8.3f} {cos_64k:<9.3f} {cos_seed:<11.3f} {verdict}")
            joint[fid] = {
                "cos_4k": cos_4k, "cos_64k": cos_64k, "cos_seed42": cos_seed,
                "fully_stable": n_stable == 3,
            }
        print(f"\n  fully stable (across 4k + 64k + seed42): {stable_count}/{len(demo_splits)}")
    else:
        print(f"\n(skipping joint analysis — missing width_comparison or catalog)")

    # ----------------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------------
    results = {
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "n_features": n_features,
        "k": sae_a.k,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "threshold_counts_all_features": threshold_counts,
        "labeled_features": labeled_results,
        "joint_width_seed_stability": joint,
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
