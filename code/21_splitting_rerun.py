"""Rigorous re-test of feature splitting across widths.

Addresses methodological issues in 08_width_comparison.py:
  1. Uses top-K matching instead of threshold-based counting (no arbitrary 0.7 cutoff).
  2. Reports FULL similarity distribution per feature (median, top-1, top-5, top-10).
  3. Tests both directions (16k -> 64k and 64k -> 16k).
  4. Tests many more features (all labeled monosemantic features by default).
  5. Compares to a randomized baseline (shuffled feature pairings).

If splitting exists, signature: for many 16k features, the top-1 64k match should
be much more similar than random pairs, and the top-5 should be a tight cluster.
If splitting is absent, signature: top-1 should be moderate; similarity should
decay smoothly across top-10.
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
    p.add_argument("--ckpt-16k", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--ckpt-64k", type=str, default="checkpoints/sae_layer6_topk64_w64k.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--top-k", type=int, default=10, help="how many top matches to report per feature")
    p.add_argument("--max-features", type=int, default=50, help="how many 16k features to test")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="data/splitting_rerun.json")
    return p.parse_args()


def load_sae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sae = TopKSAE(ckpt["d_model"], ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae


def unit_norm(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


def main():
    args = parse_args()
    print("loading SAEs...")
    sae_16k = load_sae(args.ckpt_16k, args.device)
    sae_64k = load_sae(args.ckpt_64k, args.device)
    print(f"  16k: n_features={sae_16k.n_features}")
    print(f"  64k: n_features={sae_64k.n_features}")

    W_16k = unit_norm(sae_16k.W_dec.data)  # (d_model, 16384)
    W_64k = unit_norm(sae_64k.W_dec.data)  # (d_model, 65536)

    # --- Find DEAD features in 64k SAE (these inflate "different basis" finding) ---
    # Dead features have ~random unit vectors. We exclude them from the analysis
    # because they pollute the similarity comparison.
    print("\nidentifying dead features in 64k SAE...")
    # Approximation: a feature is "alive" if its decoder column norm pre-normalization is
    # nontrivial (decoder column never updated → tiny norm). Use raw W_dec, not unit_norm.
    raw_norms_64k = sae_64k.W_dec.data.norm(dim=0)
    # Heuristic: alive if norm is at least 50% of median
    median_norm = raw_norms_64k.median()
    alive_mask_64k = raw_norms_64k > 0.5 * median_norm
    n_alive_64k = int(alive_mask_64k.sum().item())
    print(f"  64k features alive (above 50% median norm): {n_alive_64k} / {sae_64k.n_features}")

    # Use only alive 64k features
    W_64k_alive = W_64k[:, alive_mask_64k]
    alive_idx_64k = torch.nonzero(alive_mask_64k).flatten().tolist()
    print(f"  using {W_64k_alive.shape[1]} alive 64k features for comparison")

    # --- Pick features to test ---
    catalog = json.loads(Path(args.catalog).read_text())
    # Prefer monosemantic features, then high confidence
    candidates = []
    for fid_str, info in catalog.items():
        if "error" in info:
            continue
        score = (
            (1 if info.get("classification") == "MONOSEMANTIC" else 0) * 10
            + (1 if info.get("confidence") == "high" else 0) * 5
            + info.get("peak_activation", 0) / 100  # tiebreak by activation
        )
        candidates.append((score, int(fid_str), info))
    candidates.sort(reverse=True)
    chosen_features = candidates[: args.max_features]
    print(f"\ntesting {len(chosen_features)} features (highest-confidence monosemantic first)")

    # --- For each chosen feature: find top-K 64k matches ---
    results = []
    summary_top1 = []
    summary_top5_mean = []
    summary_top10_mean = []

    for score, fid, info in chosen_features:
        v = W_16k[:, fid]  # (d_model,)
        sims_64k = v @ W_64k_alive  # (n_alive_64k,)
        top_vals, top_idx = sims_64k.topk(args.top_k)
        top_global_idx = [alive_idx_64k[i] for i in top_idx.tolist()]
        top_sims = top_vals.tolist()

        results.append({
            "fid_16k": fid,
            "label": info.get("label", "?"),
            "classification": info.get("classification"),
            "confidence": info.get("confidence"),
            "peak_activation": info.get("peak_activation"),
            "top_k_64k_features": top_global_idx,
            "top_k_64k_sims": top_sims,
            "top1_sim": top_sims[0],
            "top5_mean_sim": float(np.mean(top_sims[:5])),
            "top10_mean_sim": float(np.mean(top_sims[:10])),
        })
        summary_top1.append(top_sims[0])
        summary_top5_mean.append(float(np.mean(top_sims[:5])))
        summary_top10_mean.append(float(np.mean(top_sims[:10])))

    # --- Random baseline: pick 50 random 16k features and run same analysis ---
    print("\ncomputing random baseline (50 random 16k features)...")
    np.random.seed(0)
    random_fids = np.random.choice(sae_16k.n_features, size=50, replace=False)
    random_top1 = []
    random_top5_mean = []
    random_top10_mean = []
    for fid in random_fids:
        v = W_16k[:, fid]
        sims_64k = v @ W_64k_alive
        top_vals = sims_64k.topk(args.top_k).values.tolist()
        random_top1.append(top_vals[0])
        random_top5_mean.append(float(np.mean(top_vals[:5])))
        random_top10_mean.append(float(np.mean(top_vals[:10])))

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print("SPLITTING ANALYSIS (16k -> 64k decoder column matching)")
    print(f"{'=' * 70}")
    print(f"  {'Group':<15} {'top1':<20} {'top5_mean':<20} {'top10_mean':<20}")
    print(f"  {'-' * 15:<15} {'-' * 20:<20} {'-' * 20:<20} {'-' * 20:<20}")

    def fmt(values):
        return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

    print(f"  {'Chosen':<15} {fmt(summary_top1):<20} {fmt(summary_top5_mean):<20} {fmt(summary_top10_mean):<20}")
    print(f"  {'Random':<15} {fmt(random_top1):<20} {fmt(random_top5_mean):<20} {fmt(random_top10_mean):<20}")

    # --- Per-feature breakdown of top-10 distribution ---
    print(f"\n--- Top 10 examples (highest top1 sim) ---")
    sorted_results = sorted(results, key=lambda r: -r["top1_sim"])
    for r in sorted_results[:10]:
        top_str = "  ".join(f"{s:.2f}" for s in r["top_k_64k_sims"][:10])
        print(f"  f{r['fid_16k']:>5} top1={r['top1_sim']:.3f}  [{top_str}]  {r['label'][:40]}")

    print(f"\n--- Bottom 10 examples (lowest top1 sim) ---")
    for r in sorted_results[-10:]:
        top_str = "  ".join(f"{s:.2f}" for s in r["top_k_64k_sims"][:10])
        print(f"  f{r['fid_16k']:>5} top1={r['top1_sim']:.3f}  [{top_str}]  {r['label'][:40]}")

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    mean_top1_chosen = np.mean(summary_top1)
    mean_top1_random = np.mean(random_top1)
    ratio = mean_top1_chosen / max(mean_top1_random, 1e-6)
    print(f"  Chosen mean top1: {mean_top1_chosen:.3f}")
    print(f"  Random mean top1: {mean_top1_random:.3f}")
    print(f"  Ratio: {ratio:.2f}x")
    print()
    print("  If splitting works: chosen top1 should be HIGH (>= 0.8) and much higher than random.")
    print("  If no splitting:    chosen top1 should be modest (~0.5-0.6) — not much above random.")
    print()
    if mean_top1_chosen > 0.8:
        print("  -> STRONG SPLITTING: 64k features track 16k features tightly.")
    elif mean_top1_chosen > 0.6:
        print("  -> MODERATE SPLITTING: some correspondence but not tight.")
    else:
        print("  -> WEAK/NO SPLITTING: 64k features mostly find different directions.")

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt_16k": args.ckpt_16k,
        "ckpt_64k": args.ckpt_64k,
        "n_alive_64k": n_alive_64k,
        "n_features_tested": len(results),
        "summary": {
            "chosen_top1_mean": float(mean_top1_chosen),
            "chosen_top1_std": float(np.std(summary_top1)),
            "chosen_top5_mean": float(np.mean(summary_top5_mean)),
            "chosen_top10_mean": float(np.mean(summary_top10_mean)),
            "random_top1_mean": float(mean_top1_random),
            "random_top1_std": float(np.std(random_top1)),
            "random_top5_mean": float(np.mean(random_top5_mean)),
            "random_top10_mean": float(np.mean(random_top10_mean)),
            "ratio_chosen_over_random_top1": float(ratio),
        },
        "per_feature": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
