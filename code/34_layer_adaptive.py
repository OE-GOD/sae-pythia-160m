"""Option 1 — layer-adaptive patching method.

Idea: AtP works at early downstream layers but breaks at deeper layers
(established in Finding 11). So use AtP where it works, AP where it doesn't.

For a threshold T (layer index):
  predicted[L, h] = AP[L, h]   if L >= T
  predicted[L, h] = AtP[L, h]  if L < T

Compute compute cost:
  - AtP cost: 1 backward pass per firing position (negligible per-pair amortized)
  - AP cost: 1 forward pass per (pair, position)

  total_cost = AtP_setup + n_AP_pairs * AP_per_pair_cost

We sweep T and measure both compute saved (vs full AP) and accuracy (vs full AP
as ground truth). The optimal T is the smallest threshold where AtP is still
accurate.

Also compares to:
  - Full AtP (no AP at all)
  - Full AP (the ground truth — by definition, 100% accuracy)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    ap_data = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_path_patching.json")["feature_results"]
    atp_data = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_atp.json")["feature_results"]

    common_features = sorted(set(ap_data.keys()) & set(atp_data.keys()))

    # Build paired arrays
    pairs = []  # (feature, L, h, AP, AtP)
    for fid in common_features:
        downstream_layers = ap_data[fid]["downstream_layers"]
        n_heads = ap_data[fid]["n_heads"]
        ap_arr = np.array(ap_data[fid]["mean_mediation"])
        atp_arr = np.array(atp_data[fid]["mean_atp_mediation"])

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                pairs.append({
                    "feature": fid,
                    "L": L,
                    "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atp": float(atp_arr[li, h]),
                })

    n_total = len(pairs)
    downstream_layers = sorted(set(p["L"] for p in pairs))
    print(f"loaded {n_total} pairs across layers {downstream_layers}")

    ap_all = np.array([p["ap"] for p in pairs])
    atp_all = np.array([p["atp"] for p in pairs])

    # --- Baselines ---
    print("\n=== BASELINES ===")
    pearson_atp, _ = stats.pearsonr(ap_all, atp_all)
    rmse_atp = float(np.sqrt(np.mean((ap_all - atp_all) ** 2)))
    print(f"  Full AtP only:")
    print(f"    Pearson with AP truth: {pearson_atp:.4f}")
    print(f"    RMSE vs AP truth:      {rmse_atp:.5f}")
    print(f"    AP forward passes:     0")
    print(f"    AtP backward passes:   3 (one per firing-position-feature)")
    print(f"  Full AP (ground truth):")
    print(f"    Pearson with AP truth: 1.0000")
    print(f"    RMSE vs AP truth:      0.0")
    print(f"    AP forward passes:     {n_total}")

    # --- Sweep threshold T ---
    print("\n=== LAYER-ADAPTIVE SWEEP ===")
    print(f"  {'T':>3}  {'AP layers':<20}  {'AtP layers':<20}  "
          f"{'Pearson':>8}  {'RMSE':>8}  {'n AP':>5}  {'n AtP':>6}  {'cost vs full AP':>16}")

    results_by_threshold = []
    for T in [7, 8, 9, 10, 11, 12]:  # T=7 means AP for layers >=7 (all), T=12 means AtP for everything
        predicted = np.zeros_like(ap_all)
        n_ap = 0
        n_atp = 0
        for i, p in enumerate(pairs):
            if p["L"] >= T:
                predicted[i] = p["ap"]
                n_ap += 1
            else:
                predicted[i] = p["atp"]
                n_atp += 1

        pearson, _ = stats.pearsonr(ap_all, predicted)
        rmse = float(np.sqrt(np.mean((ap_all - predicted) ** 2)))
        cost_pct = 100 * n_ap / n_total

        ap_layers = [L for L in downstream_layers if L >= T]
        atp_layers = [L for L in downstream_layers if L < T]
        ap_layers_str = ",".join(str(L) for L in ap_layers) or "(none)"
        atp_layers_str = ",".join(str(L) for L in atp_layers) or "(none)"

        print(f"  {T:>3}  {ap_layers_str:<20}  {atp_layers_str:<20}  "
              f"{pearson:>8.4f}  {rmse:>8.5f}  {n_ap:>5}  {n_atp:>6}  {cost_pct:>15.1f}%")

        results_by_threshold.append({
            "T": T,
            "ap_layers": ap_layers,
            "atp_layers": atp_layers,
            "n_ap": n_ap,
            "n_atp": n_atp,
            "pearson": pearson,
            "rmse": rmse,
            "cost_pct_vs_full_ap": cost_pct,
        })

    # --- Find best Pareto point ---
    print("\n=== ANALYSIS ===")
    # The interesting tradeoff is "how much accuracy do we get back per unit of compute spent?"
    # Best T = smallest cost_pct that achieves a target accuracy (say, Pearson >= 0.99)

    target_pearson = 0.99
    print(f"  Threshold achieving Pearson >= {target_pearson}:")
    for r in results_by_threshold:
        if r["pearson"] >= target_pearson:
            print(f"    T={r['T']}: Pearson={r['pearson']:.4f}, "
                  f"RMSE={r['rmse']:.5f}, "
                  f"cost={r['cost_pct_vs_full_ap']:.1f}% of full AP")

    # The "best" T: smallest n_ap such that pearson >= 0.99
    candidates = [r for r in results_by_threshold if r["pearson"] >= target_pearson]
    if candidates:
        best = min(candidates, key=lambda r: r["n_ap"])
        print(f"\n  BEST T (min compute, >= {target_pearson} Pearson):")
        print(f"    T = {best['T']}")
        print(f"    AP layers: {best['ap_layers']}")
        print(f"    AtP layers: {best['atp_layers']}")
        print(f"    Pearson: {best['pearson']:.4f}  (vs full AtP {pearson_atp:.4f})")
        print(f"    RMSE: {best['rmse']:.5f}  (vs full AtP {rmse_atp:.5f})")
        print(f"    Compute: {best['cost_pct_vs_full_ap']:.1f}% of full AP forward passes")
        print(f"    Speedup vs full AP: {100/best['cost_pct_vs_full_ap']:.2f}x")
    else:
        print(f"  No threshold achieves Pearson >= {target_pearson}")

    # --- Plot Pareto frontier ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Pearson vs compute cost
    ax = axes[0]
    ts = [r["T"] for r in results_by_threshold]
    pearsons = [r["pearson"] for r in results_by_threshold]
    costs = [r["cost_pct_vs_full_ap"] for r in results_by_threshold]

    ax.scatter(costs, pearsons, c="#1f77b4", s=80, zorder=3,
               edgecolors="black", linewidth=0.5)
    for r in results_by_threshold:
        atp_label = "AtP only" if r["n_ap"] == 0 else (
            "AP only" if r["n_atp"] == 0 else f"T={r['T']}"
        )
        ax.annotate(atp_label, (r["cost_pct_vs_full_ap"], r["pearson"]),
                    xytext=(7, -3), textcoords="offset points", fontsize=9)
    # Baseline points
    ax.scatter([0], [pearson_atp], c="#d62728", s=120, marker="X",
                zorder=4, label="Full AtP (no AP)", edgecolors="black", linewidth=0.7)
    ax.scatter([100], [1.0], c="#2ca02c", s=120, marker="X",
                zorder=4, label="Full AP (truth)", edgecolors="black", linewidth=0.7)
    ax.axhline(target_pearson, color="gray", linestyle="--", alpha=0.5,
                label=f"Target Pearson = {target_pearson}")
    ax.set_xlabel("Compute cost (% of full activation patching)", fontsize=11)
    ax.set_ylabel("Pearson correlation with ground truth", fontsize=11)
    ax.set_title("Layer-adaptive method: accuracy vs compute", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: RMSE vs compute
    ax = axes[1]
    rmses = [r["rmse"] for r in results_by_threshold]
    ax.scatter(costs, rmses, c="#1f77b4", s=80, zorder=3,
                edgecolors="black", linewidth=0.5)
    for r in results_by_threshold:
        atp_label = "AtP only" if r["n_ap"] == 0 else (
            "AP only" if r["n_atp"] == 0 else f"T={r['T']}"
        )
        ax.annotate(atp_label, (r["cost_pct_vs_full_ap"], r["rmse"]),
                    xytext=(7, -3), textcoords="offset points", fontsize=9)
    ax.scatter([0], [rmse_atp], c="#d62728", s=120, marker="X",
                zorder=4, label="Full AtP", edgecolors="black", linewidth=0.7)
    ax.scatter([100], [0.0], c="#2ca02c", s=120, marker="X",
                zorder=4, label="Full AP (truth)", edgecolors="black", linewidth=0.7)
    ax.set_xlabel("Compute cost (% of full activation patching)", fontsize=11)
    ax.set_ylabel("RMSE vs ground truth", fontsize=11)
    ax.set_title("Layer-adaptive method: error vs compute", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "results/figures/fig7_layer_adaptive.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    Path("checkpoints/sae_layer6_topk64_full.layer_adaptive.json").write_text(
        json.dumps({
            "n_total": n_total,
            "baseline_full_atp": {"pearson": pearson_atp, "rmse": rmse_atp,
                                   "ap_forward_passes": 0},
            "baseline_full_ap": {"pearson": 1.0, "rmse": 0.0,
                                  "ap_forward_passes": n_total},
            "threshold_sweep": results_by_threshold,
            "target_pearson": target_pearson,
            "best_T": best if candidates else None,
        }, indent=2)
    )
    print("saved checkpoints/sae_layer6_topk64_full.layer_adaptive.json")


if __name__ == "__main__":
    main()
