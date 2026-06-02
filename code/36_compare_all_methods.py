"""Compare all four methods on the same 180 (feature, head) pairs.

Methods:
  AP   — activation patching (ground truth) — script 30
  AtP  — attribution patching (first-order, 1 backward pass per position) — script 32
  IG   — integrated gradients (10-alpha average) — script 35
  Adaptive — layer-adaptive AtP+AP (AtP for L<=10, AP for L=11)

Outputs:
  - Per-layer Pearson and RMSE for each method against AP truth
  - Overall correlation
  - Single comparison figure
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
    ig_data = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_ig.json")["feature_results"]

    common = sorted(set(ap_data.keys()) & set(atp_data.keys()) & set(ig_data.keys()))

    rows = []
    for fid in common:
        downstream_layers = ap_data[fid]["downstream_layers"]
        n_heads = ap_data[fid]["n_heads"]
        ap_arr = np.array(ap_data[fid]["mean_mediation"])
        atp_arr = np.array(atp_data[fid]["mean_atp_mediation"])
        ig_arr = np.array(ig_data[fid]["mean_ig_mediation"])

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                ap_val = float(ap_arr[li, h])
                atp_val = float(atp_arr[li, h])
                ig_val = float(ig_arr[li, h])
                # Layer-adaptive: use AP for L==11, AtP otherwise
                adaptive_val = ap_val if L == 11 else atp_val

                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "AP": ap_val, "AtP": atp_val, "IG": ig_val,
                    "Adaptive": adaptive_val,
                })

    n_total = len(rows)
    print(f"loaded {n_total} pairs")

    ap_all = np.array([r["AP"] for r in rows])

    # Per-method overall correlation
    print("\n=== OVERALL ===")
    for method in ["AtP", "IG", "Adaptive"]:
        vals = np.array([r[method] for r in rows])
        pearson, _ = stats.pearsonr(ap_all, vals)
        rmse = float(np.sqrt(np.mean((ap_all - vals) ** 2)))
        print(f"  {method:>10}: Pearson={pearson:.4f}  RMSE={rmse:.5f}")

    # Per-layer
    print("\n=== PER LAYER (Pearson vs AP) ===")
    layers = sorted(set(r["L"] for r in rows))
    method_layer_pearson = {m: {} for m in ["AtP", "IG", "Adaptive"]}
    method_layer_rmse = {m: {} for m in ["AtP", "IG", "Adaptive"]}
    print(f"  {'Layer':>6} {'AtP':>10} {'IG':>10} {'Adaptive':>10}")
    for L in layers:
        sub = [r for r in rows if r["L"] == L]
        ap_sub = np.array([r["AP"] for r in sub])
        vals = {}
        for method in ["AtP", "IG", "Adaptive"]:
            arr = np.array([r[method] for r in sub])
            if np.std(arr) == 0 or np.std(ap_sub) == 0:
                pr, rmse = float("nan"), float(np.sqrt(np.mean((ap_sub - arr) ** 2)))
            else:
                pr, _ = stats.pearsonr(ap_sub, arr)
                rmse = float(np.sqrt(np.mean((ap_sub - arr) ** 2)))
            method_layer_pearson[method][L] = pr
            method_layer_rmse[method][L] = rmse
            vals[method] = pr
        print(f"  {L:>6} {vals['AtP']:>10.4f} {vals['IG']:>10.4f} {vals['Adaptive']:>10.4f}")

    # Cost summary
    print("\n=== COST PER (feature, position) ===")
    print(f"  AP:       60 forward passes (one per head)")
    print(f"  AtP:      1 forward + 1 backward")
    print(f"  IG:       10 forward + 10 backward (N_alphas=10)")
    print(f"  Adaptive: 1 forward + 1 backward + 12 forward (AP for L=11 only)")

    # Compute relative cost (where AP forward = 1, backward ≈ 1)
    cost_ap = 60
    cost_atp = 2  # 1 forward + 1 backward
    cost_ig = 20  # 10 forward + 10 backward
    cost_adaptive = 2 + 12  # AtP cost + AP for L=11 only

    print(f"\n  Relative cost (model passes per feature-position):")
    print(f"    Full AP:       {cost_ap}")
    print(f"    Full AtP:      {cost_atp}  ({100*cost_atp/cost_ap:.1f}% of AP)")
    print(f"    Full IG:       {cost_ig}  ({100*cost_ig/cost_ap:.1f}% of AP)")
    print(f"    Adaptive:      {cost_adaptive}  ({100*cost_adaptive/cost_ap:.1f}% of AP)")

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: per-layer Pearson (the failure-mode plot)
    ax = axes[0]
    method_colors = {"AtP": "#d62728", "IG": "#ff7f0e", "Adaptive": "#2ca02c"}
    method_markers = {"AtP": "o", "IG": "s", "Adaptive": "^"}
    for method in ["AtP", "IG", "Adaptive"]:
        ys = [method_layer_pearson[method][L] for L in layers]
        ax.plot(layers, ys, marker=method_markers[method], color=method_colors[method],
                 markersize=10, linewidth=2, label=method, markeredgecolor="black",
                 markeredgewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect (= AP truth)")
    ax.set_xlabel("Downstream layer", fontsize=11)
    ax.set_ylabel("Pearson correlation with AP", fontsize=11)
    ax.set_title("Per-layer accuracy of each method vs ground truth", fontsize=12)
    ax.set_xticks(layers)
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: cost vs overall Pearson (Pareto plot)
    ax = axes[1]
    method_costs = {"AP": cost_ap, "AtP": cost_atp, "IG": cost_ig, "Adaptive": cost_adaptive}
    method_pearson_overall = {}
    for method in ["AtP", "IG", "Adaptive"]:
        vals = np.array([r[method] for r in rows])
        pr, _ = stats.pearsonr(ap_all, vals)
        method_pearson_overall[method] = pr
    method_pearson_overall["AP"] = 1.0

    colors = {"AP": "#7f7f7f", "AtP": "#d62728", "IG": "#ff7f0e", "Adaptive": "#2ca02c"}
    for method in ["AP", "AtP", "IG", "Adaptive"]:
        ax.scatter(method_costs[method], method_pearson_overall[method],
                    s=180, c=colors[method], edgecolors="black", linewidth=1, zorder=3)
        ax.annotate(method, (method_costs[method], method_pearson_overall[method]),
                     xytext=(8, -8), textcoords="offset points", fontsize=11,
                     fontweight="bold")

    ax.set_xlabel("Compute cost (model passes per feature-position)", fontsize=11)
    ax.set_ylabel("Overall Pearson with AP", fontsize=11)
    ax.set_title("Pareto frontier: accuracy vs compute", fontsize=12)
    ax.set_xlim(-3, cost_ap * 1.1)
    ax.set_ylim(0.92, 1.01)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "results/figures/fig8_all_methods_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    Path("checkpoints/sae_layer6_topk64_full.all_methods_comparison.json").write_text(
        json.dumps({
            "n_total": n_total,
            "overall_pearson": {m: float(method_pearson_overall[m])
                                  for m in method_pearson_overall},
            "per_layer_pearson": {m: {str(L): float(v) for L, v in d.items()}
                                    for m, d in method_layer_pearson.items()},
            "relative_cost": {m: int(method_costs[m]) for m in method_costs},
        }, indent=2)
    )
    print("saved checkpoints/sae_layer6_topk64_full.all_methods_comparison.json")


if __name__ == "__main__":
    main()
