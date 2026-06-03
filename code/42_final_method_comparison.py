"""Final comparison of all five methods + GELU-linearization diagnostic.

Methods (all compared to AP as ground truth on 180 pairs):
  AP                    — script 30, 60 passes/feature-pos
  AtP                   — script 32, 2 passes
  IG (N=10)             — script 35, 20 passes
  Adaptive (T=11)       — Finding 12, 14 passes
  AtP-corrected (N=2 trapezoidal) — script 40, 4 passes
  Per-pair adaptive     — script 39, ~14-15 passes (config-dependent)

Diagnostic (L11 only):
  AP_gelu_lin           — script 41, AP with GELU at L11 linearized
                          If ≈ AP_full: GELU not the AtP killer
                          If ≈ AtP_full at L11: GELU is the killer
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    ap = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_path_patching.json")["feature_results"]
    atp = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_atp.json")["feature_results"]
    ig = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_ig.json")["feature_results"]
    corrected = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_atp_corrected.json")["feature_results"]
    per_pair_sweep = load_json("checkpoints/sae_layer6_topk64_full.per_pair_adaptive.json")["sweep_results"]
    gelu_lin = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_gelu_lin_L11.json")["feature_results"]

    common = sorted(set(ap.keys()) & set(atp.keys()) & set(ig.keys()) & set(corrected.keys()))

    rows = []
    for fid in common:
        downstream_layers = ap[fid]["downstream_layers"]
        n_heads = ap[fid]["n_heads"]
        ap_arr = np.array(ap[fid]["mean_mediation"])
        atp_arr = np.array(atp[fid]["mean_atp_mediation"])
        ig_arr = np.array(ig[fid]["mean_ig_mediation"])
        corr_arr = np.array(corrected[fid]["mean_corrected_mediation"])

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "AP": float(ap_arr[li, h]),
                    "AtP": float(atp_arr[li, h]),
                    "IG": float(ig_arr[li, h]),
                    "Corrected": float(corr_arr[li, h]),
                    "Adaptive": float(ap_arr[li, h]) if L == 11 else float(atp_arr[li, h]),
                })

    n_total = len(rows)
    ap_all = np.array([r["AP"] for r in rows])
    methods_simple = ["AtP", "IG", "Adaptive", "Corrected"]

    print(f"=== ALL METHODS: PEARSON & RMSE vs AP ({n_total} pairs) ===")
    print(f"  {'Method':>15} {'Pearson':>10} {'RMSE':>10} {'cost':>8}")

    method_costs = {"AP": 60, "AtP": 2, "IG": 20, "Adaptive": 14, "Corrected": 4}
    overall = {}
    for m in methods_simple:
        v = np.array([r[m] for r in rows])
        pr, _ = stats.pearsonr(ap_all, v)
        rmse = float(np.sqrt(np.mean((ap_all - v) ** 2)))
        overall[m] = {"pearson": pr, "rmse": rmse}
        print(f"  {m:>15} {pr:>10.4f} {rmse:>10.5f} {method_costs[m]:>8}")

    # Per-pair adaptive — take the best on the Pareto frontier
    print(f"\n  Best per-pair adaptive configs:")
    pp_sorted = sorted(per_pair_sweep, key=lambda x: x["pearson"], reverse=True)
    for entry in pp_sorted[:5]:
        print(f"    probe_thr={entry['probe_thr']:.3f}, lin_tol={entry['lin_tol']:.2f}: "
              f"Pearson={entry['pearson']:.4f}, cost={entry['cost_per_fpos']:.1f} passes")

    # Per-layer
    print("\n=== PER LAYER (Pearson vs AP) ===")
    layers = sorted(set(r["L"] for r in rows))
    print(f"  {'Layer':>6} {'AtP':>10} {'IG':>10} {'Adaptive':>10} {'Corrected':>10}")
    per_layer = {m: {} for m in methods_simple}
    for L in layers:
        sub = [r for r in rows if r["L"] == L]
        ap_sub = np.array([r["AP"] for r in sub])
        vals_str = []
        for m in methods_simple:
            arr = np.array([r[m] for r in sub])
            if np.std(arr) == 0 or np.std(ap_sub) == 0:
                pr = float("nan")
            else:
                pr, _ = stats.pearsonr(ap_sub, arr)
            per_layer[m][L] = pr
            vals_str.append(f"{pr:>10.4f}")
        print(f"  {L:>6} " + " ".join(vals_str))

    # GELU diagnostic — compare AP_gelu_lin to AtP and AP at L11
    print("\n=== GELU LINEARIZATION DIAGNOSTIC (L11 only) ===")
    print("  If AP_gelu_lin ≈ AtP_L11: GELU is the AtP killer at L11")
    print("  If AP_gelu_lin ≈ AP_L11: GELU is NOT the killer; look elsewhere")
    print()
    print(f"  {'feature':>10} {'head':>4} {'AP_full':>10} {'AP_gelu_lin':>12} "
          f"{'AtP':>10} {'lin→AtP?':>10} {'lin→AP?':>10}")
    gelu_rows = []
    for fid in common:
        if fid not in gelu_lin:
            continue
        n_h = gelu_lin[fid]["n_heads"]
        mean_lin = np.array(gelu_lin[fid]["mean_mediation_gelu_linearized"])
        ap_arr = np.array(ap[fid]["mean_mediation"])
        atp_arr = np.array(atp[fid]["mean_atp_mediation"])
        # L11 is index = downstream_layers.index(11)
        downstream_layers = ap[fid]["downstream_layers"]
        l11_idx = downstream_layers.index(11)
        for h in range(n_h):
            ap_v = float(ap_arr[l11_idx, h])
            atp_v = float(atp_arr[l11_idx, h])
            lin_v = float(mean_lin[h])
            close_to_atp = abs(lin_v - atp_v) < abs(lin_v - ap_v)
            gelu_rows.append({"f": fid, "h": h, "ap": ap_v, "lin": lin_v, "atp": atp_v})
            print(f"  f{fid:>9} {h:>4} {ap_v:>+10.4f} {lin_v:>+12.4f} {atp_v:>+10.4f} "
                  f"{('YES' if close_to_atp else 'no'):>10} "
                  f"{('YES' if not close_to_atp else 'no'):>10}")

    # Summary stats for GELU diagnostic
    ap_l11 = np.array([r["ap"] for r in gelu_rows])
    lin_l11 = np.array([r["lin"] for r in gelu_rows])
    atp_l11 = np.array([r["atp"] for r in gelu_rows])
    pr_lin_atp, _ = stats.pearsonr(lin_l11, atp_l11)
    pr_lin_ap, _ = stats.pearsonr(lin_l11, ap_l11)
    pr_atp_ap, _ = stats.pearsonr(atp_l11, ap_l11)
    rmse_lin_atp = float(np.sqrt(np.mean((lin_l11 - atp_l11) ** 2)))
    rmse_lin_ap = float(np.sqrt(np.mean((lin_l11 - ap_l11) ** 2)))
    print(f"\n  Pearson(AP_gelu_lin, AtP_L11) = {pr_lin_atp:.4f}, RMSE = {rmse_lin_atp:.5f}")
    print(f"  Pearson(AP_gelu_lin, AP_L11)  = {pr_lin_ap:.4f}, RMSE = {rmse_lin_ap:.5f}")
    print(f"  Pearson(AtP_L11, AP_L11) [baseline] = {pr_atp_ap:.4f}")

    if pr_lin_atp > pr_lin_ap:
        print("  => Linearizing GELU pushes AP toward AtP. GELU IS contributing to AtP failure.")
    else:
        print("  => Linearizing GELU does NOT pull AP toward AtP. GELU is NOT the AtP killer.")

    # ---- Final figure: Pareto + per-layer ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: per-layer Pearson
    ax = axes[0]
    colors = {"AtP": "#d62728", "IG": "#ff7f0e",
              "Adaptive": "#2ca02c", "Corrected": "#9467bd"}
    markers = {"AtP": "o", "IG": "s", "Adaptive": "^", "Corrected": "D"}
    for m in methods_simple:
        ys = [per_layer[m][L] for L in layers]
        ax.plot(layers, ys, marker=markers[m], color=colors[m],
                 markersize=10, linewidth=2, label=m, markeredgecolor="black",
                 markeredgewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect (= AP)")
    ax.set_xlabel("Downstream layer", fontsize=11)
    ax.set_ylabel("Pearson correlation with AP", fontsize=11)
    ax.set_title("Per-layer accuracy", fontsize=12)
    ax.set_xticks(layers)
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: full Pareto including per-pair sweep
    ax = axes[1]
    # Static method points
    method_pearson_overall = {"AP": 1.0}
    for m in methods_simple:
        method_pearson_overall[m] = overall[m]["pearson"]

    colors_pareto = {"AP": "#7f7f7f", "AtP": "#d62728", "IG": "#ff7f0e",
                     "Adaptive": "#2ca02c", "Corrected": "#9467bd"}

    for m in ["AP", "AtP", "IG", "Adaptive", "Corrected"]:
        ax.scatter(method_costs[m], method_pearson_overall[m],
                    s=180, c=colors_pareto[m], edgecolors="black", linewidth=1, zorder=3)
        ax.annotate(m, (method_costs[m], method_pearson_overall[m]),
                     xytext=(8, -8), textcoords="offset points", fontsize=11,
                     fontweight="bold")

    # Per-pair sweep as a line
    pp_costs = [r["cost_per_fpos"] for r in per_pair_sweep]
    pp_pears = [r["pearson"] for r in per_pair_sweep]
    order = np.argsort(pp_costs)
    pp_costs = [pp_costs[i] for i in order]
    pp_pears = [pp_pears[i] for i in order]
    ax.plot(pp_costs, pp_pears, color="#1f77b4", marker="x", markersize=8,
             alpha=0.7, label="Per-pair adaptive (sweep)", linewidth=1.5)

    ax.set_xlabel("Compute cost (model passes per feature-position)", fontsize=11)
    ax.set_ylabel("Overall Pearson with AP", fontsize=11)
    ax.set_title("Pareto frontier: all methods", fontsize=12)
    ax.set_xlim(-3, 65)
    ax.set_ylim(0.92, 1.01)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "results/figures/fig10_all_methods_with_per_pair.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    Path("checkpoints/sae_layer6_topk64_full.final_comparison.json").write_text(
        json.dumps({
            "n_pairs": n_total,
            "overall": {m: overall[m] for m in methods_simple},
            "method_costs": method_costs,
            "per_layer_pearson": {m: {str(L): float(v) for L, v in per_layer[m].items()}
                                   for m in methods_simple},
            "gelu_l11_diagnostic": {
                "n_pairs_l11": len(gelu_rows),
                "pearson_lin_vs_atp": pr_lin_atp,
                "pearson_lin_vs_ap": pr_lin_ap,
                "pearson_atp_vs_ap_baseline": pr_atp_ap,
                "rmse_lin_vs_atp": rmse_lin_atp,
                "rmse_lin_vs_ap": rmse_lin_ap,
                "verdict": ("GELU contributes to AtP failure"
                            if pr_lin_atp > pr_lin_ap
                            else "GELU is NOT the AtP killer"),
            },
        }, indent=2)
    )


if __name__ == "__main__":
    main()
