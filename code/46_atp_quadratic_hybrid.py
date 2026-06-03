"""AtP + quadratic-fit hybrid: AtP everywhere, S3 (quadratic 3-point fit)
only at layers where AtP fails, only for non-trivial pairs.

Method:
  1. Run AtP once (2 passes). Gets all (L, h) cheaply.
  2. For each (feature, position, L, h) where L is in "failure layers"
     AND |AtP_estimate| > threshold:
       Run AP at alpha ∈ {0.25, 0.5, 0.75} (3 forward passes).
       Fit quadratic, predict at alpha=1.0.
  3. Use AtP estimate for everything else.

Sweep over (failure_layers, threshold) configurations to find the
Pareto frontier vs AP.

Also evaluate midpoint AtP (script 43 output) for comparison.
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
    midpoint_data = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_midpoint_atp_a0.5.json")["feature_results"]
    alpha_data = load_json("checkpoints/sae_layer6_topk64_full.alpha_scaling_all_layers.json")["feature_results"]

    common = sorted(set(ap_data.keys()) & set(atp_data.keys()) & set(midpoint_data.keys()) & set(alpha_data.keys()))

    rows = []
    for fid in common:
        ap_arr = np.array(ap_data[fid]["mean_mediation"])
        atp_arr = np.array(atp_data[fid]["mean_atp_mediation"])
        midpoint_arr = np.array(midpoint_data[fid]["mean_midpoint_atp_mediation"])
        downstream_layers = ap_data[fid]["downstream_layers"]
        n_heads = ap_data[fid]["n_heads"]

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                drops_025 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.25"]
                drops_05 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.5"]
                drops_075 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.75"]
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atp": float(atp_arr[li, h]),
                    "midpoint": float(midpoint_arr[li, h]),
                    "probe025": float(np.mean(drops_025)) if drops_025 else 0.0,
                    "probe05": float(np.mean(drops_05)) if drops_05 else 0.0,
                    "probe075": float(np.mean(drops_075)) if drops_075 else 0.0,
                })

    n_total = len(rows)
    ap = np.array([r["ap"] for r in rows])
    atp_arr_flat = np.array([r["atp"] for r in rows])
    midpoint_arr_flat = np.array([r["midpoint"] for r in rows])
    layers = sorted(set(r["L"] for r in rows))
    n_features = len(common)
    n_positions = 3  # from script 32

    # Compute S3 (quadratic) for each row
    s3_pred = np.zeros(n_total)
    for i, r in enumerate(rows):
        xs = np.array([0.25, 0.5, 0.75])
        ys = np.array([r["probe025"], r["probe05"], r["probe075"]])
        if np.std(ys) < 1e-10:
            s3_pred[i] = 0.0
            continue
        coeffs = np.polyfit(xs, ys, 2)
        s3_pred[i] = np.polyval(coeffs, 1.0)

    print(f"=== BASELINES ({n_total} pairs) ===")
    pearson_atp, _ = stats.pearsonr(ap, atp_arr_flat)
    pearson_midpoint, _ = stats.pearsonr(ap, midpoint_arr_flat)
    rmse_midpoint = float(np.sqrt(np.mean((ap - midpoint_arr_flat) ** 2)))
    print(f"  Plain AtP (alpha=0):    Pearson = {pearson_atp:.4f}, cost = 2")
    print(f"  Midpoint AtP (alpha=0.5): Pearson = {pearson_midpoint:.4f}, RMSE = {rmse_midpoint:.5f}, cost = 2")

    # Per-layer midpoint
    print("\n=== MIDPOINT AtP PER LAYER ===")
    print(f"  {'Layer':>6} {'AtP':>8} {'Midpoint AtP':>14}")
    for L in layers:
        mask = np.array([r["L"] == L for r in rows])
        ap_L = ap[mask]
        atp_L = atp_arr_flat[mask]
        mp_L = midpoint_arr_flat[mask]
        if np.std(ap_L) < 1e-10:
            continue
        pr_atp, _ = stats.pearsonr(ap_L, atp_L)
        pr_mp, _ = stats.pearsonr(ap_L, mp_L)
        print(f"  {L:>6} {pr_atp:>8.4f} {pr_mp:>14.4f}")

    # ---- AtP + S3 hybrid sweeps ----
    print("\n=== AtP + S3-AT-L11 HYBRID SWEEP ===")
    print(f"  {'thr':>8} {'n_S3_calls':>12} {'Pearson':>10} {'RMSE':>10} "
          f"{'cost/fpos':>10} {'cost%':>8} {'speedup':>8}")

    sweep_results = []
    cost_full_ap = 60  # passes per feature-position
    atp_cost = 2
    s3_cost_per_pair = 3  # 3 alphas per probed pair

    failure_layers_options = [{11}, {10, 11}]
    thresholds = [0.001, 0.003, 0.005, 0.01, 0.02]

    for failure_layers in failure_layers_options:
        for thr in thresholds:
            predicted = np.zeros_like(ap)
            n_s3 = 0
            for i, r in enumerate(rows):
                if r["L"] in failure_layers and abs(r["atp"]) >= thr:
                    predicted[i] = s3_pred[i]
                    n_s3 += 1
                else:
                    predicted[i] = r["atp"]
            pr, _ = stats.pearsonr(ap, predicted)
            rmse = float(np.sqrt(np.mean((ap - predicted) ** 2)))
            # Cost per feature-position
            avg_s3 = n_s3 / (n_features * n_positions)
            cost = atp_cost + avg_s3 * s3_cost_per_pair
            cost_pct = 100 * cost / cost_full_ap
            sweep_results.append({
                "failure_layers": sorted(failure_layers),
                "threshold": thr, "n_s3_calls": n_s3,
                "pearson": pr, "rmse": rmse,
                "cost_per_fpos": cost, "cost_pct": cost_pct,
                "speedup": cost_full_ap / cost,
            })
            fl_str = ",".join(str(L) for L in sorted(failure_layers))
            print(f"  L{{{fl_str}}} thr={thr:.3f}: "
                  f"{n_s3:>12} {pr:>10.4f} {rmse:>10.5f} "
                  f"{cost:>10.1f} {cost_pct:>7.1f}% {cost_full_ap/cost:>7.2f}x")

    # ---- Final summary ----
    print("\n=== FINAL FRONTIER (all methods) ===")
    print(f"  {'Method':>40} {'Pearson':>10} {'Cost':>8} {'Speedup':>10}")
    print(f"  {'Full AP (truth)':>40} {'1.0000':>10} {'60':>8} {'1.00x':>10}")
    print(f"  {'AtP':>40} {pearson_atp:>10.4f} {'2':>8} {cost_full_ap/2:>9.2f}x")
    print(f"  {'Midpoint AtP (alpha=0.5)':>40} {pearson_midpoint:>10.4f} {'2':>8} {cost_full_ap/2:>9.2f}x")
    print(f"  {'IG N=10':>40} {'0.9472':>10} {'20':>8} {'3.00x':>10}")
    print(f"  {'Layer-adaptive T=11':>40} {'0.9919':>10} {'14':>8} {'4.29x':>10}")
    print(f"  {'Per-pair adaptive (probe_thr=0.003, lin_tol=0.10)':>40} {'0.9987':>10} {'15.6':>8} {'3.85x':>10}")
    # Pick best hybrid
    pareto_hybrid = max(sweep_results, key=lambda r: r["pearson"] - 0.001 * r["cost_per_fpos"])
    fl_str = ",".join(str(L) for L in pareto_hybrid["failure_layers"])
    label = f"AtP+S3@L{{{fl_str}}} thr={pareto_hybrid['threshold']:.3f}"
    print(f"  {label:>40} "
          f"{pareto_hybrid['pearson']:>10.4f} {pareto_hybrid['cost_per_fpos']:>8.1f} "
          f"{pareto_hybrid['speedup']:>9.2f}x")

    # Find best by accuracy target
    print("\n=== BEST HYBRID PER ACCURACY TARGET ===")
    for target in [0.99, 0.995, 0.999, 0.9995]:
        candidates = [r for r in sweep_results if r["pearson"] >= target]
        if not candidates:
            print(f"  Pearson >= {target}: no hybrid config achieves this")
            continue
        best = min(candidates, key=lambda r: r["cost_per_fpos"])
        fl_str = ",".join(str(L) for L in best["failure_layers"])
        print(f"  Pearson >= {target}: AtP+S3@L{{{fl_str}}} thr={best['threshold']:.3f}, "
              f"Pearson={best['pearson']:.4f}, cost={best['cost_per_fpos']:.1f} passes "
              f"({best['cost_pct']:.1f}% of AP), speedup={best['speedup']:.2f}x")

    # Save
    Path("checkpoints/sae_layer6_topk64_full.atp_s3_hybrid.json").write_text(
        json.dumps({
            "method": "atp_s3_quadratic_hybrid",
            "baselines": {
                "atp": {"pearson": pearson_atp},
                "midpoint_atp": {"pearson": pearson_midpoint, "rmse": rmse_midpoint},
            },
            "sweep_results": sweep_results,
        }, indent=2)
    )
    print("\nsaved checkpoints/sae_layer6_topk64_full.atp_s3_hybrid.json")


if __name__ == "__main__":
    main()
