"""Hybrid: efficient AtP* + per-pair AP verification.

Idea: efficient AtP* gives Pearson 0.993 at 2 passes (Finding 19). The
remaining 0.007 Pearson gap comes from a small number of pairs where
AtP* still mispredicts. If we can detect those pairs and fall back to
AP just for them, we'd close the gap at minimal extra cost.

Decision rule (using existing alpha-scaling data):
  For each (feature, head, position):
    1. Use efficient AtP* estimate as base prediction.
    2. If |AtP*| < probe_threshold: skip probe, predict AtP* (effect tiny).
    3. Else: compute disagreement = |AtP* - 2 * probe(0.5)|.
       (If response is linear, 2*probe ≈ AP ≈ AtP*; if nonlinear, they diverge)
    4. If disagreement / max(|AtP*|, eps) > linearity_tol:
         pair flagged uncertain → fall back to AP.
       Else: trust AtP*.

Cost per feature-position:
  - AtP*: 2 passes
  - Probes (per pair with |AtP*| > threshold): 1 pass each
  - AP fallback (per pair flagged uncertain): 1 pass each

Expected: AtP* base is more accurate than AtP base (per-pair adaptive's
starting point), so fewer AP fallbacks needed. Should give Pearson 0.999
at cost between AtP*'s 2 and per-pair adaptive's 16.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    ap = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_path_patching.json")["feature_results"]
    atpstar = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_atpstar_efficient.json")["feature_results"]
    alpha = load_json("checkpoints/sae_layer6_topk64_full.alpha_scaling_all_layers.json")["feature_results"]

    common = sorted(set(ap.keys()) & set(atpstar.keys()) & set(alpha.keys()))
    rows = []
    for fid in common:
        ds = ap[fid]["downstream_layers"]
        nh = ap[fid]["n_heads"]
        ap_arr = np.array(ap[fid]["mean_mediation"])
        atpstar_arr = np.array(atpstar[fid]["mean_atpstar_efficient_mediation"])
        for li, L in enumerate(ds):
            for h in range(nh):
                probe_drops = alpha[fid]["logp_drops"][str(L)][str(h)]["0.5"]
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atpstar": float(atpstar_arr[li, h]),
                    "probe05": float(np.mean(probe_drops)) if probe_drops else 0.0,
                })

    n_total = len(rows)
    ap_v = np.array([r["ap"] for r in rows])
    atpstar_v = np.array([r["atpstar"] for r in rows])
    probe_v = np.array([r["probe05"] for r in rows])

    pearson_atpstar, _ = stats.pearsonr(ap_v, atpstar_v)
    rmse_atpstar = float(np.sqrt(np.mean((ap_v - atpstar_v) ** 2)))

    print(f"=== BASELINE ===")
    print(f"  Efficient AtP*: Pearson = {pearson_atpstar:.4f}, RMSE = {rmse_atpstar:.5f}, cost = 2")
    print()

    n_features = len(common)
    n_positions = 3
    cost_full_ap = 60
    atpstar_cost = 2

    print("=== HYBRID SWEEP (efficient AtP* + per-pair AP fallback) ===")
    print(f"  {'probe_thr':>10} {'lin_tol':>8} {'n_probe':>8} {'n_AP_fb':>8} "
          f"{'Pearson':>10} {'RMSE':>10} {'cost/fpos':>10} {'cost%':>7} {'speedup':>8}")

    sweep = []
    for probe_thr in [0.001, 0.003, 0.005, 0.01, 0.02]:
        for lin_tol in [0.10, 0.20, 0.30, 0.50]:
            pred = atpstar_v.copy()
            n_probe = 0
            n_ap_fb = 0
            for i, r in enumerate(rows):
                if abs(r["atpstar"]) < probe_thr:
                    # tiny: trust AtP*
                    continue
                n_probe += 1
                expected_probe = 0.5 * r["atpstar"]
                if abs(r["atpstar"]) < 1e-8:
                    continue
                # Disagreement: how much does 2*probe differ from AtP*?
                # If linear: probe ≈ AP * 0.5 ≈ AtP* * 0.5
                # If concave (AP > AtP*): probe < AP * 0.5 < AtP* * 0.5, so 2*probe < AtP*
                disagreement = abs(r["atpstar"] - 2 * r["probe05"])
                rel_disagreement = disagreement / max(abs(r["atpstar"]), 1e-6)
                if rel_disagreement > lin_tol:
                    # uncertain → AP fallback
                    pred[i] = r["ap"]
                    n_ap_fb += 1
                # else: trust AtP*

            pr, _ = stats.pearsonr(ap_v, pred)
            rmse = float(np.sqrt(np.mean((ap_v - pred) ** 2)))
            avg_probes = n_probe / (n_features * n_positions)
            avg_ap_fb = n_ap_fb / (n_features * n_positions)
            cost = atpstar_cost + avg_probes + avg_ap_fb
            cost_pct = 100 * cost / cost_full_ap
            sweep.append({
                "probe_thr": probe_thr, "lin_tol": lin_tol,
                "n_probe": n_probe, "n_ap_fb": n_ap_fb,
                "pearson": pr, "rmse": rmse,
                "cost_per_fpos": cost, "cost_pct": cost_pct,
                "speedup": cost_full_ap / cost,
            })
            print(f"  {probe_thr:>10.3f} {lin_tol:>8.2f} {n_probe:>8} {n_ap_fb:>8} "
                  f"{pr:>10.4f} {rmse:>10.5f} {cost:>10.1f} {cost_pct:>6.1f}% {cost_full_ap/cost:>7.2f}x")

    print("\n=== PARETO BEST PER ACCURACY TARGET ===")
    for target in [0.99, 0.995, 0.999, 0.9995, 0.9999]:
        cands = [r for r in sweep if r["pearson"] >= target]
        if not cands:
            print(f"  Pearson >= {target}: no config achieves this")
            continue
        best = min(cands, key=lambda r: r["cost_per_fpos"])
        print(f"  Pearson >= {target}: probe_thr={best['probe_thr']}, lin_tol={best['lin_tol']}, "
              f"Pearson={best['pearson']:.4f}, cost={best['cost_per_fpos']:.1f} ({best['cost_pct']:.1f}% of AP), "
              f"speedup={best['speedup']:.2f}x")

    print("\n=== COMPARISON TO EVERYTHING ===")
    print(f"  Full AP                Pearson 1.0000   cost 60     speedup 1x")
    print(f"  Per-pair adaptive      Pearson 0.9987   cost 15.6   speedup 3.85x")
    print(f"  Efficient AtP*         Pearson 0.9933   cost 2      speedup 30x")
    print(f"  Plain AtP              Pearson 0.9466   cost 2      speedup 30x")
    best_hybrid_995 = next((r for r in sweep if r["pearson"] >= 0.995), None)
    if best_hybrid_995:
        b = min((r for r in sweep if r["pearson"] >= 0.995), key=lambda r: r["cost_per_fpos"])
        print(f"  Hybrid (≥0.995)        Pearson {b['pearson']:.4f}   cost {b['cost_per_fpos']:.1f}   speedup {b['speedup']:.2f}x")
    best_hybrid_999 = next((r for r in sweep if r["pearson"] >= 0.999), None)
    if best_hybrid_999:
        b = min((r for r in sweep if r["pearson"] >= 0.999), key=lambda r: r["cost_per_fpos"])
        print(f"  Hybrid (≥0.999)        Pearson {b['pearson']:.4f}   cost {b['cost_per_fpos']:.1f}   speedup {b['speedup']:.2f}x")

    Path("checkpoints/sae_layer6_topk64_full.hybrid_atpstar_per_pair.json").write_text(
        json.dumps({
            "method": "hybrid_efficient_atpstar_per_pair",
            "n_total": n_total,
            "baseline_atpstar": {"pearson": pearson_atpstar, "rmse": rmse_atpstar},
            "sweep": sweep,
        }, indent=2)
    )
    print("\nsaved checkpoints/sae_layer6_topk64_full.hybrid_atpstar_per_pair.json")


if __name__ == "__main__":
    main()
