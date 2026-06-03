"""Probe-then-predict: use the alpha=0.5 probe value as the AP estimate
directly, via linear extrapolation: AP_pred = 2 * probe(0.5).

Per-pair adaptive (Finding 13) uses the probe as a BINARY decision rule
(use AtP if linear, fall back to AP if nonlinear). But this throws away
information: the probe value itself is a *measurement* of the effect at
half strength, which already constrains AP at full strength.

This script tests several prediction schemes that USE the probe value:

  S1. probe_x2_only:     AP_pred = 2 * probe(0.5)
  S2. probe_extrap:      AP_pred = 4 * probe(0.25) - 4 * probe(0.5)  (linear extrap from 2 pts)
  S3. probe_quadfit:     fit a quadratic through (0,0), (0.5, probe05),
                          (1, ???) — actually predict the unknown 1 endpoint
                          using AtP_estimate as the slope at 0
                          Then AP_pred = AtP/2 + 2*probe05 - 2*probe05·0  hm this needs care
  S4. linreg_per_layer:  per layer, regress AP on (AtP, probe05). Tests how
                          well a 2-feature linear model predicts AP.

All schemes use existing data (no new model passes). Cost analysis:
  - S1, S2: only need probe(s), no AtP. Cost = N_probes per (L, h) = ~60 passes.
            Same as AP! Not a savings unless we use AtP to filter and only
            probe for the non-trivial pairs.
  - S3, S4: also need AtP. Cost = AtP (2) + probe (1 per non-tiny pair).
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
    alpha_data = load_json("checkpoints/sae_layer6_topk64_full.alpha_scaling_all_layers.json")["feature_results"]

    common = sorted(set(ap_data.keys()) & set(atp_data.keys()) & set(alpha_data.keys()))

    # Build per-(feature, L, h) records combining all sources
    rows = []
    for fid in common:
        ap_arr = np.array(ap_data[fid]["mean_mediation"])
        atp_arr = np.array(atp_data[fid]["mean_atp_mediation"])
        downstream_layers = ap_data[fid]["downstream_layers"]
        n_heads = ap_data[fid]["n_heads"]

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                drops_025 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.25"]
                drops_05 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.5"]
                drops_075 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.75"]
                drops_10 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["1.0"]
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atp": float(atp_arr[li, h]),
                    "probe025": float(np.mean(drops_025)) if drops_025 else 0.0,
                    "probe05": float(np.mean(drops_05)) if drops_05 else 0.0,
                    "probe075": float(np.mean(drops_075)) if drops_075 else 0.0,
                    "ap_alpha": float(np.mean(drops_10)) if drops_10 else 0.0,
                })

    n_total = len(rows)
    ap = np.array([r["ap"] for r in rows])
    atp = np.array([r["atp"] for r in rows])
    p025 = np.array([r["probe025"] for r in rows])
    p05 = np.array([r["probe05"] for r in rows])
    p075 = np.array([r["probe075"] for r in rows])

    layers = sorted(set(r["L"] for r in rows))

    print(f"=== PROBE-THEN-PREDICT METHODS ({n_total} pairs) ===\n")

    methods = {}

    # S1: 2 * probe(0.5) — pure linear extrap from the midpoint measurement
    pred_s1 = 2 * p05
    methods["S1: 2*probe(0.5)"] = pred_s1

    # S2: 4*probe(0.25) used as proxy for AtP (slope near 0), then doubled?
    # If response linear: drop(alpha) = alpha * slope, so slope = drop(0.25)/0.25 = 4*drop(0.25)
    # AP_pred = slope (since secant from 0 to 1 = slope under linearity)
    pred_s2 = 4 * p025
    methods["S2: empirical-AtP from alpha=0.25"] = pred_s2

    # S3: Quadratic interpolation through (0, 0), (0.5, probe05), (slope_at_0 = AtP_est)
    # The quadratic is f(a) = AtP*a + c*a^2 where 2c*0.5 + AtP = derivative at 0.5
    # Actually, simpler: fit through 3 points (0, 0), (0.5, p05), (1, 2*p05 - AtP_est*0)?
    # Let me just fit through 3 actual points: (0.25, p025), (0.5, p05), (0.75, p075)
    # quadratic = a + b*x + c*x^2. Predict f(1) for AP.
    pred_s3 = np.zeros(n_total)
    for i, r in enumerate(rows):
        # 3-point quadratic fit at alphas {0.25, 0.5, 0.75}, predict at 1.0
        xs = np.array([0.25, 0.5, 0.75])
        ys = np.array([r["probe025"], r["probe05"], r["probe075"]])
        if np.std(ys) < 1e-10:
            pred_s3[i] = ys.mean() / 0.5  # fallback
            continue
        # Fit y = a + b*x + c*x^2
        coeffs = np.polyfit(xs, ys, 2)  # returns [c, b, a]
        pred_s3[i] = np.polyval(coeffs, 1.0)
    methods["S3: quadratic 3-point fit, eval at 1.0"] = pred_s3

    # S4: per-layer linear regression: AP ≈ a * AtP + b * probe05 + c
    # Use leave-one-feature-out CV to fit + evaluate (avoids overfitting)
    feature_ids = sorted(set(r["feature"] for r in rows))
    pred_s4 = np.zeros(n_total)
    for held_out in feature_ids:
        train_mask = np.array([r["feature"] != held_out for r in rows])
        test_mask = np.array([r["feature"] == held_out for r in rows])

        # Fit per layer
        for L in layers:
            layer_mask_train = np.array([r["L"] == L for r in rows]) & train_mask
            layer_mask_test = np.array([r["L"] == L for r in rows]) & test_mask
            X_train = np.column_stack([atp[layer_mask_train], p05[layer_mask_train],
                                       np.ones(layer_mask_train.sum())])
            y_train = ap[layer_mask_train]
            if X_train.shape[0] < 4:
                pred_s4[layer_mask_test] = atp[layer_mask_test]  # fallback
                continue
            try:
                coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                X_test = np.column_stack([atp[layer_mask_test], p05[layer_mask_test],
                                           np.ones(layer_mask_test.sum())])
                pred_s4[layer_mask_test] = X_test @ coef
            except Exception:
                pred_s4[layer_mask_test] = atp[layer_mask_test]
    methods["S4: per-layer lin-reg (a*AtP + b*probe05) LOFO-CV"] = pred_s4

    # ---- Evaluate ----
    print(f"  {'Method':>55} {'Pearson':>10} {'RMSE':>10}")
    method_results = {}
    for name, pred in methods.items():
        pr, _ = stats.pearsonr(ap, pred)
        rmse = float(np.sqrt(np.mean((ap - pred) ** 2)))
        method_results[name] = {"pearson": pr, "rmse": rmse, "predictions": pred.tolist()}
        print(f"  {name:>55} {pr:>10.4f} {rmse:>10.5f}")

    # Per-layer breakdown
    print("\n=== PER LAYER PEARSON ===")
    print(f"  {'Layer':>6} " + " ".join(f"{name[:18]:>20}" for name in methods.keys()))
    for L in layers:
        layer_mask = np.array([r["L"] == L for r in rows])
        ap_L = ap[layer_mask]
        if np.std(ap_L) < 1e-10:
            continue
        prs = []
        for name, pred in methods.items():
            p_L = pred[layer_mask]
            if np.std(p_L) < 1e-10:
                prs.append(float("nan"))
            else:
                pr, _ = stats.pearsonr(ap_L, p_L)
                prs.append(pr)
        print(f"  {L:>6} " + " ".join(f"{p:>20.4f}" for p in prs))

    # Save
    Path("checkpoints/sae_layer6_topk64_full.probe_then_predict.json").write_text(
        json.dumps({
            "method": "probe_then_predict",
            "n_total": n_total,
            "method_results": {k: {"pearson": v["pearson"], "rmse": v["rmse"]}
                               for k, v in method_results.items()},
        }, indent=2)
    )
    print("\nsaved checkpoints/sae_layer6_topk64_full.probe_then_predict.json")


if __name__ == "__main__":
    main()
