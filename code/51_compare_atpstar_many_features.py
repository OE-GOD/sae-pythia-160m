"""Compare efficient AtP* vs AP on 15 diverse features (5 newline contexts +
8 non-newline categories + 2 features from semantic clusters).

Tests within-model generalization of efficient AtP*: does the 0.993 Pearson
result on 3 newline drivers hold across diverse feature semantics?
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    ap = json.load(open("checkpoints/sae_layer6_topk64_full.ap_15features.json"))["feature_results"]
    atpstar = json.load(open("checkpoints/sae_layer6_topk64_full.atpstar_efficient_15features.json"))["feature_results"]
    cat = json.load(open("data/feature_catalog.json"))

    common = sorted(set(ap.keys()) & set(atpstar.keys()), key=lambda x: int(x))
    print(f"common features: {len(common)}")

    rows = []
    for fid in common:
        ds = ap[fid]["downstream_layers"]
        nh = ap[fid]["n_heads"]
        ap_arr = np.array(ap[fid]["mean_mediation"])
        atpstar_arr = np.array(atpstar[fid]["mean_atpstar_efficient_mediation"])
        for li, L in enumerate(ds):
            for h in range(nh):
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atpstar": float(atpstar_arr[li, h]),
                })

    n_total = len(rows)
    ap_v = np.array([r["ap"] for r in rows])
    atpstar_v = np.array([r["atpstar"] for r in rows])

    print(f"\n=== OVERALL ({n_total} pairs from {len(common)} features) ===")
    pr, _ = stats.pearsonr(ap_v, atpstar_v)
    rmse = float(np.sqrt(np.mean((ap_v - atpstar_v) ** 2)))
    print(f"  Pearson(AtP*, AP) = {pr:.4f}, RMSE = {rmse:.5f}")
    print(f"  Sign agreement: {np.mean(np.sign(ap_v) == np.sign(atpstar_v)):.1%}")

    print(f"\n=== PER LAYER ===")
    layers = sorted(set(r["L"] for r in rows))
    print(f"  {'Layer':>6} {'n':>5} {'Pearson':>10} {'RMSE':>10}")
    for L in layers:
        sub = [r for r in rows if r["L"] == L]
        a = np.array([r["ap"] for r in sub])
        s = np.array([r["atpstar"] for r in sub])
        if np.std(a) < 1e-10 or np.std(s) < 1e-10:
            continue
        pr_L, _ = stats.pearsonr(a, s)
        rmse_L = float(np.sqrt(np.mean((a - s) ** 2)))
        print(f"  {L:>6} {len(sub):>5} {pr_L:>10.4f} {rmse_L:>10.5f}")

    print(f"\n=== PER FEATURE ===")
    print(f"  {'feature':>10} {'label':<46} {'Pearson':>10} {'RMSE':>10}")
    per_feature = {}
    for fid in common:
        sub = [r for r in rows if r["feature"] == fid]
        a = np.array([r["ap"] for r in sub])
        s = np.array([r["atpstar"] for r in sub])
        if np.std(a) < 1e-10 or np.std(s) < 1e-10:
            pr_f = float("nan")
        else:
            pr_f, _ = stats.pearsonr(a, s)
        rmse_f = float(np.sqrt(np.mean((a - s) ** 2)))
        label = cat.get(fid, {}).get("label", "(no label)")[:44]
        per_feature[fid] = {"pearson": pr_f, "rmse": rmse_f, "label": label}
        print(f"  f{fid:>9} {label:<46} {pr_f:>10.4f} {rmse_f:>10.5f}")

    # Show pairs with largest disagreements
    print(f"\n=== TOP 10 LARGEST AbsDIFF (AtP* - AP) ===")
    rows_sorted = sorted(rows, key=lambda r: -abs(r["atpstar"] - r["ap"]))[:10]
    print(f"  {'feature':>10} {'L':>3} {'h':>3} {'AP':>10} {'AtP*':>10} {'|diff|':>9}")
    for r in rows_sorted:
        print(f"  f{r['feature']:>9} {r['L']:>3} {r['h']:>3} "
              f"{r['ap']:>+10.4f} {r['atpstar']:>+10.4f} {abs(r['ap']-r['atpstar']):>9.4f}")

    # Save
    Path("checkpoints/sae_layer6_topk64_full.atpstar_15features_analysis.json").write_text(
        json.dumps({
            "method": "atpstar_vs_ap_15features",
            "n_features": len(common),
            "n_pairs": n_total,
            "overall_pearson": pr,
            "overall_rmse": rmse,
            "per_feature": per_feature,
        }, indent=2)
    )
    print("\nsaved checkpoints/sae_layer6_topk64_full.atpstar_15features_analysis.json")


if __name__ == "__main__":
    main()
