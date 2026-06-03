"""Compare AtP* against AtP and AP on the 180 (feature, head) test set.

If AtP* closes the L11 Pearson gap (AtP: 0.78, AP: 1.0):
  - The softmax-correction principle is validated
  - We can build an efficient closed-form version (~2 passes per feature-pos)
  - That version would be the new Pareto winner across the entire frontier
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
    atpstar = load_json("checkpoints/sae_layer6_topk64_full.sae_feature_atpstar.json")["feature_results"]

    common = sorted(set(ap.keys()) & set(atp.keys()) & set(atpstar.keys()))
    print(f"common features: {common}")

    rows = []
    for fid in common:
        downstream_layers = ap[fid]["downstream_layers"]
        n_heads = ap[fid]["n_heads"]
        ap_arr = np.array(ap[fid]["mean_mediation"])
        atp_arr = np.array(atp[fid]["mean_atp_mediation"])
        atpstar_arr = np.array(atpstar[fid]["mean_atpstar_mediation"])
        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": float(ap_arr[li, h]),
                    "atp": float(atp_arr[li, h]),
                    "atpstar": float(atpstar_arr[li, h]),
                })

    ap_v = np.array([r["ap"] for r in rows])
    atp_v = np.array([r["atp"] for r in rows])
    atpstar_v = np.array([r["atpstar"] for r in rows])

    print(f"\n=== OVERALL ({len(rows)} pairs) ===")
    pr_atp, _ = stats.pearsonr(ap_v, atp_v)
    pr_atpstar, _ = stats.pearsonr(ap_v, atpstar_v)
    rmse_atp = float(np.sqrt(np.mean((ap_v - atp_v) ** 2)))
    rmse_atpstar = float(np.sqrt(np.mean((ap_v - atpstar_v) ** 2)))
    print(f"  AtP:   Pearson={pr_atp:.4f}, RMSE={rmse_atp:.5f}")
    print(f"  AtP*:  Pearson={pr_atpstar:.4f}, RMSE={rmse_atpstar:.5f}")

    print(f"\n=== PER LAYER (Pearson vs AP) ===")
    print(f"  {'Layer':>6} {'AtP':>10} {'AtP*':>10}")
    layers = sorted(set(r["L"] for r in rows))
    per_layer_results = {}
    for L in layers:
        sub = [r for r in rows if r["L"] == L]
        ap_L = np.array([r["ap"] for r in sub])
        atp_L = np.array([r["atp"] for r in sub])
        atpstar_L = np.array([r["atpstar"] for r in sub])
        if np.std(ap_L) < 1e-10:
            continue
        pr_a, _ = stats.pearsonr(ap_L, atp_L)
        pr_as, _ = stats.pearsonr(ap_L, atpstar_L)
        per_layer_results[L] = {"atp": pr_a, "atpstar": pr_as}
        print(f"  {L:>6} {pr_a:>10.4f} {pr_as:>10.4f}")

    # Show the largest disagreements between AtP and AtP*, ordered by abs gap
    print(f"\n=== INDIVIDUAL PAIRS WHERE AtP* DIFFERS MOST FROM AtP ===")
    diffs = sorted(rows, key=lambda r: -abs(r["atpstar"] - r["atp"]))[:10]
    print(f"  {'feature':>10} {'L':>3} {'h':>3} {'AP':>10} {'AtP':>10} {'AtP*':>10} {'AtP* fixes?':>13}")
    for r in diffs:
        atp_err = abs(r["ap"] - r["atp"])
        atpstar_err = abs(r["ap"] - r["atpstar"])
        fixes = "YES" if atpstar_err < atp_err * 0.5 else ("partial" if atpstar_err < atp_err else "no")
        print(f"  f{r['feature']:>9} {r['L']:>3} {r['h']:>3} "
              f"{r['ap']:>+10.4f} {r['atp']:>+10.4f} {r['atpstar']:>+10.4f} "
              f"{fixes:>13}")

    # Save
    out_path = "checkpoints/sae_layer6_topk64_full.atpstar_analysis.json"
    Path(out_path).write_text(json.dumps({
        "method": "atpstar_vs_atp",
        "n_pairs": len(rows),
        "overall": {
            "atp": {"pearson": pr_atp, "rmse": rmse_atp},
            "atpstar": {"pearson": pr_atpstar, "rmse": rmse_atpstar},
        },
        "per_layer": {str(L): v for L, v in per_layer_results.items()},
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
