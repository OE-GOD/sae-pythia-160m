"""Compare attribution patching (AtP) vs activation patching (AP).

Loads the per-position results from both methods, computes:
  - Pearson and Spearman correlation per feature
  - Scatter plot AtP vs AP
  - Identifies failure modes: where do they disagree?

Failure mode analysis groups by:
  - Layer (do earlier or later layers diverge more?)
  - Effect magnitude (does AtP underestimate large effects?)
  - Sign agreement
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    ap_path = "checkpoints/sae_layer6_topk64_full.sae_feature_path_patching.json"
    atp_path = "checkpoints/sae_layer6_topk64_full.sae_feature_atp.json"

    ap_data = load_json(ap_path)["feature_results"]
    atp_data = load_json(atp_path)["feature_results"]

    # Note: script 30's JSON keys feature_results by feature_id as INT, but JSON
    # serializes int keys as strings. So both should have string keys here.
    common_features = set(ap_data.keys()) & set(atp_data.keys())
    print(f"common features: {sorted(common_features)}")

    # Collect paired (AP, AtP) values
    rows = []  # list of dicts: feature, position, L, h, ap, atp
    for fid in common_features:
        ap_feature = ap_data[fid]
        atp_feature = atp_data[fid]

        downstream_layers = ap_feature["downstream_layers"]
        n_heads = ap_feature["n_heads"]

        # AP per-position results structure: per_position_results, list of
        # dicts each with {"position", ..., logp_drop per (L, h)}
        # But looking at script 30, per_position_results is a list of dicts
        # with the full mediation matrix.
        # Actually script 30 saves: per_position_results is a list of dicts
        # with position, original_activation, etc. but the mediation matrix is
        # NOT stored per position — only the mean is stored as a separate field.

        # Re-checking script 30 output structure:
        # "per_position_results" = list of dicts (one per firing position)
        #   each dict has individual per-position data
        # mean_mediation (mediation averaged over positions)
        # So we need to use mean_mediation since we don't have per-position
        # for AP.

        ap_mean = np.array(ap_feature["mean_mediation"])
        atp_mean = np.array(atp_feature["mean_atp_mediation"])

        assert ap_mean.shape == atp_mean.shape, f"shape mismatch for f{fid}"

        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                rows.append({
                    "feature": fid,
                    "L": L,
                    "h": h,
                    "ap": float(ap_mean[li, h]),
                    "atp": float(atp_mean[li, h]),
                })

    print(f"total paired points: {len(rows)}")

    # ---------- Overall correlation ----------
    ap_arr = np.array([r["ap"] for r in rows])
    atp_arr = np.array([r["atp"] for r in rows])

    pearson_r, pearson_p = stats.pearsonr(ap_arr, atp_arr)
    spearman_r, spearman_p = stats.spearmanr(ap_arr, atp_arr)
    print(f"\n=== OVERALL (all {len(rows)} points) ===")
    print(f"  Pearson r:  {pearson_r:+.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:+.4f}  (p = {spearman_p:.2e})")

    # ---------- Per-feature correlation ----------
    print(f"\n=== PER FEATURE ===")
    per_feature = {}
    for fid in common_features:
        sub = [r for r in rows if r["feature"] == fid]
        ap_sub = np.array([r["ap"] for r in sub])
        atp_sub = np.array([r["atp"] for r in sub])
        pr, _ = stats.pearsonr(ap_sub, atp_sub)
        sr, _ = stats.spearmanr(ap_sub, atp_sub)
        per_feature[fid] = {"pearson": pr, "spearman": sr, "n": len(sub)}
        print(f"  f{fid}: Pearson {pr:+.4f}, Spearman {sr:+.4f}  (n={len(sub)})")

    # ---------- Per-layer correlation ----------
    print(f"\n=== PER LAYER ===")
    per_layer = {}
    layers = sorted(set(r["L"] for r in rows))
    for L in layers:
        sub = [r for r in rows if r["L"] == L]
        ap_sub = np.array([r["ap"] for r in sub])
        atp_sub = np.array([r["atp"] for r in sub])
        if np.std(ap_sub) == 0 or np.std(atp_sub) == 0:
            pr = sr = float("nan")
        else:
            pr, _ = stats.pearsonr(ap_sub, atp_sub)
            sr, _ = stats.spearmanr(ap_sub, atp_sub)
        per_layer[L] = {"pearson": pr, "spearman": sr, "n": len(sub)}
        print(f"  Layer {L}: Pearson {pr:+.4f}, Spearman {sr:+.4f}  (n={len(sub)})")

    # ---------- Magnitude-binned analysis ----------
    print(f"\n=== BY ACTIVATION-PATCHING EFFECT MAGNITUDE ===")
    abs_ap = np.abs(ap_arr)
    bins = [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, np.inf)]
    for lo, hi in bins:
        mask = (abs_ap >= lo) & (abs_ap < hi)
        if mask.sum() < 3:
            continue
        ap_sub = ap_arr[mask]
        atp_sub = atp_arr[mask]
        pr, _ = stats.pearsonr(ap_sub, atp_sub)
        # AtP/AP ratio: how well does AtP match in magnitude?
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(ap_sub) > 1e-6, atp_sub / ap_sub, np.nan)
        ratio_median = float(np.nanmedian(ratio))
        ratio_mean = float(np.nanmean(ratio))
        print(f"  |AP| in [{lo:.3f}, {hi if hi != np.inf else '∞'}]: "
              f"n={mask.sum()}, Pearson={pr:+.4f}, "
              f"AtP/AP median={ratio_median:+.3f}")

    # ---------- Sign agreement ----------
    sign_agreement = np.mean(np.sign(ap_arr) == np.sign(atp_arr))
    print(f"\n=== SIGN AGREEMENT ===")
    print(f"  Same sign: {sign_agreement:.1%}")

    # Disagreement examples (top 5)
    print(f"\n=== LARGEST ABSOLUTE DISAGREEMENTS (top 10) ===")
    disagreements = sorted(rows, key=lambda r: -abs(r["ap"] - r["atp"]))[:10]
    print(f"  {'feature':<8} {'L':>3} {'h':>3} {'AP':>10} {'AtP':>10} {'|Δ|':>9}")
    for r in disagreements:
        print(f"  f{r['feature']:<7} {r['L']:>3} {r['h']:>3} "
              f"{r['ap']:>+10.4f} {r['atp']:>+10.4f} {abs(r['ap']-r['atp']):>9.4f}")

    # ---------- Scatter plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: overall scatter colored by feature
    colors_per_feature = {fid: c for fid, c in zip(sorted(common_features),
                                                     ["#1f77b4", "#ff7f0e", "#2ca02c"])}
    ax = axes[0]
    for fid in sorted(common_features):
        sub = [r for r in rows if r["feature"] == fid]
        ap_sub = [r["ap"] for r in sub]
        atp_sub = [r["atp"] for r in sub]
        ax.scatter(ap_sub, atp_sub, c=colors_per_feature[fid],
                    label=f"f{fid} (n={len(sub)})", alpha=0.55, s=24,
                    edgecolors="black", linewidths=0.3)

    # y=x line
    lim_min = min(min(ap_arr), min(atp_arr)) * 1.05
    lim_max = max(max(ap_arr), max(atp_arr)) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4,
            linewidth=1, label="y = x (perfect agreement)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Activation patching effect (logp_drop)", fontsize=11)
    ax.set_ylabel("Attribution patching estimate (logp_drop)", fontsize=11)
    ax.set_title(f"AtP vs Activation Patching\n"
                 f"Pearson r = {pearson_r:.3f}, Spearman r = {spearman_r:.3f}",
                 fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: AtP/AP ratio by |AP| bin
    ax = axes[1]
    bin_centers = []
    ratio_medians = []
    ratio_iqr_lo = []
    ratio_iqr_hi = []
    bin_n = []
    for lo, hi in bins:
        mask = (abs_ap >= lo) & (abs_ap < hi)
        if mask.sum() < 3:
            continue
        ap_sub = ap_arr[mask]
        atp_sub = atp_arr[mask]
        ratio = np.where(np.abs(ap_sub) > 1e-6, atp_sub / ap_sub, np.nan)
        ratio = ratio[~np.isnan(ratio)]
        hi_str = f"{hi:.3f}" if hi != np.inf else "∞"
        bin_label = f"[{lo:.3f}, {hi_str})"
        bin_centers.append((lo + (hi if hi != np.inf else lo * 4)) / 2)
        ratio_medians.append(np.median(ratio))
        ratio_iqr_lo.append(np.percentile(ratio, 25))
        ratio_iqr_hi.append(np.percentile(ratio, 75))
        bin_n.append(mask.sum())

    x_pos = np.arange(len(bin_centers))
    ax.errorbar(x_pos, ratio_medians,
                 yerr=[np.array(ratio_medians) - np.array(ratio_iqr_lo),
                       np.array(ratio_iqr_hi) - np.array(ratio_medians)],
                 fmt="o", color="#1f77b4", markersize=8, capsize=6, capthick=1.5)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfect (AtP/AP = 1)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"|AP|\n[{lo:.3f},\n{hi if hi != np.inf else '∞'})\n(n={n})"
         for (lo, hi), n in zip(bins[:len(bin_centers)], bin_n)],
        fontsize=9
    )
    ax.set_ylabel("AtP / AP ratio (median ± IQR)", fontsize=11)
    ax.set_title("AtP magnitude calibration across |AP| bins", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = "results/figures/fig6_atp_vs_ap_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out_path}")

    # Save analysis JSON
    Path("checkpoints/sae_layer6_topk64_full.atp_vs_ap_comparison.json").write_text(
        json.dumps({
            "n_points": len(rows),
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "sign_agreement": float(sign_agreement),
            "per_feature": per_feature,
            "per_layer": {str(L): v for L, v in per_layer.items()},
            "top_disagreements": disagreements[:10],
        }, indent=2)
    )
    print("saved checkpoints/sae_layer6_topk64_full.atp_vs_ap_comparison.json")


if __name__ == "__main__":
    main()
