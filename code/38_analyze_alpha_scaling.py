"""Analyze the alpha-scaling diagnostic from script 37.

Per (feature, head, position) at each test layer, we have logp_drop at
alphas {0.25, 0.5, 0.75, 1.0}.

If AtP is exact at this layer/head, logp_drop should be LINEAR in alpha:
    logp_drop(alpha) = alpha * slope
Equivalently:
    logp_drop(1.0) / (4 * logp_drop(0.25)) == 1.0

If the response is CONCAVE (saturated) — the slope at alpha=0 (= AtP) is
larger than the avg slope (= AP at alpha=1):
    logp_drop(1.0) < 4 * logp_drop(0.25)
    -> ratio < 1.0
    -> AtP overestimates the alpha=1 effect

If the response is CONVEX:
    ratio > 1.0
    -> AtP underestimates

We also compare:
  - "Empirical AtP" = logp_drop(0.25) / 0.25 (slope near alpha=0, robust est of AtP)
  - "AP" = logp_drop(1.0)
  - Empirical AtP / AP ratio tells us by how much AtP overestimates

Cross-layer comparison answers: is the AtP-vs-AP gap at deep layers
explained by curvature, or by something else (noise, sign flips)?
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    data = load_json("checkpoints/sae_layer6_topk64_full.alpha_scaling.json")["feature_results"]

    rows = []
    for fid, fdata in data.items():
        alphas = [float(a) for a in fdata["alphas"]]
        layers = [int(L) for L in fdata["layers_to_test"]]
        n_heads = fdata["n_heads"]
        n_pos = len(fdata["per_position_meta"])

        for L in layers:
            for h in range(n_heads):
                # logp_drop[alpha] is a list of n_pos values
                drops_by_alpha = {a: fdata["logp_drops"][str(L)][str(h)][str(a)]
                                  for a in alphas}
                # Per position
                for pos_i in range(n_pos):
                    drops = {a: drops_by_alpha[a][pos_i] for a in alphas}
                    rows.append({
                        "feature": fid,
                        "L": L,
                        "h": h,
                        "pos": pos_i,
                        "drops": drops,
                    })

    print(f"loaded {len(rows)} (feature, layer, head, position) measurements")

    # ---- Linearity diagnostic ----
    # ratio = logp_drop(1.0) / (4 * logp_drop(0.25))
    # Filter to pairs where the effect is non-trivial (|drop_at_1| > 0.005)
    # to avoid noise-dominated regimes.
    alphas = [0.25, 0.5, 0.75, 1.0]

    print("\n=== LINEARITY ANALYSIS (effect at alpha=1 vs 4x effect at alpha=0.25) ===")
    print("  Ratio < 1.0 => concave / saturated (AtP overestimates)")
    print("  Ratio = 1.0 => linear (AtP exact)")
    print("  Ratio > 1.0 => convex (AtP underestimates)")
    print()
    print(f"  {'Layer':>6}  {'n_nontrivial':>12}  {'ratio mean':>10}  {'ratio median':>12}  "
          f"{'ratio p25':>10}  {'ratio p75':>10}  {'frac concave':>12}")

    layer_results = {}
    for L in sorted(set(r["L"] for r in rows)):
        sub = [r for r in rows if r["L"] == L]
        ratios = []
        for r in sub:
            d1 = r["drops"][1.0]
            d025 = r["drops"][0.25]
            # Only use pairs with substantial effect AND consistent sign
            if abs(d1) < 0.005 or np.sign(d1) != np.sign(d025) or abs(d025) < 1e-6:
                continue
            ratios.append(d1 / (4 * d025))

        ratios = np.array(ratios)
        if len(ratios) == 0:
            print(f"  L{L:<5}  (no non-trivial points)")
            continue
        frac_concave = float(np.mean(ratios < 1.0))
        layer_results[L] = {
            "ratios": ratios.tolist(),
            "n": len(ratios),
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
            "p25": float(np.percentile(ratios, 25)),
            "p75": float(np.percentile(ratios, 75)),
            "frac_concave": frac_concave,
        }
        print(f"  L{L:<5}  {len(ratios):>12}  "
              f"{np.mean(ratios):>10.3f}  {np.median(ratios):>12.3f}  "
              f"{np.percentile(ratios, 25):>10.3f}  {np.percentile(ratios, 75):>10.3f}  "
              f"{frac_concave:>11.0%}")

    # ---- Empirical "AtP" vs "AP" ratio ----
    print("\n=== EMPIRICAL AtP vs AP RATIO ===")
    print("  Empirical AtP = slope at alpha=0 = drop(0.25) / 0.25")
    print("  AP = drop(1.0)")
    print("  AtP / AP ratio > 1.0 => AtP overestimates (saturation)")
    print()
    print(f"  {'Layer':>6}  {'n':>5}  {'AtP/AP mean':>12}  {'AtP/AP median':>14}")
    for L in sorted(set(r["L"] for r in rows)):
        sub = [r for r in rows if r["L"] == L]
        ratios = []
        for r in sub:
            d1 = r["drops"][1.0]
            d025 = r["drops"][0.25]
            if abs(d1) < 0.005 or np.sign(d1) != np.sign(d025) or abs(d025) < 1e-6:
                continue
            emp_atp = d025 / 0.25
            ratios.append(emp_atp / d1)
        if not ratios:
            continue
        ratios = np.array(ratios)
        print(f"  L{L:<5}  {len(ratios):>5}  {np.mean(ratios):>12.3f}  {np.median(ratios):>14.3f}")

    # ---- Plot: response curves at L7 vs L11 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # For each layer plotted, show ~20 randomly sampled response curves
    layers_to_plot = sorted(set(r["L"] for r in rows))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(layers_to_plot)))

    for ax, L, color in zip(axes, layers_to_plot, colors):
        sub = [r for r in rows if r["L"] == L]
        # Filter to non-trivial cases
        nontrivial = []
        for r in sub:
            d1 = r["drops"][1.0]
            d025 = r["drops"][0.25]
            if abs(d1) >= 0.005 and np.sign(d1) == np.sign(d025) and abs(d025) > 1e-6:
                nontrivial.append(r)

        # Plot all non-trivial response curves (normalize so endpoint = 1.0
        # so we see SHAPE not MAGNITUDE)
        for r in nontrivial:
            d = r["drops"]
            normalized = [d[a] / d[1.0] for a in alphas]
            ax.plot(alphas, normalized, color=color, alpha=0.18, linewidth=1)

        # Plot the perfect linear reference
        ax.plot(alphas, alphas, "k--", linewidth=1.5, alpha=0.7,
                 label="Linear (AtP exact)")
        ax.set_xlabel(r"$\alpha$ (perturbation scaling)", fontsize=11)
        ax.set_ylabel("Normalized logp_drop ($/ $ drop at $\\alpha=1$)", fontsize=11)
        ax.set_title(f"Layer {L}: response curves ({len(nontrivial)} pairs)",
                     fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "results/figures/fig9_alpha_scaling_diagnostic.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    # ---- Save analysis ----
    Path("checkpoints/sae_layer6_topk64_full.alpha_scaling_analysis.json").write_text(
        json.dumps({
            "method": "alpha_scaling_analysis",
            "layer_results": {str(L): v for L, v in layer_results.items()},
        }, indent=2)
    )
    print("saved checkpoints/sae_layer6_topk64_full.alpha_scaling_analysis.json")


if __name__ == "__main__":
    main()
