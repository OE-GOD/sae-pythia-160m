"""Generate figures for the writeup:
  1. Magnitude sweep curves (the main figure — magnitude × concept-logit-diff)
  2. Three-way comparison bar chart (steering vs whole-patch vs SAE-feature)
  3. Threshold sensitivity heatmap

Outputs all to results/figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plot_magnitude_sweep(sweep_data, out_path):
    """Main figure: magnitude curves per feature, colored by feature category."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Color by category (newline vs other)
    colors = {
        "newline": "#1f77b4",       # blue
        "punctuation_formatting": "#ff7f0e",
        "file_path": "#d62728",      # red
        "citation_latex": "#9467bd",
        "logical_operators": "#8c564b",
        "decimal_numerical": "#2ca02c",
        "math_notation": "#e377c2",
        "subword_bpe": "#7f7f7f",
        "code_keyword": "#bcbd22",
        "french": "#17becf",
    }

    multipliers = sweep_data["magnitude_multipliers"]

    # Separate newline vs other for clarity
    newline_lines = []
    other_lines = []
    for r in sweep_data["results"]:
        cat = r["category"]
        diffs = [m["mean_concept_logit_diff"] for m in r["magnitude_sweep"]]
        color = colors.get(cat, "#444444")
        is_newline = (cat == "newline")
        line = ax.plot(
            multipliers, diffs,
            "o-" if is_newline else "s--",
            color=color,
            linewidth=2 if is_newline else 1.5,
            alpha=0.85,
            markersize=6,
            label=f"f{r['feature_id']}  ({cat})",
        )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(1, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.text(1.05, ax.get_ylim()[1] * 0.95, "natural\nmagnitude", fontsize=8, alpha=0.6, va="top")

    ax.set_xlabel("Intervention magnitude (× natural f_clean)", fontsize=11)
    ax.set_ylabel("Mean concept-logit diff (patched − corrupted)", fontsize=11)
    ax.set_title("SAE feature intervention: magnitude vs causal effect on labeled concept",
                 fontsize=12)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(multipliers)
    ax.set_xticklabels([f"{m:g}x" for m in multipliers])

    # Legend separated by class
    ax.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_three_way_comparison(steering_data, whole_data, feature_data, out_path):
    """Bar chart: driver rate across three intervention methods."""
    methods = []
    driver_counts = []
    therm_counts = []
    n_totals = []

    if steering_data:
        results = steering_data["results"]
        methods.append("Steering\n(synthetic α × W_dec)")
        d = sum(1 for r in results if r["verdict"] == "driver")
        t = sum(1 for r in results if r["verdict"] == "thermometer")
        driver_counts.append(d)
        therm_counts.append(t)
        n_totals.append(len(results))

    if whole_data:
        results = whole_data["results"]
        methods.append("Whole-residual\npatching")
        d = sum(1 for r in results if r["verdict"] == "driver")
        t = sum(1 for r in results if r["verdict"] == "thermometer")
        driver_counts.append(d)
        therm_counts.append(t)
        n_totals.append(len(results))

    if feature_data:
        results = feature_data["results"]
        methods.append("SAE-feature\npatching")
        d = sum(1 for r in results if r["verdict"] == "driver")
        t = sum(1 for r in results if r["verdict"] == "thermometer")
        driver_counts.append(d)
        therm_counts.append(t)
        n_totals.append(len(results))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    width = 0.4

    driver_pct = [100 * d / n for d, n in zip(driver_counts, n_totals)]
    therm_pct = [100 * t / n for t, n in zip(therm_counts, n_totals)]

    bars1 = ax.bar(x - width / 2, driver_pct, width, label="Driver", color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, therm_pct, width, label="Thermometer", color="#d62728", edgecolor="black", linewidth=0.5)

    for bar, pct, n in zip(bars1, driver_pct, n_totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=10)
    for bar, pct in zip(bars2, therm_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Percentage of tested features", fontsize=11)
    ax.set_title("Driver/thermometer classification depends on intervention method", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")

    # n annotation
    for i, n in enumerate(n_totals):
        ax.text(i, -5, f"n={n}", ha="center", fontsize=9, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_threshold_sensitivity(thresh_data, out_path):
    """Bar chart: driver rate across threshold pairs."""
    if not thresh_data:
        return
    sweep = thresh_data["threshold_sweep"]
    labels = [f"d≥{r['driver_threshold']}\nt≤{r['thermometer_threshold']}" for r in sweep]
    therm_pct = [r["pct_thermometer"] for r in sweep]
    driver_pct = [r["pct_driver"] for r in sweep]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, therm_pct, color="#d62728", edgecolor="black", linewidth=0.5,
           label="Thermometer", alpha=0.85)
    ax.bar(x, driver_pct, bottom=therm_pct, color="#1f77b4", edgecolor="black", linewidth=0.5,
           label="Driver", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Percentage of features", fontsize=11)
    ax.set_title("Threshold sensitivity: thermometer-majority is robust across 10 threshold pairs",
                 fontsize=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_two_by_two_classification(noising_data, out_path):
    """2x2 classification: necessity (noising) × sufficiency (denoising)."""
    results = noising_data["results"]
    classes = ["true_driver", "or_circuit", "and_circuit", "thermometer", "ambiguous"]
    class_display = {
        "true_driver": "TRUE\nDRIVER",
        "or_circuit": "OR-circuit\n(redundant)",
        "and_circuit": "AND-circuit\n(needs teammates)",
        "thermometer": "THERMOMETER\n(no causal role)",
        "ambiguous": "Ambiguous",
    }
    class_colors = {
        "true_driver": "#1f77b4",
        "or_circuit": "#ff7f0e",
        "and_circuit": "#2ca02c",
        "thermometer": "#d62728",
        "ambiguous": "#7f7f7f",
    }

    counts = {c: 0 for c in classes}
    feature_lists = {c: [] for c in classes}
    for r in results:
        cls_key = r["combined_classification"].lower()
        if cls_key in counts:
            counts[cls_key] += 1
            feature_lists[cls_key].append(f"f{r['feature_id']}")

    n = len(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                     gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Left: 2x2 grid as a heatmap-style table
    grid_data = [
        ["TRUE DRIVER",       "AND-circuit"],   # row 0: necessary
        ["OR-circuit",        "THERMOMETER"],   # row 1: not necessary
    ]
    grid_counts = [
        [counts["true_driver"], counts["and_circuit"]],
        [counts["or_circuit"],  counts["thermometer"]],
    ]
    grid_colors = [
        ["#1f77b4", "#2ca02c"],
        ["#ff7f0e", "#d62728"],
    ]

    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.invert_yaxis()
    for i in range(2):
        for j in range(2):
            count = grid_counts[i][j]
            label = grid_data[i][j]
            color = grid_colors[i][j]
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, alpha=0.25,
                                  edgecolor="black", linewidth=1.5)
            ax1.add_patch(rect)
            ax1.text(j + 0.5, i + 0.32, label, ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color)
            ax1.text(j + 0.5, i + 0.65, f"n = {count}", ha="center", va="center",
                     fontsize=14, color="black")

    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(["Sufficient\n(denoising = driver)",
                          "Not sufficient\n(denoising = thermometer)"], fontsize=10)
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(["Necessary\n(noising)", "Not necessary\n(noising)"], fontsize=10)
    ax1.tick_params(axis="both", which="both", length=0)
    ax1.set_title(f"2×2 Necessity × Sufficiency Classification (n={n})",
                  fontsize=12)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Right: per-feature breakdown
    feature_rows = []
    for cls in ["true_driver", "or_circuit", "and_circuit", "thermometer", "ambiguous"]:
        for fid_str in feature_lists[cls]:
            feature_rows.append((cls, fid_str))

    ax2.barh(
        range(len(feature_rows)),
        [1] * len(feature_rows),
        color=[class_colors[cls] for cls, _ in feature_rows],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    feature_labels = []
    for cls, fid_str in feature_rows:
        r = next(rr for rr in results if f"f{rr['feature_id']}" == fid_str)
        feature_labels.append(f"{fid_str}  {r['label'][:32]}")
    ax2.set_yticks(range(len(feature_rows)))
    ax2.set_yticklabels(feature_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ax2.set_xlim(0, 1)
    ax2.set_title("Per-feature classification", fontsize=12)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Legend showing class colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=class_colors[c], edgecolor="black", linewidth=0.5,
              label=class_display[c].replace("\n", " "))
        for c in ["true_driver", "or_circuit", "and_circuit", "thermometer", "ambiguous"]
        if counts[c] > 0
    ]
    ax2.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-prefix", type=str, default="checkpoints/sae_layer6_topk64_full")
    p.add_argument("--out-dir", type=str, default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_data = load_json(f"{args.ckpt_prefix}.magnitude_sweep.json")
    steering_data = load_json(f"{args.ckpt_prefix}.thermometer_at_scale.json")
    whole_data = load_json(f"{args.ckpt_prefix}.patching_driver_thermometer.json")
    feature_data = load_json(f"{args.ckpt_prefix}.sae_feature_patching.json")
    thresh_data = load_json(f"{args.ckpt_prefix}.threshold_sensitivity.json")
    noising_data = load_json(f"{args.ckpt_prefix}.sae_feature_noising.json")

    print("generating figures...")
    if sweep_data:
        plot_magnitude_sweep(sweep_data, str(out_dir / "fig1_magnitude_sweep.png"))
    if steering_data or whole_data or feature_data:
        plot_three_way_comparison(steering_data, whole_data, feature_data,
                                   str(out_dir / "fig2_three_way_comparison.png"))
    if thresh_data:
        plot_threshold_sensitivity(thresh_data, str(out_dir / "fig3_threshold_sensitivity.png"))
    if noising_data:
        plot_two_by_two_classification(noising_data,
                                        str(out_dir / "fig4_two_by_two_classification.png"))

    print(f"\nfigures in {out_dir}")


if __name__ == "__main__":
    main()
