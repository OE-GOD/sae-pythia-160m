"""Multi-metric SAE quality scorecard.

Aggregates results from prior evaluations to produce a single multi-axis
quality profile for a trained SAE. Addresses the open metric problem in
mech interp: no single number tells the full story of "how well an SAE
captures the model." This tool reads existing JSON outputs and produces a
unified scorecard.

Axes scored:
  1. Variance explained         (reconstruction in activation space)
  2. CE delta loss recovered    (predictive content preserved)
  3. Dead-feature rate          (capacity utilization, lower better)
  4. Cross-seed stability       (replication-resistant features)
  5. Cross-arch stability       (architecture-resistant features)
  6. Auto-interp monosemanticity (feature interpretability rate)

Each axis is normalised to [0, 1] where 1 = best. Reports a profile (per-
axis bars) and a flag for the "weakest axis" of this SAE.

Usage:
    python 11_sae_scorecard.py --ckpt checkpoints/sae_layer6_topk64_full.pt

Reads (if present):
    {ckpt}.eval.json                — basic reconstruction metrics
    {ckpt}.ce_delta.json            — CE-delta evaluation (T3)
    data/replication_analysis.json  — cross-seed (T4)
    data/matryoshka_analysis.json   — cross-arch (T6)
    data/feature_catalog.json       — auto-interp (T5)
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--replication", type=str, default="data/replication_analysis.json")
    p.add_argument("--cross-arch", type=str, default="data/matryoshka_analysis.json")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--out", type=str, default=None,
                   help="Markdown output path. Default: same dir as ckpt, name = scorecard.md")
    return p.parse_args()


def bar(value: float, width: int = 20, max_value: float = 1.0) -> str:
    """ASCII progress bar for value in [0, max_value]."""
    if value is None or not (0.0 <= value <= max_value):
        return f"[{'?' * width}]"
    fill = int((value / max_value) * width)
    return "[" + "█" * fill + "░" * (width - fill) + "]"


def safe_read(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    ckpt_stem = ckpt_path.with_suffix("")

    eval_json = safe_read(str(ckpt_stem) + ".eval.json")
    ce_json = safe_read(str(ckpt_stem) + ".ce_delta.json")
    rep_json = safe_read(args.replication)
    arch_json = safe_read(args.cross_arch)
    cat_json = safe_read(args.catalog)

    # ----------------------------------------------------------------------
    # Extract each axis. Missing data → None (will print as unknown).
    # ----------------------------------------------------------------------
    axes = {}

    # 1. Variance explained
    axes["variance_explained"] = {
        "label": "Reconstruction (variance explained)",
        "value": eval_json.get("var_explained"),
        "interpretation": "fraction of activation variance captured",
        "source": "eval.json",
    }

    # 2. CE delta loss recovered
    loss_rec = ce_json.get("loss_recovered_pct")
    axes["loss_recovered"] = {
        "label": "Predictive preservation (CE-delta loss recovered)",
        "value": (loss_rec / 100.0) if loss_rec is not None else None,
        "interpretation": "fraction of layer's predictive contribution preserved",
        "source": "ce_delta.json",
    }

    # 3. Dead-feature rate — invert so higher = better (1 - dead%)
    pct_dead = eval_json.get("pct_dead")
    axes["alive_rate"] = {
        "label": "Capacity utilization (1 - dead fraction)",
        "value": (1.0 - pct_dead / 100.0) if pct_dead is not None else None,
        "interpretation": "fraction of features that activate on at least one input",
        "source": "eval.json",
    }

    # 4. Cross-seed stability (cos > 0.9 fraction)
    rep_thresholds = rep_json.get("threshold_counts_all_features", {})
    seed_09 = rep_thresholds.get("0.9", {}).get("fraction")
    axes["seed_stability"] = {
        "label": "Cross-seed stability (cos > 0.9 vs different-seed SAE)",
        "value": seed_09,
        "interpretation": "fraction of features replicating across random seeds",
        "source": "replication_analysis.json",
    }

    # 5. Cross-arch stability (cos > 0.9 fraction Matryoshka vs TopK)
    arch_thresholds = arch_json.get("cross_arch_match", {})
    arch_09 = arch_thresholds.get("0.9", {}).get("fraction")
    axes["arch_stability"] = {
        "label": "Cross-architecture stability (cos > 0.9 Matryoshka vs flat)",
        "value": arch_09,
        "interpretation": "fraction of features replicating across SAE architectures",
        "source": "matryoshka_analysis.json",
    }

    # 6. Auto-interp monosemanticity rate
    if cat_json:
        labeled = [v for v in cat_json.values()
                   if isinstance(v, dict) and "classification" in v]
        if labeled:
            mono = sum(1 for v in labeled if v["classification"] == "MONOSEMANTIC")
            mono_rate = mono / len(labeled)
        else:
            mono_rate = None
    else:
        mono_rate = None
    axes["monosemanticity"] = {
        "label": "Auto-interp monosemanticity rate",
        "value": mono_rate,
        "interpretation": "fraction of labeled features classified MONOSEMANTIC",
        "source": "feature_catalog.json",
    }

    # ----------------------------------------------------------------------
    # Render
    # ----------------------------------------------------------------------
    out_lines = []
    out_lines.append(f"# SAE Scorecard — {ckpt_path.name}")
    out_lines.append("")
    out_lines.append("Multi-axis quality profile. Each axis is in [0, 1] where 1 = best.")
    out_lines.append("")
    out_lines.append("```")
    out_lines.append(f"{'AXIS':<48}  {'BAR':<22}  {'VALUE':<8}  SOURCE")
    out_lines.append("-" * 100)
    for key, info in axes.items():
        v = info["value"]
        vstr = f"{v:.4f}" if v is not None else "  n/a "
        b = bar(v) if v is not None else "[" + "?" * 20 + "]"
        out_lines.append(
            f"{info['label']:<48}  {b}  {vstr}  {info['source']}"
        )
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("## Axis-by-axis")
    out_lines.append("")
    for key, info in axes.items():
        out_lines.append(f"**{info['label']}**")
        v = info["value"]
        vstr = f"{v:.4f}" if v is not None else "n/a (missing input)"
        out_lines.append(f"- Value: `{vstr}`")
        out_lines.append(f"- Interpretation: {info['interpretation']}")
        out_lines.append(f"- Source: `{info['source']}`")
        out_lines.append("")

    # Identify weakest axis
    scored = [(k, info["value"]) for k, info in axes.items() if info["value"] is not None]
    if scored:
        weakest = min(scored, key=lambda x: x[1])
        strongest = max(scored, key=lambda x: x[1])
        gap = strongest[1] - weakest[1]
        out_lines.append("## Diagnosis")
        out_lines.append("")
        out_lines.append(f"- Strongest axis: **{axes[strongest[0]]['label']}** ({strongest[1]:.4f})")
        out_lines.append(f"- Weakest axis:   **{axes[weakest[0]]['label']}** ({weakest[1]:.4f})")
        out_lines.append(f"- Range across axes: **{gap:.4f}**")
        out_lines.append("")
        if gap > 0.5:
            out_lines.append(
                f"The {gap:.0%}-point gap between the strongest and weakest axis is the "
                f"central evidence for the 'no single metric tells the full story' "
                f"argument. Reporting any one axis without the others would misrepresent "
                f"this SAE's quality."
            )
        elif gap > 0.2:
            out_lines.append(
                "Notable disagreement between axes. Reporting only the strongest axis "
                "would overstate this SAE's quality."
            )
        else:
            out_lines.append(
                "All axes agree within 0.2. This SAE is uniformly strong or uniformly "
                "weak — unusual; double-check that the cross-seed/arch comparisons used "
                "appropriate counterpart checkpoints."
            )
        out_lines.append("")

    out_lines.append("## How to read this")
    out_lines.append("")
    out_lines.append("- **Variance explained** alone (the metric most papers report) describes only")
    out_lines.append("  reconstruction quality in activation space. It is necessary but not sufficient.")
    out_lines.append("- **CE-delta loss recovered** is the rigorous check that the reconstruction")
    out_lines.append("  preserves the model's downstream behavior.")
    out_lines.append("- **Cross-seed and cross-arch stability** test whether the SAE's *features* are")
    out_lines.append("  a property of the model or an artifact of the SAE training procedure.")
    out_lines.append("- A clean SAE result is one where the weakest axis is still acceptable. A misleading")
    out_lines.append("  SAE result is one with a very strong reconstruction axis but poor stability axes.")
    out_lines.append("")

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".scorecard.md")
    out_path.write_text("\n".join(out_lines))
    print("\n".join(out_lines))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
