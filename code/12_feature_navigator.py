"""Width-navigation tool — per-feature stability profile across SAE conditions.

Given a reference SAE and a set of comparison SAEs (different widths, seeds,
architectures), this tool:

  1. For each feature in the reference SAE, finds its best cosine match in
     every comparison SAE's decoder columns.
  2. Builds a stability profile: how many comparison conditions does this
     feature replicate at cos > 0.9?
  3. Identifies the "fully stable" subset — features that survive every
     comparison. These are the strongest candidates for "real model
     features" rather than SAE-training artifacts.
  4. Cross-references the labeled features (T5 auto-interp catalog) so we
     can name which concepts are stable.

For our setup the reference is the W2 TopK 16k SAE. Comparisons are:
  - TopK 4k       (different width — smaller)
  - TopK 64k      (different width — larger)
  - TopK seed=42  (different random seed)
  - Matryoshka    (different architecture)

Outputs:
  data/feature_navigator.json   — full per-feature profiles
  notes/stable_features.md      — human-readable shortlist + statistics

Usage:
    python 12_feature_navigator.py
    python 12_feature_navigator.py --query 12117          # look up one feature
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # TopKSAE
_mat_src = (Path(__file__).parent / "02c_train_sae_matryoshka.py").read_text().split("def parse_args")[0]
exec(_mat_src)  # MatryoshkaSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--ckpt-4k", type=str, default="checkpoints/sae_layer6_topk64_w4k.pt")
    p.add_argument("--ckpt-64k", type=str, default="checkpoints/sae_layer6_topk64_w64k.pt")
    p.add_argument("--ckpt-seed42", type=str, default="checkpoints/sae_layer6_topk64_seed42.pt")
    p.add_argument("--ckpt-matryoshka", type=str, default="checkpoints/sae_layer6_matryoshka.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="cos sim threshold for counting a feature as 'stable' under a comparison")
    p.add_argument("--out-json", type=str, default="data/feature_navigator.json")
    p.add_argument("--out-md", type=str, default="notes/stable_features.md")
    p.add_argument("--query", type=int, default=None,
                   help="If set, print full stability profile for this feature id and exit")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae_W(path: str, device: str) -> tuple[torch.Tensor, dict]:
    """Load any SAE checkpoint (TopK or Matryoshka), return its decoder W_dec
    and metadata."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "topk")
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    if arch == "matryoshka":
        sae = MatryoshkaSAE(d_model, n_features, k_levels=ckpt["k_levels"]).to(device)  # noqa: F821
    else:
        sae = TopKSAE(d_model, n_features, k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae.W_dec.data, {"arch": arch, "n_features": n_features, "ckpt_path": path}


def unit_norm(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"\nloading reference: {args.reference}")
    W_ref, ref_meta = load_sae_W(args.reference, device)
    W_ref = unit_norm(W_ref)
    n_features = ref_meta["n_features"]
    print(f"  n_features={n_features}  arch={ref_meta['arch']}")

    # Load all comparison SAEs that exist
    comparison_paths = {
        "topk_4k":     args.ckpt_4k,
        "topk_64k":    args.ckpt_64k,
        "topk_seed42": args.ckpt_seed42,
        "matryoshka":  args.ckpt_matryoshka,
    }
    comparisons = {}
    for name, path in comparison_paths.items():
        if Path(path).exists():
            print(f"loading {name}: {path}")
            W, meta = load_sae_W(path, device)
            comparisons[name] = {"W": unit_norm(W), "meta": meta}
        else:
            print(f"  (skipping {name}: {path} not found)")

    if not comparisons:
        print("\nERROR: no comparison SAEs found. Train T1/T4/T6 checkpoints first.")
        return

    # Load labeled catalog if present
    catalog_path = Path(args.catalog)
    catalog = {}
    if catalog_path.exists():
        raw = json.loads(catalog_path.read_text())
        catalog = {int(k): v for k, v in raw.items() if isinstance(v, dict) and "error" not in v}
        print(f"\nloaded {len(catalog)} labeled features from catalog")

    # ------------------------------------------------------------------
    # Compute best-match cosine sim per feature, across each comparison
    # ------------------------------------------------------------------
    print(f"\ncomputing best-match cosine similarity per feature across {len(comparisons)} comparisons...")
    best_idx = {}
    best_sim = {}
    for name, c in comparisons.items():
        # sims: (n_features, n_features_comparison)
        sims = W_ref.T @ c["W"]
        vals, idxs = sims.max(dim=1)
        best_sim[name] = vals.cpu().numpy()
        best_idx[name] = idxs.cpu().numpy()

    # ------------------------------------------------------------------
    # Build per-feature profile
    # ------------------------------------------------------------------
    n_comparisons = len(comparisons)
    stability_score = np.zeros(n_features, dtype=np.int32)
    for name in comparisons:
        stability_score += (best_sim[name] > args.threshold).astype(np.int32)

    # If user asked for a query, print and exit
    if args.query is not None:
        fid = args.query
        if fid < 0 or fid >= n_features:
            print(f"\nERROR: feature id {fid} out of range [0, {n_features})")
            return
        label = catalog.get(fid, {}).get("label", "(unlabeled)")
        print(f"\n=== FEATURE #{fid}: {label} ===")
        print(f"  stability score: {stability_score[fid]} / {n_comparisons}")
        print(f"  per-comparison best matches (threshold cos > {args.threshold}):")
        for name in comparisons:
            cos = best_sim[name][fid]
            idx = best_idx[name][fid]
            mark = "✓" if cos > args.threshold else " "
            print(f"    {name:<14} → feature #{int(idx):<6}  cos={cos:.4f}  {mark}")
        return

    # ------------------------------------------------------------------
    # Aggregate stats: distribution of stability scores
    # ------------------------------------------------------------------
    print(f"\n=== STABILITY DISTRIBUTION (threshold cos > {args.threshold}) ===")
    print(f"  {n_features:,} features, {n_comparisons} comparison conditions")
    print()
    print(f"  stability score | n_features | fraction | description")
    print(f"  --------------- | ---------- | -------- | --------------")
    for s in range(n_comparisons + 1):
        n = int((stability_score == s).sum())
        frac = n / n_features
        desc = "unstable everywhere" if s == 0 else (
            f"stable in {s}/{n_comparisons} condition" if s == 1
            else f"stable in {s}/{n_comparisons} conditions"
        )
        if s == n_comparisons:
            desc += " — ★ FULLY STABLE ★"
        print(f"  {s:>15} | {n:>10,} | {frac:>8.4f} | {desc}")

    n_fully_stable = int((stability_score == n_comparisons).sum())
    fully_stable_ids = np.where(stability_score == n_comparisons)[0].tolist()

    # ------------------------------------------------------------------
    # Build full output JSON
    # ------------------------------------------------------------------
    nav = {
        "reference": args.reference,
        "threshold": args.threshold,
        "n_features": n_features,
        "n_comparisons": n_comparisons,
        "comparison_paths": {name: c["meta"]["ckpt_path"] for name, c in comparisons.items()},
        "stability_distribution": {
            str(s): int((stability_score == s).sum()) for s in range(n_comparisons + 1)
        },
        "n_fully_stable": n_fully_stable,
        "fully_stable_feature_ids": fully_stable_ids,
        "per_feature": {},
    }

    # Per-feature entries (only emit for labeled or fully-stable to keep size sane)
    relevant_ids = set(catalog.keys()) | set(fully_stable_ids)
    for fid in sorted(relevant_ids):
        entry = {
            "feature_id": int(fid),
            "stability_score": int(stability_score[fid]),
            "label": catalog.get(int(fid), {}).get("label", None),
            "classification": catalog.get(int(fid), {}).get("classification", None),
            "matches": {
                name: {
                    "best_idx": int(best_idx[name][fid]),
                    "cos_sim": float(best_sim[name][fid]),
                    "stable": bool(best_sim[name][fid] > args.threshold),
                }
                for name in comparisons
            },
        }
        nav["per_feature"][int(fid)] = entry

    Path(args.out_json).parent.mkdir(exist_ok=True)
    Path(args.out_json).write_text(json.dumps(nav, indent=2))
    print(f"\nsaved {args.out_json}")

    # ------------------------------------------------------------------
    # Build human-readable markdown report
    # ------------------------------------------------------------------
    md = []
    md.append(f"# Stable Features — {Path(args.reference).name}")
    md.append("")
    md.append(f"Reference SAE: `{args.reference}`")
    md.append(f"Threshold for 'stable': cos sim > **{args.threshold}**")
    md.append("")
    md.append(f"## Comparison conditions ({n_comparisons})")
    md.append("")
    for name, c in comparisons.items():
        md.append(f"- `{name}` → `{c['meta']['ckpt_path']}` (n_features={c['meta']['n_features']}, arch={c['meta']['arch']})")
    md.append("")
    md.append("## Stability distribution")
    md.append("")
    md.append("| Stability score | # features | fraction |")
    md.append("|-----------------|-----------:|---------:|")
    for s in range(n_comparisons + 1):
        n = int((stability_score == s).sum())
        frac = n / n_features
        md.append(f"| {s} / {n_comparisons} | {n:,} | {frac:.4f} |")
    md.append("")
    md.append(
        f"**{n_fully_stable} of {n_features} features** ({100*n_fully_stable/n_features:.2f}%) "
        f"are stable across ALL {n_comparisons} comparison conditions. These are the strongest "
        f"candidates for 'real model features' — their decoder direction replicates regardless "
        f"of SAE width, seed, or architecture."
    )
    md.append("")

    # Fully stable features — listed
    md.append(f"## Fully stable features ({n_fully_stable})")
    md.append("")
    if n_fully_stable == 0:
        md.append("_No features stable across all conditions at this threshold._")
    else:
        md.append("Features whose decoder direction has cos > {} across every comparison:".format(args.threshold))
        md.append("")
        md.append("| Feature ID | Label (if known) | Min cos across comparisons |")
        md.append("|-----------:|------------------|---------------------------:|")
        for fid in fully_stable_ids[:50]:
            label = catalog.get(fid, {}).get("label", "_unlabeled_")
            min_cos = min(best_sim[name][fid] for name in comparisons)
            md.append(f"| {fid} | {label} | {min_cos:.4f} |")
        if n_fully_stable > 50:
            md.append(f"| ... | _and {n_fully_stable - 50} more_ | ... |")
        md.append("")

    # Labeled features and their stability
    md.append("## Labeled features — stability ranking")
    md.append("")
    if not catalog:
        md.append("_(no labeled catalog found)_")
    else:
        md.append(f"For each of the {len(catalog)} auto-labeled features (T5), how many "
                  f"comparison conditions does it survive?")
        md.append("")
        md.append("| Feature ID | Label | Stability | " + " | ".join(comparisons.keys()) + " |")
        md.append("|-----------:|-------|----------:|" + "|".join(["----------:" for _ in comparisons]) + "|")
        labeled_sorted = sorted(catalog.keys(), key=lambda fid: -int(stability_score[fid]))
        for fid in labeled_sorted:
            label = catalog[fid].get("label", "?")[:50]
            score = int(stability_score[fid])
            cos_str = " | ".join(
                f"{best_sim[name][fid]:.3f}{'★' if best_sim[name][fid] > args.threshold else ''}"
                for name in comparisons
            )
            md.append(f"| {fid} | {label} | {score}/{n_comparisons} | {cos_str} |")
        md.append("")

    md.append("## How to read this")
    md.append("")
    md.append("- **High stability score (4/4)** = feature replicates under every comparison condition we tested. Strongest evidence the feature is a real direction in Pythia-160M's residual stream, not an artifact of SAE training.")
    md.append("- **Low stability score (0-1/4)** = feature is specific to this particular SAE training run. Probably not safe to build circuits or interpretations on.")
    md.append("- **Intermediate (2-3/4)** = stable under some conditions but not all. Worth deeper analysis to understand which conditions break the match.")
    md.append("")

    Path(args.out_md).parent.mkdir(exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    print(f"saved {args.out_md}")


if __name__ == "__main__":
    main()
