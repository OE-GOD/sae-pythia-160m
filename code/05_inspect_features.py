"""Generate a human-readable markdown report of SAE features for manual inspection.

Selects features by a few different criteria so the report is informative:
  - Top by peak activation (strongest features overall)
  - Top by mean activation across top-K (consistently strong features)
  - Random sample of alive features (sanity check)

Output: notes/feature_inspection.md
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", type=str, default="data/feature_examples.json")
    p.add_argument("--top-features", type=str, default="data/top_features.npz")
    p.add_argument("--n-by-peak", type=int, default=30, help="features ranked by single-snippet peak activation")
    p.add_argument("--n-by-mean", type=int, default=30, help="features ranked by mean-of-top-K activation")
    p.add_argument("--n-random", type=int, default=20, help="random sample of alive features")
    p.add_argument("--out", type=str, default="notes/feature_inspection.md")
    return p.parse_args()


def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def format_feature(fid: int, examples: list[dict], max_snippet_len: int = 120) -> str:
    out = [f"### Feature #{fid}", ""]
    out.append(f"Top {len(examples)} max-activating snippets:")
    out.append("")
    for i, ex in enumerate(examples):
        snippet = truncate(ex["snippet"], max_snippet_len)
        target = ex["target_token"].replace("\n", "\\n")
        out.append(f"{i+1}. `act={ex['activation']:.2f}` target=`{target!r}` snippet: `{snippet}`")
    out.append("")
    return "\n".join(out)


def main():
    args = parse_args()
    print(f"loading {args.examples}...")
    feature_examples = json.loads(Path(args.examples).read_text())
    print(f"  {len(feature_examples):,} alive features")

    print(f"loading {args.top_features}...")
    npz = np.load(args.top_features)
    top_values = npz["top_values"]  # (n_features, K)
    n_features, K = top_values.shape

    # Convert string keys to int
    fe_by_id = {int(k): v for k, v in feature_examples.items()}

    # Rank by peak activation
    peak = top_values[:, 0]  # highest activation per feature
    alive_mask = peak > 0
    alive_ids = np.where(alive_mask)[0]
    by_peak = sorted(alive_ids.tolist(), key=lambda i: -peak[i])

    # Rank by mean of top-K (more "consistent" features)
    mean_topk = top_values.mean(axis=1)
    by_mean = sorted(alive_ids.tolist(), key=lambda i: -mean_topk[i])

    # Random sample
    rng = random.Random(42)
    random_ids = rng.sample(alive_ids.tolist(), k=min(args.n_random, len(alive_ids)))

    sections = []
    sections.append("# SAE Feature Inspection — Pythia-160M, layer 6, TopK k=64")
    sections.append("")
    sections.append(f"- Total alive features: **{len(alive_ids):,} / {n_features:,}**")
    sections.append(f"- Each row below shows a feature's top-K max-activating snippets.")
    sections.append(f"- `act` is the SAE feature activation value at that token.")
    sections.append(f"- `target` is the token being scored (decoded).")
    sections.append("")
    sections.append("Use this report to manually identify which features look monosemantic")
    sections.append("(snippets share a clear theme) vs polysemantic (mixed concepts).")
    sections.append("")

    sections.append(f"## Section A — Top {args.n_by_peak} features by single-snippet peak activation")
    sections.append("")
    sections.append("These features have the strongest single activations. Often these are 'sharp' features")
    sections.append("triggered by specific tokens or patterns.")
    sections.append("")
    for fid in by_peak[: args.n_by_peak]:
        examples = fe_by_id.get(int(fid), [])
        if not examples:
            continue
        sections.append(format_feature(int(fid), examples))

    sections.append("")
    sections.append(f"## Section B — Top {args.n_by_mean} features by mean of top-K activations")
    sections.append("")
    sections.append("These features have the most consistent strong activations across their top examples.")
    sections.append("Often more 'reliable' interpretable features than Section A's peak-driven ones.")
    sections.append("")
    for fid in by_mean[: args.n_by_mean]:
        examples = fe_by_id.get(int(fid), [])
        if not examples:
            continue
        sections.append(format_feature(int(fid), examples))

    sections.append("")
    sections.append(f"## Section C — Random sample of {args.n_random} alive features (sanity check)")
    sections.append("")
    sections.append("Random sample — typical features, not curated. Use this to estimate the prevalence")
    sections.append("of monosemantic vs polysemantic features in the SAE overall.")
    sections.append("")
    for fid in random_ids:
        examples = fe_by_id.get(int(fid), [])
        if not examples:
            continue
        sections.append(format_feature(int(fid), examples))

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(sections))
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Section A (peak):    top {args.n_by_peak}")
    print(f"  Section B (mean):    top {args.n_by_mean}")
    print(f"  Section C (random):  {args.n_random} alive features")
    print(f"\nOpen with: open {out_path}")


if __name__ == "__main__":
    main()
