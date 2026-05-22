"""Stable-feature co-activation analysis (Build #3).

Hypothesis: if stable features (those that survived cross-validation in
Build #2) are real building blocks of the model's representations, they
should show coherent co-activation patterns — pairs that fire together
more often than chance. Unstable features should not (since they're
substantially training-noise artifacts).

Procedure:
  1. Load reference SAE + activation cache.
  2. Load the 351 fully-stable feature IDs from feature_navigator.json.
  3. Sample N matched unstable features (stability score = 0) for comparison.
  4. Run SAE forward on a chunk of tokens; for each token, record which
     stable + matched-unstable features fire.
  5. Compute pairwise joint-activation counts within each group.
  6. Compute PMI = log( P(i,j) / (P(i) * P(j)) ) for all pairs.
  7. Compare PMI distributions: stable-stable vs unstable-unstable
     vs cross.

Output:
  data/coactivation_analysis.json
  notes/stable_feature_circuits.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--navigator", type=str, default="data/feature_navigator.json")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--n-tokens", type=int, default=200_000)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=7,
                   help="seed for sampling matched-unstable features")
    p.add_argument("--out-json", type=str, default="data/coactivation_analysis.json")
    p.add_argument("--out-md", type=str, default="notes/stable_feature_circuits.md")
    p.add_argument("--top-pairs", type=int, default=20,
                   help="number of top co-activating pairs to print")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # ----------------------------------------------------------------------
    # Load SAE + activations + stability info
    # ----------------------------------------------------------------------
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    sae = TopKSAE(d_model, n_features, k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    print(f"  d_model={d_model}  n_features={n_features}  k={ckpt['k']}")

    print(f"loading activations: {args.acts}")
    acts = np.load(args.acts)
    n_total = acts.shape[0]
    n_use = min(args.n_tokens, n_total)
    print(f"  using {n_use:,} of {n_total:,} tokens")

    print(f"loading navigator: {args.navigator}")
    nav = json.loads(Path(args.navigator).read_text())
    stable_ids = sorted([int(fid) for fid in nav.get("fully_stable_feature_ids", [])])
    print(f"  fully stable features: {len(stable_ids)}")

    # Build the unstable comparison set: features with stability score 0
    # We sample the same number as the stable set, matched in size.
    per_feat = nav.get("per_feature", {})
    unstable_candidates = [
        fid for fid in range(n_features)
        if fid not in set(stable_ids)
        and per_feat.get(str(fid), {}).get("stability_score", 0) == 0
    ]
    rng = np.random.default_rng(args.seed)
    if len(unstable_candidates) > 0:
        # Some unstable features may not have per_feature entries (we only
        # serialized entries for labeled+stable). Fall back to "not stable".
        non_stable = list(set(range(n_features)) - set(stable_ids))
        unstable_sample = rng.choice(non_stable, size=min(len(stable_ids), len(non_stable)),
                                     replace=False).tolist()
    else:
        non_stable = list(set(range(n_features)) - set(stable_ids))
        unstable_sample = rng.choice(non_stable, size=min(len(stable_ids), len(non_stable)),
                                     replace=False).tolist()
    unstable_sample = sorted(int(x) for x in unstable_sample)
    print(f"  matched-unstable sample: {len(unstable_sample)} features")

    # Combined index — we'll index into the SAE feature axis at these positions
    all_features = sorted(set(stable_ids) | set(unstable_sample))
    n_track = len(all_features)
    feat_to_local = {fid: i for i, fid in enumerate(all_features)}
    stable_local = np.array([feat_to_local[fid] for fid in stable_ids], dtype=np.int64)
    unstable_local = np.array([feat_to_local[fid] for fid in unstable_sample], dtype=np.int64)
    print(f"  tracking {n_track} features ({len(stable_ids)} stable + {len(unstable_sample)} unstable)")

    # ----------------------------------------------------------------------
    # Pass through SAE in chunks; record activations of tracked features only
    # ----------------------------------------------------------------------
    print(f"\ncomputing activations across {n_use:,} tokens (chunk={args.chunk})...")
    # We'll record a binary matrix: is_active[t, k] in {0,1}
    is_active = np.zeros((n_use, n_track), dtype=np.uint8)
    write_idx = 0
    t0 = time.time()

    feat_idx_tensor = torch.tensor(all_features, dtype=torch.long, device=device)

    rng_acts = np.random.default_rng(args.seed)
    sample_idx = rng_acts.permutation(n_total)[:n_use]
    sample_idx.sort()

    with torch.no_grad():
        for i in range(0, n_use, args.chunk):
            batch_idx = sample_idx[i : i + args.chunk]
            batch_np = acts[batch_idx].astype(np.float32)
            x = torch.from_numpy(batch_np).to(device)
            f = sae.encode(x)  # (B, n_features), mostly zero
            # Select tracked features
            tracked = f.index_select(dim=1, index=feat_idx_tensor)  # (B, n_track)
            is_active[write_idx : write_idx + tracked.shape[0]] = (
                (tracked > 0).cpu().numpy().astype(np.uint8)
            )
            write_idx += tracked.shape[0]
            if (i // args.chunk) % 10 == 0:
                rate = write_idx / max(time.time() - t0, 1e-9)
                print(f"  {write_idx:,}/{n_use:,}  ({rate:,.0f} tok/s)", flush=True)
    print(f"  done in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------------
    # Compute joint and marginal activation counts
    # ----------------------------------------------------------------------
    print("\ncomputing co-activation statistics...")
    # marginal: (n_track,)
    marginal = is_active.sum(axis=0).astype(np.float64)
    p_marginal = marginal / n_use
    # joint: (n_track, n_track) — count of tokens where both i and j fire
    # Cast to int32 to avoid overflow for 200k tokens
    joint = (is_active.astype(np.int32).T @ is_active.astype(np.int32)).astype(np.float64)
    p_joint = joint / n_use

    # PMI matrix (log domain)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = p_marginal[:, None] * p_marginal[None, :]
        pmi = np.where(
            (p_joint > 0) & (expected > 0),
            np.log(p_joint / expected),
            -np.inf,
        )
    # Set diagonal to NaN (self-PMI is not meaningful)
    np.fill_diagonal(pmi, np.nan)

    # ----------------------------------------------------------------------
    # Slice into three submatrices
    # ----------------------------------------------------------------------
    pmi_ss = pmi[np.ix_(stable_local, stable_local)]
    pmi_uu = pmi[np.ix_(unstable_local, unstable_local)]
    pmi_su = pmi[np.ix_(stable_local, unstable_local)]

    def summarize(name, m):
        finite = m[np.isfinite(m)]
        if finite.size == 0:
            return {"name": name, "n_pairs": 0}
        return {
            "name": name,
            "n_pairs": int(np.isfinite(m).sum()),
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
            "fraction_pmi_above_1": float((finite > 1.0).sum() / finite.size),
            "fraction_pmi_above_2": float((finite > 2.0).sum() / finite.size),
            "fraction_pmi_above_5": float((finite > 5.0).sum() / finite.size),
        }

    summaries = {
        "stable_stable": summarize("stable × stable", pmi_ss),
        "unstable_unstable": summarize("unstable × unstable", pmi_uu),
        "stable_unstable": summarize("stable × unstable (cross)", pmi_su),
    }

    print("\n=== CO-ACTIVATION SUMMARIES (PMI distributions) ===")
    print(f"{'pairing':<30} {'n_pairs':<10} {'mean':<8} {'p90':<8} {'p99':<8} {'frac>1':<8} {'frac>2':<8} {'frac>5':<8}")
    for k, s in summaries.items():
        if s.get("n_pairs", 0) == 0:
            continue
        print(f"  {s['name']:<28} {s['n_pairs']:<10,} {s['mean']:<8.3f} {s['p90']:<8.3f} {s['p99']:<8.3f} "
              f"{s['fraction_pmi_above_1']:<8.4f} {s['fraction_pmi_above_2']:<8.4f} {s['fraction_pmi_above_5']:<8.4f}")

    # ----------------------------------------------------------------------
    # Top co-activating stable pairs
    # ----------------------------------------------------------------------
    print(f"\n=== TOP {args.top_pairs} STABLE-STABLE PAIRS BY PMI ===")
    # Avoid duplicates by taking upper triangle only
    iu, ju = np.triu_indices(pmi_ss.shape[0], k=1)
    pmi_vals = pmi_ss[iu, ju]
    finite_mask = np.isfinite(pmi_vals)
    sort_order = np.argsort(-pmi_vals[finite_mask])
    iu_f = iu[finite_mask][sort_order]
    ju_f = ju[finite_mask][sort_order]
    pmi_f = pmi_vals[finite_mask][sort_order]

    catalog_path = Path(args.catalog)
    catalog = {}
    if catalog_path.exists():
        raw = json.loads(catalog_path.read_text())
        catalog = {int(k): v for k, v in raw.items() if isinstance(v, dict)}

    top_pairs = []
    print(f"  {'fid_a':<8} {'fid_b':<8} {'PMI':<8} {'P(a)':<8} {'P(b)':<8} {'P(a,b)':<8} labels")
    for n in range(min(args.top_pairs, len(iu_f))):
        a_local, b_local = iu_f[n], ju_f[n]
        a, b = stable_ids[a_local], stable_ids[b_local]
        pmi_val = pmi_f[n]
        p_a = p_marginal[stable_local[a_local]]
        p_b = p_marginal[stable_local[b_local]]
        p_ab = p_joint[stable_local[a_local], stable_local[b_local]]
        label_a = (catalog.get(a, {}).get("label", "?") or "?")[:35]
        label_b = (catalog.get(b, {}).get("label", "?") or "?")[:35]
        print(f"  f{a:<7} f{b:<7} {pmi_val:<8.3f} {p_a:<8.5f} {p_b:<8.5f} {p_ab:<8.5f}  {label_a} | {label_b}")
        top_pairs.append({
            "fid_a": int(a), "fid_b": int(b), "pmi": float(pmi_val),
            "p_a": float(p_a), "p_b": float(p_b), "p_ab": float(p_ab),
            "label_a": label_a, "label_b": label_b,
        })

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    results = {
        "n_tokens_analyzed": int(n_use),
        "n_stable_features": len(stable_ids),
        "n_unstable_features_sampled": len(unstable_sample),
        "pmi_summaries": summaries,
        "top_stable_pairs": top_pairs,
        "fully_stable_feature_ids": stable_ids,
        "unstable_sample_feature_ids": unstable_sample,
    }
    Path(args.out_json).parent.mkdir(exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out_json}")

    # ----------------------------------------------------------------------
    # Markdown summary
    # ----------------------------------------------------------------------
    md = []
    md.append("# Stable-Feature Co-Activation Analysis")
    md.append("")
    md.append(f"Computed on {n_use:,} held-out tokens from the SAE training stream.")
    md.append(f"Tracked: {len(stable_ids)} fully-stable features + "
              f"{len(unstable_sample)} matched-unstable features.")
    md.append("")
    md.append("## PMI distribution by pairing")
    md.append("")
    md.append("PMI(i,j) = log( P(i,j) / (P(i) · P(j)) ). Higher = features fire together more than chance.")
    md.append("")
    md.append("| Pairing | n pairs | mean | p90 | p99 | frac > 1 | frac > 2 | frac > 5 |")
    md.append("|---|------:|-----:|----:|----:|---------:|---------:|---------:|")
    for s in summaries.values():
        if s.get("n_pairs", 0) == 0:
            continue
        md.append(
            f"| {s['name']} | {s['n_pairs']:,} | {s['mean']:.3f} | {s['p90']:.3f} | "
            f"{s['p99']:.3f} | {s['fraction_pmi_above_1']:.4f} | "
            f"{s['fraction_pmi_above_2']:.4f} | {s['fraction_pmi_above_5']:.4f} |"
        )
    md.append("")
    ss_mean = summaries["stable_stable"]["mean"]
    uu_mean = summaries["unstable_unstable"]["mean"]
    gap = ss_mean - uu_mean
    md.append(
        f"**Mean PMI gap (stable-stable vs unstable-unstable): {gap:+.3f}.**"
    )
    md.append("")
    if gap > 0.3:
        md.append("Stable-stable PMI is meaningfully higher than unstable-unstable. "
                  "Stable features tend to fire together more than chance — "
                  "consistent with proto-circuit structure.")
    elif gap < -0.3:
        md.append("Unstable-unstable PMI is higher than stable-stable. "
                  "Stable features are MORE INDEPENDENT than unstable ones — "
                  "they are atomic, not compositional.")
    else:
        md.append("PMI distributions are similar. Stability does not strongly predict "
                  "co-activation behavior on this data.")
    md.append("")
    md.append("## Top stable-stable pairs by PMI")
    md.append("")
    md.append("| f_a | label_a | f_b | label_b | PMI | P(a,b) |")
    md.append("|----:|---------|----:|---------|----:|-------:|")
    for tp in top_pairs:
        md.append(
            f"| f{tp['fid_a']} | {tp['label_a']} | f{tp['fid_b']} | "
            f"{tp['label_b']} | {tp['pmi']:.3f} | {tp['p_ab']:.5f} |"
        )
    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append("- High-PMI pairs are candidate building blocks of circuits — features that "
              "systematically fire together on the same tokens.")
    md.append("- If stable features dominate the high-PMI tail relative to unstable features, "
              "stability + co-activation is a useful filter for downstream circuit discovery.")
    md.append("- PMI alone does not establish causality. To go from co-activation to circuit "
              "requires ablation experiments (forcing one feature off, observing the other).")
    md.append("")

    Path(args.out_md).parent.mkdir(exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    print(f"saved {args.out_md}")


if __name__ == "__main__":
    main()
