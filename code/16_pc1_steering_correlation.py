"""Test the prediction: features with higher projection onto cluster PC1
should be the ones that steer most cleanly toward the cluster's concept.

From 18_manifold_rank.py: the newline cluster has one dominant PC (the
'newline-detector' direction). Each newline feature has its own projection
onto this PC. If our 'one shared atom plus N specializations' framing is right,
features with HIGH PC1 projection should drive newline behavior strongly when
steered; features with LOW PC1 projection should be specializations that drive
different behavior.

This script computes the PC1 projection for each newline feature, then steers
on a representative sample to confirm the prediction.

Usage:
    python 16_pc1_steering_correlation.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


NEUTRAL_PROMPTS = [
    "The recipe for chocolate cake is",
    "My favorite color is",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--label-contains", type=str, default="newline")
    p.add_argument("--alpha-mult", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def make_hook(direction, alpha):
    delta = (alpha * direction).to(torch.float32)
    def hook(activation, hook):
        return activation + delta.to(activation.dtype)
    return hook


@torch.no_grad()
def generate_with_hook(model, prompt, hook_fn, hook_name, max_new_tokens, temperature):
    tokens = model.to_tokens(prompt)
    if hook_fn is None:
        out = model.generate(tokens, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True, verbose=False)
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(tokens, max_new_tokens=max_new_tokens,
                                 temperature=temperature, do_sample=True, verbose=False)
    return model.to_string(out[0])


def count_newlines(text: str) -> int:
    return text.count("\n")


def count_whitespace_runs(text: str) -> int:
    """Count consecutive whitespace runs of length 2+ (heuristic for 'newline-y' output)."""
    import re
    return len(re.findall(r"\s{2,}", text))


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    sae = TopKSAE(d_model, ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    # --- Find cluster ---
    catalog = json.loads(Path(args.catalog).read_text())
    cluster = [(int(fid), info) for fid, info in catalog.items()
               if args.label_contains.lower() in info.get("label", "").lower()]
    cluster_ids = [fid for fid, _ in cluster]
    print(f"\nfound {len(cluster_ids)} '{args.label_contains}' features")

    # --- Compute PC1 of the cluster via SVD ---
    cluster_cols = sae.W_dec[:, cluster_ids].detach().T.cpu().numpy().astype(np.float64)  # (n, d_model)
    # SVD: M = U S V^T, V rows are principal axes in input space
    U, S, Vt = np.linalg.svd(cluster_cols, full_matrices=False)
    PC1 = Vt[0]  # (d_model,) — the principal direction in residual space
    print(f"  PC1 captures {(S[0]**2)/((S**2).sum())*100:.1f}% of cluster variance")

    # --- Compute each feature's PC1 projection ---
    pc1_t = torch.from_numpy(PC1.astype(np.float32)).to(device)
    projections = []
    for fid in cluster_ids:
        v = sae.W_dec[:, fid].to(torch.float32)
        proj = float((v @ pc1_t).item())
        projections.append((fid, abs(proj), proj, catalog[str(fid)]["label"]))
    projections.sort(key=lambda x: -x[1])

    print(f"\n--- PC1 projection magnitude per feature ---")
    print(f"  {'fid':>6} {'|proj|':>8} {'sign':>6}  label")
    for fid, abs_p, p, lbl in projections:
        print(f"  f{fid:>5} {abs_p:>8.4f} {'+' if p>0 else '-':>6}  {lbl[:50]}")

    # --- Load model and steer top-3 and bottom-3 by |PC1| ---
    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"\nloading {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    test_features = projections[:3] + projections[-3:]
    print(f"\nsteering on {len(test_features)} features (top-3 + bottom-3 by |PC1|)...")

    results = []
    t0 = time.time()
    for fid, abs_p, p, lbl in test_features:
        peak = catalog[str(fid)]["peak_activation"]
        alpha = args.alpha_mult * peak
        decoder_col = sae.W_dec[:, fid].detach().clone()

        print(f"\n{'=' * 70}")
        print(f"f{fid}  |PC1|={abs_p:.4f}  sign={'+' if p>0 else '-'}  alpha={alpha:.2f}")
        print(f"  label: {lbl}")
        print(f"{'=' * 70}")

        feature_summary = {
            "feature_id": fid,
            "label": lbl,
            "pc1_projection_signed": p,
            "pc1_projection_abs": abs_p,
            "alpha": alpha,
            "newline_counts": [],
            "whitespace_run_counts": [],
            "outputs": [],
        }

        for prompt in NEUTRAL_PROMPTS:
            torch.manual_seed(args.seed)
            baseline = generate_with_hook(model, prompt, None, hook_name,
                                          args.max_new_tokens, args.temperature)
            torch.manual_seed(args.seed)
            steered = generate_with_hook(model, prompt, make_hook(decoder_col, alpha),
                                         hook_name, args.max_new_tokens, args.temperature)

            base_nl = count_newlines(baseline)
            steer_nl = count_newlines(steered)
            base_ws = count_whitespace_runs(baseline)
            steer_ws = count_whitespace_runs(steered)

            print(f"\n  prompt:    {prompt!r}")
            print(f"  baseline:  newlines={base_nl}  ws_runs={base_ws}")
            print(f"             {baseline!r}")
            print(f"  steered:   newlines={steer_nl}  ws_runs={steer_ws}  Δnl={steer_nl-base_nl}")
            print(f"             {steered!r}")

            feature_summary["newline_counts"].append({
                "prompt": prompt,
                "baseline_nl": base_nl,
                "steered_nl": steer_nl,
                "delta_nl": steer_nl - base_nl,
            })
            feature_summary["whitespace_run_counts"].append({
                "prompt": prompt,
                "baseline_ws": base_ws,
                "steered_ws": steer_ws,
                "delta_ws": steer_ws - base_ws,
            })
            feature_summary["outputs"].append({
                "prompt": prompt,
                "baseline": baseline,
                "steered": steered,
            })

        feature_summary["mean_delta_nl"] = float(np.mean(
            [d["delta_nl"] for d in feature_summary["newline_counts"]]
        ))
        results.append(feature_summary)

    # --- Correlation analysis ---
    print(f"\n{'=' * 70}")
    print("CORRELATION: |PC1 projection| vs newline-induction by steering")
    print(f"{'=' * 70}")
    print(f"  {'fid':>6} {'|PC1|':>8} {'mean Δnl':>10}  label")
    for r in results:
        print(f"  f{r['feature_id']:>5} {r['pc1_projection_abs']:>8.4f} "
              f"{r['mean_delta_nl']:>10.2f}  {r['label'][:40]}")

    pc1_vals = [r["pc1_projection_abs"] for r in results]
    nl_deltas = [r["mean_delta_nl"] for r in results]
    if len(pc1_vals) >= 3:
        corr = float(np.corrcoef(pc1_vals, nl_deltas)[0, 1])
        print(f"\n  Pearson correlation: {corr:.3f}")
        print(f"  (positive correlation predicted by 'one shared atom' framing)")

    # --- Save ---
    out_path = Path(args.ckpt).with_suffix(".pc1_steering.json")
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "label_contains": args.label_contains,
        "alpha_mult": args.alpha_mult,
        "pc1_variance_explained": float((S[0]**2)/((S**2).sum())),
        "per_feature": results,
        "correlation_pc1_vs_delta_newlines": corr if len(pc1_vals) >= 3 else None,
    }, indent=2))
    print(f"\nsaved {out_path}")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
