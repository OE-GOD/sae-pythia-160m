"""Causal validation: ablate feature direction; measure CE delta on active tokens.

For each candidate feature f, project out v_f from layer-6 residual stream
during forward pass. Measure DeltaCE per token, split by whether feature
was active. Compare three groups: 3 stable features (Build #2: f12117 citations,
f5747 paths, f12697 logical), 3 unstable newline-variant features, 3 random.

Hypothesis: stable features show DeltaCE_active >> DeltaCE_inactive (causally
load-bearing). Unstable/random features show no consistent difference.

Outputs: data/causal_validation.json, notes/causal_validation.md
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
from transformer_lens import HookedTransformer  # noqa: E402
from datasets import load_dataset  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--n-tokens", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--skip-docs", type=int, default=8000)
    p.add_argument("--stable", type=str, default="12117,5747,12697")
    p.add_argument("--unstable", type=str, default="2255,6767,7448")
    p.add_argument("--random-features", type=str, default=None)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-json", type=str, default="data/causal_validation.json")
    p.add_argument("--out-md", type=str, default="notes/causal_validation.md")
    return p.parse_args()


def per_token_nll(logits, targets):
    """logits (B, T, V), targets (B, T). Return NLL at positions [:-1] predicting [1:]."""
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_targets.reshape(-1),
        reduction="none",
    )
    return nll.reshape(shift_targets.shape)


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    sae = TopKSAE(d_model, n_features, k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"  d_model={d_model}, n_features={n_features}, layer={layer}")

    stable = [int(x) for x in args.stable.split(",")]
    unstable = [int(x) for x in args.unstable.split(",")]
    excluded = set(stable + unstable)
    if args.random_features:
        random_feats = [int(x) for x in args.random_features.split(",")]
    else:
        rng = np.random.default_rng(args.seed)
        pool = list(set(range(n_features)) - excluded)
        random_feats = sorted(int(x) for x in rng.choice(pool, size=3, replace=False))
    feats = stable + unstable + random_feats
    groups = (["stable"] * len(stable) + ["unstable"] * len(unstable)
              + ["random"] * len(random_feats))
    print(f"stable={stable}  unstable={unstable}  random={random_feats}")

    print("loading Pythia-160m...")
    model = HookedTransformer.from_pretrained("pythia-160m", device=device)
    model.eval()

    print(f"loading Pile-10k, skip {args.skip_docs} docs...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    held = list(ds)[args.skip_docs:]
    all_tokens: list[int] = []
    needed = args.n_tokens + args.seq_len * args.batch_size
    for doc in held:
        toks = model.to_tokens(doc["text"], prepend_bos=False)[0].tolist()
        all_tokens.extend(toks)
        if len(all_tokens) >= needed:
            break
    n_seqs = args.n_tokens // args.seq_len
    total = n_seqs * args.seq_len
    token_tensor = torch.tensor(all_tokens[:total], dtype=torch.long).reshape(n_seqs, args.seq_len)
    print(f"  {total:,} tokens in {n_seqs} sequences")

    W_dec = sae.W_dec.data
    feature_dirs = {fid: (W_dec[:, fid] / W_dec[:, fid].norm().clamp(min=1e-8))
                    for fid in feats}

    # ----- Pass 1: clean baseline + record activations for feats -----
    print(f"\n[pass 1/{len(feats) + 1}] clean run + record activations...")
    clean_nll = np.zeros((n_seqs, args.seq_len - 1), dtype=np.float32)
    active = {fid: np.zeros((n_seqs, args.seq_len), dtype=bool) for fid in feats}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_seqs, args.batch_size):
            batch = token_tensor[i : i + args.batch_size].to(device)
            logits, cache = model.run_with_cache(batch, names_filter=[hook_name])
            h = cache[hook_name]  # (B, T, d_model)
            f = sae.encode(h.reshape(-1, d_model).to(torch.float32)).reshape(h.shape[0], h.shape[1], -1)
            for fid in feats:
                active[fid][i : i + h.shape[0]] = (f[:, :, fid] > 0).cpu().numpy()
            nll = per_token_nll(logits, batch).cpu().numpy()
            clean_nll[i : i + h.shape[0]] = nll
            if (i // args.batch_size) % 5 == 0:
                print(f"  {i + h.shape[0]}/{n_seqs} seqs  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  clean done in {time.time()-t0:.1f}s")
    baseline_mean_nll = float(clean_nll.mean())
    print(f"  baseline CE per token: {baseline_mean_nll:.4f} nats")

    # ----- Pass 2..k+1: per-feature ablation -----
    results_per_feature = []
    for f_idx, fid in enumerate(feats):
        v = feature_dirs[fid].to(device)  # (d_model,)
        def ablate_hook(activation, hook, v=v):
            # Project out v's direction: h' = h - (h . v) v
            proj = (activation * v).sum(dim=-1, keepdim=True)  # (B, T, 1)
            return activation - proj * v

        ablated_nll = np.zeros((n_seqs, args.seq_len - 1), dtype=np.float32)
        t1 = time.time()
        with torch.no_grad():
            for i in range(0, n_seqs, args.batch_size):
                batch = token_tensor[i : i + args.batch_size].to(device)
                logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, ablate_hook)])
                nll = per_token_nll(logits, batch).cpu().numpy()
                ablated_nll[i : i + batch.shape[0]] = nll

        delta = ablated_nll - clean_nll  # (n_seqs, T-1)
        # active mask is shape (n_seqs, T); align with positions [0, T-2] which predict [1, T-1]
        is_active = active[fid][:, :-1]  # active at position t predicts token at t+1
        mask_active = is_active
        mask_inactive = ~is_active

        n_active = int(mask_active.sum())
        n_inactive = int(mask_inactive.sum())
        delta_active = float(delta[mask_active].mean()) if n_active > 0 else float("nan")
        delta_inactive = float(delta[mask_inactive].mean()) if n_inactive > 0 else float("nan")
        delta_global = float(delta.mean())
        ratio = delta_active / delta_inactive if (n_inactive > 0 and abs(delta_inactive) > 1e-9) else float("nan")
        active_frac = n_active / (n_active + n_inactive) if (n_active + n_inactive) > 0 else 0.0

        results_per_feature.append({
            "feature_id": fid,
            "group": groups[f_idx],
            "n_active_tokens": n_active,
            "active_fraction": active_frac,
            "delta_ce_global": delta_global,
            "delta_ce_active": delta_active,
            "delta_ce_inactive": delta_inactive,
            "ratio_active_inactive": ratio,
        })
        print(f"  [{f_idx+1}/{len(feats)}] f{fid} ({groups[f_idx]:<8})  active={active_frac:.4f}  "
              f"deltaCE_active={delta_active:+.4f}  deltaCE_inactive={delta_inactive:+.4f}  "
              f"ratio={ratio:.2f}  ({time.time()-t1:.1f}s)", flush=True)

    # ----- Aggregate by group -----
    print(f"\n=== AGGREGATE BY GROUP ===")
    group_summary = {}
    for g in ("stable", "unstable", "random"):
        rows = [r for r in results_per_feature if r["group"] == g]
        if not rows:
            continue
        active_means = [r["delta_ce_active"] for r in rows if not np.isnan(r["delta_ce_active"])]
        inactive_means = [r["delta_ce_inactive"] for r in rows if not np.isnan(r["delta_ce_inactive"])]
        ratios = [r["ratio_active_inactive"] for r in rows if not np.isnan(r["ratio_active_inactive"])]
        group_summary[g] = {
            "n_features": len(rows),
            "mean_delta_ce_active": float(np.mean(active_means)) if active_means else None,
            "mean_delta_ce_inactive": float(np.mean(inactive_means)) if inactive_means else None,
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
        }
        print(f"  {g:<10} mean deltaCE_active={group_summary[g]['mean_delta_ce_active']:+.4f}  "
              f"mean deltaCE_inactive={group_summary[g]['mean_delta_ce_inactive']:+.4f}  "
              f"mean ratio={group_summary[g]['mean_ratio']:.2f}")

    # ----- Save -----
    results = {
        "baseline_ce": baseline_mean_nll,
        "n_eval_tokens": int(n_seqs * (args.seq_len - 1)),
        "feats_stable": stable, "feats_unstable": unstable, "feats_random": random_feats,
        "per_feature": results_per_feature,
        "group_summary": group_summary,
    }
    Path(args.out_json).parent.mkdir(exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out_json}")

    # ----- Markdown -----
    md = ["# Causal Validation: Stable vs Unstable Features", "",
          f"Baseline CE: **{baseline_mean_nll:.4f}** nats/token on "
          f"{n_seqs * (args.seq_len - 1):,} held-out predicted tokens.", "",
          "## Per-feature results", "",
          "| Feature | Group | Active % | ΔCE active | ΔCE inactive | Ratio |",
          "|--------:|-------|---------:|-----------:|-------------:|------:|"]
    for r in results_per_feature:
        md.append(f"| f{r['feature_id']} | {r['group']} | {r['active_fraction']*100:.2f}% "
                  f"| {r['delta_ce_active']:+.4f} | {r['delta_ce_inactive']:+.4f} "
                  f"| {r['ratio_active_inactive']:.2f} |")
    md.append("")
    md.append("## Group means")
    md.append("")
    md.append("| Group | mean ΔCE active | mean ΔCE inactive | mean ratio |")
    md.append("|---|---:|---:|---:|")
    for g, s in group_summary.items():
        md.append(f"| {g} | {s['mean_delta_ce_active']:+.4f} | {s['mean_delta_ce_inactive']:+.4f} | {s['mean_ratio']:.2f} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- Higher **ΔCE active** (cost of ablation on tokens where the feature was firing) indicates the feature carries information the model uses.")
    md.append("- **Ratio ΔCE_active / ΔCE_inactive** isolates the feature-specific effect: a value >> 1 means ablation hurts much more on the feature's own tokens than elsewhere.")
    md.append("- Stable features expected to show higher ratio than unstable / random features.")
    Path(args.out_md).parent.mkdir(exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    print(f"saved {args.out_md}")


if __name__ == "__main__":
    main()