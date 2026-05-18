"""Evaluate a trained SAE: MSE, L0 sparsity, dead features, variance explained.

Auto-detects whether checkpoint is L1 (vanilla) or TopK by reading ckpt['arch'].

Usage:
    python 03_eval_sae.py
    python 03_eval_sae.py --ckpt checkpoints/sae_layer6_topk64.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Import the two SAE classes by exec-importing both training scripts (without running main).
_l1_src = (Path(__file__).parent / "02_train_sae.py").read_text().split("def parse_args")[0]
exec(_l1_src)  # defines SparseAutoencoder

_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6.pt")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--n-eval", type=int, default=50_000)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    arch = ckpt.get("arch", "l1")

    if arch == "topk":
        k = ckpt["k"]
        sae = TopKSAE(d_model, n_features, k=k).to(device)  # noqa: F821
        print(f"  arch=TopK  d_model={d_model}  n_features={n_features}  k={k}  "
              f"trained_steps={ckpt['steps']}")
    else:
        sae = SparseAutoencoder(d_model, n_features).to(device)  # noqa: F821
        print(f"  arch=L1  d_model={d_model}  n_features={n_features}  "
              f"trained_steps={ckpt['steps']}  lambda={ckpt.get('lambda_sparsity')}")

    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    print(f"\nloading activations: {args.acts}")
    acts = np.load(args.acts, mmap_mode="r")
    n_total = acts.shape[0]
    n_eval = min(args.n_eval, n_total)
    idx = np.random.choice(n_total, size=n_eval, replace=False)
    print(f"  evaluating on {n_eval:,} held-out activations from {n_total:,} total")

    chunk = 4096
    total_recon = 0.0
    total_l1 = 0.0
    total_var_x = 0.0
    total_var_resid = 0.0
    feat_active_count = torch.zeros(n_features, dtype=torch.long, device=device)
    n_seen = 0
    l0_per_token = []

    with torch.no_grad():
        for i in range(0, n_eval, chunk):
            batch_np = acts[idx[i : i + chunk]].astype(np.float32)
            x = torch.from_numpy(batch_np).to(device)
            x_hat, f = sae(x)

            recon_per = (x - x_hat).pow(2).sum(dim=-1)
            total_recon += recon_per.sum().item()
            total_l1 += f.abs().sum().item()

            xc = x - x.mean(dim=0, keepdim=True)
            total_var_x += xc.pow(2).sum().item()
            total_var_resid += (x - x_hat).pow(2).sum().item()

            feat_active_count += (f > 0).long().sum(dim=0)
            l0_per_token.append((f > 0).float().sum(dim=-1).cpu().numpy())
            n_seen += x.shape[0]

    mse = total_recon / n_seen
    avg_l1 = total_l1 / n_seen
    var_explained = 1.0 - (total_var_resid / total_var_x) if total_var_x > 0 else float("nan")
    l0_arr = np.concatenate(l0_per_token)
    n_dead = int((feat_active_count == 0).sum().item())
    n_alive = n_features - n_dead
    pct_dead = 100.0 * n_dead / n_features

    print(f"\n=== SAE EVALUATION ===")
    print(f"  reconstruction MSE (sum over d_model):  {mse:.3f}")
    print(f"  variance explained:                     {var_explained:.4f}")
    print(f"  avg L1 (sum |f|):                       {avg_l1:.2f}")
    print(f"  L0 (avg # active features per token):   {l0_arr.mean():.1f} ± {l0_arr.std():.1f}")
    print(f"  L0 percentiles: p10={np.percentile(l0_arr, 10):.0f}  "
          f"p50={np.percentile(l0_arr, 50):.0f}  p90={np.percentile(l0_arr, 90):.0f}")
    print(f"  dead features:    {n_dead:,} / {n_features:,}  ({pct_dead:.1f}%)")
    print(f"  alive features:   {n_alive:,}")

    alive_counts = feat_active_count[feat_active_count > 0].cpu().numpy()
    if len(alive_counts) > 0:
        print(f"\n  alive feature firing rates (fraction of tokens active):")
        for q in [0.1, 0.5, 0.9, 0.99]:
            v = np.quantile(alive_counts, q) / n_seen
            print(f"    p{int(q*100):02d}: {v:.4f}  (= 1 in ~{int(1/max(v,1e-9))} tokens)")

    results = {
        "arch": arch,
        "mse": float(mse),
        "var_explained": float(var_explained),
        "avg_l1": float(avg_l1),
        "L0_mean": float(l0_arr.mean()),
        "L0_std": float(l0_arr.std()),
        "n_dead": n_dead,
        "n_alive": n_alive,
        "pct_dead": pct_dead,
        "n_eval": n_seen,
    }
    out_path = Path(args.ckpt).with_suffix(".eval.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved eval results to {out_path}")


if __name__ == "__main__":
    main()
