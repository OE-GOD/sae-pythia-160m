"""Analyze the Matryoshka SAE vs the flat TopK SAE.

Produces:
  - Per-level MSE / var_explained / L0 / dead-features at each Matryoshka level
  - Cross-architecture cosine similarity: do Matryoshka decoder columns
    match the flat TopK decoder columns?
  - "Hierarchy" check: are features at k=16 a strict subset of features at k=64?
    (By construction yes; we verify and report the fraction of tokens where
    this holds in practice.)

Output: data/matryoshka_analysis.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
# Import both SAE classes
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # TopKSAE
_mat_src = (Path(__file__).parent / "02c_train_sae_matryoshka.py").read_text().split("def parse_args")[0]
exec(_mat_src)  # MatryoshkaSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-matryoshka", type=str,
                   default="checkpoints/sae_layer6_matryoshka.pt")
    p.add_argument("--ckpt-topk", type=str,
                   default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--n-eval", type=int, default=50_000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="data/matryoshka_analysis.json")
    return p.parse_args()


def unit_norm(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"loading Matryoshka SAE: {args.ckpt_matryoshka}")
    mat_ckpt = torch.load(args.ckpt_matryoshka, map_location=device, weights_only=False)
    k_levels = mat_ckpt["k_levels"]
    d_model = mat_ckpt["d_model"]
    n_features = mat_ckpt["n_features"]
    mat = MatryoshkaSAE(d_model, n_features, k_levels).to(device)  # noqa: F821
    mat.load_state_dict(mat_ckpt["model_state"])
    mat.eval()
    print(f"  d_model={d_model}  n_features={n_features}  k_levels={k_levels}")

    print(f"loading TopK SAE: {args.ckpt_topk}")
    topk_ckpt = torch.load(args.ckpt_topk, map_location=device, weights_only=False)
    topk = TopKSAE(topk_ckpt["d_model"], topk_ckpt["n_features"],  # noqa: F821
                   k=topk_ckpt["k"]).to(device)
    topk.load_state_dict(topk_ckpt["model_state"])
    topk.eval()
    print(f"  TopK k={topk_ckpt['k']}")

    print(f"\nloading eval activations from {args.acts}")
    acts = np.load(args.acts, mmap_mode="r")
    n_total = acts.shape[0]
    n_eval = min(args.n_eval, n_total)
    idx = np.random.choice(n_total, size=n_eval, replace=False)

    # =====================================================================
    # Eval each Matryoshka level + the flat TopK SAE
    # =====================================================================
    chunk = 4096
    per_level_recon = {k: 0.0 for k in k_levels}
    per_level_l0 = {k: [] for k in k_levels}
    per_level_active_count = {k: torch.zeros(n_features, dtype=torch.long, device=device)
                              for k in k_levels}
    topk_recon = 0.0
    topk_l0 = []
    topk_active = torch.zeros(topk_ckpt["n_features"], dtype=torch.long, device=device)

    total_var_x = 0.0
    var_resid_per_level = {k: 0.0 for k in k_levels}
    var_resid_topk = 0.0
    n_seen = 0

    with torch.no_grad():
        for i in range(0, n_eval, chunk):
            batch_np = acts[idx[i : i + chunk]].astype(np.float32)
            x = torch.from_numpy(batch_np).to(device)

            # Matryoshka: get x_hats per level
            x_hats, feature_levels = mat(x)
            for k, xh, f in zip(k_levels, x_hats, feature_levels):
                resid = (x - xh).pow(2).sum(dim=-1)
                per_level_recon[k] += resid.sum().item()
                var_resid_per_level[k] += resid.sum().item()
                per_level_l0[k].append((f > 0).float().sum(dim=-1).cpu().numpy())
                per_level_active_count[k] += (f > 0).long().sum(dim=0)

            # Flat TopK
            x_hat_topk, f_topk = topk(x)
            resid_topk = (x - x_hat_topk).pow(2).sum(dim=-1)
            topk_recon += resid_topk.sum().item()
            var_resid_topk += resid_topk.sum().item()
            topk_l0.append((f_topk > 0).float().sum(dim=-1).cpu().numpy())
            topk_active += (f_topk > 0).long().sum(dim=0)

            xc = x - x.mean(dim=0, keepdim=True)
            total_var_x += xc.pow(2).sum().item()
            n_seen += x.shape[0]

    print(f"\n=== PER-LEVEL EVAL (n_eval={n_seen}) ===")
    print(f"  {'level':<12} {'MSE':<10} {'var_exp':<10} {'L0_mean':<10} {'dead%':<8}")
    results = {"matryoshka_levels": {}, "topk_flat": {}}
    for k in k_levels:
        mse = per_level_recon[k] / n_seen
        ve = 1.0 - var_resid_per_level[k] / max(total_var_x, 1e-9)
        l0_arr = np.concatenate(per_level_l0[k])
        n_dead = int((per_level_active_count[k] == 0).sum().item())
        pct_dead = 100.0 * n_dead / n_features
        print(f"  mat k={k:<6} {mse:<10.3f} {ve:<10.4f} {l0_arr.mean():<10.1f} {pct_dead:<8.2f}")
        results["matryoshka_levels"][str(k)] = {
            "mse": mse, "var_explained": ve, "L0_mean": float(l0_arr.mean()),
            "n_dead": n_dead, "pct_dead": pct_dead,
        }

    mse_t = topk_recon / n_seen
    ve_t = 1.0 - var_resid_topk / max(total_var_x, 1e-9)
    l0_t = np.concatenate(topk_l0)
    n_dead_t = int((topk_active == 0).sum().item())
    pct_dead_t = 100.0 * n_dead_t / topk_ckpt["n_features"]
    print(f"  topk k=64    {mse_t:<10.3f} {ve_t:<10.4f} {l0_t.mean():<10.1f} {pct_dead_t:<8.2f}")
    results["topk_flat"] = {
        "k": topk_ckpt["k"], "mse": mse_t, "var_explained": ve_t,
        "L0_mean": float(l0_t.mean()), "n_dead": n_dead_t, "pct_dead": pct_dead_t,
    }

    # =====================================================================
    # Nested-subset property check
    # =====================================================================
    print(f"\n=== NESTED-SUBSET PROPERTY (Matryoshka invariant) ===")
    print(f"  By construction, features active at k_i should be a strict subset of features at k_{{i+1}}.")
    print(f"  Sanity-checking on first {min(n_eval, 4096)} tokens:")
    chunk = min(n_eval, 4096)
    batch_np = acts[idx[:chunk]].astype(np.float32)
    x = torch.from_numpy(batch_np).to(device)
    with torch.no_grad():
        _, fls = mat(x)
    nesting_holds = True
    for i in range(len(k_levels) - 1):
        smaller, larger = fls[i] > 0, fls[i+1] > 0
        subset_per_token = (smaller & larger).sum(dim=-1) == smaller.sum(dim=-1)
        frac = subset_per_token.float().mean().item()
        ok = "✓" if frac > 0.999 else "✗"
        print(f"  k={k_levels[i]} ⊂ k={k_levels[i+1]}:  {frac*100:.2f}% of tokens  {ok}")
        if frac < 0.999:
            nesting_holds = False
    results["nesting_invariant_holds"] = nesting_holds

    # =====================================================================
    # Cross-architecture decoder cosine similarity
    # =====================================================================
    print(f"\n=== CROSS-ARCHITECTURE DECODER SIMILARITY ===")
    print(f"  For each Matryoshka feature, find its best TopK-SAE match (cos sim):")
    W_mat = unit_norm(mat.W_dec.data)
    W_topk = unit_norm(topk.W_dec.data)
    sims = W_mat.T @ W_topk  # (n_features, n_features_topk)
    best = sims.max(dim=1).values.cpu().numpy()
    thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
    print(f"  threshold   |  count above  |  fraction")
    print(f"  ----------- + ------------- + ----------")
    cross_arch_match = {}
    for t in thresholds:
        n_above = int((best > t).sum())
        frac = n_above / n_features
        print(f"  cos > {t:.2f}  | {n_above:>7,}    | {frac:.4f}")
        cross_arch_match[str(t)] = {"count": n_above, "fraction": frac}
    results["cross_arch_match"] = cross_arch_match

    # =====================================================================
    # Save
    # =====================================================================
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
