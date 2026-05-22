"""Matryoshka SAE — nested-K reconstruction.

Adapted from 02b_train_sae_topk.py. The architectural change:

  Standard TopK SAE:
      f = TopK(z, k)            # one bottleneck
      x_hat = decode(f)         # one reconstruction
      loss = ||x - x_hat||^2

  Matryoshka SAE (Bussmann et al. 2025):
      For nested k_1 < k_2 < ... < k_N:
          f_i = TopK(z, k_i)            # top-k_i features (subset of higher levels)
          x_hat_i = decode(f_i)         # reconstruct at this level
      loss = sum_i ||x - x_hat_i||^2

The nesting (k_1 < k_2 < ... = same z used, different cutoffs) forces a
hierarchy among the features: features that survive the smallest k are the
"most important" — they have to reconstruct alone. Features active only at
larger k are "specializations" that refine the reconstruction.

Default level config: k_levels = {16, 64, 256}. The largest k (256) is the
effective L_0 for any token (the SAE can still use up to max_k features).

Usage:
    python 02c_train_sae_matryoshka.py --steps 20000 --k-levels 16,64,256
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaSAE(nn.Module):
    """SAE with nested TopK bottlenecks. Shares encoder + decoder across levels."""

    def __init__(self, d_model: int, n_features: int, k_levels: list[int]):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        # k_levels is the list of nested cutoffs, smallest first
        self.k_levels = sorted(list(k_levels))
        assert all(1 <= k <= n_features for k in self.k_levels)
        self.max_k = self.k_levels[-1]

        # Decoder columns initialised unit-norm
        W_dec = torch.randn(d_model, n_features)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Tied init for encoder
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n_features))

    def encode_levels(self, x):
        """Return a list of feature tensors, one per k_level (nested).

        Each f_i has at most k_i nonzero entries, all of which are also nonzero
        in f_{i+1} — i.e., the smaller bottlenecks are strict subsets of the
        larger ones, by construction (top-k of the same z).
        """
        z = (x - self.b_dec) @ self.W_enc.T + self.b_enc  # (B, N)
        # Take top max_k once; smaller levels are prefixes
        topk_vals, topk_idx = z.topk(self.max_k, dim=-1)  # both (B, max_k)
        topk_vals = F.relu(topk_vals)
        levels = []
        for k in self.k_levels:
            level_vals = topk_vals[:, :k]                  # (B, k)
            level_idx = topk_idx[:, :k]                    # (B, k)
            f = torch.zeros_like(z).scatter_(-1, level_idx, level_vals)
            levels.append(f)
        return levels  # length = len(k_levels)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        feature_levels = self.encode_levels(x)
        x_hats = [self.decode(f) for f in feature_levels]
        return x_hats, feature_levels

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--meta", type=str, default="data/acts_layer6_meta.json")
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--k-levels", type=str, default="16,64,256",
                   help="Comma-separated nested k values, smallest first")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints/sae_layer6_matryoshka.pt")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    k_levels = [int(k) for k in args.k_levels.split(",") if k.strip()]
    print(f"device: {device}")
    print(f"k_levels: {k_levels}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"seed: {args.seed}")

    meta = json.loads(Path(args.meta).read_text())
    d_model = meta["d_model"]
    n_tokens = meta["n_tokens"]
    print(f"loaded meta: d_model={d_model}, n_tokens={n_tokens:,}")

    print("loading activations into RAM (fp16)...")
    acts = np.load(args.acts)
    assert acts.shape == (n_tokens, d_model)
    print(f"loaded acts: {acts.shape} {acts.dtype} ({acts.nbytes / 1e9:.2f} GB)")

    sample_idx = np.random.choice(n_tokens, size=min(50_000, n_tokens), replace=False)
    sample = acts[sample_idx].astype(np.float32)
    act_mean = torch.tensor(sample.mean(axis=0), dtype=torch.float32, device=device)

    sae = MatryoshkaSAE(d_model, args.n_features, k_levels).to(device)
    sae.b_dec.data = act_mean.clone()
    print(
        f"MatryoshkaSAE: d_model={d_model}  n_features={args.n_features}  "
        f"levels={k_levels}  params={sum(p.numel() for p in sae.parameters()):,}"
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    print(f"\ntraining {args.steps} steps  batch={args.batch_size}  lr={args.lr}")
    Path(args.out).parent.mkdir(exist_ok=True)
    log_per_level = {k: [] for k in k_levels}
    log_total = []
    t0 = time.time()

    sae.train()
    for step in range(args.steps):
        idx = np.random.randint(0, n_tokens, size=args.batch_size)
        batch_np = acts[idx].astype(np.float32)
        x = torch.from_numpy(batch_np).to(device)

        x_hats, feature_levels = sae(x)
        # MSE per level, summed over d_model dim, averaged over batch
        recon_per_level = [
            (x - xh).pow(2).sum(dim=-1).mean() for xh in x_hats
        ]
        loss = sum(recon_per_level)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                per_level_str = "  ".join(
                    f"k={k}:recon={r.item():.2f}"
                    for k, r in zip(k_levels, recon_per_level)
                )
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  step {step:6d}  total={loss.item():.2f}  {per_level_str}  ({rate:.1f} step/s)",
                flush=True,
            )
            log_total.append(loss.item())
            for k, r in zip(k_levels, recon_per_level):
                log_per_level[k].append(r.item())

    torch.save(
        {
            "model_state": sae.state_dict(),
            "d_model": d_model,
            "n_features": args.n_features,
            "k_levels": k_levels,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "log_total": log_total,
            "log_per_level": log_per_level,
            "meta": meta,
            "arch": "matryoshka",
        },
        args.out,
    )
    print(f"\nsaved checkpoint to {args.out}")
    print(f"total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
