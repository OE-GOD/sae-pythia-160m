"""Train a TopK sparse autoencoder on cached layer-6 activations.

Difference vs 02_train_sae.py:
  - Encoder applies a TopK selection instead of ReLU + L1 penalty.
  - Hard sparsity constraint: at most K features active per token, by construction.
  - No L1 penalty in the loss. Just reconstruction.

Usage:
    python 02b_train_sae_topk.py --steps 1500 --k 64

The K argument controls how many features fire per token. Standard choices: 32, 64, 128.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """SAE with TopK activation: at most k features fire per token, by construction."""

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        # Decoder columns: dictionary atoms (the v_i's). Initialize unit-norm.
        W_dec = torch.randn(d_model, n_features)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: tied init.
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n_features))

    def encode(self, x):
        z = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        # TopK + ReLU: keep top-k pre-activations, zero everything else.
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        topk_vals = F.relu(topk_vals)            # in case any top-k are negative
        f = torch.zeros_like(z).scatter_(-1, topk_idx, topk_vals)
        return f

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--meta", type=str, default="data/acts_layer6_meta.json")
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--k", type=int, default=64, help="TopK sparsity (features active per token)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--log-every", type=int, default=300)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for SAE init + training sampling. Reproducibility for T4 study.")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_path = args.out or f"checkpoints/sae_layer6_topk{args.k}.pt"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"seed: {args.seed}")

    meta = json.loads(Path(args.meta).read_text())
    d_model = meta["d_model"]
    n_tokens = meta["n_tokens"]
    print(f"loaded meta: d_model={d_model}, n_tokens={n_tokens:,}")

    print("loading activations into RAM (no mmap — avoids disk-thrash)...")
    acts = np.load(args.acts)  # full load into RAM as fp16
    assert acts.shape == (n_tokens, d_model)
    print(f"loaded acts: {acts.shape} {acts.dtype} ({acts.nbytes / 1e9:.2f} GB)")

    # Center input around the data mean
    sample_idx = np.random.choice(n_tokens, size=min(50_000, n_tokens), replace=False)
    sample = acts[sample_idx].astype(np.float32)
    act_mean = torch.tensor(sample.mean(axis=0), dtype=torch.float32, device=device)

    sae = TopKSAE(d_model, args.n_features, k=args.k).to(device)
    sae.b_dec.data = act_mean.clone()
    print(
        f"TopKSAE: d_model={d_model}  n_features={args.n_features}  k={args.k}  "
        f"params={sum(p.numel() for p in sae.parameters()):,}"
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    print(f"\ntraining for {args.steps} steps  batch={args.batch_size}  lr={args.lr}  k={args.k}")
    Path(out_path).parent.mkdir(exist_ok=True)
    log_recon, log_l0 = [], []
    t0 = time.time()

    sae.train()
    for step in range(args.steps):
        idx = np.random.randint(0, n_tokens, size=args.batch_size)
        batch_np = acts[idx].astype(np.float32)
        x = torch.from_numpy(batch_np).to(device)

        x_hat, f = sae(x)
        recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        loss = recon  # no L1 penalty — TopK gives hard sparsity guarantee

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = (f > 0).float().sum(dim=-1).mean().item()
                # Variance explained on this batch
                xc = x - x.mean(dim=0, keepdim=True)
                var_x = xc.pow(2).sum().item()
                var_resid = (x - x_hat).pow(2).sum().item()
                ve = 1.0 - var_resid / max(var_x, 1e-9)
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  step {step:6d}  recon={recon.item():.3f}  L0={l0:.1f}  "
                f"var_explained={ve:.4f}  ({rate:.1f} step/s)",
                flush=True,
            )
            log_recon.append(recon.item())
            log_l0.append(l0)

    torch.save(
        {
            "model_state": sae.state_dict(),
            "d_model": d_model,
            "n_features": args.n_features,
            "k": args.k,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "log_recon": log_recon,
            "log_l0": log_l0,
            "meta": meta,
            "arch": "topk",
        },
        out_path,
    )
    print(f"\nsaved checkpoint to {out_path}")
    print(f"total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
