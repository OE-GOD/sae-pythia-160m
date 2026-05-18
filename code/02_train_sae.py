"""Train a sparse autoencoder on cached layer-6 activations.

Usage:
    python 02_train_sae.py --steps 2000     # smoke
    python 02_train_sae.py --steps 50000    # real run

The SAE is the Anthropic "Towards Monosemanticity" recipe:
  - Encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)
    (subtracting b_dec centers the input on the data mean — stabilizes training)
  - Decoder: f -> W_dec @ f + b_dec
  - Loss:    ||x - x_hat||^2 + lambda * ||f||_1
  - Constraint: W_dec columns are unit-norm (renormalized after each step)
  - Init: tied — W_enc = W_dec.T at start
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Vanilla L1 SAE with tied init and decoder unit-norm constraint."""

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Decoder: columns are the dictionary atoms (the v_i's from the paper)
        W_dec = torch.randn(d_model, n_features)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)  # unit-norm columns
        self.W_dec = nn.Parameter(W_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: tied init (W_enc = W_dec.T) — standard practice
        self.W_enc = nn.Parameter(W_dec.T.clone())  # (n_features, d_model)
        self.b_enc = nn.Parameter(torch.zeros(n_features))

    def encode(self, x):
        # Center input on b_dec, project up, ReLU
        z = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        return F.relu(z)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder(self):
        """Project W_dec columns back to unit-norm. Prevents the SAE from cheating
        the L1 penalty by making W_dec huge and f tiny."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--meta", type=str, default="data/acts_layer6_meta.json")
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--lambda-sparsity", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints/sae_layer6.pt")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    # Load metadata + memmap activations
    meta = json.loads(Path(args.meta).read_text())
    d_model = meta["d_model"]
    n_tokens = meta["n_tokens"]
    print(f"loaded meta: d_model={d_model}, n_tokens={n_tokens:,}")

    acts = np.load(args.acts, mmap_mode="r")
    assert acts.shape == (n_tokens, d_model)
    print(f"loaded acts: {acts.shape} {acts.dtype} ({acts.nbytes / 1e9:.2f} GB)")

    # Compute mean of activations — used to initialize b_dec to data center
    print("computing activation mean for b_dec init...")
    sample_idx = np.random.choice(n_tokens, size=min(50_000, n_tokens), replace=False)
    sample = acts[sample_idx].astype(np.float32)
    act_mean = torch.tensor(sample.mean(axis=0), dtype=torch.float32, device=device)
    print(f"  mean stats: |mean|={act_mean.abs().mean():.3f}  max={act_mean.abs().max():.3f}")

    # Build SAE
    sae = SparseAutoencoder(d_model, args.n_features).to(device)
    sae.b_dec.data = act_mean.clone()
    print(
        f"SAE: d_model={d_model}  n_features={args.n_features}  "
        f"params={sum(p.numel() for p in sae.parameters()):,}"
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    # Training loop
    print(f"\ntraining for {args.steps} steps  batch={args.batch_size}  lr={args.lr}  "
          f"lambda={args.lambda_sparsity}")
    Path(args.out).parent.mkdir(exist_ok=True)
    log_recon, log_sparse, log_l0 = [], [], []
    t0 = time.time()

    sae.train()
    for step in range(args.steps):
        idx = np.random.randint(0, n_tokens, size=args.batch_size)
        batch_np = acts[idx].astype(np.float32)
        x = torch.from_numpy(batch_np).to(device)

        x_hat, f = sae(x)
        recon = (x - x_hat).pow(2).sum(dim=-1).mean()  # MSE summed over d_model
        sparsity = f.abs().sum(dim=-1).mean()           # L1 of features per token
        loss = recon + args.lambda_sparsity * sparsity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = (f > 0).float().sum(dim=-1).mean().item()
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  step {step:6d}  recon={recon.item():.3f}  "
                f"sparse={sparsity.item():.2f}  L0={l0:.1f}  "
                f"loss={loss.item():.3f}  ({rate:.1f} step/s)",
                flush=True,
            )
            log_recon.append(recon.item())
            log_sparse.append(sparsity.item())
            log_l0.append(l0)

    # Save checkpoint
    torch.save(
        {
            "model_state": sae.state_dict(),
            "d_model": d_model,
            "n_features": args.n_features,
            "lambda_sparsity": args.lambda_sparsity,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "log_recon": log_recon,
            "log_sparse": log_sparse,
            "log_l0": log_l0,
            "meta": meta,
        },
        args.out,
    )
    print(f"\nsaved checkpoint to {args.out}")
    print(f"total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
