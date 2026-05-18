"""Measure CE delta — how much information the SAE actually loses.

Standard mech-interp evaluation:
  1. For a held-out text stream, run Pythia normally and record per-token NLL → CE_clean
  2. Run Pythia again, but at layer 6 replace h with SAE(h) → record per-token NLL → CE_sae
  3. ΔCE = CE_sae - CE_clean

A low ΔCE means the SAE's reconstruction preserved the information Pythia uses
downstream. A high ΔCE means our 98.4% variance-explained number was misleading —
the SAE was throwing away exactly the bits that matter.

Reference baseline (zero ablation): replace h with the data mean instead of SAE(h).
This is "how bad does it get if we destroy layer 6 entirely?" Used to put ΔCE into
context.

Usage:
    python 06_ce_delta.py
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
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--n-tokens", type=int, default=100_000,
                   help="Number of held-out tokens to evaluate on")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-docs", type=int, default=8000,
                   help="Skip first N docs (already used for SAE training); use later docs for eval")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    k_active = ckpt["k"]
    sae = TopKSAE(d_model, n_features, k=k_active).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    print(f"  d_model={d_model}  n_features={n_features}  k={k_active}")

    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"\nloading {model_name}, hook={hook_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # Held-out text: skip the docs we trained on, take fresh ones
    print(f"\nloading Pile-10k, skipping first {args.skip_docs} docs (training set)...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    held_out_docs = list(ds)[args.skip_docs:]
    print(f"  {len(held_out_docs)} held-out docs available")

    # Tokenize until we have enough
    all_tokens: list[int] = []
    needed = args.n_tokens + args.seq_len * args.batch_size
    for doc in held_out_docs:
        toks = model.to_tokens(doc["text"], prepend_bos=False)[0].tolist()
        all_tokens.extend(toks)
        if len(all_tokens) >= needed:
            break
    n_seqs = args.n_tokens // args.seq_len
    total_tokens = n_seqs * args.seq_len
    all_tokens = all_tokens[:total_tokens]
    token_tensor = torch.tensor(all_tokens, dtype=torch.long).reshape(n_seqs, args.seq_len)
    print(f"  prepared {total_tokens:,} eval tokens in {n_seqs} sequences")

    # ----------------------------------------------------------------------
    # Run 1: clean forward pass — get baseline NLL per token
    # ----------------------------------------------------------------------
    print(f"\n[run 1/3] clean Pythia (baseline CE)...")
    clean_nll_total = 0.0
    n_pred_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, n_seqs, args.batch_size):
            batch = token_tensor[bi : bi + args.batch_size].to(device)
            logits = model(batch)  # (B, T, V)
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_targets.reshape(-1),
                reduction="sum",
            )
            clean_nll_total += ce.item()
            n_pred_tokens += shift_targets.numel()
    clean_nll_mean = clean_nll_total / n_pred_tokens
    print(f"  clean CE per token: {clean_nll_mean:.4f}  ({time.time()-t0:.1f}s)")

    # ----------------------------------------------------------------------
    # Run 2: SAE-substituted forward pass — replace h_6 with SAE(h_6)
    # ----------------------------------------------------------------------
    print(f"\n[run 2/3] SAE-substituted Pythia...")

    def sae_hook(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        original_shape = activation.shape
        flat = activation.reshape(-1, d_model).to(torch.float32)
        with torch.no_grad():
            x_hat, _ = sae(flat)
        return x_hat.to(activation.dtype).reshape(original_shape)

    sae_nll_total = 0.0
    n_pred_tokens_sae = 0
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, n_seqs, args.batch_size):
            batch = token_tensor[bi : bi + args.batch_size].to(device)
            logits = model.run_with_hooks(
                batch, fwd_hooks=[(hook_name, sae_hook)]
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_targets.reshape(-1),
                reduction="sum",
            )
            sae_nll_total += ce.item()
            n_pred_tokens_sae += shift_targets.numel()
    sae_nll_mean = sae_nll_total / n_pred_tokens_sae
    print(f"  SAE-patched CE per token: {sae_nll_mean:.4f}  ({time.time()-t0:.1f}s)")

    # ----------------------------------------------------------------------
    # Run 3: zero-ablation baseline — replace h_6 with the data mean
    # Used as "context" for how bad CE could be.
    # ----------------------------------------------------------------------
    print(f"\n[run 3/3] zero-ablation Pythia (replace h with mean)...")
    # b_dec already holds the data mean from training init
    mean_h = sae.b_dec.detach().clone()

    def mean_hook(activation, hook):
        # Replace all activations with the global mean
        return mean_h.expand_as(activation).to(activation.dtype)

    abl_nll_total = 0.0
    n_pred_tokens_abl = 0
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, n_seqs, args.batch_size):
            batch = token_tensor[bi : bi + args.batch_size].to(device)
            logits = model.run_with_hooks(
                batch, fwd_hooks=[(hook_name, mean_hook)]
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_targets.reshape(-1),
                reduction="sum",
            )
            abl_nll_total += ce.item()
            n_pred_tokens_abl += shift_targets.numel()
    abl_nll_mean = abl_nll_total / n_pred_tokens_abl
    print(f"  zero-ablation CE per token: {abl_nll_mean:.4f}  ({time.time()-t0:.1f}s)")

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------
    ce_delta = sae_nll_mean - clean_nll_mean
    ablation_delta = abl_nll_mean - clean_nll_mean
    recovery = 1.0 - (ce_delta / ablation_delta) if ablation_delta > 0 else float("nan")
    bpb_clean = clean_nll_mean / np.log(2)
    bpb_sae = sae_nll_mean / np.log(2)
    bpb_delta = bpb_sae - bpb_clean

    print(f"\n{'=' * 60}")
    print("CE DELTA RESULTS")
    print(f"{'=' * 60}")
    print(f"  clean CE per token:        {clean_nll_mean:.4f}  ({bpb_clean:.4f} bits/token)")
    print(f"  SAE-patched CE per token:  {sae_nll_mean:.4f}  ({bpb_sae:.4f} bits/token)")
    print(f"  zero-ablation CE per token:{abl_nll_mean:.4f}")
    print(f"")
    print(f"  ΔCE  (SAE - clean):        {ce_delta:.4f} nats/token")
    print(f"        in bits:             {bpb_delta:.4f} bits/token")
    print(f"  zero-ablation Δ:           {ablation_delta:.4f} nats/token")
    print(f"")
    print(f"  Loss recovered:            {recovery * 100:.1f}%")
    print(f"    (fraction of zero-ablation loss the SAE preserves)")
    print(f"    1.0 = perfect, 0.0 = no better than destroying layer 6")
    print(f"{'=' * 60}")

    results = {
        "clean_ce_per_token": clean_nll_mean,
        "sae_ce_per_token": sae_nll_mean,
        "zero_ablation_ce_per_token": abl_nll_mean,
        "ce_delta_nats": ce_delta,
        "ce_delta_bits": bpb_delta,
        "zero_ablation_delta_nats": ablation_delta,
        "loss_recovered_pct": float(recovery * 100),
        "n_eval_tokens": n_pred_tokens,
        "ckpt": args.ckpt,
        "layer": layer,
    }
    out_path = Path(args.ckpt).with_suffix(".ce_delta.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
