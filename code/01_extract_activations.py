"""Extract layer-6 residual-stream activations from Pythia-160M and save to disk.

Usage:
    python 01_extract_activations.py --n-tokens 100_000     # smoke (~2 min)
    python 01_extract_activations.py --n-tokens 1_000_000   # mid (~20 min)
    python 01_extract_activations.py --n-tokens 10_000_000  # full (~3-6 hr)

Output:
    data/acts_{layer}.npy      fp16, shape (n_tokens, d_model)
    data/acts_{layer}_meta.json  metadata for sanity-checking later

The dataset is NeelNanda/pile-10k — 10,000 documents from The Pile, the standard
mech-interp benchmark dataset. Small, fast, no auth needed.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Suppress the MPS-may-be-incorrect warning (we accept the risk for a small SAE).
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402
from datasets import load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--model", type=str, default="pythia-160m")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="data")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"device:    {device}")
    print(f"model:     {args.model}")
    print(f"layer:     {args.layer}")
    print(f"n_tokens:  {args.n_tokens:,}")
    print(f"seq_len:   {args.seq_len}")
    print(f"batch:     {args.batch_size}")

    print("\nloading model...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    d_model = model.cfg.d_model
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    print(f"  d_model: {d_model}, hook: {hook_name}")

    print("\nloading dataset (NeelNanda/pile-10k)...")
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=False)
    print(f"  {len(ds):,} docs available")

    # Pre-tokenize documents and concatenate into a long stream.
    # Then chunk into seq_len pieces. Standard mech-interp prep.
    print("\ntokenizing...")
    t0 = time.time()
    all_tokens: list[int] = []
    needed = args.n_tokens + args.seq_len * args.batch_size  # small overhead
    for doc in ds:
        toks = model.to_tokens(doc["text"], prepend_bos=False)[0].tolist()
        all_tokens.extend(toks)
        if len(all_tokens) >= needed:
            break
    print(f"  collected {len(all_tokens):,} tokens in {time.time() - t0:.1f}s")

    # Reshape into (n_seqs, seq_len)
    n_seqs = args.n_tokens // args.seq_len
    all_tokens = all_tokens[: n_seqs * args.seq_len]
    token_tensor = torch.tensor(all_tokens, dtype=torch.long).reshape(n_seqs, args.seq_len)
    print(f"  reshaped to ({n_seqs}, {args.seq_len}) — {n_seqs * args.seq_len:,} tokens")

    # Allocate output as fp16 memmap
    out_path = out_dir / f"acts_layer{args.layer}.npy"
    total_tokens = n_seqs * args.seq_len
    activations = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(total_tokens, d_model)
    )
    print(f"\nallocated {out_path} — {activations.nbytes / 1e9:.2f} GB")

    # Run forward passes batch by batch, stash residual-stream activations.
    print("\nrunning forward passes...")
    write_idx = 0
    n_batches = (n_seqs + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            batch = token_tensor[bi * args.batch_size : (bi + 1) * args.batch_size].to(device)
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
            act = cache[hook_name]  # (batch, seq_len, d_model)
            act_flat = act.reshape(-1, d_model).to(torch.float16).cpu().numpy()
            n = act_flat.shape[0]
            activations[write_idx : write_idx + n] = act_flat
            write_idx += n

            if (bi + 1) % 10 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                rate = write_idx / elapsed if elapsed > 0 else 0
                eta = (total_tokens - write_idx) / rate if rate > 0 else 0
                print(
                    f"  batch {bi + 1}/{n_batches}  "
                    f"tokens {write_idx:,}/{total_tokens:,}  "
                    f"{rate:,.0f} tok/s  ETA {eta:.0f}s",
                    flush=True,
                )

    activations.flush()
    del activations  # ensure mmap is closed

    meta = {
        "model": args.model,
        "layer": args.layer,
        "hook": hook_name,
        "d_model": d_model,
        "n_tokens": total_tokens,
        "seq_len": args.seq_len,
        "dtype": "float16",
        "dataset": "NeelNanda/pile-10k",
    }
    meta_path = out_dir / f"acts_layer{args.layer}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDONE  total {time.time() - t0:.1f}s")
    print(f"  activations: {out_path}")
    print(f"  metadata:    {meta_path}")


if __name__ == "__main__":
    main()
