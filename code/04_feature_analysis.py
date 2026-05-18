"""Extract top-K max-activating text snippets for every SAE feature.

For each of the 16,384 features in the trained SAE:
  1. Find the K text positions where that feature activated most strongly.
  2. Pull a context window (12 tokens before + 4 after) for each position.

Output:
  data/top_features.npz   # top_values (n_features, K), top_global_ids (n_features, K)
  data/token_stream.npy   # int32 array of all tokens we processed
  data/feature_examples.json  # human-readable {feature_id: [{activation, snippet, ...}]}

Used by W3 auto-interp + manual inspection.

Usage:
    python 04_feature_analysis.py
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
from datasets import load_dataset  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--n-tokens", type=int, default=1_000_000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=10, help="Max-activating examples per feature")
    p.add_argument("--ctx-before", type=int, default=12)
    p.add_argument("--ctx-after", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="data")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load SAE
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    k_active = ckpt["k"]
    sae = TopKSAE(d_model, n_features, k=k_active).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    print(f"  d_model={d_model}  n_features={n_features}  k={k_active}")

    # Load Pythia (same model that produced the activations)
    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"\nloading {model_name} (hook: {hook_name})...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # Tokenize Pile-10k into a flat stream
    print("\nloading + tokenizing Pile-10k...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    all_tokens: list[int] = []
    target = args.n_tokens + args.seq_len * args.batch_size
    for doc in ds:
        toks = model.to_tokens(doc["text"], prepend_bos=False)[0].tolist()
        all_tokens.extend(toks)
        if len(all_tokens) >= target:
            break

    n_seqs = args.n_tokens // args.seq_len
    total_tokens = n_seqs * args.seq_len
    all_tokens = all_tokens[:total_tokens]
    token_stream = np.array(all_tokens, dtype=np.int32)
    token_tensor = torch.tensor(all_tokens, dtype=torch.long).reshape(n_seqs, args.seq_len)
    print(f"  prepared {total_tokens:,} tokens in {n_seqs} sequences of {args.seq_len}")

    # Save the token stream so we can produce snippets later
    np.save(out_dir / "token_stream.npy", token_stream)

    # Init top-K trackers on device
    K = args.top_k
    top_values = torch.full((n_features, K), float("-inf"), device=device)
    top_global_ids = torch.full((n_features, K), -1, dtype=torch.long, device=device)

    n_batches = (n_seqs + args.batch_size - 1) // args.batch_size
    print(f"\nrunning forward passes + extracting features  ({n_batches} batches)...")
    t0 = time.time()
    global_offset = 0
    with torch.no_grad():
        for bi in range(n_batches):
            batch_tokens = token_tensor[bi * args.batch_size : (bi + 1) * args.batch_size].to(device)
            n_tokens_this_batch = batch_tokens.numel()

            # Pythia forward
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            h = cache[hook_name].reshape(-1, d_model).to(torch.float32)  # (B*T, d_model)

            # SAE forward — gives sparse features
            f = sae.encode(h)  # (B*T, n_features)

            # Top-K per feature within this chunk: torch.topk over the token axis
            chunk_top_vals, chunk_top_pos = f.topk(K, dim=0)  # both (K, n_features)
            chunk_top_global = chunk_top_pos + global_offset  # global token IDs

            # Merge with running top-K. Shapes: (n_features, 2K)
            combined_vals = torch.cat([top_values, chunk_top_vals.T], dim=1)
            combined_ids = torch.cat([top_global_ids, chunk_top_global.T], dim=1)
            merged_vals, sel = combined_vals.topk(K, dim=1)
            top_values = merged_vals
            top_global_ids = combined_ids.gather(1, sel)

            global_offset += n_tokens_this_batch

            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                rate = global_offset / elapsed if elapsed > 0 else 0
                eta = (total_tokens - global_offset) / rate if rate > 0 else 0
                print(
                    f"  batch {bi + 1}/{n_batches}  tokens {global_offset:,}/{total_tokens:,}  "
                    f"{rate:,.0f} tok/s  ETA {eta:.0f}s",
                    flush=True,
                )

    print(f"\nfeature extraction done in {time.time() - t0:.1f}s")

    # Save raw top-K
    top_values_np = top_values.cpu().numpy().astype(np.float32)
    top_global_ids_np = top_global_ids.cpu().numpy().astype(np.int64)
    np.savez(
        out_dir / "top_features.npz",
        top_values=top_values_np,
        top_global_ids=top_global_ids_np,
    )
    print(f"saved data/top_features.npz")

    # Build a human-readable JSON of top features for interactive inspection.
    # Limit to alive features (those with any nonzero top value).
    print("\nbuilding feature_examples.json (decoding snippets)...")
    feature_examples = {}
    alive_mask = top_values_np[:, 0] > 0
    alive_ids = np.where(alive_mask)[0]
    print(f"  alive features: {len(alive_ids)} / {n_features}")

    for fid in alive_ids:
        examples = []
        for k in range(K):
            v = float(top_values_np[fid, k])
            gid = int(top_global_ids_np[fid, k])
            if not np.isfinite(v) or v <= 0:
                break
            ctx_start = max(0, gid - args.ctx_before)
            ctx_end = min(total_tokens, gid + args.ctx_after + 1)
            snippet_tokens = token_stream[ctx_start:ctx_end].tolist()
            target_pos = gid - ctx_start
            snippet_str = model.tokenizer.decode(snippet_tokens, skip_special_tokens=False)
            target_tok = model.tokenizer.decode([int(token_stream[gid])], skip_special_tokens=False)
            examples.append({
                "activation": v,
                "token_id": int(token_stream[gid]),
                "target_token": target_tok,
                "snippet": snippet_str,
                "target_position_in_snippet": target_pos,
                "global_token_id": gid,
            })
        feature_examples[int(fid)] = examples

    out_json = out_dir / "feature_examples.json"
    out_json.write_text(json.dumps(feature_examples, indent=2, ensure_ascii=False))
    print(f"saved {out_json}  ({out_json.stat().st_size / 1e6:.1f} MB)")

    # Quick top-summary: print the 10 features that fire most often
    fire_counts = (top_values > 0).sum(dim=1).cpu().numpy()
    most_active = np.argsort(-fire_counts)[:10]
    print(f"\nfeatures with strongest top-K (full top-K all positive):")
    for fid in most_active[:10]:
        if not feature_examples.get(int(fid)):
            continue
        ex0 = feature_examples[int(fid)][0]
        snippet = ex0["snippet"].replace("\n", " ")[:80]
        print(f"  f{int(fid):5d}  max_act={ex0['activation']:.2f}  → {snippet!r}")


if __name__ == "__main__":
    main()
