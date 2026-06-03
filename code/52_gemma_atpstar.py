"""Gemma 2 2B replication of efficient AtP* using Gemma Scope SAE.

Architecture differences from Pythia-160M:
  - 26 layers (vs 12)
  - GQA: 8 Q heads, 4 K/V heads (per-Q-head ablation only)
  - RMSNorm instead of LayerNorm (interface identical via ln1 module)
  - d_head 256 (vs 64)
  - Sliding-window attention in some layers

Intervention: per-Q-head q_input ablation at downstream layer (subtract
f_clean * decoder_col_X from blocks.L.hook_q_input[batch, -1, q_head, :]).
Different from Pythia setup where we ablated q+k+v together; here we only
ablate Q-input due to GQA (K/V are shared across Q groups).

Setup:
  - SAE: gemma-scope-2b-pt-res, layer 12, width 16k, l0=82
  - Downstream layers: 13..25 (13 layers — far more depth to test than Pythia's 5)

Method (same as script 49):
  1. Single clean forward + backward. Cache:
     - pre-LN q_input, rotated q/k, attn scores, attn pattern (per L, per Q head)
     - gradient of metric w.r.t. attention pattern
  2. For each (L, h_q): closed-form Δpattern via softmax(clean + Δscores_from_q_only)
  3. AtP* estimate: -∇M_pattern · Δpattern

Compare to AP: actually run the perturbed forward per (L, h_q) and measure.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402
from sae_lens import SAE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gemma-2-2b")
    p.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res")
    p.add_argument("--sae_id", type=str, default="layer_12/width_16k/average_l0_82")
    p.add_argument("--sae_layer", type=int, default=12)
    p.add_argument("--context_window", type=int, default=30)
    p.add_argument("--n_features", type=int, default=3, help="Number of features to test")
    p.add_argument("--n_positions", type=int, default=2)
    p.add_argument("--top_firing_sample", type=int, default=5000,
                    help="Number of token positions to sample for firing-position search")
    p.add_argument("--feature_ids", type=str, default=None,
                    help="Comma-separated feature IDs to test (overrides n_features)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--run_ap", action="store_true",
                    help="Also run full AP for comparison (expensive)")
    p.add_argument("--downstream_subset", type=str, default=None,
                    help="Comma-sep subset of downstream layers to test (e.g. '13,18,22,25')")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    print(f"loading SAE from sae_lens: {args.sae_release}/{args.sae_id}")
    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=device)
    sae_hook_name = sae.cfg.metadata["hook_name"]
    print(f"  d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}, hook: {sae_hook_name}")

    print(f"loading {args.model}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()
    model.set_use_split_qkv_input(True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads  # Q heads
    n_kv = model.cfg.n_key_value_heads
    d_head = model.cfg.d_head
    sae_layer = args.sae_layer
    if args.downstream_subset:
        downstream_layers = [int(x) for x in args.downstream_subset.split(",")]
    else:
        downstream_layers = list(range(sae_layer + 1, n_layers))
    print(f"  n_layers={n_layers}, n_Q_heads={n_heads}, n_KV_heads={n_kv}, d_head={d_head}")
    print(f"  SAE at layer {sae_layer}, downstream: {downstream_layers}")

    # Get some prompts for sampling firing positions
    # Use a small slice of pile-like data
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=False)
    # tokenize a small batch
    prompts = []
    for i in range(50):
        prompts.append(ds[i]["text"])
    print(f"  loaded {len(prompts)} prompts for firing-position search")

    # Build a corpus of token sequences
    tokenizer = model.tokenizer
    token_seqs = []
    for p in prompts:
        toks = tokenizer.encode(p)[:args.context_window]
        if len(toks) >= args.context_window:
            token_seqs.append(toks)
    print(f"  {len(token_seqs)} token sequences of length {args.context_window}")

    # Find features and their firing positions
    # For simplicity, scan a few sequences and find tokens where features fire strongly
    print(f"\n--- Finding firing positions ---")

    all_features_data = {}

    # If user specified feature_ids, use those
    if args.feature_ids:
        target_features = [int(x) for x in args.feature_ids.split(",")]
    else:
        target_features = None  # discover from scanning

    # Scan corpus, encode at SAE layer, find top firing positions per feature
    # Only consider positions late in context (>= half) so the model has enough
    # context to develop the SAE feature and predict a meaningful next token.
    min_pos = args.context_window // 2
    feature_max_act = {}  # feature_id -> list of (act, seq_idx, pos)
    n_to_scan = min(len(token_seqs), 30)
    print(f"  scanning {n_to_scan} sequences for firing positions...")
    for seq_idx in range(n_to_scan):
        tokens = torch.tensor([token_seqs[seq_idx]], device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[sae_hook_name])
            resid = cache[sae_hook_name][0]  # [seq, d_model]
            f_acts = sae.encode(resid)  # [seq, d_sae]
            for pos in range(f_acts.shape[0]):
                if pos < min_pos:  # Skip early-context positions
                    continue
                top_k = 64
                top_vals, top_ids = f_acts[pos].topk(top_k)
                for v, fid in zip(top_vals.cpu().numpy(), top_ids.cpu().numpy()):
                    if v < 0.5:
                        continue
                    fid = int(fid)
                    if target_features and fid not in target_features:
                        continue
                    if fid not in feature_max_act:
                        feature_max_act[fid] = []
                    feature_max_act[fid].append((float(v), seq_idx, pos))
        if seq_idx % 5 == 0:
            print(f"    scanned {seq_idx+1}/{n_to_scan} sequences, found {len(feature_max_act)} features")

    # Pick top features by max firing strength (or use specified)
    if target_features:
        candidate_features = [f for f in target_features if f in feature_max_act]
    else:
        feature_strengths = [(fid, max(v for v, _, _ in firings)) for fid, firings in feature_max_act.items()]
        feature_strengths.sort(key=lambda x: -x[1])
        candidate_features = [fid for fid, _ in feature_strengths[:args.n_features * 3]]

    print(f"\n  feature candidates: {len(candidate_features)}")

    # For each candidate, get top firing positions
    selected_features = []
    for fid in candidate_features[:args.n_features * 2]:
        firings = sorted(feature_max_act[fid], key=lambda x: -x[0])[:5]
        if len(firings) >= 1:
            selected_features.append((fid, firings))
        if len(selected_features) >= args.n_features:
            break

    print(f"  using {len(selected_features)} features")

    # Hook names for caching
    rot_k_hooks = [f"blocks.{L}.attn.hook_rot_k" for L in downstream_layers]
    rot_q_hooks = [f"blocks.{L}.attn.hook_rot_q" for L in downstream_layers]
    scores_hooks = [f"blocks.{L}.attn.hook_attn_scores" for L in downstream_layers]
    pattern_hooks = [f"blocks.{L}.attn.hook_pattern" for L in downstream_layers]
    q_input_hooks = [f"blocks.{L}.hook_q_input" for L in downstream_layers]
    cache_hook_names = rot_k_hooks + rot_q_hooks + scores_hooks + pattern_hooks + q_input_hooks

    all_results = {}
    t0 = time.time()

    for fid, firings in selected_features:
        print(f"\n{'=' * 70}\nFEATURE f{fid}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[fid].detach().to(torch.float32)  # [d_in=d_model]

        per_position_atpstar = []
        per_position_ap = [] if args.run_ap else None
        per_position_meta = []

        for cand_act, seq_idx, pos in firings:
            if len(per_position_atpstar) >= args.n_positions:
                break
            if pos < 1:
                continue
            cand_tokens = token_seqs[seq_idx][:pos + 1]
            if len(cand_tokens) < 2:
                continue
            tokens = torch.tensor([cand_tokens], device=device)
            seq_len = tokens.shape[1]
            last_pos = seq_len - 1

            # Clean forward + backward
            saved = {}

            def make_capture(name):
                def hook(activation, hook):
                    activation.retain_grad()
                    saved[name] = activation
                    return activation
                return hook

            hooks_list = [(n, make_capture(n)) for n in cache_hook_names]

            with model.hooks(fwd_hooks=hooks_list):
                clean_logits = model(tokens)

            with torch.no_grad():
                _, c2 = model.run_with_cache(tokens, names_filter=[sae_hook_name])
                clean_resid = c2[sae_hook_name][0, last_pos, :]
                f_clean = sae.encode(clean_resid.unsqueeze(0))[0, fid].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(cand_tokens[-1])  # Predict the existing last token
            # Actually we want to predict the NEXT token after `last_pos`, but we only
            # cached tokens up to pos+1. Let's use the next-token prediction from the
            # token BEFORE last_pos as the target. We'll predict the last token in our
            # sequence using context [0..last_pos-1].
            # Use sequence [0..last_pos-1] and predict token at last_pos.
            target_token = cand_tokens[-1]

            # Re-run with shortened input
            tokens_short = tokens[:, :-1]
            seq_len = tokens_short.shape[1]
            last_pos = seq_len - 1
            with model.hooks(fwd_hooks=hooks_list):
                clean_logits = model(tokens_short)

            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[target_token].item()
            metric = torch.log_softmax(clean_logits[0, -1, :], dim=-1)[target_token]
            model.zero_grad()
            metric.backward()

            # Snapshot
            clean_rot_k = {L: saved[f"blocks.{L}.attn.hook_rot_k"].detach().clone()
                           for L in downstream_layers}
            clean_rot_q = {L: saved[f"blocks.{L}.attn.hook_rot_q"].detach().clone()
                           for L in downstream_layers}
            clean_scores = {L: saved[f"blocks.{L}.attn.hook_attn_scores"].detach().clone()
                            for L in downstream_layers}
            clean_patterns = {L: saved[f"blocks.{L}.attn.hook_pattern"].detach().clone()
                              for L in downstream_layers}
            clean_q_inputs = {L: saved[f"blocks.{L}.hook_q_input"].detach().clone()
                              for L in downstream_layers}
            pattern_grads = {}
            for L in downstream_layers:
                pn = f"blocks.{L}.attn.hook_pattern"
                g = saved[pn].grad
                pattern_grads[L] = g.detach().clone() if g is not None else None
            saved.clear()

            print(f"  seq{seq_idx} pos{pos}: f_X={f_clean:.2f}, target_token={target_token}, "
                  f"baseline logP={baseline_logp:+.3f}")

            sqrt_d_head = math.sqrt(d_head)
            atpstar_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)
            ap_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)

            for li, L in enumerate(downstream_layers):
                attn = model.blocks[L].attn
                ln1 = model.blocks[L].ln1
                W_Q = attn.W_Q  # [Q_head, d_model, d_head]
                clean_q_input_lh = clean_q_inputs[L][0, last_pos, :, :].to(torch.float32)
                pert_q_input_lh = clean_q_input_lh - f_clean * decoder_col_X[None, :]
                with torch.no_grad():
                    clean_ln = ln1(clean_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
                    pert_ln = ln1(pert_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
                delta_ln = pert_ln - clean_ln  # [Q_head, d_model]

                for h_q in range(n_heads):
                    # Only this Q head's input changes
                    dq = delta_ln[h_q] @ W_Q[h_q].to(torch.float32)  # [d_head]
                    # Rotate at position last
                    x = torch.zeros(1, seq_len, n_heads, d_head, device=device, dtype=dq.dtype)
                    x[0, last_pos, h_q, :] = dq
                    drot_q = attn.apply_rotary(x)[0, last_pos, h_q, :].to(torch.float32)

                    # Δscores: only Q changes, so for this Q head's row last:
                    # Need clean_rot_k for the K head this Q head reads from.
                    # With GQA: Q head h_q reads from KV head h_q // (n_heads // n_kv)
                    kv_group = n_heads // n_kv  # how many Q heads share each KV
                    h_kv = h_q // kv_group
                    # rot_k shape: [batch, seq, n_kv, d_head] for GQA TL? Or [batch, seq, n_heads, d_head]?
                    # Check shape
                    if clean_rot_k[L].shape[2] == n_kv:
                        clean_rot_k_h = clean_rot_k[L][0, :, h_kv, :].to(torch.float32)
                    else:
                        clean_rot_k_h = clean_rot_k[L][0, :, h_q, :].to(torch.float32)

                    delta_scores_row = (drot_q @ clean_rot_k_h.T) / sqrt_d_head

                    clean_scores_row = clean_scores[L][0, h_q, last_pos, :].to(torch.float32)
                    patched_scores_row = clean_scores_row + delta_scores_row
                    patched_pattern_row = torch.softmax(patched_scores_row, dim=-1)
                    clean_pattern_row = clean_patterns[L][0, h_q, last_pos, :].to(torch.float32)
                    delta_pattern_row = patched_pattern_row - clean_pattern_row

                    g_pat = pattern_grads[L][0, h_q, last_pos, :].to(torch.float32) \
                        if pattern_grads[L] is not None else None
                    effect_qk = -float(torch.dot(g_pat, delta_pattern_row).item()) \
                        if g_pat is not None else 0.0
                    atpstar_mediation[li, h_q] = effect_qk

                    # AP comparison (if requested): run actual perturbed forward
                    if args.run_ap:
                        def make_qhook(head_idx, layer_idx):
                            def hook_fn(activation, hook):
                                activation[:, -1, head_idx, :] = (
                                    activation[:, -1, head_idx, :] - f_clean * decoder_col_X
                                )
                                return activation
                            return hook_fn
                        q_hook = make_qhook(h_q, L)
                        with torch.no_grad():
                            pl = model.run_with_hooks(
                                tokens_short,
                                fwd_hooks=[(f"blocks.{L}.hook_q_input", q_hook)]
                            )
                            ppl = torch.log_softmax(pl[0, -1, :], dim=-1)[target_token].item()
                        ap_mediation[li, h_q] = baseline_logp - ppl

            per_position_atpstar.append(atpstar_mediation)
            if args.run_ap:
                per_position_ap.append(ap_mediation)
            per_position_meta.append({
                "seq_idx": int(seq_idx),
                "pos": int(pos),
                "f_clean": f_clean,
                "target_token": int(target_token),
                "baseline_logp": baseline_logp,
            })
            top_idx = np.argsort(-np.abs(atpstar_mediation).flatten())[:5]
            print(f"    top |AtP*|: " + ", ".join(
                f"L{downstream_layers[i // n_heads]}H{i % n_heads}={atpstar_mediation.flatten()[i]:+.4f}"
                for i in top_idx
            ))

        if not per_position_atpstar:
            continue

        # Aggressive memory cleanup between features
        del clean_rot_k, clean_rot_q, clean_scores, clean_patterns
        del clean_q_inputs, pattern_grads
        if device == "mps":
            torch.mps.empty_cache()

        mean_atpstar = np.mean(np.stack(per_position_atpstar), axis=0)
        result = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_atpstar),
            "per_position_meta": per_position_meta,
            "mean_atpstar_mediation": mean_atpstar.tolist(),
        }
        if args.run_ap:
            mean_ap = np.mean(np.stack(per_position_ap), axis=0)
            result["mean_ap_mediation"] = mean_ap.tolist()
        all_results[fid] = result

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or f"checkpoints/gemma2_2b_layer{args.sae_layer}_atpstar.json"
    Path(out_path).write_text(json.dumps({
        "model": args.model,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "sae_layer": args.sae_layer,
        "downstream_layers": downstream_layers,
        "method": "atp_star_efficient_per_q_head_gemma",
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
