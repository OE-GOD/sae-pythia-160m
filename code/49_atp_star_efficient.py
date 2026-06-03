"""Efficient AtP* — closed-form softmax-corrected AtP at AtP's cost (2 passes).

Builds on script 47 (validation AtP*) which used extra forward passes to
get patched patterns. This version computes patched patterns in CLOSED FORM
using cached clean Q/K/scores and the model's rotary embedding function.

Algorithm:
  1. Single clean forward + backward. Cache:
     - q, k at all downstream layers (rotated AND unrotated)
     - clean attention scores and patterns
     - ∇M w.r.t. attention pattern (per layer, per head)
     - ∇M w.r.t. v_input (per layer, per head, last-position)
  2. For each (L, h):
     - Compute Δq = (-f_clean * decoder_col_X) @ W_Q[L, h]  (closed form)
     - Compute Δk similarly
     - Apply rotary at position 'last' via model.blocks[L].attn.apply_rotary
       (which is a deterministic function of position)
     - Compute Δscores[last, :] in closed form using clean rot_k
     - patched_scores[last, :] = clean_scores[L, h, last, :] + Δscores[last, :]
     - patched_pattern[last, :] = softmax(patched_scores)
     - Δpattern[last, :] = patched_pattern[last, :] - clean_pattern[L, h, last, :]
     - effect_qk = -∇M_pattern[L, h, last, :] · Δpattern[last, :]
     - effect_v  = ∇M_v_input[L, last, h, :] · (f_clean * decoder_col_X)
     - logp_drop_pred = effect_qk + effect_v

Cost: 1 forward + 1 backward + per-(L, h) closed-form math = 2 model passes.

This is the principled efficient implementation that the diagnosis (Finding 17)
suggested would work.
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

sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--tokens", type=str, default="data/token_stream.npy")
    p.add_argument("--context_window", type=int, default=30)
    p.add_argument("--feature_ids", type=str, default="10047,13131,15245")
    p.add_argument("--n_positions", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_top_firing_positions(sae, feature_id, acts, device, top_k=10,
                               sample_size=100000, batch_size=8192):
    n_total = acts.shape[0]
    if n_total > sample_size:
        sample_idx = np.random.choice(n_total, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n_total)

    feature_activations = np.zeros(len(sample_idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sample_idx), batch_size):
            batch = sample_idx[start:start + batch_size]
            batch_acts = torch.from_numpy(acts[batch].astype(np.float32)).to(device)
            f = sae.encode(batch_acts)
            feature_activations[start:start + len(batch)] = f[:, feature_id].cpu().numpy()
            del batch_acts, f
            if device == "mps":
                torch.mps.empty_cache()

    top_local_idx = np.argsort(-feature_activations)[:top_k]
    return [(int(sample_idx[lidx]), float(feature_activations[lidx]))
            for lidx in top_local_idx]


def apply_rotary_at_position(attn_module, vec_d_head, position, n_heads, d_head,
                              head_idx, seq_len, device):
    """Apply attn_module.apply_rotary to a vector at a specific (position, head).
    Constructs a [1, seq_len, n_heads, d_head] tensor with vec at (0, position, head_idx, :),
    zeros elsewhere, then calls apply_rotary and extracts the result at that slot.
    """
    x = torch.zeros(1, seq_len, n_heads, d_head, device=device, dtype=vec_d_head.dtype)
    x[0, position, head_idx, :] = vec_d_head
    rotated = attn_module.apply_rotary(x)
    return rotated[0, position, head_idx, :]


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    sae = TopKSAE(d_model, ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"loading {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    model.set_use_split_qkv_input(True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    downstream_layers = list(range(layer + 1, n_layers))
    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, "
          f"downstream={downstream_layers}")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    # Hooks for caching
    rot_k_hooks = [f"blocks.{L}.attn.hook_rot_k" for L in downstream_layers]
    rot_q_hooks = [f"blocks.{L}.attn.hook_rot_q" for L in downstream_layers]
    scores_hooks = [f"blocks.{L}.attn.hook_attn_scores" for L in downstream_layers]
    pattern_hooks = [f"blocks.{L}.attn.hook_pattern" for L in downstream_layers]
    v_input_hooks = [f"blocks.{L}.hook_v_input" for L in downstream_layers]
    # hook_q_input is PRE-LN (same as resid_pre). We need it to recompute LN(perturbed) - LN(clean)
    q_input_hooks = [f"blocks.{L}.hook_q_input" for L in downstream_layers]

    cache_hooks = rot_k_hooks + rot_q_hooks + scores_hooks + pattern_hooks + v_input_hooks + q_input_hooks

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)
        per_position_atpstar = []
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_atpstar) >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )
            seq_len = cand_tokens_tensor.shape[1]
            last_pos = seq_len - 1

            # ===== Clean forward + backward with grad-tracking hooks =====
            saved = {}

            def make_capture_hook(name):
                def hook(activation, hook):
                    activation.retain_grad()
                    saved[name] = activation
                    return activation
                return hook

            hooks_list = [(n, make_capture_hook(n)) for n in cache_hooks]

            with model.hooks(fwd_hooks=hooks_list):
                clean_logits = model(cand_tokens_tensor)

            with torch.no_grad():
                _, c2 = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
                clean_resid = c2[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            metric = torch.log_softmax(clean_logits[0, -1, :], dim=-1)[actual_next_token]
            model.zero_grad()
            metric.backward()

            # Snapshot needed activations and gradients
            clean_rot_k = {L: saved[f"blocks.{L}.attn.hook_rot_k"].detach().clone()
                           for L in downstream_layers}  # [b, seq, head, d_head]
            clean_rot_q = {L: saved[f"blocks.{L}.attn.hook_rot_q"].detach().clone()
                           for L in downstream_layers}
            clean_scores = {L: saved[f"blocks.{L}.attn.hook_attn_scores"].detach().clone()
                            for L in downstream_layers}  # [b, head, sq, sk]
            clean_patterns = {L: saved[f"blocks.{L}.attn.hook_pattern"].detach().clone()
                              for L in downstream_layers}
            clean_q_inputs = {L: saved[f"blocks.{L}.hook_q_input"].detach().clone()
                              for L in downstream_layers}  # PRE-LN [b, seq, head, d_model]
            pattern_grads = {L: saved[f"blocks.{L}.attn.hook_pattern"].grad.detach().clone()
                             if saved[f"blocks.{L}.attn.hook_pattern"].grad is not None else None
                             for L in downstream_layers}
            v_grads = {L: saved[f"blocks.{L}.hook_v_input"].grad.detach().clone()
                       if saved[f"blocks.{L}.hook_v_input"].grad is not None else None
                       for L in downstream_layers}
            saved.clear()

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, baseline logP={baseline_logp:+.3f}")

            # ===== Closed-form patched-pattern computation per (L, h) =====
            delta = (f_clean * decoder_col_X).to(torch.float32)  # positive delta
            sqrt_d_head = math.sqrt(d_head)

            atpstar_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)

            for li, L in enumerate(downstream_layers):
                attn_module = model.blocks[L].attn
                ln1_module = model.blocks[L].ln1
                W_Q = attn_module.W_Q  # [head, d_model, d_head]
                W_K = attn_module.W_K
                # The actual computation is q = ln1(q_input) @ W_Q + b_Q
                # hook_q_input is PRE-LN. So Δq involves ΔLN(q_input).
                # Compute ΔLN = ln1(clean - f*decoder) - ln1(clean) (per head, position)
                clean_q_input_lh = clean_q_inputs[L][0, last_pos, :, :].to(torch.float32)  # [head, d_model]
                pert_q_input_lh = clean_q_input_lh - f_clean * decoder_col_X[None, :]  # broadcast across heads
                # Run LN on both. ln1 expects shape [..., d_model]; works on broadcasting.
                with torch.no_grad():
                    clean_ln = ln1_module(clean_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
                    pert_ln = ln1_module(pert_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
                delta_ln = pert_ln - clean_ln  # [head, d_model]

                for h in range(n_heads):
                    # Δq = ΔLN[h] @ W_Q[h]
                    dq = delta_ln[h] @ W_Q[h].to(torch.float32)  # [d_head]
                    dk = delta_ln[h] @ W_K[h].to(torch.float32)  # [d_head]

                    # Rotary at position 'last' for head h
                    drot_q = apply_rotary_at_position(
                        attn_module, dq, last_pos, n_heads, d_head, h, seq_len, device
                    ).to(torch.float32)
                    drot_k = apply_rotary_at_position(
                        attn_module, dk, last_pos, n_heads, d_head, h, seq_len, device
                    ).to(torch.float32)

                    # Δscores[last, :] for this head
                    clean_rot_k_h = clean_rot_k[L][0, :, h, :].to(torch.float32)  # [seq, d_head]
                    clean_rot_q_h_last = clean_rot_q[L][0, last_pos, h, :].to(torch.float32)
                    # row contribution from delta_q (affects all key positions j)
                    delta_scores_row = (drot_q @ clean_rot_k_h.T) / sqrt_d_head  # [seq_k]
                    # at j=last, additional contribution from delta_k (and cross term)
                    delta_scores_row[last_pos] = delta_scores_row[last_pos] + (
                        clean_rot_q_h_last @ drot_k + drot_q @ drot_k
                    ) / sqrt_d_head

                    # patched_scores[last] = clean_scores[last] + delta_scores
                    clean_scores_row = clean_scores[L][0, h, last_pos, :].to(torch.float32)
                    patched_scores_row = clean_scores_row + delta_scores_row

                    # Causal mask: positions > last_pos masked; since last_pos = seq_len - 1
                    # all positions are valid for the last query. No masking needed.

                    # Softmax to get patched_pattern[last, :]
                    patched_pattern_row = torch.softmax(patched_scores_row, dim=-1)
                    clean_pattern_row = clean_patterns[L][0, h, last_pos, :].to(torch.float32)
                    delta_pattern_row = patched_pattern_row - clean_pattern_row

                    # effect_qk = -g_pattern · delta_pattern  (negated for logp_drop)
                    g_pattern_last = pattern_grads[L][0, h, last_pos, :].to(torch.float32) \
                        if pattern_grads[L] is not None else None
                    effect_qk = -float(torch.dot(g_pattern_last, delta_pattern_row).item()) \
                        if g_pattern_last is not None else 0.0

                    # V-side: standard AtP
                    g_v_last = v_grads[L][0, last_pos, h, :].to(torch.float32) \
                        if v_grads[L] is not None else None
                    effect_v = float(torch.dot(g_v_last, delta).item()) if g_v_last is not None else 0.0

                    atpstar_mediation[li, h] = effect_qk + effect_v

            per_position_atpstar.append(atpstar_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })

            top_idx = np.argsort(-atpstar_mediation.flatten())[:5]
            print(f"    top efficient AtP* mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}="
                      f"{atpstar_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))

        if not per_position_atpstar:
            continue

        mean_atpstar = np.mean(np.stack(per_position_atpstar), axis=0)
        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_atpstar),
            "per_position_meta": per_position_meta,
            "mean_atpstar_efficient_mediation": mean_atpstar.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_atpstar_efficient.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "atp_star_efficient_closed_form",
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
