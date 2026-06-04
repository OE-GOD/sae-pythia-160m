"""GradDrop on top of efficient AtP* — addresses the cancellation failure mode
from Kramár et al. 2024.

Standard efficient AtP* (script 49):
  ΔM ≈ ∇M_pattern[L, h] · Δpattern[L, h] + ∇M_v[L, h] · Δv

This fails when direct + indirect effects cancel. Example: a feature affects
head X positively (direct) and head Y negatively (indirect via residual
stream cascade); standard AtP estimates the *sum*, which can be ≈ 0 if the
two cancel — masking the real magnitudes.

GradDrop fix:
  For each layer ℓ ∈ downstream_layers:
    Detach blocks[ℓ].hook_resid_post during forward (block gradient flow
    backward through that layer's residual contribution).
    Run forward + backward with this modification.
    Compute AtP* using these modified gradients (call result AtP*_drop_ℓ).
  Final estimate: ĉ_GD(feature, head) = (1/K) · Σ_ℓ |AtP*_drop_ℓ(feature, head)|

By averaging *absolute values*, cancellation can no longer hide magnitude:
even if direct and indirect cancel in the no-drop estimate, one drop will
isolate one and the absolute value preserves its magnitude.

Cost per feature-position: 2 (standard AtP*) + 2K (one fwd+bwd per dropped
layer), where K is the number of downstream layers. For Pythia 160M with
5 downstream layers: 2 + 10 = 12 model passes.

Tests whether the remaining 0.007 Pearson gap between efficient AtP* and
full AP comes from cancellation. If GradDrop closes it, we've fully
reproduced Kramár AtP*. If not, the remaining gap is something else
(probably second-order pattern effects).
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
    p.add_argument("--drop_set", type=str, default="all_downstream",
                    choices=["all_downstream", "deepest_only"])
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


def compute_atpstar_estimate(model, tokens, hook_name, sae, fid, decoder_col_X,
                              actual_next_token, downstream_layers, drop_layer=None):
    """Run forward + backward (with optional GradDrop), compute efficient AtP*.
    If drop_layer is set, detach blocks[drop_layer].hook_resid_post during forward.
    Returns: atpstar_mediation [n_downstream, n_heads]
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    seq_len = tokens.shape[1]
    last_pos = seq_len - 1
    sqrt_d_head = math.sqrt(d_head)

    rot_k_hooks = [f"blocks.{L}.attn.hook_rot_k" for L in downstream_layers]
    rot_q_hooks = [f"blocks.{L}.attn.hook_rot_q" for L in downstream_layers]
    scores_hooks = [f"blocks.{L}.attn.hook_attn_scores" for L in downstream_layers]
    pattern_hooks = [f"blocks.{L}.attn.hook_pattern" for L in downstream_layers]
    v_input_hooks = [f"blocks.{L}.hook_v_input" for L in downstream_layers]
    q_input_hooks = [f"blocks.{L}.hook_q_input" for L in downstream_layers]

    saved = {}

    def make_capture(name):
        def hook(activation, hook):
            activation.retain_grad()
            saved[name] = activation
            return activation
        return hook

    def make_zero_grad_backward(activation, hook):
        # Forward: pass through unchanged so downstream still tracks gradients.
        # Backward: when gradient flows back to this node, return zeros so it
        # cannot propagate further upstream. This is the "do(n_out_ℓ ← clean)"
        # intervention from Kramár et al. 2024 — block gradient through layer ℓ.
        if activation.requires_grad:
            activation.register_hook(lambda grad: torch.zeros_like(grad))
        return activation

    cache_hooks = (rot_k_hooks + rot_q_hooks + scores_hooks + pattern_hooks +
                    v_input_hooks + q_input_hooks)
    fwd_hooks = [(n, make_capture(n)) for n in cache_hooks]
    if drop_layer is not None:
        fwd_hooks.append((f"blocks.{drop_layer}.hook_resid_post", make_zero_grad_backward))

    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)

    metric = torch.log_softmax(logits[0, -1, :], dim=-1)[actual_next_token]
    model.zero_grad()
    metric.backward()

    # Snapshot
    delta = (sae.encode(model.run_with_cache(tokens, names_filter=[hook_name])[1][hook_name][:, -1, :])[0, fid].item() *
             decoder_col_X).to(torch.float32)

    atpstar = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)

    for li, L in enumerate(downstream_layers):
        attn = model.blocks[L].attn
        ln1 = model.blocks[L].ln1
        W_Q = attn.W_Q
        W_K = attn.W_K  # noqa: F841
        clean_q_input_lh = saved[f"blocks.{L}.hook_q_input"][0, last_pos, :, :].detach().to(torch.float32)
        f_clean = (delta.norm() / decoder_col_X.to(torch.float32).norm()).item()  # recover f_clean
        pert_q_input_lh = clean_q_input_lh - delta[None, :]
        with torch.no_grad():
            clean_ln = ln1(clean_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
            pert_ln = ln1(pert_q_input_lh.to(model.cfg.dtype)).to(torch.float32)
        delta_ln = pert_ln - clean_ln
        clean_rot_k = saved[f"blocks.{L}.attn.hook_rot_k"].detach()
        clean_rot_q = saved[f"blocks.{L}.attn.hook_rot_q"].detach()
        clean_scores = saved[f"blocks.{L}.attn.hook_attn_scores"].detach()
        clean_pattern = saved[f"blocks.{L}.attn.hook_pattern"].detach()
        pattern_grad = saved[f"blocks.{L}.attn.hook_pattern"].grad
        v_grad = saved[f"blocks.{L}.hook_v_input"].grad

        for h in range(n_heads):
            dq = delta_ln[h] @ W_Q[h].to(torch.float32)
            dk = delta_ln[h] @ W_K[h].to(torch.float32) if W_K.shape[0] > h else delta_ln[h] @ W_K[h % W_K.shape[0]].to(torch.float32)
            x = torch.zeros(1, seq_len, n_heads, d_head, device=tokens.device, dtype=dq.dtype)
            x[0, last_pos, h, :] = dq
            with torch.no_grad():
                drot_q = attn.apply_rotary(x.to(model.cfg.dtype))[0, last_pos, h, :].to(torch.float32)
                xk = torch.zeros_like(x)
                xk[0, last_pos, h, :] = dk
                drot_k = attn.apply_rotary(xk.to(model.cfg.dtype))[0, last_pos, h, :].to(torch.float32)
            clean_rot_k_h = clean_rot_k[0, :, h, :].to(torch.float32)
            clean_rot_q_h_last = clean_rot_q[0, last_pos, h, :].to(torch.float32)
            delta_scores_row = (drot_q @ clean_rot_k_h.T) / sqrt_d_head
            delta_scores_row[last_pos] = delta_scores_row[last_pos] + (
                clean_rot_q_h_last @ drot_k + drot_q @ drot_k
            ) / sqrt_d_head
            clean_scores_row = clean_scores[0, h, last_pos, :].to(torch.float32)
            patched_pattern_row = torch.softmax(clean_scores_row + delta_scores_row, dim=-1)
            clean_pattern_row = clean_pattern[0, h, last_pos, :].to(torch.float32)
            delta_pattern_row = patched_pattern_row - clean_pattern_row

            g_pat = pattern_grad[0, h, last_pos, :].to(torch.float32) if pattern_grad is not None else None
            effect_qk = -float(torch.dot(g_pat, delta_pattern_row).item()) if g_pat is not None else 0.0
            g_v = v_grad[0, last_pos, h, :].to(torch.float32) if v_grad is not None else None
            effect_v = float(torch.dot(g_v, delta).item()) if g_v is not None else 0.0
            atpstar[li, h] = effect_qk + effect_v

    saved.clear()
    return atpstar


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sae = TopKSAE(ckpt["d_model"], ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    model.set_use_split_qkv_input(True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    downstream_layers = list(range(layer + 1, n_layers))
    if args.drop_set == "all_downstream":
        drop_layers = downstream_layers[:]
    else:  # deepest_only
        drop_layers = [downstream_layers[-1]]
    print(f"  downstream: {downstream_layers}, drop_set ({args.drop_set}): {drop_layers}")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)
        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)

        per_position_estimates = []  # list of dicts {label: ndarray}
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_estimates) >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )

            with torch.no_grad():
                _, c = model.run_with_cache(cand_tokens_tensor, names_filter=[hook_name])
                clean_resid = c[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, target={actual_next_token}")

            # Standard AtP* (no drop)
            est_nodrop = compute_atpstar_estimate(
                model, cand_tokens_tensor, hook_name, sae, feature_id,
                decoder_col_X, actual_next_token, downstream_layers, drop_layer=None
            )

            # AtP* with each drop
            drop_estimates = {}
            for dl in drop_layers:
                est = compute_atpstar_estimate(
                    model, cand_tokens_tensor, hook_name, sae, feature_id,
                    decoder_col_X, actual_next_token, downstream_layers, drop_layer=dl
                )
                drop_estimates[dl] = est

            # GradDrop estimate: mean(|.|) across drops
            if drop_estimates:
                stack = np.stack([np.abs(v) for v in drop_estimates.values()])
                est_graddrop = stack.mean(axis=0)
            else:
                est_graddrop = est_nodrop

            per_position_estimates.append({
                "nodrop": est_nodrop,
                "graddrop": est_graddrop,
                "drop_per_layer": {dl: drop_estimates[dl] for dl in drop_estimates},
            })
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
            })
            print(f"    elapsed: {time.time() - t0:.0f}s")

        if not per_position_estimates:
            continue

        mean_nodrop = np.mean(np.stack([e["nodrop"] for e in per_position_estimates]), axis=0)
        mean_graddrop = np.mean(np.stack([e["graddrop"] for e in per_position_estimates]), axis=0)
        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "drop_layers": drop_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_estimates),
            "per_position_meta": per_position_meta,
            "mean_atpstar_nodrop": mean_nodrop.tolist(),
            "mean_atpstar_graddrop": mean_graddrop.tolist(),
        }

    print(f"\ncompleted in {time.time() - t0:.1f}s")
    out_path = args.out or "checkpoints/sae_layer6_topk64_full.sae_feature_graddrop.json"
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "efficient_atpstar_with_graddrop",
        "drop_set": args.drop_set,
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
