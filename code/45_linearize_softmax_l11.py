"""Localize the AtP failure at L11 by linearizing the attention SOFTMAX.

Companion to script 41 (which tested GELU and found it's not the AtP killer).
With GELU ruled out, softmax in attention is the leading suspect for the
nonlinearity that breaks AtP at deep layers.

Method:
  1. Run a clean forward. Cache:
     - blocks.11.attn.hook_attn_scores  (pre-softmax, [batch, head, seq_q, seq_k])
     - blocks.11.attn.hook_pattern      (post-softmax)
  2. For each (feature, head, position): run a perturbed forward with
     per-head q/k/v_input ablation. Install hooks that:
     - capture the current (perturbed) attn_scores
     - replace the pattern with linearized softmax:
         pattern_lin = clean_pattern + Jacobian(softmax)|clean · (cur_scores - clean_scores)
       where Jacobian is the standard softmax Jacobian:
         delta_p[i] = p[i] * (delta_s[i] - <p, delta_s>)
  3. Measure logp drop. Call this AP_softmax_lin.

Interpretation (same as script 41):
  - If AP_softmax_lin ≈ AtP: softmax IS the AtP killer at L11. Build AtP*
    or softmax-corrected AtP next.
  - If AP_softmax_lin ≈ AP: softmax isn't the killer either. The nonlinearity
    must be propagated from L7-L10 — there's no single component to fix.
  - In-between: softmax is partially responsible.

Notes:
  - Causal mask handled by clipping -inf scores to -1e9 before subtraction.
    Since clean_pattern is 0 at masked positions, the correction is zero
    there anyway.
  - We linearize at L11 only; other layers run as normal.
"""
import argparse
import json
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
    p.add_argument("--target_layer", type=int, default=11)
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
    target_layer = args.target_layer
    print(f"  n_layers={n_layers}, n_heads={n_heads}, target_layer={target_layer} "
          f"(softmax linearized)")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    scores_hook_name = f"blocks.{target_layer}.attn.hook_attn_scores"
    pattern_hook_name = f"blocks.{target_layer}.attn.hook_pattern"

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)
        per_position_lin = []
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_lin) >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )

            # Clean forward: cache attn_scores and pattern at target layer
            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(
                    cand_tokens_tensor,
                    names_filter=[hook_name, scores_hook_name, pattern_hook_name]
                )
                clean_resid = clean_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            clean_scores = clean_cache[scores_hook_name].detach()  # [b, h, sq, sk]
            clean_pattern = clean_cache[pattern_hook_name].detach()  # same shape

            # Clip -inf for masked positions (those will be zeroed by clean_pattern=0)
            clean_scores_safe = torch.where(
                torch.isinf(clean_scores),
                torch.full_like(clean_scores, -1e9),
                clean_scores
            )

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, baseline logP={baseline_logp:+.3f}")
            print(f"    clean_scores shape: {tuple(clean_scores.shape)}, "
                  f"max pattern entropy = {(-clean_pattern * torch.log(clean_pattern.clamp_min(1e-12))).sum(-1).max().item():.3f}")

            # Per-head mediation under softmax linearization
            mediation_lin = np.zeros(n_heads, dtype=np.float32)

            for h in range(n_heads):
                state = {"current_scores": None}

                def capture_scores_hook(activation, hook):
                    state["current_scores"] = activation
                    return activation

                def linearize_pattern_hook(activation, hook):
                    cur_s = state["current_scores"]
                    if cur_s is None:
                        return activation
                    cur_s_safe = torch.where(
                        torch.isinf(cur_s),
                        torch.full_like(cur_s, -1e9),
                        cur_s
                    )
                    delta_s = cur_s_safe - clean_scores_safe
                    # softmax Jacobian: delta_p[i] = p[i] * (delta_s[i] - <p, delta_s>)
                    weighted = (clean_pattern * delta_s).sum(dim=-1, keepdim=True)
                    delta_p = clean_pattern * (delta_s - weighted)
                    return clean_pattern + delta_p

                def make_qkv_hook(head_idx):
                    def hook_fn(activation, hook):
                        activation[:, -1, head_idx, :] = (
                            activation[:, -1, head_idx, :] - f_clean * decoder_col_X
                        )
                        return activation
                    return hook_fn

                qkv_hook = make_qkv_hook(h)
                state["current_scores"] = None

                fwd_hooks = [
                    (f"blocks.{target_layer}.hook_q_input", qkv_hook),
                    (f"blocks.{target_layer}.hook_k_input", qkv_hook),
                    (f"blocks.{target_layer}.hook_v_input", qkv_hook),
                    (scores_hook_name, capture_scores_hook),
                    (pattern_hook_name, linearize_pattern_hook),
                ]
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        cand_tokens_tensor, fwd_hooks=fwd_hooks
                    )
                    patched_logprobs = torch.log_softmax(
                        patched_logits[0, -1, :], dim=-1
                    )
                    patched_logp = patched_logprobs[actual_next_token].item()
                mediation_lin[h] = baseline_logp - patched_logp

            per_position_lin.append(mediation_lin)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
                "baseline_logp": baseline_logp,
            })

            top_idx = np.argsort(-np.abs(mediation_lin))[:5]
            print(f"    top |mediation_lin|: "
                  + ", ".join(f"L{target_layer}H{i}={mediation_lin[i]:+.4f}" for i in top_idx))

        if not per_position_lin:
            print(f"  WARN: no positions; skipping")
            continue

        mean_lin = np.mean(np.stack(per_position_lin), axis=0)
        all_results[feature_id] = {
            "target_layer": target_layer,
            "n_heads": n_heads,
            "n_positions": len(per_position_lin),
            "per_position_meta": per_position_meta,
            "mean_mediation_softmax_linearized": mean_lin.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(
        f".sae_feature_softmax_lin_L{target_layer}.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "ap_with_softmax_linearized",
        "target_layer": target_layer,
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
