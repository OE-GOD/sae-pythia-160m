"""AtP* (softmax-corrected AtP) — validation implementation.

Motivated by Finding 17 (script 45): linearizing the L11 attention softmax
makes AP collapse to AtP (Pearson 0.9963). So softmax curvature is what
AtP misses.

Fix: instead of using softmax_jacobian(clean_scores) to convert pattern-
gradients to score-gradients (what plain AtP does), compute the actual
patched pattern via the full softmax(patched_scores), and use this
nonlinear quantity in the chain rule.

AtP* estimate per (L, h):
  Effect_QK = ∇M_pattern[L, h, last, :] · (patched_pattern - clean_pattern)[L, h, last, :]
  Effect_V  = ∇M_v[L, h, last, :] · Δv   (standard AtP V-side)
  AtP*[L, h] = Effect_QK + Effect_V

This implementation runs an EXTRA FORWARD per (L, h) intervention to capture
patched_pattern. Cost ≈ 62 passes per feature-position — same as full AP.
This is a VALIDATION of whether softmax correction closes the L11 gap;
an efficient closed-form version (computing patched_pattern in closed form
via rotary + score arithmetic) would bring cost back to ~2 passes.
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
    downstream_layers = list(range(layer + 1, n_layers))
    print(f"  n_layers={n_layers}, n_heads={n_heads}, "
          f"downstream_layers={downstream_layers}")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    pattern_hooks = [f"blocks.{L}.attn.hook_pattern" for L in downstream_layers]
    v_hooks = [f"blocks.{L}.hook_v_input" for L in downstream_layers]

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

            # ===== Clean forward + backward to get gradients at clean state =====
            saved_clean = {}

            def make_capture_hook(name_full):
                def hook(activation, hook):
                    activation.retain_grad()
                    saved_clean[name_full] = activation
                    return activation
                return hook

            clean_hooks = [(name, make_capture_hook(name))
                           for name in pattern_hooks + v_hooks]

            with model.hooks(fwd_hooks=clean_hooks):
                clean_logits = model(cand_tokens_tensor)

            with torch.no_grad():
                clean_resid_cache = {}
                # SAE check
                _, c_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
                clean_resid = c_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()
            metric = torch.log_softmax(clean_logits[0, -1, :], dim=-1)[actual_next_token]
            model.zero_grad()
            metric.backward()

            # Snapshot clean patterns and gradients
            clean_patterns = {}
            pattern_grads = {}
            v_grads = {}
            for L in downstream_layers:
                pn = f"blocks.{L}.attn.hook_pattern"
                vn = f"blocks.{L}.hook_v_input"
                clean_patterns[L] = saved_clean[pn].detach().clone()
                pattern_grads[L] = saved_clean[pn].grad.detach().clone() if saved_clean[pn].grad is not None else None
                v_grads[L] = saved_clean[vn].grad.detach().clone() if saved_clean[vn].grad is not None else None
            saved_clean.clear()

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, baseline logP={baseline_logp:+.3f}")

            # ===== Per-(L, h) perturbed forwards to get patched_pattern =====
            delta = (f_clean * decoder_col_X).to(torch.float32)
            atpstar_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)

            for li, L in enumerate(downstream_layers):
                pn = f"blocks.{L}.attn.hook_pattern"
                for h in range(n_heads):
                    def make_qkv_hook(head_idx):
                        def hook_fn(activation, hook):
                            activation[:, -1, head_idx, :] = (
                                activation[:, -1, head_idx, :] - f_clean * decoder_col_X
                            )
                            return activation
                        return hook_fn

                    captured = {"patched_pattern": None}

                    def capture_pattern_hook(activation, hook):
                        captured["patched_pattern"] = activation.detach().clone()
                        return activation

                    qkv_hook = make_qkv_hook(h)
                    fwd_hooks = [
                        (f"blocks.{L}.hook_q_input", qkv_hook),
                        (f"blocks.{L}.hook_k_input", qkv_hook),
                        (f"blocks.{L}.hook_v_input", qkv_hook),
                        (pn, capture_pattern_hook),
                    ]
                    with torch.no_grad():
                        model.run_with_hooks(
                            cand_tokens_tensor, fwd_hooks=fwd_hooks
                        )

                    patched_pattern = captured["patched_pattern"]  # [b, head, seq_q, seq_k]
                    # Last-position row, this head
                    pp_last = patched_pattern[0, h, -1, :].to(torch.float32)
                    cp_last = clean_patterns[L][0, h, -1, :].to(torch.float32)
                    delta_pattern = pp_last - cp_last

                    # Effect via Q+K (softmax-aware)
                    g_pattern_last = pattern_grads[L][0, h, -1, :].to(torch.float32) if pattern_grads[L] is not None else None
                    effect_qk = float(torch.dot(g_pattern_last, delta_pattern).item()) if g_pattern_last is not None else 0.0

                    # Effect via V (standard AtP)
                    # g_v · delta directly equals the logp_drop contribution from V
                    # (sign convention matches script 32)
                    g_v_last = v_grads[L][0, -1, h, :].to(torch.float32) if v_grads[L] is not None else None
                    logp_drop_v = float(torch.dot(g_v_last, delta).item()) if g_v_last is not None else 0.0

                    # logp_drop from softmax-aware Q+K:
                    # ΔM_pattern = g_pattern · Δpattern; logp_drop = -ΔM
                    logp_drop_qk = -effect_qk

                    atpstar_mediation[li, h] = logp_drop_qk + logp_drop_v

            per_position_atpstar.append(atpstar_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })

            top_idx = np.argsort(-atpstar_mediation.flatten())[:5]
            print(f"    top AtP* mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}="
                      f"{atpstar_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))
            print(f"    elapsed so far: {time.time() - t0:.0f}s")

        if not per_position_atpstar:
            continue

        mean_atpstar = np.mean(np.stack(per_position_atpstar), axis=0)
        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_atpstar),
            "per_position_meta": per_position_meta,
            "mean_atpstar_mediation": mean_atpstar.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_atpstar.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "atp_star_softmax_corrected",
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
