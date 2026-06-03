"""Localize the AtP failure at L11 by linearizing the MLP GELU.

Hypothesis (from script 37's alpha-scaling result): AtP fails at L11
because the model's response to the perturbation is nonlinear by that
depth. The question this script answers: is the responsible nonlinearity
the L11 MLP GELU, or is it elsewhere (softmax, LayerNorm)?

Method:
  1. Run a clean forward pass. Cache `blocks.11.mlp.hook_pre`
     (the pre-GELU activation at L11) and compute the GELU Jacobian
     at this clean point.
  2. For each (feature, head, position): run a perturbed forward pass
     (per-head q/k/v_input ablation as in script 30). But install a
     hook at `blocks.11.mlp.hook_post` that REPLACES the actual post-
     activation with its first-order Taylor approximation around the
     clean pre-activation:
         post_linearized = GELU(clean_pre) + GELU'(clean_pre) * (cur_pre - clean_pre)
  3. Measure the logp drop and call this AP_gelu_lin.

Interpretation:
  - If AP_gelu_lin ≈ AtP (drops the deep-layer effect by 14x): GELU is
    the AtP killer. Use AtP* / N=2 IG / GELU correction.
  - If AP_gelu_lin ≈ AP (close to ground truth, large drop): GELU is
    not the problem. Look elsewhere (softmax or LayerNorm).
  - In-between: GELU is *partially* responsible; we'd need to compose
    linearizations to fully localize.

We focus on L11 only (the failure regime); L7-10 have AtP near 1.0 so
no localization needed there.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    p.add_argument("--target_layer", type=int, default=11,
                   help="Downstream layer to linearize GELU at.")
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


def compute_gelu_jacobian(clean_pre):
    """Returns GELU'(clean_pre) elementwise, same shape as clean_pre."""
    x = clean_pre.detach().clone().requires_grad_(True)
    y = F.gelu(x)
    grad = torch.autograd.grad(y.sum(), x)[0]
    return grad.detach()


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
    print(f"  n_layers={n_layers}, n_heads={n_heads}, "
          f"target_layer={target_layer} (GELU linearized)")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    pre_hook_name = f"blocks.{target_layer}.mlp.hook_pre"
    post_hook_name = f"blocks.{target_layer}.mlp.hook_post"

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)

        per_position_lin = []  # (n_heads,) arrays for target_layer
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_lin) >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )

            # Clean forward: cache clean pre-activation at target layer and SAE hook
            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(
                    cand_tokens_tensor,
                    names_filter=[hook_name, pre_hook_name]
                )
                clean_resid = clean_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            clean_pre = clean_cache[pre_hook_name].detach()  # [batch, seq, d_mlp]
            clean_post = F.gelu(clean_pre)  # what GELU should output at clean
            gelu_jac = compute_gelu_jacobian(clean_pre)  # GELU'(clean_pre) elementwise

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, baseline logP={baseline_logp:+.3f}")
            print(f"    clean_pre shape: {tuple(clean_pre.shape)}, "
                  f"mean GELU' = {gelu_jac.mean().item():.3f}")

            # State container for capturing perturbed pre-activation
            state = {"perturbed_pre": None}

            def capture_pre_hook(activation, hook):
                state["perturbed_pre"] = activation
                return activation

            def linearize_post_hook(activation, hook):
                # Replace actual post (GELU output) with linear approximation
                # around the clean point
                cur_pre = state["perturbed_pre"]
                if cur_pre is None:
                    return activation
                linearized = clean_post + gelu_jac * (cur_pre - clean_pre)
                return linearized

            mediation_lin = np.zeros(n_heads, dtype=np.float32)

            for h in range(n_heads):
                def make_qkv_hook(head_idx):
                    def hook_fn(activation, hook):
                        activation[:, -1, head_idx, :] = (
                            activation[:, -1, head_idx, :] - f_clean * decoder_col_X
                        )
                        return activation
                    return hook_fn

                qkv_hook = make_qkv_hook(h)

                # Reset state so each (h) run starts fresh
                state["perturbed_pre"] = None

                fwd_hooks = [
                    (f"blocks.{target_layer}.hook_q_input", qkv_hook),
                    (f"blocks.{target_layer}.hook_k_input", qkv_hook),
                    (f"blocks.{target_layer}.hook_v_input", qkv_hook),
                    (pre_hook_name, capture_pre_hook),
                    (post_hook_name, linearize_post_hook),
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
            "mean_mediation_gelu_linearized": mean_lin.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(
        f".sae_feature_gelu_lin_L{target_layer}.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "ap_with_gelu_linearized",
        "target_layer": target_layer,
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
