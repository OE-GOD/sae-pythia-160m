"""Option 2 — integrated gradients (IG) for SAE feature mediation.

Same setup as script 32 (AtP), but averages the gradient over N points along
the path from `a_clean` to `a_patch` instead of taking it at just `a_clean`.

For each alpha in {0, 1/N, 2/N, ..., (N-1)/N}:
  1. Add a hook at the SAE layer that subtracts alpha * f_clean * decoder_col_X
     from the residual at the firing position. This makes the model see
     "feature partially ablated."
  2. Forward + backward pass.
  3. Cache per-head q/k/v_input gradients.

Average the gradients across alphas. IG estimate per (L, h):
  IG_mediation[L, h] = sum_qkv of (avg_grad[L, h] · (f_clean * decoder_col_X))

Cost: N forward + N backward passes per firing position (vs 1 for AtP).
For N=10: 10x AtP cost, still much cheaper than activation patching at scale.

Comparison: this script produces IG estimates that can be compared to AP
(ground truth) and AtP (1-point baseline). The question is whether IG
fixes the deep-layer failure mode AtP has.
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
    p.add_argument("--n_alphas", type=int, default=10,
                    help="Number of alpha steps for IG (more = more accurate, more expensive)")
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
    out = []
    for lidx in top_local_idx:
        out.append((int(sample_idx[lidx]), float(feature_activations[lidx])))
    return out


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
    print(f"  n_layers={n_layers}, downstream_layers={downstream_layers}")
    print(f"  n_alphas (IG steps): {args.n_alphas}")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    qkv_hooks = []
    for L in downstream_layers:
        for kind in ["q_input", "k_input", "v_input"]:
            qkv_hooks.append(f"blocks.{L}.hook_{kind}")

    all_results = {}
    t0 = time.time()

    # IG alpha grid: equally spaced in [0, 1)
    # alpha=0 corresponds to the clean activation
    # alpha=1 would correspond to the fully ablated activation
    alphas = [(i + 0.5) / args.n_alphas for i in range(args.n_alphas)]

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)
        per_position_ig = []
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_ig) >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )

            with torch.no_grad():
                _, c_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
                clean_resid = c_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            delta = (f_clean * decoder_col_X).to(torch.float32)

            # IG: average gradients over alpha values
            accumulated_grads = {name: None for name in qkv_hooks}

            for alpha in alphas:
                # Hook at SAE layer: subtract alpha * f_clean * decoder_col at last pos
                def sae_ablation_hook(activation, hook, alpha_val=alpha):
                    activation[:, -1, :] = (
                        activation[:, -1, :] - alpha_val * f_clean * decoder_col_X
                    )
                    return activation

                # Save hooks
                saved = {}
                def make_save_hook(hook_name_full):
                    def hook(activation, hook):
                        activation.retain_grad()
                        saved[hook_name_full] = activation
                        return activation
                    return hook

                fwd_hooks = [(hook_name, sae_ablation_hook)] + \
                    [(name, make_save_hook(name)) for name in qkv_hooks]

                with model.hooks(fwd_hooks=fwd_hooks):
                    logits = model(cand_tokens_tensor)

                logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)
                metric = logprobs[actual_next_token]
                model.zero_grad()
                metric.backward()

                for name in qkv_hooks:
                    g = saved[name].grad
                    if g is None:
                        continue
                    g_detached = g.detach().clone()
                    if accumulated_grads[name] is None:
                        accumulated_grads[name] = g_detached
                    else:
                        accumulated_grads[name] = accumulated_grads[name] + g_detached

                saved.clear()

            # Average across alphas
            n_alphas = args.n_alphas
            ig_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)
            for li, L in enumerate(downstream_layers):
                for h in range(n_heads):
                    contribution = 0.0
                    for kind in ["q_input", "k_input", "v_input"]:
                        name = f"blocks.{L}.hook_{kind}"
                        if accumulated_grads[name] is None:
                            continue
                        avg_grad = accumulated_grads[name] / n_alphas
                        g = avg_grad[0, -1, h, :].to(torch.float32)
                        contribution += float(torch.dot(g, delta).item())
                    ig_mediation[li, h] = contribution

            per_position_ig.append(ig_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })

            top_idx = np.argsort(-ig_mediation.flatten())[:5]
            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, top IG mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}="
                      f"{ig_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))

        if not per_position_ig:
            print(f"  WARN: no positions; skipping")
            continue

        mean_ig = np.mean(np.stack(per_position_ig), axis=0)
        flat = mean_ig.flatten()
        top_idx = np.argsort(-flat)[:10]
        print(f"\n  TOP 10 IG MEDIATORS (avg over {len(per_position_ig)} positions):")
        for rank, i in enumerate(top_idx, 1):
            L = downstream_layers[i // n_heads]
            h = i % n_heads
            print(f"    {rank:>2}. L{L}H{h}: IG={flat[i]:+.4f}")

        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_ig),
            "n_alphas": args.n_alphas,
            "per_position_meta": per_position_meta,
            "mean_ig_mediation": mean_ig.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_ig.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "integrated_gradients_per_head",
        "n_alphas": args.n_alphas,
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
