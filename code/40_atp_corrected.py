"""AtP-corrected: 2-point trapezoidal integrated gradients.

Standard AtP:
    effect ≈ ∇L|clean · Δx
Trapezoidal AtP (this script):
    effect ≈ 0.5 * (∇L|clean + ∇L|fully_ablated) · Δx

This is mathematically equivalent to N=2 integrated gradients with
endpoints {0, 1}. The motivation: if the metric's response to the
perturbation is concave (which Finding 11 + script 37 show happens at
deep layers), the average of the start and end gradients better
estimates the secant slope (which is what AP measures) than either
endpoint alone.

Cost: 2 forward + 2 backward passes per firing position (vs 1+1 for
AtP, 10+10 for IG with N=10). So 4 passes vs 2 (AtP) vs 20 (IG).

Question: at the deep-layer failure regime where AtP collapses to
Pearson 0.78 with AP, does 2-point trapezoidal recover near-IG-N=10
accuracy (Pearson 0.97 at L11)?

Same setup as script 32: per-head q/k/v_input gradient capture for
the 3 TRUE driver features × 3 firing positions × 5 downstream layers
× 12 heads = 180 (feature, head) pairs.
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


def compute_gradient_at_state(model, tokens, qkv_hooks, hook_name, ablation_alpha,
                               f_clean, decoder_col_X, actual_next_token):
    """Run forward+backward at the state where SAE-layer residual is
    perturbed by -alpha * f_clean * decoder_col_X at the last position.
    Returns: dict of qkv-hook-name -> grad tensor."""
    saved = {}

    def make_save_hook(name_full):
        def hook(activation, hook):
            activation.retain_grad()
            saved[name_full] = activation
            return activation
        return hook

    def sae_ablation_hook(activation, hook):
        if ablation_alpha != 0.0:
            activation[:, -1, :] = (
                activation[:, -1, :] - ablation_alpha * f_clean * decoder_col_X
            )
        return activation

    fwd_hooks = [(hook_name, sae_ablation_hook)] + \
        [(name, make_save_hook(name)) for name in qkv_hooks]

    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)

    logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)
    metric = logprobs[actual_next_token]
    model.zero_grad()
    metric.backward()

    grads = {}
    for name in qkv_hooks:
        g = saved[name].grad
        grads[name] = g.detach().clone() if g is not None else None
    saved.clear()
    return grads


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

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    qkv_hooks = []
    for L in downstream_layers:
        for kind in ["q_input", "k_input", "v_input"]:
            qkv_hooks.append(f"blocks.{L}.hook_{kind}")

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)
        per_position_corrected = []
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_corrected) >= args.n_positions:
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

            # Two gradient evaluations: at alpha=0 (clean) and alpha=1 (fully ablated)
            grads_clean = compute_gradient_at_state(
                model, cand_tokens_tensor, qkv_hooks, hook_name, 0.0,
                f_clean, decoder_col_X, actual_next_token
            )
            grads_patched = compute_gradient_at_state(
                model, cand_tokens_tensor, qkv_hooks, hook_name, 1.0,
                f_clean, decoder_col_X, actual_next_token
            )

            delta = (f_clean * decoder_col_X).to(torch.float32)

            corrected_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)
            for li, L in enumerate(downstream_layers):
                for h in range(n_heads):
                    contribution = 0.0
                    for kind in ["q_input", "k_input", "v_input"]:
                        name = f"blocks.{L}.hook_{kind}"
                        g_c = grads_clean.get(name)
                        g_p = grads_patched.get(name)
                        if g_c is None or g_p is None:
                            continue
                        # Trapezoidal avg
                        g_avg = 0.5 * (g_c[0, -1, h, :].to(torch.float32) +
                                        g_p[0, -1, h, :].to(torch.float32))
                        contribution += float(torch.dot(g_avg, delta).item())
                    corrected_mediation[li, h] = contribution

            per_position_corrected.append(corrected_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })

            top_idx = np.argsort(-corrected_mediation.flatten())[:5]
            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, top corrected mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}="
                      f"{corrected_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))

        if not per_position_corrected:
            print(f"  WARN: no positions; skipping")
            continue

        mean_corrected = np.mean(np.stack(per_position_corrected), axis=0)
        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_corrected),
            "per_position_meta": per_position_meta,
            "mean_corrected_mediation": mean_corrected.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_atp_corrected.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "atp_corrected_2pt_trapezoidal",
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
