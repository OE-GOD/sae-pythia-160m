"""Midpoint AtP: take the gradient at the half-ablated state (alpha=0.5)
instead of at the clean state (alpha=0).

Lesson from script 40 (AtP-corrected trapezoidal): sampling at alpha=1
(fully ablated) gives a noisy gradient because the model is far OOD.
Averaging clean grad + ablated grad made things worse, not better.

Mathematical motivation: the secant slope from alpha=0 to alpha=1 (which
is what AP measures) equals the average gradient over [0, 1]. By the
midpoint rule for integration:
    secant ≈ gradient at alpha = 0.5
For a quadratic response function, this is EXACT. For higher-order
functions, the midpoint estimate is much better than the endpoint
estimate as long as the function is smooth.

Why this might beat standard AtP:
  - At deep layers, AP_curve is concave (script 37 showed this clearly).
    For concave functions, the slope at 0 (AtP) overestimates the secant.
    The slope at 0.5 underestimates it less (the curve is closer to its
    average there).

Why it might NOT beat AtP:
  - The alpha=0.5 state is still OOD, just less so than alpha=1.
  - At shallow layers where AtP is already 0.999 accurate, switching to
    midpoint loses some precision since clean is the truly-on-distribution
    point.

Cost: 1 forward + 1 backward (same as AtP), with a single extra hook to
subtract half the perturbation at the SAE layer during the forward pass.
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
    p.add_argument("--midpoint_alpha", type=float, default=0.5,
                   help="Where to take the gradient (0 = clean = AtP, "
                        "1 = fully ablated, 0.5 = midpoint).")
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
    midpoint_alpha = args.midpoint_alpha
    print(f"  midpoint_alpha={midpoint_alpha}, downstream_layers={downstream_layers}")

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
        per_position_atp = []
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_atp) >= args.n_positions:
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

            saved = {}

            def make_save_hook(name_full):
                def hook(activation, hook):
                    activation.retain_grad()
                    saved[name_full] = activation
                    return activation
                return hook

            # Midpoint hook: subtract midpoint_alpha * f_clean * decoder_col_X
            # at the SAE layer to move forward state to the chosen midpoint
            def sae_midpoint_hook(activation, hook):
                activation[:, -1, :] = (
                    activation[:, -1, :] - midpoint_alpha * f_clean * decoder_col_X
                )
                return activation

            fwd_hooks = [(hook_name, sae_midpoint_hook)] + \
                [(name, make_save_hook(name)) for name in qkv_hooks]

            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(cand_tokens_tensor)

            logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)
            metric = logprobs[actual_next_token]
            model.zero_grad()
            metric.backward()

            delta = (f_clean * decoder_col_X).to(torch.float32)

            atp_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)
            for li, L in enumerate(downstream_layers):
                for h in range(n_heads):
                    contribution = 0.0
                    for kind in ["q_input", "k_input", "v_input"]:
                        name = f"blocks.{L}.hook_{kind}"
                        g = saved[name].grad
                        if g is None:
                            continue
                        g_vec = g[0, -1, h, :].to(torch.float32)
                        contribution += float(torch.dot(g_vec, delta).item())
                    atp_mediation[li, h] = contribution

            per_position_atp.append(atp_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })
            saved.clear()

            top_idx = np.argsort(-atp_mediation.flatten())[:5]
            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, top midpoint-AtP: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}="
                      f"{atp_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))

        if not per_position_atp:
            continue

        mean_atp = np.mean(np.stack(per_position_atp), axis=0)
        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_atp),
            "midpoint_alpha": midpoint_alpha,
            "per_position_meta": per_position_meta,
            "mean_midpoint_atp_mediation": mean_atp.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(
        f".sae_feature_midpoint_atp_a{midpoint_alpha}.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": f"midpoint_atp_alpha_{midpoint_alpha}",
        "midpoint_alpha": midpoint_alpha,
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
