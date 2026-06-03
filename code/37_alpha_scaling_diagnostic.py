"""Diagnose WHY AtP degrades with depth via empirical alpha-scaling.

Premise (from Finding 11): AtP-Pearson with AP drops from 0.999 at L7 to
0.78 at L11. AtP is exact iff the metric responds *linearly* to the
perturbation. So we measure that response directly.

For each (feature, head, position) pair, scale the perturbation:
    perturbation = -alpha * f_clean * decoder_col_X
for alpha in {0.25, 0.5, 0.75, 1.0}, and measure the actual logp_drop.

If response(alpha) is linear in alpha:
    -> AtP is exact (slope = AtP estimate)
    -> No fix needed; if Pearson < 1.0 at this layer, the gap is noise
If response(alpha) is concave / saturated:
    -> AtP at alpha=0 overestimates the alpha=1 effect
    -> Curvature lives downstream of the SAE layer
    -> A more accurate cheap method needs to account for it

Control: run on L7 (where AtP is accurate). The response should look linear.
Treatment: run on L11 (where AtP fails). The response should look curved.

Output: per-layer per-pair response curves, and a "linearity score" =
    (response(1.0)) / (4 * response(0.25))  # = 1.0 if linear
The L7 distribution of this score should center at ~1.0; if L11's centers
elsewhere (and Pearson(AtP, AP) is < 1.0), we've localized the failure to
nonlinearity in downstream layers.

Cost: 4 alphas * 12 heads * 3 positions * 3 features * 2 layers (L7 + L11)
    = 864 model forward passes. ~12 minutes on MPS.
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
    p.add_argument("--layers_to_test", type=str, default="7,11",
                   help="Comma-separated downstream layers to characterize.")
    p.add_argument("--alphas", type=str, default="0.25,0.5,0.75,1.0",
                   help="Scaling factors for the perturbation.")
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

    n_heads = model.cfg.n_heads
    layers_to_test = [int(x) for x in args.layers_to_test.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    print(f"  n_heads={n_heads}, layers_to_test={layers_to_test}, alphas={alphas}")

    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)
    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    # results[feature_id][L][h][alpha_idx] = list of logp_drops across positions
    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}\nFEATURE f{feature_id}\n{'=' * 70}")
        decoder_col_X = sae.W_dec[:, feature_id].detach().to(torch.float32)

        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)

        feature_results = {L: {h: {a: [] for a in alphas} for h in range(n_heads)}
                           for L in layers_to_test}
        per_position_meta = []
        n_pos_done = 0

        for cand_pos, cand_act in candidates:
            if n_pos_done >= args.n_positions:
                break

            cand_tokens = token_stream[max(0, cand_pos - args.context_window + 1):cand_pos + 1]
            cand_tokens_tensor = torch.tensor(
                [cand_tokens.tolist()], dtype=torch.long, device=device
            )

            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
                clean_resid = clean_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, baseline logP={baseline_logp:+.3f}")

            # delta = f_clean * decoder_col_X is the "full" perturbation
            # at alpha=1 we subtract the full thing, at alpha=0.25 we subtract 1/4
            for L in layers_to_test:
                for h in range(n_heads):
                    for alpha in alphas:
                        def make_hook(head_idx, alpha_val):
                            def hook_fn(activation, hook):
                                activation[:, -1, head_idx, :] = (
                                    activation[:, -1, head_idx, :]
                                    - alpha_val * f_clean * decoder_col_X
                                )
                                return activation
                            return hook_fn

                        hook_fn = make_hook(h, alpha)
                        fwd_hooks = [
                            (f"blocks.{L}.hook_q_input", hook_fn),
                            (f"blocks.{L}.hook_k_input", hook_fn),
                            (f"blocks.{L}.hook_v_input", hook_fn),
                        ]
                        with torch.no_grad():
                            patched_logits = model.run_with_hooks(
                                cand_tokens_tensor, fwd_hooks=fwd_hooks
                            )
                            patched_logprobs = torch.log_softmax(
                                patched_logits[0, -1, :], dim=-1
                            )
                            patched_logp = patched_logprobs[actual_next_token].item()
                        logp_drop = baseline_logp - patched_logp
                        feature_results[L][h][alpha].append(logp_drop)

            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
                "baseline_logp": baseline_logp,
            })
            n_pos_done += 1
            print(f"    done position {n_pos_done}/{args.n_positions} "
                  f"(elapsed {time.time() - t0:.0f}s)")

        all_results[feature_id] = {
            "per_position_meta": per_position_meta,
            "layers_to_test": layers_to_test,
            "alphas": alphas,
            "n_heads": n_heads,
            # nested dict -> arrays for easier downstream loading
            "logp_drops": {
                str(L): {
                    str(h): {
                        str(a): feature_results[L][h][a]
                        for a in alphas
                    } for h in range(n_heads)
                } for L in layers_to_test
            }
        }

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".alpha_scaling.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "alpha_scaling_diagnostic",
        "feature_results": all_results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
