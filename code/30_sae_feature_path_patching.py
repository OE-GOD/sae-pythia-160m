"""Per-head mediation analysis for SAE driver features.

Path-patching question: for each TRUE driver SAE feature X, which downstream
attention heads MEDIATE its effect on the model's output? I.e., when we surgically
remove X's contribution from one specific head's view of the residual stream,
how much does the model's next-token prediction drop?

This is a directional-ablation form of path patching. For each (layer L, head h)
with L > 6 (the SAE's layer):

  1. Run the clean prompt (feature X fires); record baseline log P(next token).
  2. Hook the per-head q_input, k_input, v_input at (L, h) and subtract
     f_X^clean * decoder_col_X from each, at the firing position.
     This forces head (L, h) to "see" the residual as if feature X did not fire,
     while every other head still sees the natural residual.
  3. Measure patched log P(next token).
  4. mediation = baseline_logp - patched_logp. Positive ⇒ this head was
     using feature X's contribution.

If the model has no other paths from X to the output, then ablating X from
ANY mediating head should drop the prediction. If the effect is concentrated
in a small number of heads, those heads are the bottleneck.

For each driver feature, produces a (5 layers × 12 heads) mediation matrix.
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
    p.add_argument("--feature_ids", type=str, default="10047,13131,15245",
                    help="Comma-separated driver feature IDs to test.")
    p.add_argument("--n_positions", type=int, default=3,
                    help="Average mediation across this many firing positions per feature.")
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

    # --- Load SAE ---
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

    # Enable per-head q/k/v inputs so we can hook them individually
    model.set_use_split_qkv_input(True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    downstream_layers = list(range(layer + 1, n_layers))  # 7..11 for Pythia-160M
    print(f"  n_layers={n_layers}, n_heads={n_heads}, downstream_layers={downstream_layers}")

    # --- Load data ---
    print(f"loading activations / tokens...")
    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)

    feature_ids = [int(x) for x in args.feature_ids.split(",")]
    print(f"\ntesting features: {feature_ids}")

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}")
        print(f"FEATURE f{feature_id}")
        print(f"{'=' * 70}")

        decoder_col_X = sae.W_dec[:, feature_id].detach()  # (d_model,)

        # Find top firing positions
        candidates = find_top_firing_positions(sae, feature_id, acts, device, top_k=10)

        per_position_mediation = []  # list of (n_downstream_layers, n_heads) arrays
        per_position_meta = []

        for cand_pos, cand_act in candidates:
            if len(per_position_mediation) >= args.n_positions:
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
            with torch.no_grad():
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])
            actual_next_str = model.tokenizer.decode([actual_next_token])

            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, next={actual_next_str!r}, "
                  f"baseline logP={baseline_logp:+.3f}")

            # For each downstream (layer, head): ablate this head's view of X
            mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)

            for li, L in enumerate(downstream_layers):
                for h in range(n_heads):
                    # Hook factory: ablate this specific (L, h)
                    head_idx = h

                    def make_hook(head_idx):
                        def hook_fn(activation, hook):
                            # activation: [batch, seq, head, d_model]
                            activation[:, -1, head_idx, :] = (
                                activation[:, -1, head_idx, :] - f_clean * decoder_col_X
                            )
                            return activation
                        return hook_fn

                    hook_fn = make_hook(h)
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

                    mediation[li, h] = baseline_logp - patched_logp

            per_position_mediation.append(mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": actual_next_str,
                "baseline_logp": baseline_logp,
            })

            top_idx = np.argsort(-mediation.flatten())[:5]
            print(f"    top mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}={mediation.flatten()[i]:+.3f}"
                      for i in top_idx
                  ))

        if not per_position_mediation:
            print(f"  WARN: no positions re-fired; skipping feature {feature_id}")
            continue

        # Average across positions
        mean_mediation = np.mean(np.stack(per_position_mediation), axis=0)

        # Top mediators
        flat = mean_mediation.flatten()
        top_idx = np.argsort(-flat)[:10]
        print(f"\n  TOP 10 MEDIATORS (averaged across {len(per_position_mediation)} positions):")
        for rank, i in enumerate(top_idx, 1):
            L = downstream_layers[i // n_heads]
            h = i % n_heads
            print(f"    {rank:>2}. L{L}H{h}: mediation={flat[i]:+.4f}")

        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_mediation),
            "per_position_meta": per_position_meta,
            "mean_mediation": mean_mediation.tolist(),
        }

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"completed in {elapsed:.1f}s")

    # Save
    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_path_patching.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "sae_feature_path_patching_per_head_mediation",
        "feature_results": all_results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
