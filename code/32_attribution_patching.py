"""Attribution patching (AtP) for SAE feature → downstream head mediation.

Mirrors script 30's setup but uses gradient-based first-order approximation
instead of running one forward pass per head.

Activation patching (script 30):
  For each (feature, layer L, head h):
    Run forward pass with head h's per-head q/k/v_input ablated by
    subtracting f_clean * decoder_col_X at the firing position.
    Measure logp_drop = baseline_logp - patched_logp.

  Cost: O(N_heads) forward passes per firing position.

Attribution patching (this script):
  Compute gradient of log P(actual_next_token) with respect to every
  hook_q_input, hook_k_input, hook_v_input at downstream layers,
  in a single backward pass. Then for each (L, h):
    AtP estimate of logp_drop ≈
        sum over q/k/v of grad[L][batch, -1, h, :] · (f_clean * decoder_col_X)

  Cost: 1 forward + 1 backward pass per firing position, regardless of
  N_heads.

Output JSON has both methods' estimates per (feature, L, h, position), so
we can compute correlation and identify failure modes.

The clean intended use: pair with script 30's output to build a like-for-like
comparison dataset. Same feature IDs, same firing positions, same downstream
heads — different methods.
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
                    help="Same as script 30 default — TRUE driver features.")
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

    # Enable per-head q/k/v inputs (matches script 30)
    model.set_use_split_qkv_input(True)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    downstream_layers = list(range(layer + 1, n_layers))
    print(f"  n_layers={n_layers}, n_heads={n_heads}, downstream_layers={downstream_layers}")

    # --- Load data ---
    acts = np.load(args.acts)
    token_stream = np.load(args.tokens)

    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    # Hook names to compute gradients on
    qkv_hooks = []
    for L in downstream_layers:
        for kind in ["q_input", "k_input", "v_input"]:
            qkv_hooks.append(f"blocks.{L}.hook_{kind}")

    all_results = {}
    t0 = time.time()

    for feature_id in feature_ids:
        print(f"\n{'=' * 70}")
        print(f"FEATURE f{feature_id}")
        print(f"{'=' * 70}")

        decoder_col_X = sae.W_dec[:, feature_id].detach()  # (d_model,)

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

            # Re-encode to confirm firing
            with torch.no_grad():
                _, c_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
                clean_resid = c_cache[hook_name][:, -1, :]
                f_clean = sae.encode(clean_resid)[0, feature_id].item()
            if f_clean <= 0.5:
                continue

            actual_next_token = int(token_stream[cand_pos + 1])

            # --- AtP: forward, save activations with retained grads, backward ---
            saved = {}

            def make_hook(hook_name_full):
                def hook(activation, hook):
                    activation.retain_grad()
                    saved[hook_name_full] = activation
                    return activation
                return hook

            fwd_hooks = [(name, make_hook(name)) for name in qkv_hooks]

            # Need activations to track gradients. Keeping param requires_grad
            # alive ensures autograd carries grad info through to our hooks.
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(cand_tokens_tensor)

            # Metric: log P(actual_next_token) at the last position
            logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)
            metric = logprobs[actual_next_token]

            # Backward
            model.zero_grad()
            metric.backward()

            # AtP estimate: for each (L, h), summed over q/k/v inputs at the last position
            # change in input = -f_clean * decoder_col_X (this is what activation patching does)
            # predicted change in metric = sum_qkv grad · (-f_clean * decoder_col_X)
            # logp_drop = baseline_logp - patched_logp ≈ -predicted_change
            #           = sum_qkv grad · (f_clean * decoder_col_X)
            delta = (f_clean * decoder_col_X).to(torch.float32)

            atp_mediation = np.zeros((len(downstream_layers), n_heads), dtype=np.float32)
            for li, L in enumerate(downstream_layers):
                for h in range(n_heads):
                    contribution = 0.0
                    for kind in ["q_input", "k_input", "v_input"]:
                        name = f"blocks.{L}.hook_{kind}"
                        grad = saved[name].grad  # [batch, seq, head, d_model]
                        if grad is None:
                            continue
                        # Get this head's input grad at the last position
                        g = grad[0, -1, h, :].to(torch.float32)
                        contribution += float(torch.dot(g, delta).item())
                    atp_mediation[li, h] = contribution

            per_position_atp.append(atp_mediation)
            per_position_meta.append({
                "position": cand_pos,
                "f_clean": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": model.tokenizer.decode([actual_next_token]),
            })

            top_idx = np.argsort(-atp_mediation.flatten())[:5]
            print(f"  pos {cand_pos}: f_X={f_clean:.2f}, top AtP mediators: "
                  + ", ".join(
                      f"L{downstream_layers[i // n_heads]}H{i % n_heads}={atp_mediation.flatten()[i]:+.4f}"
                      for i in top_idx
                  ))

            # Clean up
            saved.clear()

        if not per_position_atp:
            print(f"  WARN: no positions; skipping")
            continue

        mean_atp = np.mean(np.stack(per_position_atp), axis=0)

        # Top AtP mediators averaged
        flat = mean_atp.flatten()
        top_idx = np.argsort(-flat)[:10]
        print(f"\n  TOP 10 AtP MEDIATORS (avg over {len(per_position_atp)} positions):")
        for rank, i in enumerate(top_idx, 1):
            L = downstream_layers[i // n_heads]
            h = i % n_heads
            print(f"    {rank:>2}. L{L}H{h}: AtP={flat[i]:+.4f}")

        all_results[feature_id] = {
            "downstream_layers": downstream_layers,
            "n_heads": n_heads,
            "n_positions": len(per_position_atp),
            "per_position_meta": per_position_meta,
            "mean_atp_mediation": mean_atp.tolist(),
            "per_position_atp_mediation": [a.tolist() for a in per_position_atp],
        }

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"completed in {elapsed:.1f}s")

    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_atp.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "attribution_patching_per_head",
        "feature_results": all_results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
