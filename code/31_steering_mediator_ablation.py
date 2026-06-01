"""Validate Finding 9: steering + downstream head ablation (logit-based metric).

Tests whether the per-head path-patching mediators identified in Finding 9
are causally responsible for the feature's effect on output, or just
correlated.

Metric (logit-based, much sharper than generation-and-count):
  Steering effect on concept = sum_{t in concept_tokens} logprob(t | context)
  Computed as logsumexp over concept-token logprobs at the last position.

Four conditions per (feature, mediator_head, expected_direction) tuple:

  [N] NO_STEERING_NO_ABLATION:  baseline concept logprob
  [A] STEERING_ONLY:            concept logprob with steering
                                steering_effect = [A] - [N]
  [B] STEERING + MEDIATOR_ABLATED:  ablate target head's z at last position
                                ablation_effect = [A] - [B]  (how much
                                  the mediator was contributing to steering)
  [C] STEERING + RANDOM_ABLATED (averaged over n_random_controls):
                                ablation_effect_random = [A] - [C]

Predictions:

  - For a POSITIVE mediator: ablating reduces concept logprob more than
    random ablation → [B] < [C] (so [A] - [B] > [A] - [C]).
    Means the specific head was contributing positively to the steering
    effect.
  - For a NEGATIVE mediator: ablating INCREASES concept logprob (because
    the suppressive head is removed) → [B] > [C].

If selectivity = ([A] - [B]) - ([A] - [C]) = [C] - [B] is large enough in
the predicted direction, the mediator interpretation is validated.
"""
import argparse
import json
import os
import re
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


NEUTRAL_PROMPTS = [
    "The recipe for chocolate cake is",
    "My favorite color is",
    "Yesterday I went to the park and saw",
]


# Feature, primary mediator (positive direction), label
# From Finding 9: TRUE driver features and their top mediator heads.
TEST_PAIRS = [
    # (feature_id, layer, head, expected_direction)
    # Expected: "positive" = ablating reduces newlines (mediator was helping);
    #           "negative" = ablating increases newlines (mediator was suppressing).
    (10047, 8, 10, "positive"),
    (10047, 8,  9, "negative"),
    (13131, 8, 10, "positive"),
    (15245, 8,  9, "positive"),
    (15245, 8, 10, "negative"),
]


def newline_token_match(s):
    return "\n" in s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--alpha_mult", type=float, default=3.0,
                    help="alpha = alpha_mult * peak activation.")
    p.add_argument("--peak_activation", type=float, default=20.0,
                    help="Approximate peak activation for the TRUE driver features.")
    p.add_argument("--n_random_controls", type=int, default=10,
                    help="Number of random heads to use as ablation control.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def make_steering_hook(decoder_col, alpha):
    delta = (alpha * decoder_col).to(torch.float32)
    def hook(activation, hook):
        return activation + delta.to(activation.dtype)
    return hook


def make_head_ablation_hook(head_idx):
    """Zero out the specified head's z output at the last position only."""
    def hook(activation, hook):
        # activation shape: [batch, seq, head, d_head]
        activation[:, -1, head_idx, :] = 0.0
        return activation
    return hook


@torch.no_grad()
def concept_logprob(model, prompt, fwd_hooks, concept_token_ids):
    """Return logsumexp over concept-token logprobs at the last position.

    This is log P(next token ∈ concept set | context, intervention).
    Normalized — bounded by 0 — and immune to "breaking the model" artifacts
    that affect raw mean-logit metrics.
    """
    tokens = model.to_tokens(prompt)
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    logprobs = torch.log_softmax(last_logits, dim=-1)
    return torch.logsumexp(logprobs[concept_token_ids], dim=-1).item()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

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

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    downstream_heads = [(L, h) for L in range(layer + 1, n_layers) for h in range(n_heads)]
    print(f"  n_layers={n_layers}, n_heads={n_heads}, "
          f"n_downstream_heads={len(downstream_heads)}")

    alpha = args.alpha_mult * args.peak_activation
    print(f"  alpha = {alpha} ({args.alpha_mult}x peak={args.peak_activation})")

    # Concept tokens = newline-containing tokens
    vocab_size = model.cfg.d_vocab
    concept_token_ids = []
    for tid in range(vocab_size):
        s = model.tokenizer.decode([tid])
        if newline_token_match(s):
            concept_token_ids.append(tid)
    print(f"  n concept tokens (newline-containing): {len(concept_token_ids)}")

    results = []
    t0 = time.time()

    for fid, target_L, target_h, expected_dir in TEST_PAIRS:
        print(f"\n{'=' * 80}")
        print(f"FEATURE f{fid}  TARGET L{target_L}H{target_h}  ({expected_dir} mediator)")
        print(f"{'=' * 80}")

        decoder_col = sae.W_dec[:, fid].detach()
        steering_hook = make_steering_hook(decoder_col, alpha)

        # Random control heads (different from target)
        all_other_heads = [(L, h) for (L, h) in downstream_heads
                            if not (L == target_L and h == target_h)]
        rng.shuffle(all_other_heads)
        control_heads = all_other_heads[:args.n_random_controls]

        # ---------- Per prompt: 4 conditions ----------
        per_prompt = []
        for prompt in NEUTRAL_PROMPTS:
            # [N] baseline: no steering, no ablation
            lp_baseline = concept_logprob(model, prompt, fwd_hooks=[],
                                            concept_token_ids=concept_token_ids)

            # [A] steering only
            lp_steered = concept_logprob(
                model, prompt,
                fwd_hooks=[(hook_name, steering_hook)],
                concept_token_ids=concept_token_ids,
            )

            # [B] steering + mediator ablated
            ablation_hook_target = make_head_ablation_hook(target_h)
            lp_steered_mediator_ablated = concept_logprob(
                model, prompt,
                fwd_hooks=[
                    (hook_name, steering_hook),
                    (f"blocks.{target_L}.attn.hook_z", ablation_hook_target),
                ],
                concept_token_ids=concept_token_ids,
            )

            # [C] steering + random head ablated (avg over n_random_controls)
            random_ablated_lps = []
            for (ctrl_L, ctrl_h) in control_heads:
                ablation_hook_ctrl = make_head_ablation_hook(ctrl_h)
                lp = concept_logprob(
                    model, prompt,
                    fwd_hooks=[
                        (hook_name, steering_hook),
                        (f"blocks.{ctrl_L}.attn.hook_z", ablation_hook_ctrl),
                    ],
                    concept_token_ids=concept_token_ids,
                )
                random_ablated_lps.append(lp)

            per_prompt.append({
                "prompt": prompt,
                "lp_baseline": lp_baseline,
                "lp_steered": lp_steered,
                "lp_steered_mediator_ablated": lp_steered_mediator_ablated,
                "lp_steered_random_ablated_mean": float(np.mean(random_ablated_lps)),
                "lp_steered_random_ablated_std": float(np.std(random_ablated_lps)),
                "random_ablated_per_head": random_ablated_lps,
            })

        # Aggregate across prompts
        lp_baseline = float(np.mean([p["lp_baseline"] for p in per_prompt]))
        lp_steered = float(np.mean([p["lp_steered"] for p in per_prompt]))
        lp_mediator = float(np.mean([p["lp_steered_mediator_ablated"] for p in per_prompt]))
        lp_random = float(np.mean([p["lp_steered_random_ablated_mean"] for p in per_prompt]))

        steering_effect = lp_steered - lp_baseline
        mediator_contribution = lp_steered - lp_mediator  # how much the mediator was contributing
        random_contribution = lp_steered - lp_random
        selectivity = mediator_contribution - random_contribution

        # Verdict
        # For "positive" mediator: ablating it should REDUCE concept logprob more than random.
        #   mediator_contribution > random_contribution → selectivity > 0
        # For "negative" mediator: ablating it should INCREASE concept logprob more than random.
        #   mediator_contribution < random_contribution → selectivity < 0 (with margin)
        if expected_dir == "positive":
            verdict_strong = (selectivity > 0.05)  # mediator at least 0.05 nats more contribution
            verdict_weak = (selectivity > 0)
            verdict_short = "POS✓✓" if verdict_strong else "POS✓" if verdict_weak else "POS×"
        else:  # negative
            verdict_strong = (selectivity < -0.05)
            verdict_weak = (selectivity < 0)
            verdict_short = "NEG✓✓" if verdict_strong else "NEG✓" if verdict_weak else "NEG×"

        print(f"  [N] no steering:                          logP(concept)={lp_baseline:+.4f}")
        print(f"  [A] steering only:                        logP(concept)={lp_steered:+.4f}  "
              f"(steering effect = {steering_effect:+.4f})")
        print(f"  [B] steering + L{target_L}H{target_h} ablated:        "
              f"logP(concept)={lp_mediator:+.4f}  "
              f"(mediator contribution = {mediator_contribution:+.4f})")
        print(f"  [C] steering + random head ablated (avg): "
              f"logP(concept)={lp_random:+.4f}  "
              f"(random contribution = {random_contribution:+.4f})")
        print(f"  → Selectivity (mediator − random) = {selectivity:+.4f}  "
              f"({expected_dir}: {verdict_short})")

        results.append({
            "feature_id": fid,
            "target_layer": target_L,
            "target_head": target_h,
            "expected_direction": expected_dir,
            "alpha": alpha,
            "n_random_controls": args.n_random_controls,
            "control_heads": [{"layer": L, "head": h} for (L, h) in control_heads],
            "per_prompt": per_prompt,
            "lp_baseline": lp_baseline,
            "lp_steered": lp_steered,
            "lp_steered_mediator_ablated": lp_mediator,
            "lp_steered_random_ablated": lp_random,
            "steering_effect": steering_effect,
            "mediator_contribution": mediator_contribution,
            "random_contribution": random_contribution,
            "selectivity": selectivity,
            "verdict": verdict_short,
        })

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"completed in {elapsed:.1f}s")

    print(f"\n{'=' * 80}")
    print("OVERALL VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'feature':<10} {'target':<8} {'expected':<10} "
          f"{'steer_eff':>10} {'med_contr':>10} {'rand_contr':>10} {'select':>8} {'verdict':>9}")
    for r in results:
        print(f"  f{r['feature_id']:<9} L{r['target_layer']}H{r['target_head']:<5} "
              f"{r['expected_direction']:<10} "
              f"{r['steering_effect']:>+10.4f} {r['mediator_contribution']:>+10.4f} "
              f"{r['random_contribution']:>+10.4f} {r['selectivity']:>+8.4f} "
              f"{r['verdict']:>9}")

    # Save
    out_path = args.out or str(Path(args.ckpt).with_suffix(".steering_mediator_ablation.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "steering_plus_mediator_ablation_logit",
        "alpha_mult": args.alpha_mult,
        "peak_activation": args.peak_activation,
        "n_random_controls": args.n_random_controls,
        "neutral_prompts": NEUTRAL_PROMPTS,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
