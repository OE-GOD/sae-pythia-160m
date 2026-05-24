"""Random-direction control for causal intervention.

The original causal intervention (19_causal_intervention.py) showed that
forcing a feature on changes the model's output. But maybe ANY intervention
on the residual stream produces drift — not specifically real features.

This script tests that hypothesis. For each real feature, we also steer using
a matched random unit vector. If random steering produces equally interpretable
drift, the causal claim is weakened. If random steering produces only noise,
the claim is strengthened.

Outputs side-by-side comparison: baseline, real-feature steering, random steering.

Usage:
    python 15_causal_random_control.py
    python 15_causal_random_control.py --features 2255,12520
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


NEUTRAL_PROMPTS = [
    "The recipe for chocolate cake is",
    "Yesterday I went to the park and saw",
    "My favorite color is",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated feature IDs (default: f12520,f2255 — known steerers)")
    p.add_argument("--alpha-mult", type=float, default=1.0,
                   help="Alpha as multiple of feature's peak activation")
    p.add_argument("--n-random-controls", type=int, default=3,
                   help="How many random directions to test per feature")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def make_hook(direction: torch.Tensor, alpha: float):
    delta = (alpha * direction).to(torch.float32)
    def hook(activation, hook):
        return activation + delta.to(activation.dtype)
    return hook


@torch.no_grad()
def generate_with_hook(model, prompt, hook_fn, hook_name, max_new_tokens, temperature):
    tokens = model.to_tokens(prompt)
    if hook_fn is None:
        out = model.generate(tokens, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True, verbose=False)
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(tokens, max_new_tokens=max_new_tokens,
                                 temperature=temperature, do_sample=True, verbose=False)
    return model.to_string(out[0])


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load SAE ---
    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    n_features = ckpt["n_features"]
    sae = TopKSAE(d_model, n_features, k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    # --- Load model ---
    model_name = ckpt["meta"].get("model", "pythia-160m")
    layer = ckpt["meta"].get("layer", 6)
    hook_name = ckpt["meta"].get("hook", f"blocks.{layer}.hook_resid_post")
    print(f"loading {model_name}, hook={hook_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # --- Load catalog and pick features ---
    catalog = json.loads(Path(args.catalog).read_text())
    if args.features:
        chosen_ids = [int(x) for x in args.features.split(",")]
    else:
        # Default: features that worked in 19_causal_intervention
        chosen_ids = [12520, 2255]

    chosen = [(fid, catalog[str(fid)]) for fid in chosen_ids if str(fid) in catalog]
    if not chosen:
        raise SystemExit("No features matched")

    print(f"\ntesting {len(chosen)} features with {args.n_random_controls} random controls each")

    # --- Pre-generate random unit vectors ---
    # All random directions normalized to unit length in d_model space
    random_dirs = torch.randn(args.n_random_controls, d_model, device=device)
    random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True)

    # --- Run interventions ---
    results = []
    t0 = time.time()
    for fid, info in chosen:
        label = info["label"]
        peak = info["peak_activation"]
        alpha = args.alpha_mult * peak

        decoder_col = sae.W_dec[:, fid].detach().clone()

        print(f"\n{'=' * 70}")
        print(f"FEATURE f{fid}  —  {label}  (alpha = {alpha:.2f})")
        print(f"{'=' * 70}")

        feature_results = {
            "feature_id": fid,
            "label": label,
            "alpha": alpha,
            "comparisons": [],
        }

        for prompt in NEUTRAL_PROMPTS:
            print(f"\n  prompt: {prompt!r}")

            torch.manual_seed(args.seed)
            baseline = generate_with_hook(model, prompt, None, hook_name,
                                          args.max_new_tokens, args.temperature)
            print(f"\n  [baseline]")
            print(f"    {baseline!r}")

            # Real feature
            real_hook = make_hook(decoder_col, alpha)
            torch.manual_seed(args.seed)
            real_steered = generate_with_hook(model, prompt, real_hook, hook_name,
                                              args.max_new_tokens, args.temperature)
            print(f"\n  [REAL feature steering]")
            print(f"    {real_steered!r}")

            # Random directions
            random_outputs = []
            for ri, rdir in enumerate(random_dirs):
                rand_hook = make_hook(rdir, alpha)
                torch.manual_seed(args.seed)
                rand_steered = generate_with_hook(model, prompt, rand_hook, hook_name,
                                                  args.max_new_tokens, args.temperature)
                random_outputs.append(rand_steered)
                print(f"\n  [RANDOM control #{ri+1}]")
                print(f"    {rand_steered!r}")

            feature_results["comparisons"].append({
                "prompt": prompt,
                "baseline": baseline,
                "real_steered": real_steered,
                "random_steered": random_outputs,
            })

        results.append(feature_results)

    # --- Save ---
    out_path = Path(args.ckpt).with_suffix(".causal_random_control.json")
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "alpha_mult": args.alpha_mult,
        "n_random_controls": args.n_random_controls,
        "seed": args.seed,
        "results": results,
    }, indent=2))

    print(f"\n{'=' * 70}")
    print(f"saved {out_path}")
    print(f"total time: {time.time() - t0:.1f}s")
    print(f"\nManual reading:")
    print(f"  - If REAL output drifts toward feature's concept (e.g., newlines) and")
    print(f"    RANDOM outputs are noise/repetition/unrelated → causal claim STRENGTHENED")
    print(f"  - If REAL and RANDOM outputs look equally drifted → causal claim WEAKENED")


if __name__ == "__main__":
    main()
