"""LLM judge for steering success — graded replacement for the regex-based
concept-token classifier in script 22.

For each high-confidence monosemantic feature:
  1. Re-run steering at alpha = peak activation on neutral prompts.
  2. Send (label, baseline, steered) to an LLM judge.
  3. Get a 0-3 score for how much the steered output drifted toward the
     label's concept.
  4. Classify driver / thermometer / ambiguous based on mean score.
  5. Compare to script 22's regex-based classification.

The LLM judge addresses the biggest caveat in script 22: regex token-matching
under-counts conceptual drift for fuzzy categories (file paths, citations,
BPE continuations). An LLM judging "did this output drift toward [concept]?"
captures the intuition more cleanly.

Usage:
    export MOONSHOT_API_KEY=...
    python 24_llm_judge_steering.py
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
    "My favorite color is",
    "Yesterday I went to the park and saw",
]


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an intervention on a language model caused its output to drift toward a specific concept.

Concept: "{label}"

BASELINE output (no intervention):
{baseline!r}

STEERED output (intervention applied):
{steered!r}

Question: Did the STEERED output drift toward the concept "{label}" relative to the BASELINE?

Score on this exact scale:
0 = No drift. Steered output shows no more of the concept than baseline.
1 = Weak drift. Some mild drift toward the concept, but not dominant.
2 = Clear drift. The steered output is meaningfully more about the concept than baseline.
3 = Strong drift. The steered output is dominated by the concept (e.g., most tokens are concept-related).

Reply with ONLY a single digit (0, 1, 2, or 3) on its own line. No explanation, no other text."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--regex-results", type=str,
                   default="checkpoints/sae_layer6_topk64_full.thermometer_at_scale.json",
                   help="For agreement comparison with regex classification.")
    p.add_argument("--max-new-tokens", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha-mult", type=float, default=1.0)
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--judge-provider", type=str, default="moonshot",
                   choices=["moonshot", "openai", "anthropic"])
    p.add_argument("--judge-model", type=str, default="moonshot-v1-32k")
    p.add_argument("--driver-threshold", type=float, default=2.0,
                   help="mean LLM score >= this => driver")
    p.add_argument("--thermometer-threshold", type=float, default=0.5,
                   help="mean LLM score <= this => thermometer")
    p.add_argument("--max-features", type=int, default=None,
                   help="Cap number of features tested (None = all)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def get_client(provider):
    """Return an OpenAI-compatible (or Anthropic) client."""
    if provider == "moonshot":
        from openai import OpenAI
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise SystemExit("Set MOONSHOT_API_KEY environment variable.")
        return OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
    elif provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key)
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY environment variable.")
        return anthropic.Anthropic(api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")


def judge_drift(client, model, provider, label, baseline, steered, max_retries=2):
    """Send the triple to the LLM judge. Return 0-3 score or None on parse failure."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(label=label, baseline=baseline, steered=steered)
    for attempt in range(max_retries + 1):
        try:
            if provider in ("moonshot", "openai"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
            else:  # anthropic
                resp = client.messages.create(
                    model=model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()

            for c in text:
                if c in "0123":
                    return int(c)
            # Couldn't parse — fall through to retry
        except Exception as e:
            if attempt == max_retries:
                print(f"    judge error after {max_retries+1} attempts: {e}")
                return None
            time.sleep(1)
    return None


def make_hook(decoder_col, alpha):
    delta = (alpha * decoder_col).to(torch.float32)
    def hook(activation, hook):
        return activation + delta.to(activation.dtype)
    return hook


@torch.no_grad()
def generate_with_hook(model, prompt, hook_fn, hook_name, max_new, temp):
    tokens = model.to_tokens(prompt)
    if hook_fn is None:
        out = model.generate(tokens, max_new_tokens=max_new, temperature=temp,
                             do_sample=True, verbose=False)
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(tokens, max_new_tokens=max_new, temperature=temp,
                                 do_sample=True, verbose=False)
    return model.to_string(out[0])


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- Load SAE and model ---
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

    # --- Load regex results for comparison ---
    regex_data = json.loads(Path(args.regex_results).read_text())
    fids_to_test = [r["feature_id"] for r in regex_data["results"]]
    regex_verdicts = {r["feature_id"]: r["verdict"] for r in regex_data["results"]}
    if args.max_features:
        fids_to_test = fids_to_test[: args.max_features]
    print(f"\ntesting {len(fids_to_test)} features")

    catalog = json.loads(Path(args.catalog).read_text())

    # --- Set up LLM judge ---
    client = get_client(args.judge_provider)
    print(f"LLM judge: provider={args.judge_provider}  model={args.judge_model}")

    # --- Pre-generate baselines ---
    prompts = NEUTRAL_PROMPTS[: args.n_prompts]
    print(f"\ngenerating baselines for {len(prompts)} prompts...")
    baselines = {}
    for p in prompts:
        torch.manual_seed(args.seed)
        baselines[p] = generate_with_hook(model, p, None, hook_name,
                                          args.max_new_tokens, args.temperature)

    # --- Loop over features ---
    results = []
    t0 = time.time()
    for i, fid in enumerate(fids_to_test):
        info = catalog.get(str(fid))
        if info is None:
            print(f"  WARN: f{fid} not in catalog, skipping")
            continue
        label = info["label"]
        peak = info["peak_activation"]
        alpha = args.alpha_mult * peak
        decoder_col = sae.W_dec[:, fid].detach().clone()

        scores = []
        triples = []
        for prompt in prompts:
            torch.manual_seed(args.seed)
            steered = generate_with_hook(model, prompt, make_hook(decoder_col, alpha),
                                          hook_name, args.max_new_tokens, args.temperature)
            baseline = baselines[prompt]
            score = judge_drift(client, args.judge_model, args.judge_provider,
                                label, baseline, steered)
            scores.append(score)
            triples.append({
                "prompt": prompt,
                "baseline": baseline,
                "steered": steered,
                "llm_score": score,
            })

        valid = [s for s in scores if s is not None]
        mean_score = float(np.mean(valid)) if valid else None

        if mean_score is None:
            verdict = "judge_failed"
        elif mean_score >= args.driver_threshold:
            verdict = "driver"
        elif mean_score <= args.thermometer_threshold:
            verdict = "thermometer"
        else:
            verdict = "ambiguous"

        regex_v = regex_verdicts.get(fid, "?")
        mark = "✓" if verdict == regex_v else "✗"
        print(f"  [{i+1:>2}/{len(fids_to_test)}] f{fid:>5}  "
              f"llm={verdict:>11}  regex={regex_v:>11}  {mark}  "
              f"score={mean_score if mean_score is not None else 'NA'!s:>5}  "
              f"[{label[:40]}]")

        results.append({
            "feature_id": fid,
            "label": label,
            "alpha": alpha,
            "scores_per_prompt": scores,
            "mean_llm_score": mean_score,
            "llm_verdict": verdict,
            "regex_verdict": regex_v,
            "agreement": verdict == regex_v,
            "triples": triples,
        })

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("LLM JUDGE RESULTS")
    print(f"{'=' * 70}")
    n = len(results)
    by_verdict = {"driver": 0, "thermometer": 0, "ambiguous": 0, "judge_failed": 0}
    agreements = 0
    for r in results:
        by_verdict[r["llm_verdict"]] += 1
        if r["agreement"]:
            agreements += 1
    print(f"  Total features judged: {n}")
    for v, c in by_verdict.items():
        if c > 0:
            print(f"    {v:>12}: {c:>3} ({100*c/n:.1f}%)")
    print(f"\n  Agreement with regex classifier: {agreements}/{n} ({100*agreements/n:.1f}%)")

    # Where they disagree
    print(f"\n{'=' * 70}")
    print("DISAGREEMENTS (LLM judge vs regex)")
    print(f"{'=' * 70}")
    disagreements = [r for r in results if not r["agreement"]]
    if not disagreements:
        print("  No disagreements.")
    else:
        for r in disagreements:
            print(f"  f{r['feature_id']:>5}  llm={r['llm_verdict']:>11}  "
                  f"regex={r['regex_verdict']:>11}  score={r['mean_llm_score']}  "
                  f"[{r['label'][:40]}]")

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".llm_judge_steering.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "alpha_mult": args.alpha_mult,
        "driver_threshold": args.driver_threshold,
        "thermometer_threshold": args.thermometer_threshold,
        "n_features": n,
        "summary_by_verdict": by_verdict,
        "agreement_with_regex": agreements / n if n > 0 else None,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
