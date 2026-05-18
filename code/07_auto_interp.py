"""Auto-interpret SAE features using Claude API.

For each selected feature:
  - Pull its top-10 max-activating snippets from feature_examples.json
  - Send to Claude with a prompt asking for:
      1. A one-sentence concept label
      2. Classification: MONOSEMANTIC / POLYSEMANTIC / UNCLEAR
      3. A confidence score (low/medium/high)

Saves a structured catalog: data/feature_catalog.json.

Selection: top 25 by peak activation + top 25 by mean activation (deduplicated).

Usage:
    export ANTHROPIC_API_KEY=...
    python 07_auto_interp.py --n-features 50
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", type=str, default="data/feature_examples.json")
    p.add_argument("--top-features-npz", type=str, default="data/top_features.npz")
    p.add_argument("--n-features", type=int, default=50)
    p.add_argument("--out", type=str, default="data/feature_catalog.json")
    p.add_argument("--provider", type=str, default="anthropic",
                   choices=["anthropic", "openai", "kimi", "moonshot"],
                   help="LLM provider. 'kimi'/'moonshot' use OpenAI-compatible Moonshot endpoint.")
    p.add_argument("--model", type=str, default=None,
                   help="Model name. Defaults: claude-sonnet-4-6 (anthropic), gpt-4o (openai), kimi-k2-0905-preview (kimi)")
    p.add_argument("--base-url", type=str, default=None,
                   help="OpenAI-compatible base URL. Used for kimi/deepseek/together/etc. Ignored for anthropic.")
    p.add_argument("--api-key-env", type=str, default=None,
                   help="Env var holding the API key. Defaults vary by provider.")
    p.add_argument("--max-snippet-chars", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_client(args):
    """Return (client, model_name, call_fn) for the chosen provider.

    call_fn(system, user) -> (text, input_tokens, output_tokens)
    """
    provider = args.provider.lower()

    if provider == "anthropic":
        import anthropic
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        if not os.environ.get(key_env):
            raise RuntimeError(f"Set {key_env}")
        client = anthropic.Anthropic(api_key=os.environ[key_env])
        model = args.model or "claude-sonnet-4-6"
        def call_fn(system, user):
            r = client.messages.create(
                model=model,
                max_tokens=400,
                temperature=args.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
        return client, model, call_fn

    # OpenAI-compatible: openai, kimi, deepseek, together, groq, etc.
    import openai
    if provider in ("kimi", "moonshot"):
        base_url = args.base_url or "https://api.moonshot.ai/v1"
        model = args.model or "kimi-k2-0905-preview"
        key_env = args.api_key_env or "MOONSHOT_API_KEY"
        if not os.environ.get(key_env):
            # Try alternate env var names commonly used
            for alt in ("KIMI_API_KEY", "MOONSHOT_API_KEY"):
                if os.environ.get(alt):
                    key_env = alt
                    break
            else:
                raise RuntimeError(f"Set MOONSHOT_API_KEY (or KIMI_API_KEY)")
    else:
        base_url = args.base_url  # default OpenAI if None
        model = args.model or "gpt-4o"
        key_env = args.api_key_env or "OPENAI_API_KEY"
        if not os.environ.get(key_env):
            raise RuntimeError(f"Set {key_env}")

    client = openai.OpenAI(
        api_key=os.environ[key_env],
        base_url=base_url,
    )

    def call_fn(system, user):
        r = client.chat.completions.create(
            model=model,
            max_tokens=400,
            temperature=args.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (
            r.choices[0].message.content,
            r.usage.prompt_tokens,
            r.usage.completion_tokens,
        )

    return client, model, call_fn


SYSTEM_PROMPT = """You are an expert at interpreting sparse autoencoder features from neural language models.

You are given the top-K text snippets that strongly activate a single feature. The target token in each snippet is the token whose hidden state caused this feature to fire most strongly.

Your task: identify what single concept (if any) this feature represents.

Return ONLY a JSON object with these fields:
  "label": one short sentence naming the concept (e.g., "Python function-definition keywords", "Newline tokens inside JSON/XML blocks", "Decimal points in numerical contexts").
  "classification": one of MONOSEMANTIC, POLYSEMANTIC, or UNCLEAR.
    - MONOSEMANTIC: all/most snippets share one clear concept
    - POLYSEMANTIC: snippets span multiple unrelated concepts
    - UNCLEAR: not enough signal or snippets too sparse
  "confidence": one of "low", "medium", "high".
  "evidence_note": one short sentence summarizing what the snippets have in common (or why they don't).

Examples of good labels:
  "Academic figure-reference XML tags (`[Fig. 1A](#F1){ref-type=\"fig\"}`)"
  "Subword BPE continuations of multi-piece words (e.g., 'rons' in 'squadrons')"
  "Closing parenthesis/bracket tokens"
  "Japanese script characters"
"""


USER_PROMPT_TEMPLATE = """Feature ID: {fid}

Top {n_snippets} max-activating snippets (target token shown in angle brackets):

{snippets}

Identify the concept this feature represents. Return ONLY the JSON object."""


def format_snippet(ex: dict, max_chars: int) -> str:
    snippet = ex["snippet"]
    target_pos = ex.get("target_position_in_snippet", -1)
    target = ex.get("target_token", "?").replace("\n", "\\n")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    snippet = snippet.replace("\n", "\\n")
    return f"  act={ex['activation']:.2f}  target=<{target}>  snippet: `{snippet}`"


def build_user_prompt(fid: int, examples: list[dict], max_chars: int) -> str:
    snippet_lines = [format_snippet(e, max_chars) for e in examples]
    return USER_PROMPT_TEMPLATE.format(
        fid=fid, n_snippets=len(examples), snippets="\n".join(snippet_lines)
    )


def extract_json(text: str) -> dict:
    """Extract the first JSON object from Claude's response."""
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"No JSON in response: {text[:200]!r}")


def main():
    args = parse_args()

    # Load feature examples + top-K matrix
    print(f"loading {args.examples}...")
    feature_examples = json.loads(Path(args.examples).read_text())
    feature_examples = {int(k): v for k, v in feature_examples.items()}
    print(f"  {len(feature_examples):,} alive features")

    print(f"loading {args.top_features_npz}...")
    npz = np.load(args.top_features_npz)
    top_values = npz["top_values"]  # (n_features, K)
    n_features, K = top_values.shape
    print(f"  shape ({n_features}, {K})")

    # Selection: top by peak, top by mean, deduplicated
    peak = top_values[:, 0]
    mean_topk = top_values.mean(axis=1)
    alive_mask = peak > 0
    alive_ids = np.where(alive_mask)[0]
    n_select = args.n_features

    by_peak = sorted(alive_ids.tolist(), key=lambda i: -peak[i])
    by_mean = sorted(alive_ids.tolist(), key=lambda i: -mean_topk[i])

    selected = list(dict.fromkeys(
        by_mean[: n_select // 2] + by_peak[: n_select // 2]
    ))[:n_select]
    print(f"\nselected {len(selected)} features for auto-interp")
    print(f"  union of top {n_select // 2} by peak + top {n_select // 2} by mean")

    if args.dry_run:
        print("\nDRY RUN — would label these features:")
        for fid in selected[:10]:
            examples = feature_examples.get(fid, [])
            if not examples:
                continue
            preview = examples[0]["snippet"].replace("\n", " ")[:60]
            print(f"  f{fid:5d}  peak={top_values[fid, 0]:.2f}  → {preview!r}")
        remaining = max(0, len(selected) - 10)
        if remaining:
            print(f"  ... and {remaining} more")
        return

    try:
        _client, model_name, call_fn = make_client(args)
    except Exception as e:
        print(f"ERROR setting up provider {args.provider}: {e}")
        return
    print(f"\nprovider: {args.provider}  model: {model_name}")

    catalog = {}
    total_in = 0
    total_out = 0
    t0 = time.time()

    for i, fid in enumerate(selected):
        examples = feature_examples.get(fid, [])
        if not examples:
            continue
        user_prompt = build_user_prompt(fid, examples, args.max_snippet_chars)

        try:
            raw_text, in_tok, out_tok = call_fn(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json(raw_text)
            total_in += in_tok
            total_out += out_tok

            entry = {
                "feature_id": fid,
                "peak_activation": float(top_values[fid, 0]),
                "mean_top_k_activation": float(mean_topk[fid]),
                "label": parsed.get("label", ""),
                "classification": parsed.get("classification", "UNCLEAR"),
                "confidence": parsed.get("confidence", "low"),
                "evidence_note": parsed.get("evidence_note", ""),
                "n_examples": len(examples),
            }
            catalog[fid] = entry

            label = entry["label"][:60]
            cls = entry["classification"]
            print(f"  [{i+1:2d}/{len(selected)}] f{fid:5d}  {cls:13s}  {label}", flush=True)
        except Exception as e:
            print(f"  [{i+1:2d}/{len(selected)}] f{fid:5d}  ERROR: {type(e).__name__}: {e}", flush=True)
            catalog[fid] = {"feature_id": fid, "error": str(e)}

    elapsed = time.time() - t0
    # Rough cost rates per million tokens
    rates = {
        "anthropic": (3.0, 15.0),       # Claude Sonnet
        "openai":    (2.5, 10.0),       # GPT-4o approx
        "kimi":      (0.6, 2.5),        # Kimi K2 approx
        "moonshot":  (0.6, 2.5),
    }
    in_rate, out_rate = rates.get(args.provider, (3.0, 15.0))
    cost_usd = total_in / 1e6 * in_rate + total_out / 1e6 * out_rate

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(catalog, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print(f"DONE  {elapsed:.1f}s  cost ~${cost_usd:.2f}")
    print(f"  features labeled: {len([c for c in catalog.values() if 'error' not in c])}/{len(selected)}")
    print(f"  tokens: {total_in:,} in + {total_out:,} out")
    print(f"  saved: {args.out}")

    # Summary table
    by_class = {"MONOSEMANTIC": 0, "POLYSEMANTIC": 0, "UNCLEAR": 0}
    for e in catalog.values():
        cls = e.get("classification")
        if cls in by_class:
            by_class[cls] += 1
    print(f"\nclassification breakdown:")
    for cls, count in by_class.items():
        print(f"  {cls:13s}  {count:3d}")


if __name__ == "__main__":
    main()
