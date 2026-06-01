"""SAE-feature NOISING for necessity testing.

Complements script 26 (sae_feature_patching = denoising = sufficiency test).
Tests the OTHER direction: take a clean prompt where the feature fires, ZERO
OUT that feature's contribution to the residual stream, and measure whether
the concept-logit drops. If it drops, the feature is NECESSARY for the
concept.

Method (per Heimersheim & Nanda 2024):
  For each labeled feature X:
    1. Find clean positions where feature X fires hard (top-3 by activation).
    2. For each clean position:
       a. Get clean_resid at firing position; encode through SAE to get
          f_X^clean.
       b. Get baseline concept-logit from clean forward run.
       c. Apply ablation hook: subtract f_X^clean * W_dec[:, X] from the
          residual at the firing position (zero out feature X).
       d. Get patched concept-logit from patched forward run.
       e. logit_drop = baseline - patched (POSITIVE means feature was
          necessary; removing it dropped the concept).
    3. Average logit_drop across positions.

Verdicts:
  necessary:     mean_logit_drop >= necessary_threshold (default 0.5)
  not_necessary: mean_logit_drop <= weak_threshold (default 0.05)
  ambiguous:     otherwise

Combined with existing denoising results, produces the 2x2 classification:

                       Sufficiency (denoising)
                       PASS                    FAIL
  Necessity   PASS  │ TRUE DRIVER          │ AND-circuit component │
  (noising)         │ (necessary AND       │ (necessary but only   │
                    │  sufficient alone)   │  with teammates)      │
              ──────┼──────────────────────┼───────────────────────┤
              FAIL  │ OR-circuit component │ THERMOMETER           │
                    │ (sufficient alone    │ (no causal role)      │
                    │  but redundant)      │                       │

This is the methodological strengthening the Heimersheim & Nanda paper
recommends: don't claim a feature is a "driver" without testing both
directions.
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


LABEL_CATEGORIES = [
    {"name": "newline", "label_keywords": ["newline"],
     "token_match": lambda s: "\n" in s},
    {"name": "punctuation_formatting", "label_keywords": ["punctuation", "formatting"],
     "token_match": lambda s: bool(re.fullmatch(r"[\s\.,;:!?\-()\[\]{}\"'\\/]+", s)) and len(s) > 0},
    {"name": "file_path", "label_keywords": ["file path", "directory", "path"],
     "token_match": lambda s: ("/" in s) or ("\\" in s) or
                              any(s.rstrip().endswith(ext) for ext in
                                  [".py", ".js", ".html", ".css", ".cpp", ".java", ".md", ".txt"])},
    {"name": "french", "label_keywords": ["french"],
     "token_match": lambda s: any(c in s.lower() for c in "àâäçéèêëîïôöùûüÿ") or
                              any(w in s.lower() for w in [" la ", " le ", " du ", " rue ", " des "])},
    {"name": "math_notation", "label_keywords": ["math", "equation", "mathematical", "exponent"],
     "token_match": lambda s: any(c in s for c in "=+-*/^∑∏∫π√") and len(s) <= 5},
    {"name": "citation_latex", "label_keywords": ["bibtex", "latex", "citation"],
     "token_match": lambda s: any(t in s for t in ["\\cite", "\\ref", "\\bibitem", "@", "et al"])},
    {"name": "logical_operators", "label_keywords": ["logical operator", "negation"],
     "token_match": lambda s: s.strip() in ["!", "not", "&&", "||", "!=", "==", "&", "|"]},
    {"name": "code_keyword", "label_keywords": ["def keyword", "function definition", "code definition", "class"],
     "token_match": lambda s: s.strip() in ["def", "class", "function", "return", "import", "from"]},
    {"name": "decimal_numerical", "label_keywords": ["decimal", "numerical"],
     "token_match": lambda s: bool(re.fullmatch(r"[\d\.\-]+", s.strip())) and len(s.strip()) > 0},
    {"name": "subword_bpe", "label_keywords": ["subword", "bpe", "continuation"],
     "token_match": lambda s: len(s) >= 2 and not s.startswith(" ") and s[0].islower()},
    {"name": "abbreviation_acronym", "label_keywords": ["acronym", "abbreviation"],
     "token_match": lambda s: bool(re.fullmatch(r"[A-Z]{2,5}\.?", s.strip()))},
]


def classify_label(label: str):
    label_lower = label.lower()
    for cat in LABEL_CATEGORIES:
        if any(kw in label_lower for kw in cat["label_keywords"]):
            return cat
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--acts", type=str, default="data/acts_layer6.npy")
    p.add_argument("--tokens", type=str, default="data/token_stream.npy")
    p.add_argument("--context_window", type=int, default=30)
    p.add_argument("--n_clean_positions", type=int, default=3,
                    help="Number of top-firing clean positions to average noising over.")
    p.add_argument("--necessary_threshold", type=float, default=0.5,
                    help="Logit drop above this counts as necessary.")
    p.add_argument("--not_necessary_threshold", type=float, default=0.05,
                    help="Logit drop below this counts as not necessary.")
    p.add_argument("--feature_ids", type=str, default=None,
                    help="Comma-separated feature IDs to restrict to. "
                         "Defaults to features in denoising results JSON.")
    p.add_argument("--denoising_json", type=str,
                    default="checkpoints/sae_layer6_topk64_full.sae_feature_patching.json")
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


def get_context_around_position(token_stream, position, window):
    start = max(0, position - window + 1)
    return token_stream[start:position + 1]


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
    print(f"loading {model_name}, hook={hook_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # --- Load data ---
    print(f"loading activations from {args.acts}...")
    acts = np.load(args.acts)
    print(f"  shape: {acts.shape}")
    print(f"loading tokens from {args.tokens}...")
    token_stream = np.load(args.tokens)

    # --- Load catalog ---
    catalog = json.loads(Path(args.catalog).read_text())
    classified_all = []
    for fid_str, info in catalog.items():
        if "error" in info or info.get("classification") != "MONOSEMANTIC":
            continue
        cat = classify_label(info.get("label", ""))
        if cat is None:
            continue
        classified_all.append((int(fid_str), info, cat))

    # --- Restrict to features from denoising results (default) ---
    if args.feature_ids:
        target_fids = set(int(x) for x in args.feature_ids.split(","))
    elif Path(args.denoising_json).exists():
        d = json.loads(Path(args.denoising_json).read_text())
        target_fids = set(r["feature_id"] for r in d["results"])
        print(f"\nrestricting to {len(target_fids)} features from denoising results")
    else:
        target_fids = None

    if target_fids:
        classified = [(fid, info, cat) for fid, info, cat in classified_all
                       if fid in target_fids]
    else:
        classified = classified_all
    print(f"{len(classified)} features to test\n")

    # --- Main loop ---
    results = []
    t0 = time.time()

    for i, (fid, info, cat) in enumerate(classified):
        label = info["label"]
        print(f"\n[{i+1:>2}/{len(classified)}] f{fid}  [{cat['name']}]  {label[:50]}")

        # 1. Find top-10 firing positions; collect up to n_clean_positions that re-fire.
        candidates = find_top_firing_positions(sae, fid, acts, device, top_k=10)

        # 2. Concept tokens
        vocab_size = model.cfg.d_vocab
        concept_token_ids = []
        for tid in range(vocab_size):
            s = model.tokenizer.decode([tid])
            if cat["token_match"](s):
                concept_token_ids.append(tid)
        if not concept_token_ids:
            print(f"  WARN: no concept tokens; skipping")
            continue

        # 3. Decoder column for feature X
        decoder_col_X = sae.W_dec[:, fid].detach()  # (d_model,)

        # 4. For each clean firing position, ablate the feature and measure logit drop.
        per_position_results = []
        for cand_pos, cand_act in candidates:
            if len(per_position_results) >= args.n_clean_positions:
                break

            cand_tokens = get_context_around_position(token_stream, cand_pos, args.context_window)
            cand_tokens_tensor = torch.tensor([cand_tokens.tolist()], dtype=torch.long, device=device)

            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(
                    cand_tokens_tensor, names_filter=[hook_name]
                )
            clean_resid = clean_cache[hook_name][:, -1, :]
            with torch.no_grad():
                clean_features = sae.encode(clean_resid)
            f_clean = clean_features[0, fid].item()
            if f_clean <= 0.5:
                continue  # feature didn't re-fire

            # Cleanest metric: logprob of the ACTUAL next token in the corpus.
            # No "breaking the model" false positive — uniform output would
            # give logprob ≈ log(1/vocab) which is very low for any specific
            # token.
            actual_next_token = int(token_stream[cand_pos + 1])
            actual_next_str = model.tokenizer.decode([actual_next_token])

            clean_logprobs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)
            baseline_logp = clean_logprobs[actual_next_token].item()

            # Ablation: subtract f_clean * decoder_col at last position.
            def ablation_hook(activation, hook):
                activation[:, -1, :] = activation[:, -1, :] - f_clean * decoder_col_X
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    cand_tokens_tensor,
                    fwd_hooks=[(hook_name, ablation_hook)],
                )
                patched_logprobs = torch.log_softmax(patched_logits[0, -1, :], dim=-1)
                patched_logp = patched_logprobs[actual_next_token].item()

            # logp_drop > 0 means baseline > patched → ablation hurt the
            # model's prediction of the actual continuation → feature was necessary.
            logp_drop = baseline_logp - patched_logp

            # Only count this position if the actual next token is "concept-like"
            # (model should've been predicting a concept token here for the test
            # to be meaningful). Track this for filtering downstream.
            is_concept_next = actual_next_token in set(concept_token_ids)

            per_position_results.append({
                "position": cand_pos,
                "original_activation": cand_act,
                "f_clean_reencoded": f_clean,
                "actual_next_token": actual_next_token,
                "actual_next_str": actual_next_str,
                "is_concept_next": is_concept_next,
                "baseline_logp": baseline_logp,
                "patched_logp": patched_logp,
                "logp_drop": logp_drop,
            })
            concept_flag = "[concept]" if is_concept_next else "[non-concept]"
            print(f"    pos {cand_pos}: f_X={f_clean:.2f}, "
                  f"next={actual_next_str!r} {concept_flag} "
                  f"baseline logP={baseline_logp:+.3f}, "
                  f"patched={patched_logp:+.3f}, drop={logp_drop:+.4f}")

        if not per_position_results:
            print(f"  WARN: no positions re-fired; skipping")
            continue

        # Filter to only concept-next positions for the verdict — they're the
        # only ones where "necessity for the concept" is even a meaningful question.
        concept_positions = [r for r in per_position_results if r["is_concept_next"]]
        all_positions = per_position_results

        if not concept_positions:
            print(f"  WARN: no firing positions had concept-next token; reporting on all positions")
            verdict_positions = all_positions
        else:
            verdict_positions = concept_positions

        mean_logp_drop = float(np.mean([r["logp_drop"] for r in verdict_positions]))
        mean_logp_drop_all = float(np.mean([r["logp_drop"] for r in all_positions]))

        if mean_logp_drop >= args.necessary_threshold:
            verdict = "necessary"
        elif mean_logp_drop <= args.not_necessary_threshold:
            verdict = "not_necessary"
        else:
            verdict = "ambiguous"

        print(f"  n concept-next positions: {len(concept_positions)}/{len(all_positions)}")
        print(f"  mean logp_drop (concept-next only): {mean_logp_drop:+.4f}")
        print(f"  mean logp_drop (all positions):     {mean_logp_drop_all:+.4f}")
        print(f"  verdict: {verdict.upper()}")

        results.append({
            "feature_id": fid,
            "label": label,
            "category": cat["name"],
            "n_positions": len(per_position_results),
            "n_concept_next_positions": len(concept_positions),
            "per_position_results": per_position_results,
            "mean_logp_drop": mean_logp_drop,
            "mean_logp_drop_all": mean_logp_drop_all,
            "verdict": verdict,
        })

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # --- 2x2 combination with denoising results ---
    print(f"\n{'=' * 90}")
    print("2x2 CLASSIFICATION (sufficiency × necessity)")
    print(f"{'=' * 90}")

    denoising_verdicts = {}
    if Path(args.denoising_json).exists():
        d = json.loads(Path(args.denoising_json).read_text())
        denoising_verdicts = {r["feature_id"]: r["verdict"] for r in d["results"]}

    classification_counts = {
        "true_driver": [],
        "or_circuit": [],
        "and_circuit": [],
        "thermometer": [],
        "ambiguous": [],
    }

    print(f"\n  {'fid':>6} {'label':<35} {'denoise':>10} {'noise':>14} {'combined':>20}")
    print(f"  {'-'*6:>6} {'-'*35:<35} {'-'*10:>10} {'-'*14:>14} {'-'*20:>20}")

    for r in results:
        fid = r["feature_id"]
        denoise = denoising_verdicts.get(fid, "?")
        noise = r["verdict"]

        # Map denoising verdict: driver=sufficient, thermometer=not_sufficient, ambiguous=?
        suff = (denoise == "driver")
        not_suff = (denoise == "thermometer")
        nec = (noise == "necessary")
        not_nec = (noise == "not_necessary")

        if suff and nec:
            combined = "TRUE_DRIVER"
            classification_counts["true_driver"].append(fid)
        elif suff and not_nec:
            combined = "OR_CIRCUIT"
            classification_counts["or_circuit"].append(fid)
        elif not_suff and nec:
            combined = "AND_CIRCUIT"
            classification_counts["and_circuit"].append(fid)
        elif not_suff and not_nec:
            combined = "THERMOMETER"
            classification_counts["thermometer"].append(fid)
        else:
            combined = "AMBIGUOUS"
            classification_counts["ambiguous"].append(fid)

        r["denoising_verdict"] = denoise
        r["combined_classification"] = combined
        print(f"  f{fid:>5} {r['label'][:35]:<35} {denoise:>10} {noise:>14} {combined:>20}")

    n = len(results)
    print(f"\n{'=' * 90}")
    print(f"COMBINED CLASSIFICATION SUMMARY (n={n})")
    print(f"{'=' * 90}")
    for kind, fids in classification_counts.items():
        print(f"  {kind:<15}: {len(fids):>2}/{n} ({100*len(fids)/n:>5.1f}%)  features: {fids}")

    # Save
    out_path = args.out or str(Path(args.ckpt).with_suffix(".sae_feature_noising.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "method": "sae_feature_noising",
        "necessary_threshold": args.necessary_threshold,
        "not_necessary_threshold": args.not_necessary_threshold,
        "n_features": n,
        "combined_counts": {k: len(v) for k, v in classification_counts.items()},
        "combined_feature_ids": classification_counts,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
