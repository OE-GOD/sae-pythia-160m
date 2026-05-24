"""Test whether logit-weight-for-newlines predicts steering success.

Hypothesis: each feature's decoder column, multiplied by the model's unembedding
matrix, gives a vector saying "how much does activating this feature push each
output token's probability up or down?" The entries for newline tokens should
predict whether forcing the feature on actually produces newlines.

If logit weight predicts steering success better than PC1 alignment did, we've
found the right axis for understanding what makes a newline feature actually
drive newline output.

Usage:
    python 17_logit_weights.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from transformer_lens import HookedTransformer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
_topk_src = (Path(__file__).parent / "02b_train_sae_topk.py").read_text().split("def parse_args")[0]
exec(_topk_src)  # defines TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/sae_layer6_topk64_full.pt")
    p.add_argument("--catalog", type=str, default="data/feature_catalog.json")
    p.add_argument("--label-contains", type=str, default="newline")
    p.add_argument("--steering-results", type=str,
                   default="checkpoints/sae_layer6_topk64_full.pc1_steering.json",
                   help="Steering results from script 16 to cross-reference")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_newline_token_ids(model) -> dict:
    """Identify newline-related token IDs in the model's vocabulary."""
    # Test several strings that should produce newline tokens
    candidates = ["\n", "\n\n", " \n", "\n\t", "\n  ", "\r\n"]
    newline_tokens = {}
    for s in candidates:
        try:
            ids = model.to_tokens(s, prepend_bos=False)[0].tolist()
            for tid in ids:
                tok_str = model.tokenizer.decode([tid])
                if "\n" in tok_str:
                    newline_tokens[tid] = tok_str
        except Exception:
            pass

    # Also scan the full vocab for tokens containing \n (slow but thorough)
    vocab_size = model.cfg.d_vocab
    for tid in range(vocab_size):
        tok_str = model.tokenizer.decode([tid])
        if "\n" in tok_str:
            newline_tokens[tid] = tok_str

    return newline_tokens


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"loading SAE: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sae = TopKSAE(ckpt["d_model"], ckpt["n_features"], k=ckpt["k"]).to(device)  # noqa: F821
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()

    # --- Load model and unembedding ---
    model_name = ckpt["meta"].get("model", "pythia-160m")
    print(f"loading {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # Unembedding matrix W_U: shape (d_model, vocab_size)
    W_U = model.W_U.detach()
    print(f"  unembedding shape: {W_U.shape}")

    # --- Find newline token IDs ---
    print("\nidentifying newline tokens in vocab...")
    newline_tokens = find_newline_token_ids(model)
    print(f"  found {len(newline_tokens)} tokens containing '\\n'")
    print(f"  examples: {list(newline_tokens.items())[:10]}")

    # --- Find cluster ---
    catalog = json.loads(Path(args.catalog).read_text())
    cluster = [(int(fid), info) for fid, info in catalog.items()
               if args.label_contains.lower() in info.get("label", "").lower()]
    cluster_ids = [fid for fid, _ in cluster]
    print(f"\nfound {len(cluster_ids)} '{args.label_contains}' features")

    # --- Compute logit weights for each cluster feature ---
    # decoder_col shape: (d_model,)
    # W_U shape: (d_model, vocab_size)
    # logit_weights shape: (vocab_size,)
    print("\ncomputing logit weights...")
    results = []
    newline_ids = list(newline_tokens.keys())
    for fid, info in cluster:
        v = sae.W_dec[:, fid].detach()  # (d_model,)
        logit_w = v @ W_U  # (vocab_size,)
        logit_w_np = logit_w.cpu().numpy()

        # Logit weights for newline tokens
        nl_weights = logit_w_np[newline_ids]
        # Various summary stats
        max_nl = float(nl_weights.max())
        sum_nl = float(nl_weights.sum())
        mean_nl = float(nl_weights.mean())

        # Center around median (since softmax is shift-invariant)
        median_w = float(np.median(logit_w_np))
        centered = logit_w_np - median_w
        nl_centered = centered[newline_ids]
        max_nl_centered = float(nl_centered.max())
        sum_nl_centered = float(nl_centered.sum())

        # Top tokens this feature promotes
        top_idx = np.argsort(-centered)[:10]
        top_tokens = [(int(i), model.tokenizer.decode([int(i)]), float(centered[i])) for i in top_idx]

        results.append({
            "feature_id": fid,
            "label": info["label"],
            "peak_activation": info["peak_activation"],
            "max_nl_logit_weight": max_nl_centered,
            "sum_nl_logit_weight": sum_nl_centered,
            "mean_nl_logit_weight_raw": mean_nl,
            "median_overall": median_w,
            "top_promoted_tokens": top_tokens,
        })

    # --- Print results sorted by max nl logit weight ---
    results.sort(key=lambda r: -r["max_nl_logit_weight"])
    print(f"\n{'=' * 80}")
    print("LOGIT WEIGHT FOR NEWLINE TOKENS (per feature)")
    print(f"{'=' * 80}")
    print(f"  {'fid':>6} {'max_nl_w':>9} {'sum_nl_w':>10}  label")
    for r in results:
        print(f"  f{r['feature_id']:>5} {r['max_nl_logit_weight']:>9.3f} "
              f"{r['sum_nl_logit_weight']:>10.3f}  {r['label'][:50]}")

    # --- Show top promoted tokens for top and bottom features by nl_weight ---
    print(f"\n--- Top 5 features by max_nl_logit_weight ---")
    for r in results[:5]:
        print(f"\n  f{r['feature_id']}  ({r['label']})  max_nl_w = {r['max_nl_logit_weight']:.3f}")
        for tid, tok, w in r["top_promoted_tokens"][:5]:
            print(f"    tok {tid:>6} {tok!r:>20} w={w:+.3f}")

    print(f"\n--- Bottom 5 features by max_nl_logit_weight ---")
    for r in results[-5:]:
        print(f"\n  f{r['feature_id']}  ({r['label']})  max_nl_w = {r['max_nl_logit_weight']:.3f}")
        for tid, tok, w in r["top_promoted_tokens"][:5]:
            print(f"    tok {tid:>6} {tok!r:>20} w={w:+.3f}")

    # --- Cross-reference with previous steering results if available ---
    steering_path = Path(args.steering_results)
    if steering_path.exists():
        print(f"\n{'=' * 80}")
        print(f"CROSS-REFERENCE WITH STEERING RESULTS")
        print(f"{'=' * 80}")
        steering = json.loads(steering_path.read_text())
        per_feature = {r["feature_id"]: r for r in steering["per_feature"]}

        print(f"  {'fid':>6} {'max_nl_w':>9} {'|PC1|':>8} {'mean_Δnl':>10}  label")
        rows = []
        for r in results:
            fid = r["feature_id"]
            if fid in per_feature:
                s = per_feature[fid]
                rows.append({
                    "fid": fid,
                    "max_nl_w": r["max_nl_logit_weight"],
                    "pc1_abs": s["pc1_projection_abs"],
                    "mean_delta_nl": s["mean_delta_nl"],
                    "label": r["label"],
                })
                print(f"  f{fid:>5} {r['max_nl_logit_weight']:>9.3f} "
                      f"{s['pc1_projection_abs']:>8.4f} {s['mean_delta_nl']:>10.2f}  {r['label'][:40]}")

        if len(rows) >= 3:
            nl_w_arr = np.array([row["max_nl_w"] for row in rows])
            pc1_arr = np.array([row["pc1_abs"] for row in rows])
            delta_nl_arr = np.array([row["mean_delta_nl"] for row in rows])
            corr_nl_w = float(np.corrcoef(nl_w_arr, delta_nl_arr)[0, 1])
            corr_pc1 = float(np.corrcoef(pc1_arr, delta_nl_arr)[0, 1])
            print(f"\n  Pearson correlation with steering Δ newlines:")
            print(f"    max_nl_logit_weight vs Δnl: {corr_nl_w:.3f}")
            print(f"    |PC1 projection|    vs Δnl: {corr_pc1:.3f}")
            if abs(corr_nl_w) > abs(corr_pc1):
                print(f"\n  -> Logit weight is a STRONGER predictor than PC1 alignment.")
            else:
                print(f"\n  -> Logit weight is NOT a stronger predictor than PC1.")
    else:
        print(f"\n  (No steering results at {steering_path} — skipping cross-reference)")

    # --- Save ---
    out_path = args.out or str(Path(args.ckpt).with_suffix(".logit_weights.json"))
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "label_contains": args.label_contains,
        "n_newline_tokens": len(newline_tokens),
        "per_feature": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
