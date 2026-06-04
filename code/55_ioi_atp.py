"""Apply attribution patching (AtP) to Indirect Object Identification (IOI)
on Pythia 160M. Compare to ground-truth full activation patching (AP).

IOI task (Wang et al. 2023):
  Clean prompt: "When John and Mary went to the store, Mary gave a drink to"
                                                                              ↑ predict IO = " John"
  Noise prompt (ABBA→BABA swap): "When Mary and John went to the store, John gave a drink to"
                                                                                              ↑ predict IO = " Mary"
  Metric: logit_diff = logit(IO) - logit(S)
  (positive when model gets it right)

Intervention: for each (layer, head), patch the head's output (hook_z) from
clean to noise. Measure resulting change in logit_diff.

Two methods to estimate per-(layer, head) importance:
  - Full AP: 144 forwards per prompt (one per head)
  - AtP: 1 forward + 1 backward per prompt — closed-form gradient · delta

Compare the top-K heads identified by each. If AtP recovers the same heads
that full AP finds, we've validated AtP works on a real circuit-discovery
task (not just our SAE-feature setup).

The Wang et al. paper on GPT-2 small identifies specific head categories
(Name Movers, S-Inhibition, etc.). Pythia 160M may have different circuit
structure (different training, different tokenization) but should still
solve IOI; the methodology test is "does AtP find what AP finds?"
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


# Hand-crafted IOI prompts with single-token names (for clean logit_diff)
# Format: ("clean prompt", " IO_token", " S_token", "noise prompt", " noise IO", " noise S")
# Names chosen to be single-token in Pythia tokenizer.
IOI_PROMPTS = [
    ("When John and Mary went to the store, Mary gave a drink to", " John", " Mary",
     "When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("When Alice and Bob went to the park, Bob handed the ball to", " Alice", " Bob",
     "When Bob and Alice went to the park, Alice handed the ball to", " Bob", " Alice"),
    ("When Tom and Sarah went to the cafe, Sarah passed the menu to", " Tom", " Sarah",
     "When Sarah and Tom went to the cafe, Tom passed the menu to", " Sarah", " Tom"),
    ("When James and Emma went to the office, Emma showed the file to", " James", " Emma",
     "When Emma and James went to the office, James showed the file to", " Emma", " James"),
    ("When Mark and Lisa went to the gym, Lisa threw the towel to", " Mark", " Lisa",
     "When Lisa and Mark went to the gym, Mark threw the towel to", " Lisa", " Mark"),
    ("When David and Anna went to the museum, Anna pointed the painting to", " David", " Anna",
     "When Anna and David went to the museum, David pointed the painting to", " Anna", " David"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="pythia-160m")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n_prompts", type=int, default=6)
    p.add_argument("--run_ap", action="store_true", default=True,
                    help="Run full AP for comparison (default True)")
    p.add_argument("--out", type=str, default="checkpoints/ioi_pythia160m_atp.json")
    return p.parse_args()


def logit_diff(logits, io_id, s_id):
    """logits: [d_vocab], scalar logit_diff = logit(io) - logit(s)."""
    return logits[io_id] - logits[s_id]


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"loaded {args.model}: {n_layers}L x {n_heads}H, d_head={model.cfg.d_head}")

    prompts = IOI_PROMPTS[:args.n_prompts]
    tok = model.tokenizer

    # Validate each prompt: name tokens must be single-token, and clean prompt
    # should correctly predict IO over S
    # Pythia tokenizer prepends BOS; strip it
    BOS = tok.encode("")[0] if len(tok.encode("")) > 0 else None
    def encode_name(s):
        ids = tok.encode(s)
        # remove leading BOS if present
        if ids and ids[0] == BOS:
            ids = ids[1:]
        return ids

    valid_prompts = []
    for clean, io, s, noise, n_io, n_s in prompts:
        clean_tokens = encode_name(clean)
        noise_tokens = encode_name(noise)
        if len(clean_tokens) != len(noise_tokens):
            print(f"  SKIP (token len mismatch): {clean!r}")
            continue
        io_ids = encode_name(io)
        s_ids = encode_name(s)
        if len(io_ids) != 1 or len(s_ids) != 1:
            print(f"  SKIP (multi-token name): {io!r}({io_ids}) or {s!r}({s_ids})")
            continue
        n_io_ids = encode_name(n_io)
        n_s_ids = encode_name(n_s)
        if len(n_io_ids) != 1 or len(n_s_ids) != 1:
            continue
        # Quick sanity check on clean model output
        toks = torch.tensor([clean_tokens], device=device)
        with torch.no_grad():
            logits = model(toks)[0, -1, :]
        clean_logit_diff = logit_diff(logits, io_ids[0], s_ids[0]).item()
        print(f"  CLEAN: '{clean}' -> logit_diff(IO={io}, S={s}) = {clean_logit_diff:+.3f}")
        if clean_logit_diff > 0:
            valid_prompts.append({
                "clean": clean, "io": io, "s": s,
                "noise": noise, "n_io": n_io, "n_s": n_s,
                "clean_tokens": clean_tokens, "noise_tokens": noise_tokens,
                "io_id": io_ids[0], "s_id": s_ids[0],
                "n_io_id": n_io_ids[0], "n_s_id": n_s_ids[0],
                "clean_logit_diff": clean_logit_diff,
            })
        else:
            print(f"  SKIP (model can't do IOI on this prompt)")

    if not valid_prompts:
        print("No valid prompts. Pythia 160M may struggle with IOI.")
        return

    print(f"\nUsing {len(valid_prompts)} valid prompts.\n")

    # ============================================================
    # For each prompt: compute AP and AtP per (L, h)
    # ============================================================
    per_prompt_ap = []   # [n_prompts, n_layers, n_heads]
    per_prompt_atp = []  # [n_prompts, n_layers, n_heads]

    t0 = time.time()

    for prompt_i, p in enumerate(valid_prompts):
        clean_toks = torch.tensor([p["clean_tokens"]], device=device)
        noise_toks = torch.tensor([p["noise_tokens"]], device=device)
        io_id, s_id = p["io_id"], p["s_id"]
        clean_ld = p["clean_logit_diff"]

        # Run clean & noise forward with cache of head outputs (hook_z)
        z_hooks = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_toks, names_filter=z_hooks)
            _, noise_cache = model.run_with_cache(noise_toks, names_filter=z_hooks)
        clean_z = {L: clean_cache[f"blocks.{L}.attn.hook_z"].detach() for L in range(n_layers)}
        noise_z = {L: noise_cache[f"blocks.{L}.attn.hook_z"].detach() for L in range(n_layers)}

        # AtP estimate: gradient of logit_diff w.r.t. z, dot with Δz = clean_z - noise_z
        # (We're measuring how much logit_diff drops if we patch clean → noise; same direction as below)
        # Standard AtP for "patch from noise" intervention: -grad · (clean - noise)
        # But our intervention is "patch clean head to noise value", so we want effect of going noise→clean direction
        # Easier: just compute AtP as grad · (noise - clean) = predicted change in metric

        # Cache hooks for AtP: retain_grad on each layer's hook_z
        saved = {}
        def make_capture(name):
            def hook(activation, hook):
                activation.retain_grad()
                saved[name] = activation
                return activation
            return hook
        fwd_hooks = [(n, make_capture(n)) for n in z_hooks]
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(clean_toks)
        ld = logit_diff(logits[0, -1, :], io_id, s_id)
        model.zero_grad()
        ld.backward()

        atp_per_head = np.zeros((n_layers, n_heads), dtype=np.float32)
        for L in range(n_layers):
            z_grad = saved[z_hooks[L]].grad  # [batch, seq, head, d_head]
            if z_grad is None:
                continue
            delta_z = noise_z[L] - clean_z[L]  # all positions
            # Sum over seq, since intervention is at all positions (we patch the head's full output)
            for h in range(n_heads):
                # AtP: predicted change in logit_diff if z went from clean to noise
                contrib = float(torch.sum(z_grad[0, :, h, :] * delta_z[0, :, h, :]).item())
                atp_per_head[L, h] = contrib

        saved.clear()

        # AP: actually patch each head's z from clean to noise, measure logit_diff change
        ap_per_head = np.zeros((n_layers, n_heads), dtype=np.float32)
        if args.run_ap:
            for L in range(n_layers):
                for h in range(n_heads):
                    def make_patch_hook(layer_idx, head_idx):
                        def hook_fn(activation, hook):
                            activation[:, :, head_idx, :] = noise_z[layer_idx][:, :, head_idx, :]
                            return activation
                        return hook_fn
                    patch_hook = make_patch_hook(L, h)
                    with torch.no_grad():
                        patched_logits = model.run_with_hooks(
                            clean_toks, fwd_hooks=[(z_hooks[L], patch_hook)]
                        )
                    patched_ld = logit_diff(patched_logits[0, -1, :], io_id, s_id).item()
                    ap_per_head[L, h] = patched_ld - clean_ld

        per_prompt_atp.append(atp_per_head)
        per_prompt_ap.append(ap_per_head)
        print(f"  prompt {prompt_i+1}/{len(valid_prompts)}: '{p['clean'][:40]}...' "
              f"elapsed: {time.time()-t0:.0f}s")

    # Average across prompts
    mean_ap = np.mean(np.stack(per_prompt_ap), axis=0)
    mean_atp = np.mean(np.stack(per_prompt_atp), axis=0)

    # Top heads by each method
    print(f"\n=== TOP 10 NEGATIVE EFFECT HEADS (most important for IOI) ===")
    print(f"  (most negative effect = patching this head from clean→noise most hurts IOI)")
    print(f"  {'rank':>4} {'AP':>14} {'AP val':>10}  {'AtP':>14} {'AtP val':>10}")
    flat_ap = mean_ap.flatten()
    flat_atp = mean_atp.flatten()
    ap_order = np.argsort(flat_ap)  # most negative first
    atp_order = np.argsort(flat_atp)
    for rank in range(10):
        ap_i = ap_order[rank]
        atp_i = atp_order[rank]
        ap_label = f"L{ap_i // n_heads}H{ap_i % n_heads}"
        atp_label = f"L{atp_i // n_heads}H{atp_i % n_heads}"
        print(f"  {rank+1:>4} {ap_label:>14} {flat_ap[ap_i]:>+10.3f}  "
              f"{atp_label:>14} {flat_atp[atp_i]:>+10.3f}")

    # Overall correlation
    from scipy import stats
    pearson, _ = stats.pearsonr(flat_ap, flat_atp)
    spearman, _ = stats.spearmanr(flat_ap, flat_atp)
    # Overlap of top-K heads
    for K in [5, 10, 20]:
        ap_topK = set(ap_order[:K].tolist())
        atp_topK = set(atp_order[:K].tolist())
        overlap = len(ap_topK & atp_topK)
        print(f"\n  Top-{K} overlap (AP vs AtP): {overlap}/{K} ({100*overlap/K:.0f}%)")

    print(f"\n  Overall Pearson (AP vs AtP): {pearson:.4f}")
    print(f"  Overall Spearman (AP vs AtP): {spearman:.4f}")

    Path(args.out).write_text(json.dumps({
        "model": args.model,
        "n_prompts": len(valid_prompts),
        "n_layers": int(n_layers), "n_heads": int(n_heads),
        "mean_ap_per_head": mean_ap.tolist(),
        "mean_atp_per_head": mean_atp.tolist(),
        "pearson_ap_atp": float(pearson),
        "spearman_ap_atp": float(spearman),
        "top10_ap": [(int(i // n_heads), int(i % n_heads), float(flat_ap[i])) for i in ap_order[:10]],
        "top10_atp": [(int(i // n_heads), int(i % n_heads), float(flat_atp[i])) for i in atp_order[:10]],
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
