# Sparse Autoencoder on Pythia-160M Layer 6

Small-scale mechanistic-interpretability study: trained a TopK sparse autoencoder (SAE) on the residual stream at layer 6 of Pythia-160M, evaluated it five ways, looked at what features it found.

**This is an in-progress portfolio piece.** Three of five planned tracks complete; two remaining. Final writeup pending.

## Headline numbers (16k-feature SAE, k=64)

| Metric | Value |
|---|---|
| Variance explained | **98.43%** |
| L0 (active features per token) | **64.0** (TopK enforces) |
| Dead features | **5.2%** (855 / 16,384) |
| CE delta (info loss) | **0.276 nats/token (0.40 bits/token)** |
| **Loss recovered vs. zero-ablation** | **95.4%** |
| Auto-interp monosemantic rate | **88.5%** (23/26 features labeled) |

The 95.4% loss-recovered number is the rigorous reconstruction metric (replace the model's layer-6 with the SAE's reconstruction; measure how much language-modeling CE degrades on held-out text). This is competitive with the strong end of the published small-model SAE band (Anthropic *Towards Monosemanticity* ~90%, DeepMind *Gemma Scope* 80-95%).

## Setup

- **Model:** Pythia-160M (160M-param open-source transformer)
- **Layer:** 6 (middle residual stream, hook `blocks.6.hook_resid_post`)
- **Training data:** 1M tokens from `NeelNanda/pile-10k` (skipping training docs for held-out eval)
- **SAE architecture:** TopK linear-encoder SAE with K = 64 active features per token, decoder columns renormalized after each step, b_dec initialized to the data mean
- **Training:** Adam, lr=1e-3, batch 4096, 20,000 steps, no L1 penalty (TopK gives hard sparsity)
- **Hardware:** Apple M-series via PyTorch MPS

## Why not vanilla L1?

We tried vanilla L1 first. A 20× sweep of λ ∈ {0.005, 0.02, 0.1} kept L0 stuck at ~1300 (target was 30-100). The L1 penalty shrinks feature magnitudes but doesn't actually zero them — the textbook "L1 shrinkage" failure mode. Switched to TopK; L0 becomes exact by construction. Documented in the code history.

## Tracks (research questions addressed)

Each track corresponds to a published open problem in mech interp:

| Track | Problem | Status | Result |
|---|---|---|---|
| T3 — CE delta evaluation | #4 | ✅ Done | 95.4% loss recovered, ΔCE = 0.276 nats/token |
| T5 — Auto-interp via Kimi K2 | #7 | ✅ Done | 26 features labeled, 88.5% monosemantic; surfaced feature-splitting in newline variants |
| T1 — Multi-width sweep (4k / 16k / 64k) | #3, #5, #6 | ✅ Done | Pareto curve flat above 4k; 64% dead features at 64k width; merging signal clear, splitting muted |
| T2 — TopK vs JumpReLU | #1, #2 | ⏳ Pending | — |
| T4 — Replication across seeds | #11 | ⏳ Pending | — |

## What the SAE found

**Confirmed monosemantic features** (sample from 26 auto-labeled):

| ID | Concept | Auto-interp confidence |
|---|---|---|
| f1989 | Decimal points in numerical contexts | high |
| f5196 | Exponentiation notation in math | high |
| f5747 | File paths and directory structures | high |
| f7448 | Newline tokens inside code blocks | high |
| f10045 | Subword BPE continuations of multi-piece words | high |
| f12117 | BibTeX/LaTeX citation references | high |
| f12697 | Logical operators and negation symbols | high |

**Width-invariant features** (same direction in 4k, 16k, AND 64k SAEs at cos > 0.99):
- f12117 (citation refs)
- f5747 (file paths)

These are the strongest candidates for "real" features of the model — their decoder direction is stable regardless of SAE capacity.

**Feature splitting / merging:**
- One 4k feature (f2175) is the best cos-sim match for 5+ different 16k newline-variant features → merging at smaller width is clear
- At 64k, the wider SAE largely found a *different* basis rather than cleanly splitting 16k features → splitting is muted in practice

## Honest limitations

- Single layer (layer 6) only — per-layer myopia (Problem #8) not addressed
- 1M training tokens at 16k SAE width; the model has 100B+ training tokens of complexity unrepresented
- Auto-interp by Kimi K2 has known weaknesses: 10+ features all labeled "newline tokens" though clearly distinct by their snippets — the labeling collapses fine-grained splits the SAE found
- No causal-intervention validation of features (Problem #10) — features are correlationally interpretable, not causally validated
- A linear-encoder SAE is computationally matched to a single MLP layer; this constrains the class of features findable, and within that class some features may be reconstruction-correlated rather than model-used

## Layout

```
sae_project/
├── README.md                      this file
├── code/
│   ├── 00_verify.py               load Pythia, run one forward pass
│   ├── 01_extract_activations.py  build 1M-token activation cache
│   ├── 02_train_sae.py            vanilla L1 SAE (kept for reference)
│   ├── 02b_train_sae_topk.py      TopK SAE — what we actually use
│   ├── 03_eval_sae.py             MSE / L0 / dead-feature eval
│   ├── 04_feature_analysis.py     find top-K max-activating examples per feature
│   ├── 05_inspect_features.py     generate human-readable feature report
│   ├── 06_ce_delta.py             T3 — CE delta evaluation
│   ├── 07_auto_interp.py          T5 — Claude/Kimi/OpenAI auto-interp
│   └── 08_width_comparison.py     T1 — Pareto + splitting analysis
├── data/
│   ├── feature_catalog.json       26 labeled features (T5 output)
│   ├── width_comparison.json      Pareto + splitting numbers (T1 output)
│   └── top_features.npz           raw top-K activations per feature
└── notes/
    ├── feature_inspection.md      human-readable report on 80 highlight features
    └── why_this_project.md        project rationale
```

**Not in repo** (too large or reproducible):
- `data/feature_examples.json` (43.7 MB) — regenerable via `04_feature_analysis.py`
- `data/acts_layer6.npy` (1.5 GB) — regenerable via `01_extract_activations.py`
- `data/token_stream.npy` (4 MB) — regenerable
- `checkpoints/*.pt` (200 MB - 800 MB) — regenerable

To reproduce: run the pipeline below.

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch transformer_lens datasets numpy openai

# 1. extract 1M-token activations  (~80s on Mac MPS)
.venv/bin/python code/01_extract_activations.py --n-tokens 1_000_000

# 2. train TopK SAE k=64, 16k features  (~50 min on Mac, faster on H100)
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 \
    --out checkpoints/sae_layer6_topk64_full.pt --log-every 1000

# 3. eval: MSE, L0, dead features
.venv/bin/python code/03_eval_sae.py --ckpt checkpoints/sae_layer6_topk64_full.pt

# 4. CE delta (T3)
.venv/bin/python code/06_ce_delta.py --ckpt checkpoints/sae_layer6_topk64_full.pt

# 5. feature analysis: top-K examples per feature  (~2.5 min)
.venv/bin/python code/04_feature_analysis.py

# 6. inspect manually
.venv/bin/python code/05_inspect_features.py
open notes/feature_inspection.md

# 7. auto-interp (T5)
export MOONSHOT_API_KEY=...    # or ANTHROPIC_API_KEY / OPENAI_API_KEY
.venv/bin/python code/07_auto_interp.py --provider kimi --model moonshot-v1-32k --n-features 50

# 8. multi-width sweep (T1)  — train SAEs at 4k and 64k widths
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 --n-features 4096 \
    --out checkpoints/sae_layer6_topk64_w4k.pt
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 --n-features 65536 \
    --out checkpoints/sae_layer6_topk64_w64k.pt
.venv/bin/python code/08_width_comparison.py
```

## Context

This project demonstrates competence in the protocol Anthropic's interpretability team uses on frontier models, applied at 1/100,000 the scale on a small open model. Targeted at MATS / Apollo / Goodfire applications.

Done as solo work, on a MacBook (Apple M-series MPS), under a $50 compute budget. The author has no prior mech-interp publications — this is the first.

## References

- Bricken et al., *Towards Monosemanticity* (Anthropic 2023)
- Templeton et al., *Scaling Monosemanticity* (Anthropic 2024)
- Gao et al., *Scaling and Evaluating Sparse Autoencoders* (OpenAI 2024) — TopK SAE
- Lieberum et al., *Gemma Scope* (DeepMind 2024) — CE delta methodology
- Rajamanoharan et al., *Improving Dictionary Learning with Gated SAEs* (DeepMind 2024)

## License

MIT.

## Contact

irving46764@gmail.com / [github.com/OE-GOD](https://github.com/OE-GOD)
