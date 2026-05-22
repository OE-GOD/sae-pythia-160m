# Sparse Autoencoder on Pythia-160M Layer 6

**Reconstruction Is Not Replication** — small-scale mechanistic-interpretability study that cross-validates a TopK sparse autoencoder (SAE) across width, random seed, and architecture on the residual stream at layer 6 of Pythia-160M.

**Headline finding:** standard SAE quality metrics (variance explained $0.984$, CE-delta loss recovered $0.954$, auto-interp monosemanticity $0.885$) overstate feature reliability by approximately $8\times$. Only **$2.14\%$ of features ($351$ of $16{,}384$)** are stable across all four cross-validation conditions.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) — 11 pages, *Reconstruction Is Not Replication: Cross-Validating Sparse Autoencoder Features on Pythia-160M*

## The 6-axis scorecard

Same SAE, six independent quality axes:

```
Reconstruction (var. explained)         [████████████████████]  0.984
Predictive preservation (CE-delta)      [███████████████████░]  0.954
Capacity utilization (alive features)   [██████████████████░░]  0.948
Auto-interp monosemanticity rate        [█████████████████░░░]  0.885
Cross-seed stability (cos > 0.9)        [███░░░░░░░░░░░░░░░░░]  0.199
Cross-architecture stability (cos > 0.9)[██░░░░░░░░░░░░░░░░░░]  0.106
```

**88-percentage-point gap** between the strongest and weakest axis. The standard suite cannot be the only quality claim attached to an SAE.

## Tracks (empirical results)

Each track addresses a published open problem in mech interp:

| Track | Open problem | Status | Headline result |
|---|---|---|---|
| T1 — Multi-width sweep ($4$k / $16$k / $64$k) | Choosing $N$, splitting | ✅ | Pareto saturates near $N \approx 4$k; $63.6\%$ dead at $N = 65$k |
| T3 — CE-delta evaluation | Loss recovery | ✅ | $95.4\%$ loss recovered, $\Delta$CE $= 0.276$ nats/token |
| T4 — Cross-seed replication | Stability (#11) | ✅ | Only $19.9\%$ of features at cos $> 0.9$ across seeds |
| T5 — Auto-interp catalog (Kimi K2) | Auto-interp (#7) | ✅ | $26$ features labeled, $88.5\%$ monosemantic — but most fail replication |
| T6 — Matryoshka SAE replication | Architecture (#1, #2) | ✅ | $10.6\%$ at cos $> 0.9$ across architectures |
| T2 — TopK vs JumpReLU | Dead features (#1, #2) | ⏳ Pending | — |

## Tools we contribute

| Build | What it does | Output |
|---|---|---|
| **#1 — Multi-metric scorecard** | Aggregates the six axes above into one quality profile for any SAE checkpoint | `*.scorecard.md` |
| **#2 — Feature navigator** | Per-feature stability profile across all comparison SAEs; identifies the fully stable subset | `data/feature_navigator.json`, `notes/stable_features.md` |
| **#3 — Co-activation analysis** | PMI distribution among stable vs unstable features; surfaces candidate circuit-shaped pairs | `data/coactivation_analysis.json`, `notes/stable_feature_circuits.md` |

## The 3 fully stable features

Of $26$ features auto-labeled monosemantic, only $3$ survive all four cross-validation conditions at cos $> 0.9$:

| Feature | Concept | Min cos across comparisons |
|---:|---|---:|
| `f12117` | BibTeX/LaTeX citation references | $0.997$ |
| `f5747` | File paths and directory structures | $0.997$ |
| `f12697` | Logical operators and negation symbols | $0.977$ |

These are the strongest "real feature" candidates — their decoder direction replicates regardless of SAE width, seed, or architecture.

## Setup

- **Model:** Pythia-160M (160M params, 12 layers, $d_{\text{model}} = 768$)
- **Layer target:** 6 (middle residual stream, `blocks.6.hook_resid_post`)
- **Training corpus:** 1M tokens from `NeelNanda/pile-10k`
- **Reference SAE:** TopK $k = 64$, $N = 16{,}384$ features, 20k steps, Adam lr $= 10^{-3}$
- **Comparison SAEs:** 4k width, 64k width, second random seed, Matryoshka $k_{\text{levels}} = \{16, 64, 256\}$
- **Hardware:** Apple Silicon (MPS), $\$50$ compute budget (actual spend: $\$0$)

## Why not vanilla L1?

We tried L1 first. A $20\times$ sweep of $\lambda \in \{0.005, 0.02, 0.1\}$ kept $L_0$ stuck at $\sim 1300$ (target band: $30$-$100$). The L1 penalty is indifferent between "one feature at strength $S$" and "$S$ features at strength $1$" — the textbook L1-shrinkage failure mode. Switched to TopK; $L_0 = k$ becomes exact by construction. Detailed in the paper (Appendix A).

## Layout

```
sae_project/
├── README.md                            this file
├── paper/
│   ├── main.tex                         LaTeX source (11 pages)
│   ├── main.pdf                         compiled paper
│   └── sae_pythia_arxiv.zip             arXiv submission bundle
├── code/
│   ├── 00_verify.py                     load Pythia + sample activation
│   ├── 01_extract_activations.py        build 1M-token activation cache
│   ├── 02_train_sae.py                  vanilla L1 SAE (reference baseline)
│   ├── 02b_train_sae_topk.py            TopK SAE (the reference)
│   ├── 02c_train_sae_matryoshka.py      Matryoshka SAE (T6)
│   ├── 03_eval_sae.py                   MSE / L0 / dead-features eval
│   ├── 04_feature_analysis.py           top-K max-activating examples per feature
│   ├── 05_inspect_features.py           human-readable feature report
│   ├── 06_ce_delta.py                   T3 — CE-delta evaluation
│   ├── 07_auto_interp.py                T5 — Claude/Kimi/OpenAI auto-interp
│   ├── 08_width_comparison.py           T1 — Pareto + splitting/merging
│   ├── 09_replication_analysis.py       T4 — cross-seed replication
│   ├── 10_matryoshka_analysis.py        T6 — cross-architecture analysis
│   ├── 11_sae_scorecard.py              Build #1 — 6-axis scorecard
│   ├── 12_feature_navigator.py          Build #2 — per-feature stability profile
│   └── 13_coactivation_analysis.py      Build #3 — stable-feature co-activation
├── data/
│   ├── feature_catalog.json             T5 labeled features (Kimi K2)
│   ├── width_comparison.json            T1 Pareto + splitting
│   ├── replication_analysis.json        T4 cross-seed
│   ├── matryoshka_analysis.json         T6 cross-arch
│   ├── feature_navigator.json           Build #2 per-feature profiles
│   └── top_features.npz                 raw top-K activations
└── notes/
    ├── feature_inspection.md            human-readable report (80 features)
    ├── stable_features.md               Build #2 shortlist
    └── why_this_project.md              project rationale
```

**Not in repo** (regenerable):
- `data/feature_examples.json` (43.7 MB) — via `04_feature_analysis.py`
- `data/acts_layer6.npy` (1.5 GB) — via `01_extract_activations.py`
- `checkpoints/*.pt` (200 MB - 800 MB each) — via the training scripts

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch transformer_lens datasets numpy openai anthropic

# 1. extract 1M-token activations  (~80s on Mac MPS)
.venv/bin/python code/01_extract_activations.py --n-tokens 1_000_000

# 2. train TopK SAE k=64, 16k features (reference)
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 \
    --out checkpoints/sae_layer6_topk64_full.pt --log-every 1000

# 3. eval reference SAE
.venv/bin/python code/03_eval_sae.py --ckpt checkpoints/sae_layer6_topk64_full.pt

# 4. CE delta (T3)
.venv/bin/python code/06_ce_delta.py --ckpt checkpoints/sae_layer6_topk64_full.pt

# 5. feature analysis + auto-interp (T5)
.venv/bin/python code/04_feature_analysis.py
export MOONSHOT_API_KEY=...
.venv/bin/python code/07_auto_interp.py --provider kimi --model moonshot-v1-32k --n-features 50

# 6. multi-width sweep (T1)
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 --n-features 4096 \
    --out checkpoints/sae_layer6_topk64_w4k.pt
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 --n-features 65536 \
    --out checkpoints/sae_layer6_topk64_w64k.pt
.venv/bin/python code/08_width_comparison.py

# 7. cross-seed (T4)
.venv/bin/python code/02b_train_sae_topk.py --steps 20000 --k 64 --seed 42 \
    --out checkpoints/sae_layer6_topk64_seed42.pt
.venv/bin/python code/03_eval_sae.py --ckpt checkpoints/sae_layer6_topk64_seed42.pt
.venv/bin/python code/09_replication_analysis.py

# 8. Matryoshka SAE (T6)
.venv/bin/python code/02c_train_sae_matryoshka.py --steps 20000 --k-levels 16,64,256 \
    --out checkpoints/sae_layer6_matryoshka.pt
.venv/bin/python code/10_matryoshka_analysis.py

# 9. Build #1 scorecard
.venv/bin/python code/11_sae_scorecard.py

# 10. Build #2 feature navigator
.venv/bin/python code/12_feature_navigator.py

# 11. Build #3 co-activation
.venv/bin/python code/13_coactivation_analysis.py
```

## Honest limitations

- Single layer, single model — claim is methodological, not "the 2.14% number is universal"
- Cross-seed measured at one pair (reference + seed-42); more seeds may yield lower stability
- No causal-intervention validation — stability is a cheap proxy, not a substitute
- Auto-interpretation by Kimi K2 collapses fine-grained splits we know exist; a stronger labeler would likely raise monosemanticity rate further without raising cross-validation stability — *widening* the gap, not closing it
- Linear-encoder TopK is computationally matched to a single MLP layer; stronger SAE variants could find features the model itself cannot use, exacerbating the cross-validation gap

## Context

Solo work by a community-college student, in 4 weeks, on a MacBook (Apple M-series MPS), under a $50 compute budget. Targeted at MATS / Apollo / Goodfire / EleutherAI SOAR applications.

The author has no prior mech-interp publications — this is the first.

## References

- Bricken et al., *Towards Monosemanticity* (Anthropic 2023)
- Templeton et al., *Scaling Monosemanticity* (Anthropic 2024)
- Gao et al., *Scaling and Evaluating Sparse Autoencoders* (OpenAI 2024) — TopK
- Rajamanoharan et al., *Gated SAEs* / *JumpReLU SAEs* (DeepMind 2024)
- Lieberum et al., *Gemma Scope* (DeepMind 2024) — CE-delta methodology
- Bussmann et al., *Matryoshka Sparse Autoencoders* (2025) — nested-K architecture
- Biderman et al., *Pythia* (2023)

## License

MIT.

## Contact

[irving46764@gmail.com](mailto:irving46764@gmail.com) · [github.com/OE-GOD](https://github.com/OE-GOD)
