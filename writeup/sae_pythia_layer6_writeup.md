# Auto-interp Labels Conflate Driver and Thermometer Features: A Case Study on Pythia-160M

**A nine-experiment characterization of TopK SAE features in Pythia-160M layer 6, with a methodological finding about how auto-interpretation can mislead causal interpretation.**

---

## TL;DR

- Trained a TopK SAE (16k features, k=64) on Pythia-160M residual stream at layer 6. Standard reconstruction quality: 95.4% loss recovered, 5.2% dead features, 98.4% variance explained.
- **Headline finding:** SAE features in this model split into two distinct kinds — **stable atomic features** (replicate across runs, low individual causal impact) and **unstable manifold-partition features** (don't replicate, large causal impact per active token). Stability and importance are orthogonal axes, not the same axis.
- **Methodological finding:** Within the unstable cluster, features that look identical to auto-interp can have entirely different causal roles. Two features both labeled "Newlines" have logit weights of 0.49 (driver) and 0.04 (thermometer); when force-activated, one produces 19 newlines per 30 tokens, the other produces zero. **Auto-interp labels conflate driver and thermometer features; logit weight analysis discriminates them cheaply.**
- The "one shared atom plus N specializations" geometric story (from SVD on the newline cluster) explains the cluster structure but does not fully predict which features are drivers vs. thermometers. Causal effect requires both alignment with the shared direction AND alignment with downstream output paths.

---

## Motivation

The standard mech-interp workflow for SAE features involves:

1. Train an SAE on a layer's activations.
2. For each feature, find top-activating examples.
3. Have a human or LLM label the feature based on those examples.
4. Treat the label as a hypothesis about what the feature "represents."

This pipeline has a known weakness: the label is generated from *what makes the feature fire*, not from *what the feature causally does*. A feature can correlate with concept X (so it gets labeled X) without driving concept X in the model's output. The community has flagged this — Bricken et al. (2023) explicitly distinguish "the feature fires on" from "the feature's downstream effects" — but few small-scale studies systematically test the gap.

This project does that test, on a small open model, with a complete pipeline. The result: in Pythia-160M layer 6, the gap is real, measurable, and concentrated in specific feature clusters.

---

## Setup

| | Value |
|---|---|
| Base model | Pythia-160M (160M-param open transformer) |
| Hook point | `blocks.6.hook_resid_post` (middle residual stream) |
| Training data | 1M tokens from `NeelNanda/pile-10k` |
| SAE width N | 16,384 features (8× the residual dim of 768) |
| Sparsity | TopK with k=64 active features per token |
| Loss | L2 reconstruction only (no L1 — TopK gives hard sparsity) |
| Optimizer | Adam, lr=1e-3, batch 4096, 20k steps |
| Hardware | Apple M-series via PyTorch MPS, ~$50 compute budget |

Standard best practices: decoder columns renormalized to unit length after every optimizer step (prevents the optimizer from gaming the sparsity metric by rescaling decoder weights). `b_dec` initialized to the data mean (centering trick from Bricken et al.). Tied encoder/decoder initialization for faster convergence.

**Why TopK and not L1.** Vanilla L1 hit textbook shrinkage failure. A 20× hyperparameter sweep of λ ∈ {0.005, 0.02, 0.1} kept L0 stuck at ~1300 (target was 30–100): the L1 penalty distorted feature magnitudes without truly zeroing them. TopK enforces L0=k by construction. Documented in the open-source code history.

**Reconstruction quality** (held-out 100k tokens):

| Metric | Value |
|---|---|
| Variance explained | 98.43% |
| L0 (active features per token) | 64.0 (TopK enforced) |
| Dead features | 5.2% (855 / 16,384) |
| ΔCE (information loss) | 0.276 nats/token |
| Loss recovered vs. zero-ablation | 95.4% |

The 95.4% loss-recovery is competitive with the published small-model band — Bricken et al. report ~90% on their 1-layer toy; DeepMind's Gemma Scope reports 80–95% across Gemma 2 layers.

Auto-interp labeled 26 features via the Kimi K2 API, achieving 88.5% monosemantic confidence (23/26 features labeled as monosemantic with high/medium confidence).

---

## The Nine Experiments

The interpretive analysis comprises nine experiments, each addressing a distinct question. Numbers in brackets identify the open problems from Anthropic and DeepMind work that each experiment addresses.

| # | Experiment | Question | Result |
|---|---|---|---|
| 1 | Width comparison [#3, #5, #6] | Right SAE size? | 4k → 16k → 64k: VE saturates at 16k; 64k has 64% dead features |
| 2 | CE delta [#4] | How much of the MLP is preserved? | 95.4% loss recovered |
| 3 | Auto-interp [#7] | Are features interpretable? | 88.5% monosemantic by LLM labeling |
| 4 | Coactivation PMI | Do features fire together? | Unstable features: PMI +0.53; stable: −0.16 |
| 5 | Ablation | Per-feature importance? | Stable: ΔCE ≈ 10⁻³; unstable: ΔCE ≈ +15 nats |
| 6 | Cluster geometry | Do feature directions cluster? | Newline cluster cos 0.19; random 0.006 (34× ratio) |
| 7 | Splitting (rigorous re-test) | Do features split across widths? | Atomic features: yes (cos > 0.99); manifold features: no |
| 8 | Causal intervention | Do features causally steer? | 2 of 4 high-confidence features pass; 1 fails despite identical label |
| 9 | Logit weights | What predicts causal effect? | Drivers vs thermometers visible; correlation with steering r = 0.43 |

The pattern across these experiments converges on one finding: **two distinct kinds of features exist in this SAE.**

---

## Finding 1: Two kinds of features (the atom/manifold distinction)

Cross-referencing the experiments:

**Stable atomic features** (examples: f5747 file paths, f12117 citation refs, f12697 logical operators):
- Replicate near-perfectly across SAE training seeds (cosine ≥ 0.99 in cross-run match).
- Replicate near-perfectly across SAE widths (4k feature ↔ 16k feature ↔ 64k feature, all cos ≥ 0.99).
- Fire on 30–50% of tokens.
- Ablation has negligible impact (ΔCE ≈ 10⁻³–10⁻² nats/token on active tokens).
- Coactivate independently from each other (mean PMI = −0.16).

**Unstable manifold-partition features** (examples: ~16 features labeled "Newline tokens in various contexts"):
- Don't replicate cleanly across seeds (best cross-run match cos < 0.5 typically).
- Don't replicate cleanly across widths (best 64k match for any 16k newline feature: cos ≈ 0.35–0.50).
- Fire on 1–2% of tokens.
- Ablation has catastrophic impact (ΔCE = +15 nats on active tokens).
- Coactivate together (mean PMI = +0.53 within the cluster, vs −0.16 within the stable group).
- Share a common subspace: pairwise cosine between decoder columns is 0.19, vs 0.006 for random pairs (34× ratio).

**Interpretation:** atomic features represent clean, independent concepts the model uses everywhere. Manifold features are partition fragments of a higher-dimensional underlying structure the model relies on heavily — the SAE keeps finding the same region of activation space but partitions it differently across training runs.

This refines the "feature splitting" picture from Bricken et al. (2023): clean splitting holds for atomic features in our data, but for manifold-shaped clusters, increasing SAE width produces a *different basis* rather than refining the existing one.

## Finding 2: The newline cluster is "one shared atom plus N specializations"

The newline cluster cosine of 0.19 looks weak, but in 768-dim space it's 34× the random baseline. SVD on the 16 newline decoder columns reveals the underlying structure:

| Singular value rank | Value | Variance captured |
|---|---|---|
| 1 | 1.98 | 24.4% |
| 2 | 0.97 | 5.9% |
| ... | ~0.83–0.97 | ~5–6% each |
| 16 | 0.82 | 4.2% |

Compare to random unit vectors:

| Singular value rank | Random value |
|---|---|
| 1 | 1.10 |
| 2 | 1.09 |
| ... | ~0.93–1.07 |
| 16 | 0.89 |

The pattern: **one dramatically dominant singular value (1.98 vs ~0.95 for the rest)**, then 15 nearly-uniform residuals. This is the spectral signature of "one shared common direction + N nearly-independent residual directions." Not a low-dim manifold; not random unit vectors; something specific.

**Interpretation:** the newline cluster is one shared "newline-detector" direction (PC1, capturing 24% of cluster variance) plus 15 nearly-orthogonal context-specific specializations (newlines in code, newlines in CSS/HTML, newlines in academic text, etc.). The shared direction is what gives the small but significant pairwise cosine of 0.19; the specializations explain why cosines aren't higher.

## Finding 3: Within the cluster, drivers and thermometers are not distinguishable by auto-interp alone

This is the methodological contribution.

We tested 4 high-confidence monosemantic features via causal intervention (force the feature to fire at α = peak activation, observe model output on neutral prompts). Two passed; one failed despite having the same auto-interp label as a passing feature.

Logit weight analysis explains the discrepancy. For each feature, we compute `decoder_col @ unembedding_matrix`, then look at the entries for newline-containing tokens. This number measures how much activating the feature pushes the model's output distribution toward newline tokens.

Cross-reference table:

| Feature | Label (auto-interp) | logit weight for newlines | PC1 alignment | Newlines induced (per 30 tokens) |
|---|---|---|---|---|
| **f2255** | "Newlines" | **0.494** | 0.522 | **19** |
| f2757 | "Newlines (code/text)" | 0.489 | 0.469 | 2 |
| f15245 | "Newlines (XML/HTML/CSS)" | 0.477 | 0.472 | 0 |
| f14584 | "Newlines (code/markup)" | 0.360 | 0.462 | 0 |
| f15230 | "Newlines" | 0.237 | 0.527 | 0.5 |
| **f6767** | "Newlines" | **0.042** | 0.525 | **0** |

f2255 and f6767 both labeled "Newlines in various contexts" by auto-interp; both have similar PC1 alignment (0.52); both fire on tokens containing newlines (that's why auto-interp clustered them).

But:
- **f2255 has logit weight 0.494** for newline tokens. Steering produces 19 newlines.
- **f6767 has logit weight 0.042** for newline tokens. Steering produces 0 newlines.

f6767 is a **thermometer feature**: it correlates with newline tokens (so auto-interp called it "newlines") but its decoder direction doesn't push toward newlines in the output. f2255 is a **driver feature**: it both correlates with newlines and causally drives newline production.

**Methodological recommendation:** before claiming an SAE feature represents X causally, compute its logit weight for X-tokens (a cheap O(d_model × vocab_size) operation per feature) and confirm it's substantially above zero. Auto-interp alone produces labels that conflate drivers and thermometers; logit weight discriminates them at scale.

## Finding 4: Random-direction control validates the steering experiments

A natural skeptic's response to causal intervention: maybe ANY large perturbation to the residual stream causes drift, not specifically feature directions. We control by steering with random unit vectors at matched magnitude.

Real-feature steering produced the predicted output (newlines, whitespace) consistently across prompts.

Random-direction steering (three independent random vectors per feature) produced generic garbage (word repetition, BPE fragments) — never specifically newlines. The drift is direction-specific, not a generic "any-perturbation" effect.

This rules out the "any large intervention causes drift" alternative explanation and strengthens the causal claim for the driver features.

---

## Honest limitations

1. **Causal intervention sample is small (n=4).** A robust evaluation would test 30+ features and report a quantitative success rate. The driver/thermometer distinction is demonstrated in this small sample, but its frequency across the SAE is uncharacterized.

2. **Manual judgment of steering success.** The "19 newlines vs 0 newlines" count uses simple string matching. A more rigorous version would use a classifier (e.g., LLM judge) and report graded success.

3. **Single layer studied (layer 6).** Cross-layer features (per the crosscoder literature) are not captured. Per-layer myopia is a real limitation.

4. **Single model.** Whether the driver/thermometer pattern holds at frontier scale (Llama 3 8B, Gemma 27B) is unknown.

5. **Replication track (T4) and architecture comparison (T2 — TopK vs JumpReLU) are pending.** Two of the originally planned tracks remain incomplete.

6. **Auto-interp by Kimi K2** has documented weaknesses: it clustered 10+ distinct features all as "newline tokens" because they all fire on newline tokens. The discrimination among them required follow-up analysis.

7. **Logit weight is itself a linear approximation** (ignores layer norm nonlinearity and middle-layer attention dynamics). The 0.43 correlation between logit weight and steering effect leaves substantial unexplained variance, indicating other factors matter.

---

## Future directions

Each limitation suggests a concrete next experiment:

- **Scale the causal intervention sample to 30+ features**, with an automated success classifier. Report quantitative driver/thermometer rates.
- **Run T2 (TopK vs JumpReLU)** to test whether the driver/thermometer split is architecture-dependent.
- **Run T4 (replication across seeds)** with formal cross-run matching. Currently the stable/unstable distinction is binary; T4 would put it on rigorous footing.
- **Investigate why f2255 is a driver and f6767 is a thermometer when both fire on identical-looking newline contexts.** Likely candidates: encoder pattern differences, downstream attention interactions, or differential interaction with the model's layer-7+ pathways.
- **Apply logit weight discrimination to a larger set of monosemantic features** to estimate the prevalence of thermometer features in real SAEs.
- **Test the framework on a multi-layer SAE setup (crosscoder)** to see whether driver/thermometer status transfers across layers.

---

## Reproducibility

All experiments run on a MacBook (Apple M-series via PyTorch MPS) under a $50 compute budget. Full pipeline reproducible from `code/00_verify.py` through `code/17_logit_weights.py` in the repository.

Key dependencies: `torch`, `transformer_lens`, `datasets`, `numpy`, `openai` (for auto-interp; or `moonshot` for Kimi K2).

Training the 16k TopK SAE: ~50 minutes on M-series, much faster on H100. Full analysis pipeline (all nine experiments): ~3 hours total on M-series.

Code: [github.com/OE-GOD/sae-pythia-160m](https://github.com/OE-GOD/sae-pythia-160m) (license MIT).

---

## Acknowledgments

This work was done solo, as a portfolio piece, with no prior mech-interp publications. The analysis builds heavily on the methodological foundations laid by:

- Bricken et al., *Towards Monosemanticity* (Anthropic 2023)
- Templeton et al., *Scaling Monosemanticity* (Anthropic 2024)
- Gao et al., *Scaling and Evaluating Sparse Autoencoders* (OpenAI 2024)
- Rajamanoharan et al., *Improving Dictionary Learning with Gated Sparse Autoencoders* (DeepMind 2024)
- Lieberum et al., *Gemma Scope* (DeepMind 2024)

Auto-interp via Moonshot's Kimi K2 (chosen for its 32k-context efficiency at low cost).

---

## Contact

irving46764@gmail.com / [github.com/OE-GOD](https://github.com/OE-GOD)
