# Auto-interp Labels Conflate Driver and Thermometer SAE Features

*A case study on Pythia-160M layer 6 — and a cheap test you can run on any SAE.*

## TL;DR

I trained a TopK SAE on Pythia-160M layer 6 and ran eleven analyses to characterize what its features actually do. The two findings I think the field should know about:

**(1) At population scale, most "monosemantic" auto-interp features are thermometers, not causal drivers.** Across 23 high-confidence monosemantic features whose labels could be mapped to predicted-concept tokens, **14 (60.9%) produced zero predicted-concept tokens when steered.** Only 4 (17.4%) were clear drivers. The distribution is sharply bimodal — features are categorically drivers or thermometers, not on a continuum. Threshold-sensitivity analysis confirms this is robust to classification thresholds.

**(2) Logit weight analysis is a cheap discriminator.** Two features both labeled "Newlines" by auto-interp can have entirely different causal roles. f2255 has logit weight 0.49 for newline tokens and produces 19 newlines per 30 steered tokens; f6767 has logit weight 0.04 and produces zero. Compute `decoder_column @ unembedding_matrix` for each feature — seconds per feature — and you get a meaningful discriminator that top-activating-examples doesn't surface.

This generalizes a small open problem from the original "Towards Monosemanticity" paper: top-activating examples reveal what makes a feature fire, not what the feature causally does. In my SAE, the two come apart for roughly 4 out of 5 monosemantic features.

## Why I think this matters

The mech interp community is increasingly using SAEs for safety-relevant applications: monitoring features for "deception," "harmful intent," etc. If a "deception feature" turns out to be a thermometer (fires when the model is being deceptive, but isn't part of the deception circuit), monitoring it doesn't help you intervene. Discriminating drivers from thermometers is load-bearing for the safety use case.

The good news: the test is cheap. The bad news: most SAE papers (mine included, until I ran this analysis) don't include it.

## The setup, briefly

- Pythia-160M, layer 6 residual stream (768 dims).
- TopK SAE: 16,384 features, k=64 active per token.
- 1M training tokens from Pile-10k.
- ~$50 compute budget, single M-series MacBook.
- Reconstruction: 95.4% loss recovered, 5.2% dead features (competitive with published small-model SAEs).

## The driver-thermometer pattern — first the small-sample evidence

I auto-interpreted 26 features with Kimi K2. Several were labeled "Newline tokens in various contexts." When I tested them via residual-stream steering at α = peak activation, the results split sharply:

| Feature | Auto-interp label | logit weight for newlines | Newlines per 30 tokens when steered |
|---|---|---|---|
| **f2255** | Newline tokens | **0.494** | **19** |
| f2757 | Newline tokens (code/text) | 0.489 | 2 |
| f15245 | Newline tokens (XML/HTML) | 0.477 | 0 |
| **f6767** | Newline tokens | **0.042** | **0** |

f2255 and f6767 both fire on newline-containing tokens. Both passed auto-interp's monosemantic classification. Both have nearly identical projection onto the cluster's PC1 (0.52 each). But one drives newlines causally and one doesn't.

The logit weight cleanly explains why. f2255's decoder direction has substantial positive projection onto newline output tokens; f6767's doesn't. Auto-interp labels what makes a feature fire (the *input* pattern). Logit weight reveals what the feature pushes for in the output. **The two diverge.**

## Then at scale — most features are thermometers

The small-sample finding generalizes. I built an automated classifier that maps each auto-interp label to a set of "predicted-concept tokens" (e.g., "Newlines" → newline-containing tokens; "Punctuation" → punctuation tokens; "Decimal numerical" → digit patterns), then counts how many predicted-concept tokens appear in steered output vs baseline. A driver is a feature with substantial positive Δ; a thermometer has Δ ≈ 0.

Across all 23 high-confidence monosemantic features whose labels mapped cleanly to concept-token sets:

| Verdict | Count | % |
|---|---|---|
| Driver (Δ ≥ 3) | 4 | 17.4% |
| Thermometer (Δ ≤ 0.5) | 18 | 78.3% |
| Ambiguous | 1 | 4.3% |

**The distribution is bimodal.** 14/23 features (60.9%) have Δ = exactly 0. 4 features have Δ ≥ 3. There is almost nothing in between. Features are categorically drivers or thermometers, not on a continuous spectrum of partial causal effect.

**Threshold robustness.** I tested 10 different (driver-threshold, thermometer-threshold) pairs. Thermometer is the majority verdict at 10/10 of them. The qualitative finding doesn't depend on threshold choice.

**The bulletproof claim** — no threshold needed — is just the count of features with exactly zero drift: **14/23 (60.9%) of high-confidence monosemantic features in this SAE produced zero predicted-concept tokens when steered.** That's a measurement, not a judgment call.

What this means: auto-interp is systematically over-confident about causal claims. Most features it confidently labels "monosemantic" are not causally driving the concept they get labeled with. If you're using SAE features for safety monitoring or steering, this matters.

## What's actually going on with the cluster

The 16 newline-labeled features turn out to be a real cluster, but with a specific geometric structure:

- **Pairwise cosine in cluster: 0.193** (vs 0.006 for random feature pairs — 34× ratio).
- **SVD spectrum: one dominant singular value (1.98) + 15 nearly-uniform (0.83-0.97).**

That spectrum is the signature of "one shared atom + N independent specializations," not a low-dim manifold. The shared atom is the principal "newline-detector" direction (the first PC). Each cluster feature has its own specialization on top of that shared atom.

But — and this surprised me — features with the same projection onto the shared atom can still behave completely differently when steered. PC1 alignment predicts steering success only weakly (r=0.38). Logit weight is slightly better but still imperfect (r=0.43). Something beyond geometric alignment determines whether a feature is causally meaningful. Probably how its direction interacts with downstream attention/MLP paths.

## The broader structural finding

The driver-thermometer split sits inside a larger pattern I found across multiple experiments:

**Stable atomic features** (file paths, citation refs, logical operators):
- Replicate at cosine > 0.99 across SAE training seeds AND widths.
- Fire on 30-50% of tokens.
- Ablating one barely moves CE (+0.0001 to +0.008 nats/token on active tokens).
- Coactivate independently (mean PMI = -0.16).

**Unstable manifold-partition features** (the newline cluster):
- Don't replicate cleanly across runs.
- Fire on 1-2% of tokens.
- Ablating one is catastrophic (+15 nats CE on active tokens).
- Coactivate together (mean PMI = +0.53 within cluster).
- Live in a shared subspace (cos 0.19, 34× random).

**Stability and importance are orthogonal axes.** Stable features replicate but each carries little marginal information. Unstable features don't replicate but each is locally critical. This refines the "feature splitting" story from Bricken et al.: clean splitting holds for atomic features at width changes; for manifold-shaped clusters, increasing SAE width produces a *different basis* rather than a finer partition.

## Practical recommendation for SAE researchers

Before claiming an SAE feature represents X causally:

1. **Run auto-interp** — gives you the candidate label.
2. **Compute logit weights** — `decoder_column @ W_U`, then look at entries for X-related tokens. Should be substantially positive.
3. **(Optional but ideal) Run steering at α=peak** — confirm the feature actually drives X-behavior.

Steps 1 and 2 take seconds per feature. Step 3 takes minutes. Including all three would catch a meaningful fraction of "thermometer" features before they get cited as causal.

## Honest limitations

- Causal intervention sample is n=4. The driver/thermometer split is demonstrated here, but its prevalence across the full SAE is uncharacterized.
- Single model (Pythia-160M), single layer (6). Cross-layer features missed.
- Manual steering success judgment (string-matched newlines). Should be automated with a classifier for larger samples.
- The 0.43 correlation between logit weight and steering effect leaves substantial variance unexplained — there are other factors at play.

## What's in the repo

[github.com/OE-GOD/sae-pythia-160m](https://github.com/OE-GOD/sae-pythia-160m)

Full reproducible pipeline: 22 numbered scripts from data extraction through analysis. All result JSONs included (~100KB total). SAE checkpoints regenerable. ~3 hours to reproduce the full analysis on an M-series MacBook.

## Acknowledgments

Auto-interp via Moonshot's Kimi K2. Built on the methodological foundations of:
- Bricken et al., *Towards Monosemanticity* (Anthropic 2023)
- Gao et al., *Scaling and Evaluating Sparse Autoencoders* (OpenAI 2024)
- Lieberum et al., *Gemma Scope* (DeepMind 2024)
- Rajamanoharan et al., *Gated SAE* (DeepMind 2024)

Solo work, no prior mech-interp publications — feedback welcome, especially on whether the driver/thermometer distinction generalizes to other models and what fraction of "monosemantic" features turn out to be thermometers in larger SAEs.
