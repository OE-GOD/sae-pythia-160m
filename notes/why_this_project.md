# Why this project

## What I'm building

A small sparse autoencoder (SAE) trained on the residual stream of a small open
language model (Pythia-160M or Qwen2.5-0.5B), with a feature catalog and one
empirical finding.

## Why this, not something else

Three categories of frontier-lab work I considered:

| | Plays to my edges? | Stretches me? | Lab fit |
|---|---|---|---|
| Evals (faithfulness / eval-gaming) | Yes — measurement-integrity | Low | METR, Apollo |
| Agents | Some — same instinct different surface | Medium | Anthropic agents, OpenAI |
| **Interpretability (SAE)** | Some — analytical rigor transfers | **High** | **Anthropic interp, Apollo, Goodfire, MATS** |

I picked SAE because:
1. Anthropic's core research direction is here. MATS pipeline runs through it.
2. Compute fits my budget (~$30-50 total). Small model, one H100-hour.
3. The toolkit (SAELens, TransformerLens) is open and well-documented.
4. At small model scales the field is genuinely undersaturated — original
   findings are reachable, not pre-claimed by frontier teams.
5. It's a real stretch — I'd be learning new territory, which is the point.

## What I'm explicitly NOT trying to do

- Replicate Anthropic's full *Scaling Monosemanticity* result. That requires
  Claude-scale compute. My target is a clean small-scale finding.
- Compete on benchmark numbers. The artifact is the catalog + finding, not a
  leaderboard placement.
- Pretend this is novel mechanistic theory. It's empirical SAE work at small
  scale, framed honestly.

## Success criteria

By end of W6:
- [ ] One trained SAE checkpoint, public on HuggingFace or GitHub
- [ ] Feature catalog (20 features, examples + labels) as a public JSON
- [ ] One blog post or LessWrong post documenting an empirical finding
- [ ] Applied to MATS next cohort
- [ ] At least one cold email sent to a known SAE researcher with the writeup

If I get 3 out of 5 of these in 6 weeks, the project succeeded regardless of
whether it lands me a job. The artifact stays valuable.
