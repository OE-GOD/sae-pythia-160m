# Stable-Feature Co-Activation Analysis

Computed on 200,000 held-out tokens from the SAE training stream.
Tracked: 351 fully-stable features + 351 matched-unstable features.

## PMI distribution by pairing

PMI(i,j) = log( P(i,j) / (P(i) · P(j)) ). Higher = features fire together more than chance.

| Pairing | n pairs | mean | p90 | p99 | frac > 1 | frac > 2 | frac > 5 |
|---|------:|-----:|----:|----:|---------:|---------:|---------:|
| stable × stable | 90,146 | -0.158 | 0.938 | 2.423 | 0.0900 | 0.0190 | 0.0001 |
| unstable × unstable | 16,142 | 0.530 | 2.561 | 4.802 | 0.3295 | 0.1567 | 0.0076 |
| stable × unstable (cross) | 47,630 | 0.016 | 1.544 | 3.344 | 0.1856 | 0.0579 | 0.0010 |

**Mean PMI gap (stable-stable vs unstable-unstable): -0.688.**

Unstable-unstable PMI is higher than stable-stable. Stable features are MORE INDEPENDENT than unstable ones — they are atomic, not compositional.

## Top stable-stable pairs by PMI

| f_a | label_a | f_b | label_b | PMI | P(a,b) |
|----:|---------|----:|---------|----:|-------:|
| f2034 | ? | f10306 | ? | 5.500 | 0.00018 |
| f8020 | ? | f12413 | ? | 5.314 | 0.00004 |
| f8635 | ? | f10155 | ? | 5.054 | 0.00008 |
| f480 | ? | f9905 | ? | 5.021 | 0.00001 |
| f10312 | ? | f11850 | ? | 5.012 | 0.00005 |
| f5820 | ? | f6421 | ? | 4.946 | 0.00006 |
| f1040 | ? | f1789 | ? | 4.874 | 0.00076 |
| f4357 | ? | f9408 | ? | 4.784 | 0.00131 |
| f480 | ? | f2675 | ? | 4.758 | 0.00001 |
| f480 | ? | f14461 | ? | 4.696 | 0.00001 |
| f5154 | ? | f8189 | ? | 4.643 | 0.00003 |
| f3227 | ? | f14175 | ? | 4.618 | 0.00005 |
| f2775 | ? | f3182 | ? | 4.519 | 0.00079 |
| f2588 | ? | f13157 | ? | 4.468 | 0.00001 |
| f4239 | ? | f4373 | ? | 4.410 | 0.00155 |
| f1049 | ? | f13346 | ? | 4.378 | 0.00006 |
| f7941 | ? | f8189 | ? | 4.302 | 0.00001 |
| f6699 | ? | f8635 | ? | 4.295 | 0.00001 |
| f8016 | ? | f11042 | ? | 4.294 | 0.00001 |
| f12986 | ? | f14117 | ? | 4.255 | 0.00013 |

## How to read this

- High-PMI pairs are candidate building blocks of circuits — features that systematically fire together on the same tokens.
- If stable features dominate the high-PMI tail relative to unstable features, stability + co-activation is a useful filter for downstream circuit discovery.
- PMI alone does not establish causality. To go from co-activation to circuit requires ablation experiments (forcing one feature off, observing the other).
