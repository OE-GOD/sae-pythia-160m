# Stable Features — sae_layer6_topk64_full.pt

Reference SAE: `checkpoints/sae_layer6_topk64_full.pt`
Threshold for 'stable': cos sim > **0.9**

## Comparison conditions (4)

- `topk_4k` → `checkpoints/sae_layer6_topk64_w4k.pt` (n_features=4096, arch=topk)
- `topk_64k` → `checkpoints/sae_layer6_topk64_w64k.pt` (n_features=65536, arch=topk)
- `topk_seed42` → `checkpoints/sae_layer6_topk64_seed42.pt` (n_features=16384, arch=topk)
- `matryoshka` → `checkpoints/sae_layer6_matryoshka.pt` (n_features=16384, arch=matryoshka)

## Stability distribution

| Stability score | # features | fraction |
|-----------------|-----------:|---------:|
| 0 / 4 | 11,974 | 0.7308 |
| 1 / 4 | 1,799 | 0.1098 |
| 2 / 4 | 1,272 | 0.0776 |
| 3 / 4 | 988 | 0.0603 |
| 4 / 4 | 351 | 0.0214 |

**351 of 16384 features** (2.14%) are stable across ALL 4 comparison conditions. These are the strongest candidates for 'real model features' — their decoder direction replicates regardless of SAE width, seed, or architecture.

## Fully stable features (351)

Features whose decoder direction has cos > 0.9 across every comparison:

| Feature ID | Label (if known) | Min cos across comparisons |
|-----------:|------------------|---------------------------:|
| 18 | _unlabeled_ | 0.9426 |
| 76 | _unlabeled_ | 0.9787 |
| 78 | _unlabeled_ | 0.9073 |
| 109 | _unlabeled_ | 0.9228 |
| 146 | _unlabeled_ | 0.9057 |
| 223 | _unlabeled_ | 0.9225 |
| 286 | _unlabeled_ | 0.9072 |
| 305 | _unlabeled_ | 0.9399 |
| 433 | _unlabeled_ | 0.9241 |
| 480 | _unlabeled_ | 0.9159 |
| 482 | _unlabeled_ | 0.9472 |
| 509 | _unlabeled_ | 0.9422 |
| 557 | _unlabeled_ | 0.9090 |
| 613 | _unlabeled_ | 0.9329 |
| 625 | _unlabeled_ | 0.9661 |
| 699 | _unlabeled_ | 0.9218 |
| 737 | _unlabeled_ | 0.9093 |
| 881 | _unlabeled_ | 0.9075 |
| 889 | _unlabeled_ | 0.9391 |
| 920 | _unlabeled_ | 0.9253 |
| 947 | _unlabeled_ | 0.9355 |
| 973 | _unlabeled_ | 0.9217 |
| 996 | _unlabeled_ | 0.9175 |
| 997 | _unlabeled_ | 0.9409 |
| 1006 | _unlabeled_ | 0.9269 |
| 1040 | _unlabeled_ | 0.9263 |
| 1049 | _unlabeled_ | 0.9372 |
| 1073 | _unlabeled_ | 0.9494 |
| 1076 | _unlabeled_ | 0.9251 |
| 1128 | _unlabeled_ | 0.9634 |
| 1134 | _unlabeled_ | 0.9092 |
| 1216 | _unlabeled_ | 0.9519 |
| 1353 | _unlabeled_ | 0.9279 |
| 1361 | _unlabeled_ | 0.9155 |
| 1406 | _unlabeled_ | 0.9188 |
| 1548 | _unlabeled_ | 0.9467 |
| 1592 | _unlabeled_ | 0.9046 |
| 1686 | _unlabeled_ | 0.9330 |
| 1703 | _unlabeled_ | 0.9155 |
| 1711 | _unlabeled_ | 0.9449 |
| 1789 | _unlabeled_ | 0.9113 |
| 1801 | _unlabeled_ | 0.9240 |
| 1828 | _unlabeled_ | 0.9283 |
| 1830 | _unlabeled_ | 0.9200 |
| 1842 | _unlabeled_ | 0.9618 |
| 1858 | _unlabeled_ | 0.9164 |
| 1871 | _unlabeled_ | 0.9729 |
| 1936 | _unlabeled_ | 0.9419 |
| 2034 | _unlabeled_ | 0.9684 |
| 2045 | _unlabeled_ | 0.9428 |
| ... | _and 301 more_ | ... |

## Labeled features — stability ranking

For each of the 26 auto-labeled features (T5), how many comparison conditions does it survive?

| Feature ID | Label | Stability | topk_4k | topk_64k | topk_seed42 | matryoshka |
|-----------:|-------|----------:|----------:|----------:|----------:|----------:|
| 12117 | BibTeX/LaTeX citation references | 4/4 | 0.997★ | 0.999★ | 0.999★ | 0.999★ |
| 5747 | Tokens related to file paths and directory structu | 4/4 | 0.997★ | 0.999★ | 0.999★ | 0.999★ |
| 12697 | Logical operators and negation symbols | 4/4 | 0.977★ | 0.995★ | 0.992★ | 0.992★ |
| 13227 | Punctuation and special characters in academic and | 3/4 | 0.900 | 0.995★ | 0.997★ | 0.976★ |
| 121 | Acronym and abbreviation expansions | 1/4 | 0.839 | 0.716 | 0.919★ | 0.702 |
| 6630 | French street names and abbreviations | 0/4 | 0.609 | 0.571 | 0.675 | 0.560 |
| 2255 | Newline tokens in various contexts | 0/4 | 0.395 | 0.463 | 0.464 | 0.455 |
| 12520 | Punctuation and formatting tokens | 0/4 | 0.393 | 0.430 | 0.435 | 0.416 |
| 6767 | Newline tokens in various contexts | 0/4 | 0.372 | 0.393 | 0.397 | 0.394 |
| 6484 | Newline tokens inside XML/HTML/CSS blocks | 0/4 | 0.344 | 0.398 | 0.399 | 0.391 |
| 15230 | Newline tokens in various contexts | 0/4 | 0.351 | 0.408 | 0.375 | 0.425 |
| 7615 | Newline tokens in various contexts | 0/4 | 0.371 | 0.368 | 0.371 | 0.352 |
| 10047 | Newline tokens inside CSS/XSL/HTML/XML blocks | 0/4 | 0.348 | 0.410 | 0.406 | 0.376 |
| 13131 | Newline tokens in various contexts | 0/4 | 0.325 | 0.352 | 0.361 | 0.354 |
| 11488 | Newline tokens in various contexts | 0/4 | 0.335 | 0.375 | 0.375 | 0.382 |
| 10925 | Newline tokens inside academic/technical text | 0/4 | 0.410 | 0.392 | 0.389 | 0.381 |
| 10045 | Subword BPE continuations of multi-piece words (e. | 0/4 | 0.335 | 0.385 | 0.421 | 0.372 |
| 13821 | Newline tokens inside code and prose blocks | 0/4 | 0.318 | 0.379 | 0.382 | 0.368 |
| 15245 | Newline tokens inside XML/HTML/CSS blocks | 0/4 | 0.328 | 0.349 | 0.349 | 0.353 |
| 15577 | Newline tokens and decimal points in various conte | 0/4 | 0.360 | 0.356 | 0.366 | 0.363 |
| 7448 | Newline tokens inside code blocks | 0/4 | 0.316 | 0.362 | 0.328 | 0.372 |
| 1989 | Decimal points in numerical contexts | 0/4 | 0.358 | 0.366 | 0.329 | 0.330 |
| 14584 | Newline tokens inside code and markup blocks | 0/4 | 0.301 | 0.351 | 0.356 | 0.338 |
| 10545 | Newline tokens inside code blocks | 0/4 | 0.310 | 0.358 | 0.373 | 0.338 |
| 2757 | Newline tokens in code and text formatting context | 0/4 | 0.328 | 0.351 | 0.344 | 0.328 |
| 5196 | Exponentiation notation in mathematical contexts | 0/4 | 0.382 | 0.798 | 0.828 | 0.853 |

## How to read this

- **High stability score (4/4)** = feature replicates under every comparison condition we tested. Strongest evidence the feature is a real direction in Pythia-160M's residual stream, not an artifact of SAE training.
- **Low stability score (0-1/4)** = feature is specific to this particular SAE training run. Probably not safe to build circuits or interpretations on.
- **Intermediate (2-3/4)** = stable under some conditions but not all. Worth deeper analysis to understand which conditions break the match.
