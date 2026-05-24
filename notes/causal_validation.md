# Causal Validation: Stable vs Unstable Features

Baseline CE: **3.0240** nats/token on 49,530 held-out predicted tokens.

## Per-feature results

| Feature | Group | Active % | ΔCE active | ΔCE inactive | Ratio |
|--------:|-------|---------:|-----------:|-------------:|------:|
| f12117 | stable | 49.17% | +0.0001 | +0.0060 | 0.02 |
| f5747 | stable | 40.12% | +0.0007 | +0.0115 | 0.06 |
| f12697 | stable | 38.15% | +0.0076 | +0.0986 | 0.08 |
| f2255 | unstable | 1.56% | +15.2027 | +0.6192 | 24.55 |
| f6767 | unstable | 1.57% | +13.2625 | +0.5652 | 23.47 |
| f7448 | unstable | 1.56% | +13.0069 | +0.4854 | 26.80 |
| f2105 | random | 0.05% | +0.2614 | +0.0024 | 107.08 |
| f2191 | random | 0.00% | +nan | +0.0234 | nan |
| f13060 | random | 0.00% | +nan | +0.0013 | nan |

## Group means

| Group | mean ΔCE active | mean ΔCE inactive | mean ratio |
|---|---:|---:|---:|
| stable | +0.0028 | +0.0387 | 0.05 |
| unstable | +13.8240 | +0.5566 | 24.94 |
| random | +0.2614 | +0.0091 | 107.08 |

## Interpretation

- Higher **ΔCE active** (cost of ablation on tokens where the feature was firing) indicates the feature carries information the model uses.
- **Ratio ΔCE_active / ΔCE_inactive** isolates the feature-specific effect: a value >> 1 means ablation hurts much more on the feature's own tokens than elsewhere.
- Stable features expected to show higher ratio than unstable / random features.