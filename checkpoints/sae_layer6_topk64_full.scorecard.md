# SAE Scorecard — sae_layer6_topk64_full.pt

Multi-axis quality profile. Each axis is in [0, 1] where 1 = best.

```
AXIS                                              BAR                     VALUE     SOURCE
----------------------------------------------------------------------------------------------------
Reconstruction (variance explained)               [███████████████████░]  0.9843  eval.json
Predictive preservation (CE-delta loss recovered)  [███████████████████░]  0.9542  ce_delta.json
Capacity utilization (1 - dead fraction)          [██████████████████░░]  0.9478  eval.json
Cross-seed stability (cos > 0.9 vs different-seed SAE)  [███░░░░░░░░░░░░░░░░░]  0.1986  replication_analysis.json
Cross-architecture stability (cos > 0.9 Matryoshka vs flat)  [██░░░░░░░░░░░░░░░░░░]  0.1065  matryoshka_analysis.json
Auto-interp monosemanticity rate                  [█████████████████░░░]  0.8846  feature_catalog.json
```

## Axis-by-axis

**Reconstruction (variance explained)**
- Value: `0.9843`
- Interpretation: fraction of activation variance captured
- Source: `eval.json`

**Predictive preservation (CE-delta loss recovered)**
- Value: `0.9542`
- Interpretation: fraction of layer's predictive contribution preserved
- Source: `ce_delta.json`

**Capacity utilization (1 - dead fraction)**
- Value: `0.9478`
- Interpretation: fraction of features that activate on at least one input
- Source: `eval.json`

**Cross-seed stability (cos > 0.9 vs different-seed SAE)**
- Value: `0.1986`
- Interpretation: fraction of features replicating across random seeds
- Source: `replication_analysis.json`

**Cross-architecture stability (cos > 0.9 Matryoshka vs flat)**
- Value: `0.1065`
- Interpretation: fraction of features replicating across SAE architectures
- Source: `matryoshka_analysis.json`

**Auto-interp monosemanticity rate**
- Value: `0.8846`
- Interpretation: fraction of labeled features classified MONOSEMANTIC
- Source: `feature_catalog.json`

## Diagnosis

- Strongest axis: **Reconstruction (variance explained)** (0.9843)
- Weakest axis:   **Cross-architecture stability (cos > 0.9 Matryoshka vs flat)** (0.1065)
- Range across axes: **0.8778**

The 88%-point gap between the strongest and weakest axis is the central evidence for the 'no single metric tells the full story' argument. Reporting any one axis without the others would misrepresent this SAE's quality.

## How to read this

- **Variance explained** alone (the metric most papers report) describes only
  reconstruction quality in activation space. It is necessary but not sufficient.
- **CE-delta loss recovered** is the rigorous check that the reconstruction
  preserves the model's downstream behavior.
- **Cross-seed and cross-arch stability** test whether the SAE's *features* are
  a property of the model or an artifact of the SAE training procedure.
- A clean SAE result is one where the weakest axis is still acceptable. A misleading
  SAE result is one with a very strong reconstruction axis but poor stability axes.
