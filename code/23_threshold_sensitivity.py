"""Threshold-sensitivity analysis for the driver/thermometer classifier.

22_thermometer_at_scale.py classifies features based on:
  - mean Δ predicted-concept tokens (steered minus baseline)
  - thresholds: driver if Δ >= 3.0; thermometer if Δ <= 0.5

The 78% thermometer rate from that script depends on these specific thresholds.
This script tests whether the qualitative finding ("most features are
thermometers") is robust to threshold choice by re-classifying with multiple
threshold pairs and reporting the resulting distributions.

No new steering required — we use the per-feature deltas already saved by
script 22.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str,
                   default="checkpoints/sae_layer6_topk64_full.thermometer_at_scale.json")
    p.add_argument("--out", type=str,
                   default="checkpoints/sae_layer6_topk64_full.threshold_sensitivity.json")
    return p.parse_args()


def classify(delta, drv_thresh, therm_thresh):
    if delta >= drv_thresh:
        return "driver"
    if delta <= therm_thresh:
        return "thermometer"
    return "ambiguous"


def main():
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    results = data["results"]
    n = len(results)
    deltas = np.array([r["mean_delta_concept_tokens"] for r in results])

    print(f"loaded {n} feature results from {args.input}")
    print(f"\nΔ distribution:")
    print(f"  min={deltas.min():.2f}  max={deltas.max():.2f}  mean={deltas.mean():.2f}  "
          f"median={np.median(deltas):.2f}")
    print(f"  Δ values: {sorted(deltas.tolist(), reverse=True)}")

    # --- Threshold grid ---
    # Pairs of (driver_threshold, thermometer_threshold)
    # Driver threshold typically >= thermometer threshold.
    threshold_pairs = [
        (1.0, 0.5),
        (2.0, 0.5),
        (3.0, 0.5),   # the default from script 22
        (5.0, 0.5),
        (10.0, 0.5),
        (3.0, 0.0),
        (3.0, 1.0),
        (3.0, 2.0),
        (1.0, 1.0),   # any positive Δ is driver
        (5.0, 2.0),
    ]

    print(f"\n{'=' * 80}")
    print("THRESHOLD SENSITIVITY")
    print(f"{'=' * 80}")
    print(f"  {'drv':>5} {'therm':>5} | {'driver':>8} {'thermo':>8} {'amb':>5} | qualitative finding")
    print(f"  {'-' * 5:>5} {'-' * 5:>5} + {'-' * 8:>8} {'-' * 8:>8} {'-' * 5:>5} + {'-' * 30}")

    sensitivity_results = []
    for drv_t, therm_t in threshold_pairs:
        by_verdict = {"driver": 0, "thermometer": 0, "ambiguous": 0}
        for d in deltas:
            v = classify(d, drv_t, therm_t)
            by_verdict[v] += 1
        d_pct = 100 * by_verdict["driver"] / n
        t_pct = 100 * by_verdict["thermometer"] / n
        a_pct = 100 * by_verdict["ambiguous"] / n
        # Qualitative summary
        if t_pct > 60:
            qual = "majority thermometer"
        elif d_pct > 60:
            qual = "majority driver"
        elif t_pct > d_pct:
            qual = "more thermometer than driver"
        elif d_pct > t_pct:
            qual = "more driver than thermometer"
        else:
            qual = "balanced"
        print(f"  {drv_t:>5.1f} {therm_t:>5.1f} | "
              f"{by_verdict['driver']:>3} ({d_pct:>4.1f}%) {by_verdict['thermometer']:>3} ({t_pct:>4.1f}%) "
              f"{by_verdict['ambiguous']:>2}  | {qual}")
        sensitivity_results.append({
            "driver_threshold": drv_t,
            "thermometer_threshold": therm_t,
            "n_driver": by_verdict["driver"],
            "n_thermometer": by_verdict["thermometer"],
            "n_ambiguous": by_verdict["ambiguous"],
            "pct_driver": d_pct,
            "pct_thermometer": t_pct,
            "qualitative": qual,
        })

    # --- Continuous summary: at what driver-threshold does driver rate cross 50%? ---
    print(f"\n{'=' * 80}")
    print("AT WHAT DRIVER THRESHOLD DOES DRIVER RATE EQUAL VARIOUS LEVELS?")
    print(f"{'=' * 80}")
    sorted_deltas = np.sort(deltas)[::-1]  # descending
    for target_pct in [10, 25, 50, 75, 90]:
        # What threshold T gives exactly target_pct% drivers (Δ >= T)?
        idx = int(np.ceil(n * target_pct / 100)) - 1
        idx = max(0, min(n - 1, idx))
        threshold = sorted_deltas[idx]
        print(f"  {target_pct:>3}% drivers requires driver_threshold <= {threshold:>6.2f}")

    # --- Robustness verdict ---
    print(f"\n{'=' * 80}")
    print("ROBUSTNESS VERDICT")
    print(f"{'=' * 80}")
    therm_majority_count = sum(1 for r in sensitivity_results if r["pct_thermometer"] > 50)
    print(f"  Thermometer is the majority verdict at {therm_majority_count}/{len(sensitivity_results)} "
          f"of the tested threshold pairs.")
    if therm_majority_count / len(sensitivity_results) >= 0.7:
        print(f"  -> Finding is ROBUST: thermometer-dominance holds across most reasonable thresholds.")
    else:
        print(f"  -> Finding is THRESHOLD-DEPENDENT: choose thresholds carefully when reporting.")

    # Compute the "neutral" classification: just count Δ > 0 vs Δ <= 0
    n_positive_delta = sum(1 for d in deltas if d > 0)
    n_zero_or_neg_delta = sum(1 for d in deltas if d <= 0)
    print(f"\n  Neutral version (any positive Δ vs zero/negative):")
    print(f"    Δ > 0:   {n_positive_delta}/{n} ({100 * n_positive_delta / n:.1f}%)")
    print(f"    Δ <= 0:  {n_zero_or_neg_delta}/{n} ({100 * n_zero_or_neg_delta / n:.1f}%)")
    print(f"    (Any nonzero drift toward concept counts as 'driver' here — even weak drift.)")

    # Save
    Path(args.out).write_text(json.dumps({
        "input": args.input,
        "n_features": n,
        "delta_stats": {
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
        },
        "threshold_sweep": sensitivity_results,
        "n_with_positive_delta": int(n_positive_delta),
        "n_with_nonpositive_delta": int(n_zero_or_neg_delta),
        "n_threshold_pairs_with_therm_majority": therm_majority_count,
        "robust": therm_majority_count / len(sensitivity_results) >= 0.7,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
