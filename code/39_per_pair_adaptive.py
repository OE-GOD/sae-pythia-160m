"""Per-pair adaptive method: use linearity probe at alpha=0.5 to decide
AtP vs AP per (feature, head, position).

Rule:
  For each (feature, position, L, h):
    1. Compute AtP estimate (free; comes from script 32's output)
    2. If |AtP_est| < probe_threshold => the effect is tiny; predict 0 (cheap)
    3. Else: run AP at alpha=0.5 (1 forward pass) => probe_drop
       a. expected if linear: 0.5 * AtP_est
       b. linearity ratio = probe_drop / (0.5 * AtP_est)
       c. If |ratio - 1.0| < linearity_tol => trust AtP; predict AtP_est
       d. Else => run AP at alpha=1.0 => use AP

Cost amortization:
  - AtP setup: 2 model passes per (feature, position)
  - probe: 1 forward pass per non-tiny (L, h)
  - AP fallback: 1 forward pass per non-linear (L, h)

Compare against:
  - Full AP (truth): 60 passes/feature-pos
  - Adaptive (T=11): 14 passes/feature-pos, Pearson 0.992 (from Finding 12)
  - This per-pair: hoped to match or beat 0.992 at less cost

Hyperparameter sweep:
  - probe_threshold ∈ {0.001, 0.005, 0.01, 0.02}: smaller = more probes, more cost
  - linearity_tol ∈ {0.10, 0.20, 0.30}: tighter = more AP fallbacks
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ap_path", type=str,
                        default="checkpoints/sae_layer6_topk64_full.sae_feature_path_patching.json")
    parser.add_argument("--atp_path", type=str,
                        default="checkpoints/sae_layer6_topk64_full.sae_feature_atp.json")
    parser.add_argument("--alpha_path", type=str,
                        default="checkpoints/sae_layer6_topk64_full.alpha_scaling_all_layers.json")
    args = parser.parse_args()

    ap_data = load_json(args.ap_path)["feature_results"]
    atp_data = load_json(args.atp_path)["feature_results"]
    alpha_data = load_json(args.alpha_path)["feature_results"]

    common = sorted(set(ap_data.keys()) & set(atp_data.keys()) & set(alpha_data.keys()))
    print(f"common features: {common}")

    # Build per-(feature, L, h) records combining all three
    # AP and AtP are mean over positions; alpha-scaling is per-position with all alphas
    # To make this clean, we compute the per-position probe drop and AtP from existing data
    # NB: scripts 30, 32, 37 all hit the same firing positions (find_top_firing_positions
    # is deterministic with seed 0). So position indices align.

    rows = []
    for fid in common:
        ap_arr = np.array(ap_data[fid]["mean_mediation"])  # (n_L, n_h)
        atp_arr = np.array(atp_data[fid]["mean_atp_mediation"])
        downstream_layers = ap_data[fid]["downstream_layers"]
        n_heads = ap_data[fid]["n_heads"]
        layers_alpha = [int(L) for L in alpha_data[fid]["layers_to_test"]]
        n_pos = len(alpha_data[fid]["per_position_meta"])
        assert layers_alpha == downstream_layers, \
            f"layer mismatch: alpha {layers_alpha} vs AP {downstream_layers}"

        # For each (L, h), pull the probe values: mean over positions of
        # logp_drop at alpha=0.5
        for li, L in enumerate(downstream_layers):
            for h in range(n_heads):
                ap = float(ap_arr[li, h])
                atp = float(atp_arr[li, h])
                # alpha-scaling gives us per-position drops at alpha=0.5
                drops_05 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["0.5"]
                probe = float(np.mean(drops_05)) if drops_05 else 0.0
                drops_10 = alpha_data[fid]["logp_drops"][str(L)][str(h)]["1.0"]
                ap_from_alpha = float(np.mean(drops_10)) if drops_10 else ap
                rows.append({
                    "feature": fid, "L": L, "h": h,
                    "ap": ap, "atp": atp,
                    "probe_alpha_05": probe,
                    "ap_alpha_10_from_alpha_data": ap_from_alpha,
                })

    n_total = len(rows)
    print(f"\nTotal (feature, L, h) entries: {n_total}")

    # Sanity check: AP from script 30 should approximately match alpha-data AP at alpha=1.0
    # They use the same intervention but possibly different normalization.
    diffs = [r["ap"] - r["ap_alpha_10_from_alpha_data"] for r in rows]
    print(f"AP-vs-alpha_data AP: mean diff {np.mean(diffs):.5f}, "
          f"max abs {np.max(np.abs(diffs)):.5f}")

    ap_all = np.array([r["ap"] for r in rows])

    # Hyperparameter sweep
    print("\n=== PER-PAIR ADAPTIVE: HYPERPARAMETER SWEEP ===")
    print(f"  {'probe_thr':>10} {'lin_tol':>8}  "
          f"{'n_probe':>8} {'n_AP_fb':>8}  {'Pearson':>9} {'RMSE':>8}  "
          f"{'cost/AP':>9}  {'speedup':>8}")

    sweep_results = []
    n_features = len(common)
    n_positions = 3  # from script 37
    n_layers = 5
    n_heads = 12

    # Per (feature, position) base cost: AtP = 2 passes
    # Per (L, h): probe = 1 pass, AP_fallback = 1 pass
    # Total cost = n_features * n_positions * 2 (AtP) + n_probes * 1 + n_AP_fb * 1
    # but n_probes/n_AP_fb counted per (feature, L, h) since we average over positions
    # We treat n_features * n_positions = 9 "AtP setups", and the rest scale by 9 too

    # cost normalization:
    # full AP cost = n_features * n_positions * n_layers * n_heads = 9 * 60 = 540 passes
    # (this is per-position aggregate; we report per-feature-position)
    cost_full_ap = n_layers * n_heads  # 60 passes per feature-position
    atp_setup_cost = 2

    for probe_thr in [0.001, 0.003, 0.005, 0.01, 0.02]:
        for lin_tol in [0.10, 0.20, 0.30]:
            predicted = np.zeros_like(ap_all)
            n_probe = 0
            n_ap_fb = 0
            for i, r in enumerate(rows):
                atp = r["atp"]
                if abs(atp) < probe_thr:
                    # tiny effect: predict 0 (or AtP, doesn't really matter)
                    predicted[i] = atp
                    continue
                # probe
                n_probe += 1
                probe = r["probe_alpha_05"]
                expected = 0.5 * atp
                if abs(expected) < 1e-8:
                    predicted[i] = atp
                    continue
                ratio = probe / expected
                if abs(ratio - 1.0) < lin_tol:
                    # linear: trust AtP
                    predicted[i] = atp
                else:
                    # not linear: fall back to AP
                    n_ap_fb += 1
                    predicted[i] = r["ap"]

            pr, _ = stats.pearsonr(ap_all, predicted)
            rmse = float(np.sqrt(np.mean((ap_all - predicted) ** 2)))

            # Cost per feature-position
            # Total entries = n_total = 9 features-positions × 60 layer-heads = 540
            # Per feature-position layer-head count = 60
            # n_probe and n_ap_fb are summed over feature-positions × pairs
            # Average per feature-position:
            avg_probes_per_fpos = n_probe / (n_features * n_positions)
            avg_ap_fb_per_fpos = n_ap_fb / (n_features * n_positions)
            cost_per_fpos = atp_setup_cost + avg_probes_per_fpos + avg_ap_fb_per_fpos
            cost_pct = 100 * cost_per_fpos / cost_full_ap

            sweep_results.append({
                "probe_thr": probe_thr, "lin_tol": lin_tol,
                "n_probe": n_probe, "n_ap_fb": n_ap_fb,
                "pearson": pr, "rmse": rmse,
                "cost_per_fpos": cost_per_fpos,
                "cost_pct_vs_full_ap": cost_pct,
                "speedup": cost_full_ap / cost_per_fpos,
            })
            print(f"  {probe_thr:>10.3f} {lin_tol:>8.2f}  "
                  f"{n_probe:>8} {n_ap_fb:>8}  {pr:>9.4f} {rmse:>8.5f}  "
                  f"{cost_pct:>8.1f}%  {cost_full_ap/cost_per_fpos:>7.2f}x")

    # Find Pareto-best per Pearson target
    print("\n=== BEST PER-PAIR ADAPTIVE CONFIGS ===")
    for target in [0.99, 0.995, 0.999]:
        candidates = [r for r in sweep_results if r["pearson"] >= target]
        if not candidates:
            print(f"  No config achieves Pearson >= {target}")
            continue
        best = min(candidates, key=lambda r: r["cost_per_fpos"])
        print(f"  Pearson >= {target}: probe_thr={best['probe_thr']}, "
              f"lin_tol={best['lin_tol']}, "
              f"Pearson={best['pearson']:.4f}, "
              f"cost={best['cost_per_fpos']:.1f} passes ({best['cost_pct_vs_full_ap']:.1f}% of AP), "
              f"speedup={best['speedup']:.2f}x")

    # Compare to baselines
    print("\n=== COMPARISON TO BASELINES ===")
    atp_all = np.array([r["atp"] for r in rows])
    atp_pearson, _ = stats.pearsonr(ap_all, atp_all)
    print(f"  Full AtP:           Pearson={atp_pearson:.4f}, cost=2 passes (3.3% of AP)")
    print(f"  Layer-adaptive T=11: Pearson=0.9919, cost=14 passes (23% of AP) -- from Finding 12")
    print(f"  Full IG (N=10):     Pearson=0.9466, cost=20 passes (33% of AP) -- from Finding 12")
    print(f"  Full AP (truth):    Pearson=1.0000, cost=60 passes (100%)")

    # Save
    out_path = "checkpoints/sae_layer6_topk64_full.per_pair_adaptive.json"
    Path(out_path).write_text(json.dumps({
        "method": "per_pair_adaptive",
        "n_total_entries": n_total,
        "n_features": n_features,
        "n_positions": n_positions,
        "sweep_results": sweep_results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
