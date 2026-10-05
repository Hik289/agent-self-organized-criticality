from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import benjamini_hochberg, field, read_input, records, threshold_events, write_output

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

RHO, D = 0.20, 1


def build_cells():
    from wave2a.statefulpuzzle import StatefulPuzzleConfig

    cells, rho_by = [], {}
    for H in [8, 16, 32, 64, 128, 256, 512]:
        cid = f"H_{H}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"H": H, "rho": RHO})); rho_by[cid] = RHO
    return cells, rho_by


def analyze_horizons(rows, threshold=0.5, horizon_field="meta.H", pair_field="seed"):
    groups = {}
    for index, row in enumerate(rows):
        horizon = field(row, horizon_field)
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) or not math.isfinite(horizon) or horizon < 1 or int(horizon) != horizon:
            raise ValueError(f"Row {index}: horizon must be a positive integer")
        pair = field(row, pair_field)
        if isinstance(pair, (dict, list, bool)) or pair is None:
            raise ValueError(f"Row {index}: pair identifier must be a string or number")
        if isinstance(pair, float) and not math.isfinite(pair):
            raise ValueError(f"Row {index}: pair identifier must be finite")
        pair = str(pair)
        events = threshold_events(field(row, "e_series"), threshold)
        if not events["n_steps"] or events["n_steps"] > horizon:
            raise ValueError(f"Row {index}: observed series length must be in [1, horizon]")
        group = groups.setdefault(int(horizon), {})
        if pair in group:
            raise ValueError(f"Duplicate pair identifier {pair!r} at horizon {horizon}; filter different experimental conditions before analysis")
        group[pair] = {
            "row_index": index,
            "observed_steps": events["n_steps"],
            "n_events": events["n_events"],
            "total_exceedance_count": events["n_exceedances"],
            "total_event_mass": float(sum(event["size"] for event in events["events"])),
            "max_event_size": float(max((event["size"] for event in events["events"]), default=0)),
            "max_event_duration": max((event["duration"] for event in events["events"]), default=0),
            "touches_start_boundary": bool(events["events"] and events["events"][0]["start"] == 0),
            "touches_end_boundary": bool(events["events"] and events["events"][-1]["end"] == events["n_steps"] - 1),
        }
    summaries = {}
    for horizon, group in sorted(groups.items()):
        summaries[str(horizon)] = {"n_trajectories": len(group), "n_ended_before_horizon": sum(row["observed_steps"] < horizon for row in group.values())}
        for metric in ("total_exceedance_count", "total_event_mass", "max_event_size", "max_event_duration"):
            values = np.asarray([row[metric] for row in group.values()], dtype=float)
            summaries[str(horizon)][metric] = {
                "mean": float(values.mean()),
                "maximum": float(values.max()),
                "p90": float(np.quantile(values, 0.9)),
            }
        summaries[str(horizon)]["trajectories"] = group
    comparisons = []
    horizons = sorted(groups)
    for lower, upper in zip(horizons, horizons[1:]):
        matched = sorted(groups[lower].keys() & groups[upper].keys())
        comparison = {"lower_horizon": lower, "upper_horizon": upper, "n_pairs": len(matched), "n_unmatched_lower": len(groups[lower]) - len(matched), "n_unmatched_upper": len(groups[upper]) - len(matched), "metrics": {}}
        for metric in ("max_event_size", "total_exceedance_count"):
            differences = [groups[upper][pair][metric] - groups[lower][pair][metric] for pair in matched]
            positives = sum(value > 0 for value in differences)
            negatives = sum(value < 0 for value in differences)
            nonzero = positives + negatives
            pvalue = float(binomtest(positives, nonzero, 0.5, alternative="greater").pvalue) if nonzero else 1.0 if matched else None
            comparison["metrics"][metric] = {"mean_paired_difference": float(np.mean(differences)) if matched else None, "n_positive": positives, "n_negative": negatives, "n_ties": len(matched) - nonzero, "p_greater": pvalue}
        comparisons.append(comparison)
    for metric in ("max_event_size", "total_exceedance_count"):
        adjustment = benjamini_hochberg([comparison["metrics"][metric]["p_greater"] for comparison in comparisons])
        for comparison, adjusted_p in zip(comparisons, adjustment["p_adjusted"]):
            comparison["metrics"][metric]["p_adjusted_bh"] = adjusted_p
    return {
        "status": "ok" if summaries else "insufficient_data",
        "event_definition": "strict_contiguous_error_above_threshold; size=sum_of_errors",
        "legacy_A_i_equivalent": "total_exceedance_count",
        "boundary_policy": "observed_runs_retained_with_both_boundary_flags; sizes_of_runs_touching_either_observation_boundary_may_be_incomplete",
        "pairing": pair_field,
        "test": "paired_sign_test_for_positive_upper_minus_lower_difference",
        "fdr_family": "adjacent_horizon_tests_separately_for_each_metric",
        "per_horizon": summaries,
        "adjacent_comparisons": comparisons,
    }


def analyze(rows, threshold=0.5, horizon_field="meta.H", pair_field="seed"):
    by_H = {}
    for r in rows: by_H.setdefault(int(r["meta"]["H"]), []).append(r)
    per_H, all_A = {}, {}
    for H, rs in sorted(by_H.items()):
        A = np.array([r["A_i"] for r in rs]); C = np.array([r["collapse_indicator"] for r in rs])
        wsf = np.array([r["min_F"] for r in rs])
        per_H[H] = {"n": len(rs), "A_mean": float(A.mean()), "A_std": float(A.std()),
                    "A_max_cutoff": int(A.max()), "A_p90": float(np.quantile(A, 0.90)),
                    "collapse_rate": float(C.mean()), "WSF_floor_mean": float(wsf.mean())}
        all_A[H] = A
    pairwise = {}
    if _HAS_SCIPY:
        Hs = sorted(all_A.keys())
        for i in range(len(Hs)-1):
            Ha, Hb = Hs[i], Hs[i+1]
            w, p = stats.mannwhitneyu(all_A[Ha], all_A[Hb], alternative="less")
            pairwise[f"{Ha}<{Hb}"] = {"U": float(w), "p_value": float(p)}
    return {
        "per_H": per_H,
        "adjacent_H_A_mean_less_p": pairwise,
        "horizon_diagnostics": analyze_horizons(
            rows,
            threshold=threshold,
            horizon_field=horizon_field,
            pair_field=pair_field,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--source", type=Path, default=HERE / "results.json")
    ap.add_argument("--output", type=Path, help="New diagnostics JSON path for --analyze-only.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--horizon-field", default="meta.H")
    ap.add_argument("--pair-field", default="seed")
    args = ap.parse_args()
    parameters = {
        "threshold": args.threshold,
        "horizon_field": args.horizon_field,
        "pair_field": args.pair_field,
    }
    if args.analyze_only:
        try:
            data, source = read_input(args.source)
            diagnostics = analyze_horizons(records(data), **parameters)
            output = write_output(
                args.output or HERE / "analysis_outputs" / "diagnostics.json",
                diagnostics,
                [source],
                parameters,
            )
        except (ValueError, TypeError, OSError) as error:
            ap.error(str(error))
        print(str(output))
        return
    if args.output is not None:
        ap.error("--output requires --analyze-only")
    from wave2a.sp_runner import run_experiment

    cells, rho_by = build_cells()
    summary = run_experiment(exp_id="exp_10_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho_by,
        concurrency=2)
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"], **parameters)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
