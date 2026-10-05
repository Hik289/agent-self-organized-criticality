from __future__ import annotations
import json
import sys
from pathlib import Path
from numbers import Real
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import avalanche_nulls, benjamini_hochberg, field, fit_tail_models, read_input, records, threshold_events, write_output


def build_task_list(n_max: int = 200):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    tasks = []
    for i, t in enumerate(TASKS_TEST[:n_max]):
        tasks.append((i, {"task_index": i, "annotator": getattr(t, "annotator", None)}))
    return tasks


def analyze(rows):
    if not rows:
        return {"n": 0, "note": "no rows to analyze"}
    A = np.array([r["A_i"] for r in rows], dtype=float)
    R = np.array([r["reward"] for r in rows], dtype=float)
    C = np.array([r["collapse_indicator"] for r in rows], dtype=int)
    n = len(rows)
    A_max_int = int(A.max()) if len(A) else 0

    bins_edges = [0, 3, 6, 11, 21, max(A_max_int + 1, 22)]
    labels = ["1-2", "3-5", "6-10", "11-20", ">20"]
    hist_bin_counts = []
    for i, (lo, hi) in enumerate(zip(bins_edges[:-1], bins_edges[1:])):
        mask = (A >= lo) & (A < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            hist_bin_counts.append({"bin": labels[i], "n": 0})
            continue
        c_col = int(C[mask].sum())
        hist_bin_counts.append({
            "bin": labels[i],
            "n": n_bin,
            "n_collapsed": c_col,
            "final_collapse_fraction": c_col / n_bin,
            "reward_mean": float(R[mask].mean()),
        })

    A_sorted = np.sort(A[A > 0])
    tail_alpha = None
    tail_r2 = None
    if len(A_sorted) >= 5:
        ranks = np.arange(1, len(A_sorted) + 1) / len(A_sorted)
        surv = 1.0 - ranks + 1e-9
        lx = np.log(A_sorted[:-1] + 1e-9)
        ly = np.log(surv[:-1])
        from scipy import stats
        slope, _, r, _, _ = stats.linregress(lx, ly)
        tail_alpha = -float(slope)
        tail_r2 = float(r * r)

    minor = A[(A > 0) & (A <= 5)]
    major = A[A >= 10]
    from scipy import stats
    if len(minor) >= 3 and len(major) >= 3:
        ks_stat, ks_p = stats.ks_2samp(minor, major)
    else:
        ks_stat, ks_p = None, None

    return {
        "n": n,
        "A_mean": float(A.mean()),
        "A_std": float(A.std()),
        "A_median": float(np.median(A)),
        "A_max": int(A.max()),
        "reward_mean": float(R.mean()),
        "collapse_rate_overall": float(C.mean()),
        "avalanche_bins": hist_bin_counts,
        "tail_power_law_alpha": tail_alpha,
        "tail_power_law_r2": tail_r2,
        "minor_vs_major_ks_stat": ks_stat,
        "minor_vs_major_ks_p": ks_p,
    }


def _tail_input(data, args):
    if isinstance(data, list) and data and all(isinstance(value, Real) and not isinstance(value, bool) for value in data):
        return data, {"source": "explicit_numeric_array", "n_samples": len(data)}
    rows = records(data)
    if args.samples_field:
        samples = [field(row, args.samples_field) for row in rows]
        return samples, {"source": "explicit_record_field", "field": args.samples_field, "n_samples": len(samples)}
    samples = []
    n_boundary_excluded = 0
    for row in rows:
        events = threshold_events(field(row, args.series_field), args.threshold)
        for event in events["events"]:
            if not args.include_boundary_events and (event["touches_start_boundary"] or event["touches_end_boundary"]):
                n_boundary_excluded += 1
            else:
                samples.append(event["size"])
    return samples, {
        "source": "strict_contiguous_threshold_event_error_sums",
        "series_field": args.series_field,
        "n_trajectories": len(rows),
        "n_samples": len(samples),
        "n_boundary_events_excluded": n_boundary_excluded,
        "include_boundary_events": args.include_boundary_events,
        "likelihood_scope": "pooled_event_marginal; no_assumption_of_independent_events_for_significance_tests",
    }


def analyze_nulls(rows, args):
    sequences = [field(row, args.series_field) for row in rows]
    result = avalanche_nulls(sequences, threshold=args.threshold, n_simulations=args.n_simulations, seed=args.analysis_seed)
    result["event_catalog"] = [{"row_index": index, **threshold_events(sequence, args.threshold)} for index, sequence in enumerate(sequences)]
    return result


def analyze_tails(data, args):
    samples, selection = _tail_input(data, args)
    return {"sample_selection": selection, **fit_tail_models(samples, xmin=args.xmin, min_tail_size=args.min_tail_size)}


def analyze_diagnostics(rows, args, tail_data=None):
    return {
        "threshold_event_nulls": analyze_nulls(rows, args),
        "continuous_event_tail_models": analyze_tails(rows if tail_data is None else tail_data, args),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--analysis-kind", choices=("all", "avalanches", "tails"), default="all")
    ap.add_argument("--source", default=str(HERE / "results.json"))
    ap.add_argument("--output")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--series-field", default="e_series")
    ap.add_argument("--n-simulations", type=int, default=1000)
    ap.add_argument("--analysis-seed", type=int, default=42)
    ap.add_argument("--xmin", type=float)
    ap.add_argument("--min-tail-size", type=int, default=20)
    ap.add_argument("--include-boundary-events", action="store_true")
    ap.add_argument("--tail-source")
    ap.add_argument("--samples-field")
    ap.add_argument("--fdr-input")
    ap.add_argument("--fdr-q", type=float, default=0.05)
    args = ap.parse_args()
    if args.analysis_kind != "all" and not args.analyze_only and not args.fdr_input:
        ap.error("--analysis-kind requires --analyze-only")
    try:
        parameters = {key: value for key, value in vars(args).items() if key not in {"source", "output", "tail_source", "fdr_input"}}
        if args.fdr_input:
            data, source = read_input(args.fdr_input)
            tests = records(data, "tests")
            pvalues = [field(test, "p_value") for test in tests]
            result = benjamini_hochberg(pvalues, q=args.fdr_q)
            result["tests"] = [{"id": test.get("id", index), "p_value": pvalues[index], "p_adjusted": result["p_adjusted"][index], "rejected": result["rejected"][index]} for index, test in enumerate(tests)]
            output = write_output(args.output or HERE / "analysis_outputs/fdr.json", result, [source], parameters)
            print(str(output))
            return
        if args.analyze_only:
            data, source = read_input(args.source)
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
        else:
            from wave2b.tb_runner import run_experiment
            tasks = build_task_list(n_max=args.n)
            summary = run_experiment(
                exp_id="exp_1_2",
                tasks=tasks,
                out_dir=HERE,
                domain="retail",
                seed=42,
                max_num_steps=args.max_steps,
                concurrency=2,
                save_raw=True,
            )
            data, source = read_input(HERE / "results.json")
        sources = [source]
        tail_data = None
        if args.tail_source:
            tail_data, tail_source = read_input(args.tail_source)
            sources.append(tail_source)
        if args.analysis_kind == "tails":
            result = analyze_tails(data if tail_data is None else tail_data, args)
        elif args.analysis_kind == "avalanches":
            result = analyze_nulls(records(data), args)
        else:
            rows = records(data)
            agg = analyze(rows)
            agg["diagnostics"] = analyze_diagnostics(rows, args, tail_data=tail_data)
            result = {"summary": summary, "aggregates": agg}
        if args.analyze_only or args.output:
            output = write_output(args.output or HERE / "analysis_outputs/diagnostics.json", result, sources, parameters)
            print(str(output))
        else:
            (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2, allow_nan=False))
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except (ValueError, TypeError, OSError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
