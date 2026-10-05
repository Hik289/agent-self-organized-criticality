from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import (
    benjamini_hochberg,
    bootstrap_alpha_vs_shuffled,
    dfa_exponent,
    field,
    read_input,
    records,
    spectral_slope,
    write_output,
)

RHO, D = 0.20, 1


def build_cells():
    from wave2a.statefulpuzzle import StatefulPuzzleConfig

    cells, rho_by = [], {}
    for H in [64, 128, 256, 512]:
        cid = f"H_{H}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"H": H, "rho": RHO})); rho_by[cid] = RHO
    return cells, rho_by


def _finite_fit(fit):
    return {
        name: float(value) if isinstance(value, (float, np.floating)) and np.isfinite(value)
        else None if isinstance(value, (float, np.floating))
        else value
        for name, value in fit.items()
    }


def analyze_temporal(rows, series_field="e_series", group_field=None,
                     n_permutations=1000, min_dfa_length=128, seed=42):
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, Integral) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if isinstance(min_dfa_length, bool) or not isinstance(min_dfa_length, Integral) or min_dfa_length < 64:
        raise ValueError("min_dfa_length must be an integer of at least 64")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    exclusions = Counter()
    groups = {}
    for index, row in enumerate(rows):
        try:
            values = np.asarray(field(row, series_field), dtype=float)
        except (TypeError, ValueError):
            exclusions["missing_or_invalid_series"] += 1
            continue
        if values.ndim != 1 or not np.isfinite(values).all():
            exclusions["nonfinite_or_nonvector_series"] += 1
            continue
        if len(values) < 8 or np.allclose(values, values.mean()):
            exclusions["short_or_constant_series"] += 1
            continue
        if group_field is None:
            group = "all"
        else:
            group = field(row, group_field)
            if isinstance(group, (list, dict)) or group is None:
                raise ValueError(f"Row {index}: group_field must identify a scalar group")
            group = str(group)
        spectrum = _finite_fit(spectral_slope(values))
        if spectrum["alpha"] is None:
            exclusions["unidentified_spectral_fit"] += 1
            continue
        dfa = _finite_fit(dfa_exponent(values)) if len(values) >= min_dfa_length else None
        lag = None
        if np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
            lag = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        groups.setdefault(group, []).append((values, {
            "row_index": index,
            "n_steps": len(values),
            "spectral_fit": spectrum,
            "lag_one_autocorrelation": lag if lag is not None and np.isfinite(lag) else None,
            "dfa": dfa,
            "dfa_status": "fit" if dfa is not None and dfa.get("H_dfa") is not None
            else "unidentified" if dfa is not None else "below_minimum_length",
        }))
    output = {}
    pvalues = []
    for group, items in sorted(groups.items()):
        observed = float(np.mean([item[1]["spectral_fit"]["alpha"] for item in items]))
        null = []
        for _ in range(n_permutations):
            alphas = [spectral_slope(rng.permutation(item[0]))["alpha"] for item in items]
            if all(alpha is not None and np.isfinite(alpha) for alpha in alphas):
                null.append(float(np.mean(alphas)))
        pvalue = float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (n_permutations + 1)) if len(null) == n_permutations else None
        dfa_values = [item[1]["dfa"]["H_dfa"] for item in items if item[1]["dfa_status"] == "fit"]
        output[group] = {
            "n_trajectories": len(items),
            "mean_psd_exponent": observed,
            "n_dfa_fits": len(dfa_values),
            "mean_dfa_exponent": float(np.mean(dfa_values)) if dfa_values else None,
            "null_mean_psd_exponent": float(np.mean(null)) if null else None,
            "null_interval_95": np.quantile(null, [0.025, 0.975]).tolist() if null else None,
            "n_valid_permutations": len(null),
            "p_greater": pvalue,
            "trajectories": [item[1] for item in items],
        }
        pvalues.append(pvalue)
    adjusted = benjamini_hochberg(pvalues)
    for item, adjusted_p in zip(output.values(), adjusted["p_adjusted"]):
        item["p_adjusted_bh"] = adjusted_p
    return {
        "status": "ok" if output else "insufficient_data",
        "spectral_estimand": "negative_log_log_Welch_PSD_slope",
        "dfa_estimand": "legacy_linear_detrending_fluctuation_slope",
        "null": "independent_within_trajectory_permutation_of_observed_values",
        "test_statistic": "unweighted_mean_PSD_exponent_across_trajectories",
        "alternative": "observed_mean_greater_than_permuted_mean",
        "fdr_family": "one_mean_spectral_test_per_group",
        "n_input_rows": len(rows),
        "n_excluded_rows": sum(exclusions.values()),
        "excluded_reasons": dict(exclusions),
        "min_dfa_length": min_dfa_length,
        "groups": output,
    }


def analyze(rows, series_field="e_series", group_field=None, n_permutations=1000, min_dfa_length=128, seed=42):
    by_H = {}
    for r in rows: by_H.setdefault(int(r["meta"]["H"]), []).append(r)
    per_H = {}
    for H, rs in sorted(by_H.items()):
        alphas_e, alphas_F, dfa_vals, p_vals = [], [], [], []
        Cs = []
        for r in rs:
            e = np.asarray(r["e_series"]); F = np.asarray(r["F_series"])
            ae = spectral_slope(e)["alpha"]; aF = spectral_slope(F)["alpha"]
            if ae is not None: alphas_e.append(ae)
            if aF is not None: alphas_F.append(aF)
            dv = dfa_exponent(e)["H_dfa"]
            if dv is not None: dfa_vals.append(dv)
            pv = bootstrap_alpha_vs_shuffled(e, seed=r["seed"], n_boot=50)["p_value"]
            if pv is not None: p_vals.append(pv)
            Cs.append(r["collapse_indicator"])
        per_H[H] = {
            "n": len(rs),
            "alpha_e_mean": float(np.mean(alphas_e)) if alphas_e else None,
            "alpha_e_std": float(np.std(alphas_e)) if alphas_e else None,
            "alpha_F_mean": float(np.mean(alphas_F)) if alphas_F else None,
            "DFA_H_mean": float(np.mean(dfa_vals)) if dfa_vals else None,
            "bootstrap_p_e_median": float(np.median(p_vals)) if p_vals else None,
            "bootstrap_p_e_lt_0p01_frac": float(np.mean(np.array(p_vals) < 0.01)) if p_vals else None,
            "collapse_rate": float(np.mean(Cs)),
        }
    return {
        "per_H": per_H,
        "temporal_diagnostics": analyze_temporal(
            rows,
            series_field=series_field,
            group_field=group_field,
            n_permutations=n_permutations,
            min_dfa_length=min_dfa_length,
            seed=seed,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--source", type=Path, default=HERE / "results.json")
    ap.add_argument("--output", type=Path, help="New diagnostics JSON path for --analyze-only.")
    ap.add_argument("--series-field", default="e_series")
    ap.add_argument("--group-field")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--min-dfa-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    parameters = {
        "series_field": args.series_field,
        "group_field": args.group_field,
        "n_permutations": args.n_permutations,
        "min_dfa_length": args.min_dfa_length,
        "seed": args.seed,
    }
    if args.analyze_only:
        try:
            data, source = read_input(args.source)
            diagnostics = analyze_temporal(records(data), **parameters)
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
    summary = run_experiment(exp_id="exp_6_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho_by)
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"], **parameters)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
