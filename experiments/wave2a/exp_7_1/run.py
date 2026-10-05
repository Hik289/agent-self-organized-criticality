from __future__ import annotations
import json, sys, time
import math
from collections import Counter
from numbers import Integral
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import fit_power_and_exp, read_input, records, write_output

D_LEVELS = [1, 2, 4, 6, 8]
H, V, S = 128, 8, 20
RHO = 0.10
SEED_BASE = 42
CONCURRENCY = 4

def _number(value):
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None

def _metadata(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    number = _number(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else number

def _series(row, key):
    if not isinstance(row, dict):
        return None, "record_not_mapping"
    if key not in row or row[key] is None:
        return None, f"missing_{key}"
    try:
        values = np.asarray(row[key], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None, f"nonnumeric_{key}"
    if values.ndim != 1:
        return None, f"nonvector_{key}"
    if not values.size:
        return None, f"empty_{key}"
    if not np.isfinite(values).all():
        return None, f"nonfinite_{key}"
    return values, None

def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None

def _invalid(reason):
    return {"valid": False, "reason": reason}

def _fit_divergence(values):
    positive = values[values > 0]
    if positive.size < 4:
        return _invalid("fewer_than_four_positive_observations")
    if np.ptp(positive) == 0:
        return _invalid("constant_positive_response")
    try:
        fit = fit_power_and_exp(np.arange(1, values.size + 1, dtype=float), values)
    except (TypeError, ValueError, ArithmeticError) as exc:
        result = _invalid("fit_failed")
        result["error_type"] = type(exc).__name__
        return result
    if not isinstance(fit, dict) or fit.get("power") is None or fit.get("exp") is None:
        return _invalid("fit_unavailable")
    result = {}
    for model, fields in (
        ("power", ("beta", "log_c", "r2", "aic", "bic")),
        ("exp", ("lambda", "log_c", "r2", "aic", "bic")),
    ):
        if not isinstance(fit[model], dict):
            return _invalid("invalid_fit_schema")
        result[model] = {}
        for field in fields:
            value = _number(fit[model].get(field))
            if value is None:
                return _invalid("nonfinite_or_missing_fit_output")
            result[model][field] = value
    delta_aic = _number(fit.get("delta_aic_exp_minus_power"))
    if delta_aic is None:
        return _invalid("nonfinite_or_missing_delta_aic")
    n_fit = _number(fit.get("n"))
    if n_fit != positive.size:
        return _invalid("fit_sample_count_mismatch")
    return {
        "valid": True,
        "reason": None,
        "n_fit": int(positive.size),
        "n_zero_observations_excluded": int(values.size - positive.size),
        "power": result["power"],
        "exp": result["exp"],
        "delta_aic_exp_minus_power": delta_aic,
        "best": "power" if delta_aic > 0 else "exp" if delta_aic < 0 else "tie",
    }

def _tail_band(values, tolerance, window):
    if values.size < 2 * window:
        return _invalid("fewer_than_two_windows")
    tail = np.sort(values[-window:])
    middle = window // 2
    tail_median = float(tail[middle]) if window % 2 else float(tail[middle - 1] / 2 + tail[middle] / 2)
    maximum = float(np.max(values))
    half_width = tolerance * maximum
    outside = np.flatnonzero(np.abs(values - tail_median) > half_width)
    first = int(outside[-1]) + 1 if outside.size else 0
    suffix_length = int(values.size - first)
    met = suffix_length >= window
    return {
        "valid": True,
        "reason": None,
        "criterion_met": bool(met),
        "tail_median": tail_median,
        "observed_maximum": maximum,
        "band_half_width": half_width,
        "first_tail_band_step": first + 1 if met else None,
        "stable_suffix_length": suffix_length,
    }

def _exclusions(records, key):
    return dict(sorted(Counter(record[key]["reason"] for record in records if not record[key]["valid"]).items()))

def analyze_divergence(pairs, saturation_tolerance=0.1, saturation_window=10):
    tolerance = _number(saturation_tolerance)
    window_number = _number(saturation_window)
    if tolerance is None or not 0 <= tolerance <= 1:
        raise ValueError("saturation_tolerance must be finite and between 0 and 1.")
    if window_number is None or not window_number.is_integer() or window_number < 2:
        raise ValueError("saturation_window must be an integer of at least 2.")
    window = int(window_number)
    records = []
    by_depth = {}
    for index, pair in enumerate(pairs):
        depth_number = _number(pair.get("D")) if isinstance(pair, dict) else None
        depth = int(depth_number) if depth_number is not None and depth_number.is_integer() and depth_number >= 1 else None
        record = {
            "pair_index": index,
            "D": depth,
            "seed": _metadata(pair.get("seed")) if isinstance(pair, dict) else None,
        }
        values, reason = _series(pair, "delta_series")
        record["trajectory_kind"] = None
        if values is not None and not np.any(values < 0):
            record["trajectory_kind"] = "all_zero" if np.all(values == 0) else "constant_positive" if np.ptp(values) == 0 else "nonconstant"
        if reason is None and depth is None:
            reason = "missing_or_invalid_dependency_depth"
        if reason is None and np.any(values < 0):
            reason = "negative_divergence"
        if reason is None and np.all(values == 0):
            reason = "all_zero_divergence"
        record["n_steps"] = int(values.size) if values is not None else None
        if reason is not None:
            record["fit"] = _invalid(reason)
            record["saturation"] = _invalid(reason)
        else:
            record["fit"] = _fit_divergence(values)
            record["saturation"] = _tail_band(values, tolerance, window)
        records.append(record)
        if depth is not None:
            by_depth.setdefault(depth, []).append(record)
    groups = []
    for depth, group in sorted(by_depth.items()):
        fits = [record["fit"] for record in group if record["fit"]["valid"]]
        bands = [record["saturation"] for record in group if record["saturation"]["valid"]]
        matched = [band for band in bands if band["criterion_met"]]
        groups.append({
            "D": depth,
            "n_pairs": len(group),
            "n_valid_fits": len(fits),
            "n_fit_excluded": len(group) - len(fits),
            "fit_exclusions": _exclusions(group, "fit"),
            "mean_delta_aic_exp_minus_power": _mean([fit["delta_aic_exp_minus_power"] for fit in fits]),
            "fraction_power_selected": sum(fit["best"] == "power" for fit in fits) / len(fits) if fits else None,
            "n_aic_ties": sum(fit["best"] == "tie" for fit in fits),
            "n_valid_tail_band_diagnostics": len(bands),
            "n_tail_band_met": len(matched),
            "fraction_tail_band_met": len(matched) / len(bands) if bands else None,
            "mean_first_tail_band_step_when_met": _mean([float(band["first_tail_band_step"]) for band in matched]),
            "saturation_exclusions": _exclusions(group, "saturation"),
        })
    n_valid_fits = sum(record["fit"]["valid"] for record in records)
    n_valid_bands = sum(record["saturation"]["valid"] for record in records)
    return {
        "analysis": "paired_observed_divergence",
        "fit_protocol": {
            "response": "log(delta_series)",
            "predictors": {"power": "log(step)", "exp": "step"},
            "step_origin": 1,
            "fit_support": "identical strictly positive divergence observations for both models",
            "selection": "minimum AIC on the common log response; exact ties reported separately",
        },
        "saturation_protocol": {
            "interpretation": "operational tail-band diagnostic; does not establish chaos or its absence",
            "tail_window": window,
            "minimum_series_length": 2 * window,
            "tolerance": tolerance,
            "center": "median of the final window",
            "half_width": "tolerance times observed maximum divergence",
            "criterion": "a suffix of at least window observations stays inside the tail band through the final observation",
        },
        "n_pairs": len(records),
        "n_pairs_missing_seed_metadata": sum(record["seed"] is None for record in records),
        "n_valid_fits": n_valid_fits,
        "n_fit_excluded": len(records) - n_valid_fits,
        "fit_exclusions": _exclusions(records, "fit"),
        "n_valid_tail_band_diagnostics": n_valid_bands,
        "n_saturation_excluded": len(records) - n_valid_bands,
        "saturation_exclusions": _exclusions(records, "saturation"),
        "per_D": groups,
        "pairs": records,
    }

def _run_traj(client, cfg, obs0_delta):
    from wave2a.azure_client import chat
    from wave2a.statefulpuzzle import StatefulPuzzleSOC, run_stepwise_trajectory, DEFAULT_SYSTEM
    from wave2a.pipeline import analyze_trajectory
    env = StatefulPuzzleSOC(cfg)
    beliefs, steps, meta = run_stepwise_trajectory(client, cfg, env, llm_call=chat,
        K_history=3, system_prompt=DEFAULT_SYSTEM, obs0_perturb_delta=obs0_delta)
    traj = {"case": f"D{cfg.D}_d{obs0_delta}",
            "config": {"H": cfg.H, "S": cfg.S, "D": cfg.D, "V": cfg.V,
                       "seed": cfg.seed, "perturbation": None, "rho": cfg.rho},
            "gold_series": env.gold.tolist(), "trajectory": steps}
    ana = analyze_trajectory(traj)
    return beliefs, {"llm": meta, "F_series": ana["F_series"],
                     "A_i": ana["avalanche"]["A"], "collapse_indicator": ana["collapse_indicator"]}

def _run_pair(client, D, j):
    from wave2a.statefulpuzzle import StatefulPuzzleConfig
    seed_j = SEED_BASE + j
    cfg = StatefulPuzzleConfig(H=H, S=S, D=D, V=V, seed=seed_j, rho=RHO, perturbation=None)
    bA, mA = _run_traj(client, cfg, 0)
    bB, mB = _run_traj(client, cfg, 1)
    delta = np.array([(int(a)-int(b)) % V for a,b in zip(bA,bB)], dtype=float)
    delta_dist = np.minimum(delta, V - delta)
    fit = fit_power_and_exp(np.arange(1, len(delta_dist)+1), delta_dist)
    return {"D": D, "pair_idx": j, "seed": seed_j,
            "delta_series": delta_dist.tolist(), "fit": fit,
            "F_A_min": float(min(mA["F_series"])), "F_B_min": float(min(mB["F_series"])),
            "A_A": mA["A_i"], "A_B": mB["A_i"],
            "C_A": mA["collapse_indicator"], "C_B": mB["collapse_indicator"],
            "cost_usd": float(mA["llm"]["cost_usd"] + mB["llm"]["cost_usd"])}

def main():
    global N_PAIRS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--source", default=str(HERE / "results.json"))
    ap.add_argument("--output", help="New offline diagnostic output path; existing files are never overwritten.")
    ap.add_argument("--saturation-tolerance", type=float, default=0.1)
    ap.add_argument("--saturation-window", type=int, default=10)
    args = ap.parse_args()
    if args.analyze_only:
        data, source = read_input(args.source)
        result = analyze_divergence(records(data, "pairs"), saturation_tolerance=args.saturation_tolerance, saturation_window=args.saturation_window)
        output = write_output(args.output or HERE / "analysis_outputs" / "diagnostics.json", result, [source], vars(args))
        print(output)
        return
    if args.output is not None:
        ap.error("--output requires --analyze-only")
    from wave2a.azure_client import build_client
    from wave2a.statefulpuzzle import require_statefulpuzzle_harness
    N_PAIRS = args.n
    HERE.mkdir(parents=True, exist_ok=True)
    require_statefulpuzzle_harness()
    client = build_client()
    t0 = time.perf_counter(); total_cost = 0.0; all_pairs = []
    log = (HERE / "run.log").open("w")
    log.write(f"exp_7_1 D_levels={D_LEVELS} n_pairs={N_PAIRS} H={H} rho={RHO}\n"); log.flush()
    for D in D_LEVELS:
        pair_rows = [None] * N_PAIRS; done = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futs = {pool.submit(_run_pair, client, D, j): j for j in range(N_PAIRS)}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    log.write(f"  D={D}[{j}] EX {type(e).__name__}: {e}\n"); log.flush(); continue
                pair_rows[j] = rec; total_cost += rec["cost_usd"]; done += 1
                if done % 5 == 0: log.write(f"  D={D} {done}/{N_PAIRS} cost=${total_cost:.4f}\n"); log.flush()
        for rec in pair_rows:
            if rec: all_pairs.append(rec)
        log.write(f"D_DONE D={D} cost=${total_cost:.4f}\n"); log.flush()
    log.close()
    by_D = {}
    for r in all_pairs: by_D.setdefault(r["D"], []).append(r)
    per_D = {}
    for D, rs in sorted(by_D.items()):
        n = len(rs)
        pow_beta = [r["fit"]["power"]["beta"] for r in rs if r["fit"].get("power")]
        exp_lam = [r["fit"]["exp"]["lambda"] for r in rs if r["fit"].get("exp")]
        best_power = sum(1 for r in rs if r["fit"].get("best") == "power")
        d_aic = [r["fit"].get("delta_aic_exp_minus_power") for r in rs
                 if r["fit"].get("delta_aic_exp_minus_power") is not None]
        per_D[D] = {"n_pairs": n,
            "power_beta_mean": float(np.mean(pow_beta)) if pow_beta else None,
            "exp_lambda_mean": float(np.mean(exp_lam)) if exp_lam else None,
            "frac_power_beats_exp": best_power/max(1,n),
            "mean_delta_aic_exp_minus_power": float(np.mean(d_aic)) if d_aic else None,
            "collapse_rate_either": float(np.mean([max(r["C_A"],r["C_B"]) for r in rs]))}
    summary = {"exp_id": "exp_7_1", "n_pairs_total": len(all_pairs),
               "wall_seconds": round(time.perf_counter()-t0,1), "total_cost_usd": round(total_cost,6)}
    with (HERE/"results.json").open("w") as f: json.dump({"summary": summary, "pairs": all_pairs}, f)
    aggregates = {"per_D": per_D, "divergence_diagnostics": analyze_divergence(all_pairs, saturation_tolerance=args.saturation_tolerance, saturation_window=args.saturation_window)}
    with (HERE/"aggregates.json").open("w") as f: json.dump(aggregates, f, indent=2)
    with (HERE/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(json.dumps({"summary": summary, "per_D": per_D}, indent=2))

if __name__ == "__main__": main()
