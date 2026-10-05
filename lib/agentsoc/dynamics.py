from __future__ import annotations

import math
from collections import Counter
from numbers import Integral

import numpy as np

from wave2a.metrics import fit_power_and_exp


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


def _invalid(reason):
    return {"valid": False, "reason": reason}


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


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


def _residence_runs(states, basin):
    runs = []
    start = 0
    while start < len(states):
        if states[start] != basin:
            start += 1
            continue
        stop = start + 1
        while stop < len(states) and states[stop] == basin:
            stop += 1
        left = start == 0
        right = stop == len(states)
        runs.append({
            "start_index": start,
            "last_index": stop - 1,
            "exit_index": None if right else stop,
            "n_observed_states": stop - start,
            "observed_duration_steps": stop - start - int(right),
            "left_censored": left,
            "right_censored": right,
            "completed": not left and not right,
        })
        start = stop
    return runs


def _basin_summary(runs, at_risk, exits):
    complete = [run["observed_duration_steps"] for run in runs if run["completed"]]
    censored = [run["observed_duration_steps"] for run in runs if not run["completed"]]
    return {
        "n_at_risk_transitions": at_risk,
        "n_exits": exits,
        "one_step_exit_probability": exits / at_risk if at_risk else None,
        "n_residence_runs": len(runs),
        "n_completed_runs": len(complete),
        "n_censored_runs": len(censored),
        "n_left_censored_runs": sum(run["left_censored"] for run in runs),
        "n_right_censored_runs": sum(run["right_censored"] for run in runs),
        "completed_residence_steps": complete,
        "censored_observed_duration_steps": censored,
        "mean_completed_residence_steps": _mean([float(value) for value in complete]),
    }


def analyze_recovery(rows, wrong_threshold=0.5, correct_threshold=0.9):
    wrong = _number(wrong_threshold)
    correct = _number(correct_threshold)
    if wrong is None or correct is None or not 0 <= wrong < correct <= 1:
        raise ValueError("Thresholds must satisfy 0 <= wrong_threshold < correct_threshold <= 1.")
    records = []
    all_runs = {"wrong": [], "correct": []}
    transitions = {source: {target: 0 for target in ("wrong", "intermediate", "correct")} for source in ("wrong", "intermediate", "correct")}
    exclusion_counts = Counter()
    for index, row in enumerate(rows):
        record = {
            "row_index": index,
            "seed": _metadata(row.get("seed")) if isinstance(row, dict) else None,
            "cell_id": _metadata(row.get("cell_id")) if isinstance(row, dict) else None,
        }
        values, reason = _series(row, "F_series")
        if reason is None and np.any((values < 0) | (values > 1)):
            reason = "fidelity_outside_unit_interval"
        if reason is None and values.size < 2:
            reason = "fewer_than_two_fidelity_observations"
        if reason is not None:
            record.update({"valid": False, "reason": reason})
            exclusion_counts[reason] += 1
            records.append(record)
            continue
        states = ["wrong" if value < wrong else "correct" if value >= correct else "intermediate" for value in values]
        row_transitions = {source: {target: 0 for target in transitions} for source in transitions}
        for source, target in zip(states, states[1:]):
            row_transitions[source][target] += 1
            transitions[source][target] += 1
        record.update({"valid": True, "reason": None, "n_steps": len(states), "transition_counts": row_transitions, "basins": {}})
        for basin in ("wrong", "correct"):
            runs = _residence_runs(states, basin)
            all_runs[basin].extend(runs)
            at_risk = sum(row_transitions[basin].values())
            exits = at_risk - row_transitions[basin][basin]
            record["basins"][basin] = {**_basin_summary(runs, at_risk, exits), "runs": runs}
        records.append(record)
    basins = {}
    for basin in ("wrong", "correct"):
        at_risk = sum(transitions[basin].values())
        exits = at_risk - transitions[basin][basin]
        basins[basin] = _basin_summary(all_runs[basin], at_risk, exits)
    n_valid = sum(record["valid"] for record in records)
    return {
        "analysis": "operational_state_fidelity_basins",
        "interpretation": "F_series is an observed state-fidelity proxy, not a measurement of latent beliefs; final rewards are not used to label steps",
        "protocol": {
            "wrong_threshold": wrong,
            "correct_threshold": correct,
            "wrong_basin": "F_t < wrong_threshold",
            "correct_basin": "F_t >= correct_threshold",
            "intermediate_basin": "wrong_threshold <= F_t < correct_threshold",
            "exit_event": "the next observed step leaves the source basin, including transitions into the intermediate basin",
            "denominator": "within-trajectory transitions whose source observation belongs to the basin and whose next observation exists",
            "residence_unit": "observed interaction-step transitions; a terminal censored run has one fewer observed transitions than states",
            "censoring": "runs starting at the first observation are left-censored; runs ending at the last observation are right-censored; completed runs have neither condition",
            "invalid_series_policy": "exclude the entire series; never bridge missing or nonfinite observations",
            "residence_summary": "completed-run means are descriptive and do not estimate a censoring-adjusted mean residence time",
        },
        "n_rows": len(records),
        "n_valid_rows": n_valid,
        "n_excluded_rows": len(records) - n_valid,
        "exclusions": dict(sorted(exclusion_counts.items())),
        "transition_counts": transitions,
        "basins": basins,
        "rows": records,
    }
