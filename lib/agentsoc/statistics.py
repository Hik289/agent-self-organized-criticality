from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy import integrate, optimize, stats


def _vector(values, name, minimum=None, maximum=None):
    if values is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if minimum is not None and np.any(array < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and np.any(array > maximum):
        raise ValueError(f"{name} must be at most {maximum}")
    return array


def _threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("threshold must be strictly between zero and one")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def threshold_events(series, threshold=0.5):
    threshold = _threshold(threshold)
    values = _vector(series, "series", minimum=0.0, maximum=1.0)
    above = values > threshold
    changes = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    events = [
        {
            "start": int(start),
            "end": int(end),
            "duration": int(end - start + 1),
            "size": float(values[start : end + 1].sum()),
            "peak": float(values[start : end + 1].max()),
            "touches_start_boundary": bool(start == 0),
            "touches_end_boundary": bool(end == len(values) - 1),
        }
        for start, end in zip(starts, ends)
    ]
    return {
        "index_base": 0,
        "end_inclusive": True,
        "threshold": threshold,
        "comparison": ">",
        "size_definition": "sum_of_errors_in_contiguous_threshold_exceedance",
        "n_steps": int(values.size),
        "n_exceedances": int(above.sum()),
        "n_events": len(events),
        "events": events,
    }


def _indicator_summary(sequences):
    durations = []
    n_steps = sum(len(sequence) for sequence in sequences)
    n_exceedances = sum(int(sequence.sum()) for sequence in sequences)
    for sequence in sequences:
        changes = np.diff(np.r_[False, sequence, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        durations.extend((ends - starts).tolist())
    return {
        "n_steps": n_steps,
        "n_exceedances": n_exceedances,
        "n_events": len(durations),
        "exceedance_rate": float(n_exceedances / n_steps) if n_steps else None,
        "mean_duration": float(np.mean(durations)) if durations else None,
        "max_duration": max(durations) if durations else 0,
    }


def _null_summary(observed, simulated):
    output = {}
    for key in ("n_exceedances", "n_events", "mean_duration", "max_duration"):
        values = np.asarray([row[key] for row in simulated if row[key] is not None], dtype=float)
        observation = observed[key]
        complete = len(values) == len(simulated)
        output[key] = {
            "observed": observation,
            "n_identified_simulations": int(len(values)),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=1)) if values.size > 1 else None,
            "p025": float(np.quantile(values, 0.025)) if values.size else None,
            "p975": float(np.quantile(values, 0.975)) if values.size else None,
            "p_greater": float((1 + np.sum(values >= observation)) / (len(values) + 1))
            if observation is not None and complete and values.size
            else None,
            "p_less": float((1 + np.sum(values <= observation)) / (len(values) + 1))
            if observation is not None and complete and values.size
            else None,
        }
    return output


def avalanche_nulls(series_list, threshold=0.5, n_simulations=1000, seed=42):
    threshold = _threshold(threshold)
    n_simulations = _integer(n_simulations, "n_simulations", 1)
    seed = _integer(seed, "seed", 0)
    if series_list is None:
        raise ValueError("series_list is required")
    sequences = [
        _vector(values, f"series_list[{index}]", minimum=0.0, maximum=1.0) > threshold
        for index, values in enumerate(series_list)
    ]
    lengths = [len(sequence) for sequence in sequences]
    observed = _indicator_summary(sequences)
    result = {
        "threshold": threshold,
        "n_trajectories": len(sequences),
        "trajectory_lengths": lengths,
        "n_simulations": n_simulations,
        "seed": seed,
        "statistic_scale": "binary_threshold_exceedance_runs",
        "weighted_size_comparison": None,
        "pvalue_method": "plug_in_parametric_simulation_add_one",
        "observed": observed,
        "bernoulli": None,
        "markov": None,
    }
    if not observed["n_steps"]:
        result.update(status="unidentified", reason="no_observed_steps")
        return result
    rng_bernoulli, rng_markov = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    probability = observed["exceedance_rate"]
    bernoulli = [
        _indicator_summary([rng_bernoulli.random(length) < probability for length in lengths])
        for _ in range(n_simulations)
    ]
    result["bernoulli"] = {
        "status": "identified",
        "error_probability": probability,
        "matching": "pooled_exceedance_rate_and_individual_trajectory_lengths",
        "statistics": _null_summary(observed, bernoulli),
    }
    counts = np.zeros((2, 2), dtype=np.int64)
    initial = []
    for sequence in sequences:
        if len(sequence):
            initial.append(int(sequence[0]))
        if len(sequence) > 1:
            np.add.at(counts, (sequence[:-1].astype(int), sequence[1:].astype(int)), 1)
    row_counts = counts.sum(axis=1)
    initial_probability = float(np.mean(initial))
    result["markov"] = {
        "transition_counts": counts.tolist(),
        "initial_error_probability": initial_probability,
        "initialization": "empirical_first_state_distribution",
        "matching": "pooled_within_trajectory_transition_counts_and_individual_trajectory_lengths",
    }
    if np.any(row_counts == 0):
        result["markov"].update(
            status="unidentified",
            reason="at_least_one_transition_row_has_no_observations",
            transition_matrix=None,
            statistics=None,
        )
    else:
        transition = counts / row_counts[:, None]
        simulations = []
        for _ in range(n_simulations):
            paths = []
            for length in lengths:
                path = np.zeros(length, dtype=bool)
                if length:
                    path[0] = rng_markov.random() < initial_probability
                    uniforms = rng_markov.random(max(0, length - 1))
                    for index in range(1, length):
                        path[index] = uniforms[index - 1] < transition[int(path[index - 1]), 1]
                paths.append(path)
            simulations.append(_indicator_summary(paths))
        result["markov"].update(
            status="identified",
            transition_matrix=transition.tolist(),
            statistics=_null_summary(observed, simulations),
        )
    result["status"] = "identified"
    return result


def benjamini_hochberg(pvalues, q=0.05):
    q = float(q)
    if not math.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between zero and one")
    if pvalues is None:
        raise ValueError("pvalues is required")
    values = list(pvalues)
    valid_indices = []
    valid_values = []
    for index, value in enumerate(values):
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("pvalues must contain only probabilities or None")
        valid_indices.append(index)
        valid_values.append(value)
    adjusted = [None] * len(values)
    rejected = [None] * len(values)
    if valid_values:
        order = np.argsort(valid_values, kind="stable")
        ordered = np.asarray(valid_values)[order]
        correction = np.minimum.accumulate((ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1])[::-1]
        correction = np.minimum(correction, 1.0)
        for rank, position in enumerate(order):
            index = valid_indices[int(position)]
            adjusted[index] = float(correction[rank])
            rejected[index] = bool(correction[rank] <= q)
    return {
        "method": "Benjamini-Hochberg",
        "q": q,
        "n_tests": len(valid_values),
        "n_unidentified": len(values) - len(valid_values),
        "p_adjusted": adjusted,
        "rejected": rejected,
    }


def _ks_distance(sample, cdf):
    sample = np.sort(sample)
    theoretical = np.clip(np.asarray(cdf(sample), dtype=float), 0.0, 1.0)
    if theoretical.shape != sample.shape or not np.all(np.isfinite(theoretical)):
        return None
    n = len(sample)
    return float(max(np.max(np.arange(1, n + 1) / n - theoretical), np.max(theoretical - np.arange(n) / n)))


def _fit_record(log_likelihood, n, n_parameters, parameters, ks, optimization=None):
    aic = float(2 * n_parameters - 2 * log_likelihood)
    bic = float(n_parameters * math.log(n) - 2 * log_likelihood)
    if not all(math.isfinite(value) for value in (log_likelihood, aic, bic)) or ks is None:
        return {"status": "unidentified", "reason": "nonfinite_fit"}
    if not all(math.isfinite(float(value)) for value in parameters.values()):
        return {"status": "unidentified", "reason": "nonfinite_parameter"}
    result = {
        "status": "identified",
        "parameters": parameters,
        "n_parameters": n_parameters,
        "log_likelihood": float(log_likelihood),
        "aic": aic,
        "bic": bic,
        "ks_statistic": ks,
        "ks_pvalue": None,
        "ks_interpretation": "descriptive_distance_with_parameters_fitted_on_the_same_tail",
    }
    if optimization is not None:
        result["optimization"] = optimization
    return result


def _truncated_normalizer(alpha, rate):
    upper = math.log1p((max(0.0, 1.0 - alpha) + 80.0) / rate)
    mode = max(0.0, math.log((1.0 - alpha) / rate)) if alpha < 1.0 else 0.0
    offset = (1.0 - alpha) * mode - rate * math.expm1(mode)
    integral, error = integrate.quad(
        lambda value: math.exp((1.0 - alpha) * value - rate * math.expm1(value) - offset),
        0.0,
        upper,
        epsabs=1e-10,
        epsrel=1e-9,
        limit=100,
    )
    if not math.isfinite(integral) or integral <= 0.0 or error > max(1e-8, integral * 1e-5):
        return None
    return offset + math.log(integral), offset, upper


def fit_tail_models(samples, xmin=None, min_tail_size=20):
    min_tail_size = _integer(min_tail_size, "min_tail_size", 3)
    values = _vector(samples, "samples", minimum=0.0)
    positives = values[values > 0.0]
    if xmin is not None:
        xmin = float(xmin)
        if not math.isfinite(xmin) or xmin <= 0:
            raise ValueError("xmin must be positive and finite")
    result = {
        "model_kind": "continuous_left_truncated_densities",
        "n_input": int(len(values)),
        "n_zero_excluded": int(np.sum(values == 0.0)),
        "min_tail_size": min_tail_size,
        "xmin_selection": "specified" if xmin is not None else "minimum_positive_observation",
        "xmin": xmin,
        "models": {},
    }
    if not positives.size:
        result.update(status="unidentified", reason="no_positive_samples", n_tail=0)
        return result
    xmin = float(positives.min()) if xmin is None else xmin
    tail = positives[positives >= xmin]
    result.update(xmin=xmin, n_tail=int(len(tail)), n_positive_below_xmin=int(np.sum(positives < xmin)))
    if len(tail) < min_tail_size:
        result.update(status="unidentified", reason="insufficient_tail_samples")
        return result
    if np.ptp(tail) == 0.0:
        result.update(status="unidentified", reason="constant_tail_has_no_identifiable_continuous_scale")
        return result
    scaled = tail / xmin
    if not np.all(np.isfinite(scaled)):
        result.update(status="unidentified", reason="tail_dynamic_range_exceeds_float_support")
        return result
    logs = np.log(scaled)
    n = len(tail)
    sum_logs = float(logs.sum())
    sum_excess = float((scaled - 1.0).sum())
    if not math.isfinite(sum_logs) or not math.isfinite(sum_excess) or sum_logs <= 0.0 or sum_excess <= 0.0:
        result.update(status="unidentified", reason="tail_scale_not_numerically_identifiable")
        return result
    log_scale = math.log(xmin)
    alpha = 1.0 + n / sum_logs
    likelihood = n * math.log(alpha - 1.0) - alpha * sum_logs - n * log_scale
    result["models"]["power_law"] = _fit_record(
        likelihood,
        n,
        1,
        {"alpha": float(alpha)},
        _ks_distance(scaled, lambda x: -np.expm1((1.0 - alpha) * np.log(x))),
    )
    exp_rate = n / sum_excess
    result["models"]["exponential"] = _fit_record(
        n * math.log(exp_rate) - exp_rate * sum_excess - n * log_scale,
        n,
        1,
        {"rate": float(exp_rate / xmin)},
        _ks_distance(scaled, lambda x: -np.expm1(-exp_rate * (x - 1.0))),
    )

    def lognormal_objective(parameters):
        mu, log_sigma = parameters
        sigma = math.exp(log_sigma)
        log_survival = stats.norm.logsf(-mu / sigma)
        likelihood = -sum_logs - n * (log_sigma + 0.5 * math.log(2.0 * math.pi) + log_survival)
        likelihood -= float(np.square((logs - mu) / sigma).sum()) / 2.0
        return -likelihood if math.isfinite(likelihood) else np.inf

    initial_sigma = max(float(logs.std()), 1e-4)
    lognormal = optimize.minimize(
        lognormal_objective,
        [float(logs.mean()), math.log(initial_sigma)],
        method="L-BFGS-B",
        bounds=[(-100.0, 100.0), (-10.0, 10.0)],
    )
    if lognormal.success and math.isfinite(float(lognormal.fun)):
        mu, log_sigma = map(float, lognormal.x)
        sigma = math.exp(log_sigma)
        normalization = stats.norm.logsf(-mu / sigma)
        boundary = abs(mu) >= 99.999 or abs(log_sigma) >= 9.999
        result["models"]["lognormal"] = _fit_record(
            -float(lognormal.fun) - n * log_scale,
            n,
            2,
            {"mu_log_x": mu + log_scale, "sigma_log_x": sigma},
            _ks_distance(scaled, lambda x: -np.expm1(stats.norm.logsf((np.log(x) - mu) / sigma) - normalization)),
            {
                "converged": True,
                "parameter_bounds": {"mu_minus_log_xmin": [-100.0, 100.0], "log_sigma": [-10.0, 10.0]},
                "boundary_solution": boundary,
            },
        )
    else:
        result["models"]["lognormal"] = {"status": "unidentified", "reason": "optimization_failed"}

    def truncated_objective(parameters):
        exponent, log_rate = map(float, parameters)
        rate = math.exp(log_rate)
        normalization = _truncated_normalizer(exponent, rate)
        if normalization is None:
            return np.inf
        return exponent * sum_logs + rate * sum_excess + n * normalization[0]

    candidates = []
    for initial in ((min(alpha, 99.0), math.log(0.01)), (0.1, math.log(min(max(exp_rate, 1e-7), 999.0))), (2.0, math.log(0.1))):
        fitted = optimize.minimize(
            truncated_objective,
            initial,
            method="L-BFGS-B",
            bounds=[(0.0, 100.0), (math.log(1e-8), math.log(1e3))],
        )
        if fitted.success and math.isfinite(float(fitted.fun)):
            candidates.append(fitted)
    if candidates:
        truncated = min(candidates, key=lambda candidate: candidate.fun)
        exponent, log_rate = map(float, truncated.x)
        rate = math.exp(log_rate)
        normalization = _truncated_normalizer(exponent, rate)
        log_normalizer, offset, upper = normalization

        def truncated_cdf(x):
            output = []
            for value in x:
                bound = min(math.log(float(value)), upper)
                integral = integrate.quad(
                    lambda t: math.exp((1.0 - exponent) * t - rate * math.expm1(t) - offset),
                    0.0,
                    bound,
                    epsabs=1e-10,
                    epsrel=1e-9,
                    limit=100,
                )[0]
                output.append(integral * math.exp(offset - log_normalizer))
            return np.asarray(output)

        result["models"]["truncated_power_law"] = _fit_record(
            -float(truncated.fun) - n * log_scale,
            n,
            2,
            {"alpha": exponent, "rate": rate / xmin},
            _ks_distance(scaled, truncated_cdf),
            {
                "converged": True,
                "parameter_bounds": {"alpha": [0.0, 100.0], "rate_times_xmin": [1e-8, 1e3]},
                "boundary_solution": exponent <= 1e-6 or exponent >= 99.999 or rate <= 1.001e-8 or rate >= 999.9,
            },
        )
    else:
        result["models"]["truncated_power_law"] = {"status": "unidentified", "reason": "optimization_failed"}
    identified = {name: fit for name, fit in result["models"].items() if fit["status"] == "identified"}
    if identified:
        minimum_aic = min(fit["aic"] for fit in identified.values())
        minimum_bic = min(fit["bic"] for fit in identified.values())
        for fit in identified.values():
            fit["delta_aic"] = fit["aic"] - minimum_aic
            fit["delta_bic"] = fit["bic"] - minimum_bic
        result.update(
            status="identified",
            best_aic=min(identified, key=lambda name: identified[name]["aic"]),
            best_bic=min(identified, key=lambda name: identified[name]["bic"]),
        )
    else:
        result.update(status="unidentified", reason="all_models_unidentified")
    return result
