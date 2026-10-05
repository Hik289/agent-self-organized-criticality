from __future__ import annotations

from collections import Counter
from numbers import Integral

import numpy as np

from wave2a.metrics import dfa_exponent, spectral_slope

from .io import field
from .statistics import benjamini_hochberg


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
