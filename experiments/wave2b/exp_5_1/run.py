from __future__ import annotations
import json
import sys
from pathlib import Path
import math
import warnings
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import auroc, read_input, write_output


CONF_KWD = ("cancel", "refund", "downgrade", "conflict")


def classify_task_from_row(row, raw):
    n = row["n_gold_actions"]
    instr = (raw.get("instruction") or "").lower()
    if n <= 1:
        return "single_app"
    if n == 2:
        return "two_app"
    if n == 3:
        return "three_app"
    if any(k in instr for k in CONF_KWD):
        return "conflicting_constraints"
    return "cross_app_update"


def _sigma_slope_pre_collapse(sigma_series, T_col, window=3):
    if T_col >= len(sigma_series):
        return 0.0
    end = min(T_col, len(sigma_series))
    start = max(0, end - window)
    if end - start < 2:
        return 0.0
    xs = np.arange(start, end)
    ys = np.array(sigma_series[start:end])
    if np.ptp(ys) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def analyze(rows, raws_by_task):
    if not rows:
        return {"n": 0}
    by_dep = {}
    for r in rows:
        raw = raws_by_task.get(r["task_id"], {})
        dep = classify_task_from_row(r, raw)
        by_dep.setdefault(dep, []).append(r)
    per_dep = {}
    all_sigma_early = []
    all_F_drop = []
    for dep, rs in by_dep.items():
        sigma_slopes = [_sigma_slope_pre_collapse(r["sigma_series"], r["T_col"]) for r in rs]
        sigma_early = []
        F_drop = []
        for r in rs:
            n = len(r["sigma_series"])
            first_half = r["sigma_series"][: n // 2] if n >= 2 else r["sigma_series"]
            sigma_early.append(float(np.mean(first_half)) if first_half else 0.0)
            F_drop.append(1 if min(r["F_series"] or [1.0]) < 0.5 else 0)
        all_sigma_early.extend(sigma_early)
        all_F_drop.extend(F_drop)
        per_dep[dep] = {
            "n": len(rs),
            "mean_sigma_slope_pre_collapse": float(np.mean(sigma_slopes)),
            "collapse_rate": float(np.mean([r["collapse_indicator"] for r in rs])),
            "mean_reward": float(np.mean([r["reward"] for r in rs])),
            "mean_sigma_early": float(np.mean(sigma_early)),
            "F_drop_rate": float(np.mean(F_drop)),
        }
    au = auroc(np.array(all_F_drop), np.array(all_sigma_early))
    return {"per_dep_class": per_dep,
            "AUROC_sigma_early_predicts_F_drop": au}



def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _finite_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _eligible_row(row, prefix_steps):
    if not isinstance(row, Mapping):
        return None, "row_not_mapping"
    stop_reason = str(row.get("stop_reason") or "").lower()
    if row.get("error") or any(marker in stop_reason for marker in ("llm_error", "api_error", "runtime_error")):
        return None, "runner_error"
    reward = _finite_number(row.get("reward"))
    if reward is None or not 0.0 <= reward <= 1.0:
        return None, "missing_or_invalid_reward"
    difficulty = _finite_number(row.get("n_gold_actions"))
    if difficulty is None or difficulty < 0 or not difficulty.is_integer():
        return None, "missing_or_invalid_n_gold_actions"
    domain = row.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        return None, "missing_or_invalid_domain"
    task_id = row.get("task_id")
    if isinstance(task_id, (bool, np.bool_)) or not isinstance(task_id, (str, Integral)):
        return None, "missing_or_invalid_task_id"
    task_id = str(task_id).strip()
    if not task_id:
        return None, "missing_or_invalid_task_id"
    series = row.get("sigma_series")
    if not isinstance(series, (list, tuple, np.ndarray)):
        return None, "missing_or_invalid_sigma_series"
    if isinstance(series, np.ndarray) and series.ndim != 1:
        return None, "missing_or_invalid_sigma_series"
    if len(series) < prefix_steps:
        return None, "fewer_than_prefix_steps"
    if len(series) == prefix_steps:
        return None, "no_post_prefix_observation"
    prefix = [_finite_number(value) for value in series[:prefix_steps]]
    if any(value is None or value < 0 for value in prefix):
        return None, "nonfinite_nonnumeric_or_negative_prefix_stress"
    early_stress = math.fsum(value / prefix_steps for value in prefix)
    if not math.isfinite(early_stress):
        return None, "nonfinite_prefix_mean"
    return {
        "domain": domain.strip(),
        "task_id": task_id,
        "reward": reward,
        "failure": int(reward < 1.0),
        "early_stress": early_stress,
        "n_gold_actions": difficulty,
    }, None


def analyze_early_stress(rows, prefix_steps=5, n_splits=5, seed=42):
    prefix_steps = _integer(prefix_steps, "prefix_steps", 1)
    n_splits = _integer(n_splits, "n_splits", 2)
    seed = _integer(seed, "seed", 0)
    if seed > 2**32 - 1:
        raise ValueError("seed must be at most 2**32 - 1")
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("rows must be an iterable of trajectory mappings")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise ValueError("rows must be an iterable of trajectory mappings") from exc
    eligible = []
    exclusions = Counter()
    n_input = 0
    for index, row in enumerate(iterator):
        n_input += 1
        record, reason = _eligible_row(row, prefix_steps)
        if reason is not None:
            exclusions[reason] += 1
        else:
            record["input_index"] = index
            eligible.append(record)
    group_keys = [(row["domain"], row["task_id"]) for row in eligible]
    group_lookup = {key: index for index, key in enumerate(sorted(set(group_keys)))}
    groups = np.asarray([group_lookup[key] for key in group_keys], dtype=int)
    labels = np.asarray([row["failure"] for row in eligible], dtype=int)
    features = np.asarray([[row["early_stress"], row["n_gold_actions"]] for row in eligible], dtype=float).reshape(-1, 2)
    class_counts = {str(label): int(np.sum(labels == label)) for label in (0, 1)}
    class_groups = {str(label): set(groups[labels == label].tolist()) for label in (0, 1)}
    cv = {
        "status": "insufficient",
        "reason": None,
        "splitter": "StratifiedGroupKFold",
        "n_splits_requested": n_splits,
        "n_splits_used": 0,
        "shuffle": True,
        "preprocessing": "StandardScaler_fit_on_each_training_fold_only",
        "estimator": {"name": "LogisticRegression", "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 0.0001, "class_weight": None},
        "hyperparameter_selection": "fixed_no_tuning",
        "folds": [],
        "models": {},
        "oof_predictions": [],
    }
    result = {
        "status": "insufficient",
        "reason": None,
        "prefix_steps": prefix_steps,
        "seed": seed,
        "definitions": {
            "positive_label": "benchmark_failure_reward_less_than_1",
            "negative_label": "benchmark_success_reward_equal_to_1",
            "accepted_reward_range": [0.0, 1.0],
            "stress": "arithmetic_mean_of_sigma_series_first_prefix_steps",
            "difficulty": "exogenous_n_gold_actions",
            "group": ["domain", "task_id"],
            "metric_unit": "eligible_trajectory",
            "raw_score_direction": "higher_score_predicts_failure_without_posthoc_sign_selection",
            "cohort": "complete_cases_with_at_least_one_observation_after_the_fixed_prefix_no_imputation",
            "exclusion_counting": "first_failed_eligibility_check_per_row",
        },
        "counts": {
            "input_rows": n_input,
            "eligible_rows": len(eligible),
            "excluded_rows": sum(exclusions.values()),
            "exclusions_by_reason": dict(sorted(exclusions.items())),
            "class_rows": class_counts,
            "groups": len(group_lookup),
            "groups_containing_each_class": {key: len(value) for key, value in class_groups.items()},
            "mixed_outcome_groups": len(class_groups["0"] & class_groups["1"]),
            "fractional_reward_rows": sum(0.0 < row["reward"] < 1.0 for row in eligible),
        },
        "raw_auroc": {"status": "insufficient", "stress": None, "difficulty": None},
        "grouped_oof": cv,
    }
    if not eligible:
        result["reason"] = cv["reason"] = "no_eligible_rows"
        return result
    if min(class_counts.values()) == 0:
        result["reason"] = cv["reason"] = "both_outcome_classes_are_required"
        return result
    result["raw_auroc"] = {
        "status": "ok",
        "stress": float(roc_auc_score(labels, features[:, 0])),
        "difficulty": float(roc_auc_score(labels, features[:, 1])),
    }
    if len(group_lookup) < n_splits:
        result["reason"] = cv["reason"] = "number_of_groups_below_requested_n_splits"
        return result
    if min(class_counts.values()) < n_splits:
        result["reason"] = cv["reason"] = "class_sample_count_below_requested_n_splits"
        return result
    if min(len(value) for value in class_groups.values()) < n_splits:
        result["reason"] = cv["reason"] = "class_group_count_below_requested_n_splits"
        return result
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split(features, labels, groups))
    except ValueError as exc:
        result["reason"] = cv["reason"] = "grouped_stratification_failed"
        cv["error_type"] = type(exc).__name__
        cv["error"] = str(exc)
        return result
    fold_ids = np.full(len(eligible), -1, dtype=int)
    invalid_folds = []
    for fold, (train, test) in enumerate(splits):
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())
        train_counts = {str(label): int(np.sum(labels[train] == label)) for label in (0, 1)}
        test_counts = {str(label): int(np.sum(labels[test] == label)) for label in (0, 1)}
        if not len(train) or not len(test) or min(train_counts.values()) == 0 or min(test_counts.values()) == 0:
            invalid_folds.append(fold)
        if train_groups & test_groups or np.any(fold_ids[test] >= 0):
            result["status"] = cv["status"] = "error"
            result["reason"] = cv["reason"] = "invalid_group_partition"
            return result
        fold_ids[test] = fold
        cv["folds"].append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "train_class_rows": train_counts,
            "test_class_rows": test_counts,
            "group_overlap": 0,
        })
    if invalid_folds:
        result["reason"] = cv["reason"] = "fixed_split_produced_folds_without_both_classes"
        cv["invalid_folds"] = invalid_folds
        return result
    if len(splits) != n_splits or np.any(fold_ids < 0):
        result["status"] = cv["status"] = "error"
        result["reason"] = cv["reason"] = "incomplete_oof_partition"
        return result
    cv["n_splits_used"] = n_splits
    model_features = {"stress_only": [0], "difficulty_only": [1], "combined": [0, 1]}
    feature_names = ["early_stress", "n_gold_actions"]
    predictions = {}
    for name, columns in model_features.items():
        model_result = {"status": "ok", "reason": None, "features": [feature_names[index] for index in columns], "auroc": None, "fold_aurocs": [], "n_oof_predictions": 0}
        prediction = np.full(len(eligible), np.nan, dtype=float)
        for fold, (train, test) in enumerate(splits):
            estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, tol=0.0001, random_state=seed))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    warnings.simplefilter("error", RuntimeWarning)
                    estimator.fit(features[train][:, columns], labels[train])
                    fold_prediction = estimator.predict_proba(features[test][:, columns])[:, 1]
                if not np.all(np.isfinite(fold_prediction)) or np.any((fold_prediction < 0) | (fold_prediction > 1)):
                    raise ValueError("Nonfinite or invalid held-out failure probabilities")
                prediction[test] = fold_prediction
                model_result["fold_aurocs"].append({"fold": fold, "auroc": float(roc_auc_score(labels[test], fold_prediction))})
            except (ValueError, FloatingPointError, OverflowError, ConvergenceWarning, RuntimeWarning) as exc:
                model_result.update(status="error", reason="fold_estimation_failed", failed_fold=fold, error_type=type(exc).__name__, error=str(exc))
                break
        model_result["n_oof_predictions"] = int(np.sum(np.isfinite(prediction)))
        if model_result["status"] == "ok":
            if not np.all(np.isfinite(prediction)):
                model_result.update(status="error", reason="incomplete_oof_predictions")
            else:
                model_result["auroc"] = float(roc_auc_score(labels, prediction))
                predictions[name] = prediction.tolist()
        cv["models"][name] = model_result
    for index, row in enumerate(eligible):
        cv["oof_predictions"].append({
            "input_index": row["input_index"],
            "domain": row["domain"],
            "task_id": row["task_id"],
            "fold": int(fold_ids[index]),
            "failure": row["failure"],
            **{name: predictions[name][index] if name in predictions else None for name in model_features},
        })
    if len(predictions) != len(model_features):
        result["status"] = cv["status"] = "error"
        result["reason"] = cv["reason"] = "one_or_more_models_failed"
    else:
        result["status"] = cv["status"] = "ok"
        result["reason"] = cv["reason"] = None
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str,
                    default=str(HERE.parent / "exp_1_2/results.json"))
    ap.add_argument("--raw", type=str,
                    default=str(HERE.parent / "exp_1_2/trajectories.jsonl"))
    ap.add_argument("--prefix-steps", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--analysis-seed", type=int, default=42)
    ap.add_argument("--output")
    args = ap.parse_args()
    src_p = Path(args.source)
    raw_p = Path(args.raw)
    if not src_p.exists():
        print(f"ERROR: source {src_p} not found. Run exp_1_2 first.")
        sys.exit(2)
    try:
        data, source = read_input(src_p)
        raws_by_task = {}
        if raw_p.exists():
            for line in raw_p.open():
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    raws_by_task[raw["task_id"]] = raw
                except Exception:
                    pass
        agg = analyze(data["rows"], raws_by_task)
        agg["fixed_prefix_reward_error"] = analyze_early_stress(data["rows"], prefix_steps=args.prefix_steps, n_splits=args.n_splits, seed=args.analysis_seed)
        summary = {"exp_id": "exp_5_1_secondary",
                   "n_rows": len(data["rows"]),
                   "n_raws": len(raws_by_task),
                   "source": str(src_p),
                   "total_cost_usd": 0.0}
        result = {"summary": summary, "aggregates": agg}
        if args.output:
            parameters = {"raw": str(raw_p), "prefix_steps": args.prefix_steps, "n_splits": args.n_splits, "analysis_seed": args.analysis_seed}
            output = write_output(args.output, result, [source], parameters)
            print(str(output))
        else:
            (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2, allow_nan=False))
            (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except (ValueError, TypeError, OSError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
