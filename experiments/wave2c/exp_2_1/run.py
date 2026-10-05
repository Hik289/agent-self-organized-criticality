from __future__ import annotations
import json
import math
from collections import Counter
from numbers import Integral
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2a.metrics import read_input, records, write_output


def _visibility_prompts():
    from wave2c.alfworld_runner import DEFAULT_AGENT_SYSTEM
    return {
        "full": DEFAULT_AGENT_SYSTEM,
        "partial": (
            DEFAULT_AGENT_SYSTEM +
            "\n\nNote: Some observations may be incomplete or partially "
            "obscured. If you are unsure of the current state, use look/examine "
            "actions to verify before mutating."
        ),
        "delayed": (
            DEFAULT_AGENT_SYSTEM +
            "\n\nNote: Your observations reflect state as of a few steps ago; "
            "the actual current state may be slightly different. Prefer safe "
            "verification actions (look, examine) before mutating actions."
        ),
    }


CANONICAL_TYPES = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]


def build_task_list(per_cell: int, subset_types: list = None):
    from wave2c.alfworld_runner import list_games_by_type, CONFIG_PATH
    visibility_prompts = _visibility_prompts()
    games_by = list_games_by_type(CONFIG_PATH)
    types_to_use = subset_types or CANONICAL_TYPES
    tasks = []
    prompts_by_game = {}
    for tt in types_to_use:
        games = games_by.get(tt, [])[:per_cell]
        for vis, prompt in visibility_prompts.items():
            for g in games:
                tasks.append((g, {"task_type": tt, "visibility": vis,
                                    "game_file": g}))
                prompts_by_game[(g, vis)] = prompt
    return tasks, prompts_by_game


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

def analyze(rows):
    by_vis = {}
    for r in rows:
        by_vis.setdefault(r["cell"]["visibility"], []).append(r)
    per_vis = {}
    for vis, rs in by_vis.items():
        R = np.array([r["reward"] for r in rs])
        S = np.array([r["n_steps"] for r in rs])
        C = np.array([r["collapse_indicator"] for r in rs])

        residence = []
        for r in rs:
            sigma = r["sigma_series"]

            runs = 0
            in_run = False
            for s in sigma:
                if s >= 2:
                    if not in_run:
                        runs += 1
                        in_run = True
                else:
                    in_run = False
            residence.append(runs)
        per_vis[vis] = {
            "n": len(rs),
            "mean_reward": float(R.mean()),
            "success_rate": float((R >= 0.5).mean()),
            "mean_steps": float(S.mean()),
            "collapse_rate": float(C.mean()),
            "mean_wrong_basin_residence_episodes": float(np.mean(residence)),
        }
    return {"per_visibility": per_vis}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=15)
    ap.add_argument("--subset", type=str, default="",
                    help="comma-separated task_types to use (default all 6)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--source", default=str(HERE / "results.json"))
    ap.add_argument("--output", help="New offline diagnostic output path; existing files are never overwritten.")
    ap.add_argument("--wrong-threshold", type=float, default=0.5)
    ap.add_argument("--correct-threshold", type=float, default=0.9)
    args = ap.parse_args()
    if args.analyze_only:
        data, source = read_input(args.source)
        result = analyze_recovery(records(data, "rows"), wrong_threshold=args.wrong_threshold, correct_threshold=args.correct_threshold)
        output = write_output(args.output or HERE / "analysis_outputs" / "diagnostics.json", result, [source], vars(args))
        print(output)
        return
    if args.output is not None:
        ap.error("--output requires --analyze-only")
    from wave2c.alfworld_batch_runner import run_experiment
    visibility_prompts = _visibility_prompts()

    subset = [s.strip() for s in args.subset.split(",") if s.strip()] or None


    from wave2c import alfworld_batch_runner as br
    _orig_run_one = br._run_one

    def _run_one_with_prompt(game_file, cell_meta, *, client,
                             system_prompt=None, max_steps=30):

        vis = cell_meta.get("visibility")
        if vis and vis in visibility_prompts:
            system_prompt = visibility_prompts[vis]
        return _orig_run_one(game_file, cell_meta, client=client,
                             system_prompt=system_prompt, max_steps=max_steps)
    br._run_one = _run_one_with_prompt

    tasks, _ = build_task_list(args.per_cell, subset)
    print(f"Running {len(tasks)} tasks (per_cell={args.per_cell}, "
          f"subset={subset or 'ALL'}) ...")
    summary = run_experiment(
        exp_id="exp_2_1", tasks=tasks, out_dir=HERE,
        seed=42, max_steps=30, save_raw=True,
    )
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    agg["recovery_diagnostics"] = analyze_recovery(data["rows"], wrong_threshold=args.wrong_threshold, correct_threshold=args.correct_threshold)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
