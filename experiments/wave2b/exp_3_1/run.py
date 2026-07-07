"""Exp 3.1 (Part III): local valid, global wrong — Airline.

Airline test split (50 tasks), stratified by dependency depth:
  single_step (n_gold_actions=1)
  two_step (n_gold_actions=2)
  multi_constraint (n_gold_actions=3-4)
  policy_conflict (n_gold_actions=5-6 OR instruction contains cancel/refund/downgrade)
  state_changing (n_gold_actions>=7)

Note: due to airline test set being only 50 tasks (not the spec's 125), we
run ALL 50 tasks and post-hoc classify. Cell counts may be uneven.

Success: F drops before L, Δ^LG > 0 at policy_conflict + state_changing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment


DEP_KEYWORDS = ("cancel", "refund", "downgrade", "policy", "conflict")


def classify_task(task) -> str:
    n = len(task.actions)
    instr = (task.instruction or "").lower()
    if n == 0:
        return "info_only"
    if n == 1:
        return "single_step"
    if n == 2:
        return "two_step"
    if n <= 4:
        return "multi_constraint"
    if any(k in instr for k in DEP_KEYWORDS) or n <= 6:
        return "policy_conflict"
    return "state_changing"


def build_task_list(n_max: int = 50):
    from tau_bench.envs.airline.tasks_test import TASKS
    tasks = []
    for i, t in enumerate(TASKS[:n_max]):
        dep = classify_task(t)
        tasks.append((i, {"task_index": i, "dep_class": dep,
                          "n_gold_actions": len(t.actions)}))
    return tasks


def analyze(rows):
    if not rows:
        return {"n": 0}
    by_dep: dict = {}
    for r in rows:
        by_dep.setdefault(r["cell"]["dep_class"], []).append(r)
    per_dep = {}
    for dep, rs in by_dep.items():
        L = [np.mean(r["L_series"]) for r in rs]
        F = [np.mean(r["F_series"]) for r in rs]
        delta = [np.mean(r["delta_LG_series"]) for r in rs]
        T = np.array([r["T_col"] for r in rs])
        C = np.array([r["collapse_indicator"] for r in rs])
        # First invalid action time (first L=0)
        first_L0 = []
        for r in rs:
            for i, l in enumerate(r["L_series"]):
                if l == 0:
                    first_L0.append(i); break
            else:
                first_L0.append(len(r["L_series"]))
        # F collapse time = T_col
        per_dep[dep] = {
            "n": len(rs),
            "mean_L": float(np.mean(L)),
            "mean_F": float(np.mean(F)),
            "mean_delta_LG": float(np.mean(delta)),
            "T_col_mean": float(T.mean()),
            "first_L0_mean": float(np.mean(first_L0)),
            "collapse_rate": float(C.mean()),
            "F_lower_than_L": (float(np.mean(F)) < float(np.mean(L))),
        }
    return {"per_dep_class": per_dep, "overall_reward_mean": float(np.mean([r["reward"] for r in rows]))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    tasks = build_task_list(n_max=args.n)
    summary = run_experiment(
        exp_id="exp_3_1",
        tasks=tasks,
        out_dir=HERE,
        domain="airline",
        seed=42,
        max_num_steps=args.max_steps,
        concurrency=2,
        save_raw=True,
    )
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
