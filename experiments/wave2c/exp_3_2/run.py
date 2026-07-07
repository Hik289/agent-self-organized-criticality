"""Exp 3.2 (Part III): local valid, globally harmful — GAIA.

30 Level-1 tasks (no attachment), single seed=42, temp=0.0.
Post-hoc classify each step by step_type (initial_search, source_selection,
evidence_extraction, calculation, intermediate_conclusion, final_answer).
Success: Δ^LG > 0 in evidence_extraction step type.

Note: In single-model GAIA, step_type is determined by which tool the LLM
called, not by an external classifier.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client
from lib.gaia_runner import load_gaia_level1, run_one_task


def _local_judge_step(step: dict) -> int:
    """L=1 if action was a valid tool call with parseable kwargs. GAIA agent
    always has admissible = TOOLS_SCHEMA so L=1 by default (Part III thesis
    predicts L stays high but F drops)."""
    action = step.get("action_name") or ""
    if not action:
        return 0
    if action in ("text_response",):
        return 1
    kwargs = step.get("action_kwargs") or {}
    if action in ("web_search", "browse", "read_file"):
        # need at least one arg
        if not kwargs:
            return 0
    if action == "calc" and not kwargs.get("expression"):
        return 0
    if action == "final_answer" and not kwargs.get("answer"):
        return 0
    return 1


def _global_F_at_step(step_i: int, total_steps: int, correct: int) -> float:
    """F approx: 0 until final; then 1 iff correct."""
    if step_i < total_steps - 1:
        return 0.5  # in-progress baseline
    return float(correct)


def analyze(rows):
    # per-step-type aggregation across all rows
    by_step_type = {}
    for r in rows:
        n_steps = r["n_steps"]
        for step in r["step_events"]:
            st = step.get("step_type", "unknown")
            L = _local_judge_step(step)
            F = _global_F_at_step(step["step"], n_steps, r["correct"])
            by_step_type.setdefault(st, []).append({
                "L": L, "F": F, "delta_LG": L - F,
                "correct_traj": r["correct"],
            })

    per_st = {}
    for st, items in by_step_type.items():
        Ls = [i["L"] for i in items]
        Fs = [i["F"] for i in items]
        deltas = [i["delta_LG"] for i in items]
        per_st[st] = {
            "n_steps": len(items),
            "mean_L": float(np.mean(Ls)),
            "mean_F": float(np.mean(Fs)),
            "mean_delta_LG": float(np.mean(deltas)),
        }
    overall_correct = float(np.mean([r["correct"] for r in rows]))
    return {"per_step_type": per_st,
            "overall_correct_rate": overall_correct}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    tasks = load_gaia_level1(n_max=args.n)
    print(f"Loaded {len(tasks)} GAIA Level-1 tasks")
    client = build_client()

    all_rows = []
    total_cost = 0.0
    n_errors = 0
    log = (HERE / "run.log").open("w")
    log.write(f"exp_3_2 n_tasks={len(tasks)}\n"); log.flush()
    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        try:
            row = run_one_task(task, client=client)
        except Exception as e:
            log.write(f"  ERR i={i} {type(e).__name__}: {e}\n"); log.flush()
            n_errors += 1
            continue
        all_rows.append(row)
        total_cost += row["cost_usd"]
        if (i + 1) % 5 == 0:
            log.write(f"  done={i+1}/{len(tasks)} cost=${total_cost:.4f}\n"); log.flush()
    log.close()

    (HERE / "results.json").write_text(json.dumps({"rows": all_rows}, default=str))
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {
        "exp_id": "exp_3_2", "domain": "gaia",
        "n_tasks": len(tasks), "n_completed": len(all_rows), "n_errors": n_errors,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
