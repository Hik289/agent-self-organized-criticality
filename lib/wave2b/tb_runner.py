"""Runner scaffolding for tau-bench experiments (wave 2b).

Runs a list of (task_index, extra_meta) with concurrency, calls solve_task,
runs analyze_trajectory, aggregates. Saves results.json + summary.json.
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .azure_client import build_client
from .taubench_runner import solve_task
from .pipeline import analyze_trajectory


CONCURRENCY = 2  # tau-bench prompts are large; conservative concurrency


def _run_one(*, domain: str, task_index: int, cell_meta: dict,
             client, agent_system_prompt: str | None = None,
             max_num_steps: int = 30, seed: int = 42) -> dict:
    """Run one task, return {row, raw_trajectory}."""
    try:
        traj = solve_task(
            domain=domain,
            task_index=task_index,
            client=client,
            agent_system_prompt=agent_system_prompt,
            max_num_steps=max_num_steps,
            seed=seed,
        )
    except Exception as e:
        return {"row": None, "raw_trajectory": None,
                "error": f"{type(e).__name__}: {e}", "task_index": task_index}
    ana = analyze_trajectory(traj)
    row = {
        "task_id": traj["task_id"],
        "domain": traj["domain"],
        "reward": traj["reward"],
        "n_agent_steps": traj["n_agent_steps"],
        "n_tool_calls": traj["n_tool_calls"],
        "n_respond": traj["n_respond"],
        "stop_reason": traj["stop_reason"],
        "cost_usd": traj["cost_usd"],
        "user_sim_cost_usd": traj.get("user_sim_cost_usd", 0.0),
        "wall_seconds": traj["wall_seconds"],
        "F_series": ana["F_series"],
        "e_series": ana["e_series"],
        "L_series": ana["L_series"],
        "sigma_series": ana["sigma_series"],
        "delta_LG_series": ana["delta_LG_series"],
        "A_i": ana["avalanche"]["A"],
        "A_w": ana["avalanche"]["A_w"],
        "D_i": ana["avalanche"]["D_max"],
        "peak_error": ana["avalanche"]["peak_error"],
        "collapse_indicator": ana["collapse_indicator"],
        "submit_ok": ana["submit_ok"],
        "T_col": ana["T_col"],
        "recovery_time": ana["recovery_time"],
        "wsf_drop": ana["wsf_drop"],
        "min_F": ana["min_F"],
        "final_F": ana["final_F"],
        "n_gold_actions": ana["n_gold_actions"],
        "n_matched": ana["n_matched"],
        "cell": cell_meta,
    }
    # Compact raw trajectory (drop long messages, keep step_events)
    compact = {
        "task_id": traj["task_id"],
        "domain": traj["domain"],
        "reward": traj["reward"],
        "actions": traj["actions"],
        "gold_actions": traj["gold_actions"],
        "step_events": traj["step_events"],
        "instruction": traj["instruction"][:400],
        "system_prompt_head": traj["system_prompt_head"],
        "info_final": {k: traj["info"].get(k)
                       for k in ("reward_info", "task_reward") if k in traj["info"]},
    }
    return {"row": row, "raw_trajectory": compact}


def run_experiment(*, exp_id: str, tasks: list,
                   out_dir: Path, domain: str,
                   seed: int = 42, max_num_steps: int = 30,
                   agent_system_prompt_by_task: dict | None = None,
                   concurrency: int = CONCURRENCY,
                   save_raw: bool = True) -> dict:
    """Run a list of (task_index, cell_meta) tuples.

    Args:
        tasks: list of (task_index, cell_meta_dict)
        agent_system_prompt_by_task: optional {task_index -> system prompt string}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_client()
    agent_system_prompt_by_task = agent_system_prompt_by_task or {}
    all_rows: list[dict] = []
    raw_file = None
    if save_raw:
        raw_file = (out_dir / "trajectories.jsonl").open("w")

    t0 = time.perf_counter()
    total_cost = 0.0
    total_user_cost = 0.0
    n_errors = 0
    log = (out_dir / "run.log").open("w")
    log.write(f"exp_id={exp_id} domain={domain} n_tasks={len(tasks)} "
              f"seed={seed} max_steps={max_num_steps} concurrency={concurrency}\n")
    log.flush()

    done_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {}
        for task_index, cell_meta in tasks:
            sys_prompt = agent_system_prompt_by_task.get(task_index)
            fut = pool.submit(
                _run_one,
                domain=domain, task_index=task_index, cell_meta=cell_meta,
                client=client, agent_system_prompt=sys_prompt,
                max_num_steps=max_num_steps, seed=seed,
            )
            futs[fut] = (task_index, cell_meta)
        for fut in as_completed(futs):
            task_index, cell_meta = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                log.write(f"  task {task_index} EXCEPTION {type(e).__name__}: {e}\n"); log.flush()
                n_errors += 1
                continue
            if out.get("error"):
                log.write(f"  task {task_index} ERROR {out['error']}\n"); log.flush()
                n_errors += 1
                continue
            row = out["row"]
            all_rows.append(row)
            if raw_file is not None and out["raw_trajectory"] is not None:
                raw_file.write(json.dumps(out["raw_trajectory"]) + "\n")
            total_cost += float(row["cost_usd"])
            total_user_cost += float(row["user_sim_cost_usd"])
            done_count += 1
            if done_count % 5 == 0:
                log.write(f"  done={done_count}/{len(tasks)} cost=${total_cost:.4f} "
                          f"user_cost=${total_user_cost:.4f} errors={n_errors}\n")
                log.flush()

    if raw_file is not None:
        raw_file.close()

    wall_seconds = time.perf_counter() - t0
    summary = {
        "exp_id": exp_id,
        "domain": domain,
        "n_tasks_requested": len(tasks),
        "n_tasks_completed": len(all_rows),
        "n_errors": n_errors,
        "wall_seconds": round(wall_seconds, 1),
        "total_cost_usd": round(total_cost, 6),
        "user_sim_cost_usd": round(total_user_cost, 6),
        "grand_total_cost_usd": round(total_cost + total_user_cost, 6),
        "concurrency": concurrency,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump({"summary": summary, "rows": all_rows}, f)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.write(f"COMPLETE cost=${total_cost:.4f} user=${total_user_cost:.4f} "
              f"wall={wall_seconds:.0f}s errors={n_errors}\n")
    log.close()
    return summary
