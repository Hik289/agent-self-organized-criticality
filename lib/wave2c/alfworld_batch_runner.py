"""Batched ALFWorld runner: iterate tasks, extract, judge, aggregate.

Because ALFWorld env init is not thread-safe (mutates shared env.game_files),
we run tasks SERIALLY per experiment. Concurrency across experiments is
achieved by launching them as separate processes.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from .azure_client import build_client
from .alfworld_runner import run_alfworld_game
from .alfworld_pipeline import analyze_trajectory

CONFIG_PATH = "./experiments/wave1/envs/alfworld/base_config.yaml"


def _run_one(game_file: str, cell_meta: dict, *, client,
             system_prompt: str = None, max_steps: int = 30) -> dict:
    from .alfworld_runner import DEFAULT_AGENT_SYSTEM
    sp = system_prompt if system_prompt is not None else DEFAULT_AGENT_SYSTEM
    try:
        traj = run_alfworld_game(
            config_path=CONFIG_PATH, game_file=game_file,
            client=client, max_steps=max_steps, system_prompt=sp,
        )
    except Exception as e:
        return {"row": None, "raw_trajectory": None,
                "error": f"{type(e).__name__}: {e}",
                "game_file": game_file}
    ana = analyze_trajectory(traj)
    row = {
        "game_file": game_file,
        "task_type": traj["task_type"],
        "reward": traj["reward"],
        "done": traj["done"],
        "n_steps": traj["n_steps"],
        "n_valid_actions": traj["n_valid_actions"],
        "cost_usd": traj["cost_usd"],
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
        "cell": cell_meta,
    }
    compact = {
        "game_file": game_file, "task_type": traj["task_type"],
        "reward": traj["reward"], "done": traj["done"], "n_steps": traj["n_steps"],
        "step_events": traj["step_events"],
        "task_desc": traj["task_desc"],
        "system_prompt_head": traj["system_prompt_head"],
    }
    return {"row": row, "raw_trajectory": compact}


def run_experiment(*, exp_id: str, tasks: list,
                   out_dir: Path, seed: int = 42,
                   max_steps: int = 30,
                   system_prompt_by_task=None,
                   save_raw: bool = True) -> dict:
    """Run tasks serially (alfworld env not thread-safe)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_client()
    system_prompt_by_task = system_prompt_by_task or {}
    all_rows = []
    raw_file = (out_dir / "trajectories.jsonl").open("w") if save_raw else None

    t0 = time.perf_counter()
    total_cost = 0.0
    n_errors = 0
    log = (out_dir / "run.log").open("w")
    log.write(f"exp_id={exp_id} n_tasks={len(tasks)} seed={seed} max_steps={max_steps}\n")
    log.flush()

    done_count = 0
    for game_file, cell_meta in tasks:
        sp = system_prompt_by_task.get(game_file)
        out = _run_one(game_file, cell_meta, client=client,
                       system_prompt=sp, max_steps=max_steps)
        if out.get("error"):
            log.write(f"  ERR {out['error']} game={out['game_file']}\n"); log.flush()
            n_errors += 1
            continue
        row = out["row"]
        all_rows.append(row)
        if raw_file is not None:
            raw_file.write(json.dumps(out["raw_trajectory"]) + "\n")
        total_cost += float(row["cost_usd"])
        done_count += 1
        if done_count % 5 == 0:
            log.write(f"  done={done_count}/{len(tasks)} cost=${total_cost:.4f} errors={n_errors}\n")
            log.flush()

    if raw_file is not None:
        raw_file.close()

    wall_seconds = time.perf_counter() - t0
    summary = {
        "exp_id": exp_id, "domain": "alfworld",
        "n_tasks_requested": len(tasks),
        "n_tasks_completed": len(all_rows),
        "n_errors": n_errors,
        "wall_seconds": round(wall_seconds, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump({"summary": summary, "rows": all_rows}, f)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.write(f"COMPLETE cost=${total_cost:.4f} wall={wall_seconds:.0f}s errors={n_errors}\n")
    log.close()
    return summary
