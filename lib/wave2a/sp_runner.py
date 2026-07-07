"""Runner scaffolding for StatefulPuzzle-SOC experiments (step-by-step).

Each trajectory = H LLM calls (one per step) + env actions + submit.
Trajectories inside a cell are run concurrently (bounded thread pool) to
overlap network latency; steps within a trajectory are inherently sequential.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .azure_client import build_client, chat
from .pipeline import analyze_trajectory
from .statefulpuzzle import (
    StatefulPuzzleConfig, StatefulPuzzleSOC,
    run_stepwise_trajectory, oracle_predict, DEFAULT_SYSTEM,
)


CONCURRENCY_PER_CELL = 4  # bounded to avoid backend rate limits


def _run_one(cell_id: str, cfg_template: StatefulPuzzleConfig, traj_idx: int,
             *, client, use_llm: bool,
             seed: int, extra_meta: dict,
             K_history: int = 3,
             system_prompt: str | None = None,
             obs0_perturb_delta: int = 0,
             rho: float | None = None) -> dict:
    cfg2 = StatefulPuzzleConfig(
        H=cfg_template.H, S=cfg_template.S, D=cfg_template.D, V=cfg_template.V,
        eta=cfg_template.eta,
        rho=(rho if rho is not None else cfg_template.rho),
        B=cfg_template.B,
        seed=seed + traj_idx,
        perturbation=cfg_template.perturbation,
    )
    env = StatefulPuzzleSOC(cfg2)
    meta: dict[str, Any] = {"cell_id": cell_id, "traj_idx": traj_idx,
                            "seed": cfg2.seed, "rho_env": cfg2.rho,
                            **extra_meta}

    if use_llm:
        sys_p = system_prompt or DEFAULT_SYSTEM
        beliefs, steps, llm_meta = run_stepwise_trajectory(
            client, cfg2, env, llm_call=chat,
            K_history=K_history, system_prompt=sys_p,
            obs0_perturb_delta=obs0_perturb_delta,
        )
        meta["llm"] = llm_meta
    else:
        beliefs = oracle_predict(cfg2, env)
        steps = []
        for t in range(cfg2.H):
            env.t = t
            b = beliefs[t]
            env.record_belief(t, b)
            r = env.do("store", memory_key=f"gold_{t}", value=b)
            steps.append({"t": t, "action": "store",
                          "args": {"memory_key": f"gold_{t}", "value": b},
                          "result": r.get("result", {})})
            r = env.do("set", object=t % cfg2.S, property="value", value=b)
            steps.append({"t": t, "action": "set",
                          "args": {"object": t % cfg2.S, "property": "value", "value": b},
                          "result": r.get("result", {})})
        r = env.do("submit", answer={})
        steps.append({"t": cfg2.H - 1, "action": "submit",
                      "args": {"answer": {}}, "result": r.get("result", {})})
        meta["llm"] = None

    traj = {
        "case": f"{cell_id}#{traj_idx}",
        "config": {"H": cfg2.H, "S": cfg2.S, "D": cfg2.D, "V": cfg2.V,
                   "seed": cfg2.seed, "perturbation": cfg2.perturbation,
                   "rho": cfg2.rho},
        "gold_series": env.gold.tolist(),
        "trajectory": steps,
    }
    ana = analyze_trajectory(traj)

    row = {
        "cell_id": cell_id,
        "traj_idx": traj_idx,
        "seed": cfg2.seed,
        "F_series": ana["F_series"],
        "e_series": ana["e_series"],
        "sigma_series": ana["sigma_series"],
        "L_series_per_step": ana["L_series_per_step"],
        "delta_LG_series": ana["delta_LG_series"],
        "A_i": ana["avalanche"]["A"],
        "A_w": ana["avalanche"]["A_w"],
        "D_i": ana["avalanche"]["D_max"],
        "n_episodes": ana["avalanche"]["n_episodes"],
        "peak_error": ana["avalanche"]["peak_error"],
        "release_speed": ana["avalanche"]["release_speed"],
        "collapse_indicator": ana["collapse_indicator"],
        "submit_ok": ana["submit_ok"],
        "T_col": ana["T_col"],
        "recovery_time": ana["recovery_time"],
        "wsf_drop": ana["wsf_drop"],
        "min_F": ana["min_F"],
        "meta": meta,
    }
    return {"row": row, "raw_trajectory": traj}


def run_experiment(*, exp_id: str, cells: list,
                   n_traj_per_cell: int, out_dir: Path,
                   seed: int = 42, use_llm: bool = True,
                   rho_by_cell: dict | None = None,
                   obs0_perturb_by_cell: dict | None = None,
                   system_prompt_by_cell: dict | None = None,
                   K_history_by_cell: dict | None = None,
                   save_raw: bool = False,
                   concurrency: int = CONCURRENCY_PER_CELL) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_client() if use_llm else None
    rho_by_cell = rho_by_cell or {}
    obs0_perturb_by_cell = obs0_perturb_by_cell or {}
    system_prompt_by_cell = system_prompt_by_cell or {}
    K_history_by_cell = K_history_by_cell or {}

    all_rows: list[dict] = []
    raw_file = None
    if save_raw:
        raw_file = (out_dir / "trajectories.jsonl").open("w")

    t0 = time.perf_counter()
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    n_parse_fail = 0
    n_llm_err = 0

    log_path = out_dir / "run.log"
    with log_path.open("w") as logf:
        logf.write(f"exp_id={exp_id} seed={seed} n_cells={len(cells)} "
                   f"n_traj_per_cell={n_traj_per_cell} use_llm={use_llm} "
                   f"concurrency={concurrency}\n")
        logf.flush()

        for cell_id, cfg, extra_meta in cells:
            rho = rho_by_cell.get(cell_id, cfg.rho)
            obs0_pert = int(obs0_perturb_by_cell.get(cell_id, 0))
            sys_p = system_prompt_by_cell.get(cell_id, DEFAULT_SYSTEM)
            K = int(K_history_by_cell.get(cell_id, 3))
            cell_rows: list[dict | None] = [None] * n_traj_per_cell
            done_count = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {
                    pool.submit(_run_one, cell_id, cfg, j,
                                client=client, use_llm=use_llm,
                                seed=seed, extra_meta=extra_meta,
                                K_history=K, system_prompt=sys_p,
                                obs0_perturb_delta=obs0_pert, rho=rho): j
                    for j in range(n_traj_per_cell)
                }
                for fut in as_completed(futs):
                    j = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as e:  # noqa: BLE001
                        logf.write(f"  {cell_id}[{j}] EXCEPTION {type(e).__name__}: {e}\n")
                        logf.flush()
                        continue
                    row = out["row"]
                    cell_rows[j] = row
                    if raw_file is not None:
                        raw_file.write(json.dumps(out["raw_trajectory"]) + "\n")
                    llm = row["meta"].get("llm")
                    if llm:
                        total_cost += float(llm.get("cost_usd", 0.0))
                        total_tokens_in += int(llm.get("prompt_tokens", 0))
                        total_tokens_out += int(llm.get("completion_tokens", 0))
                        n_parse_fail += int(llm.get("n_parse_fail", 0))
                        n_llm_err += int(llm.get("n_llm_err", 0))
                    done_count += 1
                    if done_count % 5 == 0:
                        logf.write(f"  {cell_id}: {done_count}/{n_traj_per_cell} cost=${total_cost:.4f}\n")
                        logf.flush()
            for r in cell_rows:
                if r is not None:
                    all_rows.append(r)
            logf.write(f"CELL_DONE {cell_id} ({done_count}/{n_traj_per_cell}) cost=${total_cost:.4f}\n")
            logf.flush()
            # incremental save after each cell (rescue against silent SSL stalls)
            try:
                _partial_summary = {"exp_id": exp_id, "seed": seed,
                                    "n_cells_done": cells.index((cell_id, cfg, extra_meta)) + 1,
                                    "n_cells_total": len(cells),
                                    "n_traj_total_so_far": len(all_rows),
                                    "total_cost_usd": round(total_cost, 6),
                                    "last_cell": cell_id}
                (out_dir / "results_partial.json").write_text(
                    json.dumps({"summary": _partial_summary, "rows": all_rows}))
            except Exception as _e:
                logf.write(f"  WARN partial save failed: {_e}\n"); logf.flush()

    if raw_file is not None:
        raw_file.close()

    wall_seconds = time.perf_counter() - t0
    summary = {
        "exp_id": exp_id,
        "seed": seed,
        "n_cells": len(cells),
        "n_traj_per_cell": n_traj_per_cell,
        "n_traj_total": len(all_rows),
        "use_llm": use_llm,
        "wall_seconds": round(wall_seconds, 1),
        "total_cost_usd": round(total_cost, 6),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "n_parse_fail": n_parse_fail,
        "n_llm_err": n_llm_err,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump({"summary": summary, "rows": all_rows}, f)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary
