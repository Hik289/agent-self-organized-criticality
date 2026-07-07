"""Exp 7.2 (§Part VII): Game of Life weak-chaos over p_obs (STEP-BY-STEP).

p_obs ∈ {0.25, 0.5, 0.75, 1.0}. Grid L=32. Checkpoints K ∈ {1,2,4,8,16,32,64}.
INDEPENDENT LLM call per K (no multi-K self-consistency).

n=30 traj/p_obs × 4 = 120 traj × 7 K = 840 calls.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


CONCURRENCY_PER_P = 4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client, chat  # noqa: E402
from lib.gameoflife import (make_grid, rollout, mask_observation,  # noqa: E402
                             llm_predict_grid_at_K, per_grid_metrics)
from lib.metrics import fit_power_and_exp, box_counting_2d  # noqa: E402


L = 32
K_MAX = 64
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64]
DENSITY = 0.25
P_OBS_LEVELS = [0.25, 0.50, 0.75, 1.00]
SEED_BASE = 42


def _run_traj(client, p_obs: float, traj_idx: int) -> dict:
    seed = SEED_BASE + traj_idx
    g0 = make_grid(L, "random", density=DENSITY, seed=seed)
    trace = rollout(g0, K_MAX)
    obs, mask = mask_observation(g0, p_obs, seed=seed + 1)

    err_series = []
    F_series = []
    llm_metas = []
    err_mask_at_16 = None
    for K in CHECKPOINTS:
        pred, meta = llm_predict_grid_at_K(client, obs, K, llm_call=chat)
        pm = per_grid_metrics(pred, trace[K])
        err_series.append(pm["hamming"])
        F_series.append(pm["F"])
        llm_metas.append(meta)
        if K == 16:
            em = pm.get("err_mask")
            if em is not None and hasattr(em, "shape"):
                err_mask_at_16 = em
    fit = fit_power_and_exp(np.asarray(CHECKPOINTS, dtype=float),
                             np.asarray(err_series, dtype=float))
    Df = None
    if err_mask_at_16 is not None and err_mask_at_16.sum() > 0:
        Df = box_counting_2d(err_mask_at_16)
    total_cost = sum(m["cost_usd"] for m in llm_metas)
    total_in = sum(m["prompt_tokens"] for m in llm_metas)
    total_out = sum(m["completion_tokens"] for m in llm_metas)
    return {
        "p_obs": p_obs,
        "traj_idx": traj_idx,
        "seed": seed,
        "checkpoints": CHECKPOINTS,
        "err_series": err_series,
        "F_series": F_series,
        "fit": fit,
        "D_f_at_t16": Df,
        "llm_summary": {"n_calls": len(CHECKPOINTS),
                         "cost_usd": total_cost,
                         "prompt_tokens": total_in,
                         "completion_tokens": total_out},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    n_traj = args.n
    HERE.mkdir(parents=True, exist_ok=True)
    client = build_client()
    t0 = time.perf_counter()
    all_rows: list[dict] = []
    total_cost = 0.0
    log = (HERE / "run.log").open("w")
    log.write(f"exp_7_2 p_obs={P_OBS_LEVELS} n={n_traj} L={L} K={CHECKPOINTS}\n"); log.flush()
    for p in P_OBS_LEVELS:
        cell_rows: list[dict | None] = [None] * n_traj
        done = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY_PER_P) as pool:
            futs = {pool.submit(_run_traj, client, p, j): j for j in range(n_traj)}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    log.write(f"  p_obs={p}[{j}] EX {type(e).__name__}: {e}\n"); log.flush()
                    continue
                cell_rows[j] = row
                total_cost += float(row["llm_summary"]["cost_usd"])
                done += 1
                if done % 5 == 0:
                    log.write(f"  p_obs={p} {done}/{n_traj} cost=${total_cost:.4f}\n"); log.flush()
        for r in cell_rows:
            if r is not None:
                all_rows.append(r)
        log.write(f"P_DONE p_obs={p} cost=${total_cost:.4f}\n"); log.flush()
    log.close()

    by_p: dict[float, list] = {}
    for r in all_rows:
        by_p.setdefault(r["p_obs"], []).append(r)
    per_p = {}
    for p, rs in sorted(by_p.items()):
        n = len(rs)
        F_at = {k: float(np.mean([r["F_series"][i] for r in rs]))
                for i, k in enumerate(CHECKPOINTS)}
        pow_beta = [r["fit"]["power"]["beta"] for r in rs if r["fit"]["power"]]
        exp_lam = [r["fit"]["exp"]["lambda"] for r in rs if r["fit"]["exp"]]
        best_power = sum(1 for r in rs if r["fit"]["best"] == "power")
        d_aic = [r["fit"]["delta_aic_exp_minus_power"] for r in rs
                 if r["fit"]["delta_aic_exp_minus_power"] is not None]
        Dfs = [r["D_f_at_t16"]["D_f"] for r in rs
               if r["D_f_at_t16"] and r["D_f_at_t16"].get("D_f") is not None]
        per_p[p] = {
            "n": n,
            "F_at_checkpoints_mean": F_at,
            "power_beta_mean": float(np.mean(pow_beta)) if pow_beta else None,
            "exp_lambda_mean": float(np.mean(exp_lam)) if exp_lam else None,
            "frac_power_beats_exp": best_power / max(1, n),
            "mean_delta_aic_exp_minus_power": float(np.mean(d_aic)) if d_aic else None,
            "D_f_at_t16_mean": float(np.mean(Dfs)) if Dfs else None,
            "D_f_at_t16_std": float(np.std(Dfs)) if Dfs else None,
        }
    summary = {
        "exp_id": "exp_7_2",
        "n_traj_total": len(all_rows),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    for r in all_rows:
        if isinstance(r.get("D_f_at_t16"), dict):
            r["D_f_at_t16"].pop("err_mask", None)
    with (HERE / "results.json").open("w") as f:
        json.dump({"summary": summary, "rows": all_rows}, f)
    with (HERE / "aggregates.json").open("w") as f:
        json.dump({"per_p_obs": per_p}, f, indent=2)
    with (HERE / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary": summary, "per_p_obs": per_p}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
