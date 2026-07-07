"""Exp 10.2 (§Part X): Game of Life grid-size finite-size scaling (STEP-BY-STEP).

L ∈ {16, 24, 32, 48, 64}. p_obs=0.75, density=0.25, K_MAX=64.
Independent LLM call per K. Checkpoints K ∈ {1, 2, 4, 8, 16, 32, 64}.
n=30 traj/L × 5 = 150 traj × 7 K = 1050 calls.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


CONCURRENCY_PER_L = 2  # Researcher respec: lower to avoid memory accumulation at L=256

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client, chat  # noqa: E402
from lib.gameoflife import (make_grid, rollout, mask_observation,  # noqa: E402
                             llm_predict_grid_at_K, per_grid_metrics)
from lib.metrics import box_counting_2d  # noqa: E402


L_LEVELS = [16, 32, 64, 128, 256]  # updated per spec review
K_MAX = 64
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64]
DENSITY = 0.25
P_OBS = 0.75
SEED_BASE = 42


def _run_traj(client, L: int, traj_idx: int) -> dict:
    seed = SEED_BASE + traj_idx
    g0 = make_grid(L, "random", density=DENSITY, seed=seed)
    trace = rollout(g0, K_MAX)
    obs, mask = mask_observation(g0, P_OBS, seed=seed + 1)
    err_series = []
    F_series = []
    err_mask_at_16 = None
    llm_metas = []
    for K in CHECKPOINTS:
        pred, meta = llm_predict_grid_at_K(client, obs, K, llm_call=chat)
        pm = per_grid_metrics(pred, trace[K])
        err_series.append(pm["hamming"])
        F_series.append(pm["F"])
        llm_metas.append(meta)
        if K == 16:
            em = pm.get("err_mask")
            if em is not None:
                err_mask_at_16 = em
    Df = None
    if err_mask_at_16 is not None and err_mask_at_16.sum() > 0:
        Df = box_counting_2d(err_mask_at_16)
    pred_horizon = K_MAX
    for K, f in zip(CHECKPOINTS, F_series):
        if f < 0.5:
            pred_horizon = K
            break
    total_cost = sum(m["cost_usd"] for m in llm_metas)
    return {
        "L": L,
        "traj_idx": traj_idx,
        "seed": seed,
        "checkpoints": CHECKPOINTS,
        "err_series": err_series,
        "F_series": F_series,
        "pred_horizon": pred_horizon,
        "err_max_hamming": float(max(err_series)),
        "D_f_at_t16": Df,
        "llm_summary": {"n_calls": len(CHECKPOINTS), "cost_usd": total_cost},
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
    log.write(f"exp_10_2 L={L_LEVELS} n={n_traj} K={CHECKPOINTS}\n"); log.flush()
    for L in L_LEVELS:
        cell_rows: list[dict | None] = [None] * n_traj
        done = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY_PER_L) as pool:
            futs = {pool.submit(_run_traj, client, L, j): j for j in range(n_traj)}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    log.write(f"  L={L}[{j}] EX {type(e).__name__}: {e}\n"); log.flush()
                    continue
                cell_rows[j] = row
                total_cost += float(row["llm_summary"]["cost_usd"])
                done += 1
                if done % 5 == 0:
                    log.write(f"  L={L} {done}/{n_traj} cost=${total_cost:.4f}\n"); log.flush()
        for r in cell_rows:
            if r is not None:
                all_rows.append(r)
        log.write(f"L_DONE L={L} cost=${total_cost:.4f}\n"); log.flush()
        # incremental save after each cell (rescue against silent stalls)
        try:
            _tmp_partial = [dict(r) for r in all_rows]
            for r in _tmp_partial:
                if isinstance(r.get("D_f_at_t16"), dict):
                    r["D_f_at_t16"].pop("err_mask", None)
            (HERE / "results_partial.json").write_text(
                json.dumps({"summary": {"exp_id": "exp_10_2",
                                        "n_traj_total": len(_tmp_partial),
                                        "wall_seconds": round(time.perf_counter() - t0, 1),
                                        "total_cost_usd": round(total_cost, 6),
                                        "last_cell_done": f"L={L}"},
                            "rows": _tmp_partial}))
        except Exception as _e:
            log.write(f"  WARN partial save failed: {_e}\n"); log.flush()
    log.close()

    by_L: dict[int, list] = {}
    for r in all_rows:
        by_L.setdefault(r["L"], []).append(r)
    per_L = {}
    for L, rs in sorted(by_L.items()):
        F_at = {k: float(np.mean([r["F_series"][i] for r in rs]))
                for i, k in enumerate(CHECKPOINTS)}
        Dfs = [r["D_f_at_t16"]["D_f"] for r in rs
               if r["D_f_at_t16"] and r["D_f_at_t16"].get("D_f") is not None]
        per_L[L] = {
            "n": len(rs),
            "F_at_checkpoints_mean": F_at,
            "F_at_k64": F_at.get(64),
            "err_cutoff": float(np.max([r["err_max_hamming"] for r in rs])),
            "pred_horizon_median": float(np.median([r["pred_horizon"] for r in rs])),
            "D_f_at_t16_mean": float(np.mean(Dfs)) if Dfs else None,
            "D_f_at_t16_std": float(np.std(Dfs)) if Dfs else None,
        }
    summary = {
        "exp_id": "exp_10_2",
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
        json.dump({"per_L": per_L}, f, indent=2)
    with (HERE / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary": summary, "per_L": per_L}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
