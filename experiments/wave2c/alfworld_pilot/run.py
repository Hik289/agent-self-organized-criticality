"""ALFWorld saturation pilot (Director spec §Blocker 1).

Pilot 100 tasks across 6 canonical types (~16-17 per type).
Baseline: default agent prompt, max_steps=30, seed=42, temp=0.0, concurrency=2.

Decision rules:
  - For each of 6 types, keep if baseline success rate < 80%.
  - Drop types with success >= 80% (LLM too familiar).
  - If ALL types >= 80% → escalate to Director.

Output:
  aggregates.json:
    per_type: {task_type -> {n, success_rate, mean_steps, mean_cost}}
    subset_recommended: list of task_types with success < 0.80
    escalate_all_saturated: bool
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client
from lib.alfworld_runner import run_alfworld_game, list_games_by_type, TASK_TYPE_LABELS

CONFIG_PATH = "./experiments/wave1/envs/alfworld/base_config.yaml"
CONCURRENCY = 1  # alfworld env init isn't thread-safe with concurrent env.game_files mutation
SATURATION_THRESHOLD = 0.80

# 6 canonical §spec types (Pick, Put, Clean, Heat, Cool, Look).
# ALFWorld actually has 7 sub-types; map spec labels to actual sub-types:
CANONICAL = {
    "Pick_Put": "pick_and_place_simple",
    "Look": "look_at_obj_in_light",
    "Clean": "pick_clean_then_place_in_recep",
    "Heat": "pick_heat_then_place_in_recep",
    "Cool": "pick_cool_then_place_in_recep",
    "Pick_Two": "pick_two_obj_and_place",
}


def build_task_list(games_by_type: dict, per_type: int) -> list[tuple[str, dict]]:
    tasks = []
    for spec_label, tt in CANONICAL.items():
        games = games_by_type.get(tt, [])
        for i, g in enumerate(games[:per_type]):
            tasks.append((g, {"task_type": tt, "spec_label": spec_label,
                              "index_in_type": i}))
    return tasks


def _run_one(client, game_file: str, meta: dict) -> dict:
    try:
        traj = run_alfworld_game(
            config_path=CONFIG_PATH, game_file=game_file, client=client,
            max_steps=30,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "game_file": game_file, "meta": meta}
    row = {
        "game_file": game_file,
        "task_type": traj["task_type"],
        "spec_label": meta["spec_label"],
        "reward": traj["reward"],
        "done": traj["done"],
        "n_steps": traj["n_steps"],
        "n_valid_actions": traj["n_valid_actions"],
        "cost_usd": traj["cost_usd"],
        "tokens_in": traj["tokens_in"],
        "tokens_out": traj["tokens_out"],
        "wall_seconds": traj["wall_seconds"],
        "won": bool(traj["reward"] >= 0.5),
    }
    return {"row": row, "meta": meta}


def analyze(rows: list[dict]) -> dict:
    per_type = {}
    for r in rows:
        st = r["spec_label"]
        per_type.setdefault(st, []).append(r)
    per_type_stats = {}
    for st, rs in per_type.items():
        wons = [r["won"] for r in rs]
        rewards = [r["reward"] for r in rs]
        steps = [r["n_steps"] for r in rs]
        costs = [r["cost_usd"] for r in rs]
        per_type_stats[st] = {
            "n": len(rs),
            "success_rate": float(np.mean(wons)),
            "reward_mean": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
            "mean_cost_usd": float(np.mean(costs)),
        }
    subset = [st for st, s in per_type_stats.items() if s["success_rate"] < SATURATION_THRESHOLD]
    escalate = (len(subset) == 0)
    return {
        "per_type": per_type_stats,
        "subset_recommended": subset,
        "escalate_all_saturated": escalate,
        "saturation_threshold": SATURATION_THRESHOLD,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=17,
                    help="target games per type (default 17 → ~100 across 6 types)")
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    games = list_games_by_type(CONFIG_PATH)
    print(f"Available per type: { {k: len(v) for k, v in games.items()} }")
    tasks = build_task_list(games, args.per_type)
    print(f"Will run {len(tasks)} tasks total.")

    client = build_client()
    all_rows = []
    n_errors = 0
    total_cost = 0.0
    log = (HERE / "run.log").open("w")
    log.write(f"pilot alfworld per_type={args.per_type} n_total={len(tasks)} concurrency={CONCURRENCY}\n")
    log.flush()
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(_run_one, client, gf, m): (gf, m) for gf, m in tasks}
        done_count = 0
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:
                log.write(f"  EX {type(e).__name__}: {e}\n"); log.flush()
                n_errors += 1
                continue
            if "error" in out:
                log.write(f"  ERR {out['error']} game={out['game_file']}\n"); log.flush()
                n_errors += 1
                continue
            row = out["row"]
            all_rows.append(row)
            total_cost += row["cost_usd"]
            done_count += 1
            if done_count % 5 == 0:
                log.write(f"  done={done_count}/{len(tasks)} cost=${total_cost:.4f} errors={n_errors}\n")
                log.flush()

    log.write(f"COMPLETE cost=${total_cost:.4f} errors={n_errors} wall={time.perf_counter()-t0:.0f}s\n")
    log.close()

    agg = analyze(all_rows)
    (HERE / "results.json").write_text(json.dumps({"rows": all_rows}, indent=2))
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {
        "exp_id": "alfworld_pilot",
        "n_tasks": len(tasks), "n_completed": len(all_rows), "n_errors": n_errors,
        "total_cost_usd": round(total_cost, 6),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
