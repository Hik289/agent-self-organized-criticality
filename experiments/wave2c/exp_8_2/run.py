"""Exp 8.2 (Part VIII): embodied state graph fractal D_f — ALFWorld.

5 task graph types x 30 tasks = 150 tasks.

We approximate the 5 task graph types by ALFWorld task subtypes:
  localized: pick_and_place_simple      (single room, single object)
  multi_room: look_at_obj_in_light      (may traverse multiple rooms)
  container: pick_clean_then_place_in_recep (container dependency)
  inventory: pick_heat_then_place_in_recep  (heating chain)
  long_chain: pick_two_obj_and_place      (2 objects, long chain)

For each traj: build error grid from step_events (each step is a cell on a
grid); mark error cells (action NOT in admissible or reward-negative).
Box-counting D_f on the error mask.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.alfworld_batch_runner import run_experiment
from lib.alfworld_runner import list_games_by_type
_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))
from metrics import box_counting_2d  # type: ignore

CONFIG_PATH = "./experiments/wave1/envs/alfworld/base_config.yaml"

TASK_GRAPH_TYPES = {
    "localized": "pick_and_place_simple",
    "multi_room": "look_at_obj_in_light",
    "container": "pick_clean_then_place_in_recep",
    "inventory": "pick_heat_then_place_in_recep",
    "long_chain": "pick_two_obj_and_place",
}


def build_task_list(per_type: int):
    games_by = list_games_by_type(CONFIG_PATH)
    tasks = []
    for label, tt in TASK_GRAPH_TYPES.items():
        games = games_by.get(tt, [])[:per_type]
        for g in games:
            tasks.append((g, {"task_graph_type": label, "task_type": tt}))
    return tasks


def _fractal_over_error_positions(traj_row) -> dict:
    """Build error grid from L_series (steps where L=0 → error position)."""
    L = traj_row.get("L_series", [])
    n = len(L)
    if n < 4:
        return {"D_f": None, "n_scales": 0}
    # Layout: n steps -> sqrt(n) x sqrt(n) grid
    grid_side = max(4, int(np.ceil(np.sqrt(n))))
    mask = np.zeros((grid_side, grid_side), dtype=bool)
    for step_i, l in enumerate(L):
        if l == 0:
            r = step_i // grid_side
            c = step_i % grid_side
            if r < grid_side and c < grid_side:
                mask[r, c] = True
    return box_counting_2d(mask)


def analyze(rows):
    by_type = {}
    for r in rows:
        by_type.setdefault(r["cell"]["task_graph_type"], []).append(r)
    per_type = {}
    for label, rs in by_type.items():
        Dfs = []
        for r in rs:
            d = _fractal_over_error_positions(r)
            if d and d.get("D_f") is not None:
                Dfs.append(d["D_f"])
        R = np.array([r["reward"] for r in rs])
        A = np.array([r["A_i"] for r in rs])
        S = np.array([r["n_steps"] for r in rs])
        per_type[label] = {
            "n": len(rs),
            "mean_reward": float(R.mean()),
            "success_rate": float((R >= 0.5).mean()),
            "A_i_mean": float(A.mean()),
            "mean_steps": float(S.mean()),
            "D_f_mean": float(np.mean(Dfs)) if Dfs else None,
            "D_f_std": float(np.std(Dfs)) if Dfs else None,
            "n_Df_computed": len(Dfs),
        }
    return {"per_task_graph_type": per_type}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=30)
    args = ap.parse_args()
    tasks = build_task_list(args.per_type)
    print(f"Running {len(tasks)} tasks...")
    summary = run_experiment(
        exp_id="exp_8_2", tasks=tasks, out_dir=HERE,
        seed=42, max_steps=30, save_raw=True,
    )
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
