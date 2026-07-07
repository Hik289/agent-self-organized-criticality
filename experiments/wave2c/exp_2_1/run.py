"""Exp 2.1 (Part II): metastable belief basins — ALFWorld.

3 observation visibility conditions x 6 task types x 15 tasks = 270 tasks.
- full: default agent prompt (agent sees full obs + admissible)
- partial: agent prompt tells "some observations may be incomplete"
- delayed: agent prompt tells "your observations are 3-step lagged"

Note: since we can't actually modify the env obs, the "visibility" condition
is a *prompt-level* proxy encouraging different agent behavior. This is the
best single-model approximation.

Success: partial/delayed should have longer wrong-basin residence time
(measured via consecutive_look ratio and F drop time).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.alfworld_batch_runner import run_experiment
from lib.alfworld_runner import list_games_by_type, DEFAULT_AGENT_SYSTEM

CONFIG_PATH = "./experiments/wave1/envs/alfworld/base_config.yaml"

VISIBILITY_PROMPTS = {
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
    """Return (tasks, prompts_by_game).

    Each task_index_local ~= game_file. Same task appears in all 3 vis conditions
    (paired design for maximum comparability).
    """
    games_by = list_games_by_type(CONFIG_PATH)
    types_to_use = subset_types or CANONICAL_TYPES
    tasks = []
    prompts_by_game = {}
    for tt in types_to_use:
        games = games_by.get(tt, [])[:per_cell]
        for vis, prompt in VISIBILITY_PROMPTS.items():
            for g in games:
                # We must run each game separately per vis condition; the
                # "same" game file gets 3 runs. Since env init is fresh each
                # time this is OK.
                # Use game_file + "@vis" as a unique key stored in cell_meta.
                tasks.append((g, {"task_type": tt, "visibility": vis,
                                    "game_file": g}))
                # For prompts_by_task we need unique key per (game, vis).
                # But the runner uses game_file as key. Workaround: pass
                # per-task prompt via cell_meta and modify batch_runner to
                # use it — see below.
                prompts_by_game[(g, vis)] = prompt
    return tasks, prompts_by_game


def analyze(rows):
    by_vis = {}
    for r in rows:
        by_vis.setdefault(r["cell"]["visibility"], []).append(r)
    per_vis = {}
    for vis, rs in by_vis.items():
        R = np.array([r["reward"] for r in rs])
        S = np.array([r["n_steps"] for r in rs])
        C = np.array([r["collapse_indicator"] for r in rs])
        # wrong-basin residence proxy: number of consecutive_look episodes
        residence = []
        for r in rs:
            sigma = r["sigma_series"]
            # count runs of consecutive_look >= 2
            runs = 0
            in_run = False
            for s in sigma:
                if s >= 2:  # threshold for elevated stress
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
    args = ap.parse_args()

    subset = [s.strip() for s in args.subset.split(",") if s.strip()] or None

    # Patch batch_runner to accept per-(game,vis) prompts via cell_meta
    from lib import alfworld_batch_runner as br
    _orig_run_one = br._run_one

    def _run_one_with_prompt(game_file, cell_meta, *, client,
                             system_prompt=None, max_steps=30):
        # Look up prompt by (game, vis)
        vis = cell_meta.get("visibility")
        if vis and vis in VISIBILITY_PROMPTS:
            system_prompt = VISIBILITY_PROMPTS[vis]
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
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
