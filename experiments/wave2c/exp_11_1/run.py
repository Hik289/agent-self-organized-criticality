"""Exp 11.1 (Part XI): subcritical/critical/supercritical regime — ALFWorld.

5 regime settings x 30 tasks = 150 tasks.

Regimes (agent system prompt append):
  high_verify_low_explore: always look/examine, no exploration
  medium_verify_medium_explore: balance
  low_verify_high_explore: skip verification, try new actions
  memory_heavy_unchecked: chain many actions based on cached beliefs
  plan_reset_enabled: reset plan on inconsistency

Baseline task pool: mix from all 6 task types to balance difficulty.
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
_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))

CONFIG_PATH = "./experiments/wave1/envs/alfworld/base_config.yaml"

REGIME_PROMPTS = {
    "high_verify_low_explore": (
        DEFAULT_AGENT_SYSTEM +
        "\n\nRegime: HIGH VERIFY LOW EXPLORE. Before every mutating action "
        "(put, take, use, open, close), first look/examine to verify the "
        "current state. Do not attempt alternative approaches; stick to the "
        "first plan that seems valid."
    ),
    "medium_verify_medium_explore": (
        DEFAULT_AGENT_SYSTEM +
        "\n\nRegime: MEDIUM VERIFY MEDIUM EXPLORE. Verify state at key "
        "decision points (before mutating actions after long chains). "
        "Consider one alternative approach if the first fails."
    ),
    "low_verify_high_explore": (
        DEFAULT_AGENT_SYSTEM +
        "\n\nRegime: LOW VERIFY HIGH EXPLORE. Skip verification actions and "
        "try direct mutating actions immediately. If one action fails, try "
        "multiple alternative approaches."
    ),
    "memory_heavy_unchecked": (
        DEFAULT_AGENT_SYSTEM +
        "\n\nRegime: MEMORY HEAVY UNCHECKED. Rely on your memory of earlier "
        "observations. Chain multiple actions based on your remembered state "
        "without re-observing."
    ),
    "plan_reset_enabled": (
        DEFAULT_AGENT_SYSTEM +
        "\n\nRegime: PLAN RESET ENABLED. If an action's env reply contradicts "
        "your expected state, RESET: return to the initial location and "
        "re-observe before proceeding."
    ),
}

TYPES_MIX = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]


def build_task_list(per_regime: int):
    """Return list of (game_file, cell_meta) where each regime uses the same
    task pool for paired comparability."""
    games_by = list_games_by_type(CONFIG_PATH)
    # Build a balanced mix of ~30 tasks: 5 per type across 6 types = 30
    per_type_target = max(1, per_regime // len(TYPES_MIX))
    base_games = []
    for tt in TYPES_MIX:
        games = games_by.get(tt, [])[:per_type_target]
        base_games.extend(games)
    # top up to per_regime by cycling
    while len(base_games) < per_regime:
        for tt in TYPES_MIX:
            games = games_by.get(tt, [])
            if len(games) > per_type_target:
                base_games.append(games[per_type_target])
                if len(base_games) >= per_regime:
                    break

    tasks = []
    for regime in REGIME_PROMPTS:
        for g in base_games[:per_regime]:
            tasks.append((g, {"regime": regime, "game_file": g}))
    return tasks


def analyze(rows):
    by_regime = {}
    for r in rows:
        by_regime.setdefault(r["cell"]["regime"], []).append(r)
    per_regime = {}
    for reg, rs in by_regime.items():
        R = np.array([r["reward"] for r in rs])
        A = np.array([r["A_i"] for r in rs])
        C = np.array([r["collapse_indicator"] for r in rs])
        S = np.array([r["n_steps"] for r in rs])
        per_regime[reg] = {
            "n": len(rs),
            "mean_reward": float(R.mean()),
            "success_rate": float((R >= 0.5).mean()),
            "collapse_rate": float(C.mean()),
            "A_i_mean": float(A.mean()),
            "mean_steps": float(S.mean()),
        }
    ranked = sorted(per_regime.items(), key=lambda kv: -kv[1]["mean_reward"])
    return {"per_regime": per_regime,
            "ranking_by_reward": [k for k, _ in ranked]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-regime", type=int, default=30)
    args = ap.parse_args()
    tasks = build_task_list(args.per_regime)
    print(f"Running {len(tasks)} tasks...")

    # Patch batch_runner to pass regime prompt via cell_meta
    from lib import alfworld_batch_runner as br
    _orig = br._run_one

    def _patched(game_file, cell_meta, *, client, system_prompt=None, max_steps=30):
        reg = cell_meta.get("regime")
        if reg and reg in REGIME_PROMPTS:
            system_prompt = REGIME_PROMPTS[reg]
        return _orig(game_file, cell_meta, client=client,
                     system_prompt=system_prompt, max_steps=max_steps)
    br._run_one = _patched

    summary = run_experiment(
        exp_id="exp_11_1", tasks=tasks, out_dir=HERE,
        seed=42, max_steps=30, save_raw=True,
    )
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
