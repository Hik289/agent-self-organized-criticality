"""Exp 11.2 (Part XI): intervention regime shift — Retail.

6 regime setups (agent system prompt append):
  high_verify_low_explore
  medium_verify_medium_explore
  low_verify_high_explore
  memory_heavy_unchecked
  plan_reset_enabled
  no_intervention (control)

n=30 tasks per regime = 180 tasks.

Judgment: medium (near-critical) should yield highest reward; extremes
(rigid or unstable) lower.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment


REGIME_DIRECTIVES = {
    "no_intervention": "",
    "high_verify_low_explore": (
        "\n\nRegime: HIGH VERIFY LOW EXPLORE. Verify current state before every mutating "
        "action. Do not explore alternative approaches; commit to the first valid plan.\n"
    ),
    "medium_verify_medium_explore": (
        "\n\nRegime: MEDIUM VERIFY MEDIUM EXPLORE. Verify state at key decision points. "
        "Consider one alternative approach if the first fails.\n"
    ),
    "low_verify_high_explore": (
        "\n\nRegime: LOW VERIFY HIGH EXPLORE. Trust environment replies without re-verification. "
        "Consider multiple alternative approaches for each subgoal.\n"
    ),
    "memory_heavy_unchecked": (
        "\n\nRegime: MEMORY HEAVY UNCHECKED. Rely on your memory of earlier tool call results "
        "without re-checking. Chain multiple actions based on cached beliefs.\n"
    ),
    "plan_reset_enabled": (
        "\n\nRegime: PLAN RESET ENABLED. If you detect any inconsistency between belief and "
        "environment reply, RESET your plan: re-fetch base state and re-derive the plan.\n"
    ),
}


def _run_regime(regime: str, n_max: int, sub_dir: Path):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from lib.tb_runner import run_experiment
    tasks = []
    prompts = {}
    for i in range(min(n_max, len(TASKS_TEST))):
        tasks.append((i, {"task_index": i, "regime": regime}))
        prompts[i] = f"__APPEND_TO_WIKI_V112__::{regime}"
    return run_experiment(
        exp_id=f"exp_11_2::{regime}", tasks=tasks, out_dir=sub_dir,
        domain="retail", seed=42, max_num_steps=30, concurrency=2,
        agent_system_prompt_by_task=prompts, save_raw=True,
    )


def analyze(all_rows_by_regime):
    per = {}
    for regime, rows in all_rows_by_regime.items():
        if not rows:
            per[regime] = {"n": 0}
            continue
        A = np.array([r["A_i"] for r in rows])
        C = np.array([r["collapse_indicator"] for r in rows])
        R = np.array([r["reward"] for r in rows])
        per[regime] = {"n": len(rows), "A_mean": float(A.mean()),
                       "collapse_rate": float(C.mean()),
                       "mean_reward": float(R.mean())}
    ranked = sorted(per.items(), key=lambda kv: -kv[1].get("mean_reward", 0.0))
    return {"per_regime": per, "ranking_by_reward": [r[0] for r in ranked]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    from lib import taubench_runner as tr
    _orig = tr.solve_task

    def patched(*, domain, task_index, client=None, agent_system_prompt=None,
                max_num_steps=30, seed=42):
        if agent_system_prompt and agent_system_prompt.startswith("__APPEND_TO_WIKI_V112__::"):
            regime = agent_system_prompt.split("::", 1)[1]
            from tau_bench.envs import get_env
            from tau_bench.envs.user import UserStrategy
            from lib.azure_client import build_client, AZURE_DEPLOYMENT
            if client is None:
                client = build_client()
            env = get_env(env_name=domain, user_strategy=UserStrategy.HUMAN,
                          user_model=AZURE_DEPLOYMENT, user_provider="openai",
                          task_split="test", task_index=task_index)
            appended = env.wiki + REGIME_DIRECTIVES[regime]
            return _orig(domain=domain, task_index=task_index, client=client,
                         agent_system_prompt=appended,
                         max_num_steps=max_num_steps, seed=seed)
        return _orig(domain=domain, task_index=task_index, client=client,
                     agent_system_prompt=agent_system_prompt,
                     max_num_steps=max_num_steps, seed=seed)
    tr.solve_task = patched

    all_rows = {}
    total_cost = 0.0
    for r in REGIME_DIRECTIVES:
        sub = HERE / f"_regime_{r}"
        summary = _run_regime(r, args.n, sub)
        total_cost += summary["grand_total_cost_usd"]
        data = json.loads((sub / "results.json").read_text())
        all_rows[r] = data["rows"]
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_11_2", "total_cost_usd": total_cost,
               "n_regimes": len(REGIME_DIRECTIVES)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
