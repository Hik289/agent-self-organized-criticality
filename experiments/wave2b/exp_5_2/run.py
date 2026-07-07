"""Exp 5.2 (Part V): stress-triggered intervention — Retail.

4 intervention policies (via agent system prompt append):
  none                       — no additional directive
  periodic                   — check every 3 steps
  stress_triggered           — check if uncertainty/contradiction accumulating
  late_final                 — check only before submit
n=30 tasks per policy = 120 tasks.

Judgment: stress_triggered should have highest reward + lowest collapse.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment

INTERVENTION_DIRECTIVES = {
    "none": "",
    "periodic": "\n\nIntervention policy: every 3 tool calls, re-fetch the affected order/user state.\n",
    "stress_triggered": "\n\nIntervention policy: if you observe multiple tool errors, contradictions between environment replies, or the user seems confused about the state, immediately re-fetch order/user state before proceeding.\n",
    "late_final": "\n\nIntervention policy: before any single terminal action (submit, transfer_to_human_agents, cancel_pending_order at final), re-fetch order/user state to confirm.\n",
}


def _run_policy(policy: str, n_max: int, sub_dir: Path):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from lib.tb_runner import run_experiment
    tasks = []
    prompts = {}
    for i in range(min(n_max, len(TASKS_TEST))):
        tasks.append((i, {"task_index": i, "policy": policy}))
        prompts[i] = f"__APPEND_TO_WIKI_V52__::{policy}"
    return run_experiment(
        exp_id=f"exp_5_2::{policy}", tasks=tasks, out_dir=sub_dir,
        domain="retail", seed=42, max_num_steps=30, concurrency=2,
        agent_system_prompt_by_task=prompts, save_raw=True,
    )


def analyze(all_rows_by_policy):
    per_policy = {}
    for p, rows in all_rows_by_policy.items():
        if not rows:
            per_policy[p] = {"n": 0}
            continue
        A = np.array([r["A_i"] for r in rows])
        C = np.array([r["collapse_indicator"] for r in rows])
        R = np.array([r["reward"] for r in rows])
        F = [np.mean(r["F_series"]) for r in rows]
        per_policy[p] = {
            "n": len(rows), "A_mean": float(A.mean()),
            "collapse_rate": float(C.mean()),
            "mean_reward": float(R.mean()),
            "mean_F": float(np.mean(F)),
        }
    ranked = sorted(per_policy.items(),
                    key=lambda kv: -kv[1].get("mean_reward", 0.0))
    return {"per_policy": per_policy,
            "ranking_by_reward": [p[0] for p in ranked]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    from lib import taubench_runner as tr
    _orig = tr.solve_task

    def patched(*, domain, task_index, client=None, agent_system_prompt=None,
                max_num_steps=30, seed=42):
        if agent_system_prompt and agent_system_prompt.startswith("__APPEND_TO_WIKI_V52__::"):
            policy = agent_system_prompt.split("::", 1)[1]
            from tau_bench.envs import get_env
            from tau_bench.envs.user import UserStrategy
            from lib.azure_client import build_client, AZURE_DEPLOYMENT
            if client is None:
                client = build_client()
            env = get_env(env_name=domain, user_strategy=UserStrategy.HUMAN,
                          user_model=AZURE_DEPLOYMENT, user_provider="openai",
                          task_split="test", task_index=task_index)
            appended = env.wiki + INTERVENTION_DIRECTIVES[policy]
            return _orig(domain=domain, task_index=task_index, client=client,
                         agent_system_prompt=appended,
                         max_num_steps=max_num_steps, seed=seed)
        return _orig(domain=domain, task_index=task_index, client=client,
                     agent_system_prompt=agent_system_prompt,
                     max_num_steps=max_num_steps, seed=seed)
    tr.solve_task = patched

    all_rows = {}
    total_cost = 0.0
    for p in INTERVENTION_DIRECTIVES:
        sub = HERE / f"_policy_{p}"
        summary = _run_policy(p, args.n, sub)
        total_cost += summary["grand_total_cost_usd"]
        data = json.loads((sub / "results.json").read_text())
        all_rows[p] = data["rows"]
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_5_2", "total_cost_usd": total_cost,
               "n_policies": len(INTERVENTION_DIRECTIVES)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
