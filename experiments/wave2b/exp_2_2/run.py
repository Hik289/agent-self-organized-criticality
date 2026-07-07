"""Exp 2.2 (Part II): verification policy vs wrong-basin escape — Retail.

5 verification policies (agent_system_prompt override adds a verify directive
before the standard wiki):
  none                       — no additional directive
  periodic                   — verify every 3 steps
  after_api_error            — verify after any tool error
  after_contradiction        — verify if belief contradicts environment reply
  before_final_submit        — verify before any mutating action

30 Retail tasks (task_index 0..29) per policy = 150 tasks total.

Judgment: escape probability (recovery_time) monotonic with verify strength;
after_contradiction should be best (recall §Part II qualitative table).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment


VERIFY_DIRECTIVES = {
    "none": "",
    "periodic": (
        "\n\nAdditional verification protocol: every 3 tool calls, re-fetch the current "
        "state (get_order_details / get_user_details) before continuing.\n"
    ),
    "after_api_error": (
        "\n\nAdditional verification protocol: after any tool call returns an error or "
        "unexpected reply, re-fetch relevant state (get_order_details / get_user_details) "
        "before continuing.\n"
    ),
    "after_contradiction": (
        "\n\nAdditional verification protocol: if the environment reply contradicts your "
        "current belief about the order/user state, immediately re-fetch that state before "
        "taking any mutating action.\n"
    ),
    "before_final_submit": (
        "\n\nAdditional verification protocol: before any mutating action (exchange, "
        "cancel, modify, return), re-fetch the affected order's current state.\n"
    ),
}


def build_task_list(n_max: int = 30):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    tasks = []
    for policy in VERIFY_DIRECTIVES:
        for i, t in enumerate(TASKS_TEST[:n_max]):
            # Encode policy in cell_meta; task_index remains i
            tasks.append((i, {"task_index": i, "verify_policy": policy}))
    return tasks


def build_agent_prompts(n_max: int = 30):
    """Return dict {task_index -> None} — we use per-cell prompt override, not per-task.

    We handle policy variation by passing agent_system_prompt_by_task keyed by
    (task_index, policy) — but the runner API takes only task_index → prompt.
    Workaround: we split into 5 runs (one per policy) using subdirs.
    """
    return {}


def _run_one_policy(policy: str, n_max: int, sub_dir: Path):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from lib.tb_runner import run_experiment
    tasks = []
    prompts = {}
    for i in range(min(n_max, len(TASKS_TEST))):
        tasks.append((i, {"task_index": i, "verify_policy": policy}))
        # We need env.wiki + directive as prompt. But we can only override after we
        # know env's wiki. Solution: pass a special sentinel and let solve_task look up.
        # Simpler: the directive is appended INSIDE solve_task via
        # `agent_system_prompt = env.wiki + VERIFY_DIRECTIVE`. To do that we need to
        # amend taubench_runner. Below we set the prompt to a special marker.
        prompts[i] = f"__APPEND_TO_WIKI__::{policy}"
    return run_experiment(
        exp_id=f"exp_2_2::{policy}",
        tasks=tasks,
        out_dir=sub_dir,
        domain="retail",
        seed=42,
        max_num_steps=30,
        concurrency=2,
        agent_system_prompt_by_task=prompts,
        save_raw=True,
    )


def analyze_across_policies(all_rows_by_policy: dict[str, list]) -> dict:
    per_policy = {}
    for policy, rows in all_rows_by_policy.items():
        if not rows:
            per_policy[policy] = {"n": 0}
            continue
        recov = np.array([r["recovery_time"] for r in rows])
        C = np.array([r["collapse_indicator"] for r in rows])
        R = np.array([r["reward"] for r in rows])
        per_policy[policy] = {
            "n": len(rows),
            "mean_recovery_time": float(recov.mean()),
            "mean_reward": float(R.mean()),
            "collapse_rate": float(C.mean()),
        }
    # rank policies by (1 - collapse_rate) descending
    ranked = sorted(per_policy.items(),
                    key=lambda kv: kv[1].get("collapse_rate", 1.0))
    return {"per_policy": per_policy, "ranking_by_low_collapse": [p[0] for p in ranked]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    # Patch taubench_runner to handle __APPEND_TO_WIKI__ marker.
    from lib import taubench_runner as tr
    _orig_solve = tr.solve_task

    def patched_solve(*, domain, task_index, client=None, agent_system_prompt=None,
                       max_num_steps=30, seed=42):
        if agent_system_prompt and agent_system_prompt.startswith("__APPEND_TO_WIKI__::"):
            policy = agent_system_prompt.split("::", 1)[1]
            from tau_bench.envs import get_env
            from tau_bench.envs.user import UserStrategy
            from lib.azure_client import build_client, AZURE_DEPLOYMENT
            if client is None:
                client = build_client()
            env = get_env(env_name=domain, user_strategy=UserStrategy.HUMAN,
                          user_model=AZURE_DEPLOYMENT, user_provider="openai",
                          task_split="test", task_index=task_index)
            appended = env.wiki + VERIFY_DIRECTIVES[policy]
            return _orig_solve(domain=domain, task_index=task_index, client=client,
                               agent_system_prompt=appended,
                               max_num_steps=max_num_steps, seed=seed)
        return _orig_solve(domain=domain, task_index=task_index, client=client,
                           agent_system_prompt=agent_system_prompt,
                           max_num_steps=max_num_steps, seed=seed)
    tr.solve_task = patched_solve

    all_rows_by_policy = {}
    total_cost = 0.0
    for policy in VERIFY_DIRECTIVES:
        sub = HERE / f"_policy_{policy}"
        summary = _run_one_policy(policy, args.n, sub)
        total_cost += summary["grand_total_cost_usd"]
        data = json.loads((sub / "results.json").read_text())
        all_rows_by_policy[policy] = data["rows"]
    agg = analyze_across_policies(all_rows_by_policy)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_2_2", "total_cost_usd": total_cost,
               "n_policies": len(VERIFY_DIRECTIVES)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
