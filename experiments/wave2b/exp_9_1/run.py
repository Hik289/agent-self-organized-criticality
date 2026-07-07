"""Exp 9.1 (Part IX): cross-prompt macro stability — Retail.

5 prompt variants (paraphrase / format / tool-wording changes on retail wiki).
Since we're single-model, this replaces the "cross-model" spec with "cross-prompt".

n=30 tasks per variant = 150 tasks.

Judgment: local error overlap low, macro A_i distribution KS distance small.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment

_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))
from metrics import error_jaccard, dist_distance  # type: ignore


VARIANT_APPEND = {
    "identity": "",
    "paraphrase": "\n\nFrom now on, treat 'get_order_details' as 'lookup_order_info' and 'exchange_delivered_order_items' as 'perform_delivered_exchange' (same tool). Follow the wiki rules regardless of naming.\n",
    "output_format": "\n\nAdditional formatting rule: begin every text reply with 'RESPONSE:' and end with a period.\n",
    "tool_wording": "\n\nUse only tool calls for state-changing operations. Never claim to have performed a change without actually invoking the corresponding tool. Return short confirmations.\n",
    "structured_reply": "\n\nStructure your text replies as: (1) confirm understood request, (2) list tool calls you plan to make, (3) execute. Keep replies brief.\n",
}


def build_task_list(n_max: int = 30):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    tasks = []
    for variant in VARIANT_APPEND:
        for i in range(min(n_max, len(TASKS_TEST))):
            tasks.append((i, {"task_index": i, "variant": variant}))
    return tasks


def _run_variant(variant: str, n_max: int, sub_dir: Path):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from lib.tb_runner import run_experiment
    tasks = []
    prompts = {}
    for i in range(min(n_max, len(TASKS_TEST))):
        tasks.append((i, {"task_index": i, "variant": variant}))
        prompts[i] = f"__APPEND_TO_WIKI_V91__::{variant}"
    return run_experiment(
        exp_id=f"exp_9_1::{variant}", tasks=tasks, out_dir=sub_dir,
        domain="retail", seed=42, max_num_steps=30, concurrency=2,
        agent_system_prompt_by_task=prompts, save_raw=True,
    )


def analyze(all_rows_by_variant: dict[str, list]):
    per_variant = {}
    for v, rows in all_rows_by_variant.items():
        if not rows:
            per_variant[v] = {"n": 0}
            continue
        A = np.array([r["A_i"] for r in rows])
        C = np.array([r["collapse_indicator"] for r in rows])
        R = np.array([r["reward"] for r in rows])
        per_variant[v] = {
            "n": len(rows), "A_mean": float(A.mean()),
            "A_std": float(A.std()), "collapse_rate": float(C.mean()),
            "mean_reward": float(R.mean()),
        }
    # jaccard + macro dist vs identity
    id_rows = all_rows_by_variant.get("identity", [])
    id_err_ts = {r["task_id"]: [t for t, e in enumerate(r["e_series"]) if e > 0.5]
                 for r in id_rows}
    id_A = np.array([r["A_i"] for r in id_rows])
    jacc, dists = {}, {}
    for v, rows in all_rows_by_variant.items():
        if v == "identity":
            continue
        vals = []
        for r in rows:
            id_ts = id_err_ts.get(r["task_id"], [])
            v_ts = [t for t, e in enumerate(r["e_series"]) if e > 0.5]
            vals.append(error_jaccard(id_ts, v_ts))
        jacc[v] = {"mean": float(np.mean(vals)) if vals else None, "n_pairs": len(vals)}
        v_A = np.array([r["A_i"] for r in rows])
        dists[v] = {"A_i": dist_distance(id_A, v_A)}
    return {"per_variant": per_variant,
            "local_error_jaccard_vs_identity": jacc,
            "A_distribution_distance_vs_identity": dists}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    from lib import taubench_runner as tr
    _orig = tr.solve_task

    def patched(*, domain, task_index, client=None, agent_system_prompt=None,
                max_num_steps=30, seed=42):
        if agent_system_prompt and agent_system_prompt.startswith("__APPEND_TO_WIKI_V91__::"):
            v = agent_system_prompt.split("::", 1)[1]
            from tau_bench.envs import get_env
            from tau_bench.envs.user import UserStrategy
            from lib.azure_client import build_client, AZURE_DEPLOYMENT
            if client is None:
                client = build_client()
            env = get_env(env_name=domain, user_strategy=UserStrategy.HUMAN,
                          user_model=AZURE_DEPLOYMENT, user_provider="openai",
                          task_split="test", task_index=task_index)
            appended = env.wiki + VARIANT_APPEND[v]
            return _orig(domain=domain, task_index=task_index, client=client,
                         agent_system_prompt=appended,
                         max_num_steps=max_num_steps, seed=seed)
        return _orig(domain=domain, task_index=task_index, client=client,
                     agent_system_prompt=agent_system_prompt,
                     max_num_steps=max_num_steps, seed=seed)
    tr.solve_task = patched

    all_rows = {}
    total_cost = 0.0
    for v in VARIANT_APPEND:
        sub = HERE / f"_variant_{v}"
        summary = _run_variant(v, args.n, sub)
        total_cost += summary["grand_total_cost_usd"]
        data = json.loads((sub / "results.json").read_text())
        all_rows[v] = data["rows"]
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_9_1", "total_cost_usd": total_cost,
               "n_variants": len(VARIANT_APPEND)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
