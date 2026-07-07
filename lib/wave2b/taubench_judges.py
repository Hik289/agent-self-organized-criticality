"""Tau-bench local/global judges (§3.3 — isolation invariant preserved).

Local judge: sees only (current user msg / env reply, current action).
  L=1 iff:
    - action is a known tool name for the domain OR "respond"
    - action kwargs are non-empty for tool actions (basic well-formedness)
    - env did NOT reply with error text

Global judge: uses full trajectory + gold_actions + task reward.
  F_{i,t} per step = per-step action match rate (running):
    F_{i,t} = (number of gold actions matched so far) / (number of gold actions total)
  When there is no gold action for that step (agent taking extra tool), F stays.
  Final G_i = task reward (0 or 1 or fractional per tau-bench convention).
  Collapse indicator C_i = 1 if G_i < 0.5 OR min_t F_{i,t} < 0.5 at end.
"""
from __future__ import annotations
from typing import Any

# Domain tool sets (loaded from taubench_extractor imports)
RETAIL_TOOLS = frozenset({
    "find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details",
    "get_order_details", "get_product_details", "list_all_product_types",
    "exchange_delivered_order_items", "cancel_pending_order",
    "modify_pending_order_items", "return_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
    "transfer_to_human_agents", "calculate", "think",
})
AIRLINE_TOOLS = frozenset({
    "get_user_details", "search_direct_flight", "search_onestop_flight",
    "get_reservation_details", "calculate", "book_reservation",
    "cancel_reservation", "update_reservation_baggages",
    "update_reservation_flights", "update_reservation_passengers",
    "send_certificate", "transfer_to_human_agents", "think", "list_all_airports",
})


class LocalJudgeIsolationError(RuntimeError):
    pass


_FORBIDDEN_FIELDS = frozenset({
    "gold_actions", "reward", "info", "gold", "final_reward", "task_reward",
    "future_step", "F_series", "C_i",
})


def _assert_no_gold_leak(step_view: dict) -> None:
    leaked = [k for k in step_view.keys() if k in _FORBIDDEN_FIELDS]
    if leaked:
        raise LocalJudgeIsolationError(
            f"Local judge received leaked fields: {leaked} — §3.3 invariant violated."
        )


def local_judge_step(step_view: dict, *, domain: str) -> tuple[int, str]:
    """Return (L_t, reason)."""
    _assert_no_gold_leak(step_view)
    action = step_view.get("action_name", "")
    kwargs = step_view.get("action_kwargs", {}) or {}
    env_reply = (step_view.get("env_reply_head") or "").lower()

    known = RETAIL_TOOLS if domain == "retail" else AIRLINE_TOOLS
    if action == "respond":
        return 1, "respond_locally_valid"
    if not action:
        return 0, "empty_action"
    if action not in known:
        return 0, f"unknown_tool:{action}"
    # env reply error → local judge sees only what env said → L=0 if error visible
    if any(k in env_reply for k in ("error", "invalid", "not found", "failed")):
        return 0, "env_rejected_locally_visible"
    # basic arg well-formedness: tool actions need non-empty kwargs (except think/list_all)
    if not kwargs and action not in ("list_all_product_types", "list_all_airports", "think"):
        return 0, "empty_kwargs"
    return 1, "tool_call_locally_valid"


def local_judge_trajectory(traj: dict) -> list[dict]:
    domain = traj["domain"]
    out = []
    for step in traj["step_events"]:
        # strip forbidden keys defensively
        view = {k: v for k, v in step.items() if k in
                ("step", "action_name", "action_kwargs", "env_reply_head")}
        L, reason = local_judge_step(view, domain=domain)
        out.append({"t": step.get("step"), "action": step.get("action_name"),
                    "L_t": L, "reason": reason})
    return out


def _action_matches_gold(agent_action: dict, gold_action: dict) -> bool:
    """Structural match: name + kwargs subset. Tau-bench's own r_actions uses
    strict equality; we relax to name + key kwargs match for robustness."""
    if agent_action.get("name") != gold_action.get("name"):
        return False
    ga = gold_action.get("kwargs") or {}
    aa = agent_action.get("kwargs") or {}
    # Every gold-required kwarg must be present with matching value
    for k, v in ga.items():
        if k not in aa:
            return False
        # Loose match for list values (order-insensitive)
        if isinstance(v, list) and isinstance(aa[k], list):
            if sorted(map(str, v)) != sorted(map(str, aa[k])):
                return False
        elif str(aa[k]) != str(v):
            return False
    return True


def global_judge_trajectory(traj: dict, tau_F: float = 0.5) -> dict:
    """Compute per-step F_{i,t} + collapse indicator + submit_ok.

    F_t = (# gold_actions matched by step t) / max(1, min(t+1, n_gold))

    This F starts at 1 iff step 0 matches gold_action_0, drops when the agent
    takes a non-gold action (unnecessary tool call), and recovers when the
    agent gets back on the gold path. Matches the §1 semantics: F is "how well
    the world model matches gold at time t".

    C_i = 1 iff reward < tau_F OR final F_t < tau_F. We use FINAL rather than
    MIN because for a recovered task min_F drops early (before agent has taken
    any gold actions), which is not a true collapse.
    """
    gold_actions = traj.get("gold_actions", []) or []
    n_gold = len(gold_actions)
    matched = [False] * n_gold
    F_series: list[float] = []
    F_per_step_events = []
    step_events = traj["step_events"]
    n_matched_so_far = 0
    n_tool_steps = 0  # count only tool-call steps for F denominator (respond is neutral)
    for step in step_events:
        agent_act = {"name": step.get("action_name"), "kwargs": step.get("action_kwargs", {})}
        if agent_act["name"] and agent_act["name"] != "respond":
            n_tool_steps += 1
            for i, g in enumerate(gold_actions):
                if not matched[i] and _action_matches_gold(agent_act, g):
                    matched[i] = True
                    n_matched_so_far += 1
                    break
        denom = max(1, min(n_tool_steps, n_gold)) if n_gold else 1
        F_t = n_matched_so_far / denom if n_gold else 1.0
        F_series.append(F_t)
        F_per_step_events.append({"t": step.get("step"), "F_t": F_t,
                                  "n_matched": n_matched_so_far,
                                  "n_tool_steps": n_tool_steps})
    reward = float(traj.get("reward", 0.0))
    submit_ok = bool(reward >= 0.5)
    min_F = min(F_series) if F_series else 0.0
    final_F = float(F_series[-1]) if F_series else 0.0
    C_i = int((reward < tau_F) or (final_F < tau_F))
    return {
        "F_series": F_series,
        "F_per_step": F_per_step_events,
        "min_F": float(min_F),
        "final_F": final_F,
        "reward": reward,
        "submit_ok": submit_ok,
        "collapse_indicator": C_i,
        "n_gold_actions": n_gold,
        "n_matched": int(n_matched_so_far),
    }
