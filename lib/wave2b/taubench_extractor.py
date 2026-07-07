"""Tau-bench state extractor (§3.1 world_model_science_soc.md).

Adapts anchor_3/state_extractor.py's StatefulPuzzle mapping to tau-bench.

Per-step z_{i,t} 7-tuple:
  q = task progress {step_index, actions_taken, action_types_used, submitted?}
  b = belief about DB state {inferred_entity_from_last_tool_call, last_tool_reply_head}
  c = constraints {satisfied: successful_env_step, violated: env_error_seen, unconfirmed: mutating_action_count}
  u = uncertainty {consecutive_respond_count, respond_ratio, ambiguous_reply_count}
  r = risk {mutating_actions_without_verify}
  m = memory {tool_calls_history_size, unique_tool_names, unique_entities_touched}
  p = current action name (tool_call name or 'respond')

Notes on mapping vs anchor_3:
  - StatefulPuzzle had explicit gold_t; tau-bench's gold is the final DB state
    + gold_actions sequence. We measure per-step F_{i,t} using action-match rate
    (r_actions from tau-bench: 1 if action == corresponding gold action, else 0).
  - env.step gives a real-time observation → we use its text as evidence.
  - Mutating actions in retail: exchange_delivered_order_items, cancel_pending_order,
    modify_pending_order_items, return_delivered_order_items, modify_pending_order_address,
    modify_pending_order_payment, transfer_to_human_agents.
    In airline: cancel_reservation, update_reservation_baggages, update_reservation_flights,
    update_reservation_passengers, book_reservation, send_certificate, transfer_to_human_agents.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict

RETAIL_MUTATING = frozenset({
    "exchange_delivered_order_items", "cancel_pending_order", "modify_pending_order_items",
    "return_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_payment", "transfer_to_human_agents",
})
AIRLINE_MUTATING = frozenset({
    "cancel_reservation", "update_reservation_baggages", "update_reservation_flights",
    "update_reservation_passengers", "book_reservation", "send_certificate",
    "transfer_to_human_agents",
})
RETAIL_VERIFY = frozenset({
    "find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details",
    "get_order_details", "get_product_details", "list_all_product_types",
    "calculate",
})
AIRLINE_VERIFY = frozenset({
    "get_user_details", "search_direct_flight", "search_onestop_flight",
    "get_reservation_details", "calculate",
})


@dataclass
class Z:
    q: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    c: dict = field(default_factory=dict)
    u: dict = field(default_factory=dict)
    r: dict = field(default_factory=dict)
    m: dict = field(default_factory=dict)
    p: str = ""

    def to_dict(self):
        return asdict(self)


def extract_trajectory(traj: dict) -> list[dict]:
    """Extract z_{i,t} for every step of a tau-bench trajectory.

    Consumes only: traj["step_events"], traj["domain"]. All gold fields
    (gold_actions, reward, info.reward_info) are NOT read here — those are
    for the global judge only.

    Returns list[dict] one z-tuple per step.
    """
    domain = traj["domain"]
    mut = RETAIL_MUTATING if domain == "retail" else AIRLINE_MUTATING
    verify = RETAIL_VERIFY if domain == "retail" else AIRLINE_VERIFY

    tool_names_used: list[str] = []
    unique_tools: set = set()
    unique_entities: set = set()  # any string arg looking like an id
    consecutive_respond = 0
    respond_count = 0
    total_steps = 0
    unverified_mutations = 0
    env_errors_seen: list[str] = []
    env_ok_count = 0
    submitted = False
    ambiguous_replies = 0

    zs: list[dict] = []
    for step in traj["step_events"]:
        total_steps += 1
        action_name = step.get("action_name", "")
        kwargs = step.get("action_kwargs", {}) or {}
        env_reply_head = step.get("env_reply_head", "") or ""

        # Detect env error / ambiguous reply from reply head
        low = env_reply_head.lower()
        env_error = any(k in low for k in ("error", "not found", "invalid", "cannot", "failed"))
        if env_error:
            env_errors_seen.append(env_reply_head[:80])
        else:
            env_ok_count += 1
        if "not sure" in low or "please" in low or "could you" in low:
            ambiguous_replies += 1

        # Update memory/tool tracking
        if action_name and action_name != "respond":
            unique_tools.add(action_name)
            tool_names_used.append(action_name)
        for v in kwargs.values():
            if isinstance(v, str) and len(v) >= 6:
                unique_entities.add(v[:32])
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and len(x) >= 6:
                        unique_entities.add(x[:32])

        # Risk: mutating action not immediately preceded by verify tool
        if action_name in mut:
            recent = tool_names_used[-4:-1]
            if not any(t in verify for t in recent):
                unverified_mutations += 1

        # Track submission / handoff
        if action_name in ("transfer_to_human_agents",) or "done" in low or "complete" in low:
            submitted = True

        # Consecutive respond count (uncertainty proxy)
        if action_name == "respond":
            consecutive_respond += 1
            respond_count += 1
        else:
            consecutive_respond = 0

        # Compose z
        q = {
            "step_index": step.get("step", total_steps - 1),
            "actions_taken": total_steps,
            "n_tool_actions": len(tool_names_used),
            "submitted": submitted,
        }
        b = {
            "last_action": action_name,
            "last_env_reply_head": env_reply_head[:100],
        }
        c = {
            "satisfied_count": env_ok_count,
            "violated_env_errors": env_errors_seen[-3:],
            "unconfirmed": max(0, unverified_mutations),
        }
        u = {
            "consecutive_respond_count": consecutive_respond,
            "respond_ratio": respond_count / max(1, total_steps),
            "ambiguous_replies": ambiguous_replies,
        }
        r = {
            "unverified_mutations": int(unverified_mutations),
            "n_mutating_actions_so_far": sum(1 for t in tool_names_used if t in mut),
        }
        m = {
            "n_tool_calls": len(tool_names_used),
            "n_unique_tools": len(unique_tools),
            "n_unique_entities": len(unique_entities),
        }
        p = str(action_name or "")
        zs.append(Z(q=q, b=b, c=c, u=u, r=r, m=m, p=p).to_dict())

    return zs


# ---- σ_{i,t} stress extractor ----
def sigma_series_from_z(z_list: list[dict]) -> list[float]:
    """§3.4 sigma = w1 U + w2 K + w3 R + w4 V + w5 B, default w=1.

    U = uncertainty proxy = consecutive_respond_count + ambiguous_replies
    K = known contradiction count = len(violated_env_errors)
    R = retrieval conflict = 0 (no retrieval in tau-bench directly)
    V = unverified assumption debt = unverified_mutations
    B = tool/env failure debt = total env errors seen
    """
    out = []
    for z in z_list:
        U = float(z["u"].get("consecutive_respond_count", 0)) + float(z["u"].get("ambiguous_replies", 0))
        K = float(len(z["c"].get("violated_env_errors", [])))
        R = 0.0
        V = float(z["r"].get("unverified_mutations", 0))
        B = float(z["c"].get("satisfied_count", 0))  # actually satisfied is inverse... use env errors again
        # Better: use n env errors accumulated
        B = float(K)  # env errors already captured in K; use total accumulated env failures via unique count
        out.append(float(U + K + R + V + B))
    return out
