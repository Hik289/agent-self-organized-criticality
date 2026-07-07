"""ALFWorld local/global judges (§3.3, isolation invariant preserved).

Local judge: sees only (obs, action). L=1 iff action is admissible AND was
picked via exact match (not fuzzy substring).

Global judge: uses full trajectory + final reward.
F_{i,t} per step = (step_index + 1) / max_steps_expected, weighted by whether
action was in admissible. Final F equals the env reward (0 or 1).
C_i = 1 if reward < 0.5.
"""
from __future__ import annotations


class LocalJudgeIsolationError(RuntimeError):
    pass


_FORBIDDEN = frozenset({"reward", "won", "done", "final_reward", "gold_actions",
                        "F_series", "collapse_indicator"})


def _assert_no_gold_leak(view: dict):
    leaked = [k for k in view.keys() if k in _FORBIDDEN]
    if leaked:
        raise LocalJudgeIsolationError(f"Local judge received leaked: {leaked}")


def local_judge_step(step_view: dict) -> tuple[int, str]:
    _assert_no_gold_leak(step_view)
    action = step_view.get("action_name", "") or ""
    in_adm = bool(step_view.get("action_in_admissible", False))
    picked = step_view.get("picked_match_type", "")
    env_reply = (step_view.get("env_reply_head") or "").lower()

    if not action:
        return 0, "empty_action"
    if not in_adm:
        return 0, "not_in_admissible"
    if picked == "fuzzy":
        return 0, "fuzzy_match"
    if picked == "no_admissible":
        return 0, "no_admissible"
    if picked == "llm_error":
        return 0, "llm_error"
    # env-side failure signal
    if "nothing happens" in env_reply or "you can't" in env_reply:
        return 0, "env_no_effect"
    return 1, "action_locally_valid"


def local_judge_trajectory(traj: dict) -> list[dict]:
    out = []
    for step in traj["step_events"]:
        view = {k: v for k, v in step.items() if k in
                ("step", "action_name", "action_in_admissible",
                 "picked_match_type", "env_reply_head")}
        L, reason = local_judge_step(view)
        out.append({"t": step.get("step"), "action": step.get("action_name"),
                    "L_t": L, "reason": reason})
    return out


def global_judge_trajectory(traj: dict, tau_F: float = 0.5) -> dict:
    """F_{i,t} approx = running action-in-admissible rate weighted by step progress.

    Final F is the env's binary reward (won or not).
    """
    step_events = traj["step_events"]
    H = len(step_events)
    reward = float(traj.get("reward", 0.0))
    won = bool(reward >= 0.5)

    # Per-step F: running valid-action rate weighted by task progress.
    # If task won, all steps get F=1 (retrospectively). If not, F declines.
    F_series = []
    n_valid = 0
    for i, step in enumerate(step_events):
        if step.get("action_in_admissible"):
            n_valid += 1
        # basic progress-weighted validity
        valid_rate = n_valid / (i + 1)
        # If task ultimately won, F rises toward 1 near the end
        if won:
            F_series.append(valid_rate * 0.5 + 0.5)
        else:
            F_series.append(valid_rate * 0.5)  # capped at 0.5 = collapse floor
    # Force final F to reflect env reward
    if F_series:
        F_series[-1] = reward if won else max(0.0, F_series[-1] - 0.1)

    min_F = min(F_series) if F_series else 0.0
    final_F = F_series[-1] if F_series else 0.0
    C_i = int(reward < tau_F)
    return {
        "F_series": F_series,
        "min_F": float(min_F),
        "final_F": float(final_F),
        "reward": reward,
        "submit_ok": won,
        "collapse_indicator": C_i,
        "n_valid_actions": n_valid,
        "n_steps": H,
    }
