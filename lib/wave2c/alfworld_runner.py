"""ALFWorld trajectory runner adapted from experiments/wave1/envs/alfworld/run_smoke.py.

Runs a single ALFWorld game to completion or MAX_STEPS. Returns per-step
trajectory + step_events for extractor/judges.
Thread-safe: fresh env instance per call.
"""
from __future__ import annotations
import os
import re
import time
from typing import Any

from .azure_client import build_client, AZURE_DEPLOYMENT, price

# Set ALFWORLD_DATA before any alfworld import (runtime side-effect)
ALFWORLD_DATA = "./data/alfworld"
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA

# Task-type labels (§spec 6 categories: Pick, Put, Clean, Heat, Cool, Look)
TASK_TYPE_LABELS = {
    "pick_and_place_simple": "Pick_Put",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick_Two",
    "pick_and_place_with_movable_recep": "Pick_Movable_Recep",
}

DEFAULT_AGENT_SYSTEM = (
    "You are an embodied agent controlling a household robot in a text-based "
    "simulator (ALFWorld). At each step you will see an observation and a set "
    "of admissible actions. Pick ONE admissible action that best advances the "
    "task. Respond with ONLY the exact action text — no explanation, no "
    "punctuation, no quotes."
)


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_games_by_type(config_path: str, split: str = "eval_in_distribution") -> dict[str, list[str]]:
    """Return {task_type -> list of game_file paths}.

    Call once per pilot; cheap.
    """
    from alfworld.agents.environment import get_environment
    config = _load_config(config_path)
    AlfredTWEnv = get_environment("AlfredTWEnv")
    env = AlfredTWEnv(config, train_eval=split)
    games_by_type: dict[str, list[str]] = {tt: [] for tt in TASK_TYPE_LABELS}
    for game_file in env.game_files:
        parent = os.path.dirname(game_file)
        segments = parent.split(os.sep)
        for seg in reversed(segments):
            matched = False
            for tt in TASK_TYPE_LABELS:
                if seg.startswith(tt + "-"):
                    games_by_type[tt].append(game_file)
                    matched = True
                    break
            if matched:
                break
    return games_by_type


def _agent_pick_action(client, obs: str, admissible: list[str], task_desc: str,
                       step: int, history: list[str],
                       system_prompt: str = DEFAULT_AGENT_SYSTEM) -> tuple[str, dict]:
    """Ask the LLM to choose one admissible action."""
    admissible = list(admissible)
    if not admissible:
        return "", {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                    "picked_match_type": "no_admissible", "raw_response": ""}

    hist_str = " -> ".join(history[-6:]) if history else "(none)"
    user = (
        f"Task: {task_desc}\n"
        f"Step: {step}\n"
        f"Recent actions: {hist_str}\n"
        f"Observation:\n{obs}\n\n"
        f"Admissible actions ({len(admissible)}):\n"
        + "\n".join(f"- {a}" for a in admissible)
        + "\n\nRespond with exactly one of the admissible actions."
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user}],
            temperature=0.0, seed=42,
        )
    except Exception as e:
        return admissible[0], {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                               "picked_match_type": "llm_error",
                               "raw_response": f"{type(e).__name__}: {e}",
                               "error": str(e)}

    content = (resp.choices[0].message.content or "").strip()
    norm = content.lower().strip("'\"` \n\t.")
    match = next((a for a in admissible if a.lower().strip() == norm), None)
    if match is None:
        candidates = [a for a in admissible if a.lower().strip() in norm or norm in a.lower().strip()]
        if candidates:
            candidates.sort(key=lambda a: (len(a), a))
            match = candidates[0]
    if match is None:
        match = admissible[0]
    usage = resp.usage
    tokens_in = (usage.prompt_tokens if usage else 0) or 0
    tokens_out = (usage.completion_tokens if usage else 0) or 0
    cost = price(usage) if usage else 0.0
    return match, {
        "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost,
        "raw_response": content,
        "picked_match_type": ("exact" if norm == match.lower().strip() else "fuzzy"),
    }


def run_alfworld_game(*, config_path: str, game_file: str,
                      client=None, max_steps: int = 30,
                      system_prompt: str = DEFAULT_AGENT_SYSTEM,
                      split: str = "eval_in_distribution") -> dict:
    """Run one ALFWorld game to completion or max_steps.

    Returns trajectory dict with step_events, task_type, reward, etc.
    """
    if client is None:
        client = build_client()

    from alfworld.agents.environment import get_environment
    config = _load_config(config_path)
    AlfredTWEnv = get_environment("AlfredTWEnv")
    env = AlfredTWEnv(config, train_eval=split)
    env.game_files = [game_file]
    env.num_games = 1

    parent = os.path.dirname(game_file)
    task_type = "unknown"
    for seg in reversed(parent.split(os.sep)):
        for tt in TASK_TYPE_LABELS:
            if seg.startswith(tt + "-"):
                task_type = tt
                break
        if task_type != "unknown":
            break

    tw_env = env.init_env(batch_size=1)
    reset_out = tw_env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {"admissible_commands": []}
    admissible = info.get("admissible_commands", [])
    if isinstance(admissible, list) and admissible and isinstance(admissible[0], list):
        admissible = admissible[0]
    if isinstance(obs, (list, tuple)):
        obs = obs[0] if obs else ""
    if not isinstance(obs, str):
        obs = str(obs)

    m = re.search(r"Your task is to:\s*(.+)", obs)
    task_desc = m.group(1).strip() if m else "(task banner not found)"

    history: list[str] = []
    step_events: list[dict] = []
    total_cost = 0.0
    tokens_in = tokens_out = 0
    reward = 0.0
    done = False
    n_valid = 0
    t0 = time.perf_counter()

    for step in range(max_steps):
        action, meta = _agent_pick_action(client, obs, admissible, task_desc,
                                          step, history, system_prompt)
        tokens_in += meta["tokens_in"]
        tokens_out += meta["tokens_out"]
        total_cost += meta["cost_usd"]
        if action in admissible:
            n_valid += 1

        history.append(action)

        step_out = tw_env.step([action])
        obs_next, rew, done_flag, info_next = step_out
        obs = obs_next[0] if isinstance(obs_next, (list, tuple)) else obs_next
        if not isinstance(obs, str):
            obs = str(obs)
        r = rew[0] if isinstance(rew, (list, tuple)) else rew
        d = done_flag[0] if isinstance(done_flag, (list, tuple)) else done_flag
        reward = float(r) if r is not None else reward
        won = info_next.get("won", [False])
        won0 = won[0] if isinstance(won, (list, tuple)) else won
        n_admissible = len(admissible)
        # Record the step event BEFORE we overwrite admissible for next step
        step_events.append({
            "step": step,
            "action_name": action,
            "action_kwargs": {},
            "admissible_size_before": n_admissible,
            "env_reply_head": (obs or "")[:200],
            "reward_so_far": reward,
            "picked_match_type": meta.get("picked_match_type"),
            "action_in_admissible": (action in admissible),
        })
        admissible = info_next.get("admissible_commands", [])
        if isinstance(admissible, list) and admissible and isinstance(admissible[0], list):
            admissible = admissible[0]
        if d or won0:
            reward = 1.0 if won0 else reward
            done = True
            break

    wall_seconds = time.perf_counter() - t0
    return {
        "game_file": game_file,
        "task_type": task_type,
        "task_desc": task_desc,
        "domain": "alfworld",
        "reward": reward,
        "done": done,
        "n_steps": len(step_events),
        "n_valid_actions": n_valid,
        "step_events": step_events,
        "history": history,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(total_cost, 6),
        "wall_seconds": round(wall_seconds, 1),
        "system_prompt_head": (system_prompt or "")[:200],
    }
