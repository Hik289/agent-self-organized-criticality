"""StatefulPuzzle-SOC integration for wave2a — step-by-step (turn-by-turn) policy.

Design (locked per Director spec §Part V/VI/VII/II):
  * One LLM call per step.
  * At step t, prompt = (system + task_frame + short_history_summary + current_obs).
  * LLM outputs JSON {belief: int} — one integer prediction of gold[t].
  * Runner then:
      env.record_belief(t, belief)
      env.do("store", memory_key=f"gold_{t}", value=belief)
      env.do("set", object=t%S, property="value", value=belief)
  * At t>=1 the LLM sees the last K retrieved memory values (via env.do("retrieve"))
    which may be corrupted (rho > 0 in cfg) — this is the §2.1 stress mechanism.
  * Anchor_3 extractor + judges reused as-is.

Reuses env from experiments/anchor_setup/envs/statefulpuzzle_soc/env.py.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Import env from anchor_setup
_ENV_PATH = Path(__file__).resolve().parents[3] / "experiments/anchor_setup/envs/statefulpuzzle_soc"
if str(_ENV_PATH) not in sys.path:
    sys.path.insert(0, str(_ENV_PATH))
from env import StatefulPuzzleConfig, StatefulPuzzleSOC  # type: ignore


# ---------------------------------------------------------------------------
# LLM step policy (default surface prompt for §3.1)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are a StatefulPuzzle-SOC agent. The world evolves by the rule:\n"
    "  gold[t] = (sum(gold[t-D..t-1]) + increment[t]) mod V\n"
    "with gold[0] = initial_observation. At each step you are told the "
    "current time index t, parameters V, D, H, the current increment[t], "
    "and a summary of your recent memory (retrieved values of past gold — "
    "some entries may be missing or CORRUPTED with prob rho). Compute your "
    "best integer prediction gold[t] in [0, V-1]. Output ONLY {\"belief\": g}."
)


def _make_step_prompt(cfg: StatefulPuzzleConfig, t: int,
                       initial_obs: int, increment_t: int,
                       recent_history: list[tuple[int, int | None]]) -> str:
    """recent_history = list of (t_past, retrieved_value_or_None) tuples."""
    if recent_history:
        hist_str = ", ".join(
            f"gold_{k}={v if v is not None else '?'}"
            for k, v in recent_history
        )
    else:
        hist_str = "(none — this is step 0; use initial_observation as gold[0])"
    return (
        f"V = {cfg.V}\n"
        f"D = {cfg.D}\n"
        f"H = {cfg.H}\n"
        f"t = {t}\n"
        f"initial_observation = {initial_obs}\n"
        f"increment[t] = {increment_t}\n"
        f"recent_memory = {{{hist_str}}}\n\n"
        f"Return {{\"belief\": g}} with g in [0, {cfg.V-1}]."
    )


_JSON_INT_RE = re.compile(r'"belief"\s*:\s*(-?\d+)')


def _parse_belief(text: str, V: int) -> tuple[int, str]:
    """Return (belief, parse_note)."""
    # try direct JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "belief" in obj:
            return int(obj["belief"]) % V, ""
    except Exception:
        pass
    # regex hunt
    m = _JSON_INT_RE.search(text)
    if m:
        return int(m.group(1)) % V, "parsed_via_regex"
    # bare integer
    m2 = re.search(r"-?\d+", text)
    if m2:
        return int(m2.group(0)) % V, "parsed_bare_int"
    return 0, "parse_failed"


def run_stepwise_trajectory(client, cfg: StatefulPuzzleConfig,
                             env: StatefulPuzzleSOC,
                             llm_call: Callable,
                             *, K_history: int = 3,
                             system_prompt: str = DEFAULT_SYSTEM,
                             obs0_perturb_delta: int = 0
                             ) -> tuple[list[int], list[dict], dict]:
    """Run one turn-by-turn trajectory.

    Args:
        client: openai-like client
        cfg: env config (rho > 0 gives stress via retrieve corruption)
        env: fresh StatefulPuzzleSOC (in initial state)
        llm_call: chat() function from azure_client
        K_history: how many recent gold_t retrievals to show the LLM at each step
        system_prompt: allows §Part IV surface transformation
        obs0_perturb_delta: shift initial_observation shown to LLM by delta mod V
            (§Part I fixed small trigger; does NOT touch env.gold)

    Returns (beliefs, steps, meta):
        beliefs: list[H] of int
        steps: list of dicts logging every action taken (store, set, retrieve, submit)
        meta: aggregated per-traj stats
    """
    H = cfg.H
    beliefs = [0] * H

    # Initial observation the LLM sees for its base case
    initial_obs = int(env.get_observation(0))  # already respects env.cfg.perturbation
    initial_obs_shown = (initial_obs + int(obs0_perturb_delta)) % cfg.V

    steps: list[dict] = []
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    n_parse_fail = 0
    n_llm_err = 0
    total_llm_ms = 0.0

    for t in range(H):
        env.t = t

        # Build recent memory summary via env.do("retrieve", ...) — subject to rho corruption.
        recent = []
        for k in range(1, K_history + 1):
            past_t = t - k
            if past_t < 0:
                continue
            r = env.do("retrieve", memory_key=f"gold_{past_t}")
            val = r.get("result", {}).get("value")
            recent.append((past_t, val))
            # log the retrieve too (so extractor can see memory activity)
            steps.append({
                "t": t, "action": "retrieve",
                "args": {"memory_key": f"gold_{past_t}"},
                "result": r.get("result", {}),
            })
        recent.reverse()  # oldest first

        # LLM call
        inc_t = int(env.increments[t])
        prompt = _make_step_prompt(cfg, t, initial_obs_shown, inc_t, recent)
        resp = llm_call(client, system=system_prompt, user=prompt)
        total_cost += float(resp.get("cost_usd", 0.0))
        total_tokens_in += int(resp.get("prompt_tokens", 0))
        total_tokens_out += int(resp.get("completion_tokens", 0))
        total_llm_ms += float(resp.get("latency_ms", 0.0))
        if resp.get("error"):
            n_llm_err += 1

        belief_t, parse_note = _parse_belief(resp.get("content", ""), cfg.V)
        if parse_note in ("parse_failed",):
            n_parse_fail += 1
        beliefs[t] = belief_t
        env.record_belief(t, belief_t)

        # Store and set (the two write actions on gold_t)
        r_store = env.do("store", memory_key=f"gold_{t}", value=belief_t)
        steps.append({
            "t": t, "action": "store",
            "args": {"memory_key": f"gold_{t}", "value": belief_t},
            "result": r_store.get("result", {}),
        })
        r_set = env.do("set", object=t % cfg.S, property="value", value=belief_t)
        steps.append({
            "t": t, "action": "set",
            "args": {"object": t % cfg.S, "property": "value", "value": belief_t},
            "result": r_set.get("result", {}),
        })

    # Submit
    r_sub = env.do("submit", answer={})
    steps.append({
        "t": H - 1, "action": "submit",
        "args": {"answer": {}},
        "result": r_sub.get("result", {}),
    })

    meta = {
        "n_llm_calls": H,
        "K_history": K_history,
        "obs0_perturb_delta": int(obs0_perturb_delta),
        "prompt_tokens": total_tokens_in,
        "completion_tokens": total_tokens_out,
        "total_llm_ms": round(total_llm_ms, 1),
        "cost_usd": round(total_cost, 6),
        "n_parse_fail": n_parse_fail,
        "n_llm_err": n_llm_err,
        "initial_obs_true": initial_obs,
        "initial_obs_shown_to_llm": initial_obs_shown,
        "system_prompt_head": system_prompt[:80],
    }
    return beliefs, steps, meta


# ---------------------------------------------------------------------------
# Oracle policy (unchanged; used only as sanity)
# ---------------------------------------------------------------------------

def oracle_predict(cfg: StatefulPuzzleConfig, env: StatefulPuzzleSOC) -> list[int]:
    obs_0 = env.get_observation(0)
    V = cfg.V; D = cfg.D
    beliefs = np.zeros(cfg.H, dtype=int)
    beliefs[0] = int(obs_0)
    for t in range(1, cfg.H):
        prior = beliefs[max(0, t - D):t]
        beliefs[t] = (int(prior.sum()) + int(env.increments[t])) % V
    return beliefs.tolist()
