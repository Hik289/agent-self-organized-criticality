from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np


import hashlib
import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType


_LOAD_LOCK = threading.RLock()
_ENVIRONMENT_FILE = Path("anchor_setup/envs/statefulpuzzle_soc/env.py")
_JUDGE_FILES = (
    Path("anchor_3/state_extractor.py"),
    Path("anchor_3/local_judge.py"),
    Path("anchor_3/global_judge.py"),
)


class OriginalHarnessUnavailable(ImportError):
    pass


def _harness_root(required: tuple[Path, ...]) -> Path:
    configured = os.environ.get("AGENT_SOC_HARNESS_ROOT", "").strip()
    expected = ", ".join(str(path) for path in required)
    if not configured:
        raise OriginalHarnessUnavailable(
            "The original StatefulPuzzle/anchor_3 harness is not included in this release. "
            "Set AGENT_SOC_HARNESS_ROOT to its original experiments directory containing "
            f"{expected}. No replacement environment or scoring rules are substituted."
        )
    root = Path(configured).expanduser().resolve()
    missing = [str(root / path) for path in required if not (root / path).is_file()]
    if missing:
        raise OriginalHarnessUnavailable(
            "AGENT_SOC_HARNESS_ROOT is missing original harness files: " + ", ".join(missing)
        )
    return root


def _load_module(root: Path, relative: Path) -> ModuleType:
    source = root / relative
    digest = hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
    package_name = f"_agent_soc_original_harness_{digest}"
    module_name = f"{package_name}.{source.stem}"
    with _LOAD_LOCK:
        if module_name in sys.modules:
            return sys.modules[module_name]
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = [str(source.parent)]
            package.__package__ = package_name
            package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
            sys.modules[package_name] = package
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise OriginalHarnessUnavailable(
                f"Cannot import original harness module {source}: {exc}. "
                "Provide its original dependencies; sibling imports must be package-relative."
            ) from exc


def statefulpuzzle_types() -> tuple[type, type]:
    root = _harness_root((_ENVIRONMENT_FILE,))
    module = _load_module(root, _ENVIRONMENT_FILE)
    try:
        return module.StatefulPuzzleConfig, module.StatefulPuzzleSOC
    except AttributeError as exc:
        raise OriginalHarnessUnavailable(
            f"{root / _ENVIRONMENT_FILE} must define StatefulPuzzleConfig and StatefulPuzzleSOC."
        ) from exc


def anchor3_helpers():
    root = _harness_root(_JUDGE_FILES)
    modules = [_load_module(root, path) for path in _JUDGE_FILES]
    functions = []
    for module, path, name in zip(modules, _JUDGE_FILES, ("extract_trajectory", "judge_trajectory", "judge_trajectory")):
        function = getattr(module, name, None)
        if not callable(function):
            raise OriginalHarnessUnavailable(f"{root / path} must define callable {name}.")
        functions.append(function)
    return tuple(functions)


def require_statefulpuzzle_harness() -> None:
    _harness_root((_ENVIRONMENT_FILE, *_JUDGE_FILES))
    statefulpuzzle_types()
    anchor3_helpers()


def __getattr__(name: str):
    if name in {"StatefulPuzzleConfig", "StatefulPuzzleSOC"}:
        config_type, environment_type = statefulpuzzle_types()
        return {"StatefulPuzzleConfig": config_type, "StatefulPuzzleSOC": environment_type}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")






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

    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "belief" in obj:
            return int(obj["belief"]) % V, ""
    except Exception:
        pass

    m = _JSON_INT_RE.search(text)
    if m:
        return int(m.group(1)) % V, "parsed_via_regex"

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
    H = cfg.H
    beliefs = [0] * H


    initial_obs = int(env.get_observation(0))
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


        recent = []
        for k in range(1, K_history + 1):
            past_t = t - k
            if past_t < 0:
                continue
            r = env.do("retrieve", memory_key=f"gold_{past_t}")
            val = r.get("result", {}).get("value")
            recent.append((past_t, val))

            steps.append({
                "t": t, "action": "retrieve",
                "args": {"memory_key": f"gold_{past_t}"},
                "result": r.get("result", {}),
            })
        recent.reverse()


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






def oracle_predict(cfg: StatefulPuzzleConfig, env: StatefulPuzzleSOC) -> list[int]:
    obs_0 = env.get_observation(0)
    V = cfg.V; D = cfg.D
    beliefs = np.zeros(cfg.H, dtype=int)
    beliefs[0] = int(obs_0)
    for t in range(1, cfg.H):
        prior = beliefs[max(0, t - D):t]
        beliefs[t] = (int(prior.sum()) + int(env.increments[t])) % V
    return beliefs.tolist()
