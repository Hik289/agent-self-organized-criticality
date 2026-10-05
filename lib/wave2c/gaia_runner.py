from __future__ import annotations
import json
import importlib.util
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


from .azure_client import build_client, AZURE_DEPLOYMENT, price

METADATA_PATH = os.getenv("GAIA_METADATA_PATH")

MAX_STEPS = 12

DEFAULT_AGENT_SYSTEM = (
    "You are a general assistant solving a short question by using tools "
    "(web_search, browse, read_file, calc). When you are confident in the "
    "answer, call `final_answer(answer=...)` with a SHORT answer (a span, "
    "number, name, list, or yes/no). Do not include units unless the "
    "question asks for them. Do not repeat the question. Answer as concisely "
    "as possible."
)


@lru_cache(maxsize=1)
def load_gaia_harness():
    configured = os.getenv("GAIA_HARNESS_DIR")
    if not configured:
        raise ValueError("Set GAIA_HARNESS_DIR to the original GAIA harness containing tools.py and gaia_scorer.py.")
    root = Path(configured).expanduser().resolve()
    paths = [root / "tools.py", root / "gaia_scorer.py"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required GAIA harness module is missing: {path}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = []
    for path in paths:
        name = f"_agent_soc_gaia_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load GAIA harness module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    tools, scorer = modules
    schemas = getattr(tools, "TOOLS_SCHEMA", None)
    call_tool = getattr(tools, "call_tool", None)
    score = getattr(scorer, "score", None)
    if not isinstance(schemas, list) or not schemas or not callable(call_tool) or not callable(score):
        raise ValueError("GAIA harness must provide nonempty TOOLS_SCHEMA, callable call_tool, and callable score.")
    return schemas, call_tool, score


def load_gaia_level1(n_max: int = 30, metadata_path: str | Path | None = None) -> list[dict]:
    metadata_path = metadata_path or METADATA_PATH
    if not metadata_path:
        raise ValueError("Set GAIA_METADATA_PATH to the GAIA validation metadata JSONL file.")
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GAIA metadata does not exist: {path}")
    if n_max <= 0:
        raise ValueError("n_max must be positive.")
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    filtered = [
        {"task_id": r["task_id"], "level": r["Level"],
         "question": r["Question"], "final_answer": r["Final answer"],
         "file_name": r.get("file_name", "")}
        for r in rows
        if r.get("Level") == 1 and (r.get("file_name") or "") == ""
    ]
    if not filtered:
        raise ValueError("GAIA metadata contains no Level-1 tasks without file attachments.")
    return filtered[:n_max]


def _classify_step_type(step_i: int, action_name: str, kwargs: dict) -> str:
    if action_name == "web_search":
        return "initial_search" if step_i <= 1 else "search"
    if action_name == "browse":
        return "source_selection"
    if action_name == "read_file":
        return "evidence_extraction"
    if action_name == "calc":
        return "calculation"
    if action_name == "final_answer":
        return "final_answer"
    return "intermediate_conclusion"


def run_one_task(task: dict, *, client=None,
                 system_prompt: str = DEFAULT_AGENT_SYSTEM,
                 max_steps: int = MAX_STEPS,
                 seed: int = 42) -> dict:
    tools_schema, call_tool, score = load_gaia_harness()
    if client is None:
        client = build_client()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["question"]},
    ]
    tokens_in = tokens_out = 0
    total_cost = 0.0
    n_tool_calls = 0
    tool_calls_by_name: dict[str, int] = {}
    final_answer: str | None = None
    stop_reason = "loop_end"
    step_events: list[dict] = []
    last_step = -1
    t0 = time.perf_counter()

    for step in range(max_steps):
        last_step = step
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT, messages=messages,
                tools=tools_schema, temperature=0.0, seed=seed,
            )
        except Exception as e:
            stop_reason = f"llm_error: {type(e).__name__}: {e}"
            break

        if resp.usage:
            tokens_in += resp.usage.prompt_tokens or 0
            tokens_out += resp.usage.completion_tokens or 0
            total_cost += price(resp.usage)

        msg = resp.choices[0].message
        next_message: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            next_message["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(next_message)

        if not msg.tool_calls:
            stop_reason = "no_tool_call"
            if msg.content and not final_answer:
                final_answer = msg.content.strip()
            step_events.append({
                "step": step, "action_name": "text_response",
                "action_kwargs": {}, "step_type": "intermediate_conclusion",
                "env_reply_head": "",
                "action_in_admissible": True,
                "picked_match_type": "exact",
            })
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                kwargs = json.loads(tc.function.arguments or "{}")
                if not isinstance(kwargs, dict):
                    kwargs = {}
            except json.JSONDecodeError:
                kwargs = {}
            n_tool_calls += 1
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
            step_type = _classify_step_type(step, name, kwargs)

            if name == "final_answer":
                final_answer = str(kwargs.get("answer", "")).strip()
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "name": name, "content": "ok"})
                stop_reason = "final_answer_called"
                step_events.append({
                    "step": step, "action_name": name,
                    "action_kwargs": kwargs, "step_type": step_type,
                    "env_reply_head": "ok",
                    "action_in_admissible": True,
                    "picked_match_type": "exact",
                })
                break
            else:
                result = call_tool(name, kwargs)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "name": name, "content": result})
                step_events.append({
                    "step": step, "action_name": name,
                    "action_kwargs": kwargs, "step_type": step_type,
                    "env_reply_head": (result or "")[:200],
                    "action_in_admissible": True,
                    "picked_match_type": "exact",
                })

        if stop_reason == "final_answer_called":
            break

    wall = time.perf_counter() - t0


    correct = 0
    if final_answer is not None:
        correct = 1 if score(final_answer, task["final_answer"]) else 0

    return {
        "task_id": task["task_id"],
        "level": task["level"],
        "question": task["question"],
        "gold_answer": task["final_answer"],
        "final_answer": final_answer,
        "correct": correct,
        "reward": float(correct),
        "stop_reason": stop_reason,
        "n_steps": last_step + 1,
        "n_tool_calls": n_tool_calls,
        "tool_calls_by_name": tool_calls_by_name,
        "step_events": step_events,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(total_cost, 6),
        "wall_seconds": round(wall, 1),
        "domain": "gaia",
    }
