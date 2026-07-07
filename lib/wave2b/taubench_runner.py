"""Tau-bench trajectory runner adapted from anchor_setup/harness/tau_bench_retail/run_sanity.py.

Runs a single task, returns full trajectory dict with step_events for extractor.
Thread-safe: fresh env instance per call.
"""
from __future__ import annotations
import json
import time
from typing import Any

from openai import OpenAI

from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy
from tau_bench.types import Action, RESPOND_ACTION_NAME

from .azure_client import build_client, AZURE_DEPLOYMENT, price


class LLMUserSimulator:
    metadata: dict = {}

    def __init__(self, client, model: str = AZURE_DEPLOYMENT) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def build_system_prompt(self, instruction: str | None) -> str:
        instr_disp = ("\n\nInstruction: " + instruction + "\n") if instruction else ""
        return (
            f"You are a user interacting with an agent.{instr_disp}\n"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information "
            "that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction. For "
            "example, if the agent asks for the order id but it is not mentioned in the "
            "instruction, do not make up an order id, just say you do not remember or have it.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as a standalone "
            "message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, use your own "
            "words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to the "
            "personalities in the instruction."
        )

    def _generate(self) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, seed=42
        )
        content = resp.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        if resp.usage:
            self.total_tokens_in += resp.usage.prompt_tokens or 0
            self.total_tokens_out += resp.usage.completion_tokens or 0
            self.total_cost += price(resp.usage)
        return content

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._generate()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._generate()

    def get_total_cost(self) -> float:
        return self.total_cost


def message_to_action(message: dict) -> Action:
    tc = message.get("tool_calls")
    if tc:
        first = tc[0]
        fn = first["function"]
        name = fn["name"]
        try:
            kwargs = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            kwargs = {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        return Action(name=name, kwargs=kwargs)
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content") or ""})


def solve_task(*, domain: str, task_index: int,
               client: OpenAI | None = None,
               agent_system_prompt: str | None = None,
               max_num_steps: int = 30,
               seed: int = 42) -> dict:
    """Run a single tau-bench task; returns trajectory dict.

    Args:
        domain: "retail" or "airline"
        task_index: task_id in test split
        client: reuse LLM client (or lazily build)
        agent_system_prompt: override env.wiki for §Part IV/IX surface tests
        max_num_steps: episode budget
        seed: LLM seed
    """
    if client is None:
        client = build_client()

    # HUMAN user_strategy avoids external credential lookup; we replace env.user
    # immediately with our own chat-completions-based simulator.
    env = get_env(
        env_name=domain,
        user_strategy=UserStrategy.HUMAN,
        user_model=AZURE_DEPLOYMENT,
        user_provider="openai",
        task_split="test",
        task_index=task_index,
    )
    env.user = LLMUserSimulator(client)

    tools_info = env.tools_info
    wiki = agent_system_prompt if agent_system_prompt is not None else env.wiki

    reset_res = env.reset(task_index=task_index)
    obs = reset_res.observation
    info = reset_res.info.model_dump()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": wiki},
        {"role": "user", "content": obs},
    ]
    actions: list[dict[str, Any]] = []
    step_events: list[dict[str, Any]] = []

    total_cost = 0.0
    tokens_in = 0
    tokens_out = 0
    reward = 0.0
    n_agent_steps = 0
    n_tool_calls = 0
    n_respond = 0
    stop_reason = "loop_end"
    t0 = time.perf_counter()

    for step_i in range(max_num_steps):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT, messages=messages, tools=tools_info,
                temperature=0.0, seed=seed,
            )
        except Exception as e:
            stop_reason = f"agent_llm_error: {type(e).__name__}: {e}"
            break

        n_agent_steps += 1
        if resp.usage:
            tokens_in += resp.usage.prompt_tokens or 0
            tokens_out += resp.usage.completion_tokens or 0
            total_cost += price(resp.usage)

        choice = resp.choices[0]
        msg = choice.message
        next_message: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            next_message["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]

        action = message_to_action(next_message)
        actions.append({"name": action.name, "kwargs": action.kwargs})

        env_response = env.step(action)
        reward = env_response.reward
        info = {**info, **env_response.info.model_dump()}

        step_events.append({
            "step": step_i,
            "action_name": action.name,
            "action_kwargs": action.kwargs,
            "env_reply_head": (env_response.observation or "")[:200],
            "reward_so_far": float(reward),
        })

        if action.name != RESPOND_ACTION_NAME:
            tool_call = next_message["tool_calls"][0]
            next_message["tool_calls"] = [tool_call]
            messages.append(next_message)
            messages.append({
                "role": "tool", "tool_call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "content": env_response.observation,
            })
            n_tool_calls += 1
        else:
            messages.append(next_message)
            messages.append({"role": "user", "content": env_response.observation})
            n_respond += 1

        if env_response.done:
            stop_reason = "env_done"
            break

    wall_seconds = time.perf_counter() - t0

    try:
        task = env.tasks[task_index]
        gold_actions = [{"name": a.name, "kwargs": a.kwargs} for a in task.actions]
        instruction = task.instruction
        user_id = getattr(task, "user_id", None)
    except Exception:
        gold_actions = []
        instruction = ""
        user_id = None

    return {
        "task_id": task_index,
        "domain": domain,
        "reward": float(reward),
        "n_agent_steps": n_agent_steps,
        "n_tool_calls": n_tool_calls,
        "n_respond": n_respond,
        "stop_reason": stop_reason,
        "actions": actions,
        "gold_actions": gold_actions,
        "user_id": user_id,
        "instruction": instruction,
        "step_events": step_events,
        "info": info,
        "system_prompt_head": (wiki or "")[:200],
        "cost_usd": round(total_cost, 6),
        "user_sim_cost_usd": round(env.user.total_cost, 6),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "wall_seconds": round(wall_seconds, 1),
    }


# Historical alias retained for older notebooks/scripts.
AzureUserSimulator = LLMUserSimulator
