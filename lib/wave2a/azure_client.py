"""Thin wrapper around a chat-completions-compatible LLM endpoint.

Configuration is intentionally provider-neutral for public release. Set
``LLM_API_BASE_URL``, ``LLM_MODEL``, and ``LLM_API_KEY`` before running
experiments. The public code keeps the historical constant names below as
aliases so older experiment scripts continue to work.
"""
from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://YOUR_LLM_API_BASE_URL/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "YOUR_MODEL_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY")

# Backward-compatible aliases used by existing experiment scripts.
AZURE_ENDPOINT = LLM_API_BASE_URL
AZURE_DEPLOYMENT = LLM_MODEL
AZURE_CLIENT_KEY = LLM_API_KEY

PRICE_INPUT_USD_PER_MTOK = 0.25
PRICE_OUTPUT_USD_PER_MTOK = 2.00


def build_client(timeout: float = 60.0) -> OpenAI:
    """Build a chat-completions client with a hard request timeout."""
    try:
        import httpx
        _timeout = httpx.Timeout(connect=15.0, read=timeout, write=10.0, pool=5.0)
    except Exception:
        _timeout = timeout
    return OpenAI(base_url=LLM_API_BASE_URL, api_key=LLM_API_KEY,
                  timeout=_timeout, max_retries=0)


def chat(client: OpenAI, system: str, user: str,
         seed: int = 42, temperature: float = 0.0,
         max_retries: int = 3, retry_sleep: float = 2.0) -> dict:
    """One-shot chat call. Returns dict with content, usage, latency, cost."""
    last_err = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                seed=seed,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = int(usage.prompt_tokens) if usage else 0
            ct = int(usage.completion_tokens) if usage else 0
            cost = pt / 1e6 * PRICE_INPUT_USD_PER_MTOK + ct / 1e6 * PRICE_OUTPUT_USD_PER_MTOK
            return {
                "content": content,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(cost, 8),
                "finish_reason": resp.choices[0].finish_reason,
                "model_returned": resp.model,
                "attempt": attempt + 1,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(retry_sleep * (2 ** attempt))
    return {
        "content": "", "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "latency_ms": 0.0, "cost_usd": 0.0,
        "finish_reason": None, "model_returned": None,
        "attempt": max_retries, "error": last_err,
    }
