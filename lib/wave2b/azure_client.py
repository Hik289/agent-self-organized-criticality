"""Provider-neutral LLM client shared across wave 2b."""
from __future__ import annotations
import os
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


def build_client(timeout: float = 300.0) -> OpenAI:
    """Wave 2b uses longer timeout for long tool-use contexts."""
    return OpenAI(base_url=LLM_API_BASE_URL, api_key=LLM_API_KEY,
                  timeout=timeout, max_retries=0)


def price(usage) -> float:
    pt = usage.prompt_tokens or 0
    ct = usage.completion_tokens or 0
    return pt / 1e6 * PRICE_INPUT_USD_PER_MTOK + ct / 1e6 * PRICE_OUTPUT_USD_PER_MTOK
