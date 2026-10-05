from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np

def gol_step(g: np.ndarray) -> np.ndarray:
    grid = np.asarray(g)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError("Game of Life requires a nonempty two-dimensional grid.")
    if not np.all((grid == 0) | (grid == 1)):
        raise ValueError("Game of Life cells must be 0 or 1.")
    height, width = grid.shape
    padded = np.pad(grid.astype(np.int8, copy=False), 1, mode="constant")
    neighbors = np.zeros((height, width), dtype=np.int8)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                neighbors += padded[dy:dy + height, dx:dx + width]
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(int)






_PATTERNS = {
    "blinker": np.array([[1, 1, 1]], dtype=int),
    "glider": np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=int),
    "block": np.array([[1, 1], [1, 1]], dtype=int),
    "beehive": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=int),
}


def make_grid(L: int, kind: str, density: float = 0.2,
              seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.zeros((L, L), dtype=int)
    if kind == "random":
        g = (rng.random((L, L)) < density).astype(int)
    elif kind == "clustered":

        for _ in range(3):
            cy, cx = rng.integers(L // 4, 3 * L // 4, size=2)
            r = max(3, L // 6)
            for y in range(max(0, cy - r), min(L, cy + r)):
                for x in range(max(0, cx - r), min(L, cx + r)):
                    if rng.random() < density * 2:
                        g[y, x] = 1
    elif kind == "mixture":

        for name in ["glider", "blinker", "block", "beehive"]:
            for _ in range(max(1, L // 32)):
                p = _PATTERNS[name]
                y = int(rng.integers(0, L - p.shape[0]))
                x = int(rng.integers(0, L - p.shape[1]))
                g[y:y + p.shape[0], x:x + p.shape[1]] |= p
    else:
        raise ValueError(f"unknown init kind {kind}")
    return g






def rollout(g0: np.ndarray, K: int) -> list[np.ndarray]:
    trace = [g0.copy()]
    g = g0.copy()
    for _ in range(K):
        g = gol_step(g)
        trace.append(g.copy())
    return trace






def mask_observation(g0: np.ndarray, p_obs: float,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = g0.shape[0]
    mask = (rng.random((L, L)) < p_obs).astype(int)
    obs = g0.copy()
    obs[mask == 0] = -1
    return obs, mask






_LLM_SYSTEM_GOL = (
    "You are a Conway's Game of Life predictor. Rules (deterministic, "
    "dead-boundary): a live cell with 2 or 3 live neighbours survives; a "
    "dead cell with exactly 3 live neighbours becomes alive; otherwise dies. "
    "You will receive an initial grid (some cells may be marked -1 = "
    "unobserved) and a single future time step K. Predict the grid at time K. "
    "Output ONLY a JSON 2D list of 0/1 of the same shape as the input grid. "
    "No commentary."
)


def _make_gol_prompt_single(obs: np.ndarray, K: int) -> str:
    return (
        f"L = {obs.shape[0]}\n"
        f"K = {K}\n"
        f"observed_grid_t0 = {obs.tolist()}\n\n"
        f"Predict the grid at time t = {K}. Output ONLY the JSON 2D list "
        f"of size {obs.shape[0]}x{obs.shape[0]}."
    )


def _parse_gol_single(text: str, L: int) -> tuple[np.ndarray, str]:
    text = text.strip()
    err = ""
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\[\s*\[.*\]\s*\]", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if obj is None or not isinstance(obj, list):
        return np.full((L, L), -1, dtype=int), "parse_failed"
    try:
        arr = np.array(obj, dtype=int)
        if arr.shape != (L, L):
            pad = np.zeros((L, L), dtype=int)
            a, b = min(L, arr.shape[0]), min(L, arr.shape[1])
            pad[:a, :b] = arr[:a, :b]
            arr = pad
            err = "shape_padded"
        arr = np.clip(arr, 0, 1)
        return arr, err
    except Exception:
        return np.full((L, L), -1, dtype=int), "parse_grid_failed"


def llm_predict_grid_at_K(client, obs: np.ndarray, K: int,
                           llm_call: Callable) -> tuple[np.ndarray, dict]:
    prompt = _make_gol_prompt_single(obs, K)
    resp = llm_call(client, system=_LLM_SYSTEM_GOL, user=prompt)
    grid, err = _parse_gol_single(resp["content"], obs.shape[0])
    return grid, {
        "prompt_tokens": resp["prompt_tokens"],
        "completion_tokens": resp["completion_tokens"],
        "cost_usd": resp["cost_usd"],
        "latency_ms": resp["latency_ms"],
        "parse_note": err,
        "error": resp["error"],
        "K": K,
    }






def per_grid_metrics(pred: np.ndarray, gold: np.ndarray) -> dict:
    L = gold.shape[0]
    valid = (pred >= 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"hamming": 1.0, "F": 0.0, "n_valid": 0,
                "err_mask_frac_alive": 0.0}
    err_mask = (pred != gold) & valid
    hamm = float(err_mask.sum()) / (L * L)
    F = 1.0 - hamm
    return {
        "hamming": hamm, "F": F, "n_valid": n_valid,
        "err_mask": err_mask,
        "err_mask_frac_alive": float(err_mask.sum()) / max(1, L * L),
    }
