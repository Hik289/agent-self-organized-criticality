"""Single-trajectory analysis pipeline.

Reuses the anchor_3 extractor + local judge + global judge; augments with
per-t stress σ_{i,t}, avalanche stats, per-t local-global gap Δ^LG.

Input:
  raw_trajectory = {
      "case": str,
      "config": {"H":..., "S":..., "D":..., "V":..., "seed":..., ...},
      "gold_series": list[int] length H,
      "trajectory": list[step dict {t, action, args, result}],
  }

Output:
  {
     "H": H,
     "z": [z_dict per logged step],
     "F_per_step": [...],
     "F_series": list[H] (per-t last value),
     "e_series": 1-F,
     "L_series_per_logged_step": [...],
     "sigma_series": list[H] (per-t last value of aggregated sigma from z),
     "avalanche": {...},
     "collapse_indicator": int,
     "submit_ok": bool,
     "T_col": int (H+1 if none),
     "recovery_time": int,
     "wsf_drop": float,   # 1 - min(F),
     "delta_LG_series_per_t": list[H] (L(last set step at t) - F(t)),
  }
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse anchor_3 modules
_A3 = Path(__file__).resolve().parents[2] / "anchor_3"
if str(_A3) not in sys.path:
    sys.path.insert(0, str(_A3))
from state_extractor import extract_trajectory  # type: ignore
from local_judge import judge_trajectory as local_judge_traj  # type: ignore
from global_judge import judge_trajectory as global_judge_traj  # type: ignore

from .metrics import sigma_series_from_z, detect_avalanches

TAU_F = 0.5
TAU_E = 0.5


def _per_t_last(records: list[dict], H: int, key: str, default) -> list:
    out = [default] * H
    for r in records:
        t = r.get("t")
        if isinstance(t, int) and 0 <= t < H:
            v = r.get(key, default)
            if v is not None:
                out[t] = v
    return out


def analyze_trajectory(traj: dict) -> dict:
    H = int(traj["config"]["H"])
    zs = extract_trajectory(traj)
    L_records = local_judge_traj(traj)
    L_series_per_step = [r["L_t"] for r in L_records]
    global_out = global_judge_traj(traj)
    F_series = list(global_out["F_series"])
    e_series = [1.0 - f for f in F_series]

    # sigma from z (per-t last value)
    sigma_full = sigma_series_from_z(zs)
    sigma_by_t: dict[int, float] = {}
    for step, sig in zip(traj["trajectory"], sigma_full.tolist()):
        t = step.get("t")
        if isinstance(t, int) and 0 <= t < H:
            sigma_by_t[t] = float(sig)
    sigma_series = [sigma_by_t.get(t, 0.0) for t in range(H)]

    # avalanche stats on e_series
    aval = detect_avalanches(np.asarray(e_series), tau_e=TAU_E, window_w=2)

    # T_col
    T_col = H + 1
    for t, f in enumerate(F_series):
        if f < TAU_F:
            T_col = t
            break

    # recovery time: from T_col to next stable F >= 0.9 for 2 consecutive steps
    recovery = H + 1 - T_col
    if T_col <= H:
        for t in range(T_col + 1, H):
            if F_series[t] >= 0.9 and (t + 1 >= H or F_series[t + 1] >= 0.9):
                recovery = t - T_col
                break

    # delta_LG series (per t) — use L at final logged action of that t
    L_by_t_last: dict[int, int] = {}
    for step, L in zip(traj["trajectory"], L_series_per_step):
        t = step.get("t")
        if isinstance(t, int) and 0 <= t < H:
            L_by_t_last[t] = int(L)
    delta_LG = [(L_by_t_last.get(t, 1) - F_series[t]) for t in range(H)]

    return {
        "H": H,
        "n_z": len(zs),
        "F_series": F_series,
        "e_series": e_series,
        "L_series_per_step": L_series_per_step,
        "sigma_series": sigma_series,
        "avalanche": aval,
        "collapse_indicator": int(global_out["collapse_indicator"]),
        "submit_ok": bool(global_out["submit_ok"]),
        "T_col": int(T_col),
        "recovery_time": int(recovery),
        "wsf_drop": float(1.0 - min(F_series) if F_series else 0.0),
        "delta_LG_series": delta_LG,
        "min_F": float(global_out["min_F"]),
    }
