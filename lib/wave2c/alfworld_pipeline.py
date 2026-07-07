"""ALFWorld single-trajectory analysis pipeline."""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

from .alfworld_extractor import extract_trajectory, sigma_series_from_z
from .alfworld_judges import local_judge_trajectory, global_judge_trajectory

_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))
from metrics import detect_avalanches  # type: ignore

TAU_F = 0.5
TAU_E = 0.5


def analyze_trajectory(traj: dict) -> dict:
    zs = extract_trajectory(traj)
    L_records = local_judge_trajectory(traj)
    L_series = [r["L_t"] for r in L_records]
    global_out = global_judge_trajectory(traj, tau_F=TAU_F)
    F_series = global_out["F_series"]
    e_series = [1.0 - f for f in F_series]
    sigma_series = sigma_series_from_z(zs)

    aval = detect_avalanches(np.asarray(e_series, dtype=float), tau_e=TAU_E, window_w=2)

    H = len(F_series)
    T_col = H + 1
    for t, f in enumerate(F_series):
        if f < TAU_F:
            T_col = t
            break

    recovery = H + 1 - T_col
    if T_col <= H:
        for t in range(T_col + 1, H):
            if F_series[t] >= 0.9:
                recovery = t - T_col
                break

    delta_LG = [(int(L_series[t]) - float(F_series[t])) for t in range(H)]

    return {
        "H": H,
        "F_series": F_series,
        "e_series": e_series,
        "L_series": L_series,
        "sigma_series": sigma_series,
        "delta_LG_series": delta_LG,
        "avalanche": aval,
        "min_F": global_out["min_F"],
        "final_F": global_out["final_F"],
        "reward": global_out["reward"],
        "submit_ok": global_out["submit_ok"],
        "collapse_indicator": global_out["collapse_indicator"],
        "T_col": T_col,
        "recovery_time": recovery,
        "wsf_drop": float(1.0 - min(F_series) if F_series else 0.0),
        "n_valid_actions": global_out["n_valid_actions"],
        "n_steps": global_out["n_steps"],
    }
