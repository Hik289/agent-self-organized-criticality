from __future__ import annotations

import numpy as np

from .external_harness import anchor3_helpers
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
    extract_trajectory, local_judge_traj, global_judge_traj = anchor3_helpers()
    H = int(traj["config"]["H"])
    zs = extract_trajectory(traj)
    L_records = local_judge_traj(traj)
    L_series_per_step = [r["L_t"] for r in L_records]
    global_out = global_judge_traj(traj)
    F_series = list(global_out["F_series"])
    e_series = [1.0 - f for f in F_series]


    sigma_full = sigma_series_from_z(zs)
    sigma_by_t: dict[int, float] = {}
    for step, sig in zip(traj["trajectory"], sigma_full.tolist()):
        t = step.get("t")
        if isinstance(t, int) and 0 <= t < H:
            sigma_by_t[t] = float(sig)
    sigma_series = [sigma_by_t.get(t, 0.0) for t in range(H)]


    aval = detect_avalanches(np.asarray(e_series), tau_e=TAU_E, window_w=2)


    T_col = H + 1
    for t, f in enumerate(F_series):
        if f < TAU_F:
            T_col = t
            break


    recovery = H + 1 - T_col
    if T_col <= H:
        for t in range(T_col + 1, H):
            if F_series[t] >= 0.9 and (t + 1 >= H or F_series[t + 1] >= 0.9):
                recovery = t - T_col
                break


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
