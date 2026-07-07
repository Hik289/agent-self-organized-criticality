"""Exp 5.1 (Part V): stress accumulation before collapse — Retail.

**SECONDARY ANALYSIS**: reuses exp_1_2 trajectories (same 115 retail tasks).
No new API calls.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))
from metrics import auroc  # type: ignore


CONF_KWD = ("cancel", "refund", "downgrade", "conflict")


def classify_task_from_row(row, raw):
    n = row["n_gold_actions"]
    instr = (raw.get("instruction") or "").lower()
    if n <= 1:
        return "single_app"
    if n == 2:
        return "two_app"
    if n == 3:
        return "three_app"
    if any(k in instr for k in CONF_KWD):
        return "conflicting_constraints"
    return "cross_app_update"


def _sigma_slope_pre_collapse(sigma_series, T_col, window=3):
    if T_col >= len(sigma_series):
        return 0.0
    end = min(T_col, len(sigma_series))
    start = max(0, end - window)
    if end - start < 2:
        return 0.0
    xs = np.arange(start, end)
    ys = np.array(sigma_series[start:end])
    if np.ptp(ys) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def analyze(rows, raws_by_task):
    if not rows:
        return {"n": 0}
    by_dep = {}
    for r in rows:
        raw = raws_by_task.get(r["task_id"], {})
        dep = classify_task_from_row(r, raw)
        by_dep.setdefault(dep, []).append(r)
    per_dep = {}
    all_sigma_early = []
    all_F_drop = []
    for dep, rs in by_dep.items():
        sigma_slopes = [_sigma_slope_pre_collapse(r["sigma_series"], r["T_col"]) for r in rs]
        sigma_early = []
        F_drop = []
        for r in rs:
            n = len(r["sigma_series"])
            first_half = r["sigma_series"][: n // 2] if n >= 2 else r["sigma_series"]
            sigma_early.append(float(np.mean(first_half)) if first_half else 0.0)
            F_drop.append(1 if min(r["F_series"] or [1.0]) < 0.5 else 0)
        all_sigma_early.extend(sigma_early)
        all_F_drop.extend(F_drop)
        per_dep[dep] = {
            "n": len(rs),
            "mean_sigma_slope_pre_collapse": float(np.mean(sigma_slopes)),
            "collapse_rate": float(np.mean([r["collapse_indicator"] for r in rs])),
            "mean_reward": float(np.mean([r["reward"] for r in rs])),
            "mean_sigma_early": float(np.mean(sigma_early)),
            "F_drop_rate": float(np.mean(F_drop)),
        }
    au = auroc(np.array(all_F_drop), np.array(all_sigma_early))
    return {"per_dep_class": per_dep,
            "AUROC_sigma_early_predicts_F_drop": au}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str,
                    default=str(HERE.parent / "exp_1_2/results.json"))
    ap.add_argument("--raw", type=str,
                    default=str(HERE.parent / "exp_1_2/trajectories.jsonl"))
    args = ap.parse_args()
    src_p = Path(args.source)
    raw_p = Path(args.raw)
    if not src_p.exists():
        print(f"ERROR: source {src_p} not found. Run exp_1_2 first.")
        sys.exit(2)
    data = json.loads(src_p.read_text())
    raws_by_task = {}
    if raw_p.exists():
        for line in raw_p.open():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raws_by_task[raw["task_id"]] = raw
            except Exception:
                pass
    agg = analyze(data["rows"], raws_by_task)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_5_1_secondary",
               "n_rows": len(data["rows"]),
               "n_raws": len(raws_by_task),
               "source": str(src_p),
               "total_cost_usd": 0.0}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
