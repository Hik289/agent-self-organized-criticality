"""Exp 1.2 (Part I): small/large failures on same avalanche curve.

Retail full test split (~114 tasks), single seed=42, temp=0.0.

Success (Part I qualitative): minor recoverable errors and final collapse trajs
should fall on same A_i distribution curve.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.tb_runner import run_experiment


def build_task_list(n_max: int = 200):
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    tasks = []
    for i, t in enumerate(TASKS_TEST[:n_max]):
        tasks.append((i, {"task_index": i, "annotator": getattr(t, "annotator", None)}))
    return tasks


def analyze(rows):
    if not rows:
        return {"n": 0, "note": "no rows to analyze"}
    A = np.array([r["A_i"] for r in rows], dtype=float)
    R = np.array([r["reward"] for r in rows], dtype=float)
    C = np.array([r["collapse_indicator"] for r in rows], dtype=int)
    n = len(rows)
    A_max_int = int(A.max()) if len(A) else 0

    bins_edges = [0, 3, 6, 11, 21, max(A_max_int + 1, 22)]
    labels = ["1-2", "3-5", "6-10", "11-20", ">20"]
    hist_bin_counts = []
    for i, (lo, hi) in enumerate(zip(bins_edges[:-1], bins_edges[1:])):
        mask = (A >= lo) & (A < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            hist_bin_counts.append({"bin": labels[i], "n": 0})
            continue
        c_col = int(C[mask].sum())
        hist_bin_counts.append({
            "bin": labels[i],
            "n": n_bin,
            "n_collapsed": c_col,
            "final_collapse_fraction": c_col / n_bin,
            "reward_mean": float(R[mask].mean()),
        })

    A_sorted = np.sort(A[A > 0])
    tail_alpha = None
    tail_r2 = None
    if len(A_sorted) >= 5:
        ranks = np.arange(1, len(A_sorted) + 1) / len(A_sorted)
        surv = 1.0 - ranks + 1e-9
        lx = np.log(A_sorted[:-1] + 1e-9)
        ly = np.log(surv[:-1])
        from scipy import stats
        slope, _, r, _, _ = stats.linregress(lx, ly)
        tail_alpha = -float(slope)
        tail_r2 = float(r * r)

    minor = A[(A > 0) & (A <= 5)]
    major = A[A >= 10]
    from scipy import stats
    if len(minor) >= 3 and len(major) >= 3:
        ks_stat, ks_p = stats.ks_2samp(minor, major)
    else:
        ks_stat, ks_p = None, None

    return {
        "n": n,
        "A_mean": float(A.mean()),
        "A_std": float(A.std()),
        "A_median": float(np.median(A)),
        "A_max": int(A.max()),
        "reward_mean": float(R.mean()),
        "collapse_rate_overall": float(C.mean()),
        "avalanche_bins": hist_bin_counts,
        "tail_power_law_alpha": tail_alpha,
        "tail_power_law_r2": tail_r2,
        "minor_vs_major_ks_stat": ks_stat,
        "minor_vs_major_ks_p": ks_p,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    tasks = build_task_list(n_max=args.n)
    summary = run_experiment(
        exp_id="exp_1_2",
        tasks=tasks,
        out_dir=HERE,
        domain="retail",
        seed=42,
        max_num_steps=args.max_steps,
        concurrency=2,
        save_raw=True,
    )
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
