"""Exp 1.1 (§Part I): vary memory-corruption stress ρ (STEP-BY-STEP).

Design (locked after pilot v1/v2 calibration):
  H=32, D=1, V=8, S=20, seed=42, temp=0.0
  D=1 chosen so LLM baseline (σ=0, ρ=0) has ~40% clean trajectories.
  H=64 D=4 saturated the LLM at baseline, drowning the stress signal.
  Stress σ ↔ env retrieval corruption ρ (§2.1 memory corruption ratio).
  No obs0 trigger — stress-only encoding proved cleaner.

  σ ∈ {0,1,2,4,8,16} → ρ ∈ {0.00,0.05,0.10,0.20,0.40,0.80}
  n=30 per cell → 180 traj total (~6k LLM calls, ~$1.5 estimated).

Success: σ→collapse AUROC ≥ 0.75.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.sp_runner import run_experiment
from lib.statefulpuzzle import StatefulPuzzleConfig
from lib.metrics import auroc

SIGMA_TO_RHO = {0: 0.00, 1: 0.05, 2: 0.10, 4: 0.20, 8: 0.40, 16: 0.80}
H, D = 64, 1  # updated per spec review

def build_cells():
    cells, rho_by, obs0_by = [], {}, {}
    for sigma in [0, 1, 2, 4, 8, 16]:
        cid = f"sigma_{sigma}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"stress_level": sigma, "rho": SIGMA_TO_RHO[sigma]}))
        rho_by[cid] = SIGMA_TO_RHO[sigma]
        obs0_by[cid] = 0
    return cells, rho_by, obs0_by

def analyze(rows):
    by_level = {}
    for r in rows:
        by_level.setdefault(int(r["meta"]["stress_level"]), []).append(r)
    per_level, A_all, sig_all, C_all = {}, [], [], []
    for s, rs in sorted(by_level.items()):
        A = np.array([r["A_i"] for r in rs])
        C = np.array([r["collapse_indicator"] for r in rs])
        T = np.array([r["T_col"] for r in rs])
        wsf = np.array([r["wsf_drop"] for r in rs])
        rec = np.array([r["recovery_time"] for r in rs])
        sig_end = np.array([r["sigma_series"][-1] for r in rs])
        per_level[s] = {
            "n": len(rs), "A_mean": float(A.mean()), "A_std": float(A.std()),
            "A_min": int(A.min()), "A_max": int(A.max()), "A_median": float(np.median(A)),
            "collapse_rate": float(C.mean()), "T_col_mean": float(T.mean()),
            "wsf_drop_mean": float(wsf.mean()), "recovery_time_mean": float(rec.mean()),
            "sigma_extractor_end_mean": float(sig_end.mean()),
        }
        A_all.append(A); sig_all.append(np.full_like(A, s, dtype=float)); C_all.append(C)
    A_all = np.concatenate(A_all); sig_all = np.concatenate(sig_all); C_all = np.concatenate(C_all)
    q75 = float(np.quantile(A_all, 0.75))
    y_large = (A_all >= q75).astype(int)
    return {
        "per_level": per_level, "A_i_q75": q75,
        "AUROC_sigma_predicts_top25_A": auroc(y_large, sig_all),
        "AUROC_sigma_predicts_collapse": auroc(C_all, sig_all),
    }

def main():
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cells, rho, obs0 = build_cells()
    summary = run_experiment(exp_id="exp_1_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho, obs0_perturb_by_cell=obs0)
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))

if __name__ == "__main__": main()
