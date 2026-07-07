"""Exp 10.1 (§Part X): horizon FSS H ∈ {8,16,32,64,128} (STEP-BY-STEP).
D=1, ρ=0.20. n=30/H → 150 traj.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.sp_runner import run_experiment
from lib.statefulpuzzle import StatefulPuzzleConfig

try:
    from scipy import stats; _HAS_SCIPY = True
except Exception: _HAS_SCIPY = False

RHO, D = 0.20, 1

def build_cells():
    cells, rho_by = [], {}
    for H in [8, 16, 32, 64, 128, 256, 512]:  # updated per spec review
        cid = f"H_{H}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"H": H, "rho": RHO})); rho_by[cid] = RHO
    return cells, rho_by

def analyze(rows):
    by_H = {}
    for r in rows: by_H.setdefault(int(r["meta"]["H"]), []).append(r)
    per_H, all_A = {}, {}
    for H, rs in sorted(by_H.items()):
        A = np.array([r["A_i"] for r in rs]); C = np.array([r["collapse_indicator"] for r in rs])
        wsf = np.array([r["min_F"] for r in rs])
        per_H[H] = {"n": len(rs), "A_mean": float(A.mean()), "A_std": float(A.std()),
                    "A_max_cutoff": int(A.max()), "A_p90": float(np.quantile(A, 0.90)),
                    "collapse_rate": float(C.mean()), "WSF_floor_mean": float(wsf.mean())}
        all_A[H] = A
    pairwise = {}
    if _HAS_SCIPY:
        Hs = sorted(all_A.keys())
        for i in range(len(Hs)-1):
            Ha, Hb = Hs[i], Hs[i+1]
            w, p = stats.mannwhitneyu(all_A[Ha], all_A[Hb], alternative="less")
            pairwise[f"{Ha}<{Hb}"] = {"U": float(w), "p_value": float(p)}
    return {"per_H": per_H, "adjacent_H_A_mean_less_p": pairwise}

def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cells, rho_by = build_cells()
    summary = run_experiment(exp_id="exp_10_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho_by,
        concurrency=2)  # respec: lower to avoid memory accumulation at H=512
    data = json.loads((HERE/"results.json").read_text())
    agg = analyze(data["rows"])
    (HERE/"aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))

if __name__ == "__main__": main()
