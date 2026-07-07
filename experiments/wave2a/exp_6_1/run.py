"""Exp 6.1 (§Part VI): spectral analysis H ∈ {32,64,128,256} (STEP-BY-STEP).
D=1, ρ=0.20. n=30/H → 120 traj.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.sp_runner import run_experiment
from lib.statefulpuzzle import StatefulPuzzleConfig
from lib.metrics import spectral_slope, dfa_exponent, bootstrap_alpha_vs_shuffled

RHO, D = 0.20, 1

def build_cells():
    cells, rho_by = [], {}
    for H in [64, 128, 256, 512]:  # updated per spec review
        cid = f"H_{H}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"H": H, "rho": RHO})); rho_by[cid] = RHO
    return cells, rho_by

def analyze(rows):
    by_H = {}
    for r in rows: by_H.setdefault(int(r["meta"]["H"]), []).append(r)
    per_H = {}
    for H, rs in sorted(by_H.items()):
        alphas_e, alphas_F, dfa_vals, p_vals = [], [], [], []
        Cs = []
        for r in rs:
            e = np.asarray(r["e_series"]); F = np.asarray(r["F_series"])
            ae = spectral_slope(e)["alpha"]; aF = spectral_slope(F)["alpha"]
            if ae is not None: alphas_e.append(ae)
            if aF is not None: alphas_F.append(aF)
            dv = dfa_exponent(e)["H_dfa"]
            if dv is not None: dfa_vals.append(dv)
            pv = bootstrap_alpha_vs_shuffled(e, seed=r["seed"], n_boot=50)["p_value"]
            if pv is not None: p_vals.append(pv)
            Cs.append(r["collapse_indicator"])
        per_H[H] = {
            "n": len(rs),
            "alpha_e_mean": float(np.mean(alphas_e)) if alphas_e else None,
            "alpha_e_std": float(np.std(alphas_e)) if alphas_e else None,
            "alpha_F_mean": float(np.mean(alphas_F)) if alphas_F else None,
            "DFA_H_mean": float(np.mean(dfa_vals)) if dfa_vals else None,
            "bootstrap_p_e_median": float(np.median(p_vals)) if p_vals else None,
            "bootstrap_p_e_lt_0p01_frac": float(np.mean(np.array(p_vals) < 0.01)) if p_vals else None,
            "collapse_rate": float(np.mean(Cs)),
        }
    return {"per_H": per_H}

def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cells, rho_by = build_cells()
    summary = run_experiment(exp_id="exp_6_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho_by)
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))

if __name__ == "__main__": main()
