"""Exp 4.1 (§Part IV): surface transformation stability (STEP-BY-STEP).
H=32, D=1, ρ=0.20 (moderate). 6 surface prompt variants × 30 traj = 180 traj.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.sp_runner import run_experiment
from lib.statefulpuzzle import StatefulPuzzleConfig, DEFAULT_SYSTEM
from lib.metrics import error_jaccard, dist_distance

RHO, H, D = 0.20, 64, 1  # updated per spec review

SURFACE_VARIANTS = {
    "identity": DEFAULT_SYSTEM,
    "paraphrase": (
        "You act as a StatefulPuzzle-SOC solver. Rule: gold[t]=(sum(gold[t-D..t-1])+increment[t]) mod V, "
        "gold[0]=initial_observation. Each step: t, V, D, H, increment[t], recent-memory summary (may be corrupted). "
        "Emit ONLY {\"belief\": g} with g in [0, V-1]."
    ),
    "renaming": (
        "You are an EvolvingArray agent. arr[t]=(sum(arr[t-D..t-1])+delta[t]) mod V, arr[0]=initial_obs. "
        "Each step: t, V, D, H, delta[t], recent-memory (possibly corrupted). Return ONLY {\"belief\": g}."
    ),
    "distractor": (
        "You are a diligent StatefulPuzzle-SOC agent. Background noise exists but rule holds: "
        "gold[t]=(sum(gold[t-D..t-1])+increment[t]) mod V. Each step gives t, V, D, H, increment[t], "
        "corrupted-memory summary. Output ONLY {\"belief\": g}."
    ),
    "order_shuffle": (
        "OUTPUT: JSON {\"belief\": g} with g in [0, V-1].\n"
        "TASK: Compute gold[t].\n"
        "RULE: gold[t]=(sum(gold[t-D..t-1])+increment[t]) mod V.\n"
        "INPUT: t, V, D, H, increment[t], recent memory (some entries corrupted)."
    ),
    "style_change": (
        "System: StatefulPuzzle-SOC Solver v1. Each step: t, V, D, H, initial_observation, increment[t], "
        "recent_memory (last K retrieved gold values, some possibly corrupted per env.rho). "
        "Compute gold[t]=(sum(gold[t-D..t-1])+increment[t]) mod V and emit only {\"belief\": g}."
    ),
}

def build_cells():
    cells, prompts, rho_by = [], {}, {}
    for name, prompt in SURFACE_VARIANTS.items():
        cid = f"surf_{name}"
        cfg = StatefulPuzzleConfig(H=H, S=20, D=D, V=8, seed=42, perturbation=None)
        cells.append((cid, cfg, {"surface": name, "rho": RHO}))
        prompts[cid] = prompt; rho_by[cid] = RHO
    return cells, prompts, rho_by

def analyze(rows):
    by_var = {}
    for r in rows: by_var.setdefault(r["meta"]["surface"], []).append(r)
    per_variant = {}
    for v, rs in by_var.items():
        A = np.array([r["A_i"] for r in rs]); C = np.array([r["collapse_indicator"] for r in rs])
        per_variant[v] = {"n": len(rs), "A_mean": float(A.mean()), "A_std": float(A.std()),
                          "collapse_rate": float(C.mean()), "T_col_mean": float(np.mean([r["T_col"] for r in rs]))}
    identity = by_var.get("identity", [])
    err_ts = {v: [{"idx": r["traj_idx"], "err_ts": [t for t,e in enumerate(r["e_series"]) if e > 0.5]} for r in rs]
              for v, rs in by_var.items()}
    id_ts = err_ts.get("identity", [])
    jacc, dists = {}, {}
    id_A = np.array([r["A_i"] for r in identity])
    for v, rs in by_var.items():
        if v == "identity": continue
        vals = [error_jaccard(a["err_ts"], b["err_ts"]) for a, b in zip(id_ts, err_ts[v])]
        jacc[v] = {"mean": float(np.mean(vals)) if vals else None, "n_pairs": len(vals)}
        dists[v] = {"A_i": dist_distance(id_A, np.array([r["A_i"] for r in rs]))}
    return {"per_variant": per_variant, "local_error_jaccard_vs_identity": jacc,
            "A_distribution_distance_vs_identity": dists}

def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cells, prompts, rho_by = build_cells()
    summary = run_experiment(exp_id="exp_4_1", cells=cells, n_traj_per_cell=args.n,
        out_dir=HERE, seed=42, use_llm=True, rho_by_cell=rho_by, system_prompt_by_cell=prompts)
    data = json.loads((HERE / "results.json").read_text())
    agg = analyze(data["rows"])
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))

if __name__ == "__main__": main()
