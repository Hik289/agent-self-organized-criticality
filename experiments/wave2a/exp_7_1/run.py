"""Exp 7.1 (§Part VII): paired divergence D ∈ {1,2,4} (STEP-BY-STEP).
H=32, ρ=0.10. 30 pairs/D → 90 pairs = 180 traj.
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client, chat
from lib.statefulpuzzle import StatefulPuzzleConfig, StatefulPuzzleSOC, run_stepwise_trajectory, DEFAULT_SYSTEM
from lib.pipeline import analyze_trajectory
from lib.metrics import fit_power_and_exp

D_LEVELS = [1, 2, 4, 6, 8]  # updated per spec review
H, V, S = 128, 8, 20  # updated per spec review
RHO = 0.10
SEED_BASE = 42
CONCURRENCY = 4

def _run_traj(client, cfg, obs0_delta):
    env = StatefulPuzzleSOC(cfg)
    beliefs, steps, meta = run_stepwise_trajectory(client, cfg, env, llm_call=chat,
        K_history=3, system_prompt=DEFAULT_SYSTEM, obs0_perturb_delta=obs0_delta)
    traj = {"case": f"D{cfg.D}_d{obs0_delta}",
            "config": {"H": cfg.H, "S": cfg.S, "D": cfg.D, "V": cfg.V,
                       "seed": cfg.seed, "perturbation": None, "rho": cfg.rho},
            "gold_series": env.gold.tolist(), "trajectory": steps}
    ana = analyze_trajectory(traj)
    return beliefs, {"llm": meta, "F_series": ana["F_series"],
                     "A_i": ana["avalanche"]["A"], "collapse_indicator": ana["collapse_indicator"]}

def _run_pair(client, D, j):
    seed_j = SEED_BASE + j
    cfg = StatefulPuzzleConfig(H=H, S=S, D=D, V=V, seed=seed_j, rho=RHO, perturbation=None)
    bA, mA = _run_traj(client, cfg, 0)
    bB, mB = _run_traj(client, cfg, 1)
    delta = np.array([(int(a)-int(b)) % V for a,b in zip(bA,bB)], dtype=float)
    delta_dist = np.minimum(delta, V - delta)
    fit = fit_power_and_exp(np.arange(1, len(delta_dist)+1), delta_dist)
    return {"D": D, "pair_idx": j, "seed": seed_j,
            "delta_series": delta_dist.tolist(), "fit": fit,
            "F_A_min": float(min(mA["F_series"])), "F_B_min": float(min(mB["F_series"])),
            "A_A": mA["A_i"], "A_B": mB["A_i"],
            "C_A": mA["collapse_indicator"], "C_B": mB["collapse_indicator"],
            "cost_usd": float(mA["llm"]["cost_usd"] + mB["llm"]["cost_usd"])}

def main():
    global N_PAIRS
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    N_PAIRS = args.n
    HERE.mkdir(parents=True, exist_ok=True)
    client = build_client()
    t0 = time.perf_counter(); total_cost = 0.0; all_pairs = []
    log = (HERE / "run.log").open("w")
    log.write(f"exp_7_1 D_levels={D_LEVELS} n_pairs={N_PAIRS} H={H} rho={RHO}\n"); log.flush()
    for D in D_LEVELS:
        pair_rows = [None] * N_PAIRS; done = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futs = {pool.submit(_run_pair, client, D, j): j for j in range(N_PAIRS)}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    log.write(f"  D={D}[{j}] EX {type(e).__name__}: {e}\n"); log.flush(); continue
                pair_rows[j] = rec; total_cost += rec["cost_usd"]; done += 1
                if done % 5 == 0: log.write(f"  D={D} {done}/{N_PAIRS} cost=${total_cost:.4f}\n"); log.flush()
        for rec in pair_rows:
            if rec: all_pairs.append(rec)
        log.write(f"D_DONE D={D} cost=${total_cost:.4f}\n"); log.flush()
    log.close()
    by_D = {}
    for r in all_pairs: by_D.setdefault(r["D"], []).append(r)
    per_D = {}
    for D, rs in sorted(by_D.items()):
        n = len(rs)
        pow_beta = [r["fit"]["power"]["beta"] for r in rs if r["fit"].get("power")]
        exp_lam = [r["fit"]["exp"]["lambda"] for r in rs if r["fit"].get("exp")]
        best_power = sum(1 for r in rs if r["fit"].get("best") == "power")
        d_aic = [r["fit"].get("delta_aic_exp_minus_power") for r in rs
                 if r["fit"].get("delta_aic_exp_minus_power") is not None]
        per_D[D] = {"n_pairs": n,
            "power_beta_mean": float(np.mean(pow_beta)) if pow_beta else None,
            "exp_lambda_mean": float(np.mean(exp_lam)) if exp_lam else None,
            "frac_power_beats_exp": best_power/max(1,n),
            "mean_delta_aic_exp_minus_power": float(np.mean(d_aic)) if d_aic else None,
            "collapse_rate_either": float(np.mean([max(r["C_A"],r["C_B"]) for r in rs]))}
    summary = {"exp_id": "exp_7_1", "n_pairs_total": len(all_pairs),
               "wall_seconds": round(time.perf_counter()-t0,1), "total_cost_usd": round(total_cost,6)}
    with (HERE/"results.json").open("w") as f: json.dump({"summary": summary, "pairs": all_pairs}, f)
    with (HERE/"aggregates.json").open("w") as f: json.dump({"per_D": per_D}, f, indent=2)
    with (HERE/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(json.dumps({"summary": summary, "per_D": per_D}, indent=2))

if __name__ == "__main__": main()
