"""Exp 6.2 (Part VI): memory setting alters spectral slope alpha — HotpotQA.

5 memory settings x 30 questions = 150 runs.
- no_retrieval: use only supporting_only subset with k=0 (LLM must answer from parametric memory)
- top_2: k=2 (narrow field, likely misses supporting)
- top_5: k=5 (default)
- full_context: all 10 paragraphs
- contaminated: k=5, but supporting paragraphs corrupted with rho=0.5

For each setting, compute per-question error e_i = 1 - f1, then spectral
slope alpha of the e-series over question index (30 questions as time axis).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lib.azure_client import build_client
from lib.hotpotqa_runner import load_corpus, run_one_question
_W2A_LIB = Path(__file__).resolve().parents[2] / "wave2a" / "lib"
if str(_W2A_LIB) not in sys.path:
    sys.path.insert(0, str(_W2A_LIB))
from metrics import spectral_slope, dfa_exponent  # type: ignore


MEMORY_SETTINGS = {
    "no_retrieval": {"k": 0, "paragraph_filter": "supporting_only", "corrupt": 0.0},
    "top_2": {"k": 2, "paragraph_filter": "all", "corrupt": 0.0},
    "top_5": {"k": 5, "paragraph_filter": "all", "corrupt": 0.0},
    "full_context": {"k": 10, "paragraph_filter": "all", "corrupt": 0.0},
    "contaminated": {"k": 5, "paragraph_filter": "all", "corrupt": 0.5},
}


def analyze(rows):
    by_setting = {}
    for r in rows:
        by_setting.setdefault(r["setting"], []).append(r)
    per_setting = {}
    for s, rs in by_setting.items():
        rs = sorted(rs, key=lambda r: r["qid"])
        e = np.array([1.0 - r["f1"] for r in rs])
        alpha_e = spectral_slope(e)["alpha"]
        dfa_e = dfa_exponent(e)["H_dfa"]
        per_setting[s] = {
            "n": len(rs),
            "em_mean": float(np.mean([r["em"] for r in rs])),
            "f1_mean": float(np.mean([r["f1"] for r in rs])),
            "sf_recall_mean": float(np.mean([r["sf_recall"] for r in rs])),
            "alpha_e": alpha_e,
            "DFA_H": dfa_e,
        }
    return {"per_setting": per_setting}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-setting", type=int, default=30)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    n = min(args.n_per_setting, len(corpus))
    client = build_client()

    all_rows = []
    total_cost = 0.0
    n_errors = 0
    log = (HERE / "run.log").open("w")
    log.write(f"exp_6_2 n_per_setting={n} n_settings={len(MEMORY_SETTINGS)}\n"); log.flush()
    import time
    t0 = time.perf_counter()
    for s_name, s_cfg in MEMORY_SETTINGS.items():
        for i in range(n):
            try:
                row = run_one_question(
                    corpus[i], client=client,
                    k=s_cfg["k"],
                    paragraph_filter=s_cfg["paragraph_filter"],
                    memory_corruption_prob=s_cfg["corrupt"],
                    memory_corruption_seed=42 + i,
                )
            except Exception as e:
                log.write(f"  ERR setting={s_name} i={i} {type(e).__name__}: {e}\n"); log.flush()
                n_errors += 1
                continue
            row["setting"] = s_name
            all_rows.append(row)
            total_cost += row["cost_usd"]
        log.write(f"  setting_done {s_name} cost=${total_cost:.4f}\n"); log.flush()
    log.close()

    (HERE / "results.json").write_text(json.dumps({"rows": all_rows}))
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {
        "exp_id": "exp_6_2",
        "n_settings": len(MEMORY_SETTINGS),
        "n_per_setting": n,
        "n_completed": len(all_rows),
        "n_errors": n_errors,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
