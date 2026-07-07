"""Exp 4.2 (Part IV): HotpotQA surface variants macro stability.

Uses supporting-facts subset only (§Blocker 2 from Director).
5 surface variants x 30 questions (using 40 available) = 150 runs.

Success: macro dist (EM/F1 distribution) < local variance (per-question EM
overlap between variants).
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
from metrics import error_jaccard, dist_distance  # type: ignore


SYSTEM_VARIANTS = {
    "identity": None,  # default prompt
    "paraphrase": (
        "You act as a HotpotQA multi-hop answering system. Read the passages "
        "and provide a single short answer (span, name, or yes/no). No prose."
    ),
    "renaming": (
        "You are a multi-hop QA agent. Given retrieved passages and a question, "
        "output a concise final answer (a name, a span, or yes/no)."
    ),
    "distractor": (
        "You are an evidence-based QA assistant. Passages may include "
        "irrelevant context; focus on the passages that most directly support "
        "the answer. Output only a short answer."
    ),
    "structured": (
        "TASK: HotpotQA multi-hop QA.\n"
        "INPUT: retrieved passages + question.\n"
        "OUTPUT FORMAT: single short answer, no explanation.\n"
    ),
}


def analyze(rows):
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    per_variant = {}
    for v, rs in by_variant.items():
        EM = np.array([r["em"] for r in rs])
        F1 = np.array([r["f1"] for r in rs])
        per_variant[v] = {
            "n": len(rs),
            "em_mean": float(EM.mean()),
            "f1_mean": float(F1.mean()),
            "em_std": float(EM.std()),
        }
    identity_ems = {r["qid"]: r["em"] for r in by_variant.get("identity", [])}
    id_wrong = [q for q, e in identity_ems.items() if e == 0]

    jacc = {}
    for v, rs in by_variant.items():
        if v == "identity":
            continue
        v_wrong = [r["qid"] for r in rs if r["em"] == 0]
        jacc[v] = {"mean": float(error_jaccard(id_wrong, v_wrong)),
                    "n_pairs": len(rs)}

    id_f1 = np.array([r["f1"] for r in by_variant.get("identity", [])])
    dists = {}
    for v, rs in by_variant.items():
        if v == "identity":
            continue
        v_f1 = np.array([r["f1"] for r in rs])
        dists[v] = {"f1": dist_distance(id_f1, v_f1)}
    return {"per_variant": per_variant,
            "local_wrong_qid_jaccard_vs_identity": jacc,
            "f1_distribution_distance_vs_identity": dists}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-variant", type=int, default=30)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} questions")
    n = min(args.n_per_variant, len(corpus))
    client = build_client()

    all_rows = []
    total_cost = 0.0
    n_errors = 0
    log = (HERE / "run.log").open("w")
    log.write(f"exp_4_2 n_per_variant={n} n_variants={len(SYSTEM_VARIANTS)}\n"); log.flush()
    import time
    t0 = time.perf_counter()
    for v_name, prompt_override in SYSTEM_VARIANTS.items():
        for i in range(n):
            try:
                row = run_one_question(
                    corpus[i], client=client, k=5,
                    paragraph_filter="supporting_only",
                    system_override=prompt_override,
                )
            except Exception as e:
                log.write(f"  ERR variant={v_name} i={i} {type(e).__name__}: {e}\n"); log.flush()
                n_errors += 1
                continue
            row["variant"] = v_name
            all_rows.append(row)
            total_cost += row["cost_usd"]
        log.write(f"  variant_done {v_name} cost=${total_cost:.4f}\n"); log.flush()
    log.close()

    (HERE / "results.json").write_text(json.dumps({"rows": all_rows}))
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {
        "exp_id": "exp_4_2",
        "n_variants": len(SYSTEM_VARIANTS),
        "n_per_variant": n,
        "n_completed": len(all_rows),
        "n_errors": n_errors,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
