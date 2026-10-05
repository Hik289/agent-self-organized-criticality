from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "lib"))
from wave2c.azure_client import build_client
from wave2c.hotpotqa_runner import load_corpus, run_one_question, tokenize
from wave2a.metrics import box_counting_2d


DISTRACTOR_STRUCTURES = [
    "random", "clustered_same_entity", "hub_entity",
    "reasoning_path", "multi_scale",
]


def _paragraph_similarity_graph(paragraphs: list[dict]) -> np.ndarray:
    n = len(paragraphs)
    token_sets = [set(tokenize(p["title"] + " " + " ".join(p["sentences"]))) for p in paragraphs]
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(token_sets[i] & token_sets[j])
            if shared >= 8:
                A[i, j] = A[j, i] = 1
    return A


def _fractal_over_error_positions(paragraphs: list[dict], error_indices: list[int]) -> dict:
    n = len(paragraphs)
    if n == 0 or not error_indices:
        return {"D_f": None, "n_scales": 0}

    ncols = max(3, int(np.ceil(np.sqrt(n))))
    L = max(4, ncols)
    mask = np.zeros((L, L), dtype=bool)
    for idx in error_indices:
        if idx < n:
            r = idx // ncols
            c = idx % ncols
            if r < L and c < L:
                mask[r, c] = True
    return box_counting_2d(mask)


def run_one_with_graph(item, client, distractor_kind: str):
    row = run_one_question(item, client=client, k=5, paragraph_filter="all",
                            distractor_kind=distractor_kind)
    supp_titles = {t for (t, _s) in [tuple(x) for x in item["supporting_facts"]]}
    all_paragraphs = item["paragraphs"]
    retrieved_titles = row.get("retrieved_titles", [])

    error_indices = []
    for i, p in enumerate(all_paragraphs):
        if p["title"] in retrieved_titles and p["title"] not in supp_titles:
            error_indices.append(i)
    row["error_positions"] = error_indices
    D_f = _fractal_over_error_positions(all_paragraphs, error_indices)
    row["D_f_result"] = D_f
    row["distractor_structure"] = distractor_kind
    return row


def analyze(rows):
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["distractor_structure"], []).append(r)
    per_ds = {}
    for ds, rs in by_ds.items():
        Dfs = [r["D_f_result"].get("D_f") for r in rs
               if r["D_f_result"] and r["D_f_result"].get("D_f") is not None]
        per_ds[ds] = {
            "n": len(rs),
            "em_mean": float(np.mean([r["em"] for r in rs])),
            "f1_mean": float(np.mean([r["f1"] for r in rs])),
            "sf_recall_mean": float(np.mean([r["sf_recall"] for r in rs])),
            "D_f_mean": float(np.mean(Dfs)) if Dfs else None,
            "D_f_std": float(np.std(Dfs)) if Dfs else None,
            "n_Df_computed": len(Dfs),
        }
    return {"per_distractor_structure": per_ds}


def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-ds", type=int, default=30)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    n = min(args.n_per_ds, len(corpus))
    client = build_client()
    all_rows = []
    total_cost = 0.0
    n_errors = 0
    log = (HERE / "run.log").open("w")
    log.write(f"exp_8_1 n_per_ds={n} n_structures={len(DISTRACTOR_STRUCTURES)}\n"); log.flush()
    t0 = time.perf_counter()
    for ds in DISTRACTOR_STRUCTURES:
        for i in range(n):
            try:
                row = run_one_with_graph(corpus[i], client, ds)
            except Exception as e:
                log.write(f"  ERR ds={ds} i={i} {type(e).__name__}: {e}\n"); log.flush()
                n_errors += 1
                continue
            all_rows.append(row)
            total_cost += row["cost_usd"]
        log.write(f"  ds_done {ds} cost=${total_cost:.4f}\n"); log.flush()
    log.close()

    (HERE / "results.json").write_text(json.dumps({"rows": all_rows}, default=str))
    agg = analyze(all_rows)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {
        "exp_id": "exp_8_1",
        "n_distractor_structures": len(DISTRACTOR_STRUCTURES),
        "n_per_ds": n,
        "n_completed": len(all_rows),
        "n_errors": n_errors,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "total_cost_usd": round(total_cost, 6),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
