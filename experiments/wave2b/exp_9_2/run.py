"""Exp 9.2 (Part IX): collapse universality classes — Retail.

**SECONDARY ANALYSIS**: reuses exp_1_2 trajectories. No new API calls.

Cluster tasks by macro feature vector (A_i, T_col, wsf_drop, sigma_slope,
final_F, n_gold_actions). Compare cluster labels with task_family
(inferred from instruction keywords).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def infer_family(raw: dict) -> str:
    instr = (raw.get("instruction") or "").lower()
    if any(k in instr for k in ("order id", "order status", "look up", "check")):
        return "order_lookup"
    if any(k in instr for k in ("address", "shipping")):
        return "address_change"
    if any(k in instr for k in ("refund",)):
        return "refund_request"
    if any(k in instr for k in ("exchange",)):
        return "exchange_request"
    if any(k in instr for k in ("discount", "coupon", "gift")):
        return "discount"
    return "conflicting_customer_constraints"


def cluster(rows, raws_by_task):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = []
    labels = []
    for r in rows:
        raw = raws_by_task.get(r["task_id"], {})
        family = infer_family(raw)
        sigma_series = r.get("sigma_series") or [0.0]
        X.append([
            r["A_i"], r["T_col"], r["wsf_drop"], r["final_F"],
            r["n_gold_actions"], np.mean(sigma_series),
            len(sigma_series),
        ])
        labels.append(family)
    X = np.array(X, dtype=float)
    if len(X) < 6:
        return {"error": "too few rows to cluster"}
    Xs = StandardScaler().fit_transform(X)
    k = min(6, len(X))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    cluster_ids = km.labels_.tolist()
    # cross-tab family vs cluster
    from collections import Counter
    ct = {}
    for f, c in zip(labels, cluster_ids):
        ct.setdefault(f, Counter())[c] += 1

    # within-cluster spread (avg pairwise distance)
    within_spread = []
    for c in set(cluster_ids):
        mask = np.array(cluster_ids) == c
        if mask.sum() > 1:
            centroid = Xs[mask].mean(axis=0)
            within_spread.append(float(np.linalg.norm(Xs[mask] - centroid, axis=1).mean()))
    # between-cluster spread (avg centroid distance)
    centroids = km.cluster_centers_
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    return {
        "n_rows": len(rows),
        "n_families": len(set(labels)),
        "n_clusters": k,
        "family_x_cluster_crosstab": {f: dict(cnts) for f, cnts in ct.items()},
        "mean_within_cluster_spread": float(np.mean(within_spread)) if within_spread else None,
        "mean_between_centroid_distance": float(np.mean(between)) if between else None,
        "cluster_separation_ratio": (
            float(np.mean(between)) / float(np.mean(within_spread))
            if within_spread and between and np.mean(within_spread) > 0 else None
        ),
    }


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
    agg = cluster(data["rows"], raws_by_task)
    (HERE / "aggregates.json").write_text(json.dumps(agg, indent=2))
    summary = {"exp_id": "exp_9_2_secondary",
               "n_rows": len(data["rows"]),
               "n_raws": len(raws_by_task),
               "source": str(src_p),
               "total_cost_usd": 0.0}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "aggregates": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
