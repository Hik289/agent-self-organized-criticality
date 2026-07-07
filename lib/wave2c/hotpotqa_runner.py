"""HotpotQA-RAG runner."""
from __future__ import annotations
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi
from .azure_client import build_client, AZURE_DEPLOYMENT, price


CORPUS_PATH = "./experiments/wave1/harness/hotpotqa_rag/corpus.jsonl"
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(p_toks)
    r = num_same / len(g_toks)
    return 2 * p * r / (p + r)


def load_corpus() -> list[dict]:
    items = []
    with open(CORPUS_PATH) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def retrieve_bm25(query: str, paragraphs: list[dict], k: int) -> list[int]:
    if k <= 0 or not paragraphs:
        return []
    docs = [tokenize(p["title"] + " " + " ".join(p["sentences"])) for p in paragraphs]
    bm25 = BM25Okapi(docs)
    q = tokenize(query)
    scores = bm25.get_scores(q)
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return order[:k]


def build_answer_prompt(question: str, retrieved: list[dict], system_override: str = None) -> list[dict]:
    ctx_blocks = []
    for i, p in enumerate(retrieved):
        joined = " ".join(p["sentences"])
        ctx_blocks.append(f"[{i+1}] Title: {p['title']}\n{joined}")
    ctx = "\n\n".join(ctx_blocks) if ctx_blocks else "(no retrieved passages)"
    system = system_override or (
        "You are answering HotpotQA multi-hop questions using the retrieved "
        "passages. Give a single short answer (a span, a name, or yes/no). Do not explain."
    )
    user = f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_one_question(item: dict, *, client=None,
                     k: int = 5,
                     paragraph_filter: str = "all",
                     distractor_kind: str = "default",
                     system_override: str = None,
                     question_transform: str = "identity",
                     memory_corruption_prob: float = 0.0,
                     memory_corruption_seed: int = 42) -> dict:
    if client is None:
        client = build_client()

    supp = [tuple(x) for x in item["supporting_facts"]]
    supp_titles = {t for (t, _s) in supp}
    all_paragraphs = item["paragraphs"]

    if paragraph_filter == "supporting_only":
        paragraphs = [p for p in all_paragraphs if p["title"] in supp_titles]
    elif paragraph_filter == "distractor_only":
        paragraphs = [p for p in all_paragraphs if p["title"] not in supp_titles]
    else:
        paragraphs = list(all_paragraphs)

    # Part VIII: apply distractor_kind — restructure the paragraph pool
    if distractor_kind and distractor_kind != "default":
        supp_paras = [p for p in all_paragraphs if p["title"] in supp_titles]
        distr = [p for p in all_paragraphs if p["title"] not in supp_titles]
        supp_tokens_all = set()
        for p in supp_paras:
            supp_tokens_all |= set(tokenize(p["title"] + " " + " ".join(p["sentences"])))
        supp_title_tokens = set()
        for p in supp_paras:
            supp_title_tokens |= set(tokenize(p["title"]))

        def _sim(p):
            p_tokens = set(tokenize(p["title"] + " " + " ".join(p["sentences"])))
            return len(p_tokens & supp_tokens_all)

        if distractor_kind == "random":
            paragraphs = list(all_paragraphs)
        elif distractor_kind == "clustered_same_entity":
            filt = [p for p in distr if _sim(p) >= 3]
            if len(filt) < 3:
                filt = distr
            paragraphs = supp_paras + filt
        elif distractor_kind == "hub_entity":
            distr_ranked = sorted(distr, key=lambda p: -_sim(p))
            keep_n = max(2, len(distr_ranked) // 2)
            paragraphs = supp_paras + distr_ranked[:keep_n]
        elif distractor_kind == "reasoning_path":
            path_distr = [p for p in distr
                          if len(set(tokenize(p["title"])) & supp_title_tokens) >= 1
                          or any(t in " ".join(p["sentences"]).lower()
                                 for t in supp_title_tokens if len(t) >= 4)]
            if len(path_distr) < 2:
                path_distr = distr[:5]
            paragraphs = supp_paras + path_distr
        elif distractor_kind == "multi_scale":
            distr_ranked = sorted(distr, key=lambda p: -_sim(p))
            n = len(distr_ranked)
            if n >= 4:
                subset = distr_ranked[:2] + distr_ranked[-2:]
            else:
                subset = distr_ranked
            paragraphs = supp_paras + subset

    corrupted_positions = []
    if memory_corruption_prob > 0 and paragraphs:
        import numpy as np
        rng = np.random.default_rng(memory_corruption_seed)
        distr = [p for p in all_paragraphs if p["title"] not in supp_titles]
        for i, p in enumerate(paragraphs):
            if p["title"] in supp_titles and rng.random() < memory_corruption_prob:
                if distr:
                    swap = distr[rng.integers(0, len(distr))]
                    paragraphs[i] = {"title": p["title"], "sentences": list(swap["sentences"])}
                    corrupted_positions.append(i)

    q_use = item["question"]
    if question_transform == "paraphrase":
        q_use = "Please tell me: " + q_use

    top_idx = retrieve_bm25(q_use, paragraphs, k)
    retrieved = [paragraphs[i] for i in top_idx]

    supp_hit = 0
    retrieved_titles = [p["title"] for p in retrieved]
    for t in retrieved_titles:
        if t in supp_titles:
            supp_hit += 1
    sf_recall = (supp_hit / max(1, len(supp_titles))) if supp_titles else 0.0

    messages = build_answer_prompt(q_use, retrieved, system_override=system_override)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT, messages=messages, seed=42, temperature=0.0,
        )
        pred = (resp.choices[0].message.content or "").strip()
        p_tok = resp.usage.prompt_tokens or 0
        c_tok = resp.usage.completion_tokens or 0
        cost = price(resp.usage)
        err = None
    except Exception as e:
        pred = ""
        p_tok = c_tok = 0
        cost = 0.0
        err = f"{type(e).__name__}: {e}"
    latency_ms = (time.perf_counter() - t0) * 1000.0
    em = em_score(pred, item["answer"])
    f1 = f1_score(pred, item["answer"])
    return {
        "qid": item["qid"],
        "question": q_use,
        "gold_answer": item["answer"],
        "pred_answer": pred,
        "em": em, "f1": f1,
        "sf_recall": sf_recall,
        "n_retrieved": len(retrieved),
        "retrieved_titles": retrieved_titles,
        "corrupted_positions": corrupted_positions,
        "level": item.get("level"),
        "type": item.get("type"),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "error": err,
        "paragraph_filter": paragraph_filter,
        "distractor_kind": distractor_kind,
        "question_transform": question_transform,
        "memory_corruption_prob": memory_corruption_prob,
        "k": k,
    }
