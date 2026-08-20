"""Retrieval metrics with bootstrap confidence intervals.

Two different "recall" notions are reported because they answer different
questions, and conflating them is a common way to overstate a retriever:

* ``recall@k``  -- |retrieved_k ∩ gold| / |gold|.  Standard IR recall.  Because
  chunks overlap, a question can have several near-duplicate gold chunks, and
  finding only one of them caps this metric below 1.0 even though the answer
  was found.
* ``hit@k`` (a.k.a. success@k) -- 1 if *any* gold chunk is in the top k.  This
  is the operationally meaningful one for RAG: the reader only needs the answer
  to appear once in the context window.

``hit@k`` is the headline metric for this project, with ``recall@k`` reported
alongside it so the gap is visible rather than hidden.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


def _relevance_vector(ranked_ids: Sequence[str], gold: Sequence[str]) -> np.ndarray:
    gold_set = set(gold)
    return np.array([1.0 if cid in gold_set else 0.0 for cid in ranked_ids], dtype=np.float64)


def recall_at_k(ranked_ids: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    found = len(set(ranked_ids[:k]) & set(gold))
    return found / len(set(gold))


def hit_at_k(ranked_ids: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(ranked_ids[:k]) & set(gold) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """1 / rank of the first relevant chunk within the top k, else 0."""
    gold_set = set(gold)
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """Binary-relevance nDCG.

    The ideal ranking places ``min(|gold|, k)`` relevant chunks at the top, so
    a question with more gold chunks than ``k`` is not penalised for the
    impossible.
    """
    if not gold:
        return float("nan")
    rel = _relevance_vector(ranked_ids[:k], gold)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * discounts).sum())

    n_ideal = min(len(set(gold)), k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, n_ideal + 2))
    idcg = float(ideal_discounts.sum())
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    mrr_k: int = 10,
    ndcg_k: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """Score a run (qid -> ranked chunk ids) against qrels (qid -> gold ids).

    Only questions present in ``qrels`` are scored; a question missing from
    ``run`` counts as a total failure rather than being skipped, so an
    incomplete run cannot look better than a complete one.
    """
    qids = [q for q in qrels if qrels[q]]
    if not qids:
        raise ValueError("qrels contains no question with gold chunks")

    per_query: Dict[str, Dict[str, float]] = {}
    for qid in qids:
        ranked = list(run.get(qid, []))
        gold = qrels[qid]
        scores: Dict[str, float] = {}
        for k in k_values:
            scores[f"recall@{k}"] = recall_at_k(ranked, gold, k)
            scores[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        scores[f"mrr@{mrr_k}"] = reciprocal_rank(ranked, gold, mrr_k)
        scores[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked, gold, ndcg_k)
        per_query[qid] = scores

    metric_names = list(next(iter(per_query.values())).keys())
    matrix = {m: np.array([per_query[q][m] for q in qids], dtype=np.float64) for m in metric_names}

    summary: Dict[str, object] = {"n_queries": len(qids)}
    for m, values in matrix.items():
        summary[m] = float(values.mean())
    if n_bootstrap:
        summary["ci95"] = _bootstrap_ci(matrix, n_bootstrap=n_bootstrap, seed=seed)
    return {"summary": summary, "per_query": per_query}


def _bootstrap_ci(matrix: Dict[str, np.ndarray], n_bootstrap: int = 1000,
                  seed: int = 42) -> Dict[str, List[float]]:
    """Percentile bootstrap CI over questions.

    Reported because the test split has only a few hundred questions -- a 2-point
    difference in hit@5 between two retrievers is well inside the noise, and
    presenting point estimates alone would invite over-reading it.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(matrix.values())))
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    out: Dict[str, List[float]] = {}
    for m, values in matrix.items():
        means = values[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[m] = [float(lo), float(hi)]
    return out


def paired_bootstrap_test(
    per_query_a: Mapping[str, Mapping[str, float]],
    per_query_b: Mapping[str, Mapping[str, float]],
    metric: str,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap on the per-question difference B - A.

    Paired rather than independent: both systems are scored on exactly the same
    questions, so pairing removes question difficulty as a source of variance
    and gives a far tighter (and honest) test of whether B beats A.
    """
    common = sorted(set(per_query_a) & set(per_query_b))
    if not common:
        raise ValueError("no overlapping questions between the two runs")
    diffs = np.array([per_query_b[q][metric] - per_query_a[q][metric] for q in common], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided p-value: proportion of resamples on the other side of zero.
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "metric": metric,
        "n_queries": len(common),
        "mean_delta": float(diffs.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(min(1.0, p)),
        "significant_at_0.05": bool(min(1.0, p) < 0.05),
    }
