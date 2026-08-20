"""Failure taxonomy for RAG.

The point of this module is to answer *where* a failure came from, because
"F1 = 63" tells you nothing about what to fix.  Each failed question is assigned
to exactly one category, in priority order, using signals that are all
observable from the run artefacts:

============================  ============================================
category                       condition
============================  ============================================
``retrieval_miss``             no gold chunk anywhere in the retrieved set
``context_truncated``          gold chunk retrieved, but it fell outside the
                               context window actually passed to the reader
``chunk_boundary``             the answer span straddles a chunk boundary, so
                               no single chunk contains it -- a chunking
                               failure, not a retrieval or reader failure
``reranker_demotion``          gold chunk was in the first-stage candidates but
                               the reranker pushed it out of the top-k
``reader_miss``                gold answer *is* in the context and the reader
                               still got it wrong -- the model's fault
``partial_match``              overlapping but not exact (0 < F1 < 1)
``hallucination_suspect``      answer tokens absent from the context (only
                               reachable for a generative reader; an extractive
                               reader cannot produce one by construction)
============================  ============================================

The ordering matters: a question whose gold chunk was never retrieved is a
retrieval failure regardless of what the reader then said, so retrieval
categories are tested first.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .qa_metrics import normalize_answer

CATEGORIES = (
    "correct",
    "partial_match",
    "retrieval_miss",
    "chunk_boundary",
    "reranker_demotion",
    "context_truncated",
    "reader_miss",
    "hallucination_suspect",
)


def classify_failure(
    record: Dict[str, Any],
    first_stage_ids: Optional[Sequence[str]] = None,
    multi_chunk_gold: bool = False,
    f1_threshold: float = 0.99,
    partial_threshold: float = 0.0,
) -> str:
    """Assign one failure category to a per-question record."""
    f1 = float(record.get("f1", 0.0))
    if f1 >= f1_threshold:
        return "correct"

    retrieved = list(record.get("retrieved_ids", []))
    gold = set(record.get("gold_chunk_ids", []))
    gold_retrieved = bool(gold & set(retrieved))
    answer_present = float(record.get("answer_in_context", 0.0)) >= 1.0

    if not gold_retrieved:
        # Distinguish "the reranker lost it" from "it was never found at all".
        if first_stage_ids and (gold & set(first_stage_ids)):
            return "reranker_demotion"
        if multi_chunk_gold:
            return "chunk_boundary"
        return "retrieval_miss"

    # Gold chunk was retrieved. Did it actually reach the reader?
    if not answer_present:
        return "context_truncated"

    grounded = float(record.get("groundedness", 1.0))
    prediction = str(record.get("prediction", ""))
    if prediction and grounded < 0.5:
        return "hallucination_suspect"

    if f1 > partial_threshold:
        return "partial_match"
    return "reader_miss"


def analyse(
    records: Sequence[Dict[str, Any]],
    first_stage: Optional[Dict[str, Sequence[str]]] = None,
    multi_chunk_gold_qids: Optional[Sequence[str]] = None,
    n_examples: int = 5,
) -> Dict[str, Any]:
    """Categorise every question and collect representative examples."""
    multi = set(multi_chunk_gold_qids or [])
    categorised: List[Dict[str, Any]] = []
    for record in records:
        category = classify_failure(
            record,
            first_stage_ids=(first_stage or {}).get(record["qid"]),
            multi_chunk_gold=record["qid"] in multi,
        )
        categorised.append({**record, "category": category})

    counts = Counter(c["category"] for c in categorised)
    total = len(categorised)
    n_failures = total - counts.get("correct", 0)

    examples: Dict[str, List[Dict[str, Any]]] = {}
    for category in CATEGORIES:
        if category == "correct":
            continue
        pool = [c for c in categorised if c["category"] == category]
        # Worst-scoring first: the clearest instances of each failure mode.
        pool.sort(key=lambda c: (c.get("f1", 0.0), c.get("em", 0.0)))
        examples[category] = [
            {
                "qid": c["qid"],
                "question": c["question"],
                "gold": c["gold"],
                "prediction": c["prediction"],
                "f1": round(float(c.get("f1", 0.0)), 3),
                "gold_chunk_ids": c.get("gold_chunk_ids", []),
                "retrieved_ids": c.get("retrieved_ids", [])[:5],
                "answer_in_context": c.get("answer_in_context", 0.0),
                "groundedness": round(float(c.get("groundedness", 0.0)), 3),
            }
            for c in pool[:n_examples]
        ]

    return {
        "n_questions": total,
        "n_correct": counts.get("correct", 0),
        "n_failures": n_failures,
        "counts": {c: counts.get(c, 0) for c in CATEGORIES},
        "failure_share": {
            c: (round(100 * counts.get(c, 0) / n_failures, 1) if n_failures else 0.0)
            for c in CATEGORIES if c != "correct"
        },
        "examples": examples,
    }


def question_type(question: str) -> str:
    """Coarse wh-type bucket, used to see which question forms fail most."""
    q = question.strip().lower()
    for prefix in ("what", "when", "where", "who", "why", "how", "which", "does", "is", "are", "can"):
        if q.startswith(prefix):
            return prefix
    return "other"


def breakdown_by_question_type(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean F1 and count per question type."""
    buckets: Dict[str, List[float]] = {}
    for r in records:
        buckets.setdefault(question_type(r["question"]), []).append(float(r.get("f1", 0.0)))
    return {
        qtype: {"n": len(values), "mean_f1": round(100 * sum(values) / len(values), 2)}
        for qtype, values in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    }


def breakdown_by_answer_length(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean F1 bucketed by gold answer length -- long answers are much harder."""
    edges = [(0, 20, "1-20 chars"), (20, 60, "21-60"), (60, 150, "61-150"), (150, 10**9, "150+")]
    buckets: Dict[str, List[float]] = {label: [] for _, _, label in edges}
    for r in records:
        n = len(str(r.get("gold", "")))
        for lo, hi, label in edges:
            if lo <= n < hi:
                buckets[label].append(float(r.get("f1", 0.0)))
                break
    return {
        label: {"n": len(v), "mean_f1": round(100 * sum(v) / len(v), 2)}
        for label, v in buckets.items() if v
    }
