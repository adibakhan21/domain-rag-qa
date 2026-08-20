"""Shared machinery for running a retriever over an evaluation split."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.covidqa import QAExample
from ..reranking.cross_encoder import CrossEncoderReranker
from ..retrieval.base import BaseRetriever, RetrievalResult
from ..utils.runtime import get_logger

LOGGER = get_logger()


def run_retrieval(
    retriever: BaseRetriever,
    examples: Sequence[QAExample],
    top_k: int = 10,
    reranker: Optional[CrossEncoderReranker] = None,
    candidate_k: int = 50,
    batch_size: int = 32,
) -> Tuple[Dict[str, List[str]], Dict[str, List[RetrievalResult]], Dict[str, float]]:
    """Retrieve (and optionally rerank) for every example.

    Returns ``(run, hits, timings)`` where ``run`` maps qid -> ranked chunk ids
    for metric computation and ``hits`` keeps the full results for error
    analysis and citation display.

    Retrieval depth is ``candidate_k`` when a reranker is present and ``top_k``
    otherwise, so the two-stage cascade is timed as it would actually be served.
    """
    queries = [e.question for e in examples]
    depth = max(candidate_k, top_k) if reranker is not None else top_k

    retrieve_seconds = 0.0
    rerank_seconds = 0.0
    all_hits: List[List[RetrievalResult]] = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]

        t0 = time.perf_counter()
        hits = retriever.search_batch(batch, top_k=depth)
        retrieve_seconds += time.perf_counter() - t0

        if reranker is not None:
            t0 = time.perf_counter()
            hits = reranker.rerank_batch(batch, hits, top_k=top_k)
            rerank_seconds += time.perf_counter() - t0

        all_hits.extend(hits)

    run = {e.qid: [h.chunk_id for h in hits] for e, hits in zip(examples, all_hits)}
    hits_by_qid = {e.qid: hits for e, hits in zip(examples, all_hits)}
    timings = {
        "retrieve_seconds_total": retrieve_seconds,
        "rerank_seconds_total": rerank_seconds,
        "seconds_total": retrieve_seconds + rerank_seconds,
        "ms_per_query": 1000.0 * (retrieve_seconds + rerank_seconds) / max(1, len(queries)),
        "n_queries": len(queries),
    }
    return run, hits_by_qid, timings
