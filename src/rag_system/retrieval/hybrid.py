"""Hybrid retrieval by Reciprocal Rank Fusion.

Why RRF rather than a weighted score sum
----------------------------------------
BM25 scores are unbounded and corpus-dependent; cosine similarities live in
[-1, 1].  Combining them by weighted sum requires per-corpus normalisation and
a tuned weight, and the tuned weight will not transfer.  RRF fuses *ranks*
instead of scores:

    RRF(d) = sum_r  1 / (k + rank_r(d))

It is scale-free, has one insensitive hyperparameter (k, conventionally 60),
and needs no tuning set -- which matters here because the validation split is
small enough that a tuned weight would mostly fit noise.  The trade-off is that
RRF discards score magnitude, so a document that BM25 is *overwhelmingly*
confident about is treated the same as a merely top-ranked one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .base import BaseRetriever, RetrievalResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]], k: int = 60, top_k: int = 10
) -> List[RetrievalResult]:
    """Fuse several ranked lists of the same corpus into one."""
    scores: Dict[str, float] = {}
    exemplar: Dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for result in ranking:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + result.rank)
            exemplar.setdefault(result.chunk_id, result)

    ordered: List[Tuple[str, float]] = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    fused: List[RetrievalResult] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        src = exemplar[chunk_id]
        fused.append(
            RetrievalResult(
                chunk_id=chunk_id, score=float(score), rank=rank,
                text=src.text, doc_id=src.doc_id, metadata=src.metadata,
            )
        )
    return fused


class HybridRetriever(BaseRetriever):
    """BM25 + dense retrieval fused with RRF."""

    name = "hybrid"

    def __init__(self, sparse: BaseRetriever, dense: BaseRetriever, rrf_k: int = 60,
                 candidate_k: int = 50):
        if sparse.chunk_ids != dense.chunk_ids:
            raise ValueError("sparse and dense retrievers must index the same chunks in the same order")
        super().__init__(sparse.chunks)
        self.sparse, self.dense = sparse, dense
        self.rrf_k, self.candidate_k = rrf_k, candidate_k

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        depth = max(self.candidate_k, top_k)
        sparse_hits = self.sparse.search_batch(queries, top_k=depth)
        dense_hits = self.dense.search_batch(queries, top_k=depth)
        return [
            reciprocal_rank_fusion([s, d], k=self.rrf_k, top_k=top_k)
            for s, d in zip(sparse_hits, dense_hits)
        ]
