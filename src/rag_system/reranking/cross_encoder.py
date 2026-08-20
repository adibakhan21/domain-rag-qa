"""Cross-encoder reranking.

A bi-encoder must compress a passage into one vector before it ever sees the
query, so it cannot model term-level interaction.  A cross-encoder reads the
(query, passage) pair jointly and scores it directly, which is far more
accurate and far more expensive -- O(candidates) forward passes per query
instead of one.  The standard resolution, used here, is a two-stage cascade:
retrieve ``candidate_k`` cheaply, then rerank only those.

``candidate_k`` is the accuracy/latency dial: recall of the first stage at
depth ``candidate_k`` is a hard ceiling on what reranking can recover.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..retrieval.base import RetrievalResult
from ..utils.config import RerankerConfig
from ..utils.runtime import get_logger, resolve_device

LOGGER = get_logger()


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, cfg: Optional[RerankerConfig] = None):
        from sentence_transformers import CrossEncoder

        self.cfg = cfg or RerankerConfig()
        self.device = resolve_device(self.cfg.device)
        LOGGER.info("loading reranker %s on %s", self.cfg.model_name, self.device)
        self.model = CrossEncoder(self.cfg.model_name, device=self.device)

    def rerank(self, query: str, candidates: Sequence[RetrievalResult],
               top_k: Optional[int] = None) -> List[RetrievalResult]:
        return self.rerank_batch([query], [candidates], top_k=top_k)[0]

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidate_lists: Sequence[Sequence[RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalResult]]:
        """Rerank several queries at once.

        All (query, passage) pairs across every query are scored in one batched
        call, which keeps the GPU/MPS busy instead of issuing one small batch
        per query.
        """
        top_k = top_k or self.cfg.top_k
        pairs: List[tuple] = []
        spans: List[tuple] = []
        for query, candidates in zip(queries, candidate_lists):
            start = len(pairs)
            pairs.extend((query, c.text) for c in candidates)
            spans.append((start, len(pairs)))

        if not pairs:
            return [[] for _ in queries]

        scores = np.asarray(
            self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False),
            dtype=np.float32,
        )

        out: List[List[RetrievalResult]] = []
        for (start, end), candidates in zip(spans, candidate_lists):
            local = scores[start:end]
            order = np.argsort(-local)[:top_k]
            out.append([
                RetrievalResult(
                    chunk_id=candidates[i].chunk_id,
                    score=float(local[i]),
                    rank=rank,
                    text=candidates[i].text,
                    doc_id=candidates[i].doc_id,
                    metadata=candidates[i].metadata,
                )
                for rank, i in enumerate(order, start=1)
            ])
        return out
