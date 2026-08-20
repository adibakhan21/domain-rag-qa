"""The RAG pipeline: retrieve -> (rerank) -> build context -> read -> cite."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..reranking.cross_encoder import CrossEncoderReranker
from ..retrieval.base import BaseRetriever, RetrievalResult
from ..utils.config import Config
from ..utils.runtime import get_logger
from .readers import Answer, Reader

LOGGER = get_logger()


@dataclass
class RAGResponse:
    """Everything needed to inspect and trust an answer."""

    question: str
    answer: str
    score: float
    contexts: List[RetrievalResult] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    context_chars: int = 0

    def to_dict(self, include_context_text: bool = True) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "score": None if self.score != self.score else round(float(self.score), 6),  # NaN -> null
            "contexts": [
                {**c.to_dict(), "text": c.text if include_context_text else ""}
                for c in self.contexts
            ],
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "model_info": self.model_info,
            "context_chars": self.context_chars,
        }

    @property
    def citations(self) -> List[str]:
        return [c.chunk_id for c in self.contexts]


def build_context(results: Sequence[RetrievalResult], max_chars: int,
                  separator: str = "\n\n") -> str:
    """Concatenate retrieved chunks into one context string, highest-ranked first.

    Truncation is at chunk granularity rather than mid-chunk: a half-chunk can
    cut an answer span in two, which would make the reader fail for a reason
    that has nothing to do with retrieval quality.  The single exception is a
    first chunk already longer than the budget, which is cut so that the context
    is never empty.
    """
    parts: List[str] = []
    used = 0
    for r in results:
        piece = r.text
        if not parts and len(piece) > max_chars:
            parts.append(piece[:max_chars])
            used = max_chars
            break
        if used + len(piece) + (len(separator) if parts else 0) > max_chars:
            break
        used += len(piece) + (len(separator) if parts else 0)
        parts.append(piece)
    return separator.join(parts)


class RAGPipeline:
    """Question in, grounded answer plus citations out."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reader: Reader,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 5,
        candidate_k: int = 50,
        max_context_chars: int = 4000,
    ):
        self.retriever = retriever
        self.reader = reader
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.max_context_chars = max_context_chars

    @classmethod
    def from_config(cls, cfg: Config, retriever: BaseRetriever, reader: Reader,
                    reranker: Optional[CrossEncoderReranker] = None) -> "RAGPipeline":
        return cls(
            retriever=retriever,
            reader=reader,
            reranker=reranker if cfg.reranker.enabled else None,
            top_k=cfg.retrieval.top_k,
            candidate_k=cfg.retrieval.candidate_k,
            max_context_chars=cfg.generation.max_context_chars,
        )

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "retriever": self.retriever.name,
            "reranker": self.reranker.cfg.model_name if self.reranker else None,
            "reader": self.reader.name,
            "reader_model": getattr(self.reader, "model_name", None),
            "top_k": self.top_k,
            "candidate_k": self.candidate_k if self.reranker else self.top_k,
            "max_context_chars": self.max_context_chars,
        }

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        return self.query_batch([question], top_k=top_k)[0]

    def query_batch(self, questions: Sequence[str], top_k: Optional[int] = None) -> List[RAGResponse]:
        top_k = top_k or self.top_k
        depth = max(self.candidate_k, top_k) if self.reranker else top_k

        t0 = time.perf_counter()
        hits = self.retriever.search_batch(list(questions), top_k=depth)
        retrieve_ms = 1000 * (time.perf_counter() - t0)

        rerank_ms = 0.0
        if self.reranker is not None:
            t0 = time.perf_counter()
            hits = self.reranker.rerank_batch(list(questions), hits, top_k=top_k)
            rerank_ms = 1000 * (time.perf_counter() - t0)
        else:
            hits = [h[:top_k] for h in hits]

        contexts = [build_context(h, self.max_context_chars) for h in hits]

        t0 = time.perf_counter()
        answers = self.reader.answer_batch(list(questions), contexts)
        read_ms = 1000 * (time.perf_counter() - t0)

        n = max(1, len(questions))
        responses: List[RAGResponse] = []
        for question, hit_list, context, ans in zip(questions, hits, contexts, answers):
            responses.append(
                RAGResponse(
                    question=question,
                    answer=ans.text,
                    score=ans.score,
                    contexts=hit_list,
                    context_chars=len(context),
                    latency_ms={
                        "retrieval": retrieve_ms / n,
                        "rerank": rerank_ms / n,
                        "generation": read_ms / n,
                        "total": (retrieve_ms + rerank_ms + read_ms) / n,
                    },
                    model_info=self.model_info,
                )
            )
        return responses
