"""Retriever interface shared by every retrieval method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..preprocessing.chunking import Chunk


@dataclass
class RetrievalResult:
    """One retrieved chunk with its score and rank (1-indexed)."""

    chunk_id: str
    score: float
    rank: int
    text: str = ""
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "text": self.text,
            "metadata": self.metadata,
        }


class Retriever(Protocol):
    """Anything that turns queries into ranked chunks."""

    name: str

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]: ...

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievalResult]]: ...


class BaseRetriever:
    """Common plumbing: chunk bookkeeping and a default batch implementation."""

    name = "base"

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks: List[Chunk] = list(chunks)
        self.chunk_ids: List[str] = [c.chunk_id for c in self.chunks]
        self._by_id: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._by_id.get(chunk_id)

    def _make_results(self, indices: Sequence[int], scores: Sequence[float],
                      include_text: bool = True) -> List[RetrievalResult]:
        out: List[RetrievalResult] = []
        for rank, (i, s) in enumerate(zip(indices, scores), start=1):
            chunk = self.chunks[i]
            out.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(s),
                    rank=rank,
                    text=chunk.text if include_text else "",
                    doc_id=chunk.doc_id,
                    metadata=chunk.metadata,
                )
            )
        return out

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        raise NotImplementedError
