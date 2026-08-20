"""Lazily-constructed singleton holding the loaded models and index.

Models are loaded once at startup, not per request: loading the reader and
reranker takes seconds, which would dominate request latency and make the
service unusable under any concurrency.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..data.store import read_chunks
from ..generation.pipeline import RAGPipeline
from ..generation.readers import build_reader
from ..reranking.cross_encoder import CrossEncoderReranker
from ..retrieval.factory import build_retriever
from ..utils.config import ARTIFACT_DIR, CONFIG_DIR, Config, load_config
from ..utils.runtime import get_logger, resolve_device

LOGGER = get_logger()


class RAGService:
    """Holds every heavyweight object the API needs."""

    def __init__(self, config_path: Optional[Path] = None, corpus_dir: Optional[Path] = None,
                 adapter_path: Optional[str] = None):
        self.config_path = Path(config_path or CONFIG_DIR / "rag.yaml")
        self.corpus_dir = Path(corpus_dir or ARTIFACT_DIR / "corpus")
        self.cfg: Config = load_config(self.config_path)
        if adapter_path:
            self.cfg.generation.adapter_path = adapter_path

        self.started_at = time.time()
        self.device = resolve_device(self.cfg.generation.device)
        self._lock = threading.Lock()

        LOGGER.info("loading chunks from %s", self.corpus_dir)
        self.chunks = read_chunks(self.corpus_dir / "chunks.jsonl")

        self.retrievers: Dict[str, Any] = {
            "hybrid": build_retriever(self.cfg, method="hybrid", chunks=self.chunks),
        }
        # BM25 shares the chunk list; dense reuses the hybrid retriever's index.
        self.retrievers["bm25"] = self.retrievers["hybrid"].sparse
        self.retrievers["dense"] = self.retrievers["hybrid"].dense

        self.reranker = CrossEncoderReranker(self.cfg.reranker) if self.cfg.reranker.enabled else None
        self.reader = build_reader(self.cfg.generation)
        self.pipeline = RAGPipeline.from_config(
            self.cfg, self.retrievers[self.cfg.retrieval.method], self.reader, self.reranker
        )
        LOGGER.info("service ready: %d chunks, device=%s", len(self.chunks), self.device)

    def pipeline_for(self, method: Optional[str], rerank: Optional[bool]) -> RAGPipeline:
        """A pipeline variant for a single request, without reloading models."""
        method = method or self.cfg.retrieval.method
        if method not in self.retrievers:
            raise KeyError(f"unknown retrieval method {method!r}; use one of {sorted(self.retrievers)}")
        use_rerank = self.reranker is not None if rerank is None else bool(rerank)
        return RAGPipeline(
            retriever=self.retrievers[method],
            reader=self.reader,
            reranker=self.reranker if use_rerank else None,
            top_k=self.cfg.retrieval.top_k,
            candidate_k=self.cfg.retrieval.candidate_k,
            max_context_chars=self.cfg.generation.max_context_chars,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.cfg.embedding.model_name,
            "reranker_model": self.cfg.reranker.model_name if self.reranker else None,
            "reader": self.reader.name,
            "reader_model": getattr(self.reader, "model_name", None),
            "adapter": self.cfg.generation.adapter_path,
            "retrieval_method": self.cfg.retrieval.method,
            "top_k": self.cfg.retrieval.top_k,
            "chunk_size": self.cfg.chunking.chunk_size,
            "chunk_overlap": self.cfg.chunking.chunk_overlap,
        }

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at


_SERVICE: Optional[RAGService] = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> RAGService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = RAGService()
    return _SERVICE


def set_service(service: Optional[RAGService]) -> None:
    """Inject a service instance (used by tests to avoid loading real models)."""
    global _SERVICE
    _SERVICE = service
