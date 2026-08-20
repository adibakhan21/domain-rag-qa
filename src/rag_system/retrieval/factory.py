"""Build retrievers from a Config plus cached artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..data.store import index_name, load_index
from ..preprocessing.chunking import Chunk
from ..utils.config import Config
from ..utils.runtime import get_logger
from .base import BaseRetriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever, EmbeddingModel
from .hybrid import HybridRetriever

LOGGER = get_logger()

METHODS = ("bm25", "dense", "hybrid")


def build_retriever(
    cfg: Config,
    method: Optional[str] = None,
    chunks: Optional[Sequence[Chunk]] = None,
    artifacts_root: Optional[Path] = None,
) -> BaseRetriever:
    """Construct the retriever named by ``method`` (defaults to the config).

    Dense and hybrid load the prebuilt FAISS index; the loader validates that
    the index was built with this exact chunking and embedding configuration.
    """
    method = (method or cfg.retrieval.method).lower()
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method {method!r}; choose from {METHODS}")

    if method == "bm25":
        if chunks is None:
            chunks, *_ = _load(cfg, artifacts_root)
        return BM25Retriever(chunks, k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b)

    loaded_chunks, _, faiss_index, _ = _load(cfg, artifacts_root)
    dense = DenseRetriever(
        loaded_chunks,
        embedder=EmbeddingModel(cfg.embedding),
        index=faiss_index,
        cfg=cfg.embedding,
    )
    if method == "dense":
        return dense

    sparse = BM25Retriever(loaded_chunks, k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b)
    return HybridRetriever(
        sparse, dense, rrf_k=cfg.retrieval.rrf_k, candidate_k=cfg.retrieval.candidate_k
    )


def _load(cfg: Config, artifacts_root: Optional[Path]):
    name = index_name(cfg.chunking, cfg.embedding)
    return load_index(name, cfg.chunking, cfg.embedding, root=artifacts_root)
