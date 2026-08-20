"""Dense retrieval: sentence-transformer embeddings + a FAISS index.

Index choice
------------
``IndexFlatIP`` over L2-normalised vectors gives exact cosine similarity.  For
~5k chunks an approximate index (IVF/HNSW) would add tuning parameters and
recall loss for a search that already takes single-digit milliseconds, so exact
search is the correct engineering call at this scale.  The wrapper keeps the
index behind a small interface so swapping in IVF for a larger corpus is a
one-line change.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..preprocessing.chunking import Chunk
from ..utils.config import EmbeddingConfig
from ..utils.runtime import get_logger, resolve_device
from .base import BaseRetriever, RetrievalResult

LOGGER = get_logger()


class EmbeddingModel:
    """Thin wrapper over sentence-transformers with asymmetric query/passage prefixes."""

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg or EmbeddingConfig()
        self.device = resolve_device(self.cfg.device)
        LOGGER.info("loading embedding model %s on %s", self.cfg.model_name, self.device)
        self.model = SentenceTransformer(self.cfg.model_name, device=self.device)
        # get_sentence_embedding_dimension was renamed in sentence-transformers 6;
        # support both so the project runs on 3.x-6.x.
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = getter()

    def encode(self, texts: Sequence[str], is_query: bool = False,
               show_progress: bool = False) -> np.ndarray:
        prefix = self.cfg.query_prefix if is_query else self.cfg.passage_prefix
        # The prefix is only meaningful for models trained with one (BGE/E5).
        # Applying it to a symmetric model such as all-MiniLM would corrupt the
        # embedding, so it is empty for those in the config.
        payload = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            payload,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


class FaissIndex:
    """Exact inner-product FAISS index with save/load."""

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors.astype(np.float32)))

    def search(self, queries: np.ndarray, top_k: int):
        return self.index.search(np.ascontiguousarray(queries.astype(np.float32)), top_k)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def save(self, path: Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        import faiss

        index = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        return obj


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Optional[EmbeddingModel] = None,
        embeddings: Optional[np.ndarray] = None,
        index: Optional[FaissIndex] = None,
        cfg: Optional[EmbeddingConfig] = None,
        show_progress: bool = True,
    ):
        super().__init__(chunks)
        self.embedder = embedder or EmbeddingModel(cfg)
        if index is not None:
            self.index = index
        else:
            if embeddings is None:
                LOGGER.info("embedding %d chunks", len(self.chunks))
                embeddings = self.embedder.encode(
                    [c.text for c in self.chunks], is_query=False, show_progress=show_progress
                )
            self.embeddings = embeddings
            self.index = FaissIndex(embeddings.shape[1])
            self.index.add(embeddings)
        if self.index.size != len(self.chunks):
            raise ValueError(
                f"index has {self.index.size} vectors but {len(self.chunks)} chunks were provided; "
                "the index and the chunk manifest are out of sync"
            )

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        qvecs = self.embedder.encode(list(queries), is_query=True)
        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(qvecs, top_k)
        return [
            self._make_results(idx.tolist(), sc.tolist())
            for idx, sc in zip(indices, scores)
        ]
