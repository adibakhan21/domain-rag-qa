"""Shared fixtures.

The whole suite is deterministic and downloads nothing: every test builds its
own tiny corpus and, where a model would be needed, uses a stub.  That keeps
`pytest` runnable in CI and on a laptop with no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_system.preprocessing.chunking import Chunk, chunk_document  # noqa: E402

DOC_A = (
    "SARS-CoV-2 is a betacoronavirus. It causes the disease COVID-19. "
    "The incubation period is estimated at five days. "
    "Transmission occurs mainly through respiratory droplets. "
    "Severe cases may require mechanical ventilation in intensive care. "
)
DOC_B = (
    "DC-SIGNR is a receptor expressed on placental endothelial cells. "
    "Mother-to-child transmission is the main cause of HIV-1 infection in children. "
    "Genetic variants in DC-SIGNR are associated with transmission risk. "
    "Antiretroviral therapy reduces vertical transmission substantially. "
)


@pytest.fixture
def documents() -> dict:
    return {"docA": DOC_A * 3, "docB": DOC_B * 3}


@pytest.fixture
def chunks(documents) -> List[Chunk]:
    out: List[Chunk] = []
    for doc_id, text in documents.items():
        out.extend(chunk_document(text, doc_id, chunk_size=200, chunk_overlap=40, min_chunk_size=20))
    return out


class StubEmbedder:
    """Deterministic hashing embedder: no model download, but a real vector space.

    Token-hash bag-of-words projected onto a fixed-dimension vector and L2
    normalised.  Semantically crude, but it makes dense retrieval genuinely
    exercise the FAISS path and behave sensibly on exact-term overlap.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.cfg = type("Cfg", (), {"batch_size": 8, "model_name": "stub", "normalize": True})()

    def encode(self, texts, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                vecs[i, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()
