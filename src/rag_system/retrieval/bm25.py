"""Lexical retrieval baseline (BM25 Okapi).

BM25 is the right baseline for this corpus rather than a token-overlap
heuristic: COVID-QA questions are full of rare, highly discriminative technical
terms ("DC-SIGNR", "onset-to-death"), and BM25's IDF weighting is exactly what
exploits them.  Any dense model has to beat this to justify its cost.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from ..preprocessing.chunking import Chunk
from .base import BaseRetriever, RetrievalResult

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# A short, standard stoplist.  Kept small on purpose: aggressive stopword
# removal hurts biomedical queries where words like "against" or "between"
# carry relational meaning.
STOPWORDS = frozenset("""
a an the and or but if while of to in on at by for with from as is are was were be been being
this that these those it its do does did done have has had having what which who whom when where
how why can could should would may might will shall there here their his her they them we you i
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation that preserves hyphenated terms.

    Hyphen preservation matters here: splitting "SARS-CoV-2" into three tokens
    destroys the single most discriminative term in the corpus.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.k1, self.b = k1, b
        self._tokenized = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def search_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        results: List[List[RetrievalResult]] = []
        for query in queries:
            scores = self._bm25.get_scores(tokenize(query))
            k = min(top_k, len(scores))
            # argpartition then sort only the top-k slice: O(n) instead of O(n log n).
            top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
            top = top[np.argsort(-scores[top])]
            results.append(self._make_results(top.tolist(), scores[top].tolist()))
        return results
