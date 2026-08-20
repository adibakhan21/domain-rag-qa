"""Retrieval, indexing and fusion."""
from __future__ import annotations

import numpy as np
import pytest

from rag_system.retrieval.base import RetrievalResult
from rag_system.retrieval.bm25 import BM25Retriever, tokenize
from rag_system.retrieval.dense import DenseRetriever, FaissIndex
from rag_system.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def test_tokenizer_preserves_hyphenated_terms():
    tokens = tokenize("What is SARS-CoV-2 and the onset-to-death distribution?")
    assert "sars-cov-2" in tokens
    assert "onset-to-death" in tokens
    assert "the" not in tokens  # stopword


def test_bm25_finds_the_document_containing_a_rare_term(chunks):
    retriever = BM25Retriever(chunks)
    results = retriever.search("DC-SIGNR receptor placental", top_k=3)
    assert results
    assert results[0].doc_id == "docB"
    assert results[0].rank == 1


def test_bm25_ranks_are_sequential_and_scores_descending(chunks):
    results = BM25Retriever(chunks).search("mechanical ventilation intensive care", top_k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))


def test_retriever_never_returns_more_than_the_corpus(chunks):
    results = BM25Retriever(chunks).search("virus", top_k=999)
    assert len(results) <= len(chunks)


def test_faiss_index_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = FaissIndex(8)
    index.add(vectors)
    assert index.size == 20

    path = tmp_path / "test.faiss"
    index.save(path)
    reloaded = FaissIndex.load(path)
    assert reloaded.size == 20

    scores_a, idx_a = index.search(vectors[:3], 5)
    scores_b, idx_b = reloaded.search(vectors[:3], 5)
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_allclose(scores_a, scores_b, rtol=1e-6)


def test_faiss_self_retrieval_is_exact():
    """A normalised vector must retrieve itself with similarity ~1.0."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = FaissIndex(16)
    index.add(vectors)
    scores, indices = index.search(vectors, 1)
    assert (indices[:, 0] == np.arange(50)).all()
    np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-5)


def test_dense_retriever_with_stub_embedder(chunks, stub_embedder):
    retriever = DenseRetriever(chunks, embedder=stub_embedder, show_progress=False)
    results = retriever.search("mechanical ventilation intensive care", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r.score, float) for r in results)


def test_dense_retriever_rejects_index_chunk_mismatch(chunks, stub_embedder):
    vectors = stub_embedder.encode([c.text for c in chunks])
    index = FaissIndex(vectors.shape[1])
    index.add(vectors[:-1])  # one short
    with pytest.raises(ValueError, match="out of sync"):
        DenseRetriever(chunks, embedder=stub_embedder, index=index)


def test_rrf_promotes_documents_ranked_well_by_both_systems():
    a = [RetrievalResult("c1", 9.0, 1), RetrievalResult("c2", 5.0, 2), RetrievalResult("c3", 1.0, 3)]
    b = [RetrievalResult("c3", 0.9, 1), RetrievalResult("c1", 0.8, 2)]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=3)
    assert fused[0].chunk_id == "c1"          # 1st and 2nd
    assert [r.rank for r in fused] == [1, 2, 3]
    assert all(x.score >= y.score for x, y in zip(fused, fused[1:]))


def test_rrf_is_scale_invariant():
    """Multiplying one system's scores by 1000 must not change the fusion."""
    a = [RetrievalResult("c1", 9.0, 1), RetrievalResult("c2", 5.0, 2)]
    a_scaled = [RetrievalResult("c1", 9000.0, 1), RetrievalResult("c2", 5000.0, 2)]
    b = [RetrievalResult("c2", 0.9, 1), RetrievalResult("c1", 0.8, 2)]
    assert ([r.chunk_id for r in reciprocal_rank_fusion([a, b])]
            == [r.chunk_id for r in reciprocal_rank_fusion([a_scaled, b])])


def test_hybrid_requires_identical_chunk_ordering(chunks, stub_embedder):
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(list(reversed(chunks)), embedder=stub_embedder, show_progress=False)
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(sparse, dense)


def test_hybrid_returns_requested_depth(chunks, stub_embedder):
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder=stub_embedder, show_progress=False)
    hybrid = HybridRetriever(sparse, dense, candidate_k=10)
    results = hybrid.search("HIV-1 transmission children", top_k=4)
    assert len(results) == 4
    assert [r.rank for r in results] == [1, 2, 3, 4]
