"""API contract tests.

A stub service is injected so the endpoints are exercised without loading any
model or index -- these tests assert the HTTP contract (status codes, response
shape, validation, error handling), which is what actually breaks in practice.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_system.api import service as service_module
from rag_system.api.main import app
from rag_system.generation.pipeline import RAGPipeline
from rag_system.generation.readers import Answer
from rag_system.preprocessing.chunking import Chunk
from rag_system.retrieval.base import BaseRetriever


class FakeRetriever(BaseRetriever):
    name = "fake"

    def search_batch(self, queries, top_k=10):
        idx = list(range(min(top_k, len(self.chunks))))
        return [self._make_results(idx, [1.0 - 0.1 * i for i in idx]) for _ in queries]


class FakeReader:
    name = "fake-reader"
    model_name = "fake/model"

    def answer(self, question, context):
        return self.answer_batch([question], [context])[0]

    def answer_batch(self, questions, contexts):
        return [
            Answer(text=(c.split(".")[0] if c else ""), score=0.75, reader=self.name,
                   context_chars=len(c))
            for c in contexts
        ]


class FakeService:
    def __init__(self):
        self.chunks = [
            Chunk(chunk_id=f"d::{i}", doc_id="d", text=f"Sentence {i} about coronavirus. More text.",
                  start_char=0, end_char=40, chunk_index=i, metadata={"title": "Doc"})
            for i in range(6)
        ]
        retriever = FakeRetriever(self.chunks)
        self.retrievers = {"bm25": retriever, "dense": retriever, "hybrid": retriever}
        self.reranker = None
        self.reader = FakeReader()
        self.device = "cpu"
        self.cfg = _FakeCfg()
        self.pipeline = RAGPipeline(retriever, self.reader, reranker=None, top_k=5,
                                    max_context_chars=2000)
        self.uptime_seconds = 1.5
        self.info = {"embedding_model": "stub", "reader": "fake-reader"}

    def pipeline_for(self, method, rerank):
        return self.pipeline


class _FakeCfg:
    class retrieval:
        method = "hybrid"
        top_k = 5
        candidate_k = 20

    class reranker:
        model_name = "stub-reranker"

    class generation:
        max_context_chars = 2000


@pytest.fixture
def client():
    service_module.set_service(FakeService())
    with TestClient(app) as c:
        yield c
    service_module.set_service(None)


def test_health_reports_loaded_index(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["n_chunks"] == 6
    assert "models" in body


def test_health_sets_request_id_and_timing_headers(client):
    r = client.get("/health")
    assert r.headers["X-Request-ID"]
    assert float(r.headers["X-Response-Time-ms"]) >= 0


def test_retrieve_returns_ranked_results(client):
    r = client.post("/retrieve", json={"query": "coronavirus", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["n_results"] == 3
    assert [c["rank"] for c in body["results"]] == [1, 2, 3]
    assert body["latency_ms"]["total"] >= 0


def test_retrieve_rejects_unknown_method(client):
    r = client.post("/retrieve", json={"query": "x", "method": "magic"})
    assert r.status_code == 400
    assert "magic" in r.json()["detail"]


def test_retrieve_rejects_rerank_when_none_loaded(client):
    r = client.post("/retrieve", json={"query": "x", "rerank": True})
    assert r.status_code == 400


def test_retrieve_validates_empty_query(client):
    assert client.post("/retrieve", json={"query": ""}).status_code == 422


def test_retrieve_validates_top_k_bounds(client):
    assert client.post("/retrieve", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 1000}).status_code == 422


def test_query_returns_answer_with_citations_and_latency(client):
    r = client.post("/query", json={"question": "What about coronavirus?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["contexts"]) == 3
    assert set(body["latency_ms"]) >= {"retrieval", "generation", "total"}
    assert body["model_info"]["reader"] == "fake-reader"


def test_query_can_suppress_context_text(client):
    r = client.post("/query", json={"question": "q", "include_context_text": False})
    assert all(c["text"] == "" for c in r.json()["contexts"])
    # Citations must survive even when the text is suppressed.
    assert all(c["chunk_id"] for c in r.json()["contexts"])


def test_query_requires_a_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_evaluate_without_ground_truth_reports_only_proxies(client):
    r = client.post("/evaluate", json={"questions": ["a?", "b?"]})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert "exact_match" not in body["metrics"]
    assert "groundedness_proxy" in body["metrics"]
    assert "No ground truths" in body["note"]


def test_evaluate_with_ground_truth_reports_em_and_f1(client):
    r = client.post("/evaluate", json={"questions": ["a?"], "ground_truths": ["Sentence 0 about coronavirus"]})
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert "exact_match" in metrics and "f1" in metrics


def test_evaluate_rejects_mismatched_ground_truth_length(client):
    r = client.post("/evaluate", json={"questions": ["a?", "b?"], "ground_truths": ["only one"]})
    assert r.status_code == 400


def test_evaluate_rejects_empty_batch(client):
    assert client.post("/evaluate", json={"questions": []}).status_code == 422


def test_openapi_schema_is_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/query" in r.json()["paths"]
