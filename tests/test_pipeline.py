"""RAG pipeline assembly and context construction."""
from __future__ import annotations

from rag_system.generation.pipeline import RAGPipeline, build_context
from rag_system.generation.readers import Answer
from rag_system.retrieval.base import RetrievalResult
from rag_system.retrieval.bm25 import BM25Retriever


class EchoReader:
    name = "echo"
    model_name = "echo/1"

    def answer(self, question, context):
        return self.answer_batch([question], [context])[0]

    def answer_batch(self, questions, contexts):
        return [Answer(text=c[:40], score=1.0, reader=self.name, context_chars=len(c))
                for c in contexts]


def _results(n=3, size=100):
    return [RetrievalResult(f"c{i}", 1.0, i + 1, text=chr(97 + i) * size, doc_id="d")
            for i in range(n)]


def test_build_context_truncates_at_chunk_boundaries():
    ctx = build_context(_results(3, 100), max_chars=250)
    # Two 100-char chunks + one separator fit; a third would exceed the budget.
    assert ctx.count("\n\n") == 1
    assert len(ctx) <= 250


def test_build_context_keeps_at_least_one_oversized_chunk():
    ctx = build_context(_results(3, 100), max_chars=50)
    assert len(ctx) == 50 and ctx  # never returns empty


def test_build_context_preserves_rank_order():
    ctx = build_context(_results(3, 10), max_chars=1000)
    assert ctx.index("aaa") < ctx.index("bbb") < ctx.index("ccc")


def test_build_context_handles_no_results():
    assert build_context([], max_chars=100) == ""


def test_pipeline_returns_citations_and_latency(chunks):
    pipe = RAGPipeline(BM25Retriever(chunks), EchoReader(), top_k=3, max_context_chars=1000)
    response = pipe.query("HIV-1 transmission in children")
    assert response.answer
    assert len(response.citations) == 3
    assert response.latency_ms["total"] >= 0
    assert response.model_info["reader"] == "echo"


def test_pipeline_batch_matches_single_query(chunks):
    pipe = RAGPipeline(BM25Retriever(chunks), EchoReader(), top_k=2, max_context_chars=1000)
    single = pipe.query("DC-SIGNR receptor")
    batched = pipe.query_batch(["DC-SIGNR receptor"])[0]
    assert single.citations == batched.citations
    assert single.answer == batched.answer


def test_response_serialises_nan_score_as_null(chunks):
    class NanReader(EchoReader):
        def answer_batch(self, questions, contexts):
            return [Answer(text="x", score=float("nan"), reader="nan") for _ in contexts]

    pipe = RAGPipeline(BM25Retriever(chunks), NanReader(), top_k=1)
    assert pipe.query("virus").to_dict()["score"] is None
