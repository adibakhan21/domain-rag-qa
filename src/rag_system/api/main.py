"""FastAPI inference service.

Endpoints
---------
GET  /health     liveness + loaded-model report
POST /retrieve   retrieval only (no reader) -- useful for debugging grounding
POST /query      full RAG: answer + cited chunks + per-stage latency
POST /evaluate   score a batch of questions with the offline metrics

Every response carries per-stage latency and the model identifiers that produced
it, so a served answer can be traced back to an exact configuration.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..evaluation.qa_metrics import answer_in_context, exact_match, groundedness, token_f1
from ..generation.pipeline import build_context
from ..utils.runtime import get_logger
from .schemas import (EvaluateRequest, EvaluateResponse, HealthResponse, QueryRequest,
                      QueryResponse, RetrievedChunk, RetrieveRequest, RetrieveResponse)
from .service import RAGService, get_service

LOGGER = get_logger()

app = FastAPI(
    title="Scientific Literature RAG for Document QA",
    description="Retrieval-augmented question answering over a corpus of scientific research papers (COVID-QA / CORD-19).",
    version=__version__,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request logging with a correlation id and server-side latency."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("[%s] %s %s failed", request_id, request.method, request.url.path)
        raise
    elapsed_ms = 1000 * (time.perf_counter() - start)
    LOGGER.info("[%s] %s %s -> %d in %.1f ms", request_id, request.method,
                request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _to_chunks(results) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=r.chunk_id, doc_id=r.doc_id, score=float(r.score),
                       rank=r.rank, text=r.text, metadata=r.metadata)
        for r in results
    ]


@app.get("/health", response_model=HealthResponse)
def health(service: RAGService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        device=service.device,
        index_loaded=len(service.chunks) > 0,
        n_chunks=len(service.chunks),
        models=service.info,
        uptime_seconds=round(service.uptime_seconds, 1),
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, service: RAGService = Depends(get_service)) -> RetrieveResponse:
    method = req.method or service.cfg.retrieval.method
    if method not in service.retrievers:
        raise HTTPException(status_code=400,
                            detail=f"unknown method {method!r}; use one of {sorted(service.retrievers)}")
    use_rerank = (service.reranker is not None) if req.rerank is None else req.rerank
    if use_rerank and service.reranker is None:
        raise HTTPException(status_code=400, detail="reranking requested but no reranker is loaded")

    depth = service.cfg.retrieval.candidate_k if use_rerank else req.top_k
    t0 = time.perf_counter()
    hits = service.retrievers[method].search_batch([req.query], top_k=depth)[0]
    retrieve_ms = 1000 * (time.perf_counter() - t0)

    rerank_ms = 0.0
    if use_rerank:
        t0 = time.perf_counter()
        hits = service.reranker.rerank_batch([req.query], [hits], top_k=req.top_k)[0]
        rerank_ms = 1000 * (time.perf_counter() - t0)
    else:
        hits = hits[: req.top_k]

    return RetrieveResponse(
        query=req.query,
        results=_to_chunks(hits),
        latency_ms={"retrieval": round(retrieve_ms, 2), "rerank": round(rerank_ms, 2),
                    "total": round(retrieve_ms + rerank_ms, 2)},
        n_results=len(hits),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, service: RAGService = Depends(get_service)) -> QueryResponse:
    response = service.pipeline.query(req.question, top_k=req.top_k)
    payload = response.to_dict(include_context_text=req.include_context_text)
    return QueryResponse(**payload)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, service: RAGService = Depends(get_service)) -> EvaluateResponse:
    if req.ground_truths is not None and len(req.ground_truths) != len(req.questions):
        raise HTTPException(status_code=400,
                            detail="ground_truths must be the same length as questions")

    responses = service.pipeline.query_batch(req.questions, top_k=req.top_k)
    per_question: List[Dict[str, Any]] = []
    for i, (question, resp) in enumerate(zip(req.questions, responses)):
        context = build_context(resp.contexts, service.pipeline.max_context_chars)
        record: Dict[str, Any] = {
            "question": question,
            "answer": resp.answer,
            "citations": resp.citations,
            "groundedness_proxy": round(groundedness(resp.answer, context), 4),
            "latency_ms": round(resp.latency_ms["total"], 2),
        }
        if req.ground_truths is not None:
            gold = req.ground_truths[i]
            record.update({
                "ground_truth": gold,
                "exact_match": exact_match(resp.answer, gold),
                "f1": round(token_f1(resp.answer, gold), 4),
                "answer_in_context": answer_in_context(gold, context),
            })
        per_question.append(record)

    keys = [k for k in ("exact_match", "f1", "answer_in_context", "groundedness_proxy", "latency_ms")
            if k in per_question[0]]
    metrics = {k: round(sum(float(r[k]) for r in per_question) / len(per_question), 4) for k in keys}

    note = ("EM/F1 computed against supplied ground truths. groundedness_proxy is a lexical "
            "overlap measure, not an LLM-judged faithfulness score.")
    if req.ground_truths is None:
        note = "No ground truths supplied: only groundedness_proxy and latency are reported. " + note

    return EvaluateResponse(n=len(per_question), metrics=metrics, per_question=per_question, note=note)
