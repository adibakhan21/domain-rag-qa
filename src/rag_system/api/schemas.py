"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(5, ge=1, le=50)
    method: Optional[str] = Field(None, description="bm25 | dense | hybrid; defaults to config")
    rerank: Optional[bool] = Field(None, description="override reranking for this request")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any] = {}


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    latency_ms: Dict[str, float]
    n_results: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    include_context_text: bool = Field(True, description="return full chunk text with citations")


class QueryResponse(BaseModel):
    question: str
    answer: str
    score: Optional[float] = None
    contexts: List[RetrievedChunk]
    latency_ms: Dict[str, float]
    model_info: Dict[str, Any]
    context_chars: int


class EvaluateRequest(BaseModel):
    """Score a batch of question/answer pairs with the same metrics as the offline harness."""

    questions: List[str] = Field(..., min_length=1, max_length=50)
    ground_truths: Optional[List[str]] = Field(
        None, description="optional references; EM/F1 are computed only when supplied"
    )
    top_k: int = Field(5, ge=1, le=20)


class EvaluateResponse(BaseModel):
    n: int
    metrics: Dict[str, float]
    per_question: List[Dict[str, Any]]
    note: str


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    index_loaded: bool
    n_chunks: int
    models: Dict[str, Any]
    uptime_seconds: float
