#!/usr/bin/env python
"""Phase 2 ablation grid: chunk size and embedding model.

    python scripts/run_experiments.py --ablation chunking
    python scripts/run_experiments.py --ablation embedding
    python scripts/run_experiments.py --ablation all

Each cell rebuilds the corpus and the vector index from scratch for the
configuration under test, so a result can never be produced by a stale index.
Writes results/retrieval/ablation_<name>.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402  (fixes faiss/torch OpenMP load order)

from rag_system.data.covidqa import build_corpus                           # noqa: E402
from rag_system.evaluation.retrieval_metrics import evaluate_run           # noqa: E402
from rag_system.evaluation.runner import run_retrieval                     # noqa: E402
from rag_system.preprocessing.chunking import chunk_stats                  # noqa: E402
from rag_system.reranking.cross_encoder import CrossEncoderReranker        # noqa: E402
from rag_system.retrieval.bm25 import BM25Retriever                        # noqa: E402
from rag_system.retrieval.dense import DenseRetriever, EmbeddingModel      # noqa: E402
from rag_system.retrieval.hybrid import HybridRetriever                    # noqa: E402
from rag_system.utils.config import RESULTS_DIR, load_config               # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, write_json      # noqa: E402
from rag_system.utils.tracking import start_run                            # noqa: E402

LOGGER = get_logger()

CHUNK_SIZES = [500, 1000, 2000]
EMBEDDING_MODELS = [
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    ("sentence-transformers/all-MiniLM-L6-v2", ""),   # symmetric model: no query instruction
]


def evaluate_cell(cfg, split: str, reranker: Optional[CrossEncoderReranker],
                  with_rerank: bool) -> Dict[str, object]:
    """Build corpus + index for this config and score every retrieval method."""
    set_seed(cfg.experiment.seed)
    corpus = build_corpus(cfg.data, cfg.chunking)
    examples = corpus.examples_for(split)
    qrels = corpus.qrels(split)

    embedder = EmbeddingModel(cfg.embedding)
    t0 = time.perf_counter()
    dense = DenseRetriever(corpus.chunks, embedder=embedder, cfg=cfg.embedding, show_progress=False)
    index_seconds = time.perf_counter() - t0
    sparse = BM25Retriever(corpus.chunks, k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b)
    hybrid = HybridRetriever(sparse, dense, rrf_k=cfg.retrieval.rrf_k,
                             candidate_k=cfg.retrieval.candidate_k)

    systems = {"bm25": (sparse, False), "dense": (dense, False), "hybrid_rrf": (hybrid, False)}
    if with_rerank and reranker is not None:
        systems["hybrid_rrf_rerank"] = (hybrid, True)

    max_k = max(cfg.evaluation.recall_at_k)
    out: Dict[str, object] = {
        "chunk_stats": chunk_stats(corpus.chunks),
        "index_build_seconds": round(index_seconds, 2),
        "n_questions": len(examples),
        "systems": {},
    }
    for label, (retriever, use_rr) in systems.items():
        run, _, timings = run_retrieval(
            retriever, examples, top_k=max_k,
            reranker=reranker if use_rr else None,
            candidate_k=cfg.retrieval.candidate_k,
        )
        scored = evaluate_run(run, qrels, k_values=cfg.evaluation.recall_at_k,
                              mrr_k=cfg.evaluation.mrr_at_k, ndcg_k=cfg.evaluation.ndcg_at_k,
                              n_bootstrap=0)
        out["systems"][label] = {"metrics": scored["summary"], "timings": timings}
    return out


def ablation_chunking(cfg_path: Path, split: str, with_rerank: bool) -> Dict[str, object]:
    """Vary chunk size with overlap fixed at 20% of size."""
    reranker = CrossEncoderReranker(load_config(cfg_path).reranker) if with_rerank else None
    cells: Dict[str, object] = {}
    for size in CHUNK_SIZES:
        cfg = load_config(cfg_path)
        cfg.chunking.chunk_size = size
        cfg.chunking.chunk_overlap = size // 5
        LOGGER.info("=== chunk_size=%d overlap=%d ===", size, cfg.chunking.chunk_overlap)
        cell = evaluate_cell(cfg, split, reranker, with_rerank)
        cell["chunk_size"] = size
        cell["chunk_overlap"] = cfg.chunking.chunk_overlap
        cells[f"chunk_{size}"] = cell
        for label, r in cell["systems"].items():
            m = r["metrics"]
            LOGGER.info("  %-18s hit@1 %.3f hit@5 %.3f mrr@10 %.3f (%d chunks)",
                        label, m["hit@1"], m["hit@5"], m["mrr@10"], cell["chunk_stats"]["n_chunks"])
        with start_run(f"ablation-chunk-{size}", cfg.experiment.tracking_uri,
                       cfg.experiment.experiment_name) as run_logger:
            run_logger.set_tags({"phase": "ablation", "ablation": "chunking"})
            run_logger.log_params(cfg.flat_params())
            for label, r in cell["systems"].items():
                run_logger.log_metrics({f"{label}.{k}": v for k, v in r["metrics"].items()
                                        if isinstance(v, (int, float))})
    return cells


def ablation_embedding(cfg_path: Path, split: str) -> Dict[str, object]:
    """Compare embedding models at the default chunk size."""
    cells: Dict[str, object] = {}
    for model_name, prefix in EMBEDDING_MODELS:
        cfg = load_config(cfg_path)
        cfg.embedding.model_name = model_name
        cfg.embedding.query_prefix = prefix
        LOGGER.info("=== embedding=%s ===", model_name)
        cell = evaluate_cell(cfg, split, reranker=None, with_rerank=False)
        cell["embedding_model"] = model_name
        cell["query_prefix"] = prefix
        cells[model_name.split("/")[-1]] = cell
        for label, r in cell["systems"].items():
            m = r["metrics"]
            LOGGER.info("  %-18s hit@1 %.3f hit@5 %.3f mrr@10 %.3f", label, m["hit@1"], m["hit@5"], m["mrr@10"])
        with start_run(f"ablation-emb-{model_name.split('/')[-1]}", cfg.experiment.tracking_uri,
                       cfg.experiment.experiment_name) as run_logger:
            run_logger.set_tags({"phase": "ablation", "ablation": "embedding"})
            run_logger.log_params(cfg.flat_params())
            for label, r in cell["systems"].items():
                run_logger.log_metrics({f"{label}.{k}": v for k, v in r["metrics"].items()
                                        if isinstance(v, (int, float))})
    return cells


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.yaml")
    p.add_argument("--ablation", choices=["chunking", "embedding", "all"], default="all")
    p.add_argument("--split", default="test")
    p.add_argument("--with-rerank", action="store_true",
                   help="also score hybrid+rerank in the chunking ablation (slow)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "retrieval")
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.ablation in ("chunking", "all"):
        cells = ablation_chunking(args.config, args.split, args.with_rerank)
        write_json(args.out_dir / "ablation_chunking.json",
                   {"split": args.split, "with_rerank": args.with_rerank, "cells": cells})
    if args.ablation in ("embedding", "all"):
        cells = ablation_embedding(args.config, args.split)
        write_json(args.out_dir / "ablation_embedding.json", {"split": args.split, "cells": cells})
    LOGGER.info("ablation finished in %.1f min", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
