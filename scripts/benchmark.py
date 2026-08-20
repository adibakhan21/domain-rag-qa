#!/usr/bin/env python
"""Phase 8: latency, throughput and memory measurement.

    python scripts/benchmark.py --config configs/rag.yaml

Measures each pipeline stage separately (retrieval / rerank / generation) plus
end-to-end, at batch size 1 (the interactive serving case) and in batches (the
offline scoring case).  Reports P50/P95/P99 rather than means only: mean latency
hides the tail that actually determines a timeout budget.

All numbers are measured on the machine that runs the script and are labelled
with its device; nothing is extrapolated.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402

from rag_system.data.covidqa import QAExample                             # noqa: E402
from rag_system.data.store import read_chunks, read_jsonl                 # noqa: E402
from rag_system.generation.pipeline import RAGPipeline                    # noqa: E402
from rag_system.generation.readers import build_reader                    # noqa: E402
from rag_system.reranking.cross_encoder import CrossEncoderReranker       # noqa: E402
from rag_system.retrieval.factory import build_retriever                  # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, RESULTS_DIR, load_config  # noqa: E402
from rag_system.utils.runtime import (get_logger, resolve_device, rss_memory_mb,  # noqa: E402
                                      set_seed, write_json)

LOGGER = get_logger()


def percentiles(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)

    def pct(p: float) -> float:
        # Nearest-rank percentile; exact and unambiguous for small samples.
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * len(ordered) + 0.5)) - 1))
        return ordered[idx]

    return {
        "n": len(ordered),
        "mean_ms": round(statistics.mean(ordered), 2),
        "std_ms": round(statistics.pstdev(ordered), 2) if len(ordered) > 1 else 0.0,
        "min_ms": round(ordered[0], 2),
        "p50_ms": round(pct(50), 2),
        "p95_ms": round(pct(95), 2),
        "p99_ms": round(pct(99), 2),
        "max_ms": round(ordered[-1], 2),
    }


def time_calls(fn: Callable[[int], Any], n: int, warmup: int = 3) -> List[float]:
    """Run ``fn`` n times after ``warmup`` untimed calls.

    Warm-up matters on MPS/CUDA: the first calls include lazy kernel
    compilation and would otherwise dominate the p99.
    """
    for i in range(warmup):
        fn(i)
    samples: List[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        samples.append(1000 * (time.perf_counter() - t0))
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "rag.yaml")
    p.add_argument("--corpus", type=Path, default=ARTIFACT_DIR / "corpus")
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmarks" / "benchmark_results.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.generation.device)

    rows = [r for r in read_jsonl(args.corpus / "examples.jsonl") if r["split"] == "test"]
    questions = [QAExample(**r).question for r in rows][: args.n_queries]
    chunks = read_chunks(args.corpus / "chunks.jsonl")
    LOGGER.info("benchmarking %d queries over %d chunks on %s", len(questions), len(chunks), device)

    mem_before = rss_memory_mb()
    bm25 = build_retriever(cfg, method="bm25", chunks=chunks)
    hybrid = build_retriever(cfg, method="hybrid")
    dense = hybrid.dense
    reranker = CrossEncoderReranker(cfg.reranker)
    reader = build_reader(cfg.generation)
    mem_after = rss_memory_mb()

    results: Dict[str, Any] = {
        "environment": {
            "device": device,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "n_chunks": len(chunks),
            "n_queries": len(questions),
        },
        # Current RSS at three points, not a high-water mark -- RSS can fall
        # again once intermediate tensors are freed, so "peak" would be a lie.
        "memory": {
            "rss_before_models_mb": round(mem_before, 1),
            "rss_after_models_mb": round(mem_after, 1),
            "models_delta_mb": round(mem_after - mem_before, 1),
        },
        "components": {},
        "end_to_end": {},
        "throughput": {},
    }

    # --- per-component, batch size 1 (interactive serving) ---------------
    k = cfg.retrieval.top_k
    depth = cfg.retrieval.candidate_k
    component_specs = {
        "retrieval_bm25": lambda i: bm25.search_batch([questions[i % len(questions)]], top_k=k),
        "retrieval_dense": lambda i: dense.search_batch([questions[i % len(questions)]], top_k=k),
        "retrieval_hybrid": lambda i: hybrid.search_batch([questions[i % len(questions)]], top_k=k),
    }
    for name, fn in component_specs.items():
        results["components"][name] = percentiles(time_calls(fn, len(questions)))
        LOGGER.info("  %-20s p50 %7.2f ms  p95 %7.2f ms", name,
                    results["components"][name]["p50_ms"], results["components"][name]["p95_ms"])

    # Reranking and reading are timed on pre-retrieved candidates so the
    # measurement isolates the stage rather than including retrieval again.
    candidates = hybrid.search_batch(questions, top_k=depth)
    rerank_fn = lambda i: reranker.rerank_batch(
        [questions[i % len(questions)]], [candidates[i % len(questions)]], top_k=k)
    results["components"]["rerank_cross_encoder"] = percentiles(time_calls(rerank_fn, len(questions)))
    LOGGER.info("  %-20s p50 %7.2f ms  p95 %7.2f ms", "rerank_cross_encoder",
                results["components"]["rerank_cross_encoder"]["p50_ms"],
                results["components"]["rerank_cross_encoder"]["p95_ms"])

    reranked = reranker.rerank_batch(questions, candidates, top_k=k)
    contexts = ["\n\n".join(c.text for c in hits)[: cfg.generation.max_context_chars] for hits in reranked]
    read_fn = lambda i: reader.answer_batch(
        [questions[i % len(questions)]], [contexts[i % len(questions)]])
    results["components"]["reader_extractive"] = percentiles(time_calls(read_fn, len(questions)))
    LOGGER.info("  %-20s p50 %7.2f ms  p95 %7.2f ms", "reader_extractive",
                results["components"]["reader_extractive"]["p50_ms"],
                results["components"]["reader_extractive"]["p95_ms"])

    # --- end-to-end -------------------------------------------------------
    for label, pipe in {
        "rag_bm25_no_rerank": RAGPipeline(bm25, reader, reranker=None, top_k=k,
                                          max_context_chars=cfg.generation.max_context_chars),
        "rag_hybrid_rerank": RAGPipeline(hybrid, reader, reranker=reranker, top_k=k,
                                         candidate_k=depth,
                                         max_context_chars=cfg.generation.max_context_chars),
    }.items():
        fn = lambda i, _p=pipe: _p.query(questions[i % len(questions)])
        results["end_to_end"][label] = percentiles(time_calls(fn, len(questions)))
        LOGGER.info("  %-20s p50 %7.2f ms  p95 %7.2f ms", label,
                    results["end_to_end"][label]["p50_ms"], results["end_to_end"][label]["p95_ms"])

    # --- throughput -------------------------------------------------------
    full = RAGPipeline(hybrid, reader, reranker=reranker, top_k=k, candidate_k=depth,
                       max_context_chars=cfg.generation.max_context_chars)
    for batch_size in args.batch_sizes:
        batch = (questions * ((batch_size // len(questions)) + 1))[:batch_size]
        full.query_batch(batch)                      # warm-up
        t0 = time.perf_counter()
        full.query_batch(batch)
        elapsed = time.perf_counter() - t0
        results["throughput"][f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "seconds": round(elapsed, 3),
            "queries_per_second": round(batch_size / elapsed, 2),
            "ms_per_query": round(1000 * elapsed / batch_size, 2),
        }
        LOGGER.info("  batch %-3d  %6.2f q/s  (%.1f ms/query)", batch_size,
                    results["throughput"][f"batch_{batch_size}"]["queries_per_second"],
                    results["throughput"][f"batch_{batch_size}"]["ms_per_query"])

    results["memory"]["rss_at_end_mb"] = round(rss_memory_mb(), 1)
    write_json(args.out, results)
    LOGGER.info("wrote %s", args.out)

    print("\n" + "=" * 84)
    print(f"{'stage':<26}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'mean ms':>10}")
    print("-" * 84)
    for section in ("components", "end_to_end"):
        for name, m in results[section].items():
            print(f"{name:<26}{m['p50_ms']:>10.2f}{m['p95_ms']:>10.2f}{m['p99_ms']:>10.2f}{m['mean_ms']:>10.2f}")
    print("-" * 84)
    for name, m in results["throughput"].items():
        print(f"{name:<26}{m['queries_per_second']:>10.2f} q/s  ({m['ms_per_query']:.1f} ms/query)")
    print(f"\ndevice={device}   RSS after models={results['memory']['rss_after_models_mb']} MB")
    print("=" * 84)


if __name__ == "__main__":
    main()
