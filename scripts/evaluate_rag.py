#!/usr/bin/env python
"""Phase 5: end-to-end RAG evaluation against controlled baselines.

    python scripts/evaluate_rag.py --config configs/rag.yaml

Systems compared on identical test questions:

  closed_book        generative reader, no retrieval          -- the floor
  rag_bm25           extractive reader + BM25                 -- basic retrieval
  rag_hybrid_rerank  extractive reader + hybrid + reranking   -- best retrieval
  rag_hybrid_lora    as above, with the LoRA fine-tuned reader
  oracle_context     extractive reader given the gold chunk   -- the ceiling
  oracle_context_lora  LoRA reader given the gold chunk        -- ceiling, adapted

The two controls are the point.  ``closed_book`` shows what the reader knows
without evidence; ``oracle_context`` shows what it could do with perfect
retrieval.  Any RAG score between them decomposes into retrieval error
(ceiling minus score) and reader error (100 minus ceiling), which is what makes
the error analysis actionable rather than anecdotal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402

from rag_system.data.covidqa import QAExample                              # noqa: E402
from rag_system.data.store import read_chunks, read_jsonl                  # noqa: E402
from rag_system.evaluation.qa_metrics import (aggregate_qa, answer_in_context,  # noqa: E402
                                              context_relevance, exact_match,
                                              groundedness, token_f1)
from rag_system.generation.pipeline import RAGPipeline, build_context      # noqa: E402
from rag_system.generation.readers import (ExtractiveReader, GenerativeReader,  # noqa: E402
                                           build_reader)
from rag_system.reranking.cross_encoder import CrossEncoderReranker        # noqa: E402
from rag_system.retrieval.base import RetrievalResult                      # noqa: E402
from rag_system.retrieval.factory import build_retriever                   # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, RESULTS_DIR, load_config  # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, write_json      # noqa: E402
from rag_system.utils.tracking import start_run                            # noqa: E402

LOGGER = get_logger()


def score_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    agg = aggregate_qa(records)
    return {
        "exact_match": 100 * agg.get("em", 0.0),
        "f1": 100 * agg.get("f1", 0.0),
        "answer_in_context": 100 * agg.get("answer_in_context", 0.0),
        "groundedness_proxy": 100 * agg.get("groundedness", 0.0),
        "context_relevance": 100 * agg.get("context_relevance", 0.0),
        "gold_chunk_retrieved": 100 * agg.get("gold_retrieved", 0.0),
        "latency_ms_mean": agg.get("latency_ms", float("nan")),
        "n": agg.get("n", 0),
    }


def evaluate_pipeline(pipe: RAGPipeline, examples: Sequence[QAExample],
                      batch_size: int = 16) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(0, len(examples), batch_size):
        batch = list(examples[i : i + batch_size])
        responses = pipe.query_batch([e.question for e in batch])
        for ex, resp in zip(batch, responses):
            context = build_context(resp.contexts, pipe.max_context_chars)
            retrieved_ids = [c.chunk_id for c in resp.contexts]
            records.append({
                "qid": ex.qid,
                "question": ex.question,
                "gold": ex.answer_text,
                "prediction": resp.answer,
                "em": exact_match(resp.answer, ex.answer_text),
                "f1": token_f1(resp.answer, ex.answer_text),
                "answer_in_context": answer_in_context(ex.answer_text, context),
                "groundedness": groundedness(resp.answer, context),
                "context_relevance": context_relevance(ex.gold_chunk_ids, retrieved_ids),
                "gold_retrieved": float(bool(set(ex.gold_chunk_ids) & set(retrieved_ids))),
                "latency_ms": resp.latency_ms["total"],
                "retrieved_ids": retrieved_ids,
                "gold_chunk_ids": list(ex.gold_chunk_ids),
                "context_chars": len(context),
            })
    return records


def evaluate_closed_book(reader: GenerativeReader, examples: Sequence[QAExample],
                         batch_size: int = 16) -> List[Dict[str, Any]]:
    """No retrieval at all: what the reader knows unaided."""
    import time

    records: List[Dict[str, Any]] = []
    for i in range(0, len(examples), batch_size):
        batch = list(examples[i : i + batch_size])
        t0 = time.perf_counter()
        answers = reader.answer_batch([e.question for e in batch], [""] * len(batch))
        elapsed = 1000 * (time.perf_counter() - t0) / len(batch)
        for ex, ans in zip(batch, answers):
            records.append({
                "qid": ex.qid, "question": ex.question, "gold": ex.answer_text,
                "prediction": ans.text,
                "em": exact_match(ans.text, ex.answer_text),
                "f1": token_f1(ans.text, ex.answer_text),
                "answer_in_context": 0.0,     # there is no context
                "groundedness": 0.0,
                "context_relevance": 0.0,
                "gold_retrieved": 0.0,
                "latency_ms": elapsed,
                "retrieved_ids": [], "gold_chunk_ids": list(ex.gold_chunk_ids), "context_chars": 0,
            })
    return records


def evaluate_oracle(reader: ExtractiveReader, examples: Sequence[QAExample],
                    chunks, batch_size: int = 16) -> List[Dict[str, Any]]:
    """Upper bound: the reader is handed the gold chunk directly."""
    import time

    by_id = {c.chunk_id: c for c in chunks}
    usable = [e for e in examples if e.gold_chunk_ids and e.gold_chunk_ids[0] in by_id]
    records: List[Dict[str, Any]] = []
    for i in range(0, len(usable), batch_size):
        batch = usable[i : i + batch_size]
        contexts = [by_id[e.gold_chunk_ids[0]].text for e in batch]
        t0 = time.perf_counter()
        answers = reader.answer_batch([e.question for e in batch], contexts)
        elapsed = 1000 * (time.perf_counter() - t0) / len(batch)
        for ex, ctx, ans in zip(batch, contexts, answers):
            records.append({
                "qid": ex.qid, "question": ex.question, "gold": ex.answer_text,
                "prediction": ans.text,
                "em": exact_match(ans.text, ex.answer_text),
                "f1": token_f1(ans.text, ex.answer_text),
                "answer_in_context": answer_in_context(ex.answer_text, ctx),
                "groundedness": groundedness(ans.text, ctx),
                "context_relevance": 1.0,
                "gold_retrieved": 1.0,
                "latency_ms": elapsed,
                "retrieved_ids": [ex.gold_chunk_ids[0]],
                "gold_chunk_ids": list(ex.gold_chunk_ids),
                "context_chars": len(ctx),
            })
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "rag.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--corpus", type=Path, default=ARTIFACT_DIR / "corpus")
    p.add_argument("--adapter", type=Path, default=Path("artifacts/finetuned/roberta-covidqa-lora"))
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "rag" / "rag_results.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--systems", nargs="+", default=None)
    p.add_argument("--no-tracking", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    rows = [r for r in read_jsonl(args.corpus / "examples.jsonl") if r["split"] == args.split]
    examples = [QAExample(**r) for r in rows][: args.limit] if args.limit else [QAExample(**r) for r in rows]
    chunks = read_chunks(args.corpus / "chunks.jsonl")
    LOGGER.info("evaluating %d %s questions", len(examples), args.split)

    all_systems = ["closed_book", "rag_bm25", "rag_hybrid_rerank", "rag_hybrid_lora",
                   "oracle_context", "oracle_context_lora"]
    wanted = args.systems or all_systems
    results: Dict[str, Any] = {}
    per_system_records: Dict[str, List[Dict[str, Any]]] = {}

    reranker = CrossEncoderReranker(cfg.reranker)
    extractive = build_reader(cfg.generation)

    if "closed_book" in wanted:
        LOGGER.info("--- closed_book (no retrieval) ---")
        reader = GenerativeReader(cfg.generation, closed_book=True)
        per_system_records["closed_book"] = evaluate_closed_book(reader, examples)
        del reader

    if "rag_bm25" in wanted:
        LOGGER.info("--- rag_bm25 ---")
        pipe = RAGPipeline(build_retriever(cfg, method="bm25", chunks=chunks), extractive,
                           reranker=None, top_k=cfg.retrieval.top_k,
                           max_context_chars=cfg.generation.max_context_chars)
        per_system_records["rag_bm25"] = evaluate_pipeline(pipe, examples)

    if "rag_hybrid_rerank" in wanted:
        LOGGER.info("--- rag_hybrid_rerank ---")
        pipe = RAGPipeline(build_retriever(cfg, method="hybrid"), extractive, reranker=reranker,
                           top_k=cfg.retrieval.top_k, candidate_k=cfg.retrieval.candidate_k,
                           max_context_chars=cfg.generation.max_context_chars)
        per_system_records["rag_hybrid_rerank"] = evaluate_pipeline(pipe, examples)

    lora_reader = None
    needs_lora = {"rag_hybrid_lora", "oracle_context_lora"} & set(wanted)
    if needs_lora:
        if args.adapter.exists():
            lora_cfg = load_config(args.config).generation
            lora_cfg.adapter_path = str(args.adapter)
            lora_reader = ExtractiveReader(lora_cfg)
        else:
            LOGGER.warning("adapter %s not found -- skipping %s (run scripts/train.py first)",
                           args.adapter, sorted(needs_lora))

    if "rag_hybrid_lora" in wanted and lora_reader is not None:
        LOGGER.info("--- rag_hybrid_lora ---")
        pipe = RAGPipeline(build_retriever(cfg, method="hybrid"), lora_reader, reranker=reranker,
                           top_k=cfg.retrieval.top_k, candidate_k=cfg.retrieval.candidate_k,
                           max_context_chars=cfg.generation.max_context_chars)
        per_system_records["rag_hybrid_lora"] = evaluate_pipeline(pipe, examples)

    if "oracle_context" in wanted:
        LOGGER.info("--- oracle_context (gold chunk, base reader) ---")
        per_system_records["oracle_context"] = evaluate_oracle(extractive, examples, chunks)

    if "oracle_context_lora" in wanted and lora_reader is not None:
        LOGGER.info("--- oracle_context_lora (gold chunk, LoRA reader) ---")
        per_system_records["oracle_context_lora"] = evaluate_oracle(lora_reader, examples, chunks)

    for label, records in per_system_records.items():
        results[label] = score_records(records)
        m = results[label]
        LOGGER.info("  %-20s EM %5.2f  F1 %5.2f  ans-in-ctx %5.1f  gold-retrieved %5.1f  %.0f ms",
                    label, m["exact_match"], m["f1"], m["answer_in_context"],
                    m["gold_chunk_retrieved"], m["latency_ms_mean"])
        with start_run(f"rag-{label}", cfg.experiment.tracking_uri, cfg.experiment.experiment_name,
                       enabled=not args.no_tracking) as run:
            run.set_tags({"phase": "rag", "system": label, "split": args.split})
            run.log_params(cfg.flat_params())
            run.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})

    # Merge with previously saved systems: re-running a subset must not discard
    # results from systems that were not re-run.
    per_question_path = args.out.parent / f"per_question_{args.split}.json"
    if args.systems and args.out.exists():
        from rag_system.utils.runtime import read_json as _read_json

        previous = _read_json(args.out).get("systems", {})
        results = {**previous, **results}
        if per_question_path.exists():
            per_system_records = {**_read_json(per_question_path), **per_system_records}
        LOGGER.info("merged with existing results; systems now: %s", sorted(results))

    write_json(args.out, {"split": args.split, "n_questions": len(examples),
                          "config": cfg.to_dict(), "systems": results})
    write_json(per_question_path, per_system_records)
    LOGGER.info("wrote %s", args.out)

    print("\n" + "=" * 104)
    print(f"{'system':<22}{'EM':>8}{'F1':>8}{'ans in ctx':>13}{'gold retr.':>12}"
          f"{'ctx prec.':>11}{'grounded':>10}{'ms/q':>10}")
    print("-" * 104)
    for label in all_systems:
        if label not in results:
            continue
        m = results[label]
        print(f"{label:<22}{m['exact_match']:>8.2f}{m['f1']:>8.2f}{m['answer_in_context']:>13.1f}"
              f"{m['gold_chunk_retrieved']:>12.1f}{m['context_relevance']:>11.1f}"
              f"{m['groundedness_proxy']:>10.1f}{m['latency_ms_mean']:>10.0f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
