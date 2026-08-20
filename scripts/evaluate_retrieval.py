#!/usr/bin/env python
"""Phase 1 + 2: benchmark every retrieval configuration on the test split.

    python scripts/evaluate_retrieval.py --config configs/retrieval.yaml

Compares BM25, dense, hybrid (RRF) and hybrid+cross-encoder reranking on
identical questions with identical judgements, reports bootstrap confidence
intervals, and runs a paired bootstrap test of each system against BM25.

Writes results/retrieval/retrieval_results.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_system.data.covidqa import QAExample                              # noqa: E402
from rag_system.data.store import read_jsonl                               # noqa: E402
from rag_system.evaluation.retrieval_metrics import (evaluate_run,         # noqa: E402
                                                     paired_bootstrap_test)
from rag_system.evaluation.runner import run_retrieval                     # noqa: E402
from rag_system.reranking.cross_encoder import CrossEncoderReranker        # noqa: E402
from rag_system.retrieval.factory import build_retriever                   # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, RESULTS_DIR, load_config  # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, write_json      # noqa: E402
from rag_system.utils.tracking import start_run                            # noqa: E402

LOGGER = get_logger()

# (label, retrieval method, use reranker)
SYSTEMS = [
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid_rrf", "hybrid", False),
    # The reranked variants of each first stage. Without bm25_rerank and
    # dense_rerank there is no way to tell whether hybrid_rrf_rerank wins
    # because of the fusion or purely because of the cross-encoder.
    ("bm25_rerank", "bm25", True),
    ("dense_rerank", "dense", True),
    ("hybrid_rrf_rerank", "hybrid", True),
]


def load_examples(path: Path, split: str, limit=None) -> List[QAExample]:
    rows = [r for r in read_jsonl(path) if r["split"] == split]
    examples = [QAExample(**r) for r in rows]
    return examples[:limit] if limit else examples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.yaml")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--corpus", type=Path, default=ARTIFACT_DIR / "corpus")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "retrieval" / "retrieval_results.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--systems", nargs="+", default=[s[0] for s in SYSTEMS])
    p.add_argument("--no-tracking", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    limit = args.limit or cfg.evaluation.max_eval_questions

    examples = load_examples(args.corpus / "examples.jsonl", args.split, limit)
    qrels = {e.qid: e.gold_chunk_ids for e in examples if e.gold_chunk_ids}
    LOGGER.info("evaluating on %s split: %d questions, %d with gold chunks",
                args.split, len(examples), len(qrels))

    reranker = None
    if any(name in args.systems and use_rr for name, _, use_rr in SYSTEMS):
        reranker = CrossEncoderReranker(cfg.reranker)

    max_k = max(cfg.evaluation.recall_at_k)
    results: Dict[str, dict] = {}
    per_query_store: Dict[str, dict] = {}

    for label, method, use_reranker in SYSTEMS:
        if label not in args.systems:
            continue
        LOGGER.info("--- %s ---", label)
        retriever = build_retriever(cfg, method=method)

        run, _, timings = run_retrieval(
            retriever,
            examples,
            top_k=max_k,
            reranker=reranker if use_reranker else None,
            candidate_k=cfg.retrieval.candidate_k,
        )
        scored = evaluate_run(
            run, qrels,
            k_values=cfg.evaluation.recall_at_k,
            mrr_k=cfg.evaluation.mrr_at_k,
            ndcg_k=cfg.evaluation.ndcg_at_k,
            n_bootstrap=cfg.evaluation.n_bootstrap,
            seed=cfg.experiment.seed,
        )
        summary = scored["summary"]
        per_query_store[label] = scored["per_query"]

        results[label] = {
            "method": method,
            "reranker": cfg.reranker.model_name if use_reranker else None,
            "candidate_k": cfg.retrieval.candidate_k if use_reranker else max_k,
            "metrics": summary,
            "timings": timings,
        }
        LOGGER.info(
            "  hit@1 %.3f | hit@5 %.3f | hit@10 %.3f | recall@10 %.3f | mrr@10 %.3f | ndcg@10 %.3f | %.1f ms/query",
            summary["hit@1"], summary["hit@5"], summary["hit@10"],
            summary["recall@10"], summary["mrr@10"], summary["ndcg@10"],
            timings["ms_per_query"],
        )

        with start_run(f"retrieval-{label}", cfg.experiment.tracking_uri,
                       cfg.experiment.experiment_name, enabled=not args.no_tracking) as run_logger:
            run_logger.set_tags({"phase": "retrieval", "system": label, "split": args.split})
            params = cfg.flat_params()
            params["retrieval.method"] = method
            params["reranker.enabled"] = use_reranker
            run_logger.log_params(params)
            run_logger.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            run_logger.log_metrics({"ms_per_query": timings["ms_per_query"]})

    # --- significance testing --------------------------------------------
    # Every system is tested against the BM25 baseline, plus two targeted
    # head-to-heads that decide the design: does fusing a weak dense ranker help
    # once a cross-encoder is already reranking, and is reranking itself the
    # component doing the work?
    sig_metrics = ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")
    comparisons = {}
    if "bm25" in per_query_store:
        for label in per_query_store:
            if label == "bm25":
                continue
            comparisons[f"{label}_vs_bm25"] = {
                metric: paired_bootstrap_test(per_query_store["bm25"], per_query_store[label], metric)
                for metric in sig_metrics
            }
    for baseline, candidate in (("bm25_rerank", "hybrid_rrf_rerank"),
                                ("hybrid_rrf", "hybrid_rrf_rerank")):
        if baseline in per_query_store and candidate in per_query_store:
            comparisons[f"{candidate}_vs_{baseline}"] = {
                metric: paired_bootstrap_test(per_query_store[baseline], per_query_store[candidate], metric)
                for metric in sig_metrics
            }

    payload = {
        "split": args.split,
        "n_questions": len(examples),
        "n_with_gold": len(qrels),
        "corpus": {"n_chunks": len(build_retriever(cfg, method="bm25"))},
        "config": cfg.to_dict(),
        "systems": results,
        "significance_vs_bm25": comparisons,
    }
    write_json(args.out, payload)
    write_json(args.out.parent / f"per_query_{args.split}.json", per_query_store)
    LOGGER.info("wrote %s", args.out)

    print("\n" + "=" * 100)
    print(f"{'system':<22}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'recall@10':>11}{'mrr@10':>9}{'ndcg@10':>9}{'ms/query':>10}")
    print("-" * 100)
    for label, r in results.items():
        m = r["metrics"]
        print(f"{label:<22}{m['hit@1']:>8.3f}{m['hit@5']:>8.3f}{m['hit@10']:>8.3f}"
              f"{m['recall@10']:>11.3f}{m['mrr@10']:>9.3f}{m['ndcg@10']:>9.3f}{r['timings']['ms_per_query']:>10.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
