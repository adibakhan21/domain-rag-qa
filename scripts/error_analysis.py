#!/usr/bin/env python
"""Phase 6: categorise RAG failures and save representative examples.

    python scripts/error_analysis.py --system rag_hybrid_rerank

Consumes results/rag/per_question_test.json (written by evaluate_rag.py) and
writes results/error_analysis/error_analysis.json plus a readable Markdown
report.  Also recomputes the first-stage candidate list so that
"the reranker demoted the gold chunk" can be separated from "retrieval never
found it" -- a distinction that changes which component you would fix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402

from rag_system.data.covidqa import QAExample                              # noqa: E402
from rag_system.data.store import read_chunks, read_jsonl                  # noqa: E402
from rag_system.evaluation.error_analysis import (analyse,                 # noqa: E402
                                                  breakdown_by_answer_length,
                                                  breakdown_by_question_type)
from rag_system.retrieval.factory import build_retriever                   # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, RESULTS_DIR, load_config  # noqa: E402
from rag_system.utils.runtime import get_logger, read_json, set_seed, write_json  # noqa: E402

LOGGER = get_logger()


def markdown_report(system: str, report: Dict, by_type: Dict, by_length: Dict,
                    n_examples: int = 3) -> str:
    lines: List[str] = [
        f"# Error analysis — `{system}`", "",
        f"{report['n_correct']}/{report['n_questions']} questions answered with F1 = 1.0 "
        f"({100 * report['n_correct'] / report['n_questions']:.1f}%); "
        f"{report['n_failures']} failures categorised below.", "",
        "## Failure distribution", "",
        "| category | count | share of failures |", "|---|---:|---:|",
    ]
    for category, count in report["counts"].items():
        if category == "correct" or count == 0:
            continue
        lines.append(f"| `{category}` | {count} | {report['failure_share'][category]:.1f}% |")

    lines += ["", "## Mean F1 by question type", "", "| type | n | mean F1 |", "|---|---:|---:|"]
    for qtype, stats in by_type.items():
        lines.append(f"| {qtype} | {stats['n']} | {stats['mean_f1']:.1f} |")

    lines += ["", "## Mean F1 by gold answer length", "", "| length | n | mean F1 |", "|---|---:|---:|"]
    for label, stats in by_length.items():
        lines.append(f"| {label} | {stats['n']} | {stats['mean_f1']:.1f} |")

    lines += ["", "## Representative failures", ""]
    for category, examples in report["examples"].items():
        if not examples:
            continue
        lines += [f"### `{category}`", ""]
        for ex in examples[:n_examples]:
            lines += [
                f"**Q:** {ex['question']}", "",
                f"- **gold:** {ex['gold'][:300]}",
                f"- **predicted:** {ex['prediction'][:300] or '_(empty)_'}",
                f"- F1 {ex['f1']:.2f} · answer_in_context {ex['answer_in_context']:.0f} "
                f"· groundedness {ex['groundedness']:.2f}",
                f"- gold chunks `{ex['gold_chunk_ids']}` · retrieved `{ex['retrieved_ids']}`", "",
            ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "rag.yaml")
    p.add_argument("--per-question", type=Path,
                   default=RESULTS_DIR / "rag" / "per_question_test.json")
    p.add_argument("--corpus", type=Path, default=ARTIFACT_DIR / "corpus")
    p.add_argument("--system", default="rag_hybrid_rerank")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "error_analysis")
    p.add_argument("--rerank", type=lambda v: v.lower() in {"1", "true", "yes"}, default=None,
                   help="override whether this system used a reranker (default: read from config)")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    if not args.per_question.exists():
        raise SystemExit(f"{args.per_question} not found -- run scripts/evaluate_rag.py first")
    all_records = read_json(args.per_question)
    if args.system not in all_records:
        raise SystemExit(f"system {args.system!r} not in {sorted(all_records)}")
    records = all_records[args.system]

    # Questions whose answer spans a chunk boundary (no single chunk contains
    # it) are a chunking failure, not a retrieval one.
    chunks = read_chunks(args.corpus / "chunks.jsonl")
    by_id = {c.chunk_id: c for c in chunks}
    examples = {r["qid"]: QAExample(**r) for r in read_jsonl(args.corpus / "examples.jsonl")}
    boundary_qids = [
        qid for qid, ex in examples.items()
        if ex.gold_chunk_ids
        and not any(ex.answer_text in by_id[g].text for g in ex.gold_chunk_ids if g in by_id)
    ]

    # First-stage candidates, to detect reranker demotion.
    #
    # Whether a system reranks is read from the config, not guessed from its
    # name: "rag_hybrid_lora" reranks but contains no "rerank" substring, and
    # name-sniffing silently made reranker_demotion unreachable for it.
    first_stage: Dict[str, List[str]] = {}
    uses_reranker = cfg.reranker.enabled and args.system not in {"closed_book", "oracle_context",
                                                                 "oracle_context_lora", "rag_bm25"}
    if args.rerank is not None:
        uses_reranker = args.rerank
    if uses_reranker:
        LOGGER.info("recomputing first-stage candidates to detect reranker demotion")
        method = "bm25" if "bm25" in args.system else "hybrid"
        retriever = build_retriever(cfg, method=method, chunks=chunks if method == "bm25" else None)
        questions = [r["question"] for r in records]
        hits = retriever.search_batch(questions, top_k=cfg.retrieval.candidate_k)
        first_stage = {r["qid"]: [h.chunk_id for h in hit] for r, hit in zip(records, hits)}

    report = analyse(records, first_stage=first_stage, multi_chunk_gold_qids=boundary_qids)
    by_type = breakdown_by_question_type(records)
    by_length = breakdown_by_answer_length(records)

    write_json(args.out_dir / "error_analysis.json",
               {"system": args.system, "report": report,
                "by_question_type": by_type, "by_answer_length": by_length})
    md = markdown_report(args.system, report, by_type, by_length)
    (args.out_dir / "error_analysis.md").write_text(md)
    LOGGER.info("wrote %s", args.out_dir / "error_analysis.md")

    print("\n" + "=" * 72)
    print(f"error analysis — {args.system}")
    print("-" * 72)
    print(f"correct (F1=1.0): {report['n_correct']}/{report['n_questions']}")
    for category, count in report["counts"].items():
        if category == "correct" or count == 0:
            continue
        print(f"  {category:<24}{count:>5}   {report['failure_share'][category]:>5.1f}% of failures")
    print("=" * 72)


if __name__ == "__main__":
    main()
