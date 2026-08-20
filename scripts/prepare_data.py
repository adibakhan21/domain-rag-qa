#!/usr/bin/env python
"""Download COVID-QA, clean, chunk and derive retrieval ground truth.

    python scripts/prepare_data.py --config configs/retrieval.yaml

Writes to artifacts/corpus/: chunks.jsonl, examples.jsonl, documents.jsonl,
stats.json.  Idempotent -- rerunning with the same config reproduces byte-identical
output because document splits are hash-based, not shuffle-based.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_system.data.covidqa import build_corpus                    # noqa: E402
from rag_system.data.store import write_jsonl, write_chunks         # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, load_config       # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, write_json  # noqa: E402

LOGGER = get_logger()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.yaml")
    p.add_argument("--out", type=Path, default=ARTIFACT_DIR / "corpus")
    p.add_argument("--max-documents", type=int, default=None, help="smoke-test subset")
    args = p.parse_args()

    cfg = load_config(args.config if args.config.exists() else None)
    if args.max_documents:
        cfg.data.max_documents = args.max_documents
    set_seed(cfg.experiment.seed)

    corpus = build_corpus(cfg.data, cfg.chunking)
    stats = corpus.stats()

    out = Path(args.out)
    write_chunks(out / "chunks.jsonl", corpus.chunks)
    write_jsonl(out / "examples.jsonl", [e.to_dict() for e in corpus.examples])
    write_jsonl(out / "documents.jsonl", [
        {"doc_id": d.doc_id, "title": d.title, "n_chars": len(d.text),
         "raw_length": d.raw_length, **d.metadata}
        for d in corpus.documents.values()
    ])

    # Integrity check reported as data, not asserted away: how often does the
    # derived gold chunk actually contain the annotated answer string?
    by_id = {c.chunk_id: c for c in corpus.chunks}
    verbatim = sum(
        1 for e in corpus.examples
        if any(e.answer_text in by_id[g].text for g in e.gold_chunk_ids)
    )
    stats["gold_chunk_contains_answer_verbatim"] = verbatim
    stats["gold_chunk_contains_answer_verbatim_frac"] = round(verbatim / len(corpus.examples), 4)

    write_json(out / "stats.json", stats)
    cfg.save(out / "config.yaml")

    LOGGER.info("wrote corpus artefacts to %s", out)
    LOGGER.info("  %d documents, %d chunks, %d questions", stats["n_documents"],
                stats["chunks"]["n_chunks"], stats["n_questions"])
    LOGGER.info("  gold chunk contains answer verbatim: %d/%d (%.2f%%)",
                verbatim, stats["n_questions"], 100 * verbatim / stats["n_questions"])


if __name__ == "__main__":
    main()
