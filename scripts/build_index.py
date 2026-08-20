#!/usr/bin/env python
"""Embed the chunk corpus and build the FAISS index.

    python scripts/build_index.py --config configs/retrieval.yaml

The index directory name encodes the embedding model and chunking parameters, so
several configurations coexist and the ablation can switch between them without
rebuilding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_system.data.store import index_name, read_chunks, save_index   # noqa: E402
from rag_system.retrieval.dense import EmbeddingModel, FaissIndex        # noqa: E402
from rag_system.utils.config import ARTIFACT_DIR, load_config            # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, timer         # noqa: E402

LOGGER = get_logger()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.yaml")
    p.add_argument("--chunks", type=Path, default=ARTIFACT_DIR / "corpus" / "chunks.jsonl")
    p.add_argument("--embedding-model", default=None, help="override embedding.model_name")
    args = p.parse_args()

    cfg = load_config(args.config if args.config.exists() else None)
    if args.embedding_model:
        cfg.embedding.model_name = args.embedding_model
        # Only BGE-family models were trained with a query instruction.
        if "bge" not in args.embedding_model.lower():
            cfg.embedding.query_prefix = ""
    set_seed(cfg.experiment.seed)

    if not args.chunks.exists():
        raise SystemExit(f"{args.chunks} not found -- run scripts/prepare_data.py first")
    chunks = read_chunks(args.chunks)
    LOGGER.info("loaded %d chunks", len(chunks))

    embedder = EmbeddingModel(cfg.embedding)
    with timer("embedding", LOGGER) as t:
        vectors = embedder.encode([c.text for c in chunks], is_query=False, show_progress=True)

    index = FaissIndex(vectors.shape[1])
    index.add(vectors)

    name = index_name(cfg.chunking, cfg.embedding)
    path = save_index(name, chunks, vectors, index, cfg.chunking, cfg.embedding)
    LOGGER.info("embedded %d chunks in %.1fs (%.1f chunks/s) -> %s",
                len(chunks), t["seconds"], len(chunks) / t["seconds"], path)


if __name__ == "__main__":
    main()
