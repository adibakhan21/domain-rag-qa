"""COVID-QA corpus construction with derived retrieval ground truth.

Why this dataset
----------------
COVID-QA (Möller et al., 2020) is 2,019 question/answer pairs annotated by
biomedical experts over 147 *full-text* CORD-19 articles.  Two properties make
it the right choice for a project that has to evaluate retrieval *and* reading
objectively:

1. Answers carry exact character offsets into the source article, so the gold
   passage for retrieval can be **derived** (the chunk containing the annotated
   span) instead of being assumed or hand-labelled.
2. Documents are long (median ~30k characters), so chunking, retrieval and
   context construction are real problems rather than formalities.

Splitting
---------
Splits are over **documents**, not questions.  A question-level split would put
other questions from the same article in the training set, so a fine-tuned
reader would be evaluated on documents it had already read -- inflating EM/F1
for reasons that have nothing to do with generalisation.  The retrieval corpus
still spans all 147 documents, which is both realistic and harder.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.chunking import Chunk, chunk_document
from ..preprocessing.cleaning import build_raw_to_clean_map, clean_text
from ..utils.config import ChunkingConfig, DataConfig
from ..utils.runtime import get_logger

LOGGER = get_logger()


@dataclass
class Document:
    doc_id: str
    title: str
    text: str                      # cleaned text
    raw_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    """One question with its gold answer, in cleaned-document coordinates."""

    qid: str
    question: str
    doc_id: str
    answer_text: str
    answer_start: int              # offset into the cleaned document
    answer_end: int
    split: str
    gold_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Corpus:
    """Documents, chunks, QA examples and the derived relevance judgements."""

    documents: Dict[str, Document]
    chunks: List[Chunk]
    examples: List[QAExample]
    chunking: ChunkingConfig

    @property
    def chunk_index(self) -> Dict[str, int]:
        return {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def examples_for(self, split: str) -> List[QAExample]:
        return [e for e in self.examples if e.split == split]

    def qrels(self, split: Optional[str] = None) -> Dict[str, List[str]]:
        """Relevance judgements: question id -> list of gold chunk ids."""
        return {
            e.qid: list(e.gold_chunk_ids)
            for e in self.examples
            if (split is None or e.split == split) and e.gold_chunk_ids
        }

    def stats(self) -> Dict[str, Any]:
        from ..preprocessing.chunking import chunk_stats

        per_split = {}
        for split in ("train", "validation", "test"):
            ex = self.examples_for(split)
            per_split[split] = {
                "questions": len(ex),
                "documents": len({e.doc_id for e in ex}),
                "with_gold_chunk": sum(1 for e in ex if e.gold_chunk_ids),
            }
        return {
            "n_documents": len(self.documents),
            "total_corpus_chars": sum(len(d.text) for d in self.documents.values()),
            "n_questions": len(self.examples),
            "chunks": chunk_stats(self.chunks),
            "splits": per_split,
        }


def _title_of(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 15:
            return line[:300]
    return text[:120].strip()


def split_documents(
    doc_ids: Sequence[str], cfg: DataConfig
) -> Dict[str, str]:
    """Assign each document to train/validation/test deterministically.

    The assignment is a hash of the document id rather than a shuffled index, so
    adding or removing a document does not reshuffle everything else -- splits
    stay stable across runs and across changes to the corpus.
    """
    total = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    assignment: Dict[str, str] = {}
    for doc_id in doc_ids:
        digest = hashlib.sha256(f"{cfg.seed}:{doc_id}".encode()).hexdigest()
        u = int(digest[:16], 16) / float(1 << 64)
        if u < cfg.train_frac:
            assignment[doc_id] = "train"
        elif u < cfg.train_frac + cfg.val_frac:
            assignment[doc_id] = "validation"
        else:
            assignment[doc_id] = "test"
    return assignment


def build_corpus(
    data_cfg: Optional[DataConfig] = None,
    chunk_cfg: Optional[ChunkingConfig] = None,
    cache_dir: Optional[Path] = None,
) -> Corpus:
    """Load COVID-QA, clean, chunk and derive retrieval ground truth."""
    from datasets import load_dataset

    data_cfg = data_cfg or DataConfig()
    chunk_cfg = chunk_cfg or ChunkingConfig()

    LOGGER.info("loading %s (split=%s)", data_cfg.dataset_name, data_cfg.split)
    ds = load_dataset(data_cfg.dataset_name, split=data_cfg.split, cache_dir=str(cache_dir) if cache_dir else None)

    # --- documents -------------------------------------------------------
    raw_by_doc: Dict[str, str] = {}
    for row in ds:
        doc_id = str(row["document_id"])
        raw_by_doc.setdefault(doc_id, row["context"])

    doc_ids = sorted(raw_by_doc)
    if data_cfg.max_documents:
        doc_ids = doc_ids[: data_cfg.max_documents]
    keep = set(doc_ids)

    split_map = split_documents(doc_ids, data_cfg)

    documents: Dict[str, Document] = {}
    inverse_maps: Dict[str, np.ndarray] = {}
    chunks: List[Chunk] = []
    chunks_by_doc: Dict[str, List[Chunk]] = {}

    for doc_id in doc_ids:
        raw = raw_by_doc[doc_id]
        cleaned = clean_text(raw)
        inverse_maps[doc_id] = build_raw_to_clean_map(cleaned, len(raw))
        documents[doc_id] = Document(
            doc_id=doc_id,
            title=_title_of(cleaned.text),
            text=cleaned.text,
            raw_length=len(raw),
            metadata={"split": split_map[doc_id], "source": "CORD-19 / COVID-QA"},
        )
        doc_chunks = chunk_document(
            cleaned.text,
            doc_id=doc_id,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            min_chunk_size=chunk_cfg.min_chunk_size,
            strategy=chunk_cfg.strategy,
            metadata={"title": documents[doc_id].title, "split": split_map[doc_id]},
        )
        chunks_by_doc[doc_id] = doc_chunks
        chunks.extend(doc_chunks)

    # --- questions + derived gold chunks ---------------------------------
    examples: List[QAExample] = []
    for row in ds:
        doc_id = str(row["document_id"])
        if doc_id not in keep:
            continue
        answers = row["answers"]
        if not answers["text"]:
            continue
        answer_text = answers["text"][0]
        raw_start = int(answers["answer_start"][0])
        raw_end = raw_start + len(answer_text)

        inv = inverse_maps[doc_id]
        start = int(inv[min(raw_start, len(inv) - 1)])
        end = int(inv[min(raw_end, len(inv) - 1)])
        cleaned_text = documents[doc_id].text
        start, end = _snap_span(cleaned_text, answer_text, start, end)

        gold = _gold_chunks(chunks_by_doc[doc_id], start, end)
        examples.append(
            QAExample(
                qid=str(row["id"]),
                question=row["question"].strip(),
                doc_id=doc_id,
                answer_text=cleaned_text[start:end],
                answer_start=start,
                answer_end=end,
                split=split_map[doc_id],
                gold_chunk_ids=[c.chunk_id for c in gold],
            )
        )

    corpus = Corpus(documents=documents, chunks=chunks, examples=examples, chunking=chunk_cfg)
    LOGGER.info(
        "corpus: %d docs, %d chunks, %d questions (%d with gold chunk)",
        len(documents), len(chunks), len(examples),
        sum(1 for e in examples if e.gold_chunk_ids),
    )
    return corpus


def _snap_span(text: str, answer: str, start: int, end: int, window: int = 60) -> Tuple[int, int]:
    """Correct small drift between the mapped span and the actual answer string.

    Whitespace collapsing can shift a span by a few characters.  Rather than
    trust the arithmetic, the exact answer string is searched for in a window
    around the mapped position; if it is not found the mapped span is kept and
    the caller can still detect the mismatch.
    """
    normalized = " ".join(answer.split())
    if text[start:end] == normalized:
        return start, end
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    local = text[lo:hi]
    idx = local.find(normalized)
    if idx >= 0:
        return lo + idx, lo + idx + len(normalized)
    return start, min(end, len(text))


def _gold_chunks(doc_chunks: Sequence[Chunk], start: int, end: int) -> List[Chunk]:
    """Chunks that count as relevant for this answer span.

    Preference order: chunks that contain the span *entirely* (an extractive
    reader can only succeed on those).  If the span straddles a boundary, fall
    back to every chunk it overlaps, so the question still has usable
    judgements rather than being silently dropped.
    """
    full = [c for c in doc_chunks if c.contains_span(start, end)]
    if full:
        return full
    return [c for c in doc_chunks if c.overlaps_span(start, end)]
