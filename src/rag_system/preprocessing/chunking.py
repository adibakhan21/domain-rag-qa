"""Document chunking with exact character-offset bookkeeping.

Chunk boundaries are the single most consequential preprocessing choice in this
project: they determine (a) what the retriever can possibly return, (b) whether
an answer span survives intact inside one chunk, and (c) how much irrelevant
text the reader has to wade through.  Every chunk therefore carries the exact
``[start_char, end_char)`` span it came from, which lets the pipeline derive
retrieval ground truth from the dataset's answer offsets instead of guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .cleaning import split_sentences


@dataclass
class Chunk:
    """One retrievable passage."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int          # offset into the cleaned document
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def contains_span(self, start: int, end: int) -> bool:
        """True if [start, end) lies entirely inside this chunk."""
        return self.start_char <= start and end <= self.end_char

    def overlaps_span(self, start: int, end: int) -> bool:
        """True if [start, end) intersects this chunk at all."""
        return start < self.end_char and end > self.start_char


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100,
    strategy: str = "sentence",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Split one cleaned document into overlapping chunks.

    ``sentence`` strategy packs whole sentences up to ``chunk_size`` characters
    and then backs up by ``chunk_overlap`` characters' worth of trailing
    sentences.  Respecting sentence boundaries matters here for two reasons:
    citations shown to a user should be readable, and an answer span cut in half
    by a boundary is unrecoverable for an extractive reader.

    ``fixed`` is a plain character-window baseline, kept so the ablation can
    quantify what sentence-awareness actually buys.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
    if not text.strip():
        return []

    metadata = dict(metadata or {})
    if strategy == "fixed":
        spans = _fixed_spans(len(text), chunk_size, chunk_overlap)
    elif strategy == "sentence":
        spans = _sentence_spans(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"unknown chunking strategy {strategy!r}; use 'sentence' or 'fixed'")

    chunks: List[Chunk] = []
    for start, end in spans:
        body = text[start:end]
        # Trim surrounding whitespace but keep offsets honest.
        lead = len(body) - len(body.lstrip())
        trail = len(body) - len(body.rstrip())
        start, end = start + lead, end - trail
        body = text[start:end]
        if len(body) < min_chunk_size and chunks:
            continue  # too small to be a useful retrieval unit
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{len(chunks)}",
                doc_id=doc_id,
                text=body,
                start_char=start,
                end_char=end,
                chunk_index=len(chunks),
                metadata=metadata,
            )
        )
    return chunks


def _fixed_spans(length: int, chunk_size: int, overlap: int) -> List[tuple]:
    step = chunk_size - overlap
    spans, start = [], 0
    while start < length:
        spans.append((start, min(start + chunk_size, length)))
        if start + chunk_size >= length:
            break
        start += step
    return spans


def _sentence_spans(text: str, chunk_size: int, overlap: int) -> List[tuple]:
    sentences = split_sentences(text)
    if not sentences:
        return _fixed_spans(len(text), chunk_size, overlap)

    spans: List[tuple] = []
    current: List[tuple] = []
    current_len = 0

    for sent in sentences:
        sent_len = sent[1] - sent[0]
        # A single sentence longer than the budget is hard-split, otherwise it
        # would blow past chunk_size and silently break the embedding window.
        if sent_len > chunk_size:
            if current:
                spans.append((current[0][0], current[-1][1]))
                current, current_len = [], 0
            for s, e in _fixed_spans(sent_len, chunk_size, overlap):
                spans.append((sent[0] + s, sent[0] + e))
            continue

        if current_len + sent_len > chunk_size and current:
            spans.append((current[0][0], current[-1][1]))
            # Carry back trailing sentences worth ~overlap characters.
            carry, carried = [], 0
            for prev in reversed(current):
                prev_len = prev[1] - prev[0]
                if carried + prev_len > overlap:
                    break
                carry.insert(0, prev)
                carried += prev_len
            current, current_len = carry, carried

        current.append(sent)
        current_len += sent_len

    if current:
        spans.append((current[0][0], current[-1][1]))
    return spans


def chunk_stats(chunks: Sequence[Chunk]) -> Dict[str, float]:
    """Descriptive statistics used in the README and the chunking ablation."""
    if not chunks:
        return {"n_chunks": 0}
    lengths = [len(c.text) for c in chunks]
    import statistics

    return {
        "n_chunks": len(chunks),
        "n_documents": len({c.doc_id for c in chunks}),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": int(statistics.median(lengths)),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "total_chars": sum(lengths),
    }
