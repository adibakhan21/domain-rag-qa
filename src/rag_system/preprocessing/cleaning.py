"""Text normalisation for CORD-19 full-text articles.

The raw contexts are scientific papers pasted as plain text: hard-wrapped
lines, running headers, figure captions and long runs of whitespace.  Cleaning
has to be *offset-preserving-by-construction* or the gold answer spans (which
are character offsets into the raw text) stop pointing at the right characters.

The approach here is deliberately conservative: only whitespace is collapsed,
and an explicit offset map is returned so a span in cleaned coordinates can
always be mapped back to the raw document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_WS_RUN = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\n{3,}")


@dataclass
class CleanedText:
    """Cleaned text plus the map back to raw character offsets."""

    text: str
    # offset_map[i] = index in the raw string of cleaned character i
    offset_map: np.ndarray

    def to_raw(self, start: int, end: int) -> Tuple[int, int]:
        """Map a [start, end) span in cleaned coordinates back to raw coordinates."""
        if len(self.offset_map) == 0:
            return (0, 0)
        start = max(0, min(start, len(self.offset_map) - 1))
        end = max(start + 1, min(end, len(self.offset_map)))
        return int(self.offset_map[start]), int(self.offset_map[end - 1]) + 1


def clean_text(raw: str) -> CleanedText:
    """Collapse whitespace runs while tracking where every kept character came from.

    Only whitespace is touched.  Anything that removed or rewrote content
    (de-hyphenation, header stripping) would either break the offset map or
    require heuristics that can silently corrupt an answer span, so it is left
    to the chunker to skip degenerate chunks instead.
    """
    out_chars: List[str] = []
    out_offsets: List[int] = []

    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch in " \t\r\f\v":
            j = i
            while j < n and raw[j] in " \t\r\f\v":
                j += 1
            # A whitespace run adjacent to a newline is absorbed by the newline.
            if not (out_chars and out_chars[-1] == "\n") and (j < n and raw[j] != "\n"):
                out_chars.append(" ")
                out_offsets.append(i)
            i = j
            continue
        if ch == "\n":
            j = i
            while j < n and raw[j] in " \t\r\f\v\n":
                j += 1
            newline_count = raw[i:j].count("\n")
            if out_chars:
                out_chars.append("\n" if newline_count == 1 else "\n\n")
                out_offsets.append(i)
                if newline_count > 1:
                    out_offsets.append(i)
            i = j
            continue
        out_chars.append(ch)
        out_offsets.append(i)
        i += 1

    text = "".join(out_chars)
    # A "\n\n" append pushes two characters but we recorded two offsets for it.
    offsets = np.array(out_offsets[: len(text)], dtype=np.int64)
    if len(offsets) < len(text):  # pragma: no cover - defensive
        pad = np.full(len(text) - len(offsets), offsets[-1] if len(offsets) else 0, dtype=np.int64)
        offsets = np.concatenate([offsets, pad])
    return CleanedText(text=text, offset_map=offsets)


def build_raw_to_clean_map(cleaned: CleanedText, raw_length: int) -> np.ndarray:
    """Inverse of ``offset_map``: raw index -> nearest cleaned index.

    Needed to project the dataset's raw-coordinate answer spans into cleaned
    coordinates before chunking.
    """
    inverse = np.zeros(raw_length + 1, dtype=np.int64)
    if len(cleaned.offset_map) == 0:
        return inverse
    prev = 0
    for clean_idx, raw_idx in enumerate(cleaned.offset_map):
        inverse[prev : raw_idx + 1] = clean_idx
        prev = raw_idx + 1
    inverse[prev:] = len(cleaned.offset_map)
    return inverse


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\[])|\n\n+")


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """Split into sentence spans, returned as [start, end) character offsets.

    A regex splitter is used rather than a model: it is deterministic, has no
    download cost, and chunk boundaries only need to be *reasonable*, not
    linguistically perfect.  Abbreviations such as "et al." occasionally cause
    an early split, which is harmless because chunks pack many sentences.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]
