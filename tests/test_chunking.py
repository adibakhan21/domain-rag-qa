"""Chunking and cleaning: offsets must stay exact or every derived label is wrong."""
from __future__ import annotations

import pytest

from rag_system.preprocessing.chunking import chunk_document, chunk_stats
from rag_system.preprocessing.cleaning import (build_raw_to_clean_map, clean_text,
                                               split_sentences)


def test_chunk_offsets_reconstruct_source_exactly(documents):
    text = documents["docA"]
    for chunk in chunk_document(text, "docA", chunk_size=200, chunk_overlap=40):
        assert chunk.text == text[chunk.start_char:chunk.end_char]


def test_chunks_cover_all_non_whitespace_characters(documents):
    text = documents["docB"]
    chunks = chunk_document(text, "docB", chunk_size=180, chunk_overlap=30)
    covered = set()
    for c in chunks:
        covered.update(range(c.start_char, c.end_char))
    uncovered = [i for i, ch in enumerate(text) if i not in covered and not ch.isspace()]
    assert uncovered == []


def test_overlap_produces_shared_content(documents):
    chunks = chunk_document(documents["docA"], "docA", chunk_size=200, chunk_overlap=60)
    assert len(chunks) > 1
    # Consecutive chunks should overlap in character space.
    assert any(b.start_char < a.end_char for a, b in zip(chunks, chunks[1:]))


def test_chunk_size_is_respected(documents):
    chunks = chunk_document(documents["docA"] * 2, "docA", chunk_size=250, chunk_overlap=50)
    assert all(len(c.text) <= 250 for c in chunks)


def test_sentence_strategy_does_not_split_mid_sentence(documents):
    chunks = chunk_document(documents["docA"], "docA", chunk_size=200, chunk_overlap=0,
                            strategy="sentence")
    # Every chunk should start at a capital letter or digit, not mid-word.
    assert all(c.text[0].isupper() or c.text[0].isdigit() for c in chunks)


def test_oversized_sentence_is_hard_split():
    text = "word " * 400  # no sentence terminator at all
    chunks = chunk_document(text, "d", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document("abc def", "d", chunk_size=100, chunk_overlap=100)


def test_empty_document_yields_no_chunks():
    assert chunk_document("   \n  ", "d") == []


def test_contains_and_overlaps_span(documents):
    chunk = chunk_document(documents["docA"], "docA", chunk_size=200, chunk_overlap=0)[0]
    assert chunk.contains_span(chunk.start_char + 1, chunk.start_char + 5)
    assert not chunk.contains_span(chunk.end_char - 2, chunk.end_char + 50)
    assert chunk.overlaps_span(chunk.end_char - 2, chunk.end_char + 50)


def test_cleaning_preserves_offset_mapping():
    raw = "Title   Here\n\n\nAbstract:  SARS-CoV-2   is  a virus.\n\nNext."
    cleaned = clean_text(raw)
    for i, ch in enumerate(cleaned.text):
        if not ch.isspace():
            assert raw[cleaned.offset_map[i]] == ch


def test_raw_to_clean_roundtrip_recovers_the_substring():
    raw = "Intro.\n\n\nThe   term  SARS-CoV-2   appears here."
    cleaned = clean_text(raw)
    inverse = build_raw_to_clean_map(cleaned, len(raw))
    raw_idx = raw.index("SARS-CoV-2")
    clean_idx = inverse[raw_idx]
    assert cleaned.text[clean_idx:clean_idx + len("SARS-CoV-2")] == "SARS-CoV-2"


def test_split_sentences_covers_text():
    text = "One thing happened. Another followed! A third? Yes."
    spans = split_sentences(text)
    assert len(spans) >= 3
    assert all(0 <= s < e <= len(text) for s, e in spans)


def test_chunk_stats_reports_counts(chunks):
    stats = chunk_stats(chunks)
    assert stats["n_chunks"] == len(chunks)
    assert stats["n_documents"] == 2
    assert stats["min_chars"] <= stats["median_chars"] <= stats["max_chars"]
