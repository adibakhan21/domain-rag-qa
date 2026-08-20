"""Answer-quality metrics.

Exact Match and token-F1 follow the SQuAD v1.1 official implementation
(normalise: lowercase, strip articles/punctuation, collapse whitespace), so the
numbers are comparable with published extractive-QA results.

Two grounding proxies are also computed.  They are **not** RAGAS-style
LLM-judged faithfulness -- see the note on ``groundedness`` -- and are labelled
as proxies everywhere they are reported.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence


def normalize_answer(s: str) -> str:
    """SQuAD normalisation, extended to Unicode punctuation.

    The official SQuAD script strips only ``string.punctuation`` (ASCII),
    because SQuAD is ASCII.  CORD-19 full text is not: it contains U+2010
    hyphens, en/em dashes and curly quotes, so a model answering
    ``HCoV-HKU1`` against the gold string ``HCoV‐HKU1`` (U+2010) would be
    scored *wrong* on a purely typographic difference.

    Unicode punctuation is therefore stripped too, using the general category
    (``P*``) rather than a hand-written list.  This is a deliberate, documented
    deviation from the official script; it only ever removes characters, so it
    cannot turn a wrong answer into a right one -- it can only stop a right
    answer being marked wrong.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(
            ch for ch in text
            if ch not in exclude and not unicodedata.category(ch).startswith("P")
        )

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Bag-of-tokens F1 between prediction and reference."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        # Both empty counts as a match; one empty counts as a miss.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_references(fn, prediction: str, references: Sequence[str]) -> float:
    return max((fn(prediction, ref) for ref in references), default=0.0)


def answer_in_context(answer: str, context: str) -> float:
    """Is the gold answer present in the retrieved context at all?

    This is the ceiling on what any extractive reader can achieve given that
    context, so it separates retrieval failure from reading failure.
    """
    return float(normalize_answer(answer) in normalize_answer(context))


def groundedness(prediction: str, context: str) -> float:
    """Fraction of predicted answer tokens that appear in the retrieved context.

    A *lexical* faithfulness proxy, not a judgement of factual support.  It
    catches the clearest hallucination mode -- an answer containing tokens that
    appear nowhere in the provided evidence -- but it cannot detect an answer
    that recombines context tokens into a false claim, and it under-scores
    correct paraphrases.  RAGAS-style faithfulness would need an LLM judge and
    therefore an API key, which would make the project non-reproducible offline;
    that trade-off is documented rather than hidden.
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    ctx_tokens = set(normalize_answer(context).split())
    return sum(1 for t in pred_tokens if t in ctx_tokens) / len(pred_tokens)


def context_relevance(gold_chunk_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """Precision of the retrieved context: fraction of retrieved chunks that are gold."""
    if not retrieved_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    return sum(1 for cid in retrieved_ids if cid in gold) / len(retrieved_ids)


def aggregate_qa(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Mean of every numeric field across per-question records."""
    records = list(records)
    if not records:
        return {}
    numeric_keys = [k for k, v in records[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out: Dict[str, float] = {"n": len(records)}
    for k in numeric_keys:
        values = [float(r[k]) for r in records if isinstance(r.get(k), (int, float))]
        if values:
            out[k] = sum(values) / len(values)
    return out
