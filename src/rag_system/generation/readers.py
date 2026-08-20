"""Readers: turn a question plus retrieved context into an answer.

Two reader families implement the same ``Reader`` interface so the RAG pipeline
is agnostic to which is loaded, and so the fine-tuned adapter can be swapped in
without touching pipeline code:

* :class:`ExtractiveReader` -- a span-prediction model (RoBERTa/SQuAD2 head).
  It can only return substrings of the supplied context, so it is *structurally
  incapable* of hallucinating text, and it returns a usable confidence via the
  span logits.  Its ceiling is bounded by whether the answer is in the context.
* :class:`GenerativeReader` -- a seq2seq model (Flan-T5).  It can paraphrase and
  answer questions whose gold answer is not a clean span, but it can also invent
  text, and it is far slower.

The extractive reader is the default for COVID-QA because the dataset's answers
*are* annotated spans, which makes EM/F1 a fair measurement of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..retrieval.base import RetrievalResult
from ..utils.config import GenerationConfig
from ..utils.runtime import get_logger, resolve_device

LOGGER = get_logger()


@dataclass
class Answer:
    text: str
    score: float
    reader: str
    context_chars: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    source_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": round(float(self.score), 6),
            "reader": self.reader,
            "context_chars": self.context_chars,
            "source_chunk_id": self.source_chunk_id,
            "metadata": self.metadata,
        }


class Reader(Protocol):
    name: str

    def answer(self, question: str, context: str) -> Answer: ...
    def answer_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> List[Answer]: ...


class ExtractiveReader:
    """Span-extraction reader with sliding-window support for long contexts."""

    name = "extractive"

    def __init__(self, cfg: Optional[GenerationConfig] = None, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.cfg = cfg or GenerationConfig()
        self.model_name = model_name or self.cfg.extractive_model
        self.device = resolve_device(self.cfg.device)
        LOGGER.info("loading extractive reader %s on %s", self.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        if self.cfg.adapter_path:
            from peft import PeftModel

            LOGGER.info("applying LoRA adapter from %s", self.cfg.adapter_path)
            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)
            model = model.merge_and_unload()  # fold LoRA into the base weights for fast inference
            self.name = "extractive-lora"

        self.model = model.to(self.device).eval()
        self._torch = torch

    @property
    def max_length(self) -> int:
        return min(self.cfg.max_input_tokens, self.tokenizer.model_max_length)

    def answer(self, question: str, context: str) -> Answer:
        return self.answer_batch([question], [context])[0]

    def answer_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> List[Answer]:
        """Answer several (question, context) pairs.

        Long contexts are split into overlapping windows and the highest-scoring
        span across all windows of a question wins, so an answer near the end of
        a long context is not silently truncated away.
        """
        torch = self._torch
        encodings = self.tokenizer(
            list(questions),
            list(contexts),
            truncation="only_second",
            max_length=self.max_length,
            stride=self.cfg.max_input_tokens // 4,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        sample_map = encodings.pop("overflow_to_sample_mapping").tolist()
        offset_mapping = encodings.pop("offset_mapping")
        sequence_ids_per_window = [encodings.sequence_ids(i) for i in range(len(sample_map))]

        inputs = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_logits = outputs.start_logits.float().cpu().numpy()
        end_logits = outputs.end_logits.float().cpu().numpy()

        best: Dict[int, Dict[str, Any]] = {}
        for window_idx, sample_idx in enumerate(sample_map):
            seq_ids = sequence_ids_per_window[window_idx]
            offsets = offset_mapping[window_idx].tolist()
            # Only context tokens (sequence id 1) are valid answer positions.
            valid = np.array([sid == 1 for sid in seq_ids], dtype=bool)
            if not valid.any():
                continue

            span = _best_span(start_logits[window_idx], end_logits[window_idx], valid,
                              max_answer_tokens=self.cfg.max_answer_tokens)
            if span is None:
                continue
            start_tok, end_tok, score = span
            char_start, char_end = offsets[start_tok][0], offsets[end_tok][1]
            if char_end <= char_start:
                continue
            if sample_idx not in best or score > best[sample_idx]["score"]:
                best[sample_idx] = {"score": score, "start": char_start, "end": char_end}

        answers: List[Answer] = []
        for i, context in enumerate(contexts):
            hit = best.get(i)
            if hit is None:
                answers.append(Answer(text="", score=0.0, reader=self.name, context_chars=len(context)))
            else:
                answers.append(
                    Answer(
                        text=context[hit["start"]:hit["end"]].strip(),
                        score=float(hit["score"]),
                        reader=self.name,
                        context_chars=len(context),
                        start_char=hit["start"],
                        end_char=hit["end"],
                    )
                )
        return answers


def _best_span(start_logits: np.ndarray, end_logits: np.ndarray, valid: np.ndarray,
               max_answer_tokens: int = 64, top_n: int = 20):
    """Highest-scoring (start, end) pair subject to end >= start and a length cap.

    Scoring the full O(L^2) grid is unnecessary: only the top-N start and end
    positions can participate in the best pair, so the search is O(N^2).
    """
    starts = np.where(valid, start_logits, -1e9)
    ends = np.where(valid, end_logits, -1e9)
    start_idx = np.argsort(-starts)[:top_n]
    end_idx = np.argsort(-ends)[:top_n]

    best = None
    for s in start_idx:
        for e in end_idx:
            if e < s or (e - s + 1) > max_answer_tokens:
                continue
            score = float(starts[s] + ends[e])
            if best is None or score > best[2]:
                best = (int(s), int(e), score)
    return best


PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "Quote the answer from the context as briefly as possible. "
    "If the context does not contain the answer, reply exactly: unanswerable.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

CLOSED_BOOK_TEMPLATE = "Answer the question as briefly as possible.\n\nQuestion: {question}\nAnswer:"


class GenerativeReader:
    """Seq2seq reader (Flan-T5 family) with an optional closed-book mode."""

    name = "generative"

    def __init__(self, cfg: Optional[GenerationConfig] = None, model_name: Optional[str] = None,
                 closed_book: bool = False):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg or GenerationConfig()
        self.model_name = model_name or self.cfg.generative_model
        self.closed_book = closed_book
        self.device = resolve_device(self.cfg.device)
        LOGGER.info("loading generative reader %s on %s (closed_book=%s)",
                    self.model_name, self.device, closed_book)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()
        self._torch = torch
        if closed_book:
            self.name = "generative-closed-book"

    def _prompt(self, question: str, context: str) -> str:
        if self.closed_book:
            return CLOSED_BOOK_TEMPLATE.format(question=question)
        return PROMPT_TEMPLATE.format(context=context[: self.cfg.max_context_chars], question=question)

    def answer(self, question: str, context: str) -> Answer:
        return self.answer_batch([question], [context])[0]

    def answer_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> List[Answer]:
        torch = self._torch
        prompts = [self._prompt(q, c) for q, c in zip(questions, contexts)]
        encodings = self.tokenizer(
            prompts, truncation=True, max_length=self.cfg.max_input_tokens,
            padding=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **encodings,
                max_new_tokens=self.cfg.max_answer_tokens,
                num_beams=self.cfg.num_beams,
                do_sample=False,               # greedy/beam: deterministic, so results reproduce
            )
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [
            Answer(text=t.strip(), score=float("nan"), reader=self.name, context_chars=len(c))
            for t, c in zip(texts, contexts)
        ]


def build_reader(cfg: GenerationConfig, closed_book: bool = False) -> Reader:
    if closed_book:
        return GenerativeReader(cfg, closed_book=True)
    if cfg.reader_type == "extractive":
        return ExtractiveReader(cfg)
    if cfg.reader_type == "generative":
        return GenerativeReader(cfg)
    raise ValueError(f"unknown reader_type {cfg.reader_type!r}; use 'extractive' or 'generative'")
