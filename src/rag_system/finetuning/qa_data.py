"""Feature preparation for extractive QA fine-tuning.

Turning (question, context, answer-span) into model features is the fiddly part
of extractive QA, and getting it wrong produces a model that trains without
error and scores near zero.  Two details matter:

* **Sliding windows.** A context longer than ``max_seq_length`` is split into
  overlapping windows.  Windows that do not contain the answer are labelled
  with the CLS position, which is what teaches the model to abstain rather than
  guess a span from an irrelevant window.
* **Offset mapping.** Character offsets are carried through tokenisation so the
  predicted token span can be mapped back to the exact substring of the original
  context, and so the training labels land on the right tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def prepare_train_features(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_seq_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, List[Any]]:
    """Tokenise and attach ``start_positions`` / ``end_positions`` labels."""
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]

        start_char = examples["answer_start"][sample_index]
        end_char = examples["answer_end"][sample_index]

        # Locate the context span within this window.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Answer entirely outside this window -> point at CLS (the "no answer
        # here" target). Without this the model is taught a wrong span whenever
        # the window misses the answer.
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized["start_positions"].append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized["end_positions"].append(token_end_index + 1)

    return tokenized


def prepare_eval_features(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_seq_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, List[Any]]:
    """Tokenise for inference, keeping the example id and context offsets."""
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["qid"][sample_index])
        # Null out offsets for non-context tokens so postprocessing cannot pick
        # a span out of the question.
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized


def postprocess_predictions(
    examples: Sequence[Dict[str, Any]],
    features: Sequence[Dict[str, Any]],
    raw_predictions,
    n_best_size: int = 20,
    max_answer_length: int = 128,
) -> Dict[str, str]:
    """Map start/end logits back to answer strings.

    Considers the top ``n_best_size`` start and end positions per window, keeps
    only spans that are inside the context, correctly ordered and shorter than
    ``max_answer_length``, and takes the highest-scoring candidate across all
    windows of an example.
    """
    all_start_logits, all_end_logits = raw_predictions

    features_per_example: Dict[str, List[int]] = {}
    for i, feature in enumerate(features):
        features_per_example.setdefault(feature["example_id"], []).append(i)

    predictions: Dict[str, str] = {}
    for example in examples:
        qid = example["qid"]
        context = example["context"]
        best_text, best_score = "", -1e9

        for feature_index in features_per_example.get(qid, []):
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1]
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1]
            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    score = float(start_logits[s] + end_logits[e])
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0] : offsets[e][1]]

        predictions[qid] = best_text.strip()
    return predictions
