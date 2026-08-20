"""LoRA fine-tuning of an extractive QA reader on COVID-QA.

Why LoRA rather than full fine-tuning
-------------------------------------
The training set is ~1.2k questions over 102 documents.  Full fine-tuning of a
125M-parameter RoBERTa on that little in-domain data overfits quickly and needs
a full optimiser state in memory (≈3x model size), which is uncomfortable on a
16GB laptop.  LoRA freezes the backbone and learns rank-``r`` updates to the
attention projections, so only a fraction of a percent of parameters are
trainable, the optimiser state is tiny, and the strong SQuAD2 prior in the base
model is preserved rather than washed out.

What is being measured
----------------------
The base model (``deepset/roberta-base-squad2``) is already a competent
extractive reader, so this experiment does **not** ask "can we learn QA?"  It
asks the narrower and more honest question: *does in-domain adaptation on
biomedical text improve a general SQuAD2 reader?*  Both are evaluated on the
same held-out documents, so the answer is measured either way.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.covidqa import Corpus, QAExample
from ..evaluation.qa_metrics import exact_match, token_f1
from ..preprocessing.chunking import Chunk
from ..utils.config import FinetuningConfig
from ..utils.runtime import get_logger, resolve_device, set_seed
from .qa_data import postprocess_predictions, prepare_eval_features, prepare_train_features

LOGGER = get_logger()


def build_reader_examples(
    examples: Sequence[QAExample], chunks: Sequence[Chunk]
) -> List[Dict[str, Any]]:
    """Pair each question with its gold chunk as the training context.

    Training on the gold chunk gives clean span supervision.  It does introduce
    a train/inference mismatch -- at inference the reader sees *retrieved*
    context, which contains distractor passages -- and that mismatch is called
    out in the README rather than papered over.  Training on retrieved context
    instead would entangle reader quality with retriever quality and make the
    ablation uninterpretable.
    """
    by_id = {c.chunk_id: c for c in chunks}
    out: List[Dict[str, Any]] = []
    for ex in examples:
        for chunk_id in ex.gold_chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            start = ex.answer_start - chunk.start_char
            end = ex.answer_end - chunk.start_char
            # Only spans wholly inside the chunk are usable as span labels.
            if start < 0 or end > len(chunk.text) or end <= start:
                continue
            if chunk.text[start:end] != ex.answer_text:
                continue
            out.append({
                "qid": ex.qid,
                "question": ex.question,
                "context": chunk.text,
                "answer_text": ex.answer_text,
                "answer_start": start,
                "answer_end": end,
            })
            break  # one training instance per question
    return out


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def evaluate_reader(
    model, tokenizer, reader_examples: Sequence[Dict[str, Any]],
    max_seq_length: int = 384, doc_stride: int = 128, batch_size: int = 16,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Exact Match / token-F1 of a span reader on (question, context) pairs."""
    import torch
    from datasets import Dataset

    device = device or resolve_device()
    ds = Dataset.from_list(list(reader_examples))
    features = ds.map(
        lambda b: prepare_eval_features(b, tokenizer, max_seq_length, doc_stride),
        batched=True, remove_columns=ds.column_names, desc="tokenising eval",
    )

    model.eval()
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    keep = ["input_ids", "attention_mask"]
    if "token_type_ids" in features.column_names:
        keep.append("token_type_ids")

    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size]
        inputs = {k: torch.tensor(batch[k], device=device) for k in keep}
        with torch.no_grad():
            out = model(**inputs)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())

    raw = (np.concatenate(starts, axis=0), np.concatenate(ends, axis=0))
    predictions = postprocess_predictions(list(reader_examples), list(features), raw)

    em = float(np.mean([exact_match(predictions[e["qid"]], e["answer_text"]) for e in reader_examples]))
    f1 = float(np.mean([token_f1(predictions[e["qid"]], e["answer_text"]) for e in reader_examples]))
    return {"exact_match": 100 * em, "f1": 100 * f1, "n": len(reader_examples)}, predictions


def finetune(
    corpus: Corpus,
    cfg: Optional[FinetuningConfig] = None,
    output_dir: Optional[Path] = None,
    max_train_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Fine-tune with LoRA and return base-vs-finetuned metrics."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              Trainer, TrainingArguments, default_data_collator)

    cfg = cfg or FinetuningConfig()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = Path(output_dir or cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ex = build_reader_examples(corpus.examples_for("train"), corpus.chunks)
    val_ex = build_reader_examples(corpus.examples_for("validation"), corpus.chunks)
    test_ex = build_reader_examples(corpus.examples_for("test"), corpus.chunks)
    if max_train_examples:
        train_ex = train_ex[:max_train_examples]
    LOGGER.info("reader examples: train=%d val=%d test=%d", len(train_ex), len(val_ex), len(test_ex))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    # --- 1. Base model, before any adaptation ----------------------------
    LOGGER.info("evaluating BASE model %s (zero-shot on COVID-QA)", cfg.base_model)
    base_model = AutoModelForQuestionAnswering.from_pretrained(cfg.base_model).to(device)
    base_params = count_parameters(base_model)
    base_val, _ = evaluate_reader(base_model, tokenizer, val_ex, cfg.max_seq_length, cfg.doc_stride, device=device)
    base_test, base_preds = evaluate_reader(base_model, tokenizer, test_ex, cfg.max_seq_length, cfg.doc_stride, device=device)
    LOGGER.info("  base  val EM %.2f F1 %.2f | test EM %.2f F1 %.2f",
                base_val["exact_match"], base_val["f1"], base_test["exact_match"], base_test["f1"])
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    # --- 2. LoRA fine-tuning ---------------------------------------------
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.base_model)
    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            # The QA span head is randomly initialised relative to the adapter
            # and must be trained, so it is saved alongside the LoRA weights.
            modules_to_save=["qa_outputs"],
        )
        model = get_peft_model(model, lora)
    model = model.to(device)
    param_stats = count_parameters(model)
    LOGGER.info("trainable %s / %s params (%.3f%%)",
                f"{param_stats['trainable_parameters']:,}",
                f"{param_stats['total_parameters']:,}",
                param_stats["trainable_percent"])

    train_ds = Dataset.from_list(train_ex).map(
        lambda b: prepare_train_features(b, tokenizer, cfg.max_seq_length, cfg.doc_stride),
        batched=True, remove_columns=list(train_ex[0].keys()), desc="tokenising train",
    )
    val_ds = Dataset.from_list(val_ex).map(
        lambda b: prepare_train_features(b, tokenizer, cfg.max_seq_length, cfg.doc_stride),
        batched=True, remove_columns=list(val_ex[0].keys()), desc="tokenising val",
    )

    steps_per_epoch = max(1, len(train_ds) // (cfg.batch_size * cfg.gradient_accumulation_steps))
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_steps=int(cfg.warmup_ratio * steps_per_epoch * cfg.num_epochs),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=max(1, steps_per_epoch // 4),
        seed=cfg.seed,
        fp16=cfg.fp16,
        report_to=[],           # MLflow logging is handled by the caller
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - t0
    LOGGER.info("training finished in %.1fs", train_seconds)

    if cfg.use_lora:
        model.save_pretrained(str(output_dir))
    else:
        trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- 3. Fine-tuned model ---------------------------------------------
    ft_val, _ = evaluate_reader(model, tokenizer, val_ex, cfg.max_seq_length, cfg.doc_stride, device=device)
    ft_test, ft_preds = evaluate_reader(model, tokenizer, test_ex, cfg.max_seq_length, cfg.doc_stride, device=device)
    LOGGER.info("  lora  val EM %.2f F1 %.2f | test EM %.2f F1 %.2f",
                ft_val["exact_match"], ft_val["f1"], ft_test["exact_match"], ft_test["f1"])

    history = [
        {k: v for k, v in entry.items()}
        for entry in trainer.state.log_history
    ]

    results: Dict[str, Any] = {
        "base_model": cfg.base_model,
        "config": asdict(cfg),
        "device": device,
        "parameters": {"base": base_params, "with_lora": param_stats},
        "dataset": {"train": len(train_ex), "validation": len(val_ex), "test": len(test_ex),
                    "train_features": len(train_ds), "val_features": len(val_ds)},
        "training": {
            "seconds": round(train_seconds, 1),
            "minutes": round(train_seconds / 60, 2),
            "epochs": cfg.num_epochs,
            "steps": int(train_result.global_step),
            "final_train_loss": float(train_result.training_loss),
            "log_history": history,
        },
        "metrics": {
            "base": {"validation": base_val, "test": base_test},
            "finetuned": {"validation": ft_val, "test": ft_test},
            "delta_test": {
                "exact_match": round(ft_test["exact_match"] - base_test["exact_match"], 2),
                "f1": round(ft_test["f1"] - base_test["f1"], 2),
            },
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "predictions_test.json").write_text(
        json.dumps({"base": base_preds, "finetuned": ft_preds}, indent=2)
    )
    return results
