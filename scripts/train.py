#!/usr/bin/env python
"""Phase 4: LoRA fine-tune the extractive QA reader.

    python scripts/train.py --config configs/finetuning.yaml

Evaluates the base model zero-shot, fine-tunes with LoRA, re-evaluates on the
same held-out documents, and writes results/finetuning/finetuning_results.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402  (faiss/torch OpenMP load order)

from rag_system.data.covidqa import build_corpus                       # noqa: E402
from rag_system.finetuning.train_qa import finetune                    # noqa: E402
from rag_system.utils.config import RESULTS_DIR, load_config           # noqa: E402
from rag_system.utils.runtime import get_logger, set_seed, write_json  # noqa: E402
from rag_system.utils.tracking import start_run                        # noqa: E402

LOGGER = get_logger()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finetuning.yaml")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "finetuning" / "finetuning_results.json")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None, help="smoke-test subset")
    p.add_argument("--no-lora", action="store_true", help="full fine-tuning instead of LoRA")
    p.add_argument("--no-tracking", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.finetuning.num_epochs = args.epochs
    if args.lora_r is not None:
        cfg.finetuning.lora_r = args.lora_r
    if args.no_lora:
        cfg.finetuning.use_lora = False
        cfg.finetuning.learning_rate = 3e-5   # full FT needs a much smaller LR than LoRA
        cfg.finetuning.output_dir = "artifacts/finetuned/roberta-covidqa-full"
    set_seed(cfg.experiment.seed)

    corpus = build_corpus(cfg.data, cfg.chunking)
    results = finetune(corpus, cfg.finetuning, max_train_examples=args.max_train_examples)

    write_json(args.out, results)
    LOGGER.info("wrote %s", args.out)

    with start_run(f"finetune-{'lora' if cfg.finetuning.use_lora else 'full'}",
                   cfg.experiment.tracking_uri, cfg.experiment.experiment_name,
                   enabled=not args.no_tracking) as run:
        run.set_tags({"phase": "finetuning", "method": "lora" if cfg.finetuning.use_lora else "full"})
        run.log_params(cfg.flat_params())
        p_ = results["parameters"]["with_lora"]
        run.log_metrics({
            "trainable_parameters": p_["trainable_parameters"],
            "total_parameters": p_["total_parameters"],
            "trainable_percent": p_["trainable_percent"],
            "train_seconds": results["training"]["seconds"],
            "final_train_loss": results["training"]["final_train_loss"],
            "base_test_em": results["metrics"]["base"]["test"]["exact_match"],
            "base_test_f1": results["metrics"]["base"]["test"]["f1"],
            "ft_test_em": results["metrics"]["finetuned"]["test"]["exact_match"],
            "ft_test_f1": results["metrics"]["finetuned"]["test"]["f1"],
        })

    m = results["metrics"]
    print("\n" + "=" * 78)
    print(f"{'model':<28}{'val EM':>10}{'val F1':>10}{'test EM':>10}{'test F1':>10}")
    print("-" * 78)
    for label, key in (("base (SQuAD2, zero-shot)", "base"), ("LoRA fine-tuned", "finetuned")):
        print(f"{label:<28}{m[key]['validation']['exact_match']:>10.2f}{m[key]['validation']['f1']:>10.2f}"
              f"{m[key]['test']['exact_match']:>10.2f}{m[key]['test']['f1']:>10.2f}")
    print("-" * 78)
    print(f"{'delta (test)':<28}{'':<20}{m['delta_test']['exact_match']:>10.2f}{m['delta_test']['f1']:>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
