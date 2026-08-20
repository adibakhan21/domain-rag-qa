"""Typed configuration loading.

Every experiment is described by a YAML file under ``configs/``.  Configs are
parsed into dataclasses (so typos fail loudly instead of silently defaulting)
and are serialised back out beside every result file, which is what makes a run
reproducible from its artefacts alone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"

T = TypeVar("T")


@dataclass
class DataConfig:
    """Dataset and document-level split configuration."""

    dataset_name: str = "deepset/covid_qa_deepset"
    split: str = "train"
    # Splits are over *documents*, not questions: two questions on the same
    # paper must never straddle the train/test boundary or the reader is
    # evaluated on documents it was fine-tuned on.
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    max_documents: Optional[int] = None  # smoke-test escape hatch


@dataclass
class ChunkingConfig:
    """Sentence-aware chunking parameters (character units)."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    strategy: str = "sentence"  # "sentence" | "fixed"


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    device: Optional[str] = None  # None -> auto-detect (mps > cuda > cpu)
    # bge models are trained with an asymmetric query instruction; omitting it
    # measurably hurts retrieval, so it is part of the config rather than hidden.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""


@dataclass
class RetrievalConfig:
    method: str = "hybrid"  # bm25 | dense | hybrid
    top_k: int = 10
    candidate_k: int = 50   # depth retrieved before reranking
    rrf_k: int = 60         # reciprocal-rank-fusion smoothing constant
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 64
    top_k: int = 10
    device: Optional[str] = None


@dataclass
class GenerationConfig:
    """Reader configuration.

    ``reader_type`` selects between an extractive span reader and a seq2seq
    generative reader; both implement the same interface so the RAG pipeline is
    agnostic to which one is loaded.
    """

    reader_type: str = "extractive"  # extractive | generative
    extractive_model: str = "deepset/roberta-base-squad2"
    generative_model: str = "google/flan-t5-base"
    adapter_path: Optional[str] = None  # LoRA adapter dir, if any
    max_context_chars: int = 4000
    max_answer_tokens: int = 64
    max_input_tokens: int = 512
    num_beams: int = 1
    device: Optional[str] = None


@dataclass
class FinetuningConfig:
    base_model: str = "deepset/roberta-base-squad2"
    output_dir: str = "artifacts/finetuned/roberta-covidqa-lora"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    learning_rate: float = 3e-4      # LoRA tolerates (and needs) a higher LR than full FT
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 384
    doc_stride: int = 128
    seed: int = 42
    fp16: bool = False               # MPS does not support fp16 autocast reliably
    device: Optional[str] = None


@dataclass
class EvaluationConfig:
    recall_at_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    ndcg_at_k: int = 10
    mrr_at_k: int = 10
    max_eval_questions: Optional[int] = None
    n_bootstrap: int = 1000          # bootstrap resamples for confidence intervals


@dataclass
class ExperimentConfig:
    name: str = "default"
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "domain-rag-qa"
    seed: int = 42


@dataclass
class Config:
    """Root configuration object."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def flat_params(self, prefix: str = "") -> Dict[str, Any]:
        """Flatten to ``section.key`` pairs for experiment-tracker logging."""
        out: Dict[str, Any] = {}
        for section, values in self.to_dict().items():
            if isinstance(values, dict):
                for k, v in values.items():
                    out[f"{prefix}{section}.{k}"] = v if not isinstance(v, list) else ",".join(map(str, v))
            else:
                out[f"{prefix}{section}"] = values
        return out


def _build(cls: Type[T], raw: Optional[Dict[str, Any]]) -> T:
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; valid keys are {sorted(known)}")
    return cls(**raw)


def load_config(path: Optional[Path] = None, **overrides: Any) -> Config:
    """Load a YAML config, applying ``section.key=value`` overrides.

    Unknown keys raise rather than being ignored -- a silently-dropped
    ``chunk_size`` would invalidate an experiment without any visible error.
    """
    raw: Dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}

    cfg = Config(
        experiment=_build(ExperimentConfig, raw.get("experiment")),
        data=_build(DataConfig, raw.get("data")),
        chunking=_build(ChunkingConfig, raw.get("chunking")),
        embedding=_build(EmbeddingConfig, raw.get("embedding")),
        retrieval=_build(RetrievalConfig, raw.get("retrieval")),
        reranker=_build(RerankerConfig, raw.get("reranker")),
        generation=_build(GenerationConfig, raw.get("generation")),
        finetuning=_build(FinetuningConfig, raw.get("finetuning")),
        evaluation=_build(EvaluationConfig, raw.get("evaluation")),
    )
    for dotted, value in overrides.items():
        apply_override(cfg, dotted, value)
    return cfg


def apply_override(cfg: Config, dotted: str, value: Any) -> None:
    """Apply a single ``section.key`` override in place, with type coercion."""
    if "." not in dotted:
        raise ValueError(f"override must be 'section.key', got {dotted!r}")
    section_name, key = dotted.split(".", 1)
    if not hasattr(cfg, section_name):
        raise KeyError(f"unknown config section {section_name!r}")
    section = getattr(cfg, section_name)
    if not hasattr(section, key):
        raise KeyError(f"unknown key {key!r} in section {section_name!r}")

    current = getattr(section, key)
    if isinstance(value, str) and not isinstance(current, str):
        value = _coerce(value, current)
    setattr(section, key, value)


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [int(v) if v.strip().lstrip("-").isdigit() else v.strip() for v in value.split(",")]
    if current is None:
        lowered = value.lower()
        if lowered in {"none", "null", ""}:
            return None
        return value
    return value
