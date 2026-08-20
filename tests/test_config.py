"""Configuration loading, overrides and validation."""
from __future__ import annotations

import pytest
import yaml

from rag_system.utils.config import Config, apply_override, load_config


def test_defaults_load_without_a_file():
    cfg = load_config()
    assert cfg.chunking.chunk_size > 0
    assert cfg.retrieval.method in {"bm25", "dense", "hybrid"}


def test_yaml_roundtrip(tmp_path):
    cfg = load_config()
    cfg.chunking.chunk_size = 777
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert load_config(path).chunking.chunk_size == 777


def test_unknown_key_in_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"chunking": {"chunk_size": 100, "nonsense": 1}}))
    with pytest.raises(KeyError, match="nonsense"):
        load_config(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_override_coerces_types():
    cfg = load_config()
    apply_override(cfg, "chunking.chunk_size", "512")
    assert cfg.chunking.chunk_size == 512 and isinstance(cfg.chunking.chunk_size, int)
    apply_override(cfg, "reranker.enabled", "false")
    assert cfg.reranker.enabled is False


def test_override_rejects_unknown_section_and_key():
    cfg = load_config()
    with pytest.raises(KeyError):
        apply_override(cfg, "nosuch.key", 1)
    with pytest.raises(KeyError):
        apply_override(cfg, "chunking.nosuch", 1)
    with pytest.raises(ValueError):
        apply_override(cfg, "chunking", 1)


def test_flat_params_are_scalar_for_tracking():
    flat = load_config().flat_params()
    assert "chunking.chunk_size" in flat
    assert all(not isinstance(v, (dict, list)) for v in flat.values())


def test_split_fractions_must_sum_to_one():
    from rag_system.data.covidqa import split_documents
    from rag_system.utils.config import DataConfig

    with pytest.raises(ValueError, match="sum to 1.0"):
        split_documents(["a"], DataConfig(train_frac=0.5, val_frac=0.2, test_frac=0.2))


def test_document_splits_are_deterministic_and_stable():
    from rag_system.data.covidqa import split_documents
    from rag_system.utils.config import DataConfig

    cfg = DataConfig()
    ids = [str(i) for i in range(200)]
    first = split_documents(ids, cfg)
    assert first == split_documents(ids, cfg)
    # Removing a document must not reshuffle the others (hash-based assignment).
    subset = split_documents(ids[:100], cfg)
    assert all(subset[k] == first[k] for k in subset)
    assert set(first.values()) == {"train", "validation", "test"}
