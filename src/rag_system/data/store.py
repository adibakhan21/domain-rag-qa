"""On-disk artefacts: corpus, chunk manifest and vector index.

Artefacts are written under ``artifacts/`` (git-ignored) and always accompanied
by a manifest recording the exact config that produced them.  ``load_index``
refuses to load an index whose manifest disagrees with the requested config,
because silently reusing an index built with a different chunk size or a
different embedding model is the single easiest way to produce a wrong
experiment that still runs cleanly.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.chunking import Chunk
from ..utils.config import ARTIFACT_DIR, ChunkingConfig, EmbeddingConfig
from ..utils.runtime import get_logger

LOGGER = get_logger()


# --- chunks ---------------------------------------------------------------
def write_chunks(path: Path, chunks: Sequence[Chunk]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    return path


def read_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --- vector index ---------------------------------------------------------
def index_dir(name: str, root: Optional[Path] = None) -> Path:
    return Path(root or ARTIFACT_DIR) / "index" / name


def index_name(chunk_cfg: ChunkingConfig, emb_cfg: EmbeddingConfig) -> str:
    """Deterministic directory name so different settings never collide."""
    model = emb_cfg.model_name.split("/")[-1]
    return f"{model}__{chunk_cfg.strategy}_{chunk_cfg.chunk_size}_{chunk_cfg.chunk_overlap}"


def save_index(
    name: str,
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    faiss_index,
    chunk_cfg: ChunkingConfig,
    emb_cfg: EmbeddingConfig,
    root: Optional[Path] = None,
) -> Path:
    out = index_dir(name, root)
    out.mkdir(parents=True, exist_ok=True)
    write_chunks(out / "chunks.jsonl", chunks)
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))
    faiss_index.save(out / "index.faiss")
    manifest = {
        "name": name,
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "chunking": asdict(chunk_cfg),
        "embedding": asdict(emb_cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("saved index '%s' (%d chunks, dim %d) -> %s", name, len(chunks), embeddings.shape[1], out)
    return out


def load_index(
    name: str,
    chunk_cfg: Optional[ChunkingConfig] = None,
    emb_cfg: Optional[EmbeddingConfig] = None,
    root: Optional[Path] = None,
) -> Tuple[List[Chunk], np.ndarray, Any, Dict[str, Any]]:
    """Load a saved index, verifying it matches the requested configuration."""
    from ..retrieval.dense import FaissIndex

    path = index_dir(name, root)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no index at {path}. Build it first:  python scripts/build_index.py"
        )
    manifest = json.loads(manifest_path.read_text())

    if chunk_cfg is not None:
        _assert_matches(manifest["chunking"], asdict(chunk_cfg), "chunking", path)
    if emb_cfg is not None:
        # Only fields that change the vectors are compared; batch_size and
        # device legitimately differ between build time and query time.
        for field_name in ("model_name", "normalize", "query_prefix", "passage_prefix"):
            if manifest["embedding"].get(field_name) != getattr(emb_cfg, field_name):
                raise ValueError(
                    f"index at {path} was built with {field_name}="
                    f"{manifest['embedding'].get(field_name)!r} but the current config requests "
                    f"{getattr(emb_cfg, field_name)!r}. Rebuild the index or fix the config."
                )

    chunks = read_chunks(path / "chunks.jsonl")
    embeddings = np.load(path / "embeddings.npy")
    faiss_index = FaissIndex.load(path / "index.faiss")
    if len(chunks) != faiss_index.size:
        raise ValueError(f"corrupt index at {path}: {len(chunks)} chunks vs {faiss_index.size} vectors")
    return chunks, embeddings, faiss_index, manifest


def _assert_matches(stored: Dict[str, Any], requested: Dict[str, Any], label: str, path: Path) -> None:
    diffs = {k: (stored.get(k), v) for k, v in requested.items() if stored.get(k) != v}
    if diffs:
        pretty = ", ".join(f"{k}: stored={a!r} requested={b!r}" for k, (a, b) in diffs.items())
        raise ValueError(f"index at {path} has mismatched {label} config ({pretty}). Rebuild the index.")
