"""Embedding backends behind one interface.

Two backends:

* ``fast`` (default): ``BAAI/bge-small-en-v1.5`` via fastembed's CPU-ONNX runtime.
  384-dim, no torch, downloads once and caches. This is the ported v1 path.
* ``style`` (optional, extra ``[style]``): StyleDistance sentence-transformer,
  the 2025 SOTA at disentangling *style* from *topic*. Ported from RVCR's
  ``style_embed`` wrapper. Imported lazily so the base install never pulls torch;
  a missing dependency raises a clear, actionable error.

Both return L2-normalized ``float32`` vectors, so cosine similarity is a dot
product. Row-normalized corpus matrices are cached per profile and keyed on the
store's mtime, so re-ingesting or switching voices self-invalidates the cache
(ported cache-invalidation pattern from v1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from . import config

# --- backend: fast (bge-small ONNX) ------------------------------------------

_FAST_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_FAST_DIM = 384
_fast_model = None


def _fast_model_instance():
    global _fast_model
    if _fast_model is None:
        from fastembed import TextEmbedding  # lazy: keeps import time low

        _fast_model = TextEmbedding(
            model_name=_FAST_MODEL_NAME, cache_dir=str(config.embed_cache_dir())
        )
    return _fast_model


def _fast_encode(texts: list[str]) -> np.ndarray:
    model = _fast_model_instance()
    vecs = np.array(list(model.embed(list(texts))), dtype=np.float32)
    return _normalize(vecs)


# --- backend: style (StyleDistance, optional) --------------------------------
#
# StyleDistance (arXiv:2410.12757, MIT) is a RoBERTa-base sentence-transformer
# trained on synthetic near-paraphrase pairs that hold *content* fixed while
# varying a single *style* feature, so its 768-dim vectors encode style while
# staying (relatively) invariant to topic. That is exactly the critic the design
# wants: a topic-leaning embedder (bge-small) rewards anchors that share subject
# matter; a style embedder rewards anchors that share *voice*. Mean-pooled,
# 512-token cap, no query/passage prefix. We L2-normalize at encode time so a dot
# product equals cosine downstream.

_STYLE_MODEL_ENV = "MIMESIS_STYLE_MODEL"
_STYLE_DEVICE_ENV = "MIMESIS_STYLE_DEVICE"
_STYLE_MODEL_DEFAULT = "StyleDistance/styledistance"
_STYLE_DIM = 768
_style_model = None


def _style_device() -> str | None:
    """Resolve the torch device for the style model: explicit override, else CUDA if present."""
    import os

    override = os.environ.get(_STYLE_DEVICE_ENV, "").strip()
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None  # let sentence-transformers auto-select


def _style_model_instance():
    global _style_model
    if _style_model is None:
        import os

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "the 'style' embed backend needs the [style] extra "
                "(pip install 'mimesis-voice[style]'); import failed: " + str(exc)
            ) from exc
        name = os.environ.get(_STYLE_MODEL_ENV, _STYLE_MODEL_DEFAULT)
        _style_model = SentenceTransformer(name, device=_style_device())
    return _style_model


def _style_encode(texts: list[str]) -> np.ndarray:
    model = _style_model_instance()
    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
    ).astype(np.float32)
    return vecs


# --- dispatch -----------------------------------------------------------------


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode(texts: list[str], backend: str = "fast") -> np.ndarray:
    """Embed texts with the named backend. Returns an (n, d) L2-normalized array."""
    if not texts:
        return np.zeros((0, _STYLE_DIM if backend == "style" else _FAST_DIM), dtype=np.float32)
    if backend == "style":
        return _style_encode(texts)
    return _fast_encode(texts)


def embed_one(text: str, backend: str = "fast") -> np.ndarray:
    return encode([text], backend=backend)[0]


def backend_dim(backend: str) -> int:
    """Nominal vector dimension for a backend (fast is fixed; style is model-defined)."""
    if backend == "style":
        model = _style_model_instance()
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension.
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(model, attr, None)
            if callable(fn):
                dim = fn()
                if dim:
                    return int(dim)
        return _STYLE_DIM
    return _FAST_DIM


# --- corpus matrix cache ------------------------------------------------------

_matrix_cache: dict[str, dict] = {}


def _build_matrix(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, text, word_count, chunk_idx, embedding FROM chunks"
    )
    rows = cur.fetchall()
    ids, filenames, texts, word_counts, chunk_idxs, vectors = [], [], [], [], [], []
    for row in rows:
        ids.append(row["id"])
        filenames.append(row["filename"])
        texts.append(row["text"])
        word_counts.append(row["word_count"])
        chunk_idxs.append(row["chunk_idx"])
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
    if vectors:
        matrix = _normalize(np.vstack(vectors).astype(np.float32))
    else:
        matrix = np.zeros((0, _FAST_DIM), dtype=np.float32)
    return {
        "ids": ids,
        "filenames": filenames,
        "texts": texts,
        "word_counts": word_counts,
        "chunk_idxs": chunk_idxs,
        "matrix": matrix,
    }


def get_matrix(conn: sqlite3.Connection, db_path: Path, slug: str) -> dict:
    """Row-normalized embedding matrix for a profile, cached and mtime-keyed."""
    try:
        mtime = int(db_path.stat().st_mtime)
    except OSError:
        mtime = 0
    key = f"{slug}:{mtime}"
    cached = _matrix_cache.get(key)
    if cached is None:
        cached = _build_matrix(conn)
        _matrix_cache.clear()  # keep only the live profile resident (bounded memory)
        _matrix_cache[key] = cached
    return cached
