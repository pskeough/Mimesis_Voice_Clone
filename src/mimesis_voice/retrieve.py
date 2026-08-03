"""Retrieval: RRF hybrid recall, MMR diversification, transform-demo lookup.

Recall fuses a vector search (cosine over the profile's embedding backend) and an
FTS5 keyword search with Reciprocal Rank Fusion (k=60), ported from v1. The fused
candidates are then re-ranked with Maximal Marginal Relevance so the anchors
handed to the writer are *diverse*, not five paraphrases of one passage:

    mmr = lambda * sim(query, cand) - (1 - lambda) * max sim(cand, already_selected)

with lambda = 0.65 (score = 0.65*sim - 0.35*max_overlap_with_selected), implemented
fresh from the design spec. Profiles that ship AI->author transform pairs also get
contrastive demonstration retrieval (ported concept from RVCR), the strongest
anchor for rewrites.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config, embed

_RRF_K = 60
_MMR_LAMBDA = 0.65
_WORD_RE = re.compile(r"\b\w+\b")


def _open(profile: config.Profile) -> sqlite3.Connection:
    if not profile.db_path.exists():
        raise FileNotFoundError(
            f"No store for voice '{profile.slug}' at {profile.db_path}. "
            f"Run: mimesis ingest {profile.slug}"
        )
    conn = sqlite3.connect(str(profile.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fts_available(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'"
        ).fetchone()
        is not None
    )


def hybrid(
    query_text: str,
    limit: int,
    profile: config.Profile,
    exclude_files: set[str] | None = None,
) -> list[dict]:
    """RRF fusion of vector + FTS5 recall. Returns fused hits (each carries its vector).

    ``exclude_files`` drops any chunk whose source filename is in the set, the leak
    control the discrimination eval uses to keep a held-out piece (and its nearest
    neighbours) out of its own anchors.
    """
    exclude_files = exclude_files or set()
    conn = _open(profile)
    try:
        cache = embed.get_matrix(conn, profile.db_path, profile.slug)
        matrix = cache["matrix"]
        if matrix.shape[0] == 0:
            return []
        id_to_idx = {cid: i for i, cid in enumerate(cache["ids"])}

        qv = embed.embed_one(query_text, backend=profile.embed_backend)
        sims = matrix @ qv
        want = min(max(limit * 3, limit), sims.shape[0])
        top_idx = np.argpartition(-sims, want - 1)[:want]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        vector_hits = [
            {
                "id": cache["ids"][i],
                "filename": cache["filenames"][i],
                "text": cache["texts"][i],
                "word_count": cache["word_counts"][i],
                "chunk_idx": cache["chunk_idxs"][i],
                "sim": float(sims[i]),
            }
            for i in top_idx
            if cache["filenames"][i] not in exclude_files
        ]

        fts_hits: list[dict] = []
        if _fts_available(conn):
            terms = [t for t in _WORD_RE.findall(query_text.lower()) if len(t) > 3]
            if terms:
                match_query = " OR ".join(terms[:15])
                rows = conn.execute(
                    "SELECT id, text, rank FROM fts_chunks WHERE fts_chunks MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (match_query, limit * 3),
                ).fetchall()
                for row in rows:
                    fts_hits.append({"id": row["id"], "text": row["text"]})
    finally:
        conn.close()

    # Reciprocal Rank Fusion.
    scores: dict[str, float] = defaultdict(float)
    hits_by_id: dict[str, dict] = {}
    for rank, h in enumerate(vector_hits):
        scores[h["id"]] += 1.0 / (_RRF_K + rank + 1)
        hits_by_id[h["id"]] = h
    for rank, h in enumerate(fts_hits):
        idx = id_to_idx.get(h["id"])
        fname = cache["filenames"][idx] if idx is not None else h["id"].split("::")[0]
        if fname in exclude_files:
            continue
        scores[h["id"]] += 1.0 / (_RRF_K + rank + 1)
        if h["id"] not in hits_by_id:
            hits_by_id[h["id"]] = {
                "id": h["id"],
                "filename": fname,
                "text": h["text"],
                "word_count": len(h["text"].split()),
                "chunk_idx": cache["chunk_idxs"][idx] if idx is not None else 0,
                "sim": float(matrix[idx] @ qv) if idx is not None else 0.0,
            }
    fused = sorted(
        (hits_by_id[cid] for cid in scores), key=lambda h: scores[h["id"]], reverse=True
    )
    # Attach each hit's vector for downstream MMR.
    for h in fused:
        idx = id_to_idx.get(h["id"])
        h["_vec"] = matrix[idx] if idx is not None else None
    return fused


def mmr(
    query_text: str,
    candidates: list[dict],
    k: int,
    backend: str,
    lam: float = _MMR_LAMBDA,
) -> list[dict]:
    """Maximal Marginal Relevance: pick k diverse-yet-relevant candidates.

    score(c) = lam * sim(query, c) - (1 - lam) * max sim(c, selected)
    """
    pool = [c for c in candidates if c.get("_vec") is not None]
    if not pool:
        return candidates[:k]
    qv = embed.embed_one(query_text, backend=backend)
    vecs = {c["id"]: np.asarray(c["_vec"], dtype=np.float32) for c in pool}
    q_sim = {c["id"]: float(vecs[c["id"]] @ qv) for c in pool}

    selected: list[dict] = []
    remaining = list(pool)
    while remaining and len(selected) < k:
        best, best_score = None, -1e9
        for c in remaining:
            if selected:
                overlap = max(float(vecs[c["id"]] @ vecs[s["id"]]) for s in selected)
            else:
                overlap = 0.0
            score = lam * q_sim[c["id"]] - (1.0 - lam) * overlap
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve(
    query_text: str,
    limit: int,
    profile: config.Profile,
    diversify: bool = True,
    exclude_files: set[str] | None = None,
) -> list[dict]:
    """Fused recall then MMR diversification. Returns up to ``limit`` clean hits."""
    fused = hybrid(query_text, max(limit * 3, limit), profile, exclude_files=exclude_files)
    chosen = (
        mmr(query_text, fused, limit, profile.embed_backend) if diversify else fused[:limit]
    )
    for h in chosen:
        h.pop("_vec", None)
    return chosen


# --- transform demos ----------------------------------------------------------


def _load_pairs(path: Path) -> list[dict]:
    pairs: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("human_text") and obj.get("ai_text"):
            pairs.append(obj)
    return pairs


def transform_demos(query_text: str, k: int, profile: config.Profile) -> list[dict]:
    """Contrastive AI->author rewrite pairs closest in style to the query.

    Returns [] when the profile ships no ``pairs.jsonl``. Ranking is by similarity
    of the query to each pair's human (author) side under the profile backend.
    """
    if not profile.pairs_path or not profile.pairs_path.exists():
        return []
    pairs = _load_pairs(profile.pairs_path)
    if not pairs:
        return []
    human_vecs = embed.encode([p["human_text"] for p in pairs], backend=profile.embed_backend)
    qv = embed.embed_one(query_text, backend=profile.embed_backend)
    sims = human_vecs @ qv
    order = np.argsort(-sims)[: max(1, k)]
    return [pairs[i] for i in order]
