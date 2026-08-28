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


# One chunk per source document. Five anchors should be five different pieces of
# his writing, not two slices of one. Measured on the creative profile over seven
# briefs, share of anchor slots taken by the single most-retrieved document:
#
#   diversify=True   cap=inf  23%   cap=2  23% (no-op)   cap=1  17%
#   diversify=False  cap=inf  43%   cap=2  31%           cap=1  20%
#
# and worst-case chunks from one document inside a single brief: 5 -> 2 -> 1.
#
# Note what this does NOT fix. The 8.97x over-representation of a single
# stage-play document, measured across briefs, is cross-brief topical pull: that
# document is close to many queries, and a per-call cap cannot see that. Cap=1
# reduces it (23% -> 17%) without addressing the cause, which is retrieval on
# topic. The register filter is the fix for that.
_PER_DOC_CAP = 1


def _cap_per_document(hits: list[dict], k: int, cap: int = _PER_DOC_CAP) -> list[dict]:
    """At most ``cap`` chunks from any one source document.

    Chunking has no per-document limit, so document balance and retrieval balance
    are different distributions. Measured on the creative profile against its seven
    frozen briefs: three novels (2.6% of documents) own 49.7% of the chunk pool,
    and a single stage-play document supplied 11.4% of every anchor slot in the
    system -- 8.97x its share of the pool -- including two of the top five anchors
    for a philosophical-fragment brief. Novels, half the corpus, got 8.6%.

    MMR does not fix this and never did: measured mean distinct literary forms per
    anchor set was 3.57 with diversify=True and 3.57 with it off. It diversifies on
    embedding distance, which is not document identity.

    A second pass backfills rather than returning short, so a query whose best
    material genuinely lives in one document still gets ``k`` anchors.
    """
    picked, counts = [], defaultdict(int)
    for h in hits:
        fn = h.get("filename") or h["id"].split("::")[0]
        if counts[fn] < cap:
            picked.append(h)
            counts[fn] += 1
        if len(picked) >= k:
            return picked
    seen = {id(h) for h in picked}
    for h in hits:
        if len(picked) >= k:
            break
        if id(h) not in seen:
            picked.append(h)
    return picked


def retrieve(
    query_text: str,
    limit: int,
    profile: config.Profile,
    diversify: bool = True,
    exclude_files: set[str] | None = None,
    per_doc_cap: int = _PER_DOC_CAP,
) -> list[dict]:
    """Fused recall then MMR diversification. Returns up to ``limit`` clean hits."""
    fused = hybrid(query_text, max(limit * 3, limit), profile, exclude_files=exclude_files)
    ranked = (
        mmr(query_text, fused, max(limit * 3, limit), profile.embed_backend)
        if diversify else fused
    )
    chosen = _cap_per_document(ranked, limit, per_doc_cap)
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


# Every pair in pairs.jsonl already carries a ``move``: which rhetorical slot of a
# paper it belongs to. The research briefs carry a ``genre`` field whose vocabulary
# matches that move vocabulary exactly. Nothing used either: ranking was pure
# embedding cosine to the human side, which is topical.
#
# Topical selection is the documented worst case for exemplars. Wang et al. 2025
# (arXiv:2509.14543, 400+ authors, 40k+ generations) ablated it directly and found
# content-similarity exemplar selection performs SUBSTANTIALLY WORSE than random
# same-author selection: CCAT50 89.72 -> 81.05, Enron 69.33 -> 36.00, blogs 43.93
# -> 22.13. Retrieving a limitations paragraph when writing a limitations paragraph
# is the obvious alternative and, as far as a 2026-08-28 literature sweep found,
# has never been published. This is that experiment.
_MOVE_KEYWORDS = {
    "abstract": ("abstract",),
    "gap_statement": ("gap", "understudied", "unexplored"),
    "motivation": ("motivat", "why this matters"),
    "related_work": ("related work", "prior work", "literature review"),
    "contribution_statement": ("contribution", "we contribute"),
    "methods": ("method", "procedure", "protocol", "we ran", "design"),
    "methods_procedure": ("procedure", "step by step"),
    "methods_notation": ("notation", "formal", "we define"),
    "results": ("result", "we find", "we found", "finding"),
    "results_reporting": ("report the", "reporting"),
    "interpretation": ("interpret", "what this means", "explain the"),
    "discussion": ("discussion", "discuss"),
    "discussion_implications": ("implication", "what follows"),
    "limitations": ("limitation", "caveat", "threat to validity", "we did not"),
    "conclusion": ("conclusion", "conclude", "in closing"),
    "figure_caption": ("figure", "caption"),
    "table_description": ("table",),
    "reviewer_response": ("reviewer", "rebuttal", "response to"),
}


def infer_move(task: str) -> str | None:
    """Which rhetorical slot a brief is asking for, or None when it does not say.

    Deliberately keyword-based rather than a model call: this runs inside kit
    assembly on every compose, and a wrong guess costs a worse demo rather than a
    wrong answer. Longest keyword wins so "methods_procedure" beats "methods".
    """
    low = (task or "").lower()
    best, best_len = None, 0
    for move, keys in _MOVE_KEYWORDS.items():
        for kw in keys:
            if kw in low and len(kw) > best_len:
                best, best_len = move, len(kw)
    return best


def transform_demos(query_text: str, k: int, profile: config.Profile,
                    move: str | None = None) -> list[dict]:
    """Contrastive AI->author rewrite pairs for the query.

    When ``move`` is given (or inferable from the query), pairs carrying that move
    are ranked ahead of the rest, and the remainder backfills to ``k`` by style
    similarity. The corpus is small -- 26 research pairs over 18 moves, so most
    moves hold one or two -- which is exactly why this is a re-ranking rather than
    a filter: an exact-match filter would usually return fewer demos than asked for.

    Returns [] when the profile ships no ``pairs.jsonl``.
    """
    if not profile.pairs_path or not profile.pairs_path.exists():
        return []
    pairs = _load_pairs(profile.pairs_path)
    if not pairs:
        return []
    human_vecs = embed.encode([p["human_text"] for p in pairs], backend=profile.embed_backend)
    qv = embed.embed_one(query_text, backend=profile.embed_backend)
    sims = human_vecs @ qv

    want = move or infer_move(query_text)
    if want:
        # Rank matched pairs first, each block still ordered by similarity.
        matched = [i for i, p in enumerate(pairs) if p.get("move") == want]
        rest = [i for i in np.argsort(-sims) if i not in set(matched)]
        matched.sort(key=lambda i: -sims[i])
        order = (matched + list(rest))[: max(1, k)]
    else:
        order = np.argsort(-sims)[: max(1, k)]
    return [pairs[i] for i in order]
