"""Inference-time learning from accept/edit signal (Upgrade 3).

No weight training. The loop learns from two cheap signals the author already
produces:

* **Accept** -- a draft the author keeps. Appended to a per-voice accepted set
  that (a) gets extra retrieval weight as high-priority exemplars in the compose
  kit and (b) folds into the fingerprint calibration, recency-weighted, so the
  gate steers new drafts toward what the author actually keeps.
* **Edit** -- the author's changes to a draft. The pre-edit phrasing is negative
  signal (what to avoid), the post-edit is the target. We turn each edit into
  contrastive transform pairs -- the same ``ai_text``/``human_text`` anchor the
  research voice already uses -- both whole-draft and per-changed-segment.

The edit-diff is modelled on the *tracked-changes* idea (a before/after span
diff) but implemented fresh here on stdlib ``difflib``; nothing is ported from
any other project. Storage is two JSONL files under ``profiles/<slug>/accepted/``.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z']+")


# --- paths --------------------------------------------------------------------


def accepted_dir(profile: config.Profile) -> Path:
    return profile.root / "accepted"


def accepted_path(profile: config.Profile) -> Path:
    return accepted_dir(profile) / "accepted.jsonl"


def edit_pairs_path(profile: config.Profile) -> Path:
    return accepted_dir(profile) / "edit_pairs.jsonl"


def has_accepted(profile: config.Profile) -> bool:
    return accepted_path(profile).exists() or edit_pairs_path(profile).exists()


# --- io -----------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_accepted(profile: config.Profile) -> list[dict]:
    """Accepted records, oldest first (append order = recency order)."""
    return _read_jsonl(accepted_path(profile))


def load_edit_pairs(profile: config.Profile) -> list[dict]:
    return _read_jsonl(edit_pairs_path(profile))


def accepted_texts(profile: config.Profile) -> list[str]:
    """Accepted draft texts, oldest first (newest last -> highest recency weight)."""
    return [r["text"] for r in load_accepted(profile) if r.get("text", "").strip()]


# --- capture ------------------------------------------------------------------


def _next_id(records: list[dict], prefix: str) -> str:
    return f"{prefix}_{len(records):03d}"


def record_accept(
    profile: config.Profile, text: str, task: str | None = None,
    source: str | None = None, timestamp: str | None = None,
) -> dict:
    """Append one accepted draft to the voice's accepted set."""
    existing = load_accepted(profile)
    rec = {
        "id": _next_id(existing, "acc"),
        "text": text.strip(),
        "task": task,
        "source": source,
        "timestamp": timestamp,
    }
    _append_jsonl(accepted_path(profile), rec)
    return rec


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


def segment_pairs(before: str, after: str, min_words: int = 4) -> list[dict]:
    """Per-changed-span contrastive pairs from an edit (the tracked-changes diff).

    Sentence-align before/after with difflib; every ``replace`` opcode yields one
    (avoid -> target) pair. Pure insertions/deletions are skipped: a contrastive
    pair needs both a before and an after side to teach a rewrite move.
    """
    b_sents = _sentences(before)
    a_sents = _sentences(after)
    sm = difflib.SequenceMatcher(a=b_sents, b=a_sents, autojunk=False)
    pairs: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        avoid = " ".join(b_sents[i1:i2]).strip()
        target = " ".join(a_sents[j1:j2]).strip()
        if not avoid or not target or avoid == target:
            continue
        if len(_WORD.findall(target)) < min_words:
            continue
        pairs.append({"ai_text": avoid, "human_text": target})
    return pairs


def record_edit(
    profile: config.Profile, before: str, after: str, task: str | None = None,
    timestamp: str | None = None, accept_after: bool = True,
) -> dict:
    """Capture an edit: keep the post-edit text and mine contrastive pairs from it.

    Appends the whole before->after as a document-level pair plus one pair per
    changed segment, and (by default) records the post-edit text as accepted, so a
    single ``accept --from`` gives both the positive exemplar and the negative
    signal.
    """
    before, after = before.strip(), after.strip()
    existing = load_edit_pairs(profile)
    added = 0
    # document-level pair (only if the whole thing actually changed)
    if before and after and before != after:
        _append_jsonl(edit_pairs_path(profile), {
            "id": _next_id(existing, "edit"),
            "ai_text": before, "human_text": after,
            "move": "edit", "granularity": "document",
            "task": task, "timestamp": timestamp,
        })
        added += 1
    # segment-level pairs
    for k, sp in enumerate(segment_pairs(before, after)):
        _append_jsonl(edit_pairs_path(profile), {
            "id": f"{_next_id(existing, 'edit')}_seg{k:02d}",
            "ai_text": sp["ai_text"], "human_text": sp["human_text"],
            "move": "edit", "granularity": "segment",
            "task": task, "timestamp": timestamp,
        })
        added += 1
    acc = record_accept(profile, after, task=task, source="edit", timestamp=timestamp) if accept_after else None
    return {"pairs_added": added, "accepted": acc}


# --- retrieval (extra weight for what the author kept) ------------------------


def _rank_by_relevance(query: str, texts: list[str], profile: config.Profile) -> list[int]:
    """Order indices of ``texts`` by cosine to ``query`` under the profile backend,
    with a small recency nudge (later items are newer). Falls back to recency-only."""
    n = len(texts)
    if n == 0:
        return []
    try:
        from . import embed
        import numpy as np

        vecs = embed.encode(texts, backend=profile.embed_backend)
        qv = embed.embed_one(query, backend=profile.embed_backend)
        sims = vecs @ qv
        recency = np.linspace(0.0, 0.15, n)  # newest (last) gets a +0.15 nudge
        blended = sims + recency
        return list(np.argsort(-blended))
    except Exception:
        return list(range(n - 1, -1, -1))  # newest first


def retrieve_accepted(profile: config.Profile, query: str, k: int = 3) -> list[dict]:
    """Top accepted drafts to anchor a new composition (relevance + recency)."""
    recs = load_accepted(profile)
    if not recs:
        return []
    texts = [r["text"] for r in recs]
    order = _rank_by_relevance(query, texts, profile)
    return [recs[i] for i in order[: max(1, k)]]


def demo_pairs(profile: config.Profile, query: str, k: int = 3) -> list[dict]:
    """Edit-derived contrastive demos closest in style to the query.

    Same shape as ``retrieve.transform_demos`` output (``ai_text``/``human_text``/
    ``move``), so the compose kit renders them identically. Prefers segment-level
    pairs (sharper moves) then document-level, ranked by relevance of the target
    (author) side to the query.
    """
    pairs = load_edit_pairs(profile)
    if not pairs:
        return []
    targets = [p.get("human_text", "") for p in pairs]
    order = _rank_by_relevance(query, targets, profile)
    return [pairs[i] for i in order[: max(1, k)]]
