"""Recover the supervised signal that the engine has been throwing away.

Every voiced draft produced on the MCP path passes through ``scrub_ai_footprint``
or ``gate_candidates``, so its full text is written into the Claude Code session
transcript as a tool-call argument. The engine logged only a SHA-1 prefix of it
(``server._log_event``), so the drafts were never recoverable from the profile.

Measured against ``~/.claude/projects`` on 2026-09-03:

===========================================  ======
scrub calls carrying a full draft                484
gate slates (271 candidates over 110 calls)      110
accepted records on disk                          20
edit pairs on disk                                 6
===========================================  ======

Roughly 755 graded drafts existed and 26 records survived. This module reads the
transcripts back and reconstructs what the profiles should have been recording.

Three kinds of record come out, and they are kept apart on purpose because they
are not equally trustworthy:

``repair`` pairs
    Consecutive scrubs of the same piece inside one session: the model fixed its
    own draft after a flag. Useful for teaching the repair move, but the *after*
    side was written by the model, not the author. Never treat these as author
    prose. They do not feed the fingerprint.

``author-edit`` pairs
    The author pasted a revised version back into the conversation. The after
    side is genuinely his. This is the gold signal the accept loop was designed
    to capture and never got.

``slate`` preferences
    A gate slate where the discriminator picked a winner. Winner-over-loser is a
    real preference pair with no human judgment required, because the gate
    labelled it at the time.

Accept candidates are staged rather than applied. An accept changes the
fingerprint calibration, so it needs a human verdict; a negative pair does not
and lands directly.
"""
from __future__ import annotations

import difflib
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import accepted as accepted_mod
from . import config

# --- tuning -------------------------------------------------------------------

# Revision-chain thresholds, measured on the busiest real transcript
# (one research session, 270 scrub calls). Consecutive scrubs of the same piece
# scored 0.545 and 0.690; consecutive scrubs of different pieces scored 0.020 to
# 0.040. The gap is wide enough that the exact cut barely matters, so these are
# set conservatively away from both clusters rather than tuned.
SAME_PIECE_MIN = 0.50   # below this, a new piece has started
IDENTICAL_MAX = 0.995   # at or above, the same text was re-scrubbed; not an edit

# An author edit pasted back into chat has to be substantial to be worth a pair,
# and has to still resemble the draft it came from or it is a different piece.
AUTHOR_EDIT_MIN_WORDS = 40
AUTHOR_EDIT_MIN_RATIO = 0.45

_WORD = re.compile(r"[A-Za-z']+")

# The author's own rejection vocabulary, taken from docs/AUTHOR_FEEDBACK.md and
# from transcripts, not invented. These label a draft's fate only when they land
# in the user turn immediately after it. Everything else stays unlabeled: his
# persona record says continuation is the approval, and silence is too weak a
# signal to spend a fingerprint slot on.
_REJECT_MARKERS = (
    "dog shit", "dogshit", "terrible", "awful", "garbage", "trash",
    "i hate", "hate how", "hate it", "doesn't flow", "does not flow",
    "self explain", "self-explain", "overexplain", "over explain",
    "ai tell", "ai-tell", "sounds like ai", "sounds ai", "so ai",
    "too formal", "doesn't sound like me", "does not sound like me",
    "not my voice", "feels amiss", "doesn't hit", "does not hit",
    "not right", "no one talks like that", "pisses me off", "not human",
    "rewrite", "redo it", "try again", "start over", "scrap",
)
_ACCEPT_MARKERS = (
    "that's good", "thats good", "this is good", "looks good", "sounds good",
    "i like that", "i like this", "that works", "perfect", "nice one",
    "keep that", "keep it", "use that", "ship it", "that's it", "thats it",
    "yeah good", "good sentence", "distinctly human", "feels like my writing",
)


# --- records ------------------------------------------------------------------


@dataclass
class Draft:
    """One draft text observed in a transcript, with where it came from."""

    text: str
    ts: str
    voice: str | None
    tool: str
    session: str
    project: str
    task: str | None = None

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def words(self) -> int:
        return len(_WORD.findall(self.text))


@dataclass
class Harvest:
    """What one pass over the transcripts found."""

    repair_pairs: list[dict] = field(default_factory=list)
    author_pairs: list[dict] = field(default_factory=list)
    slate_prefs: list[dict] = field(default_factory=list)
    accept_candidates: list[dict] = field(default_factory=list)
    unattributed: int = 0
    transcripts_read: int = 0
    drafts_seen: int = 0
    # Which project directory each voice-less call came from. A third of the
    # archive passed no voice= argument, and dropping it loses more than guessing
    # would; showing the breakdown lets the routing decision be made on evidence.
    unattributed_by_project: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "repair_pairs": len(self.repair_pairs),
            "author_pairs": len(self.author_pairs),
            "slate_prefs": len(self.slate_prefs),
            "accept_candidates": len(self.accept_candidates),
        }


# --- transcript reading -------------------------------------------------------


def transcript_root() -> Path:
    return Path(os.path.expanduser("~/.claude/projects"))


def iter_transcripts(root: Path | None = None):
    """Every Claude Code session transcript, newest last.

    Skips the ``memory`` subtrees: those hold auto-memory markdown, not sessions.
    """
    root = root or transcript_root()
    if not root.exists():
        return
    for fp in sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True)):
        parts = Path(fp).parts
        if "memory" in parts:
            continue
        yield Path(fp)


def _bump(d: dict, key: str) -> None:
    d[key] = d.get(key, 0) + 1


def _blocks(rec: dict) -> list:
    content = (rec.get("message") or {}).get("content")
    return content if isinstance(content, list) else []


def _user_text(rec: dict) -> str:
    """Plain text of a user turn, or '' when it is a tool result or notification."""
    msg = rec.get("message") or {}
    if msg.get("role") != "user":
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        return ""
    # Harness plumbing, not the author speaking.
    if text.lstrip().startswith(("<task-notification>", "<system-reminder>",
                                "<local-command", "Caveat:")):
        return ""
    return text.strip()


def read_session(path: Path) -> tuple[list, str, str]:
    """Ordered (kind, ts, payload) events for one transcript.

    Kinds: ``scrub`` and ``gate`` carry draft text; ``user`` carries the author's
    reply, which is what grades the draft before it.
    """
    events: list[tuple[str, str, dict]] = []
    session = path.stem
    project = path.parent.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], session, project
    for line in raw:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("timestamp") or ""
        ut = _user_text(rec)
        if ut:
            events.append(("user", ts, {"text": ut}))
        if "mimesis" not in line:
            continue
        for b in _blocks(rec):
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name = b.get("name", "")
            if "mimesis" not in name:
                continue
            inp = b.get("input") or {}
            voice = inp.get("voice")
            if name.endswith("scrub_ai_footprint") and (inp.get("text") or "").strip():
                events.append(("scrub", ts, {"text": inp["text"], "voice": voice}))
            elif name.endswith("gate_candidates"):
                cands = [c for c in (inp.get("candidates") or []) if (c or "").strip()]
                if cands:
                    events.append(("gate", ts, {
                        "candidates": cands, "voice": voice,
                        "task": inp.get("task"), "id": b.get("id"),
                    }))
    return events, session, project


def gate_results(path: Path) -> dict[str, int]:
    """Map tool_use_id -> winning candidate index, read from the gate's own report.

    The winner is parsed out of the report text rather than recomputed. The gate
    decided with the calibration that was live at the time; re-scoring now would
    use today's fingerprint and silently relabel history.
    """
    out: dict[str, int] = {}
    pat = re.compile(r"winner:\s*candidate\s*(\d+)", re.IGNORECASE)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in raw:
        if "winner" not in line.lower():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for b in _blocks(rec):
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                continue
            content = b.get("content")
            text = content if isinstance(content, str) else " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ) if isinstance(content, list) else ""
            m = pat.search(text or "")
            if m and b.get("tool_use_id"):
                out[b["tool_use_id"]] = int(m.group(1)) - 1  # report is 1-based
    return out


# --- labelling ----------------------------------------------------------------


def label_reaction(text: str) -> str:
    """Conservative verdict on the user turn that followed a draft.

    Returns ``reject``, ``accept`` or ``unlabeled``. Deliberately biased toward
    ``unlabeled``: a wrong accept feeds the fingerprint, and the ledger's whole
    complaint is that the gate already drifts toward the wrong target.
    """
    low = (text or "").lower()
    if not low.strip():
        return "unlabeled"
    hit_reject = any(m in low for m in _REJECT_MARKERS)
    hit_accept = any(m in low for m in _ACCEPT_MARKERS)
    if hit_reject and not hit_accept:
        return "reject"
    if hit_accept and not hit_reject:
        return "accept"
    return "unlabeled"


def _looks_like_author_paste(user_text: str, draft: str) -> bool:
    """Did the author paste an edited version of the draft back into chat?"""
    if len(_WORD.findall(user_text)) < AUTHOR_EDIT_MIN_WORDS:
        return False
    ratio = difflib.SequenceMatcher(None, draft, user_text, autojunk=False).ratio()
    return AUTHOR_EDIT_MIN_RATIO <= ratio < IDENTICAL_MAX


# A paste almost always arrives with a sentence of chat in front of it: "made a
# few edits- think this is the final version: The river was low...". Left in, that instruction becomes the target side of a transform
# pair and teaches the model to open a piece by announcing what it is about, which
# is the first-person meta-announcement the ledger already bans. Found in the very
# first author pair this harvester recovered, so it is not hypothetical.
_PREAMBLE_CUT = re.compile(r"[.:\n]\s+")


def strip_paste_preamble(draft: str, pasted: str, min_block: int = 60) -> str:
    """Drop the conversational lead-in from a pasted revision.

    Aligns the paste against the draft it revises and cuts at the boundary
    immediately before the first substantial shared block. Falls back to the whole
    paste when no block is large enough to locate a boundary confidently.
    """
    sm = difflib.SequenceMatcher(None, draft, pasted, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= min_block]
    if not blocks:
        return pasted.strip()
    start = blocks[0].b
    head = pasted[:start]
    # Back up only as far as the nearest sentence or line boundary, so a preamble
    # sharing a line with the content is still removed.
    cuts = [m.end() for m in _PREAMBLE_CUT.finditer(head)]
    return pasted[(cuts[-1] if cuts else start):].strip()


# --- mining -------------------------------------------------------------------


def mine_session(path: Path, only_voice: str | None = None,
                 assume_voice: str | None = None) -> Harvest:
    """Reconstruct pairs, preferences and accept candidates from one transcript."""
    h = Harvest()
    events, session, project = read_session(path)
    if not events:
        return h
    # Count a transcript only when it actually carries voiced drafts. Nearly every
    # session has user turns, so counting those would report the whole archive.
    if not any(k in ("scrub", "gate") for k, _, _ in events):
        return h
    h.transcripts_read = 1
    winners = gate_results(path)

    def resolve(voice: str | None) -> str | None:
        if voice:
            return voice
        if assume_voice:
            return assume_voice
        return None

    chain: list[Draft] = []   # consecutive scrubs of one evolving piece
    pending_task: str | None = None

    def flush_chain(reaction: str | None):
        """Close a revision chain: emit repair pairs and one accept candidate."""
        if not chain:
            return
        for a, b in zip(chain, chain[1:]):
            ratio = difflib.SequenceMatcher(None, a.text, b.text, autojunk=False).ratio()
            if not (SAME_PIECE_MIN <= ratio < IDENTICAL_MAX):
                continue
            h.repair_pairs.append({
                "ai_text": a.text, "human_text": b.text,
                "move": "repair", "origin": "model-repair", "granularity": "document",
                "voice": b.voice, "similarity": round(ratio, 4),
                "timestamp": b.ts, "session": session, "project": project,
                "harvest_sha": hashlib.sha1(
                    (a.sha + b.sha).encode("utf-8")).hexdigest()[:16],
            })
        last = chain[-1]
        h.accept_candidates.append({
            "text": last.text, "task": last.task or pending_task,
            "voice": last.voice, "timestamp": last.ts,
            "session": session, "project": project,
            "revisions": len(chain), "words": last.words,
            "reaction": (reaction or "")[:800],
            "label": label_reaction(reaction or ""),
            "harvest_sha": last.sha,
        })

    for i, (kind, ts, payload) in enumerate(events):
        if kind == "user":
            utext = payload["text"]
            # An author paste that resembles the live draft is a real author edit.
            if chain and _looks_like_author_paste(utext, chain[-1].text):
                utext = strip_paste_preamble(chain[-1].text, utext)
                h.author_pairs.append({
                    "ai_text": chain[-1].text, "human_text": utext.strip(),
                    "move": "author-edit", "origin": "author-edit",
                    "granularity": "document", "voice": chain[-1].voice,
                    "timestamp": ts, "session": session, "project": project,
                    "harvest_sha": hashlib.sha1(
                        (chain[-1].sha + utext[:200]).encode("utf-8")).hexdigest()[:16],
                })
                for k, sp in enumerate(accepted_mod.segment_pairs(chain[-1].text, utext)):
                    h.author_pairs.append({
                        "ai_text": sp["ai_text"], "human_text": sp["human_text"],
                        "move": "author-edit", "origin": "author-edit",
                        "granularity": "segment", "voice": chain[-1].voice,
                        "timestamp": ts, "session": session, "project": project,
                        "harvest_sha": hashlib.sha1(
                            (sp["ai_text"][:120] + sp["human_text"][:120]).encode("utf-8")
                        ).hexdigest()[:16],
                    })
            else:
                pending_task = utext[:400]
            continue

        if kind == "gate":
            voice = resolve(payload.get("voice"))
            if voice is None:
                h.unattributed += 1
                _bump(h.unattributed_by_project, project)
            if only_voice and voice != only_voice:
                continue
            cands = payload["candidates"]
            h.drafts_seen += len(cands)
            win = winners.get(payload.get("id") or "")
            if win is not None and 0 <= win < len(cands) and len(cands) > 1:
                for j, c in enumerate(cands):
                    if j == win:
                        continue
                    h.slate_prefs.append({
                        "chosen": cands[win], "rejected": c,
                        "origin": "gate-slate", "voice": voice,
                        "task": payload.get("task"), "timestamp": ts,
                        "session": session, "project": project,
                        "harvest_sha": hashlib.sha1(
                            (cands[win][:120] + c[:120]).encode("utf-8")).hexdigest()[:16],
                    })
            continue

        # kind == "scrub"
        voice = resolve(payload.get("voice"))
        if voice is None:
            h.unattributed += 1
            _bump(h.unattributed_by_project, project)
        if only_voice and voice != only_voice:
            continue
        h.drafts_seen += 1
        d = Draft(text=payload["text"], ts=ts, voice=voice, tool="scrub",
                  session=session, project=project, task=pending_task)
        if not chain:
            chain = [d]
            continue
        ratio = difflib.SequenceMatcher(None, chain[-1].text, d.text, autojunk=False).ratio()
        if ratio >= IDENTICAL_MAX:
            continue  # same text re-scrubbed to verify a fix
        if ratio >= SAME_PIECE_MIN:
            chain.append(d)
            continue
        # A different piece started. Grade the finished one on the next user turn.
        reaction = ""
        for k in range(i + 1, min(i + 6, len(events))):
            if events[k][0] == "user":
                reaction = events[k][2]["text"]
                break
        flush_chain(reaction)
        chain = [d]

    if chain:
        reaction = ""
        for k in range(len(events) - 1, -1, -1):
            if events[k][0] == "user":
                reaction = events[k][2]["text"]
                break
        flush_chain(reaction)
    return h


def mine_all(only_voice: str | None = None, assume_voice: str | None = None,
             since: str | None = None, root: Path | None = None,
             project: str | None = None, progress=None) -> Harvest:
    """Walk every transcript and merge the findings.

    ``project`` restricts the walk to transcript directories whose name contains
    that substring, which is how a voice-less run is scoped before
    ``assume_voice`` routes it.
    """
    total = Harvest()
    cutoff = None
    if since:
        try:
            cutoff = datetime.fromisoformat(since).timestamp()
        except ValueError:
            cutoff = None
    for path in iter_transcripts(root):
        if cutoff and path.stat().st_mtime < cutoff:
            continue
        if project and project.lower() not in path.parent.name.lower():
            continue
        h = mine_session(path, only_voice=only_voice, assume_voice=assume_voice)
        if not h.transcripts_read:
            continue
        total.transcripts_read += h.transcripts_read
        total.drafts_seen += h.drafts_seen
        total.unattributed += h.unattributed
        for k, v in h.unattributed_by_project.items():
            total.unattributed_by_project[k] = total.unattributed_by_project.get(k, 0) + v
        total.repair_pairs.extend(h.repair_pairs)
        total.author_pairs.extend(h.author_pairs)
        total.slate_prefs.extend(h.slate_prefs)
        total.accept_candidates.extend(h.accept_candidates)
        if progress:
            progress(path, h)
    return total


# --- writing ------------------------------------------------------------------


def harvest_dir(profile: config.Profile) -> Path:
    return profile.root / "harvest"


def _existing_shas(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("harvest_sha"):
            out.add(obj["harvest_sha"])
    return out


def _append_new(path: Path, records: list[dict]) -> int:
    """Append only records whose harvest_sha is not already on disk."""
    if not records:
        return 0  # do not scaffold an empty file for a voice with no harvest
    seen = _existing_shas(path)
    added = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            sha = r.get("harvest_sha")
            if sha and sha in seen:
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if sha:
                seen.add(sha)
            added += 1
    return added


def apply_harvest(profile: config.Profile, h: Harvest) -> dict[str, int]:
    """Write one voice's harvest to disk. Idempotent by ``harvest_sha``.

    Negative and preference signal lands directly: it never touches the
    fingerprint. Accept candidates are staged in ``harvest/accept_pending.jsonl``
    and require ``mimesis harvest --accept`` after review, because an accept is
    folded into the calibration and a wrong one moves the target.
    """
    d = harvest_dir(profile)
    slug = profile.slug
    keep = lambda rs: [r for r in rs if r.get("voice") == slug]  # noqa: E731
    return {
        "author_pairs": _append_new(d / "author_pairs.jsonl", keep(h.author_pairs)),
        "repair_pairs": _append_new(d / "repair_pairs.jsonl", keep(h.repair_pairs)),
        "slate_prefs": _append_new(d / "slate_preferences.jsonl", keep(h.slate_prefs)),
        "accept_pending": _append_new(d / "accept_pending.jsonl", keep(h.accept_candidates)),
    }


def promote_accepts(profile: config.Profile, labels: tuple[str, ...] = ("accept",),
                    min_words: int = 25) -> int:
    """Move reviewed accept candidates into the real accepted set.

    Only candidates whose ``label`` is in ``labels`` are promoted, so the default
    promotes nothing that the marker vocabulary did not positively confirm.
    """
    pending = harvest_dir(profile) / "accept_pending.jsonl"
    if not pending.exists():
        return 0
    promoted = 0
    already = {
        hashlib.sha1(r.get("text", "").encode("utf-8")).hexdigest()[:16]
        for r in accepted_mod.load_accepted(profile)
    }
    for line in pending.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("label") not in labels:
            continue
        text = (rec.get("text") or "").strip()
        if len(_WORD.findall(text)) < min_words:
            continue
        if hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] in already:
            continue
        accepted_mod.record_accept(
            profile, text, task=rec.get("task"), source="harvest",
            timestamp=rec.get("timestamp"),
        )
        already.add(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
        promoted += 1
    return promoted


def author_pairs_for_demos(profile: config.Profile) -> Path:
    """Path the compose kit can read harvested author pairs from."""
    return harvest_dir(profile) / "author_pairs.jsonl"


# --- live capture -------------------------------------------------------------


def record_draft(profile: config.Profile, text: str, tool: str,
                 task: str | None = None, verdict: str | None = None,
                 timestamp: str | None = None) -> None:
    """Persist a draft at the moment it is scored, not months later from a transcript.

    ``server._log_event`` records a SHA-1 prefix, which identifies a draft without
    preserving it, so every draft this engine has produced had to be recovered
    from Claude Code session files. This writes the text next to the profile as it
    happens, so the transcript walk becomes a backfill rather than the only path.

    Never raises: it annotates a tool call and must not be able to break one.
    """
    try:
        text = (text or "").strip()
        if not text:
            return
        rec = {
            "text": text,
            "tool": tool,
            "task": (task or "")[:400] or None,
            "verdict": (verdict or "")[:200] or None,
            "timestamp": timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
            "words": len(_WORD.findall(text)),
            "harvest_sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
        }
        _append_new(harvest_dir(profile) / "drafts.jsonl", [rec])
    except Exception:
        pass
