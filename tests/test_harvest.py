"""Transcript harvesting: chain splitting, paste cleaning, labelling, idempotency."""
from __future__ import annotations

import json

import pytest

from mimesis_voice import harvest


# --- revision chains ----------------------------------------------------------


def _t(kind, ts, **payload):
    return (kind, ts, payload)


def test_same_piece_and_new_piece_thresholds_bracket_the_measured_gap():
    """Measured on real transcripts: same piece 0.545-0.690, different 0.020-0.040."""
    assert harvest.SAME_PIECE_MIN < 0.545
    assert harvest.SAME_PIECE_MIN > 0.040
    assert harvest.IDENTICAL_MAX > 0.690


def test_identical_rescrub_is_not_an_edit(tmp_path):
    """Re-scrubbing the same text to verify a fix must not become a repair pair."""
    body = "The threshold discards signal below it. " * 12
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _tool_rec("scrub", body, "2026-01-01T00:00:00Z"),
        _tool_rec("scrub", body, "2026-01-01T00:05:00Z"),
    ]), encoding="utf-8")
    h = harvest.mine_session(path, assume_voice="research")
    assert h.repair_pairs == []


def test_revision_of_the_same_piece_becomes_a_repair_pair(tmp_path):
    a = "The threshold discards signal below it. " * 12
    b = a.replace("discards signal below it", "removes the signal underneath it")
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _tool_rec("scrub", a, "2026-01-01T00:00:00Z"),
        _tool_rec("scrub", b, "2026-01-01T00:05:00Z"),
    ]), encoding="utf-8")
    h = harvest.mine_session(path, assume_voice="research")
    assert len(h.repair_pairs) == 1
    pair = h.repair_pairs[0]
    assert pair["origin"] == "model-repair"
    assert pair["ai_text"].startswith("The threshold discards")
    assert "removes the signal" in pair["human_text"]


def test_unrelated_drafts_start_a_new_chain(tmp_path):
    a = "The threshold discards signal below it. " * 12
    b = "Morning light on the pool and nobody in it yet. " * 12
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _tool_rec("scrub", a, "2026-01-01T00:00:00Z"),
        _tool_rec("scrub", b, "2026-01-01T00:05:00Z"),
    ]), encoding="utf-8")
    h = harvest.mine_session(path, assume_voice="creative")
    assert h.repair_pairs == []
    assert len(h.accept_candidates) == 2


# --- paste preamble -----------------------------------------------------------


def test_strip_paste_preamble_removes_the_chat_lead_in():
    """The first author pair ever harvested carried an instruction as its target.

    Left in, "made a few edits, this is the final version:" becomes the model's
    idea of how the author opens a piece, which is the first-person meta-
    announcement the ledger bans.
    """
    draft = ("Nobody in the town said much about the drought. The river was low "
             "that year and the boats stayed tied up along the bank, and the ferry "
             "stopped running.")
    pasted = ("made a few edits- think this is the final version: The river was low "
              "that year and the boats stayed tied up along the bank well past May, "
              "and the ferry stopped running.")
    out = harvest.strip_paste_preamble(draft, pasted)
    assert not out.lower().startswith("made a few edits")
    assert out.startswith("The river was low")


def test_strip_paste_preamble_keeps_text_with_no_preamble():
    draft = "The river was low that year and the boats stayed tied up."
    pasted = "The river was low that year and the boats stayed tied up past May."
    assert harvest.strip_paste_preamble(draft, pasted).startswith("The river")


# --- labelling ----------------------------------------------------------------


@pytest.mark.parametrize("reaction,expected", [
    ("Try again", "reject"),
    ("this is all dog shit honestly", "reject"),
    ("theres some Ai tells in there", "reject"),
    ("yeah that works, good sentence actually", "accept"),
    ("okay now do the figures", "unlabeled"),
    ("", "unlabeled"),
])
def test_label_reaction(reaction, expected):
    assert harvest.label_reaction(reaction) == expected


def test_mixed_signal_stays_unlabeled():
    """A wrong accept feeds the fingerprint, so ambiguity must not resolve to accept."""
    assert harvest.label_reaction("i like this but the ai tells are terrible") == "unlabeled"


# --- writing ------------------------------------------------------------------


def test_append_new_is_idempotent_by_sha(tmp_path):
    p = tmp_path / "pairs.jsonl"
    recs = [{"ai_text": "a", "human_text": "b", "harvest_sha": "deadbeef"}]
    assert harvest._append_new(p, recs) == 1
    assert harvest._append_new(p, recs) == 0
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_append_new_does_not_scaffold_an_empty_file(tmp_path):
    p = tmp_path / "pairs.jsonl"
    assert harvest._append_new(p, []) == 0
    assert not p.exists()


# --- helpers ------------------------------------------------------------------


def _tool_rec(tool: str, text: str, ts: str) -> dict:
    name = {"scrub": "mcp__mimesis__scrub_ai_footprint",
            "gate": "mcp__mimesis__gate_candidates"}[tool]
    inp = {"text": text} if tool == "scrub" else {"candidates": [text]}
    return {
        "timestamp": ts,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_x", "name": name, "input": inp},
        ]},
    }
