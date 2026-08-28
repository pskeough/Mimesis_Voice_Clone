"""Tests for blind rating packets.

The load-bearing property is that the packet does not leak which arm produced
which text. A leak is worse than no packet: it yields confident data pointing the
wrong way. So the arm-name-absence check runs against the rendered HTML, not
against the function's intent.
"""
from __future__ import annotations

import json
import types

import pytest

from mimesis_voice import blind, preference as pref


@pytest.fixture
def profile(tmp_path):
    return types.SimpleNamespace(root=str(tmp_path), slug="testvoice")


# Deliberately distinct lengths. An earlier version made all three the same
# length, which silently turned the length scorer into a constant and made the
# end-to-end benchmark test look like a code failure.
CANDS = {
    "opus_bare": "Alpha.",
    "qwen_mimesis": "Beta draft text here.",
    "human": "Gamma draft text, rather longer than the others in this fixture.",
}


def test_packet_writes_html_and_key(tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    assert p.html_path.exists() and p.key_path.exists()
    assert set(p.key.values()) == set(CANDS)
    assert len(p.key) == 3


def test_html_never_names_an_arm(tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    doc = p.html_path.read_text(encoding="utf-8")
    for arm in CANDS:
        assert arm not in doc, f"packet leaked arm name {arm!r}"


def test_all_texts_present(tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    doc = p.html_path.read_text(encoding="utf-8")
    for text in CANDS.values():
        assert text in doc


def test_shuffle_is_not_identity(tmp_path):
    """The regression this exists for: a hash order that degenerated to the
    input order, presenting as a position control while providing none."""
    for tid in ("t1", "t2", "t3", "t4", "t5"):
        p = blind.packet(tmp_path, tid, CANDS, voice="v")
        letters_in_input_order = [p.key[l] for l in sorted(p.key)]
        assert letters_in_input_order != list(CANDS), f"{tid} degenerated"


def test_shuffle_is_deterministic(tmp_path):
    a = blind.packet(tmp_path, "same", CANDS, voice="v").key
    b = blind.packet(tmp_path, "same", CANDS, voice="v").key
    assert a == b


def test_single_candidate_does_not_crash(tmp_path):
    p = blind.packet(tmp_path, "solo", {"only": "text"}, voice="v")
    assert p.key == {"A": "only"}


def test_ingest_roundtrip(profile, tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    letters = sorted(p.key)                       # A, B, C
    ranking = [letters[2], letters[0], letters[1]]
    j = blind.ingest(profile, p, ranking, candidates=CANDS,
                     grades={letters[2]: "A"},
                     rationales={letters[2]: "best rhythm"})
    assert [i.rank for i in j.items] == [1, 2, 3]
    assert j.items[0].arm == p.key[letters[2]]
    assert j.items[0].grade == "A"
    assert len(pref.load(profile)) == 1


def test_ingest_partial_ranking_drops_unranked(profile, tmp_path):
    """Silence is not a verdict; an unranked candidate must not become 'worst'."""
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    letters = sorted(p.key)
    j = blind.ingest(profile, p, [letters[0], letters[1]], candidates=CANDS)
    assert len(j.items) == 2


def test_ingest_rejects_unknown_letter(profile, tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    with pytest.raises(ValueError):
        blind.ingest(profile, p, ["Z"], candidates=CANDS)


def test_ingest_requires_candidates(profile, tmp_path):
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    with pytest.raises(ValueError):
        blind.ingest(profile, p, ["A"])


def test_ingested_judgment_feeds_benchmark(profile, tmp_path):
    """End to end: packet -> ranking -> store -> benchmark produces a number."""
    p = blind.packet(tmp_path, "t1", CANDS, voice="testvoice")
    order = sorted(p.key, key=lambda l: len(CANDS[p.key[l]]))
    blind.ingest(profile, p, order, candidates=CANDS)
    r = pref.benchmark(profile, lambda t: len(t), name="len")
    assert r.pairs == 3
    assert r.accuracy == 1.0
