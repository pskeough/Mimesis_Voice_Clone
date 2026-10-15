"""Genre guidance block: present only when configured, scoped to one move, capped.

Added 2026-09-06 after the ML4H rev6 ledger showed the gate passing a Discussion
(fit 0.201, no hard flags) that the author rejected in one line as "a bunch of
nothing". The kit had no block pointed at what a section is for. These tests pin
the three properties that keep the new block from becoming the next burstiness
advisory: absent by default, one card not a bible, and never longer than its cap.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mimesis_voice import config, guidance, retrieve


def _profile(tmp_path: Path, pack: str | None) -> config.Profile:
    root = tmp_path / "voice"
    (root / "data").mkdir(parents=True)
    return config.Profile(
        slug="voice", name="Author", embed_backend="fast", gate={}, formats={},
        anchors={}, whitelist=[], root=root, data_dir=root / "data",
        source_dir=root / "source_documents", db_path=root / "data" / "s.sqlite",
        fingerprint_path=root / "data" / "f.json", scrub_path=root / "data" / "c.json",
        presence_path=root / "data" / "p.json", guidance=pack,
    )


def test_no_pack_means_no_block(tmp_path):
    prof = _profile(tmp_path, None)
    assert guidance.block("write the discussion", prof) == ""


def test_unknown_pack_means_no_block(tmp_path):
    prof = _profile(tmp_path, "haiku")
    assert guidance.block("write the discussion", prof) == ""


def test_research_pack_ships_a_card_for_every_inferable_move():
    moves = set(guidance.available_moves("research"))
    inferable = set(retrieve._MOVE_KEYWORDS)
    missing = inferable - moves
    assert not missing, f"moves with keywords but no card: {sorted(missing)}"


def test_block_carries_common_plus_one_card_only(tmp_path):
    prof = _profile(tmp_path, "research")
    out = guidance.block("write the discussion paragraph for the audit", prof)
    assert out.startswith(guidance.HEADER)
    assert "### This passage is a discussion" in out
    # One card. The related-work card's signature line must not be present.
    assert "Organised by question or approach" not in out


def test_explicit_format_beats_inference(tmp_path):
    prof = _profile(tmp_path, "research")
    out = guidance.block("write the discussion paragraph", prof, fmt="related_work")
    assert "### This passage is a related work" in out
    assert "### This passage is a discussion" not in out


def test_format_accepts_hyphen_and_space(tmp_path):
    prof = _profile(tmp_path, "research")
    assert guidance.resolve_move("x", "related-work", "research") == "related_work"
    assert guidance.resolve_move("x", "Related Work", "research") == "related_work"


def test_uninferable_task_still_gets_common_principles(tmp_path):
    prof = _profile(tmp_path, "research")
    out = guidance.block("write something about the weather", prof)
    assert out.startswith(guidance.HEADER)
    assert "### This passage is a" not in out
    assert "one central contribution" in out


def test_author_layer_is_appended_after_the_shipped_card(tmp_path):
    prof = _profile(tmp_path, "research")
    (prof.root / "guidance").mkdir()
    (prof.root / "guidance" / "discussion.md").write_text("Built from his answers.", encoding="utf-8")
    (prof.root / "guidance" / "_common.md").write_text("No colons in prose.", encoding="utf-8")
    out = guidance.block("write the discussion", prof)
    i_card = out.index("### This passage is a discussion")
    i_author = out.index("Author's own rulings")
    assert i_card < i_author
    assert out.index("No colons in prose.") < out.index("Built from his answers.")


def test_every_shipped_card_is_under_the_cap():
    d = guidance.GUIDANCE_DIR / "research"
    for p in d.glob("*.md"):
        if p.stem in ("SOURCES",):
            continue
        cap = guidance._COMMON_MAX_CHARS if p.stem == "_common" else guidance._CARD_MAX_CHARS
        n = len(p.read_text(encoding="utf-8").strip())
        assert n <= cap, f"{p.name} is {n} chars, cap {cap}: trim the card, do not raise the cap"


def test_cards_do_not_prescribe_phrasing():
    """A card that hands the model words is the failure the scrubber exists to catch."""
    banned_openers = ("use the phrase", "say \"", "write \"", "begin with \"", "for example, \"")
    d = guidance.GUIDANCE_DIR / "research"
    for p in d.glob("*.md"):
        if p.stem == "SOURCES":
            continue
        low = p.read_text(encoding="utf-8").lower()
        for b in banned_openers:
            assert b not in low, f"{p.name} prescribes phrasing: {b!r}"


def test_read_truncates_at_paragraph_break(tmp_path):
    p = tmp_path / "long.md"
    p.write_text("para one.\n\n" + ("x" * 3000), encoding="utf-8")
    out = guidance._read(p, 100)
    assert out == "para one."


def test_kit_carries_structure_block_between_rules_and_anchors(tmp_path, monkeypatch):
    """Integration: build_kit places the block after the voice rules, before anchors."""
    from mimesis_voice import gate
    from mimesis_voice.scrub import ScrubCalibration

    prof = _profile(tmp_path, "research")
    cal = ScrubCalibration(
        banned_words=["delve"], banned_phrases=[], whitelist=[],
        burstiness_floor=5.0, hedge_ceiling=0.5, mean_sentence_len=20.0, n_pieces=10,
    )
    monkeypatch.setattr(gate, "_anchor_block", lambda *a, **k: "## ANCHORS\n(none)")
    kit = gate.build_kit("write the limitations paragraph", prof, cal)
    i_rules = kit.index("## STYLE RULES")
    i_struct = kit.index("## WHAT THIS PASSAGE MUST DO")
    i_anch = kit.index("## ANCHORS")
    assert i_rules < i_struct < i_anch
    assert "### This passage is a limitations" in kit
