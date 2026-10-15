"""First-run behaviour: corpus inspection, the setup guard, and voice activation.

A fresh clone has only the synthetic example voice. These tests pin the three
things that keep a new user from getting confident prose in a voice that is not
theirs: inspection names a corpus that is too small or mixed, every MCP writing
tool returns the intake protocol until a real voice is built, and a new voice
becomes active instead of leaving the demo in place.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import config, intake  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _words(n: int, seed: str = "river") -> str:
    base = (f"The {seed} ran low that summer and the boats stayed tied up along the bank "
            "while the town argued about the weather and the price of grain. ")
    out, wc = [], 0
    while wc < n:
        out.append(base)
        wc += len(base.split())
    return "\n\n".join(out)


def test_inspect_stops_below_the_word_floor(tmp_path):
    for i in range(3):
        (tmp_path / f"p{i}.txt").write_text(_words(200, f"river{i}"), encoding="utf-8")
    rep = intake.inspect([str(tmp_path)])
    assert len(rep.docs) == 3
    assert intake.verdict(rep) == "stop"
    assert any(sev == "STOP" for sev, _ in rep.warnings())


def test_inspect_flags_mixed_forms_and_a_dominant_document(tmp_path):
    for i in range(8):
        (tmp_path / f"post{i}.txt").write_text(_words(150, f"post{i}"), encoding="utf-8")
    for i in range(3):
        (tmp_path / f"chapter{i}.txt").write_text(_words(3500, f"chap{i}"), encoding="utf-8")
    (tmp_path / "book.txt").write_text(_words(30000), encoding="utf-8")
    msgs = " ".join(m for _, m in intake.inspect([str(tmp_path)]).warnings())
    assert "Mixed forms" in msgs
    assert "book.txt" in msgs


def test_inspect_skips_hidden_folders_and_names_unsupported_exports(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "notes.md").write_text(_words(500), encoding="utf-8")
    (tmp_path / "posts.csv").write_text("id,text\n1,hello\n", encoding="utf-8")
    (tmp_path / "essay.md").write_text("---\ndate: 2024-01-01\n---\n\n" + _words(600), encoding="utf-8")
    rep = intake.inspect([str(tmp_path)])
    assert [d.path.name for d in rep.docs] == ["essay.md"]
    assert rep.docs[0].front_matter
    assert any(".csv" in m for _, m in rep.warnings())


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A profiles/ dir holding only a copy of the shipped example voice."""
    prof = tmp_path / "profiles"
    shutil.copytree(REPO / "profiles" / "example", prof / "example",
                    ignore=shutil.ignore_patterns("*.sqlite*", "accepted", "harvest"))
    monkeypatch.setattr(config, "PROFILES_DIR", prof)
    monkeypatch.setattr(config, "STATE_PATH", prof / "state.json")
    return prof


def _fake_built_voice(prof: Path, slug: str) -> None:
    d = prof / slug
    (d / "data").mkdir(parents=True)
    (d / "source_documents").mkdir()
    (d / "config.json").write_text(json.dumps({"author_name": "Jane"}), encoding="utf-8")
    for f in ("fingerprint.json", "scrub_calibration.json"):
        (d / "data" / f).write_text("{}", encoding="utf-8")


def test_fresh_clone_needs_setup_and_every_writing_tool_says_so(sandbox):
    server = pytest.importorskip("mimesis_voice.server")
    assert config.needs_setup()
    prof, err = server._profile(None)
    assert prof is None and "NO VOICE IS SET UP" in err
    # The demo stays reachable when asked for by name.
    prof, err = server._profile("example")
    assert err is None and prof.slug == "example"
    assert "NO VOICE IS SET UP" in server.compose_in_voice("write a note")
    assert "NO VOICE IS SET UP" in server.list_voices()


def test_demo_left_active_beside_a_real_voice_is_refused(sandbox):
    server = pytest.importorskip("mimesis_voice.server")
    _fake_built_voice(sandbox, "jane-essays")
    config.write_state(active="example")
    assert not config.needs_setup()
    prof, err = server._profile(None)
    assert prof is None and "jane-essays" in err
    prof, err = server._profile("jane-essays")
    assert err is None and prof.slug == "jane-essays"


def test_unbuilt_voice_by_name_is_reported_not_used(sandbox):
    server = pytest.importorskip("mimesis_voice.server")
    (sandbox / "half").mkdir()
    (sandbox / "half" / "config.json").write_text("{}", encoding="utf-8")
    prof, err = server._profile("half")
    assert prof is None and "not built" in err


def test_a_new_voice_takes_over_from_the_demo(sandbox):
    from mimesis_voice import cli
    config.write_state(active="example")
    _fake_built_voice(sandbox, "jane-posts")
    cli._activate_if_unclaimed("jane-posts")
    assert config.read_state()["active"] == "jane-posts"
    # A second voice does not steal the slot from a real one.
    _fake_built_voice(sandbox, "jane-essays")
    cli._activate_if_unclaimed("jane-essays")
    assert config.read_state()["active"] == "jane-posts"
