"""Behaviour of the structural measures, independent of the study's data.

`test_measure_parity.py` proves the ported code reproduces the study's numbers,
and it only runs where the study tree exists. These tests run everywhere and
pin the properties the gate depends on: that each measure fires on the failure
it was adopted for, stays quiet on the author's own shape, and refuses to score
what it cannot resolve.
"""
from __future__ import annotations

import pytest

from mimesis_voice.measures import calibration as cal_mod
from mimesis_voice.measures import construction, cpd, lexdiv, otd, per, psd, report
from mimesis_voice.measures import textbase as tb


# --- splitters ---------------------------------------------------------------

def test_normalise_punct_is_measurement_only():
    assert tb.normalise_punct("a -- b") == "a—b"
    assert tb.normalise_punct("_italics_ here") == "italics here"


def test_fold_glyphs_removes_transcription_provenance():
    assert tb.fold_glyphs("“it’s”") == '"it\'s"'


def test_paragraphs_splits_on_blank_lines_only():
    assert len(tb.paragraphs("a\nb\nc")) == 1
    assert len(tb.paragraphs("a\n\nb\n\nc")) == 3


def test_prose_paragraphs_recovers_single_newline_paragraphing():
    """The case that made the paragraph line meaningless on a real corpus."""
    long_lines = "\n".join(" ".join(["word"] * 40) for _ in range(5))
    assert len(tb.paragraphs(long_lines)) == 1
    assert len(tb.prose_paragraphs(long_lines)) == 5


def test_prose_paragraphs_leaves_hard_wrapped_text_alone():
    """Hard wrapping at a fixed column is not paragraphing."""
    wrapped = "\n".join(" ".join(["word"] * 10) for _ in range(8))
    assert len(tb.prose_paragraphs(wrapped)) == 1


def test_truncate_words_keeps_paragraph_breaks():
    text = "one two three\n\nfour five six"
    assert tb.truncate_words(text, 4) == "one two three\n\nfour"


# --- per ---------------------------------------------------------------------

def _word(i):
    """A distinct alphabetic token. lexdiv's tokenizer drops digits, so
    "token1" and "token2" are the same word to it."""
    a = "abcdefghijklmnopqrstuvwxyz"
    return a[i % 26] + a[(i // 26) % 26] + a[(i // 676) % 26] + "x"


def test_per_fires_on_restatement_in_fresh_wording():
    looping = (
        "The committee rejected the proposal outright. "
        "The proposal was rejected by the committee without discussion. "
        "Rejection of the committee proposal followed immediately."
    )
    assert per.per(looping)["per"] > 0.3


def test_per_quiet_on_prose_that_moves_on():
    moving = (
        "The committee rejected the proposal outright. "
        "Rain started before anyone reached the car park. "
        "Someone had left a window open on the third floor."
    )
    assert per.per(moving)["per"] == 0.0


def test_per_windows_read_the_whole_draft():
    """The reason the study's first-600-words cap was not adopted verbatim."""
    filler = " ".join(
        f"{_word(i)} {_word(i + 500)} met {_word(i + 900)} beside {_word(i + 1300)}."
        for i in range(120)
    )
    loop = ("Everything failed at once. Failure arrived everywhere simultaneously. "
            "At once, everything had failed. ") * 20
    text = filler + " " + loop
    assert len(text.split()) > per.WORD_CAP
    assert per.per(tb.truncate_words(text, per.WORD_CAP))["per"] == 0.0
    got = per.worst_window(text)
    assert got["per"] > 0.5 and got["n_windows"] > 1 and got["start_word"] > 0


def test_echo_sentences_returns_the_offending_sentences():
    looping = ("The committee rejected the proposal outright. "
               "The proposal was rejected by the committee without discussion.")
    assert per.echo_sentences(looping)


# --- cpd ---------------------------------------------------------------------

def test_burstiness_separates_metronomic_from_varied():
    flat = [15] * 30
    varied = [4, 31, 9, 45, 6, 22, 3, 38, 11, 27, 5, 40, 8, 19, 2] * 2
    assert cpd.burstiness_an(flat) < cpd.burstiness_an(varied)
    assert cpd.burstiness_an(flat) <= -0.9


def test_burstiness_is_length_independent():
    """The whole reason A_n replaces the raw Goh-Barabasi B."""
    series = [4, 31, 9, 45, 6, 22, 3, 38, 11, 27]
    short = cpd.burstiness_an(series)
    long = cpd.burstiness_an(series * 6)
    assert abs(short - long) < 0.05


def test_cpd_refuses_a_series_with_no_shape():
    assert cpd.profile("One. Two.")["n_sent"] < cpd.MIN_SENTS


# --- psd ---------------------------------------------------------------------

def test_paragraph_cv_separates_one_size_from_varied():
    same = "\n\n".join(" ".join(["word"] * 100) for _ in range(6))
    varied = "\n\n".join(" ".join(["word"] * n) for n in (20, 180, 45, 300, 12, 90))
    assert psd.profile(same)["para_cv_words"] < 0.05
    assert psd.profile(varied)["para_cv_words"] > 0.6


# --- lexdiv ------------------------------------------------------------------

def test_lexdiv_refuses_a_text_shorter_than_the_window():
    assert lexdiv.measure("word " * 50) == {}


def test_hapax_falls_when_vocabulary_narrows():
    wide = " ".join(_word(i) for i in range(400))
    narrow = " ".join(_word(i % 12) for i in range(400))
    assert lexdiv.measure(wide)["hapax_w300"] > lexdiv.measure(narrow)["hapax_w300"]


# --- construction ------------------------------------------------------------

def test_anaphora2_counts_repeated_two_word_openers():
    text = ("I remember the coffee. I remember the rain. I remember the door. "
            "Something else happened later.")
    r = construction.rates(text)
    assert r["anaphora2"] > 50
    assert construction.anaphora_pairs(text) == ["i remember", "i remember"]


def test_long_short_counts_the_beat():
    long = " ".join(["word"] * 30) + "."
    assert construction.rates(f"{long} Yes. {long} No.")["long_short"] > 50


def test_rhetorical_question_rate():
    assert construction.rates("Why? Because. Who knows? Nobody.")["rhetorical_question"] == 50.0


def test_construction_rates_survive_empty_text():
    assert construction.rates("") == {f: 0.0 for f in construction.FEATURES}


# --- otd ---------------------------------------------------------------------

def test_js_distance_bounds():
    p = [1.0] + [0.0] * 9
    q = [0.0] * 9 + [1.0]
    assert otd.js_distance(p, p) == 0.0
    assert otd.js_distance(p, q) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not otd.available(), reason="spaCy en_core_web_sm not installed")
def test_opener_classes_use_context_not_the_bare_word():
    """The reason this measure needs a tagger and not a word list: the same
    first word lands in different classes depending on what follows it."""
    assert otd.opener_class("Left alone, she counted the tiles.") == "PART"
    assert otd.opener_class("Left hands are rarer than right ones.") == "NP"
    assert otd.opener_class("Past the church the road narrows.") == "PP"
    assert otd.opener_class("Past mistakes are not a plan.") == "NP"
    assert otd.opener_class('"Get out," she said.') == "QUOTE"
    assert otd.opener_class("But nobody moved.") == "CONJ"
    assert otd.opener_class("Although nobody moved, the door opened.") == "SUBORD"


# --- calibration and report --------------------------------------------------

def _corpus(n=40):
    """A synthetic author: varied paragraphs, varied sentence lengths, no echo."""
    out = []
    for k in range(n):
        paras = []
        for j, plen in enumerate((3, 7, 2, 9, 4)):
            sents = []
            for i in range(plen):
                w = 6 + ((i * 7 + j * 5 + k) % 28)
                sents.append(" ".join(f"w{k}{j}{i}{x}" for x in range(w)) + ".")
            paras.append(" ".join(sents))
        out.append("\n\n".join(paras))
    return out


def test_calibration_round_trips(tmp_path):
    cal = cal_mod.calibrate(_corpus())
    p = tmp_path / cal_mod.FILENAME
    cal.save(p)
    back = cal_mod.StructureCalibration.load(p)
    assert back.n_units == cal.n_units
    assert back.per_threshold() == cal.per_threshold()
    assert back.constructions == cal.constructions


def test_thin_calibration_is_reported_not_guessed():
    cal = cal_mod.calibrate(_corpus(4))
    assert cal.thin
    assert cal.per_threshold() is None
    assert cal.para_cv_band(5) is None
    text = "\n\n".join(_corpus(1))
    rendered = report.render(text, cal)
    assert "THIN CALIBRATION" in rendered


def test_report_never_prints_a_total():
    cal = cal_mod.calibrate(_corpus())
    rendered = report.render(_corpus(1)[0], cal)
    for banned in ("total", "overall score", "composite"):
        assert banned not in rendered.lower()
    keys = [ln.key for ln in report.analyse(_corpus(1)[0], cal)]
    assert keys == ["echo", "openers", "paragraphs", "cadence", "vocabulary", "constructions"]


def test_report_flags_one_size_paragraphs():
    cal = cal_mod.calibrate(_corpus())
    flat = "\n\n".join(" ".join(f"w{i}{x}" for x in range(80)) + "." for i in range(5))
    lines = {ln.key: ln for ln in report.analyse(flat, cal)}
    assert lines["paragraphs"].flagged
    assert "one size" in lines["paragraphs"].text


def test_report_declines_to_score_what_it_cannot_resolve():
    cal = cal_mod.calibrate(_corpus())
    lines = {ln.key: ln for ln in report.analyse("Short. Very short. Done.", cal)}
    assert lines["echo"].status == "--"
    assert lines["paragraphs"].status == "--"
    assert not lines["constructions"].flagged


def test_missing_calibration_says_so_rather_than_defaulting():
    assert "No structure calibration" in report.render("some text", None)


def test_construction_cut_matches_between_report_and_calibration():
    """The cut is duplicated to avoid a circular import; it must not drift."""
    assert report.Z_FLAG == cal_mod.CONSTRUCTION_Z


def test_calibration_reports_its_own_false_positive_rate():
    """DETECTORS.md's audit: every line has to say how often it is wrong on the
    author's own writing, because a line that fires on his prose is measuring
    him."""
    cal = cal_mod.calibrate(_corpus())
    for key in ("echo", "openers", "paragraphs", "cadence", "vocabulary", "constructions"):
        if key == "openers" and not otd.available():
            continue
        assert key in cal.self_flag, f"{key} has no measured self-flag rate"
        assert 0.0 <= cal.self_flag[key] <= 1.0


# --- repair notes -------------------------------------------------------------

def test_repair_notes_are_silent_on_a_clean_draft():
    cal = cal_mod.calibrate(_corpus())
    assert report.repair_notes(_corpus(1)[0], cal) == []
    assert report.repair_notes("anything", None) == []


def test_repair_notes_name_the_evidence_not_the_number():
    """A model can act on the sentences; it cannot act on 0.219."""
    cal = cal_mod.calibrate(_corpus())
    looping = "\n\n".join(
        " ".join(
            "The committee rejected the proposal outright. "
            "The proposal was rejected by the committee without discussion. "
            "Rejection of the committee proposal followed from the committee. "
            for _ in range(3)
        ) for _ in range(5)
    )
    notes = report.repair_notes(looping, cal)
    text = "\n".join(notes)
    assert any(n.startswith("COHERENCE") for n in notes)
    assert "committee rejected the proposal" in text
    assert "0.2" not in text and "z=" not in text


def test_repair_notes_never_ask_for_more_uniform_paragraphs():
    """The study's reader PREFERRED the cell that over-varied its paragraphs
    (CV 0.79 against the source's 0.34). A repair prompt that evens paragraphs
    out would optimise toward the failure this measure exists to catch, so the
    over side of every band is diagnosed and never repaired."""
    cal = cal_mod.calibrate(_corpus())
    wild = "\n\n".join(" ".join(f"w{i}{x}" for x in range(n)) + "."
                       for i, n in enumerate((8, 400, 15, 900, 6, 250)))
    lines = {ln.key: ln for ln in report.analyse(wild, cal)}
    assert lines["paragraphs"].flagged and "over the author's band" in lines["paragraphs"].text
    assert not any(n.startswith("PARAGRAPHS") for n in report.repair_notes(wild, cal))


def test_repair_notes_flag_one_size_paragraphs():
    cal = cal_mod.calibrate(_corpus())
    flat = "\n\n".join(" ".join(f"w{i}{x}" for x in range(80)) + "." for i in range(5))
    assert any(n.startswith("PARAGRAPHS") for n in report.repair_notes(flat, cal))
