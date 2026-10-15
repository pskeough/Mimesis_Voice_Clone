"""Regression tests for the second detector family (rhetoric_ext).

Every positive specimen below comes from one of two posts written in a
second author's voice by the shipped engine on 2026-09-02, both of which passed
that author's scrubber and were then rated 100% machine-written by a publishing
platform's AI detector. Proper names in them are replaced. Every negative is a
synthetic stand-in for a construction the author really writes, or ordinary
English the pattern must not touch.

The corpus false-positive numbers in rhetoric_ext.py were measured with
`python -m mimesis_voice.rhetoric_ext measure <voice>`; re-run that, not only
this file, after changing any pattern.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import rhetoric, rhetoric_ext as rx  # noqa: E402

LONG_POST_EXCERPT = """The argument runs in three movements. The Industrial Age, 1769 to 1995, built the machines and taught us the mathematics of plant and cash.

In 2025, we crossed what I call the Threshold, and there the old pattern broke. Every previous age arrived whether anyone chose it or not. This one arrives as a fork.

Organisations will drift into a Drift Age, where AI amplifies inherited weaknesses at machine speed: fear industrialised, silos automated, exhaustion measured in real time. The machine does not choose between them. Leadership does.

Two frameworks hold it together. The first names the five choices every leader now makes.

One number sits under all of it. Seventy sounded high to me the first time I read it. Then I counted the transformation programmes I have watched stall for entirely human reasons, and I now read seventy as conservative.

A fingerprint is the trace of human originality: distinct, formed by pressure and time, impossible to replicate, and entirely capable of fading. Nothing about what comes next is predetermined. That is precisely the point.

If you have been reading since June, thank you for waiting. If you are new here, welcome, and pull up a chair. The walk starts now, and I would much rather take it with you."""


def test_announcers_fire_on_the_four_specimens():
    hits = rx.find_announcers(LONG_POST_EXCERPT)
    assert "The argument runs in three movements." in hits
    assert "Two frameworks hold it together." in hits
    assert "One number sits under all of it." in hits
    assert rx.find_announcers("Here is the map in one breath.") == ["Here is the map in one breath."]


def test_announcers_leave_a_stake_naming_pivot_alone():
    # The author's own move names a stake, not a table of contents.
    assert rx.find_announcers("Here is the part that should worry anyone who runs a team.") == []
    assert rx.find_announcers("Two of them were in the room that morning, and both were angry.") == []


def test_pointer_verdicts():
    assert rx.find_pointer_verdicts("That is precisely the point.")
    assert rx.find_pointer_verdicts("That is why it exists.")
    assert rx.find_pointer_verdicts("On the cover sits a fingerprint, and that image is the whole argument in miniature.")
    # From the v6 eval set: the first family's declarative-rating pattern needs
    # "the part THAT" and missed this one.
    assert rx.find_pointer_verdicts("That is the part the inevitability story conveniently hides.")
    assert rhetoric.find_declarative_ratings("That is the part the inevitability story conveniently hides.") == []


def test_pointer_verdicts_leave_ordinary_pointing_alone():
    assert rx.find_pointer_verdicts("That is the office on the corner.") == []
    assert rx.find_pointer_verdicts("This is the third time the board has asked.") == []


def test_participial_triad():
    assert rx.find_participial_triads("fear industrialised, silos automated, exhaustion measured in real time")
    assert rx.find_participial_triads("we hired quickly, trained badly, and paid late") == []


def test_colon_twist():
    s = "A fingerprint is the trace of human originality: distinct, formed by pressure and time, impossible to replicate, and entirely capable of fading."
    assert rx.find_colon_twists(s)
    assert rx.find_colon_twists("Three things: talent, structure, and pay.") == []


def test_parallel_address():
    assert rx.find_parallel_address("If you have been reading since June, thank you for waiting. If you are new here, welcome.")
    assert rx.find_parallel_address("If you were to rate each behaviour as poor. If you were to rate each behaviour as poor.") == []


def test_conversion_narrative():
    s = "Seventy sounded high to me the first time I read it. Then I counted the programmes I have watched stall, and I now read seventy as conservative."
    assert rx.find_conversions(s)
    assert rx.find_conversions("Seventy sounded high to me. The board disagreed.") == []


def test_personification_counts_and_gates_at_three():
    text = "The question moved. The old pattern broke. The argument travels. One number sits under all of it."
    hits = rx.find_personifications(text)
    assert len(hits) >= 3
    assert "personification" in rx.analyze(text).hard_flags
    # Once in a piece is the author's own rate.
    one = "Late in the meeting the hard question lands, and nobody wants to raise it."
    assert len(rx.find_personifications(one)) == 1
    assert "personification" not in rx.analyze(one).hard_flags


def test_tail_twist_is_advisory():
    s = "The Information Age taught us to digitise, to globalise, and to measure everything that moved, and I date its opening to a world that was moving under our feet and mostly telling us it was not."
    rep = rx.analyze(s)
    assert rep.tail_twists
    assert "tail-twist" not in rep.hard_flags


def test_does_pair():
    assert rx.find_does_pairs("The machine does not choose between them. Leadership does.")
    assert rx.find_does_pairs("The machine does not choose between them. Leadership is a choice made daily.") == []


def test_then_openers_are_rate_gated():
    text = ("Then AI arrived. Then a project swallowed the season. Then I counted. "
            "The board met on Tuesday. Nobody spoke for the first ten minutes.")
    # advisory=[] is the profile saying "gate this one"; see the next test for
    # why the shipped default does not.
    strict = rx.RhetoricExtCalibration(then_p95=0.12, advisory=[])
    loose = rx.RhetoricExtCalibration(then_p95=80.0, advisory=[])
    assert "then-opener" in rx.analyze(text, strict).hard_flags
    assert "then-opener" not in rx.analyze(text, loose).hard_flags
    # A single "Then" is never hard on its own, whatever the p95.
    one = "Then AI arrived. The board met on Tuesday. Nobody spoke for the first ten minutes."
    assert "then-opener" not in rx.analyze(one, strict).hard_flags


def test_generator_phrases_measured_zero_in_the_reference_corpus():
    hits = rx.find_generator_phrases("AI amplifies inherited weaknesses at machine speed, measured in real time. Here is the map in one breath.")
    assert {h.lower() for h in hits} == {"at machine speed", "in real time", "in one breath"}


def test_intensifier_rate_gate():
    text = "It is precisely that. It is entirely so. Quietly, mostly, genuinely, exactly this."
    assert rx.analyze(text, rx.RhetoricExtCalibration(intensifier_p95=2.2)).intensifier_over


def test_demotion_moves_first_family_flags_off_the_report_when_under_rate():
    # A corpus whose real posts fire the first family's reversal detector 4 times
    # in 8 can rate-gate it instead of banning it.
    text = ("This is not hope. Hope has weight, and this has none. " + "The board met on Tuesday and talked for two hours about the budget. " * 12)
    ff = rhetoric.analyze(text)
    assert ff.reversals
    cal = rx.RhetoricExtCalibration(demote={"unheeded-reversal": 8.0})
    rep = rx.analyze(text, cal, first_family=ff)
    assert ff.reversals == []
    assert "unheeded-reversal" in rep.demoted
    assert "unheeded-reversal" not in rep.hard_flags
    # Over what the author's rate licenses for the length, it comes back as hard. One
    # instance is always licensed; three in a short text is not.
    text3 = "This is not hope. Hope has weight, and this has none. " * 3 + "The board met on Tuesday. " * 10
    ff2 = rhetoric.analyze(text3)
    rep2 = rx.analyze(text3, rx.RhetoricExtCalibration(demote={"unheeded-reversal": 0.5}), first_family=ff2)
    assert "unheeded-reversal" in rep2.hard_flags
    ff3 = rhetoric.analyze(text)
    rep3 = rx.analyze(text, rx.RhetoricExtCalibration(demote={"unheeded-reversal": 0.5}), first_family=ff3)
    assert "unheeded-reversal" not in rep3.hard_flags


def test_calibrate_never_demotes_unless_asked():
    # Off by default: a corpus with machine-written pieces in it would license
    # the generator's habits as the author's.
    reversing = [("This is not hope. Hope has weight, and this has none. " * 3 + "The board met on Tuesday and talked about the budget for an hour. " * 10)] * 6
    assert rx.calibrate(reversing).demote == {}
    assert "unheeded-reversal" in rx.calibrate(reversing, demote=True).demote
    plain = ["The board met on Tuesday and talked for two hours about the budget, and nobody left early. " * 20] * 6
    assert rx.calibrate(plain, demote=True).demote == {}


def test_negation_template_detects_and_gates_when_the_profile_asks():
    assert rx.find_negation_templates("The outcome is not fixed. It is chosen.")
    assert rx.find_negation_templates("Leadership isn't complex. It's a series of choices.")
    assert rx.find_negation_templates("They win. Not because they have the smartest people.")
    # An ordinary negation followed by an unrelated sentence is not the template.
    assert rx.find_negation_templates("The room was not ready. We started anyway at nine.") == []
    hard = rx.RhetoricExtCalibration(advisory=[])
    assert "negation-template" in rx.analyze("The outcome is not fixed. It is chosen.", hard).hard_flags


def test_question_fragment_detects_and_the_stacked_volley_is_not_one():
    assert rx.find_question_fragments("The answers? Leadership today demands more than vision.")
    assert rx.find_question_fragments("Now? It is something you build.")
    volley = "Where should a person always stay in the loop? What is the team actually for now? Who keeps the gain? Those questions deserve more than a default answer."
    assert rx.find_question_fragments(volley) == []
    assert rx.find_question_fragments("What next? That is what I want to spend the year on.") == []
    hard = rx.RhetoricExtCalibration(advisory=[])
    assert "question-fragment" in rx.analyze("Now? It is something you build.", hard).hard_flags


# --- the advisory layer ----------------------------------------------------------
#
# Three detectors ship demoted because they were measured firing on the
# reference author's own writing: negation-template on 9% of his 116 creative
# pieces ("Not everyone has ability or talent.", "Not sure."), question-fragment
# on 7% (dialogue: "It hurts?", "The next life?"), and then-opener on 5%,
# because its uncalibrated rate ceiling is 0.00 per 100 sentences. The table and
# the specimens are in the rhetoric_ext module docstring.

def test_the_three_measured_defaults_report_but_do_not_gate():
    text = ("The outcome is not fixed. It is chosen. Now? It is something you build. "
            "Then AI arrived. Then a project swallowed the season.")
    rep = rx.analyze(text, rx.RhetoricExtCalibration(then_p95=0.12))
    assert rep.hard_flags == []
    assert set(rep.advisory_flags) == {"negation-template", "question-fragment", "then-opener"}
    # Reported, not silent, and not clean.
    lines = rx.render_lines(rep, "the author")
    assert lines and all("[HIGH]" not in ln for ln in lines)
    assert any("Negation template" in ln for ln in lines)
    assert not rep.is_clean


def test_a_profile_can_gate_a_demoted_detector():
    text = "The outcome is not fixed. It is chosen."
    rep = rx.analyze(text, rx.RhetoricExtCalibration(advisory=["question-fragment"]))
    assert "negation-template" in rep.hard_flags
    assert "[HIGH] Negation template" in " ".join(rx.render_lines(rep))


def test_disabled_suppresses_the_line_advisory_only_lowers_it():
    text = "The outcome is not fixed. It is chosen."
    off = rx.analyze(text, rx.RhetoricExtCalibration(disabled=["negation-template"], advisory=[]))
    assert off.hard_flags == [] and off.advisory_flags == []
    assert rx.render_lines(off) == []
    dem = rx.analyze(text, rx.RhetoricExtCalibration(advisory=["negation-template"]))
    assert "[MEDIUM] Negation template" in " ".join(rx.render_lines(dem))


def test_calibration_round_trips_the_advisory_list(tmp_path):
    cal = rx.RhetoricExtCalibration(advisory=["then-opener"], dash_zero=True)
    f = tmp_path / rx.FILENAME
    cal.save(f)
    assert rx.RhetoricExtCalibration.load(f).advisory == ["then-opener"]
    # A file written before advisory existed falls back to the measured
    # defaults, never to hard-on-presence.
    (tmp_path / "old.json").write_text('{"then_p95": 0.5}', encoding="utf-8")
    assert rx.RhetoricExtCalibration.load(tmp_path / "old.json").advisory == list(rx.DEFAULT_ADVISORY)
    # An explicit empty list means "gate all of them", and must survive the trip.
    (tmp_path / "hard.json").write_text('{"advisory": []}', encoding="utf-8")
    assert rx.RhetoricExtCalibration.load(tmp_path / "hard.json").advisory == []


def test_calibrate_keeps_the_defaults_and_can_be_overridden():
    plain = ["The board met on Tuesday and talked for two hours about the budget. " * 30] * 4
    assert rx.calibrate(plain).advisory == list(rx.DEFAULT_ADVISORY)
    assert rx.calibrate(plain, advisory=[]).advisory == []


def test_emoji_and_hashtags():
    assert "emoji" in rx.analyze("Mental health is a leadership conversation. \U0001F9E0").hard_flags
    assert "hashtag-cluster" in rx.analyze("Come and build it with me. #leadership #AI #culture").hard_flags
    assert "hashtag-cluster" not in rx.analyze("One tag only. #leadership").hard_flags


def test_linkedin_phrases():
    hits = rx.find_linkedin_phrases("Let that sink in. Here's the thing: nobody talks about the cost.")
    assert len(hits) == 3


def test_dash_gate_follows_policy_not_corpus():
    text = "Leadership today demands more than vision or speed — it demands judgment."
    assert "dash" not in rx.analyze(text).hard_flags
    assert "dash" in rx.analyze(text, rx.RhetoricExtCalibration(dash_zero=True)).hard_flags


def test_full_excerpt_is_caught_and_clean_prose_is_not():
    rep = rx.analyze(LONG_POST_EXCERPT, rx.RhetoricExtCalibration(then_p95=0.12, intensifier_p95=2.24))
    for f in ("announcer", "pointer-verdict", "participial-triad", "colon-twist",
              "parallel-address", "conversion"):
        assert f in rep.hard_flags, f
    # One "Then" in the excerpt: over the p95 but under the two-instance floor.
    assert "then-opener" not in rep.hard_flags
    assert rep.generator_phrases
    clean = ("Most firms treat their strongest staff as rare, and build a whole system to prove it: "
             "succession plans, talent pools, fast-track schemes, retention bonuses that admit we cannot "
             "hire another one of them. Both claims cannot hold at once. Which one holds inside your firm "
             "will not be settled by the software.")
    assert rx.analyze(clean).hard_flags == []
