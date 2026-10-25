"""Minimal-diff editing: segmentation round-trips and every limit holds in code."""
from __future__ import annotations

import pytest

from mimesis_voice import contentrisk, ownedit, segment
from mimesis_voice.scrub import ScrubCalibration

CAL = ScrubCalibration(
    banned_words=["crucial", "delve"], banned_phrases=["it is important to note"], whitelist=[],
    burstiness_floor=5.0, hedge_ceiling=0.5, mean_sentence_len=20.0, n_pieces=10,
)

TEXT = (
    "We scored 345 items with three raters. Agreement was 0.78 across the full set, "
    "and Smith et al. (2021) reported 0.71 on a similar pool.\n\n"
    "The raters disagreed most on the hardest items \u2014 the ones near the cut. "
    "It is important to note that the cut was fixed before scoring. "
    "We used Qwen3-8B for the generations at temperature 0.9."
)


@pytest.mark.parametrize("text", [
    TEXT,
    "  leading space. Then more.\n",
    "\\section{Methods}\nWe ran it. Smith et al. (2020) did not.\n\n% a comment\nSee Fig. 2 for the curve. Done.",
    "",
    "One sentence with no end",
    "A. Smith wrote this, e.g. here. And i.e. there.\n\n\n\nNew para.",
])
def test_segmentation_reconstructs_exactly(text):
    assert segment.join(segment.split(text)) == text


def test_abbreviations_do_not_split():
    s = segment.sentences("Smith et al. (2024) found it. See Fig. 2 for details. J. Doe agreed.")
    assert s == ["Smith et al. (2024) found it.", "See Fig. 2 for details.", "J. Doe agreed."]


def test_latex_structure_lines_are_fixed():
    segs = segment.split("\\section{Intro}\nWe ran it.\n\\label{sec:x}\nThen we stopped.")
    assert [s.text for s in segs if s.is_prose] == ["We ran it.", "Then we stopped."]


def test_flags_land_on_the_right_sentence():
    infos = ownedit.flag_sentences(TEXT, CAL)
    assert len(infos) == 5
    assert "em-dash" in infos[2].hard
    assert any(f.startswith("banned-phrase") for f in infos[3].hard)
    assert not infos[0].flags and not infos[4].flags


def _apply(edits, **kw):
    return ownedit.apply_edits(TEXT, edits, CAL, **kw)


def test_a_clean_fix_applies_and_leaves_everything_else_byte_identical():
    res = _apply([{"n": 3, "text": "The raters disagreed most on the hardest items, the ones near the cut.",
                   "fixes": "em-dash"}])
    assert [o.status for o in res.ops] == ["applied"]
    assert res.edited == TEXT.replace(" \u2014 the ones", ", the ones")
    assert res.section_sim >= res.floor


def test_unflagged_sentence_is_refused_under_flagged_scope():
    res = _apply([{"n": 5, "text": "We used Qwen3-8B for all generations at temperature 0.9."}], scope="flagged")
    assert res.ops[0].status == "refused" and "no flag" in res.ops[0].reason
    assert res.edited == TEXT


def test_default_scope_is_hard_and_leaves_advisory_habits_alone():
    # Long enough that one swapped word stays above the 0.93 whole-text floor; on a
    # 12-word text the same swap is refused by the floor, which is intended.
    text = "The crucial step was the second rater pass. " + TEXT.split("\n\n")[0]
    assert ownedit.flag_sentences(text, CAL)[0].advisory  # banned word, advisory only
    res = ownedit.apply_edits(text, [{"n": 1, "text": "The key step was the second rater pass."}], CAL)
    assert res.scope == "hard"
    assert res.ops[0].status == "refused" and "no hard flag" in res.ops[0].reason
    ok = ownedit.apply_edits(text, [{"n": 1, "text": "The key step was the second rater pass."}], CAL,
                             scope="flagged")
    assert ok.ops[0].status == "applied"


def test_goal_scope_allows_unflagged_sentence_with_a_reason():
    res = _apply([{"n": 5, "text": "We used Qwen3-8B for all generations at temperature 0.9.",
                   "fixes": "goal: say all"}], scope="goal")
    assert res.ops[0].status == "applied"


def test_merge_or_split_is_refused():
    split = _apply([{"n": 3, "text": "The raters disagreed most on the hardest items. Those sat near the cut."}])
    assert split.ops[0].status == "refused" and "one sentence" in split.ops[0].reason


def test_heavy_rewrite_is_refused():
    res = _apply([{"n": 4, "text": "Before any scoring began, we locked the threshold in advance."}])
    assert res.ops[0].status == "refused"
    assert "keeps" in res.ops[0].reason or "adds" in res.ops[0].reason


def test_number_and_citation_changes_are_refused():
    res = _apply([{"n": 2, "text": "Agreement was 0.80 across the full set, and Smith et al. (2021) "
                                   "reported 0.71 on a similar pool.", "fixes": "goal: x"}], scope="goal")
    assert res.ops[0].status == "refused" and "number" in res.ops[0].reason


def test_introducing_a_banned_word_is_refused():
    res = _apply([{"n": 3, "text": "The raters disagreed most on the crucial hardest items, the ones near the cut.",
                   "fixes": "em-dash"}])
    assert res.ops[0].status == "refused" and "banned-word" in res.ops[0].reason


def test_introducing_a_detection_risk_move_is_refused():
    res = _apply([{"n": 4, "text": "This suggests that the cut was fixed before scoring.",
                   "fixes": "banned phrase"}])
    assert res.ops[0].status == "refused"
    assert "detection-risk" in res.ops[0].reason or "keeps" in res.ops[0].reason


def test_floor_is_clamped_and_enforced():
    res = _apply([], floor=0.5)
    assert res.floor == ownedit.MIN_FLOOR
    # Two legitimate fixes, a floor so high that one of them has to go.
    res = _apply([
        {"n": 3, "text": "The raters disagreed most on the hardest items, the ones near the cut.", "fixes": "em-dash"},
        {"n": 4, "text": "The cut was fixed before scoring.", "fixes": "banned phrase"},
    ], floor=0.99)
    assert len(res.applied) <= 1
    assert res.section_sim >= 0.99 or not res.applied


def test_render_result_carries_diff_and_provenance():
    res = _apply([{"n": 3, "text": "The raters disagreed most on the hardest items, the ones near the cut.",
                   "fixes": "em-dash"}])
    out = ownedit.render_result(res, "Jane Doe")
    assert "PROVENANCE" in out and "[-" in out and "{+" in out and "## EDITED TEXT" in out


def test_kit_lists_every_sentence_with_flags():
    infos = ownedit.flag_sentences(TEXT, CAL)
    kit = ownedit.render_kit(TEXT, infos, "Jane Doe", "research", None, "flagged", ownedit.DEFAULT_FLOOR)
    assert kit.count("\n[") >= 5 and "apply_own_edits" in kit and "HARD: em-dash" in kit


# --- content risk ---------------------------------------------------------------------

@pytest.mark.parametrize("sentence,cls", [
    ("This suggests that the model relies on surface cues.", "interpretation"),
    ("Unlike prior work, we measure the effect directly.", "ranking"),
    ("Our main contribution is a validated rubric.", "contribution"),
    ("Whether this holds for clinical text remains unexplored.", "gap"),
    ("The rest of this paper is organised as follows.", "roadmap"),
    ("In other words, the cue carries the decision.", "gloss"),
    ("Full validation would require a second rater pool.", "obligation"),
    ("The system prompt read: You are a clinician.", "specification"),
])
def test_risk_classes_fire(sentence, cls):
    assert cls in contentrisk.classify(sentence).risk


@pytest.mark.parametrize("sentence", [
    "Smith et al. (2021) found that agreement fell to 0.62 on hard items.",
    "We ran 345 items through Qwen3-8B at temperature 0.9 with seed 7.",
    "The aim of the present study was to test whether raters agree.",
    "Bisbee et al. (2024) suggest that synthetic samples understate variance.",
])
def test_reports_and_run_facts_are_not_risky(sentence):
    r = contentrisk.classify(sentence)
    assert not r.is_risk, r.risk


def test_latex_citations_are_normalised():
    r = contentrisk.classify("\\citet{smith2021} found that agreement fell on hard items.")
    assert "cited-report" in r.safe and r.citations == 1


def test_risk_report_finds_runs():
    text = ("This suggests a shortcut. In other words, the cue decides. This means the gap is real. "
            "We ran 40 items.")
    rep = contentrisk.analyze(text)
    assert rep.runs() == [(1, 3)]
    assert "DETECTION RISK" in contentrisk.render(rep)


# --- regressions from the 2026-10-25 review ---------------------------------------------

FILLER = " ".join(f"We scored batch {i} with the same three raters and the same rubric." for i in range(40))


def test_citation_regex_is_linear_on_unclosed_parentheses():
    import time
    cites = "; ".join(f"Name{c}, 20{10 + i}" for i, c in enumerate("ABCDEFGHIJ"))
    s = f"Prior studies agree (e.g., {cites}; among others; see (Table 2)."
    t0 = time.perf_counter()
    contentrisk.classify(s)
    assert time.perf_counter() - t0 < 0.5


def _one(text, op, **kw):
    return ownedit.apply_edits(text, [op], CAL, **kw).ops[0]


def test_no_hidden_sentence_after_an_abbreviation_or_initial():
    text = "Full scores \u2014 per rater \u2014 are listed in Appendix A.\n\n" + FILLER
    for new in ("Full scores, per rater, are listed in Appendix A. We checked them twice.",
                "Full scores, per rater, are listed in Fig. We checked them twice."):
        o = _one(text, {"n": 1, "text": new, "fixes": "em-dash"})
        assert o.status == "refused", new


def test_segmenter_ends_sentences_at_appendix_letters_and_no():
    assert segment.sentences("See Appendix A. We then ran it.") == ["See Appendix A.", "We then ran it."]
    assert segment.sentences("Does it hold? No. The effect vanished.") == [
        "Does it hold?", "No.", "The effect vanished."]
    assert segment.sentences("Written by A. Smith and J. R. Tolkien. Done.") == [
        "Written by A. Smith and J. R. Tolkien.", "Done."]


def test_newlines_cannot_add_structure_or_delete_a_heading():
    text = FILLER + " The raters disagreed most on the hardest items \u2014 the ones near the cut.\n\n" + FILLER
    n = 41
    for new in ("The raters disagreed most on the hardest items, the ones near the cut.\n# Added Heading.",
                "The raters disagreed most on the hardest items, the ones near the cut.\n\\section{Added}.",
                "The raters disagreed most on the hardest items.\n% the ones near the cut."):
        assert _one(text, {"n": n, "text": new, "fixes": "em-dash"}).status == "refused", new
    crlf = "Introduction\r\n\r\nThe raters disagreed \u2014 on hard items. " + FILLER
    assert segment.sentences(crlf)[0] == "The raters disagreed \u2014 on hard items."
    plain = "Methods\nThe raters disagreed \u2014 on hard items. " + FILLER
    assert segment.sentences(plain)[0] == "The raters disagreed \u2014 on hard items."


def test_latex_environment_lines_are_not_sentences():
    tex = "We fit the model.\n\\begin{equation}\nE = mc^2.\n\\end{equation}\nIt converged."
    assert segment.sentences(tex) == ["We fit the model.", "It converged."]
    assert segment.join(segment.split(tex)) == tex


def test_whole_text_hard_flags_are_a_backstop():
    text = FILLER + " The raters disagreed most on the hardest items \u2014 the ones near the cut.\n\n" + FILLER
    o = _one(text, {"n": 41, "text": "The raters disagreed most on the hardest items, the ones near the cut \U0001F642.",
                    "fixes": "em-dash"})
    assert o.status == "refused"


def test_nan_floor_falls_back_to_default():
    res = ownedit.apply_edits(TEXT, [], CAL, floor=float("nan"))
    assert res.floor == ownedit.DEFAULT_FLOOR


def test_citation_key_changes_and_added_citations_are_refused():
    text = "We follow the protocol \\parencite{smith2021a}, with \u2014 one change. " + FILLER
    o = _one(text, {"n": 1, "text": "We follow the protocol \\parencite{smith2021b}, with one change.",
                    "fixes": "em-dash"})
    assert o.status == "refused" and "citation" in o.reason
    text2 = "We follow the protocol \u2014 with one change. " + FILLER
    o2 = _one(text2, {"n": 1, "text": "We follow the protocol of Jones et al., with one change.",
                      "fixes": "em-dash"})
    assert o2.status == "refused"


def test_a_boundary_change_refuses_only_its_own_op():
    text = ("The raters disagreed \u2014 most of all on the hardest items in the pool. "
            "We followed the scoring protocol \u2014 described earlier in the methods section. "
            "Next we scored everything twice. " + FILLER)
    res = ownedit.apply_edits(text, [
        {"n": 1, "text": "The raters disagreed, mostly on the hardest items in the pool.", "fixes": "em-dash"},
        {"n": 2, "text": "We followed the scoring protocol described earlier in the methods section", "fixes": "em-dash"},
    ], CAL)
    by_n = {o.n: o for o in res.ops}
    assert by_n[1].status == "applied"
    assert by_n[2].status == "refused"


def test_a_no_change_op_does_not_block_a_later_real_one():
    res = _apply([{"n": 3, "text": TEXT.split("\n\n")[1].split(". ")[0] + "."},
                  {"n": 3, "text": "The raters disagreed most on the hardest items, the ones near the cut.",
                   "fixes": "em-dash"}])
    assert [o.status for o in res.ops] == ["refused", "applied"]


def test_bool_n_is_ignored():
    assert _apply([{"n": True, "text": "x."}]).ops == []
