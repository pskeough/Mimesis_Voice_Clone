"""Regression tests for the LaTeX rewrite failure.

Every case here is a thing that actually shipped in a voiced manuscript while the
pipeline reported CLEAN. The point of the file is that the failure is now a test
rather than a lesson: each assertion names the corruption it prevents.

The last two tests matter as much as the rest. A verifier that rejects everything
is not a verifier, so ``test_identity_is_clean`` and ``test_faithful_rewrite_passes``
guard the other direction.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from mimesis_voice import fidelity, textnorm
from mimesis_voice.scrub import count_dashes, scalpel

# Optional end-to-end regression against a real source/voiced manuscript pair.
# Point both at your own files to run it:
#     MIMESIS_FIDELITY_PAPER=... MIMESIS_FIDELITY_VOICED=... pytest
# The env vars are read as STRINGS and checked before becoming paths: Path("")
# is Path("."), which exists, so building the path first would defeat the skip
# and hand a directory to read_text(). Every other test here uses inline
# fixtures and runs anywhere.
_PAPER_ENV = os.environ.get("MIMESIS_FIDELITY_PAPER", "")
_VOICED_ENV = os.environ.get("MIMESIS_FIDELITY_VOICED", "")
_HAVE_MANUSCRIPTS = bool(_PAPER_ENV and _VOICED_ENV) and Path(_PAPER_ENV).is_file() and Path(_VOICED_ENV).is_file()
PAPER = Path(_PAPER_ENV) if _PAPER_ENV else None
VOICED = Path(_VOICED_ENV) if _VOICED_ENV else None


# --- the scalpel, which caused the corruption directly ------------------------

def test_latex_numeric_range_survives_the_scalpel():
    """0.17--0.21 became "0.17, 0.21" in three table rows of a real manuscript.

    In LaTeX "--" is an en-dash literal. It is range notation, not punctuation,
    and rewriting it changes an interval into two point estimates.
    """
    tex = r"Race & +3.77 to +4.24 & 0.17--0.21 (0.20--0.24) & $-$0.14 SD \\"
    fixed, n = scalpel(tex, fmt="latex")
    assert "0.17--0.21" in fixed
    assert "0.20--0.24" in fixed
    assert n == 0


def test_latex_range_survives_even_when_format_is_sniffed():
    """The corrupting call passed no format at all, so sniffing must be safe."""
    tex = ("\\subsection{Results}\\label{sec:r}\n"
           "The racial contrasts land at 0.17--0.21 against nulls of 0.10--0.19.\n")
    fixed, _ = scalpel(tex)
    assert "0.17--0.21" in fixed and "0.10--0.19" in fixed


def test_latex_emdash_is_still_stripped():
    """Protecting "--" must not amnesty the thing the rule actually targets."""
    tex = "The models---all four of them---avoid the endpoints."
    fixed, n = scalpel(tex, fmt="latex")
    assert "---" not in fixed and n == 2
    assert "The models, all four of them, avoid the endpoints." == fixed


def test_plain_text_behaviour_is_unchanged():
    """Back-compat: outside LaTeX, "--" is still an em-dash a writer typed."""
    fixed, n = scalpel("the room -- and the hall -- went quiet")
    assert fixed == "the room, and the hall, went quiet"
    assert n == 2


def test_math_and_citations_are_not_edited():
    tex = r"a span $x -- y$ and \cite{smith--jones2024} stay intact"
    fixed, _ = scalpel(tex, fmt="latex")
    assert "$x -- y$" in fixed and r"\cite{smith--jones2024}" in fixed


def test_rule_lines_still_survive():
    md = "HEADER\n------\nbody -- text"
    fixed, _ = scalpel(md)
    assert "------" in fixed


def test_count_dashes_agrees_with_the_scalpel():
    """The report and the fix must not disagree, or a draft is flagged for a
    dash the scalpel already declined to touch."""
    for text, fmt in ((r"0.17--0.21 and 0.20--0.24", "latex"),
                      ("a -- b -- c", "plain"),
                      ("the models---all four---avoid it", "latex")):
        _, n = scalpel(text, fmt=fmt)
        assert count_dashes(text, fmt=fmt) == n


# --- structure and scaffolding ------------------------------------------------

def test_scaffolding_token_is_hard():
    """A literal "[memory: persona/voice-style.md]" replaced \\section{Results}
    and compiled silently, because no check looked for harness artifacts."""
    src = "\\section{Results}\n\nWe report the findings below in order.\n"
    out = "[memory: persona/voice-style.md]\n\nWe report the findings below.\n"
    rep = fidelity.verify(src, out, fmt="latex")
    assert not rep.ok
    axes = {f.axis for f in rep.hard}
    assert "scaffold" in axes and "structure" in axes


def test_lost_section_is_hard():
    src = "\\section{Method}\nText one here.\n\\section{Results}\nText two here.\n"
    out = "\\section{Method}\nText one here.\nText two here.\n"
    rep = fidelity.verify(src, out, fmt="latex")
    assert any(f.axis == "structure" for f in rep.hard)


def test_retitling_a_section_is_allowed():
    """The pass was explicitly asked to rewrite headings, so titles are compared
    by count and not by text. Otherwise the gate fires on its own instructions."""
    src = "\\section{Method}\nOne two three four five six seven eight nine ten.\n"
    out = "\\section{How we did it}\nOne two three four five six seven eight ten.\n"
    rep = fidelity.verify(src, out, fmt="latex")
    assert not any(f.axis == "structure" for f in rep.findings)


def test_destroyed_range_is_hard():
    src = "Racial contrasts land at 0.17--0.21 against nulls of 0.10--0.19 here."
    out = "Racial contrasts land at 0.17, 0.21 against nulls of 0.10, 0.19 here."
    rep = fidelity.verify(src, out, fmt="latex")
    assert any(f.axis == "house-style" for f in rep.hard)


# --- rhythm -------------------------------------------------------------------

def _long_short(n: int) -> str:
    """Alternating long and short sentences: high variance, like real prose."""
    long_s = ("The unit of analysis is the design cell, a model crossed with a "
              "demographic cohort, and the thirty iterations inside it are "
              "repeated draws from one conditional response distribution rather "
              "than separate individuals in any sense that matters here. ")
    short_s = "Nothing else changed. "
    return (long_s + short_s) * n


def _flattened(n: int) -> str:
    """The same content chopped into medium sentences: the observed failure."""
    med = ("The unit of analysis is the design cell, a model crossed with a "
           "demographic cohort. The thirty iterations inside it are repeated "
           "draws from one distribution. They are not separate individuals in "
           "any sense that matters. Nothing else about the design changed here. ")
    return med * n


def test_flattened_rhythm_is_hard():
    """sd fell 9.8% in the real pass and the pipeline reported clean, because
    acceptance was assets plus a word budget and nothing measured cadence."""
    rep = fidelity.verify(_long_short(6), _flattened(6), fmt="plain")
    rhythm = [f for f in rep.hard if f.axis == "rhythm"]
    assert rhythm, fidelity.render(rep)
    assert rep.metrics["out_sd"] < rep.metrics["src_sd"]


def test_losing_the_short_sentences_is_hard():
    src = _long_short(8)
    out = src.replace("Nothing else changed. ", "")
    rep = fidelity.verify(src, out, fmt="plain")
    assert any(f.axis == "rhythm" for f in rep.hard)


# --- register and constructions -----------------------------------------------

def test_second_person_in_a_document_with_none_is_hard():
    src = ("The models avoid the endpoints of every instrument tested. "
           "A pipeline drawing patients from them receives no severe cases. "
           "The result holds across all three instruments in the same run.")
    out = ("The models avoid the endpoints of every instrument tested. "
           "If you draw your patients from them you get no severe cases. "
           "The result holds across all three instruments in the same run.")
    rep = fidelity.verify(src, out, fmt="plain")
    hits = [f for f in rep.hard if f.axis == "register"]
    assert hits and "second person" in hits[0].detail


def test_banned_construction_in_a_heading_is_hard():
    """"Race is where the error flattens, not just shifts" shipped as a heading."""
    src = "\\subsection{The error is not uniform}\nOne two three four five six.\n"
    out = ("\\subsection{Race is where the error flattens, not just shifts}\n"
           "One two three four five six.\n")
    rep = fidelity.verify(src, out, fmt="latex")
    assert any(f.axis == "construction" for f in rep.hard)


def test_dropped_hedge_is_reported():
    src = ("No released NHANES cycle carries a gender-identity item at present. "
           "We read the equity finding as an imposition rather than an erasure.")
    out = ("No NHANES cycle has ever included a gender-identity item. "
           "The equity finding is an imposition and not an erasure.")
    rep = fidelity.verify(src, out, fmt="plain")
    assert any(f.axis == "claim-strength" for f in rep.soft), fidelity.render(rep)


# --- the other direction: do not reject good work -----------------------------

def test_identity_is_clean():
    """A document compared against itself must produce nothing at all."""
    src = ("\\section{Method}\n\\subsection{Design}\\label{sec:d}\n"
           "The unit of analysis is the design cell, a model crossed with a "
           "demographic cohort, 480 per condition, and the 30 iterations inside "
           "it are repeated draws from one conditional response distribution. "
           "Pooling them would understate standard errors. "
           "Ranges run 0.17--0.21 throughout.\n")
    rep = fidelity.verify(src, src, fmt="latex")
    assert rep.ok and not rep.findings, fidelity.render(rep)


def test_faithful_rewrite_passes():
    """A real improvement -- active verbs, one reordering, same shape -- must not
    trip a single hard finding, or the gate is useless in practice."""
    src = ("\\subsection{Statistical approach}\\label{sec:stats}\n"
           "The unit of analysis in this work is taken to be the design cell, "
           "which is a model crossed with a demographic cohort, with 480 of them "
           "per condition. It should be noted that the 30 iterations contained "
           "inside each cell are repeated draws from a single conditional "
           "response distribution, and are not separate individuals. "
           "Pooling would be an error. "
           "Standard errors would be understated by a factor of four to seven if "
           "all 28,800 generations were treated as being exchangeable with one "
           "another in the analysis.\n")
    out = ("\\subsection{What a cell is, and why it is the unit}\\label{sec:stats}\n"
           "The unit of analysis is the design cell: a model crossed with a "
           "demographic cohort, 480 per condition. The 30 iterations inside each "
           "cell are repeated draws from one conditional response distribution, "
           "not separate individuals. Pooling would be an error. "
           "Treating all 28,800 generations as exchangeable understates standard "
           "errors by a factor of four to seven, which is not a rounding "
           "difference but a change in what the intervals mean.\n")
    rep = fidelity.verify(src, out, fmt="latex")
    assert rep.ok, fidelity.render(rep)


# --- end to end on the artifacts that failed ----------------------------------

@pytest.mark.skipif(not _HAVE_MANUSCRIPTS,
                    reason="set MIMESIS_FIDELITY_PAPER and MIMESIS_FIDELITY_VOICED to run")
def test_the_real_failure_is_caught():
    """The whole point, on the actual files: every corruption that shipped is a
    hard finding now, and the pass that produced them would have been rejected."""
    def block(t: str) -> str:
        return t[t.index("\\section{Method}"): t.index("\\section{Discussion}")]

    src = block(PAPER.read_text(encoding="utf-8"))
    out = block(VOICED.read_text(encoding="utf-8"))
    rep = fidelity.verify(src, out, fmt="latex")
    axes = {f.axis for f in rep.hard}
    for expected in ("scaffold", "structure", "house-style", "rhythm", "register"):
        assert expected in axes, f"{expected} missed\n{fidelity.render(rep)}"
