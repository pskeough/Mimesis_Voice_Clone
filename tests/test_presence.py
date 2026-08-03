"""Regression tests for the authorial-presence tier.

The cases here are the ones that broke earlier versions of the module, kept as tests because each
represents a real failure of a plausible design:

  * a 371-word conclusion the fingerprint scored CLEAN and a reader called machine-written on sight
  * a published EMNLP limitations section that a single-class absence rule false-positived on
  * two of the author's own short technical pieces that an all-classes-zero rule false-positived on

Any change to the thresholds has to keep all three verdicts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import presence  # noqa: E402

FLAT = """
Four widely deployed cost-tier models generate psychiatric patients that pass individual inspection
and fail population comparison. Coherence is real but model-specific, ranging from 0.35% to 6.24%
gateway violations. The fidelity failure survives every re-derived population anchor and every
composition frame, and roughly half of it survives an unconditioned clinical one, at +2.8 to +5.5
points on cisgender personas and larger pooled. Underneath both sits an instability that
regeneration alone produces, flipping a third of patients across diagnostic categories and falling
hardest on transgender personas, who also receive the most structurally distinctive representation
in the design and the only one no national instrument can check. The audit rests on benchmarks
re-derived from primary microdata, and the difference that discipline made to this paper's own
conclusions is documented here. Audit claims inherit the validity of their ground truth. We release
the pipeline, the constants, and the changelog so that inheritance is checkable. The unit of
analysis is the design cell, a model crossed with a demographic cohort, of which there are 480 per
condition. The 30 iterations inside a cell are repeated draws from one conditional response
distribution, not distinguishable individuals, so every residual, contrast and interval reported
here is computed on cell means. Cells within a model are not independent of one another either, and
two-way cluster-robust errors over model and cohort run 1.8 to 4.7 times the between-cell errors.
Residual t statistics run from 10 to 30, so nothing changes sign or significance, but those
intervals are within-model precision rather than intervals on the model class. NHANES is a complex
multistage sample and its anchors carry sampling error of their own, which we compute by Taylor
linearisation and find small enough to treat the anchors as fixed. The population distribution is
heavily right-skewed, 77.1% of adults at 4 or below and a long thin tail. No simulated distribution
has that shape.
"""

AUTHORED = """
In summary, our results give a consistent narrative about four widely deployed cost-tier models.
They generate psychiatric patients that pass individual inspection and fail population comparison.
Two things about this work surprised us and are worth stating plainly. Where the models depart from
additivity across demographic axes, they compress rather than compound, which runs against the
direction an intersectional account would predict. And the only severe cases anywhere in the corpus
belong to transgender personas. Neither result is one we designed the study to find. We had expected
the racial contrasts to be attenuated, since that is what a model with a weak grip on a small
population difference should produce, and two of the three inverted instead. What we would ask a
reader to take from this is narrower than the results themselves. Audit claims inherit the validity
of their ground truth, and an audit that never returns to primary data cannot tell the difference
between a finding and a benchmark error. We know that concretely rather than in principle. We
suspect, though we cannot show it from one paper, that this is not the only audit whose ground truth
would fail re-derivation. The pipeline and the constants are released so the inheritance here is
checkable by anyone who doubts it.
"""

# A real published limitations section. Carries no expectation-violation markers at all, which is
# normal for the form, so a rule that gates on any single class flags it.
PUBLISHED = """
While we fill a critical gap since there is no existing work on systematically detecting stereotypes
in simulations, our measure is limited in scope: it is not a comprehensive evaluation of the quality
of a simulation. We quantify susceptibility to caricature, which is a particular failure case. Our
method may yield false positives, that is, simulations that seem acceptable based on our method but
have other problems. Avoiding caricature is a necessary but insufficient criterion for simulation
quality; our metric should be used in tandem with other evaluations. As a pilot study for a
recently-emerging direction of work, we hope to lay the groundwork for a more comprehensive
evaluation of simulations in the future, perhaps in tandem with human evaluation. We notice that a
small set of identified words seem to be explicitly anti-stereotypical. We posit that these words
might in fact result from bias mitigation mechanisms. First, they suggest that current evaluation
practices may give practitioners a false sense of security. Finally, we leave these directions to
future work.
"""

# One of the author's own pieces: heavy signposting, no interpretation, legitimately authored.
TERSE_REAL = """
The approach to grading utilizes a local language model as the judge following standard practice,
with additional controls given to remove known failure modes. One such control is the use of blind
head-to-head judging. In each instance the judge is given the question, a correct answer reference,
supporting passages, and two labeled candidate answers. We obscure the identity of both candidates
to control for potential model alignment bias. Running each comparison in both orders, we only count
a decisive win when both respond with the name of the same winner, treating mixed results as a tie.
First, the evaluation is restricted to a single model family. Additionally, while rigorous the
annotation process is based on subjective judgements that can vary across raters. Finally, the
specific set of behaviors we focus on does not claim to capture the full space of possible failures.
The arena is only a 50-question preview, and thus while the ranking it produces is stable, the exact
size of each gap needs the full 345-question run for reliability of measurement.
"""


def _flagged(text, cal=None):
    return bool(presence.render_lines(presence.analyze(text, cal or presence.PresenceCalibration.default())))


def test_flat_draft_is_flagged():
    """The motivating case. Every style check passed; nobody was in the text."""
    res = presence.analyze(FLAT)
    assert res["measurable"], "371 words must be above the absence threshold"
    assert res["densities"]["epistemic"] == 0.0
    assert res["densities"]["expectation"] == 0.0
    assert _flagged(FLAT)


def test_authored_draft_passes():
    assert not _flagged(AUTHORED)


def test_published_limitations_section_passes():
    """No expectation markers, and that is normal for the form. Gating any single class breaks."""
    assert presence.analyze(PUBLISHED)["densities"]["expectation"] == 0.0
    assert not _flagged(PUBLISHED)


def test_terse_real_piece_passes():
    """Signposts heavily, interprets nothing. Presence shows up as organisation, which counts."""
    assert presence.analyze(TERSE_REAL)["densities"]["epistemic"] >= 0.0
    assert not _flagged(TERSE_REAL)


def test_short_text_is_not_judged():
    assert not _flagged("We ran the model. The output was wrong.")


def test_stance_markers_are_not_weak_hedges():
    """The bug this module was also written to fix: "seems to us" is stance, not softening."""
    stance = "It seems to us the more specific reading, and we cannot say which mechanism drives it."
    mush = "The result was sort of unclear and somehow the model kind of failed in a sense."
    assert presence.weak_hedge_density(stance) == 0.0
    assert presence.weak_hedge_density(mush) > 0.0
    assert presence.STANCE_MARKERS.search(stance)


def test_field_floor_survives_a_thin_corpus():
    """A corpus with no expectation markers must not license a floor of zero."""
    cal = presence.calibrate([TERSE_REAL * 3])
    assert cal.floors.get("expectation", 0.0) >= presence.FIELD_FLOOR["expectation"]


def test_corpus_only_classes_are_never_gated():
    """Signposting is measured, never gated: a short-form corpus signposts at long-form-implausible
    rates and would flag every extended draft."""
    cal = presence.calibrate([TERSE_REAL * 3])
    assert "signpost" not in cal.floors
