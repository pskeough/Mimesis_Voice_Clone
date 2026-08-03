"""Authorial presence: is there a person visible in this draft?

Every other check in the scrubber asks whether the prose *matches a distribution*. A draft can pass
all of them and still read as machine-written, because the thing that gives writing a human author
is not its sentence-length variance. It is that somebody in the text notices, expects, doubts,
concedes, and is occasionally wrong.

The motivating case, measured on a 21,000-word research manuscript that the fingerprint scored as
clean and that a reader immediately called machine-written:

    per 1,000 words                   published human papers      that manuscript
    epistemic verbs (find/suspect)          0.31, 0.32                  0.00
    expectation violation (surprised)       0.16, 0.16                  0.00

Zero of each, across the whole paper. Nobody ever found anything, expected anything, or was
surprised by anything. Every sentence was a verified assertion. The fingerprint could not see it
because assertion density is not one of its features, and no wordlist could see it because the tell
is an *absence*.

Two design decisions follow from that, and both cut against how the rest of this package works.

**Presence is a floor, not a ceiling.** Every existing check hunts for things the author does not
do. This one hunts for something the author *does* do and the draft does not. It fires on too
little, never too much.

**The reference is not only the corpus.** The author's own note on this: "i dont care too much if
it perfectly reflects my database, that doesnt matter." A short-form corpus of technical reports
will not contain the epistemic density of a full research paper, so calibrating presence purely
against it would enforce the very flatness we are trying to detect. The floor is therefore the
maximum of a corpus-derived value and a published-field floor, so a thin corpus cannot drag the
target below what real published prose carries.

**A correction to the hedge rule this module also fixes.** The scrubber's hedge list conflates two
different things under one ceiling:

    softening        "sort of", "kind of", "somehow", "in a sense"     weak writing, penalize
    epistemic stance "suggests", "indicates", "we suspect", "seems"    a thinking author, reward

Because both were penalized together, a calibration on a corpus that avoids the first produced a
hedge ceiling of zero, which then flagged the second. In practice that told a writer to strip the
exact markers that make prose read as human. STANCE_MARKERS below are counted as presence, never
against the hedge ceiling; WEAK_HEDGES keep the old behaviour.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"\b[\w']+\b")

# --- the five presence classes ------------------------------------------------
#
# Each is a way a person shows up in prose. They are counted separately rather than pooled because
# they fail independently: a draft can narrate its own method ("we ran", "we chose") while never
# once conceding a doubt, and that draft reads like a lab notebook rather than like a paper.

EPISTEMIC = re.compile(
    r"\b(?:we|i)\s+(?:\w+\s+){0,2}?"
    r"(?:find|found|think|thought|believe|suspect|suspected|read|interpret|take|took|argue|"
    r"argued|posit|posited|judge|judged|doubt|doubted|conclude|concluded|infer|inferred|"
    r"understand|understood|hold|held|treat|treated|prefer|preferred)\b"
    r"|\b(?:suggests?|suggesting|suggested|indicates?|indicating|indicated|implies|implying)\b",
    re.IGNORECASE,
)

EXPECTATION = re.compile(
    r"\bsurpris\w+\b|\bunexpected(?:ly)?\b|\bcounterintuitive\b|\bcontrary to\b|"
    r"\b(?:we|i)\s+(?:had\s+)?expect\w*\b|\bwe did not expect\b|\bagainst (?:our )?expectation\w*\b|"
    r"\bwe went in expecting\b|\bturned out\b|\bto our surprise\b|\bruns? against\b|"
    r"\bwe had (?:assumed|hoped|thought)\b|\bstriking\w*\b",
    re.IGNORECASE,
)

# Stance markers: a claim held at a stated strength. Counted as presence, and explicitly NOT
# counted against the hedge ceiling. This is the class the old calibration was suppressing.
STANCE_MARKERS = re.compile(
    r"\b(?:seems?|seemed|appears?|appeared)\s+(?:to|that)\b|\bit seems to (?:us|me)\b|"
    r"\b(?:probably|plausibly|arguably|presumably|likely|apparently)\b|"
    r"\b(?:we|i)\s+(?:cannot|can't|could not|do not|don't)\s+(?:say|tell|settle|separate|show|know)\b|"
    r"\bremains? open\b|\bis not settled\b|\bwe are not sure\b|\bhard to say\b",
    re.IGNORECASE,
)

SIGNPOST = re.compile(
    r"(?:^|\.\s+)(?:First|Second|Third|Finally|In summary|To summarize|In short|Regarding|"
    r"Taken together|On balance|That said|By contrast|Put differently)\b",
    re.IGNORECASE | re.MULTILINE,
)

AGENCY = re.compile(
    r"\b(?:we|i)\s+(?:ran|run|chose|choose|decided|decide|looked|look|went|go|tried|try|built|"
    r"build|asked|ask|set out|wanted|want|dropped|drop|kept|keep|rejected|reject|withdrew|"
    r"withdraw|added|add|checked|check|measured|measure|tested|test)\b",
    re.IGNORECASE,
)

CLASSES: dict[str, re.Pattern] = {
    "epistemic": EPISTEMIC,
    "expectation": EXPECTATION,
    "stance": STANCE_MARKERS,
    "signpost": SIGNPOST,
    "agency": AGENCY,
}

# Softening that genuinely weakens a sentence. Kept apart from STANCE_MARKERS so the hedge ceiling
# can penalize mush without penalizing thought.
WEAK_HEDGES = re.compile(
    r"\bsort of\b|\bkind of\b|\bsomehow\b|\bsomething like\b|\bin some way\b|\bof sorts\b|"
    r"\ba bit\b|\brather more\b|\bmore or less\b|\bto some extent\b|\bin a sense\b",
    re.IGNORECASE,
)

# Published-field floors, per 1,000 words, measured on two peer-reviewed papers in the domain the
# motivating manuscript targets (EMNLP 2023 and ACL 2023, full text minus references). They are
# deliberately modest: the point is a floor no real paper falls under, not a target to hit.
#
#   epistemic    0.32, 0.31  ->  0.25
#   expectation  0.16, 0.16  ->  0.12
#
# The other three classes were not measured on those papers and carry no field floor; they are
# reported for the writer's information and only ever flagged against the corpus.
FIELD_FLOOR: dict[str, float] = {"epistemic": 0.25, "expectation": 0.12}

# Two thresholds, because absence and density need different amounts of text.
#
# ABSENCE_WORDS: above this, a complete absence of interpretive moves is itself the signal. Any
# stretch of analytical prose this long by a human author almost always contains at least one
# "we read this as" or "the pattern suggests". Zero is not a low score, it is a different kind of
# document. Set at 220 words because the scrubber is usually handed a section, not a manuscript,
# and a check that only fires on whole papers would never fire at all: the first version of this
# module used 400 and scored a known-flat draft as passing because it was 371 words long.
#
# DENSITY_WORDS: below this a per-1,000-word rate is too noisy to compare against a floor. One
# marker in a 300-word excerpt reads as 3.3 per 1k, which says nothing. Density floors are only
# applied above it.
# Total presence, per 1,000 words, under which a passage reads as having no author. Measured
# against real text rather than chosen: the author's own pieces score 3.9, 12.8, 15.9 and 25.0, a
# published EMNLP limitations section scores 21.9, and the flat manuscript conclusion that
# motivated this module scores 0.0. Anything in the low single digits clears.
ABSENCE_TOTAL = 1.5
ABSENCE_WORDS = 220
DENSITY_WORDS = 1500
MIN_WORDS = ABSENCE_WORDS


@dataclass
class PresenceCalibration:
    """Per-class presence floors, per 1,000 words."""

    floors: dict[str, float]
    corpus_medians: dict[str, float]
    n_pieces: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"floors": self.floors, "corpus_medians": self.corpus_medians,
                 "n_pieces": self.n_pieces},
                indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PresenceCalibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(floors=d.get("floors", dict(FIELD_FLOOR)),
                   corpus_medians=d.get("corpus_medians", {}),
                   n_pieces=d.get("n_pieces", 0))

    @classmethod
    def default(cls) -> "PresenceCalibration":
        return cls(floors=dict(FIELD_FLOOR), corpus_medians={}, n_pieces=0)


def densities(text: str) -> dict[str, float]:
    """Per-class hits per 1,000 words."""
    n = len(_WORD_RE.findall(text)) or 1
    return {k: len(pat.findall(text)) * 1000 / n for k, pat in CLASSES.items()}


def weak_hedge_density(text: str) -> float:
    """Softening-only hedges per 200 words, excluding epistemic stance markers."""
    n = len(_WORD_RE.findall(text)) or 1
    return len(WEAK_HEDGES.findall(text)) * 200 / n


def calibrate(texts: list[str]) -> PresenceCalibration:
    """Derive presence floors from corpus pieces, floored at the published-field values.

    The corpus contributes its 25th percentile rather than its median: the floor should be a level
    the author clears in most pieces, not a level half of them fail. Where the corpus sits below the
    field floor, the field floor wins, which is the whole point of having one. A corpus of terse
    technical notes should not license a paper with nobody in it.
    """
    per_class: dict[str, list[float]] = {k: [] for k in CLASSES}
    n = 0
    for t in texts:
        if len(_WORD_RE.findall(t)) < MIN_WORDS:
            continue
        n += 1
        for k, v in densities(t).items():
            per_class[k].append(v)

    floors, medians = {}, {}
    for k, vals in per_class.items():
        if vals:
            medians[k] = round(statistics.median(vals), 3)
            s = sorted(vals)
            p25 = s[max(0, int(round(0.25 * (len(s) - 1))))]
        else:
            p25 = 0.0
        # Corpus values may RAISE a field floor, never introduce a new gated class. The research
        # corpus signposts at 7.8 per 1k because its pieces are 200-word notes; carrying that over
        # as a floor would flag every long-form draft for insufficient "First"/"Finally". Classes
        # outside FIELD_FLOOR are measured and reported, never gated.
        if k not in FIELD_FLOOR:
            continue
        floor = max(p25, FIELD_FLOOR[k])
        if floor > 0:
            floors[k] = round(floor, 3)
    return PresenceCalibration(floors=floors, corpus_medians=medians, n_pieces=n)


def analyze(text: str, cal: PresenceCalibration | None = None) -> dict:
    """Score a draft's authorial presence against the floors.

    Returns the per-class densities, which floors are missed, and whether the draft is long enough
    for the measurement to mean anything.
    """
    cal = cal or PresenceCalibration.default()
    n_words = len(_WORD_RE.findall(text))
    d = densities(text)
    measurable = n_words >= ABSENCE_WORDS
    # Density floors only where the rate is stable. Absence is checked from a much lower word count,
    # since zero interpretive moves in 220 words of analysis is informative in a way that
    # "0.19 per 1k against a floor of 0.25" is not.
    missed = (
        [(k, d.get(k, 0.0), f) for k, f in cal.floors.items() if d.get(k, 0.0) < f]
        if n_words >= DENSITY_WORDS else []
    )
    # The absence rule is keyed on TOTAL presence across all five classes, not on any one of them.
    # A person shows up in prose in several ways and need not use all of them: a passage can
    # interpret nothing and still be visibly authored by organising the argument out loud. Two
    # earlier versions got this wrong. Flagging a single absent class false-positived on a real
    # EMNLP limitations section, and requiring epistemic, expectation and stance all to be zero
    # still flagged 2 of 4 of the author's own pieces, both short technical passages that signpost
    # heavily and interpret nothing. Total presence separates them cleanly: those pieces score 3.9
    # and 25.0, while the flat manuscript that motivated this module scores 0.0.
    total = sum(d.values())
    both_absent = measurable and total < ABSENCE_TOTAL
    absent = (
        [k for k in ("epistemic", "expectation") if d.get(k, 0.0) == 0.0]
        if n_words >= DENSITY_WORDS else
        (["epistemic", "expectation"] if both_absent else [])
    )
    return {
        "measurable": measurable,
        "density_measurable": n_words >= DENSITY_WORDS,
        "n_words": n_words,
        "densities": {k: round(v, 3) for k, v in d.items()},
        "floors": cal.floors,
        "missed": sorted(missed, key=lambda x: x[1] - x[2]),
        "absent": absent,
        "total": round(sum(d.values()), 3),
        "weak_hedges_per200": round(weak_hedge_density(text), 3),
    }


# Repair guidance per class. Deliberately concrete: "add authorial presence" is not actionable, and
# a writer told to hedge more will produce mush. Each line names the move and warns off the failure
# mode of overdoing it.
ADVICE: dict[str, str] = {
    "epistemic": (
        "no first-person reading of the evidence. Somewhere the text should say what you make of a "
        "result, not only what the result is: \"we read this as\", \"we take it to mean\", "
        "\"the pattern suggests\". Attach these to interpretations you actually hold"
    ),
    "expectation": (
        "nothing in the draft ran against expectation. Real work surprises its author. Where a "
        "result contradicted what you set out to find, say so: \"we had expected X and found Y\". "
        "Only where it is true, and it is worth checking whether the absence means the draft is "
        "reporting rather than arguing"
    ),
    "stance": (
        "every claim is stated at the same strength. Mark the ones you are less sure of, and say "
        "plainly what the design cannot settle"
    ),
    "signpost": (
        "the argument does not tell the reader where it is. \"First\", \"Regarding\", "
        "\"In summary\", \"Taken together\" are how a person walks someone through reasoning"
    ),
    "agency": (
        "nobody appears to have done anything. Methodological choices read as facts about the "
        "world rather than as decisions someone made and could have made differently"
    ),
}


def _sentence(text: str) -> str:
    """Upper-case the first letter only. ``str.capitalize`` lower-cases everything after it, which
    mangled quoted examples like '"we read this as"' into lower case mid-advice."""
    return text[:1].upper() + text[1:] if text else text


def render_lines(res: dict, author: str = "the author") -> list[str]:
    """Report lines for the scrub output. Empty when presence clears every floor."""
    if not res.get("measurable"):
        return []
    out = []
    absent = res.get("absent", [])
    if "epistemic" in absent and "expectation" in absent:
        out.append(
            f"- [HIGH] Nobody is visibly thinking in this draft. Across {res['n_words']:,} words "
            f"there is no point where {author} finds, reads, doubts, or is surprised by anything. "
            f"Prose can clear every other check here and still read as machine-written for this "
            f"reason alone, and it is the one tell that is an absence rather than a pattern. "
            f"{_sentence(ADVICE['epistemic'])}."
        )
    else:
        for k in absent:
            out.append(f"- [HIGH] No {k} presence at all in {res['n_words']:,} words. "
                       f"{_sentence(ADVICE[k])}.")
    for k, got, floor in res["missed"]:
        if k in absent:
            continue
        out.append(
            f"- [MEDIUM] Thin authorial presence ({k}): {got:.2f} per 1k words against a floor of "
            f"{floor:.2f}. {_sentence(ADVICE[k])}."
        )
    return out
