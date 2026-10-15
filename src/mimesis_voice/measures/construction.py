"""The four constructions the study added to this package's existing detectors.

`rhetoric.py` and `rhetoric_ext.py` find twenty-nine named constructions. The
study measured all of them as rates per 1,000 words against nine authors and
added four of its own, which are the four here. They are counted per 100
sentences (per 100 words for copula density) because all four are properties of
sentence sequence, not of word supply.

What the study's table showed for each, from
`eval_methodology/results/constructional_A0.md`:

  anaphora2        the dominant machine tell at every model size: z +5 to +7 in
                   the 4B bases. "I remember the way she had..." fifteen times
                   in one cell. Replicate-floor ratio 4.8, the highest of the
                   four.
  copula_density   above the author in every adapter at every size. The failure
                   is a passage that describes states where the author moves.
  rhetorical_question  under-produced by every base and restored by the
                   adapters, so the informative direction here is DOWN.
  long_short       the long-sentence-then-short-sentence beat, over 25 words
                   followed by 6 or fewer. Clears the floor at ratio 2.0.

All four clear the replicate floor as individual features (ratio 2.0-4.8). The
study's aggregates over the full thirty-three (mean |z|, mean positive z) do not
once winsorised, ratio 1.3-1.4, and are deliberately not ported: an aggregate
that is carried by its extreme cells is not a gate, it is a list of extreme
cells with the labels removed.

Each is reported as a signed z against the author's own corpus with sigma
floored at `SD_FLOOR`, so a construction an author never uses cannot produce an
unbounded z from one accidental hit.
"""
from __future__ import annotations

import re

from .textbase import sentences

__all__ = ["FEATURES", "SD_FLOOR", "rates"]

# Per the study: sigma floored so one hit on a never-used construction does not
# read as z = 40. 0.25 is in the same units as the feature (per 100 sentences,
# or per 100 words for copula density).
SD_FLOOR = 0.25

FEATURES = ["rhetorical_question", "anaphora2", "long_short", "copula_density"]

# Signed direction that indicates the machine, for the report. Rhetorical
# questions are the one feature whose machine failure is under-production.
MACHINE_DIRECTION = {
    "rhetorical_question": -1,
    "anaphora2": +1,
    "long_short": +1,
    "copula_density": +1,
}

LABELS = {
    "rhetorical_question": "rhetorical questions per 100 sentences",
    "anaphora2": "repeated two-word openers per 100 sentence pairs",
    "long_short": "long-then-short beats per 100 sentence pairs",
    "copula_density": "copulas per 100 words",
}

_COPULA = re.compile(r"\b(?:is|was|are|were|be|been|being|am)\b", re.IGNORECASE)
_LEAD_NONALPHA = re.compile(r"^[^A-Za-z]+")
# Copula density's denominator, kept as the study wrote it. Wider than
# textbase.word_count: it counts digits and underscore-joined tokens as words.
# The difference is small on prose and the point of a rate is that both sides of
# a comparison are counted the same way, so the study's counter is the one that
# makes these z-scores comparable to its table.
_WORD_ANY = re.compile(r"\b[\w']+\b")


def rates(text: str) -> dict[str, float]:
    """The four rates. Zeros on empty text rather than an error, because the
    caller scores whatever a model returned including nothing."""
    sents = [s.strip() for s in sentences(text) if s.strip()]
    n = len(sents)
    if n == 0:
        return {f: 0.0 for f in FEATURES}
    q = sum(1 for s in sents if s.endswith("?"))
    firsts = [tuple(_LEAD_NONALPHA.sub("", s).lower().split()[:2]) for s in sents]
    ana = sum(1 for i in range(1, n) if len(firsts[i]) == 2 and firsts[i] == firsts[i - 1])
    lens = [len(s.split()) for s in sents]
    ls = sum(1 for i in range(1, n) if lens[i - 1] > 25 and lens[i] <= 6)
    nw = len(_WORD_ANY.findall(text))
    return {
        "rhetorical_question": 100.0 * q / n,
        "anaphora2": 100.0 * ana / max(n - 1, 1),
        "long_short": 100.0 * ls / max(n - 1, 1),
        "copula_density": 100.0 * len(_COPULA.findall(text)) / max(nw, 1),
    }


def anaphora_pairs(text: str) -> list[str]:
    """The sentence pairs that share a two-word opener, for quoting.

    `anaphora2` is the strongest machine tell in the study, so the report shows
    the repeated openers rather than the rate.
    """
    sents = [s.strip() for s in sentences(text) if s.strip()]
    firsts = [tuple(_LEAD_NONALPHA.sub("", s).lower().split()[:2]) for s in sents]
    out = []
    for i in range(1, len(sents)):
        if len(firsts[i]) == 2 and firsts[i] == firsts[i - 1]:
            out.append(" ".join(firsts[i]))
    return out
