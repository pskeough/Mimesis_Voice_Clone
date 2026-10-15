"""Paragraph shape: how much the paragraph lengths vary.

Paragraph architecture as a distribution rather than a mean. The headline is
the coefficient of variation of paragraph length in words; `w1_logpara` is the
1-Wasserstein distance between two passages on log paragraph length, for the
paired case where a source exists.

Every base model in the study wrote one-size paragraphs, CV 0.18-0.43, against
authors at 0.62-0.95; adapters trained on 51 briefs or more recovered it.
Replicate floor 0.024, effect 11.4x it, and the author-vs-base separation held
on 9 of 9 authors at d >= 0.5.

This is an architecture outcome, not a quality one, and the study is explicit
about the difference. On the blind item Patrick ruled on, the cell he preferred
over-varied its paragraphs badly (CV 0.79 against the source's 0.34) and he
still preferred it, so the measure disagrees with the reader in that direction.
That is why the gate line is a two-sided band around the author's own CV and
why it is reported as a band rather than as a score: it says the paragraphing
is unlike the author's, not that the passage is worse.

`psd_z` is kept for parity with the study cache and is not used in the gate:
the study found it outlier-heavy and moved to `cv_diff` and `w1_logpara`.

A caller must check `para_n` before reading anything else: under three
paragraphs there is no shape to measure.

Paragraphs are split by `textbase.prose_paragraphs`, not by the study's
blank-line-only rule. That matters more than it sounds. Calibrated off this
repo's own creative store the author's paragraph CV came out at 0.06, because
`ingest.read_pieces` rejoins ~180-word chunks with blank lines and every
"paragraph" was one chunk. Read from the documents as written, the same author
runs a median of 0.58 -- in the study's author range of 0.62-0.95.
"""
from __future__ import annotations

import math
import statistics as st

import numpy as np

from .textbase import prose_paragraphs, sentences

__all__ = ["profile", "w1", "cv_diff", "MIN_PARAS"]

# Below three paragraphs a coefficient of variation is two numbers and their
# midpoint. The study's own passages run 4-20 paragraphs.
MIN_PARAS = 3


def w1(a: list[float], b: list[float]) -> float:
    """1-Wasserstein between two 1-D samples, via their quantile functions."""
    if not a or not b:
        return float("nan")
    q = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def profile(text: str, split=prose_paragraphs) -> dict:
    """Paragraph shape. `split` is the paragraph splitter and defaults to the
    newline-tolerant one; pass `textbase.paragraphs` for the study's exact rule.
    The two agree on all 150 of the study's own passages, which the parity test
    checks by running this default against the study's cache."""
    paras = [p for p in split(text) if p.strip()]
    pw = [len(p.split()) for p in paras]
    ps = [len([s for s in sentences(p) if s.strip()]) for p in paras]
    n = len(pw)
    cv = st.pstdev(pw) / st.mean(pw) if n > 1 and st.mean(pw) > 0 else 0.0
    cvs = st.pstdev(ps) / st.mean(ps) if n > 1 and st.mean(ps) > 0 else 0.0
    first_ratio = []
    for p in paras:
        sl = [len(s.split()) for s in sentences(p) if s.strip()]
        if len(sl) >= 2:
            first_ratio.append(sl[0] / st.mean(sl))
    return {
        "para_n": float(n),
        "para_mean_words": st.mean(pw) if pw else 0.0,
        "para_mean_sents": st.mean(ps) if ps else 0.0,
        "para_cv_words": cv,
        "para_cv_sents": cvs,
        "first_sent_ratio": st.mean(first_ratio) if first_ratio else 1.0,
        "_log_para": [math.log(max(x, 1)) for x in pw],
    }


def cv_diff(text_profile: dict, reference_cv: float) -> float:
    """Signed distance of a passage's paragraph CV from the author's mean.

    Negative is the machine failure mode: paragraphs all one size.
    """
    return text_profile["para_cv_words"] - reference_cv
