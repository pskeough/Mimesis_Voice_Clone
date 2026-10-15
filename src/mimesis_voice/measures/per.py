"""Proposition-echo rate: the passage restating itself in fresh wording.

A sentence is an echo when more than `ECHO_FRAC` of its unordered content-word
pairs already appeared in an earlier sentence. PER is the share of echoing
sentences among sentences carrying at least two content words. Pairs rather
than unigrams, and 0.25 rather than a near-duplicate threshold, so it fires on
restatement that reuses no whole phrase:

    Because we are terrible parents... Because we do not... Because we are not...

The study's primary coherence row. Authors run 0.01-0.04; every model run sat
above that, and on the blind item Patrick ruled on, the candidate he said lost
the story scored 0.167 against 0.000 for the one he picked (item-1 coherence
rank correlation +0.95). Replicate floor 0.011, adapter effect 5.9x it.

Computed over 600-word windows. Uncapped, the rate correlates +0.41 with word
count and up to +0.57 on the 1.7B runs, so an uncapped number ranks long
passages as incoherent for being long.

The study took the first 600 words and stopped, which is right for ranking
thousands of cells against each other and wrong for gating one draft: a model
that loops does it late, and truncating discards the half of the draft where the
looping is. On the blind item, the candidate Patrick said lost the story scores
0.167 whole, 0.000 on its first 600 words, and 0.200 on its worst window. So the
draft is scored in 600-word windows at a 300-word stride and the worst window is
reported. Every window is exactly the object the study validated -- a passage of
at most 600 words scored by `per` -- and the overlap catches an echo that
straddles a boundary. Echoes spanning more than 600 words are missed, which
under-reports rather than over-reports.

This measures construction, not voice: it scores fluent generic prose best. It
must never be folded into a voice score, and never into any aggregate at all --
three coherence features sat averaged inside the study's geometry index for
weeks and the looping they measured was invisible for all of it.

Grounding: Holtzman et al. 2020 on degeneration; Guan and Huang 2020 for
UNION's repetition perturbation.
"""
from __future__ import annotations

import itertools
import re

from .textbase import sentences

__all__ = ["ECHO_FRAC", "STRIDE", "WORD_CAP", "content_words", "echo_sentences",
           "per", "windows", "worst_window"]

ECHO_FRAC = 0.25
WORD_CAP = 600
STRIDE = 300

STOP = set("""
a an the and or but nor so yet for of in on at to from by with without into onto over under
about above below between through during before after again further then once here there
when where why how all any both each few more most other some such no not only own same than
too very can will just don should now is are was were be been being have has had having do
does did doing i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves what which
who whom this that these those am as if because until while up down out off would could
might must shall may also one two said says say like get got go went come came see saw made
make take took thing things way still even much many back way
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def content_words(sent: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(sent) if len(w) > 2 and w.lower() not in STOP]


def per(text: str) -> dict[str, float]:
    """{"per": echo share, "per_n_sent": sentences that could echo}."""
    seen: set[tuple[str, str]] = set()
    n = echo = 0
    for s in sentences(text):
        cw = sorted(set(content_words(s)))
        if len(cw) < 2:
            continue
        pairs = set(itertools.combinations(cw, 2))
        n += 1
        if len(pairs & seen) / len(pairs) > ECHO_FRAC:
            echo += 1
        seen |= pairs
    return {"per": echo / n if n else 0.0, "per_n_sent": float(n)}


def echo_sentences(text: str) -> list[str]:
    """The sentences PER counted as echoes, in order.

    The report quotes these rather than printing the rate alone. A rate is a
    number to argue with; the three sentences that repeat each other are the
    finding.
    """
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for s in sentences(text):
        cw = sorted(set(content_words(s)))
        if len(cw) < 2:
            continue
        pairs = set(itertools.combinations(cw, 2))
        if len(pairs & seen) / len(pairs) > ECHO_FRAC:
            out.append(s.strip())
        seen |= pairs
    return out


def windows(text: str, cap: int = WORD_CAP, stride: int = STRIDE) -> list[str]:
    """The draft cut into overlapping `cap`-word windows.

    A text at or under `cap` is one window and scores exactly as the study
    scored it. A trailing stub shorter than half a window is dropped rather than
    scored: an echo rate over 90 words is three sentences and a coin flip.
    """
    w = text.split()
    if len(w) <= cap:
        return [text]
    out: list[str] = []
    i = 0
    while i < len(w):
        chunk = w[i:i + cap]
        if len(chunk) < cap // 2 and out:
            break
        out.append(" ".join(chunk))
        i += stride
    return out


def worst_window(text: str, cap: int = WORD_CAP, stride: int = STRIDE) -> dict:
    """Highest echo rate over the draft's windows, and where it was.

    `index` and `n_windows` let the caller say which stretch of the draft the
    rate came from, which is the difference between a number and a location.
    """
    ws = windows(text, cap, stride)
    scored = [(per(w), w) for w in ws]
    best = max(range(len(scored)), key=lambda i: scored[i][0]["per"])
    v, w = scored[best]
    return {
        "per": v["per"],
        "per_n_sent": v["per_n_sent"],
        "index": best,
        "n_windows": len(ws),
        "start_word": best * stride,
        "text": w,
    }
