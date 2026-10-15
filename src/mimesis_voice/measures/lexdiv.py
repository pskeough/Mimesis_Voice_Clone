"""Windowed lexical diversity: a narrowing vocabulary is looping in disguise.

Hapax proportion, MATTR-50 and MTLD over a fixed 300-token window from the start
of the text, so passages of unequal length are comparable. McCarthy and Jarvis
2010: MTLD and MATTR are length-stable, raw type-token ratio and hapax are not,
which is why the window exists rather than a whole-text count.

Adapters narrowed vocabulary at every model size in the study; bases widened
with scale. Replicate floor 0.010 on hapax, effect 4.3x it, author-vs-base
separation on 8 of 9 authors at d >= 0.5. The narrowest Chesterton 14B adapter
cell turned out to be a four-way permutation loop in Father Brown dialogue, so
narrowing and looping are one event, seen from two sides.

Reported as a signed distance from the author's own mean, two-sided. Vocabulary
far above the author is as unlike them as vocabulary far below, and the
above-direction is what a model does when it is reaching for register.

Texts under `MIN_TOKENS` return `{}`: a 300-token window over 140 tokens is not
a windowed measure.
"""
from __future__ import annotations

import re

__all__ = ["MIN_TOKENS", "WINDOW", "mattr", "measure", "mtld", "tokens"]

WINDOW = 300
MIN_TOKENS = 200
_TOK = re.compile(r"[A-Za-z][A-Za-z'-]*")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


def mattr(toks: list[str], w: int = 50) -> float:
    """Moving-average type-token ratio over a sliding window of `w`."""
    if len(toks) < w:
        return len(set(toks)) / len(toks) if toks else 0.0
    return sum(len(set(toks[i:i + w])) / w for i in range(len(toks) - w + 1)) / (len(toks) - w + 1)


def _mtld_one(toks: list[str], thr: float = 0.72) -> float:
    factors, types, count = 0.0, set(), 0
    for t in toks:
        types.add(t)
        count += 1
        if len(types) / count <= thr:
            factors += 1
            types, count = set(), 0
    if count > 0:
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - thr) if ttr > thr else 1.0
    return len(toks) / factors if factors else float(len(toks))


def mtld(toks: list[str], thr: float = 0.72) -> float:
    """Bidirectional MTLD. Both directions because a passage that narrows only
    at its end scores differently forwards and backwards, and the mean of the
    two is the standard estimator."""
    if len(toks) < 50:
        return float("nan")
    return (_mtld_one(toks, thr) + _mtld_one(list(reversed(toks)), thr)) / 2


def measure(text: str) -> dict[str, float]:
    toks = tokens(text)
    if len(toks) < MIN_TOKENS:
        return {}
    w = toks[:WINDOW]
    counts: dict[str, int] = {}
    for t in w:
        counts[t] = counts.get(t, 0) + 1
    v1 = sum(1 for c in counts.values() if c == 1)
    return {
        "hapax_w300": v1 / len(w),
        "ttr_w300": len(counts) / len(w),
        "mattr50_w300": mattr(w, 50),
        "mtld_w300": mtld(w),
        "n_tok": float(len(toks)),
    }
