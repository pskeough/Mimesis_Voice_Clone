"""Cadence: whether the sentence-length series is metronomic.

Finite-size-corrected burstiness A_n (Kim and Jo 2016, Phys. Rev. E 94, 032311)
over the sentence-length series, plus the quantile ratios q90/q50 and q50/q10.

    r    = sd / mean over n sentence lengths
    A_n  = ( sqrt(n+1) r - sqrt(n-1) ) / ( (sqrt(n+1) - 2) r + sqrt(n-1) )

A_n is -1 for a perfectly regular series, 0 for random, 1 for maximally bursty,
and unlike the raw Goh-Barabasi B it does not depend on n. The correction is
not cosmetic at this scale: B's ceiling is near 1 - 2/sqrt(n), and passages here
carry 20-60 sentences.

The 1.7B base was the most regular writer in the study at A_n -0.50 against
Patrick's -0.38, and it is the cell he called monotonous. Replicate floor 0.014,
effect 10.4x it.

Lag-1 and lag-3 autocorrelation of sentence length were in the study's version
of this profile and are deliberately NOT ported: their adapter effects were 0.7x
and 0.5x their own replicate floors, which is to say inside the noise. The
package already has `cadence.py` measuring thirteen rhythm features on a
different footing; this module adds the one number that survived the floor.
"""
from __future__ import annotations

import math
import statistics as st

import numpy as np

from .textbase import sent_lengths

__all__ = ["MIN_SENTS", "burstiness_an", "profile"]

# Below four sentences a length series has no shape. The study's profile
# returned flat zeros under this and the gate must not read those as regular.
MIN_SENTS = 4


def burstiness_an(x: list[int]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = st.mean(x)
    s = st.pstdev(x)
    if m == 0:
        return 0.0
    r = s / m
    num = math.sqrt(n + 1) * r - math.sqrt(n - 1)
    den = (math.sqrt(n + 1) - 2) * r + math.sqrt(n - 1)
    return num / den if den != 0 else 0.0


def profile(text: str) -> dict[str, float]:
    x = sent_lengths(text)
    if len(x) < MIN_SENTS:
        return {"burst_an": 0.0, "burst_b": 0.0, "q90_q50": 1.0, "q50_q10": 1.0,
                "sent_cv": 0.0, "n_sent": float(len(x))}
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9])
    m, s = st.mean(x), st.pstdev(x)
    return {
        "burst_an": burstiness_an(x),
        "burst_b": (s - m) / (s + m) if (s + m) > 0 else 0.0,
        "q90_q50": q90 / q50 if q50 else 1.0,
        "q50_q10": q50 / max(q10, 1.0),
        "sent_cv": s / m if m else 0.0,
        "n_sent": float(len(x)),
    }
