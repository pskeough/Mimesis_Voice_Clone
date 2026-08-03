"""Order-aware cadence features: the rhythm the 13-feature fingerprint cannot see.

Every feature in ``fingerprint.FEATURES`` is an order-invariant aggregate -- a
mean, a standard deviation, a percentage, a per-100-word rate. Shuffle a text's
sentences and all thirteen are unchanged. Sort them shortest-to-longest, a
monotone ramp no writer produces, and they are *still* unchanged: measured on
the creative corpus, sorting moved mean RMS-z by +0.0023 and flagged 0/93 pieces
as off-voice (``evals/cadence_blindness.py``).

So the gate was selecting candidates on a metric that is provably blind to
pacing. That is the mechanism behind "it uses my words but it doesn't sound like
me": vocabulary, punctuation density and sentence-length *distribution* were
being matched while the *sequence* -- the build, the punch, the run of three
short lines before a long one -- was unconstrained.

These 13 features are all functions of sentence ORDER or of within-sentence
clause structure. They are deliberately scale-free (ratios, correlations,
normalized deltas) so a 200-word flash piece and a 2,000-word chapter are
comparable, and they are pure stdlib so calibration stays milliseconds-cheap.

Design notes:

- Sentence *class* (short/mid/long) is cut at the author's own terciles, not at
  fixed word counts. A writer of long sentences has their own idea of "short",
  and a fixed 8-word cut would call one author's entire corpus "long".
- Autocorrelation is computed on z-scored lengths within the piece, so it
  measures shape, not length.
- Clause features use comma/semicolon/colon boundaries as a cheap proxy for
  syntactic clauses. This is not a parse, and it is not meant to be: it is a
  rhythm signal, and punctuation is where a reader actually breathes.
"""
from __future__ import annotations

import math
import re
import statistics

CADENCE_FEATURES = (
    "sl_autocorr1",        # lag-1 autocorrelation of sentence lengths
    "sl_autocorr2",        # lag-2
    "sl_delta_abs",        # mean |L[i+1]-L[i]| / mean L  -- jaggedness
    "sl_delta_sd",         # sd of successive differences / mean L
    "sl_dir_changes",      # fraction of positions where the length trend flips
    "sl_run_mean",         # mean run length of same-class sentences
    "sl_trans_entropy",    # entropy of the 3x3 class transition matrix (bits)
    "sl_short_after_long", # P(short | previous long) -- the punch after a build
    "para_open_ratio",     # mean first-sentence length / mean sentence length
    "para_close_ratio",    # mean last-sentence length  / mean sentence length
    "clause_len_mean",     # mean words per clause
    "clause_len_cv",       # clause-length coefficient of variation
    "clause_per_sent",     # mean clauses per sentence
)

# Rewrite instructions per feature, as (too-high advice, too-low advice).
#
# A rewrite prompt that says "sl_autocorr1 is +2.3 SD" tells a language model
# nothing it can act on. These are the same facts in the vocabulary a writer
# uses. Kept next to the feature definitions so the two cannot drift apart.
HINTS: dict[str, tuple[str, str]] = {
    "sl_autocorr1": (
        "sentences cluster by length -- long ones bunch together, then short ones do; break up the clumps",
        "sentence lengths alternate too mechanically; let two similar-length sentences sit together sometimes",
    ),
    "sl_autocorr2": (
        "the rhythm repeats on a fixed cycle; vary the pattern",
        "there is no medium-range shape; let a passage build over several sentences",
    ),
    "sl_delta_abs": (
        "sentence-to-sentence length jumps are too violent; smooth some transitions",
        "consecutive sentences are too close in length, giving a flat, metronomic pace; vary them more",
    ),
    "sl_delta_sd": (
        "the size of the length changes is itself too erratic",
        "every length change is the same size; the pacing is too regular",
    ),
    "sl_dir_changes": (
        "the length zigzags up-down-up-down too regularly; let a run head one direction",
        "sentence lengths drift in long monotone runs; the author reverses direction more often",
    ),
    "sl_run_mean": (
        "too many same-length sentences run consecutively; interrupt the runs",
        "the author holds a register longer; stop switching length every sentence",
    ),
    "sl_trans_entropy": (
        "the short/medium/long ordering is more scattered than the author's",
        "the short/medium/long ordering is too predictable and mechanical",
    ),
    "sl_short_after_long": (
        "too many long sentences are followed by a short punch; the move is overused",
        "long sentences rarely land on a short one afterward; the author punches more",
    ),
    "para_open_ratio": (
        "paragraphs open with sentences that run long; the author opens shorter",
        "paragraphs open too abruptly; the author opens with more weight",
    ),
    "para_close_ratio": (
        "paragraphs end on long trailing sentences; the author closes tighter",
        "paragraphs end too clipped; the author's closing lines carry more length",
    ),
    "clause_len_mean": (
        "clauses between commas run long; break the internal phrasing shorter",
        "clauses are chopped short; let phrases extend before the comma",
    ),
    "clause_len_cv": (
        "clause lengths inside sentences vary wildly; even out the internal phrasing",
        "every clause is the same length, making sentences internally monotonous",
    ),
    "clause_per_sent": (
        "sentences carry too many comma-separated clauses; simplify some",
        "sentences are too single-clause; the author builds more internal structure",
    ),
}


def hint(feature: str, z: float) -> str | None:
    """Plain-language rewrite instruction for a deviating cadence feature."""
    pair = HINTS.get(feature)
    if not pair:
        return None
    return pair[0] if z > 0 else pair[1]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z']+")
_CLAUSE_SPLIT = re.compile(r"[,;:]")

# Sentence-class terciles. Defaults are only used when no calibration is
# supplied (single-text scoring before a corpus exists); calibrate() overwrites
# them with the author's own 33rd/67th percentile sentence lengths.
DEFAULT_CUTS = (12.0, 22.0)


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


def _paragraphs(text: str) -> list[list[str]]:
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if p:
            sents = _sentences(p)
            if sents:
                out.append(sents)
    return out


def _autocorr(xs: list[float], lag: int) -> float:
    """Pearson autocorrelation at ``lag``, on the series' own z-scale.

    Returns 0.0 for series too short or with no variance -- a flat series has no
    rhythm to correlate, and 0.0 is the honest "no signal" value rather than a
    division by zero.
    """
    n = len(xs)
    if n <= lag + 1:
        return 0.0
    mean = statistics.mean(xs)
    var = sum((x - mean) ** 2 for x in xs)
    if var <= 1e-9:
        return 0.0
    cov = sum((xs[i] - mean) * (xs[i + lag] - mean) for i in range(n - lag))
    return cov / var


def _classes(lens: list[int], cuts: tuple[float, float]) -> list[int]:
    lo, hi = cuts
    return [0 if n <= lo else (2 if n > hi else 1) for n in lens]


def sentence_cuts(texts: list[str]) -> tuple[float, float]:
    """Author's own short/mid/long boundaries: 33rd and 67th percentile lengths."""
    lens: list[int] = []
    for t in texts:
        lens.extend(len(_WORD.findall(s)) for s in _sentences(t))
    lens = sorted(n for n in lens if n > 0)
    if len(lens) < 10:
        return DEFAULT_CUTS

    def pct(p: float) -> float:
        pos = p * (len(lens) - 1)
        i = int(pos)
        frac = pos - i
        j = min(i + 1, len(lens) - 1)
        return lens[i] + frac * (lens[j] - lens[i])

    return (pct(1 / 3), pct(2 / 3))


def extract_features(
    text: str, cuts: tuple[float, float] = DEFAULT_CUTS
) -> dict[str, float]:
    """Cadence vector for one text. Zeros for degenerate input."""
    sents = _sentences(text)
    lens = [len(_WORD.findall(s)) for s in sents]
    lens = [n for n in lens if n > 0]
    if len(lens) < 4:
        return {f: 0.0 for f in CADENCE_FEATURES}

    mean_len = statistics.mean(lens)
    fl = [float(n) for n in lens]
    deltas = [fl[i + 1] - fl[i] for i in range(len(fl) - 1)]

    # Direction changes: how often the length trend reverses. A writer who
    # alternates long/short scores high; a monotone ramp scores 0.
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
    nz = [s for s in signs if s != 0]
    dir_changes = (
        sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1]) / (len(nz) - 1)
        if len(nz) > 1
        else 0.0
    )

    cls = _classes(lens, cuts)

    # Runs of the same class.
    runs, cur = [], 1
    for i in range(1, len(cls)):
        if cls[i] == cls[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)

    # 3x3 transition matrix -> entropy in bits. Low entropy means a predictable,
    # mechanical rhythm; human prose sits mid-range.
    trans: dict[tuple[int, int], int] = {}
    for i in range(len(cls) - 1):
        k = (cls[i], cls[i + 1])
        trans[k] = trans.get(k, 0) + 1
    total = sum(trans.values()) or 1
    entropy = -sum((c / total) * math.log2(c / total) for c in trans.values() if c)

    # P(short | previous long): the punch line after the build.
    long_positions = [i for i in range(len(cls) - 1) if cls[i] == 2]
    short_after_long = (
        sum(1 for i in long_positions if cls[i + 1] == 0) / len(long_positions)
        if long_positions
        else 0.0
    )

    paras = _paragraphs(text)
    opens = [len(_WORD.findall(p[0])) for p in paras if p]
    closes = [len(_WORD.findall(p[-1])) for p in paras if p]

    # Clause rhythm: words between punctuation breaths.
    clause_lens: list[int] = []
    clauses_per: list[int] = []
    for s in sents:
        parts = [c for c in _CLAUSE_SPLIT.split(s) if _WORD.findall(c)]
        if not parts:
            continue
        clauses_per.append(len(parts))
        clause_lens.extend(len(_WORD.findall(c)) for c in parts)
    clause_mean = statistics.mean(clause_lens) if clause_lens else 0.0
    clause_sd = statistics.stdev(clause_lens) if len(clause_lens) > 1 else 0.0

    return {
        "sl_autocorr1": _autocorr(fl, 1),
        "sl_autocorr2": _autocorr(fl, 2),
        "sl_delta_abs": (statistics.mean(abs(d) for d in deltas) / mean_len) if deltas and mean_len else 0.0,
        "sl_delta_sd": (statistics.stdev(deltas) / mean_len) if len(deltas) > 1 and mean_len else 0.0,
        "sl_dir_changes": dir_changes,
        "sl_run_mean": statistics.mean(runs) if runs else 0.0,
        "sl_trans_entropy": entropy,
        "sl_short_after_long": short_after_long,
        "para_open_ratio": (statistics.mean(opens) / mean_len) if opens and mean_len else 0.0,
        "para_close_ratio": (statistics.mean(closes) / mean_len) if closes and mean_len else 0.0,
        "clause_len_mean": clause_mean,
        "clause_len_cv": (clause_sd / clause_mean) if clause_mean > 1e-9 else 0.0,
        "clause_per_sent": statistics.mean(clauses_per) if clauses_per else 0.0,
    }
