"""Judge-free stylometric fingerprint.

13 surface features per text, z-scored against a per-author corpus calibration,
collapsed to a single RMS-z distance. Selection and gating on this distance is
deliberately taken away from LLM taste: judge preference and stylometric
fidelity diverge (see docs/DESIGN.md thesis), so the fingerprint is the
authority on "sounds like the author" and the LLM judge is demoted to a
secondary quality signal.

Pure stdlib + math. No model downloads, no tokenizers: features are cheap
surface statistics so calibration runs on any corpus in milliseconds and the
gate can score every slate candidate on every compose call.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

FEATURES = (
    "sent_len_mean",
    "sent_len_stdev",       # burstiness
    "pct_short_sents",      # < 8 words
    "pct_long_sents",       # > 30 words
    "colons_per_100w",
    "semicolons_per_100w",
    "emdash_per_100w",
    "question_rate",        # questions per sentence
    "commas_per_sent",
    "para_len_mean",        # sentences per paragraph
    "first_person_openers", # sentences opening with I/I'm/I've/I'd/I'll/My
    "ttr",                  # type-token ratio, capped window
    "word_len_mean",
)

from . import cadence as cadence_mod

# The 13 surface features above are every one order-invariant: shuffle a text's
# sentences and all of them are unchanged. Sorting a real piece shortest-to-
# longest -- a monotone ramp no writer produces -- moved mean RMS-z by +0.0023
# and flagged 0/93 creative pieces off-voice (evals/cadence_blindness.py). So a
# v1 fingerprint cannot see cadence, and a gate selecting on it was optimising
# vocabulary and density while leaving pacing free. FEATURE_SETS["v2"] appends
# the 13 order-aware features from cadence.py. v1 is kept intact so every
# existing profile keeps scoring on the ruler it was calibrated with, and so the
# two can be A/B'd honestly.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "v1": FEATURES,
    "v2": FEATURES + cadence_mod.CADENCE_FEATURES,
}
DEFAULT_FEATURE_SET = "v1"

# Per-feature z-scores are winsorized here before the RMS.
#
# A feature the author almost never uses has a near-zero std, so an ordinary
# draft scores an enormous z on it and that one feature dominates the aggregate.
# Measured on the creative corpus, calibrating on 34 pieces gave
# emdash_per_100w mean 0.0022 / std 0.0117, and the author's OWN held-out prose
# averaged |z| = 12.05 on it -- contributing ~145 to a sum of squares where every
# other feature contributed 1-2. The RMS over 26 features had become a
# single-feature em-dash detector.
#
# 4.0 keeps the metric two-sided and ordered (4 SD out is still "definitely
# off") while bounding any one feature's share of the total. This is a fix to
# the v1 metric as much as to v2; both improve.
Z_CLIP = 4.0

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z']+")
_FIRST_PERSON = re.compile(r"^[\"'“‘(]?(I|I'm|I've|I'd|I'll|My)\b")
_TTR_WINDOW = 400  # cap so TTR is comparable across text lengths
# Dash RUNS, matching scrub._EMDASH_COUNT_RE. Counting hyphen pairs made
# "aflame--- heat" score 1 while the scalpel also treated it as 1 but left a
# stray hyphen; both now agree that a run of any length is one dash.
# A run that occupies a whole line is a horizontal RULE, not punctuation.
# Counting those made a folder README score 27 dashes per 100 words and drag
# an entire corpus mean from ~0 to 1.24, because ASCII underlines like
# "------" matched. Prose dashes always have text on at least one side.
_RULE_LINE = re.compile(r"(?m)^[ 	]*[-—–=_*]{3,}[ 	]*$")
_DASH_RUN = re.compile(r"—|–|-{2,}")


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def extract_features(text: str) -> dict[str, float]:
    """Feature vector for one text. Returns zeros for degenerate input."""
    sents = _sentences(text)
    words = _WORD.findall(text)
    if not sents or len(words) < 20:
        return {f: 0.0 for f in FEATURES}

    sent_lens = [len(_WORD.findall(s)) for s in sents]
    n_words = len(words)
    n_sents = len(sents)
    paras = _paragraphs(text)
    para_lens = [len(_sentences(p)) for p in paras] or [n_sents]
    window = words[:_TTR_WINDOW]

    per100 = 100.0 / n_words
    return {
        "sent_len_mean": statistics.mean(sent_lens),
        "sent_len_stdev": statistics.stdev(sent_lens) if n_sents > 1 else 0.0,
        "pct_short_sents": sum(1 for n in sent_lens if n < 8) / n_sents,
        "pct_long_sents": sum(1 for n in sent_lens if n > 30) / n_sents,
        "colons_per_100w": text.count(":") * per100,
        "semicolons_per_100w": text.count(";") * per100,
        "emdash_per_100w": len(_DASH_RUN.findall(_RULE_LINE.sub("", text))) * per100,
        "question_rate": sum(1 for s in sents if s.rstrip().endswith("?")) / n_sents,
        "commas_per_sent": text.count(",") / n_sents,
        "para_len_mean": statistics.mean(para_lens),
        "first_person_openers": sum(1 for s in sents if _FIRST_PERSON.match(s)) / n_sents,
        "ttr": len({w.lower() for w in window}) / len(window),
        "word_len_mean": statistics.mean(len(w) for w in words),
    }


def extract_all(
    text: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    cuts: tuple[float, float] = cadence_mod.DEFAULT_CUTS,
) -> dict[str, float]:
    """Feature vector for the named set. v1 is surface-only; v2 adds cadence."""
    feats = extract_features(text)
    if feature_set == "v2":
        feats.update(cadence_mod.extract_features(text, cuts))
    return feats


@dataclass
class Fingerprint:
    """Per-author calibration: feature means/stds plus the corpus self-baseline.

    self_baseline is the mean RMS-z of held-out corpus pieces against the
    calibration built from the rest: real writing by the author does not score
    0.0, and gate thresholds and eval reports must be read relative to this
    number, not to zero.

    fit_threshold is the 95th percentile of those same leave-one-out distances:
    the point above which a text is unlike the corpus by more than 5% of the
    author's own pieces are. Use it, not a fixed multiple of self_baseline, to
    decide "off-voice" -- corpus spread varies by voice. On the creative corpus a
    1.6x-baseline rule sat exactly on the pieces' p90 and flagged 11% of real
    writing; a gate that fires on the author's own work is one they learn to
    ignore. 0.0 means an older fingerprint that predates this field; callers
    should fall back to a ratio rule and say so.
    """

    means: dict[str, float]
    stds: dict[str, float]
    self_baseline: float = 0.0
    fit_threshold: float = 0.0
    n_pieces: int = 0
    meta: dict = field(default_factory=dict)
    feature_set: str = DEFAULT_FEATURE_SET
    cuts: tuple[float, float] = cadence_mod.DEFAULT_CUTS

    @property
    def features(self) -> tuple[str, ...]:
        return FEATURE_SETS.get(self.feature_set, FEATURES)

    def distance(self, text: str) -> float:
        return self.distance_detail(text)[0]

    def distance_detail(self, text: str) -> tuple[float, dict[str, float]]:
        """RMS-z distance plus per-feature z-scores (drives targeted rewrites)."""
        feats = extract_all(text, self.feature_set, self.cuts)
        names = self.features
        zs: dict[str, float] = {}
        for f in names:
            std = self.stds.get(f, 0.0)
            z = (feats[f] - self.means.get(f, 0.0)) / std if std > 1e-9 else 0.0
            zs[f] = max(-Z_CLIP, min(Z_CLIP, z))
        rms = math.sqrt(sum(z * z for z in zs.values()) / len(names))
        return rms, zs

    def cadence_distance(self, text: str) -> float:
        """RMS-z over the cadence block only. 0.0 on a v1 fingerprint.

        Reported separately because an aggregate over 26 features dilutes a
        rhythm failure: a draft can be badly off on pacing and still pass the
        combined threshold on the strength of 13 matching surface features.
        """
        if self.feature_set != "v2":
            return 0.0
        _, zs = self.distance_detail(text)
        names = cadence_mod.CADENCE_FEATURES
        return math.sqrt(sum(zs[f] ** 2 for f in names) / len(names))

    def worst_features(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k deviating features with signed z, for rewrite instructions."""
        _, zs = self.distance_detail(text)
        return sorted(zs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "means": self.means,
                    "stds": self.stds,
                    "self_baseline": self.self_baseline,
                    "fit_threshold": self.fit_threshold,
                    "n_pieces": self.n_pieces,
                    "meta": self.meta,
                    "feature_set": self.feature_set,
                    "cuts": list(self.cuts),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Fingerprint":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        # Fingerprints written before cadence existed carry no feature_set; they
        # are v1 by construction and must keep scoring as v1, or every stored
        # threshold silently changes meaning.
        cuts = d.get("cuts") or list(cadence_mod.DEFAULT_CUTS)
        return cls(
            means=d["means"],
            stds=d["stds"],
            self_baseline=d.get("self_baseline", 0.0),
            fit_threshold=d.get("fit_threshold", 0.0),
            n_pieces=d.get("n_pieces", 0),
            meta=d.get("meta", {}),
            feature_set=d.get("feature_set", DEFAULT_FEATURE_SET),
            cuts=(float(cuts[0]), float(cuts[1])),
        )


def _p95(values: list[float]) -> float:
    """95th percentile, linear interpolation. Small-n safe."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = 0.95 * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + frac * (s[hi] - s[lo])


def calibrate(
    texts: list[str], min_words: int = 120, feature_set: str = DEFAULT_FEATURE_SET
) -> Fingerprint:
    """Build a fingerprint from corpus pieces.

    Pieces under min_words are dropped: short fragments have unstable feature
    values and widen every std, which silently loosens the gate. The
    self-baseline is computed leave-one-out so it is honest for small corpora.

    ``feature_set="v2"`` adds the 13 order-aware cadence features and derives the
    author's own short/mid/long sentence cuts from their corpus terciles.
    """
    usable = [t for t in texts if len(_WORD.findall(t)) >= min_words]
    if len(usable) < 5:
        raise ValueError(
            f"need >=5 pieces of >={min_words} words to calibrate, got {len(usable)}"
        )

    names = FEATURE_SETS.get(feature_set, FEATURES)
    cuts = cadence_mod.sentence_cuts(usable) if feature_set == "v2" else cadence_mod.DEFAULT_CUTS
    rows = [extract_all(t, feature_set, cuts) for t in usable]
    means = {f: statistics.mean(r[f] for r in rows) for f in names}
    stds = {
        f: statistics.stdev([r[f] for r in rows]) if len(rows) > 1 else 0.0
        for f in names
    }
    fp = Fingerprint(
        means=means, stds=stds, n_pieces=len(usable), feature_set=feature_set, cuts=cuts
    )

    # Leave-one-out self-baseline.
    dists: list[float] = []
    for i in range(len(usable)):
        rest = rows[:i] + rows[i + 1 :]
        loo_means = {f: statistics.mean(r[f] for r in rest) for f in names}
        loo_stds = {
            f: statistics.stdev([r[f] for r in rest]) if len(rest) > 1 else 0.0
            for f in names
        }
        loo = Fingerprint(
            means=loo_means, stds=loo_stds, feature_set=feature_set, cuts=cuts
        )
        dists.append(loo.distance(usable[i]))
    fp.self_baseline = statistics.mean(dists)
    fp.fit_threshold = _p95(dists)
    fp.meta = dict(fp.meta or {})
    fp.meta["corpus_spread"] = corpus_spread(rows, names)
    return fp


def corpus_spread(rows: list[dict], names: tuple[str, ...]) -> float:
    """Mean pairwise z-space distance between the author's own pieces.

    This is the yardstick a slate has to reach. A set of candidates that clusters
    far tighter than the author's own writing is not a set of options, it is one
    output sampled four times, and the gate then picks among near-duplicates.
    Measured on this project: six generations spread 0.657 against a corpus
    spread of 1.065 for comparable pieces.

    Capped at 60 pieces because this is O(n^2) and runs inside calibrate; the
    estimate is stable well before that.
    """
    import random as _r
    sample = rows if len(rows) <= 60 else _r.Random(0).sample(rows, 60)
    mus = {f: statistics.mean(r[f] for r in sample) for f in names}
    sds = {f: (statistics.stdev([r[f] for r in sample]) if len(sample) > 1 else 0.0)
           for f in names}
    vecs = [[(r[f] - mus[f]) / sds[f] if sds[f] > 1e-9 else 0.0 for f in names]
            for r in sample]
    tot = pairs = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            tot += (sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j])) / len(names)) ** 0.5
            pairs += 1
    return tot / pairs if pairs else 0.0


# --- weighted recalibration (Upgrade 3: recalibrate_from_accepted) ------------


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    tot = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / tot if tot else 0.0


def _weighted_std(values: list[float], weights: list[float], mean: float) -> float:
    tot = sum(weights)
    if tot <= 0:
        return 0.0
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / tot
    return math.sqrt(var)


def recency_weights(n: int, base_weight: float = 4.0, half_life: float = 3.0) -> list[float]:
    """Weights for n accepted samples, oldest first, newest weighted most.

    The newest sample gets ``base_weight``; each step back in time halves the
    excess-over-1 every ``half_life`` samples, decaying toward the corpus's own
    per-piece weight of 1.0. This is the "recency-weighted toward what I actually
    keep" bias: the loop trusts the last few things the author kept the most.
    """
    if n <= 0:
        return []
    out = []
    for i in range(n):  # i=0 oldest, i=n-1 newest
        steps_back = (n - 1) - i
        decay = 0.5 ** (steps_back / half_life) if half_life > 0 else (1.0 if steps_back == 0 else 0.0)
        out.append(1.0 + (base_weight - 1.0) * decay)
    return out


def calibrate_weighted(
    texts: list[str],
    weights: list[float],
    min_words: int = 120,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> Fingerprint:
    """Weighted fingerprint: each text carries a weight into the mean/std.

    Corpus pieces carry weight 1.0; accepted pieces carry a recency-scaled boost,
    so the calibrated means shift toward the accepted set and the gate steers new
    generations there. The self-baseline is the unweighted leave-one-out mean over
    the same usable set, keeping the reported ruler honest.
    """
    pairs = [(t, w) for t, w in zip(texts, weights) if len(_WORD.findall(t)) >= min_words]
    if len(pairs) < 5:
        raise ValueError(
            f"need >=5 pieces of >={min_words} words to recalibrate, got {len(pairs)}"
        )
    usable = [t for t, _ in pairs]
    ws = [w for _, w in pairs]
    names = FEATURE_SETS.get(feature_set, FEATURES)
    cuts = cadence_mod.sentence_cuts(usable) if feature_set == "v2" else cadence_mod.DEFAULT_CUTS
    rows = [extract_all(t, feature_set, cuts) for t in usable]
    means = {f: _weighted_mean([r[f] for r in rows], ws) for f in names}
    stds = {f: _weighted_std([r[f] for r in rows], ws, means[f]) for f in names}
    fp = Fingerprint(
        means=means, stds=stds, n_pieces=len(usable), feature_set=feature_set, cuts=cuts
    )

    dists: list[float] = []
    for i in range(len(usable)):
        rest = rows[:i] + rows[i + 1 :]
        loo_means = {f: statistics.mean(r[f] for r in rest) for f in names}
        loo_stds = {
            f: statistics.stdev([r[f] for r in rest]) if len(rest) > 1 else 0.0
            for f in names
        }
        dists.append(
            Fingerprint(
                means=loo_means, stds=loo_stds, feature_set=feature_set, cuts=cuts
            ).distance(usable[i])
        )
    fp.meta = dict(fp.meta or {})
    fp.meta["corpus_spread"] = corpus_spread(rows, names)
    fp.self_baseline = statistics.mean(dists)
    fp.fit_threshold = _p95(dists)
    return fp


@dataclass
class AcceptedTarget:
    """A fixed-ruler distance to the accepted set, for the convergence test.

    Feature means are the *accepted set's* centroid; the z-scale is the base
    reference fingerprint's stds (stable, many pieces). ``distance(text)`` is the
    RMS-z of a text from the accepted centroid on that fixed ruler, so it is
    comparable before vs after recalibration: if new outputs move toward what the
    author kept, this number drops.
    """

    means: dict[str, float]
    scale_stds: dict[str, float]
    n: int = 0
    feature_set: str = DEFAULT_FEATURE_SET
    cuts: tuple[float, float] = cadence_mod.DEFAULT_CUTS

    def distance(self, text: str) -> float:
        names = FEATURE_SETS.get(self.feature_set, FEATURES)
        feats = extract_all(text, self.feature_set, self.cuts)
        acc = 0.0
        for f in names:
            std = self.scale_stds.get(f, 0.0)
            if std > 1e-9:
                z = (feats[f] - self.means.get(f, 0.0)) / std
                z = max(-Z_CLIP, min(Z_CLIP, z))
                acc += z * z
        return math.sqrt(acc / len(names))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "means": self.means,
                    "scale_stds": self.scale_stds,
                    "n": self.n,
                    "feature_set": self.feature_set,
                    "cuts": list(self.cuts),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AcceptedTarget":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cuts = d.get("cuts") or list(cadence_mod.DEFAULT_CUTS)
        return cls(
            means=d["means"],
            scale_stds=d["scale_stds"],
            n=d.get("n", 0),
            feature_set=d.get("feature_set", DEFAULT_FEATURE_SET),
            cuts=(float(cuts[0]), float(cuts[1])),
        )


def accepted_target(accepted_texts: list[str], scale_from: Fingerprint) -> AcceptedTarget:
    """Build the accepted-set centroid, z-scaled by a stable base fingerprint."""
    names = scale_from.features
    rows = [
        extract_all(t, scale_from.feature_set, scale_from.cuts)
        for t in accepted_texts
        if t.strip()
    ]
    if not rows:
        raise ValueError("no accepted texts to build a target from")
    means = {f: statistics.mean(r[f] for r in rows) for f in names}
    return AcceptedTarget(
        means=means,
        scale_stds=dict(scale_from.stds),
        n=len(rows),
        feature_set=scale_from.feature_set,
        cuts=scale_from.cuts,
    )
