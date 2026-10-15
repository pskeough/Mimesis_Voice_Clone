"""Per-voice calibration for the structural measures.

Everything that decides anything here is measured from the author's own corpus.
The file lives beside `rhetoric_ext_calibration.json` at
`profiles/<voice>/data/structure_calibration.json` and is rebuilt by
`mimesis calibrate`.

Length is the standing confound in every one of these measures, and a fixed
threshold does not survive it. A 250-word draft has fifteen sentences; a
1,200-word calibration unit has seventy. The opener distribution of fifteen
sentences sits further from any reference than the distribution of seventy does,
for reasons that have nothing to do with who wrote it, and the same holds for a
coefficient of variation over four paragraphs against one over fifteen.

So the calibration stores the author's own *series* -- the opener class of every
sentence, the word length of every paragraph, the word length of every sentence
-- and the threshold is computed at scoring time against the author's own units
truncated to the draft's own sentence and paragraph count. The comparison is
then matched rather than corrected. The cost is a hundred kilobytes of JSON per
voice.

Two thresholds are one-sided and two are bands, following the study's verdicts
in `docs/STRUCTURE_GATES.md`:

  echo rate     one-sided, max(mu + 3 sd, 0.08). Above the author only. An
                author who never restates is not a defect. Calibrated over the
                author's own 600-word windows, the same object the draft is
                scored on, so a long draft's extra windows are extra draws from
                a distribution measured the same way.
  opener distance  one-sided, mu + 2 sd of the author's own leave-one-out
                distances. Distance has no informative low end.
  paragraph CV  two-sided band. Under-varied is the machine failure; the study
                also found the reader preferring an over-varied cell, so the
                high side reports rather than accuses.
  cadence A_n   two-sided band, same reasoning.
  hapax         two-sided band. Vocabulary above the author is as unlike them
                as vocabulary below.

Every reference statistic is leave-one-out where the unit contributes to it.
Without that, a unit is compared to a reference it helped build and the
self-distances come out too small, which is the mistake that makes any
corpus-fitted gate look tighter than it is.
"""
from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from . import construction, cpd, lexdiv, otd, per, psd
from .textbase import prose_paragraphs, sentences

__all__ = ["FILENAME", "StructureCalibration", "calibrate", "load_for"]

FILENAME = "structure_calibration.json"

# Minimum calibration units before a threshold means anything. Below this the
# standard deviations are noise and the gate inherits the noise, the same rule
# `calibrate` prints THIN CALIBRATION for elsewhere in this package.
MIN_UNITS = 12

# Floors from the study: the echo rate's absolute ceiling regardless of how
# quiet an author's own corpus is, and the sd floor for the four constructions.
PER_FLOOR = 0.08
PER_SIGMA = 3.0
BAND_SIGMA = 2.0
OTD_SIGMA = 2.0
# Mirrors report.Z_FLAG, duplicated rather than imported because report imports
# this module. tests/test_measures.py asserts the two stay equal.
CONSTRUCTION_Z = 3.0

# One-letter codes so the stored opener series is one short string per unit.
_CODE = {"QUOTE": "Q", "CONJ": "C", "DISC": "D", "SUBORD": "S", "PP": "P",
         "ADV": "A", "PART": "T", "PRON": "R", "NP": "N", "OTHER": "O"}
assert set(_CODE) == set(otd.CLASSES) and len(set(_CODE.values())) == len(_CODE)
_DECODE = {v: k for k, v in _CODE.items()}


def _mu_sd(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def _shares(classes: list[str]) -> list[float]:
    n = len(classes)
    if not n:
        return [0.0] * len(otd.CLASSES)
    counts = {c: 0 for c in otd.CLASSES}
    for c in classes:
        counts[c] += 1
    return [counts[c] / n for c in otd.CLASSES]


@dataclass
class StructureCalibration:
    """The author's own distribution for each structural measure.

    `openers`, `paras` and `sents` are one entry per calibration unit and are
    what make the matched thresholds possible; everything else is a summary of
    them or of a length-stable measure.
    """

    n_units: int = 0
    # per unit
    openers: list[str] = field(default_factory=list)      # coded class sequence
    paras: list[list[int]] = field(default_factory=list)   # paragraph word lengths
    sents: list[list[int]] = field(default_factory=list)   # sentence word lengths
    per_values: list[float] = field(default_factory=list)  # echo rate per 600-word window
    hapax_values: list[float] = field(default_factory=list)
    # four new constructions: feature -> {"mu", "sd", "sd_raw", "fired"}
    constructions: dict = field(default_factory=dict)
    # share of the author's own units each line would flag, measured
    # leave-one-out at build time. The DETECTORS.md audit, for these measures.
    self_flag: dict = field(default_factory=dict)
    otd_available: bool = True

    # --- reference distributions ------------------------------------------

    def opener_reference(self, exclude: int | None = None) -> dict[str, float]:
        """Pooled opener-class distribution, optionally leaving one unit out."""
        counts = {c: 0 for c in otd.CLASSES}
        for i, seq in enumerate(self.openers):
            if i == exclude:
                continue
            for ch in seq:
                counts[_DECODE[ch]] += 1
        n = sum(counts.values())
        return {f"share_{c}": (counts[c] / n if n else 0.0) for c in otd.CLASSES}

    # --- matched thresholds -----------------------------------------------

    def _longest_matchable(self, series: list, n: int, floor: int) -> int | None:
        """The largest length at or below `n` that MIN_UNITS of the author's own
        units can match.

        Without this a draft longer than most of the corpus loses the line
        entirely: a 61-sentence draft against a corpus of 40-sentence pieces has
        no matched band and reports nothing. Comparing the draft's first 40
        sentences to the author's first 40 is still a matched comparison, of a
        prefix rather than the whole, and the report says which.
        """
        lengths = sorted((len(x) for x in series), reverse=True)
        if len(lengths) < MIN_UNITS:
            return None
        best = min(n, lengths[MIN_UNITS - 1])
        return best if best >= floor else None

    def otd_threshold(self, n_sent: int) -> tuple[float, int, int] | None:
        """(mu + 2 sd, units used, sentences compared) of the author's own
        leave-one-out opener distances, each computed on that unit's FIRST
        `n_sent` sentences.

        Truncating the author's units to the draft's length is what removes the
        length confound: a short draft is compared to short stretches of the
        author, not to whole pieces. When the corpus has too few units that long,
        both sides are cut to the longest length it can match.
        """
        if not self.openers:
            return None
        n = self._longest_matchable(self.openers, n_sent, floor=5)
        if n is None:
            return None
        d = []
        for i, seq in enumerate(self.openers):
            if len(seq) < n:
                continue
            ref = self.opener_reference(exclude=i)
            p = _shares([_DECODE[ch] for ch in seq[:n]])
            q = [ref[f"share_{c}"] for c in otd.CLASSES]
            d.append(otd.js_distance(p, q))
        if len(d) < MIN_UNITS:
            return None
        mu, sd = _mu_sd(d)
        return mu + OTD_SIGMA * sd, len(d), n

    def para_cv_band(self, n_para: int) -> tuple[float, float, float, int, int] | None:
        """(lo, hi, mu, units, paragraphs compared) for paragraph-length CV over
        the author's own units truncated to the draft's paragraph count."""
        if n_para < psd.MIN_PARAS:
            return None
        n_para = self._longest_matchable(self.paras, n_para, floor=psd.MIN_PARAS)
        if n_para is None:
            return None
        vals = []
        for pw in self.paras:
            if len(pw) < n_para:
                continue
            head = pw[:n_para]
            m = st.mean(head)
            if m > 0:
                vals.append(st.pstdev(head) / m)
        if len(vals) < MIN_UNITS:
            return None
        mu, sd = _mu_sd(vals)
        return mu - BAND_SIGMA * sd, mu + BAND_SIGMA * sd, mu, len(vals), n_para

    def burst_band(self, n_sent: int) -> tuple[float, float, float, int, int] | None:
        """(lo, hi, mu, units, sentences compared) for corrected burstiness. A_n
        is n-independent by construction, so the truncation here is belt and
        braces rather than a correction; it costs nothing and removes the
        question."""
        if n_sent < cpd.MIN_SENTS:
            return None
        n_sent = self._longest_matchable(self.sents, n_sent, floor=cpd.MIN_SENTS)
        if n_sent is None:
            return None
        vals = [cpd.burstiness_an(s[:n_sent]) for s in self.sents if len(s) >= n_sent]
        if len(vals) < MIN_UNITS:
            return None
        mu, sd = _mu_sd(vals)
        return mu - BAND_SIGMA * sd, mu + BAND_SIGMA * sd, mu, len(vals), n_sent

    def per_threshold(self) -> tuple[float, float] | None:
        """(threshold, author mean). max(mu + 3 sd, 0.08), from the study.

        Pooled over every window of every unit rather than one value per unit:
        a draft is scored on its worst window, so the reference distribution has
        to be the distribution of windows.
        """
        if len(self.per_values) < MIN_UNITS:
            return None
        mu, sd = _mu_sd(self.per_values)
        return max(mu + PER_SIGMA * sd, PER_FLOOR), mu

    def hapax_band(self) -> tuple[float, float, float] | None:
        if len(self.hapax_values) < MIN_UNITS:
            return None
        mu, sd = _mu_sd(self.hapax_values)
        return mu - BAND_SIGMA * sd, mu + BAND_SIGMA * sd, mu

    def construction_z(self, text: str) -> dict[str, float]:
        """Signed z of the four new constructions against the author."""
        r = construction.rates(text)
        out = {}
        for f in construction.FEATURES:
            c = self.constructions.get(f)
            if not c:
                continue
            out[f] = (r[f] - c["mu"]) / max(c["sd"], construction.SD_FLOOR)
        return out

    @property
    def thin(self) -> bool:
        return self.n_units < MIN_UNITS

    # --- io ----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_units": self.n_units,
            "openers": self.openers,
            "paras": self.paras,
            "sents": self.sents,
            "per_values": self.per_values,
            "hapax_values": self.hapax_values,
            "constructions": self.constructions,
            "self_flag": self.self_flag,
            "otd_available": self.otd_available,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructureCalibration":
        return cls(
            n_units=int(d.get("n_units", 0)),
            openers=[str(x) for x in d.get("openers") or []],
            paras=[[int(y) for y in x] for x in d.get("paras") or []],
            sents=[[int(y) for y in x] for x in d.get("sents") or []],
            per_values=[float(x) for x in d.get("per_values") or []],
            hapax_values=[float(x) for x in d.get("hapax_values") or []],
            constructions=dict(d.get("constructions") or {}),
            self_flag=dict(d.get("self_flag") or {}),
            otd_available=bool(d.get("otd_available", True)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StructureCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_for(prof) -> StructureCalibration | None:
    """The profile's structure calibration, or None when it has not been built.

    None rather than a default: there is no honest default for "how much this
    author varies their paragraphs", and a report line computed against a
    made-up mean is worse than a line that says it has no calibration.
    """
    try:
        path = Path(prof.scrub_path).parent / FILENAME
        return StructureCalibration.load(path) if path.exists() else None
    except Exception:
        return None


def calibrate(texts: list[str], min_words: int = 200) -> StructureCalibration:
    """Build the calibration from the author's own pieces.

    `texts` should be the same units the fingerprint is calibrated on -- the
    segmented ones when `mimesis calibrate` segments -- so a threshold and a
    fingerprint are measured over the same objects.
    """
    usable = [t for t in texts if len(t.split()) >= min_words]
    cal = StructureCalibration(n_units=len(usable), otd_available=otd.available())
    con_rows = []
    for t in usable:
        cal.paras.append([len(p.split()) for p in prose_paragraphs(t) if p.strip()])
        cal.sents.append([len(s.split()) for s in sentences(t) if s.strip()])
        cal.per_values.extend(per.per(w)["per"] for w in per.windows(t))
        lx = lexdiv.measure(t)
        if lx:
            cal.hapax_values.append(lx["hapax_w300"])
        if cal.otd_available:
            cal.openers.append("".join(_CODE[otd.opener_class(s)]
                                       for s in sentences(t) if s.strip()))
        con_rows.append(construction.rates(t))

    for f in construction.FEATURES:
        v = [r[f] for r in con_rows]
        mu, sd = _mu_sd(v)
        cal.constructions[f] = {
            "mu": mu,
            "sd": max(sd, construction.SD_FLOOR),
            "sd_raw": sd,
            "fired": (sum(1 for x in v if x > 0) / len(v)) if v else 0.0,
        }

    cal.self_flag = _self_flag_rates(cal, usable)
    # The construction line's own false-positive rate, scored the way the report
    # scores a draft: any of the four at z >= 3 in the machine direction.
    if con_rows:
        hits = 0
        for r in con_rows:
            for f in construction.FEATURES:
                c = cal.constructions[f]
                z = (r[f] - c["mu"]) / max(c["sd"], construction.SD_FLOOR)
                if z * construction.MACHINE_DIRECTION[f] >= CONSTRUCTION_Z:
                    hits += 1
                    break
        cal.self_flag["constructions"] = hits / len(con_rows)
    return cal


def _self_flag_rates(cal: StructureCalibration, unit_texts: list[str]) -> dict:
    """Share of the author's own units each line would flag.

    This is the DETECTORS.md false-positive audit applied to a band rather than
    to a construction. A band at two standard deviations flags about 5% of a
    normal distribution by arithmetic; what matters is what it flags on THIS
    corpus, where these features are not normal. A line that flags a fifth of
    the author's own writing is measuring the author.
    """
    out: dict[str, float] = {}
    thr = cal.per_threshold()
    if thr and cal.per_values:
        # Scored the way a draft is scored: worst window per unit, not per
        # window. A draft with four windows gets four draws at the threshold and
        # the audit has to pay the same multiplicity.
        worst = [max(per.per(w)["per"] for w in per.windows(t)) for t in unit_texts]
        out["echo"] = (sum(1 for x in worst if x > thr[0]) / len(worst)) if worst else 0.0
    band = cal.hapax_band()
    if band and cal.hapax_values:
        lo, hi, _ = band
        out["vocabulary"] = sum(1 for x in cal.hapax_values
                                if x < lo or x > hi) / len(cal.hapax_values)
    # Openers, paragraphs and cadence are scored at the median unit's own
    # length, which is the length the matched threshold will most often see.
    if cal.openers:
        n = int(st.median([len(s) for s in cal.openers]))
        t = cal.otd_threshold(n)
        if t:
            n = t[2]
            hits = tot = 0
            for i, seq in enumerate(cal.openers):
                if len(seq) < n:
                    continue
                ref = cal.opener_reference(exclude=i)
                p = _shares([_DECODE[ch] for ch in seq[:n]])
                q = [ref[f"share_{c}"] for c in otd.CLASSES]
                tot += 1
                hits += otd.js_distance(p, q) > t[0]
            if tot:
                out["openers"] = hits / tot
    if cal.paras:
        n = max(psd.MIN_PARAS, int(st.median([len(p) for p in cal.paras])))
        b = cal.para_cv_band(n)
        if b:
            lo, hi, _, _, n = b
            vals = [st.pstdev(p[:n]) / st.mean(p[:n])
                    for p in cal.paras if len(p) >= n and st.mean(p[:n]) > 0]
            if vals:
                out["paragraphs"] = sum(1 for x in vals if x < lo or x > hi) / len(vals)
    if cal.sents:
        n = max(cpd.MIN_SENTS, int(st.median([len(s) for s in cal.sents])))
        b = cal.burst_band(n)
        if b:
            lo, hi, _, _, n = b
            vals = [cpd.burstiness_an(s[:n]) for s in cal.sents if len(s) >= n]
            if vals:
                out["cadence"] = sum(1 for x in vals if x < lo or x > hi) / len(vals)
    return out


# Segment size for structural calibration. Long enough that a unit has a
# paragraph architecture to measure, short enough that a novel does not become
# one sample. Matches ingest.CALIBRATION_SEGMENT_WORDS.
SEGMENT_WORDS = 1500


def segment(texts: list[str], target: int = SEGMENT_WORDS) -> list[str]:
    """Cut over-long documents into ~`target`-word units at paragraph breaks.

    `ingest.segment_for_calibration` splits on blank lines, which does not cut a
    corpus that separates paragraphs with single newlines; this uses the same
    splitter the measures use, so a unit's paragraph boundaries are the ones the
    band is computed over. Trailing scraps under a third of the target are
    dropped rather than admitted as short, noisy samples.
    """
    out: list[str] = []
    for t in texts:
        if len(t.split()) <= target:
            out.append(t)
            continue
        buf: list[str] = []
        n = 0
        for para in prose_paragraphs(t):
            buf.append(para)
            n += len(para.split())
            if n >= target:
                out.append("\n\n".join(buf))
                buf, n = [], 0
        if buf and n >= target // 3:
            out.append("\n\n".join(buf))
    return out
