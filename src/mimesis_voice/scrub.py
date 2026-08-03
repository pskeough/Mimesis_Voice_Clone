"""AI-footprint scrubber: hard style rules, fidelity audit, optional detector.

Three tiers, merged from v1's calibrated scrubber and RVCR's fidelity audit:

1. **Hard rules** — a prose-dash ceiling read from the author's own corpus, a
   corpus-calibrated AI banlist minus a
   per-author whitelist, a burstiness floor, and a hedge ceiling. The banlist is
   the static AI-tell wordlist filtered against words the author actually uses,
   so it never fights the author's own vocabulary (ported v1 calibration).
2. **Fidelity audit** — numbers, citations, and acronyms extracted from a source
   and compared to the output; drops, alterations, and inventions are flagged.
   Critical for research voice, where a smoother sentence must never cost a
   statistic (ported RVCR concept, reimplemented).
3. **Detector (optional, extra ``[detect]``)** — a perplexity-ratio signal for
   *reporting only*, never a gate. The base install reports it as not installed.

Scalpel mode is deliberate: the scrubber strips only what is always safe and
*flags* the rest. It never rewrites prose to force the mean, because
mean-collapse kills the voice it is trying to protect. "Always safe" is itself
per-author: the em-dash strip is a genuine de-AI edit for a writer who does not
use them and pure vandalism for one who does, so it runs only below the
calibrated AUTHOR_DASH_BAND_MIN and drafts above it are flagged, not rewritten.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .fingerprint import Fingerprint
from . import presence as presence_mod
from . import quality as quality_mod
from . import rhetoric as rhetoric_mod
from . import textnorm

# A draft is called off-voice when its fingerprint RMS-z exceeds the corpus
# self-baseline by this factor. Real pieces by the author do not score 0.0, so
# the baseline (not zero) is the ruler -- see Fingerprint.self_baseline.
FIT_RATIO = 1.6

# Per-feature escape hatch for drafts whose RMS is diluted: this many features,
# each at least this many SDs out, is advisory regardless of the aggregate.
#
# Both numbers are measured, not chosen. Swept against real corpus pieces:
#   z=1.9 k=1 -> fires on 34/42/55% of real creative/personal/research pieces
#   z=2.5 k=1 -> 24/23/27%
#   z=3.0 k=1 -> 18/19/18%
#   z=3.0 k=2 -> 5/4/0%   and still catches a deliberately flat draft everywhere
# One feature at 1.9 SD is just multiple comparisons over 13 features.
SINGLE_FEATURE_Z = 3.0
SINGLE_FEATURE_K = 2

# --- lexical tables (ported from v1 calibrate.py) -----------------------------

AI_TELL_WORDS: set[str] = {
    "delve", "delves", "delving", "tapestry", "tapestries", "navigate",
    "navigating", "navigated", "realm", "realms", "testament", "intricate",
    "intricately", "multifaceted", "holistic", "leverage", "leveraging",
    "leveraged", "unlock", "unlocking", "paradigm", "paradigms", "embark",
    "embarking", "foster", "fostering", "cultivate", "cultivating", "robust",
    "myriad", "plethora", "underscore", "underscores", "underscoring",
    "showcase", "showcases", "showcasing", "elevate", "elevating", "streamline",
    "streamlining", "comprehensive", "seamless", "seamlessly", "pivotal",
    "burgeoning", "empower", "empowering", "empowered", "transformative",
    "crucial", "vital", "essential", "catalyst", "synergy", "optimize",
    "optimizing", "optimized", "unleash", "unleashing", "beacon", "resonate",
    "resonates", "resonating", "align", "aligning", "aligned", "impactful",
    "demystify", "reimagine", "furthermore", "moreover", "additionally",
}

AI_TELL_PHRASES: list[str] = [
    "in the realm of", "navigate the complexities of", "stands as a testament",
    "it's important to note", "it is important to note", "in today's fast-paced",
    "in today's digital", "ever-evolving landscape", "rich tapestry",
    "delve into", "embark on a journey", "shed light on", "in conclusion",
    "in essence", "when it comes to", "that being said", "it goes without saying",
    "play a crucial role", "play a vital role", "wealth of knowledge",
    "treasure trove", "deep dive", "a sense of", "an air of", "a feeling of",
    "unlock the potential", "catalyst for change", "game-changer",
    "look no further", "moving forward", "delve deeper", "let's dive in",
]

# Softening only. "seems to" and "appears to" used to live in this list and no longer do: they are
# epistemic stance, not mush, and penalizing them taught writers to strip the exact markers that
# make prose read as authored. A corpus that avoids softening produced a hedge ceiling of 0.00,
# which then flagged every "seems to us" as overuse. They are counted as presence instead, in
# presence.STANCE_MARKERS. See presence.py for the case that prompted the split.
_HEDGE_PATTERNS = [
    r"\bas if\b", r"\balmost as if\b", r"\bas though\b", r"\bsort of\b",
    r"\bkind of\b", r"\bin some way\b", r"\bin a sense\b", r"\bof sorts\b",
    r"\bmight have been\b", r"\bcould have been\b", r"\bsomehow\b",
    r"\bsomething like\b", r"\bmore or less\b", r"\bto some extent\b",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")
_WORD_RE = re.compile(r"\b[\w']+\b")

# --- fidelity extractors (ported from RVCR mcp_bridge) ------------------------

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITE_RE = re.compile(r"[A-Z][a-zA-Z]+ et al\.?|\([A-Z][a-zA-Z][^)]*?\d{4}[^)]*?\)|arXiv:\d{4}\.\d+")
_ACRO_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b")
_ACRO_SKIP = {
    "AI", "LLM", "LLMS", "RAG", "LORA", "QLORA", "GPT", "API", "CPU", "GPU",
    "MCP", "JSON", "PDF", "FTS", "RRF", "STEM", "ID", "URL", "IID", "HTTP",
}


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            parts.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return parts


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM_RE.findall(text):
        v = m.replace(",", "")
        if "." in v:
            v = v.rstrip("0").rstrip(".")
        out.add(v)
    return out


def _citations(text: str) -> set[str]:
    cites = {c.strip() for c in _CITE_RE.findall(text)}
    acros = {a for a in _ACRO_RE.findall(text) if a not in _ACRO_SKIP}
    return cites | acros


# --- calibration --------------------------------------------------------------


@dataclass
class ScrubCalibration:
    """Per-author scrub thresholds and banlist produced by ``mimesis calibrate``."""

    banned_words: list[str]
    banned_phrases: list[str]
    whitelist: list[str]
    burstiness_floor: float
    hedge_ceiling: float
    mean_sentence_len: float
    n_pieces: int = 0
    # Author's own 95th-percentile rates for two constructions that presence-based
    # checks get wrong. He writes both; generated text AMPLIFIES them. Measured on
    # his 157 corpus pieces: cleft p95 1.90/1000w (median 0.00), antithesis p95
    # 3.84 (median 0.00-0.96). Six generated sections ran 2.8-3.4 and 5.8-19.8.
    # Stored here rather than in a new file so every existing analyze() caller
    # picks them up without a signature change.
    cleft_p95: float = 0.0
    antithesis_p95: float = 0.0
    # Prose dashes per 1000 words, 95th percentile over the author's own
    # pieces. Zero for an author who does not use them, which is the case a
    # blanket strip-to-zero rule silently assumed for everyone.
    dash_per1k_p95: float = 0.0
    cleft_p25: float = 0.0
    antithesis_p25: float = 0.0
    density_p25: float = 0.0
    specificity_p25: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "banned_words": self.banned_words,
                    "banned_phrases": self.banned_phrases,
                    "whitelist": self.whitelist,
                    "burstiness_floor": self.burstiness_floor,
                    "hedge_ceiling": self.hedge_ceiling,
                    "mean_sentence_len": self.mean_sentence_len,
                    "n_pieces": self.n_pieces,
                    "dash_per1k_p95": self.dash_per1k_p95,
                    "cleft_p95": self.cleft_p95,
                    "antithesis_p95": self.antithesis_p95,
                    "cleft_p25": self.cleft_p25,
                    "antithesis_p25": self.antithesis_p25,
                    "density_p25": self.density_p25,
                    "specificity_p25": self.specificity_p25,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScrubCalibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            banned_words=d["banned_words"],
            banned_phrases=d["banned_phrases"],
            whitelist=d.get("whitelist", []),
            burstiness_floor=d.get("burstiness_floor", 5.0),
            hedge_ceiling=d.get("hedge_ceiling", 1.5),
            mean_sentence_len=d.get("mean_sentence_len", 18.0),
            n_pieces=d.get("n_pieces", 0),
            dash_per1k_p95=d.get("dash_per1k_p95", 0.0),
            cleft_p95=d.get("cleft_p95", 0.0),
            antithesis_p95=d.get("antithesis_p95", 0.0),
            cleft_p25=d.get("cleft_p25", 0.0),
            antithesis_p25=d.get("antithesis_p25", 0.0),
            density_p25=d.get("density_p25", 0.0),
            specificity_p25=d.get("specificity_p25", 0.0),
        )


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[idx]


# Above this calibrated rate the author demonstrably uses prose dashes as
# punctuation, and stripping them is not de-AI-ing a draft, it is deleting a
# habit. Below it, strip-to-zero stands: the em-dash is a genuine model tell.
#
# The band is read off dash_per1k_p95, which is a TAIL statistic, so the cutoff
# has to sit well above zero: a novel with no em-dashes at all still scored 0.63
# on a handful of en-dashes and hyphen runs, and a 0.5 cutoff wrongly licensed
# the habit for it. Measured reference points, same author, three corpora:
# novel 0.63 (no habit), long-form articles 12.6, short social posts 42.3.
#
# NOTE: this is an engine constant describing authors in general, not one
# measured from any corpus. scripts/portability_audit.py lists it as such.
AUTHOR_DASH_BAND_MIN = 2.0  # dashes per 1000 words


def author_uses_dashes(cal: "ScrubCalibration") -> bool:
    return cal.dash_per1k_p95 >= AUTHOR_DASH_BAND_MIN


def calibrate(texts: list[str], whitelist: list[str] | None = None) -> ScrubCalibration:
    """Calibrate scrub thresholds from corpus pieces.

    A word the author uses >= 3 times is whitelisted out of the banlist, plus any
    explicit ``whitelist`` from config. Burstiness floor is the 5th percentile of
    per-piece sentence-length stdev; hedge ceiling is the 95th percentile of
    per-piece hedge density (per 200 words).
    """
    explicit = {w.lower() for w in (whitelist or [])}
    bursts: list[float] = []
    dashes_per1k: list[float] = []
    hedges_per200: list[float] = []
    mean_lens: list[float] = []
    vocab: dict[str, int] = {}
    n = 0
    for text in texts:
        sents = _split_sentences(text)
        sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
        sent_lens = [x for x in sent_lens if x > 0]
        if len(sent_lens) < 3:
            continue
        n += 1
        wc = sum(sent_lens)
        bursts.append(statistics.stdev(sent_lens))
        mean_lens.append(statistics.mean(sent_lens))
        hedges_per200.append(len(_HEDGE_RE.findall(text)) * 200 / wc if wc else 0.0)
        dashes_per1k.append(count_dashes(text) * 1000 / wc if wc else 0.0)
        for w in _WORD_RE.findall(text.lower()):
            vocab[w] = vocab.get(w, 0) + 1

    rhet = rhetoric_mod.calibrate(texts)
    qual = quality_mod.calibrate(texts)
    author_used = {w for w, c in vocab.items() if c >= 3} | explicit
    banned = sorted(w for w in AI_TELL_WORDS if w.lower() not in author_used)
    kept = sorted((AI_TELL_WORDS & author_used) | explicit)
    return ScrubCalibration(
        banned_words=banned,
        banned_phrases=list(AI_TELL_PHRASES),
        whitelist=kept,
        burstiness_floor=_percentile(bursts, 5) if bursts else 5.0,
        hedge_ceiling=_percentile(hedges_per200, 95) if hedges_per200 else 1.5,
        dash_per1k_p95=_percentile(dashes_per1k, 95) if dashes_per1k else 0.0,
        mean_sentence_len=_percentile(mean_lens, 50) if mean_lens else 18.0,
        n_pieces=n,
        cleft_p95=rhet.cleft_p95,
        antithesis_p95=rhet.antithesis_p95,
        cleft_p25=rhet.cleft_p25,
        antithesis_p25=rhet.antithesis_p25,
        density_p25=qual.density_p25,
        specificity_p25=qual.specificity_p25,
    )


# --- optional detector (extra [detect]) ---------------------------------------


def detector_signal(
    text: str, threshold: float | None = None, calibrated: bool = False,
    direction: str = "low_is_machine",
) -> dict:
    """Binoculars perplexity-ratio detector signal.

    Returns the real score dict when the ``[detect]`` extra is installed (see
    ``detect.score``); a ``{"available": False, ...}`` stub otherwise, so callers
    degrade cleanly on a base install. Pass a calibrated ``threshold`` to get a
    ``label`` of "machine"/"human"; without one the score is reported only.
    """
    try:
        from . import detect as _detect  # type: ignore
    except Exception:
        return {"available": False, "detail": "detector not installed (pip install 'mimesis-voice[detect]')"}
    try:  # pragma: no cover - only with optional extra present
        return _detect.score(text, threshold=threshold, calibrated=calibrated, direction=direction)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "detail": f"detector error: {exc}"}


# --- analysis + report --------------------------------------------------------


@dataclass
class ScrubReport:
    """Structured scrub result. ``hard_flags`` drive the gate; the rest is advisory."""

    emdash_count: int = 0
    dash_band_per1k: float = 0.0
    word_count: int = 0
    banned_words: list[str] = field(default_factory=list)
    banned_phrases: list[str] = field(default_factory=list)
    hedge_density: float = 0.0
    hedge_ceiling: float = 0.0
    burstiness: float = 0.0
    burstiness_floor: float = 0.0
    fidelity_dropped_numbers: list[str] = field(default_factory=list)
    fidelity_added_numbers: list[str] = field(default_factory=list)
    fidelity_dropped_citations: list[str] = field(default_factory=list)
    detector: dict = field(default_factory=dict)
    checked_fidelity: bool = False
    fp_distance: float = 0.0
    fp_baseline: float = 0.0
    fp_threshold: float = 0.0
    fp_threshold_calibrated: bool = False
    fp_worst: list[tuple[str, float]] = field(default_factory=list)
    fp_checked: bool = False
    presence: dict = field(default_factory=dict)
    rhetoric: "rhetoric_mod.RhetoricReport | None" = None

    @property
    def fit_off(self) -> bool:
        """Draft is measurably unlike the corpus, in either direction.

        Every other check here is one-sided: it hunts for things the author does
        NOT do (em-dashes, cliche vocabulary, hedging, flat pacing). A draft can
        pass all of them and still be nothing like the author by being too plain
        -- shorter words, thinner vocabulary, lower reading level. That is the
        observed failure mode of voiced drafts, not slop. The fingerprint's RMS-z
        is symmetric, so it catches both directions; this exposes it here.
        """
        if not self.fp_checked or self.fp_threshold <= 0:
            return False
        return self.fp_distance > self.fp_threshold

    @property
    def fit_drifting(self) -> bool:
        """Between the ratio rule and the corpus p95: advisory, not a hard flag.

        Two tiers exist because one threshold cannot serve every voice. A p95 cut
        keeps false alarms at 5%, but on a corpus that mixes poems, flash and
        novels the spread is so wide (creative p95 = 2.24 vs baseline 0.93) that a
        flat off-voice draft scoring 1.68 slips under it. The ratio rule catches
        that draft but flags 11% of real pieces.

        So: p95 is the hard flag, and the band beneath it is reported as advice.
        The real fix is per-form fingerprints -- a poem and a novel chapter should
        not share a calibration -- at which point this band can go away.
        """
        if not self.fp_checked or self.fp_baseline <= 0 or self.fit_off:
            return False
        if self.fp_distance > self.fp_baseline * FIT_RATIO:
            return True
        # RMS over 13 features dilutes a strong signal on a few of them, so check
        # the per-feature view too. Deliberately conservative: a voiced draft that
        # sat at ttr z=-1.92 with everything else quiet is NOT caught here, and
        # should not be -- a third of the author's own pieces have some feature
        # that far out. Only a genuinely extreme, multi-feature departure counts.
        return sum(
            1 for _, z in self.fp_worst if abs(z) >= SINGLE_FEATURE_Z
        ) >= SINGLE_FEATURE_K

    @property
    def hard_flags(self) -> list[str]:
        flags = []
        if self.emdash_count:
            flags.append("em-dash")
        if self.banned_phrases:
            flags.append("banned-phrase")
        if self.fidelity_dropped_numbers or self.fidelity_added_numbers:
            flags.append("fidelity-number")
        if self.fidelity_dropped_citations:
            flags.append("fidelity-citation")
        if self.rhetoric and self.rhetoric.leakage:
            flags.append("task-leakage")
        return flags

    @property
    def rhetoric_advisory(self) -> bool:
        """Anaphoric triads, meta-commentary asides, and scene-then-absence-punches
        are real AI reflexes but each has a nonzero measured false-positive rate
        against real prose (see tests/test_rhetoric.py), so they are advisory like
        burstiness and hedging, not a hard gate. Leakage is different -- see
        hard_flags -- because it is the model breaking character, not a tic."""
        r = self.rhetoric
        return bool(r and (r.triads or r.meta_asides or r.absence_punches))

    @property
    def presence_missing(self) -> bool:
        """No first-person reading and nothing that ran against expectation.

        Reported as a hard problem rather than an advisory because it is the failure mode a clean
        fingerprint cannot see: the draft matches the author's distribution and still has no author
        in it.
        """
        p = self.presence
        if not p or not p.get("measurable"):
            return False
        d = p.get("densities", {})
        return d.get("epistemic", 0.0) == 0.0 and d.get("expectation", 0.0) == 0.0

    @property
    def is_clean(self) -> bool:
        return not self.hard_flags and not self.banned_words and not self.presence_missing


def analyze(
    text: str,
    cal: ScrubCalibration,
    source: str | None = None,
    detect: bool = False,
    fp: "Fingerprint | None" = None,
    pres: "presence_mod.PresenceCalibration | None" = None,
    rhet: "rhetoric_mod.RhetoricCalibration | None" = None,
) -> ScrubReport:
    """Score a draft against a calibration.

    Pass ``source`` to run the fidelity audit, and ``fp`` to add the two-sided
    fingerprint fit check. ``fp`` is optional so existing callers are unaffected,
    but every caller that can supply one should: without it this function only
    detects AI tells, never a draft that is simply unlike the author.
    """
    rep = ScrubReport()
    # Same counter the scalpel uses, or the report and the fix disagree.
    rep.emdash_count = count_dashes(text)
    rep.dash_band_per1k = cal.dash_per1k_p95
    rep.word_count = len(_WORD_RE.findall(text))

    lower = text.lower()
    rep.banned_words = [
        w for w in cal.banned_words if re.search(rf"\b{re.escape(w)}\b", lower)
    ]
    rep.banned_phrases = [p for p in cal.banned_phrases if p.lower() in lower]

    sents = _split_sentences(text)
    sent_lens = [n for n in (len(_WORD_RE.findall(s)) for s in sents) if n > 0]
    wc = sum(sent_lens) or 1
    rep.hedge_density = len(_HEDGE_RE.findall(text)) * 200 / wc
    rep.hedge_ceiling = cal.hedge_ceiling
    rep.burstiness = statistics.stdev(sent_lens) if len(sent_lens) >= 2 else 0.0
    rep.burstiness_floor = cal.burstiness_floor

    if source is not None and source.strip():
        rep.checked_fidelity = True
        s_nums, o_nums = _numbers(source), _numbers(text)
        rep.fidelity_dropped_numbers = sorted(s_nums - o_nums, key=lambda x: (len(x), x))
        rep.fidelity_added_numbers = sorted(o_nums - s_nums, key=lambda x: (len(x), x))
        rep.fidelity_dropped_citations = sorted(_citations(source) - _citations(text))

    if fp is not None:
        rep.fp_checked = True
        rep.fp_distance = fp.distance(text)
        rep.fp_baseline = fp.self_baseline
        rep.fp_worst = fp.worst_features(text, k=3)
        # Prefer the per-voice p95 stored at calibration. Fingerprints written
        # before that field existed fall back to the ratio rule, which is looser
        # on some corpora and tighter on others; the report says which was used.
        rep.fp_threshold_calibrated = fp.fit_threshold > 0
        rep.fp_threshold = (
            fp.fit_threshold if rep.fp_threshold_calibrated
            else fp.self_baseline * FIT_RATIO
        )

    rep.presence = presence_mod.analyze(text, pres)
    rep.rhetoric = rhetoric_mod.analyze(
        text,
        rhet or rhetoric_mod.RhetoricCalibration(
            cleft_p95=cal.cleft_p95, antithesis_p95=cal.antithesis_p95,
            cleft_p25=cal.cleft_p25, antithesis_p25=cal.antithesis_p25,
        ),
    )

    if detect:
        rep.detector = detector_signal(text)
    return rep


def render(rep: ScrubReport, author: str = "the author") -> str:
    """Human-readable scrub report (the MCP tool / CLI output)."""
    style: list[str] = []
    if rep.presence:
        style.extend(presence_mod.render_lines(rep.presence, author))
    if rep.emdash_count:
        # An author with a calibrated dash habit is held to THEIR rate, not zero.
        band = getattr(rep, "dash_band_per1k", 0.0)
        wc = getattr(rep, "word_count", 0) or 0
        rate = rep.emdash_count * 1000 / wc if wc else 0.0
        if band >= AUTHOR_DASH_BAND_MIN:
            if rate > band:
                style.append(
                    f"- [MEDIUM] Dashes: {rate:.1f}/1000 words against this author's "
                    f"calibrated {band:.1f}. Thin them out; do not remove them all."
                )
        else:
            style.append(
                f"- [HIGH] Em-dashes: found {rep.emdash_count}. Target is zero. "
                f"Replace with commas, colons, or split the sentence."
            )
    if rep.banned_phrases:
        style.append(
            f"- [HIGH] AI cliche phrases: {'; '.join(rep.banned_phrases)}. Rephrase these lines."
        )
    if rep.banned_words:
        style.append(
            f"- [MEDIUM] AI-tell vocabulary outside {author}'s lexicon: "
            f"{', '.join(rep.banned_words)}. Replace with plain equivalents."
        )
    if rep.burstiness and rep.burstiness < rep.burstiness_floor:
        # The old wording here was "mix a short punchy line with a long one",
        # and it was actively harmful. Burstiness is sentence-length STDEV, an
        # order-invariant statistic, and the cheapest way for a model to raise it
        # is rigid long/short alternation. Audited over 35 shipped generations
        # (evals/cadence_audit.py) that is exactly what came out:
        # sl_delta_abs +2.32 SD, sl_autocorr1 -2.05 SD, sl_short_after_long
        # +1.68 SD -- the draft hits the stdev target while acquiring a metronomic
        # rhythm the author does not have. Classic Goodhart: the proxy was
        # satisfied and the thing it proxied for got worse.
        style.append(
            f"- [MEDIUM] Low burstiness: sentence-length stdev {rep.burstiness:.2f} < floor "
            f"{rep.burstiness_floor:.2f}. Let some passages run long and others stay short, "
            f"rather than alternating long/short line by line: regular alternation raises this "
            f"number while making the rhythm more mechanical, not less. "
            f"(Advisory: flagged, not auto-enforced.)"
        )
    if rep.hedge_density > rep.hedge_ceiling + 0.2:
        style.append(
            f"- [MEDIUM] Hedge overuse: {rep.hedge_density:.2f}/200w > ceiling "
            f"{rep.hedge_ceiling:.2f}. Prefer definite verbs."
        )
    if rep.rhetoric:
        # render_lines already frames leakage as its own [HIGH] line, distinctly from
        # the [MEDIUM] tics -- see rhetoric.render_lines. Nothing to add here.
        style.extend(rhetoric_mod.render_lines(rep.rhetoric))
    if rep.fit_off or rep.fit_drifting:
        drift = ", ".join(
            f"{name} {abs(z):.1f} SD {'above' if z > 0 else 'below'} corpus"
            for name, z in rep.fp_worst
        )
        if rep.fit_off:
            rule = "corpus p95" if rep.fp_threshold_calibrated else "1.6x baseline, uncalibrated"
            style.append(
                f"- [HIGH] Off-voice: fingerprint distance {rep.fp_distance:.2f} > "
                f"threshold {rep.fp_threshold:.2f} ({rule}; {author}'s baseline "
                f"{rep.fp_baseline:.2f}). Furthest features: {drift}. This is not an "
                f"AI tell, it is a mismatch: the draft is measurably unlike {author} "
                f"on these axes. Move each toward the corpus."
            )
        else:
            style.append(
                f"- [MEDIUM] Drifting off-voice: fingerprint distance "
                f"{rep.fp_distance:.2f} vs {author}'s baseline {rep.fp_baseline:.2f} "
                f"(under the {rep.fp_threshold:.2f} hard threshold, so this is "
                f"advisory). Furthest features: {drift}. Worth a look if the draft "
                f"reads flat."
            )

    fidelity: list[str] = []
    if rep.checked_fidelity:
        if rep.fidelity_dropped_numbers:
            fidelity.append(
                f"- [HIGH] Numbers in source missing/altered in output: "
                f"{', '.join(rep.fidelity_dropped_numbers)}. Restore the exact value."
            )
        if rep.fidelity_added_numbers:
            fidelity.append(
                f"- [HIGH] Numbers in output not in source (possible fabrication): "
                f"{', '.join(rep.fidelity_added_numbers)}. Remove or verify each."
            )
        if rep.fidelity_dropped_citations:
            fidelity.append(
                f"- [HIGH] Citations/acronyms in source missing from output: "
                f"{', '.join(rep.fidelity_dropped_citations)}. Confirm none were dropped."
            )

    if rep.is_clean and not style:
        body = "CLEAN: matches the calibrated voice and shows no AI tells."
        if rep.fp_checked:
            body += (
                f" Fingerprint fit {rep.fp_distance:.2f} "
                f"(threshold {rep.fp_threshold:.2f}, baseline {rep.fp_baseline:.2f})."
            )
            if not rep.fp_threshold_calibrated:
                body += (
                    "\n[note] Fingerprint predates per-voice fit thresholds; using a "
                    "ratio fallback. Re-run: mimesis calibrate."
                )
        else:
            # Say so. A CLEAN with no fit check means "no AI tells found", which
            # is a much weaker claim than "sounds like the author".
            body += (
                "\n[note] No fingerprint available for this voice, so only AI "
                "tells were checked, not resemblance. Run: mimesis calibrate."
            )
    else:
        body = "AI FOOTPRINT SCRUBBER REPORT\n"
        body += "## STYLE\n" + ("\n".join(style) if style else "- Clean.") + "\n"
        if rep.checked_fidelity:
            body += "## FIDELITY (numbers, citations)\n" + (
                "\n".join(fidelity) if fidelity else "- Preserved (none dropped, altered, or invented)."
            ) + "\n"
        body += "\nREPAIR: fix every [HIGH]/[MEDIUM]; fidelity issues take priority. Never add em-dashes."

    if rep.detector:
        det = rep.detector
        body += (
            f"\n\n[detector] {'signal=' + str(det) if det.get('available') else det.get('detail', '')}"
        )
    return body


# --- scalpel ------------------------------------------------------------------

# Runs of two OR MORE hyphens, plus the en-dash. The old pattern was `--`
# exactly, which left a stray hyphen behind on longer runs: this author writes
# "aflame--- heat" and the scalpel turned it into "aflame, - heat", inserting a
# dangling hyphen into the middle of his prose. His corpus contains that habit,
# so generations reproduce it and every one of them was being mangled. En-dash is
# included because a model that has been told not to use em-dashes will often
# reach for it instead.
# A line made only of dashes is a horizontal RULE, not punctuation. Counting
# those made a folder README score 27 dashes per 100 words and dragged an entire
# corpus mean from ~0 to 1.24 (ASCII underlines like "------" matched). Rules are
# removed before counting or editing rather than excluded by lookaround, because
# the lookaround version silently rewrote "HEADER\n------\nbody" into
# "HEADER\n-, -\nbody". Prose dashes always have text on at least one side.
_RULE_LINE_RE = re.compile(r"(?m)^[ \t]*[-—–=_*]{3,}[ \t]*$")


def _without_rule_lines(text: str) -> str:
    return _RULE_LINE_RE.sub("", text)


_EMDASH_RE = re.compile(r"[ \t]*(—|–|-{2,})[ \t]*")

# In LaTeX, "--" and "---" are not punctuation the author typed: they are the
# SOURCE ENCODING for an en-dash and an em-dash. Applying the plain-text pattern
# to a .tex file turned the numeric range 0.17--0.21 into "0.17, 0.21" in three
# table rows, converting one interval into two point estimates and contradicting
# the body text that described it as a range. Six ranges were destroyed in a
# single pass and every downstream check reported clean, because both numbers
# were still present and the asset verifier compares number SETS.
#
# So the LaTeX rule targets only what actually renders as an em-dash: a literal
# em-dash character, or a run of three or more hyphens. "--" is left alone.
_EMDASH_TEX_RE = re.compile(r"[ \t]*(—|-{3,})[ \t]*")


def _dash_re(fmt: str) -> re.Pattern:
    return _EMDASH_TEX_RE if fmt == "latex" else _EMDASH_RE


def _resolve_fmt(text: str, fmt: str | None) -> str:
    """Sniff when undeclared. Sniffing is the safe default: guessing "latex" on
    plain prose requires a backslash command to be present, while guessing
    "plain" on real LaTeX is the failure that corrupted a manuscript."""
    kind = (fmt or textnorm.guess_format(text)).lower()
    return {"tex": "latex", "md": "markdown"}.get(kind, kind)


def _editable(text: str, fmt: str) -> list[tuple[int, int]]:
    """Complement of the protected spans: the regions a surface fix may touch."""
    out, cursor = [], 0
    for a, b in textnorm.protected_spans(text, fmt):
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < len(text):
        out.append((cursor, len(text)))
    return out


def count_dashes(text: str, fmt: str | None = None) -> int:
    """Prose dashes only: em, en, or a hyphen run, ignoring rule lines, markup
    and data. ``fmt`` is sniffed when not given."""
    kind = _resolve_fmt(text, fmt)
    stripped = _without_rule_lines(text)
    pat = _dash_re(kind)
    return sum(len(pat.findall(stripped[a:b])) for a, b in _editable(stripped, kind))


def scalpel(text: str, fmt: str | None = None, allow_dashes: bool = False) -> tuple[str, int]:
    """Deterministic, meaning-preserving surface fix: strip em-dashes.

    Returns (fixed_text, replacements). This is the only edit the scrubber makes
    on its own; everything else is flagged for the writer. Never rewrites prose to
    hit a target mean, since that collapse is what destroys the voice.

    Two classes of region are never touched. Rule lines, because a row of dashes
    under a heading is structure and turning it into ", " corrupts the document.
    And markup or data spans -- math, tables, citation keys, numeric ranges --
    because an edit there is a change to the content rather than to the prose.

    ``allow_dashes`` turns the strip off for an author whose own corpus uses
    prose dashes as punctuation (see AUTHOR_DASH_BAND_MIN). For them this edit
    is not de-AI-ing a draft, it is deleting a habit: one measured LinkedIn
    corpus ran 4.9 dashes per 1000 words, and stripping every one flattens the
    most distinctive punctuation in the voice. Those drafts are flagged against
    the author's calibrated rate instead of silently rewritten.
    """
    if allow_dashes:
        return text, 0
    kind = _resolve_fmt(text, fmt)
    pat = _dash_re(kind)
    spans = _editable(text, kind)
    pieces, n, cursor = [], 0, 0
    for a, b in spans:
        pieces.append(text[cursor:a])
        chunk = text[a:b]
        fixed_lines = []
        for line in chunk.split("\n"):
            if _RULE_LINE_RE.fullmatch(line):
                fixed_lines.append(line)
                continue
            n += len(pat.findall(line))
            fixed_lines.append(pat.sub(", ", line))
        pieces.append("\n".join(fixed_lines))
        cursor = b
    pieces.append(text[cursor:])
    fixed = "".join(pieces)
    fixed = re.sub(r",[ \t]*,", ",", fixed)
    return fixed, n
