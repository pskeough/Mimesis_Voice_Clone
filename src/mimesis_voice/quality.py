"""Is the writing any good? -- a separate axis from "does it sound like the author".

Every other check in this package answers one question: is this text like the
corpus. None of them can fail a draft for being boring, repetitive, or empty. A
flat, hedged, contentless passage that happens to match the author's punctuation
rates and sentence-length distribution passes the fingerprint, the scrubber, the
presence floors and the rhetoric checks simultaneously. That is a real gap, and
it is the one that lets a system produce prose that is technically in-voice and
not worth reading.

These four measures are deliberately cheap, deterministic and parse-free, so they
can run on every slate candidate without a model call:

* **redundancy** -- the highest content-word overlap between any two sentences.
  Saying the same thing twice is the most common failure in generated prose and
  the easiest to miss on a fast read, because both sentences are individually
  fine.
* **information density** -- distinct content lemmas per 100 words, with function
  words removed. Separates prose that carries information from prose that carries
  connective tissue.
* **specificity** -- the fraction of sentences containing a concrete anchor: a
  number, a proper noun, a quoted term, a named entity. Abstract prose that never
  touches ground is the register a model defaults to under uncertainty.
* **filler** -- rate of phrases that occupy space without adding content ("it is
  important to note", "in order to", "the fact that").

## Absolute or author-relative?

Both, deliberately, and the split matters.

Redundancy and filler are scored ABSOLUTELY. Repeating yourself and padding are
defects in anyone's prose; there is no author for whom a 0.8 sentence-pair
overlap is a stylistic choice worth preserving.

Density and specificity are scored AGAINST THE AUTHOR'S OWN BAND, because they
are genre-bound. A methods section is legitimately denser than a lyric essay, and
a fiction writer's specificity profile is not a researcher's. Holding those to a
universal threshold would penalise a register rather than a failure -- the same
mistake the lexical banlist made before it was calibrated per author.

Nothing here is a gate by default. Quality is the axis most likely to disagree
with a human reader, so it reports and lets the LLM judge and the writer decide.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")

# Closed-class words carry grammar rather than content. Removing them is what
# makes "distinct lemmas" mean "distinct ideas" instead of "distinct tokens".
_FUNCTION = frozenset("""
a an the and or but nor so yet for of to in on at by with from into onto upon
about over under again further then once here there all any both each few more
most other some such no not only own same than too very can will just don should
now is are was were be been being am do does did doing have has had having i you
he she it we they them his her its our their this that these those as if because
while when where which who whom whose what how why would could may might must
shall am s t d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma
mightn mustn needn shan shouldn wasn weren won wouldn
""".split())

_FILLER = re.compile(
    r"\b(?:it is important to note(?: that)?|it should be noted(?: that)?|"
    r"in order to|the fact that|due to the fact that|at this point in time|"
    r"it is worth noting(?: that)?|needless to say|as a matter of fact|"
    r"for all intents and purposes|in terms of|with regard to|"
    r"a number of|a variety of|in the process of)\b",
    re.IGNORECASE,
)

# A concrete anchor: a digit, a mid-sentence capitalised word (proper noun), a
# quoted term, or a percentage. Sentence-initial capitals are excluded, which is
# why the proper-noun branch requires a preceding word.
_ANCHOR = re.compile(
    r"\d|\b\w+\s+[A-Z][a-z]{2,}|[\"“'']\w+[\"”'']|%"
)


def _sentences(t: str) -> list[str]:
    out: list[str] = []
    for para in t.split("\n"):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return out


def _content(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s) if w.lower() not in _FUNCTION and len(w) > 2}


@dataclass
class QualityReport:
    redundancy: float = 0.0        # 0-1, worst sentence-pair content overlap
    redundant_pair: tuple[str, str] | None = None
    density: float = 0.0           # distinct content lemmas per 100 words
    specificity: float = 0.0       # fraction of sentences with a concrete anchor
    filler_per_1kw: float = 0.0
    n_words: int = 0
    # author bands, when a calibration was supplied
    density_p25: float = 0.0
    specificity_p25: float = 0.0

    # Absolute defects: true for any author.
    @property
    def is_repetitive(self) -> bool:
        return self.redundancy >= 0.60

    @property
    def is_padded(self) -> bool:
        return self.filler_per_1kw > 6.0

    # Author-relative: a register, not a universal standard.
    @property
    def is_thin(self) -> bool:
        return self.density_p25 > 0 and self.density < self.density_p25

    @property
    def is_vague(self) -> bool:
        return self.specificity_p25 > 0 and self.specificity < self.specificity_p25

    @property
    def flags(self) -> list[str]:
        f = []
        if self.is_repetitive:
            f.append("repetitive")
        if self.is_padded:
            f.append("padded")
        if self.is_thin:
            f.append("thin")
        if self.is_vague:
            f.append("vague")
        return f


@dataclass
class QualityCalibration:
    density_p25: float = 0.0
    specificity_p25: float = 0.0
    density_median: float = 0.0
    specificity_median: float = 0.0
    n_pieces: int = 0

    def to_dict(self) -> dict:
        return {
            "density_p25": self.density_p25,
            "specificity_p25": self.specificity_p25,
            "density_median": self.density_median,
            "specificity_median": self.specificity_median,
            "n_pieces": self.n_pieces,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QualityCalibration":
        return cls(**{k: d.get(k, 0) for k in
                      ("density_p25", "specificity_p25", "density_median",
                       "specificity_median", "n_pieces")})


def measure(text: str, cal: QualityCalibration | None = None) -> QualityReport:
    words = _WORD.findall(text)
    n = len(words)
    rep = QualityReport(n_words=n)
    if n < 40:
        return rep
    sents = _sentences(text)

    # Redundancy: worst pair, not the mean. One duplicated claim in an otherwise
    # varied passage is the defect; averaging hides it.
    worst, pair = 0.0, None
    csets = [(s, _content(s)) for s in sents]
    for i in range(len(csets)):
        for j in range(i + 1, len(csets)):
            a, b = csets[i][1], csets[j][1]
            if len(a) < 4 or len(b) < 4:
                continue
            ov = len(a & b) / min(len(a), len(b))
            if ov > worst:
                worst, pair = ov, (csets[i][0], csets[j][0])
    rep.redundancy, rep.redundant_pair = worst, pair

    content = [w.lower() for w in words if w.lower() not in _FUNCTION and len(w) > 2]
    rep.density = 100.0 * len(set(content)) / n
    rep.specificity = (sum(1 for s in sents if _ANCHOR.search(s)) / len(sents)) if sents else 0.0
    rep.filler_per_1kw = 1000.0 * len(_FILLER.findall(text)) / n

    if cal:
        rep.density_p25 = cal.density_p25
        rep.specificity_p25 = cal.specificity_p25
    return rep


def calibrate(texts: list[str], min_words: int = 200) -> QualityCalibration:
    """Author's own density and specificity range.

    p25 is the floor: a draft below the 25th percentile of the author's own
    corpus is thinner or vaguer than three quarters of what they write. No
    ceiling, because unusually dense or specific prose is not a defect.
    """
    usable = [t for t in texts if len(_WORD.findall(t)) >= min_words]
    if len(usable) < 5:
        return QualityCalibration()
    ds, ss = [], []
    for t in usable:
        r = measure(t)
        ds.append(r.density)
        ss.append(r.specificity)
    ds.sort(); ss.sort()
    q = lambda v, p: v[int(p * (len(v) - 1))]
    return QualityCalibration(
        density_p25=q(ds, 0.25), specificity_p25=q(ss, 0.25),
        density_median=statistics.median(ds), specificity_median=statistics.median(ss),
        n_pieces=len(usable),
    )


def render_lines(rep: QualityReport) -> list[str]:
    out: list[str] = []
    if rep.is_repetitive and rep.redundant_pair:
        a, b = rep.redundant_pair
        out.append(
            f"- [HIGH] Repetition: two sentences share {rep.redundancy:.0%} of their "
            f"content words. \"{a[:70]}...\" / \"{b[:70]}...\". Cut one or merge them."
        )
    if rep.is_padded:
        out.append(
            f"- [MEDIUM] Filler: {rep.filler_per_1kw:.1f} empty phrases per 1000 words "
            f"(\"in order to\", \"the fact that\"). Delete them; nothing is lost."
        )
    if rep.is_thin:
        out.append(
            f"- [MEDIUM] Thin: {rep.density:.1f} distinct content words per 100 vs this "
            f"author's floor of {rep.density_p25:.1f}. The prose is carrying more "
            f"connective tissue than content."
        )
    if rep.is_vague:
        out.append(
            f"- [MEDIUM] Unanchored: only {rep.specificity:.0%} of sentences contain a "
            f"number, name or quoted term, against this author's floor of "
            f"{rep.specificity_p25:.0%}. Abstraction without a referent is the register "
            f"a model defaults to when it has nothing specific to say."
        )
    return out
