"""Source-relative verification for REWRITES, as opposed to generations.

The rest of this package answers "does this read like the author?" by comparing a
draft against a corpus. That is the right question when writing something new. It
is the wrong question, on its own, when rewriting a document that already exists,
and a LaTeX rewrite pass demonstrated why: it verified numbers, citations,
cross-references, labels, environments and inline math, found all six clean,
and shipped a manuscript that had

  * lost ``\\section{Results}``, replaced by a literal ``[memory: ...]`` token
    that leaked out of the generation harness,
  * turned six numeric ranges ``0.17--0.21`` into pairs ``0.17, 0.21``,
  * reduced sentence-length variance by 9.8%,
  * introduced second person into a Results section that had none,
  * and put a banned construction into a section heading.

Every one of those is invisible to a corpus comparison, because the corpus does
not know what this document was before. They are visible immediately against the
source. That is the whole idea here: for a rewrite, THE SOURCE IS THE SPEC.

Two design rules follow from the failure:

1. A check that only reports is not a check. Everything here returns a severity,
   and ``hard`` findings are meant to reject a candidate, not annotate it. The
   LaTeX pass computed a fingerprint distance for every subsection and printed it
   to a markdown report while accepting on assets alone. The numbers were real
   and nothing consumed them.

2. Rhythm is measured against the source, not a target. "Sound more like the
   author" is a direction; "do not flatten what is already there" is a
   constraint, and only the second one survives contact with a model that will
   happily chop long sentences into medium ones and call it clarity. Burstiness
   is the best-attested human/machine discriminator in the detection literature,
   and it is precisely what a well-meaning rewrite destroys.
"""
from __future__ import annotations

import re
import statistics as st
from dataclasses import dataclass, field

from . import rhetoric as rhetoric_mod
from . import textnorm

__all__ = ["Finding", "FidelityReport", "Tolerances", "verify", "render"]


@dataclass
class Finding:
    axis: str
    severity: str            # "hard" rejects; "soft" is reported for judgment
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Tolerances:
    """Every number here is a policy choice, so they live in one visible place.

    Defaults are deliberately tight on the axes where a regression is silent and
    loose on the axes where honest variation is normal.
    """

    # Rhythm. A rewrite may vary; it may not systematically flatten.
    #
    # 5% is a policy choice, not a measured noise floor -- it is set loose enough
    # that ordinary rewriting does not trip it and tight enough to have caught the
    # -9.8% that shipped. Two observations so far: the failed whole-block pass at
    # -9.8% (rejected) and one live subsection at -4.0% (accepted). That is not
    # enough to know where run-to-run variation actually sits, so if this fires on
    # work that is genuinely good, measure the distribution over ~20 resamples of
    # one passage before loosening it, rather than nudging the number until the
    # gate goes quiet.
    sd_drop: float = 0.05
    # Tails carry burstiness. Losing a third of either one is a shape change even
    # when the standard deviation happens to survive.
    tail_drop: float = 0.34
    long_words: int = 35     # a "long" sentence
    short_words: int = 12    # a "short" sentence
    # Register. A marker absent from the source may not be introduced at all;
    # one already present may grow by this ratio before it counts as drift.
    register_growth: float = 1.5
    register_slack: int = 2  # absolute headroom, so 1 -> 2 is not a finding
    # Length. Kept here so callers stop hand-rolling it.
    max_growth: float = 0.10


# Markers whose presence is a property of REGISTER, not of voice. An author who
# writes "you" constantly in essays does not write it in a Results section, so
# the corpus cannot arbitrate this and the source document must.
REGISTER_MARKERS: dict[str, str] = {
    "contraction": r"\b\w+'(?:s|t|re|ve|ll|d|m)\b",
    "second person": r"\b(?:you|your|yours|yourself)\b",
    "intensifier": r"\b(?:exactly|simply|actually|really|very|indeed|precisely|"
                   r"quite|utterly|genuinely)\b",
    "expletive there": r"\bthere(?:'s| is| are| was| were)\b",
    "rhetorical question": r"\?",
}

# Constructions the author has explicitly rejected, HARD when they appear in a
# heading: a heading is the most-read line in a section and the least likely to
# be revised later.
#
# The cleft detector is imported rather than rewritten. rhetoric.py's version was
# narrowed against 313,820 words of this author's prose after a first attempt
# fired 113 times at 360 per million against a ~145 per million published human
# baseline, and 105 of those were it-clefts, which are his own device. A second,
# looser copy here would undo that work: an early draft of this module used
# ``wh-word ... is (that|the|a|to)`` and flagged the ordinary embedded question
# "what a cell is, and why it is the unit" as a cleft.
_NOT_JUST_RE = re.compile(r"\b(?:not|isn't|aren't|wasn't|weren't|doesn't|don't)\s+"
                          r"(?:just|merely|only|simply)\b", re.I)
_COLON_HOOK_RE = re.compile(r"^[^:]{3,45}:\s+\S")

HEADING_TICS: dict[str, object] = {
    "not-just-X": lambda s: _NOT_JUST_RE.findall(s),
    "wh-cleft": rhetoric_mod.find_wh_clefts,
    "contrarian colon": lambda s: _COLON_HOOK_RE.findall(s),
}

# Text a generation harness emits that is never part of the document. This list
# exists because one of them reached a manuscript and compiled silently.
SCAFFOLD_RE = re.compile(
    r"\[(?:memory|tool|system|assistant|note|thinking|context)\b[^\]]{0,80}\]"
    r"|^(?:Here(?:'s| is) (?:the|your|a)\b|Sure[,!]|Certainly[,!]|I've rewritten"
    r"|Rewritten version|Note:|Output:)"
    r"|^```",
    re.I | re.M,
)

HEDGES = (r"\b(?:may|might|could|can|appears? to|seems? to|tends? to|we read|"
          r"we interpret|suggests?|indicates?|largely|generally|typically|"
          r"broadly|released|reported|observed|approximately|about)\b")
UNIVERSALS = (r"\b(?:never|always|every|all|none|no \w+ (?:has|have) ever|"
              r"cannot|impossible|invariably|without exception)\b")

_RANGE_RE = re.compile(r"\d[\d,.]*\s*(?:-{2,3}|\u2013|\u2014)\s*\d[\d,.]*")
_TEX_HEADING = re.compile(r"\\((?:sub){0,2}section)\*?\{([^{}]*)\}")
_MD_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.*)$")


# --- helpers ------------------------------------------------------------------

def _sentences(prose: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.split()) >= 2]


def _lengths(prose: str) -> list[int]:
    return [len(s.split()) for s in _sentences(prose)]


def _headings(text: str, fmt: str) -> list[tuple[str, str]]:
    if fmt == "latex":
        return [(m.group(1), m.group(2)) for m in _TEX_HEADING.finditer(text)]
    if fmt == "markdown":
        return [(m.group(1), m.group(2)) for m in _MD_HEADING.finditer(text)]
    return []


def _share(lengths: list[int], pred) -> float:
    return (sum(1 for L in lengths if pred(L)) / len(lengths)) if lengths else 0.0


def _dropped(ratio_new: float, ratio_old: float, tol: float) -> bool:
    """True when ``ratio_new`` fell more than ``tol`` below ``ratio_old``."""
    return ratio_old > 0 and ratio_new < ratio_old * (1 - tol)


# --- report -------------------------------------------------------------------

@dataclass
class FidelityReport:
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def hard(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "hard"]

    @property
    def soft(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "soft"]

    @property
    def ok(self) -> bool:
        return not self.hard

    def problems(self) -> list[str]:
        """One line per hard finding, for callers that reject on a string list."""
        return [f"{f.axis}: {f.detail}" for f in self.hard]


# --- the checks ---------------------------------------------------------------

def verify(
    source: str,
    output: str,
    fmt: str | None = None,
    tol: Tolerances | None = None,
) -> FidelityReport:
    """Compare a rewrite against the document it rewrote."""
    tol = tol or Tolerances()
    kind = (fmt or textnorm.guess_format(source)).lower()
    kind = {"tex": "latex", "md": "markdown"}.get(kind, kind)
    rep = FidelityReport()
    add = rep.findings.append

    src_p = textnorm.to_prose(source, kind)
    out_p = textnorm.to_prose(output, kind)

    # 1. Scaffolding. Hard, unconditional: this is never authorial text.
    leaks = SCAFFOLD_RE.findall(output)
    if leaks:
        add(Finding("scaffold", "hard",
                    f"{len(leaks)} harness artifact(s) in the output",
                    [str(x)[:70] for x in leaks[:5]]))

    # 2. Structure. Counts per level, not titles: a rewrite is allowed to retitle
    #    a section and is not allowed to lose one. Comparing titles would have
    #    produced noise on the very change the pass was asked to make, while
    #    comparing counts catches a \section that became a stray token.
    s_head, o_head = _headings(source, kind), _headings(output, kind)
    s_lvl: dict[str, int] = {}
    o_lvl: dict[str, int] = {}
    for lvl, _ in s_head:
        s_lvl[lvl] = s_lvl.get(lvl, 0) + 1
    for lvl, _ in o_head:
        o_lvl[lvl] = o_lvl.get(lvl, 0) + 1
    for lvl in sorted(set(s_lvl) | set(o_lvl)):
        a, b = s_lvl.get(lvl, 0), o_lvl.get(lvl, 0)
        if a != b:
            add(Finding("structure", "hard",
                        f"{lvl}: {a} in source, {b} in output",
                        [t for l, t in s_head if l == lvl][:6]))

    # 3. House style: numeric ranges. A range that becomes a pair changes an
    #    interval into two point estimates. The asset checker sees both numbers
    #    present and reports clean, which is exactly how this shipped.
    s_rng, o_rng = _RANGE_RE.findall(source), _RANGE_RE.findall(output)
    if len(o_rng) < len(s_rng):
        lost = [r for r in s_rng if r not in o_rng]
        add(Finding("house-style", "hard",
                    f"{len(s_rng) - len(o_rng)} numeric range(s) no longer written "
                    f"as ranges",
                    lost[:6]))

    # 4. Rhythm. Measured on prose, against the source.
    sl, ol = _lengths(src_p), _lengths(out_p)
    if len(sl) >= 8 and len(ol) >= 8:
        s_sd, o_sd = st.pstdev(sl), st.pstdev(ol)
        s_long, o_long = (_share(sl, lambda L: L >= tol.long_words),
                          _share(ol, lambda L: L >= tol.long_words))
        s_short, o_short = (_share(sl, lambda L: L < tol.short_words),
                            _share(ol, lambda L: L < tol.short_words))
        rep.metrics.update(
            src_sentences=len(sl), out_sentences=len(ol),
            src_mean=round(st.mean(sl), 2), out_mean=round(st.mean(ol), 2),
            src_sd=round(s_sd, 2), out_sd=round(o_sd, 2),
            sd_delta=round(o_sd - s_sd, 2),
            src_long=round(s_long, 3), out_long=round(o_long, 3),
            src_short=round(s_short, 3), out_short=round(o_short, 3),
        )
        if _dropped(o_sd, s_sd, tol.sd_drop):
            add(Finding("rhythm", "hard",
                        f"sentence-length variance fell {(o_sd/s_sd - 1):+.1%} "
                        f"(sd {s_sd:.1f} -> {o_sd:.1f}); burstiness is the property "
                        f"this pass exists to protect",
                        [f"mean {st.mean(sl):.1f} -> {st.mean(ol):.1f} words"]))
        if _dropped(o_long, s_long, tol.tail_drop):
            add(Finding("rhythm", "hard",
                        f"long sentences (>={tol.long_words}w) fell "
                        f"{s_long:.0%} -> {o_long:.0%}; long sentences were chopped "
                        f"into medium ones"))
        if _dropped(o_short, s_short, tol.tail_drop):
            add(Finding("rhythm", "hard",
                        f"short sentences (<{tol.short_words}w) fell "
                        f"{s_short:.0%} -> {o_short:.0%}; the punch lines went"))

    # 5. Register lock. The source document, not the corpus, sets the ceiling.
    for name, pat in REGISTER_MARKERS.items():
        a = len(re.findall(pat, src_p, re.I))
        b = len(re.findall(pat, out_p, re.I))
        rep.metrics[f"reg:{name}"] = (a, b)
        if a == 0 and b > 0:
            hits = re.findall(pat, out_p, re.I)
            add(Finding("register", "hard",
                        f"introduced {name} ({b}x) into a document that used none",
                        [str(h) for h in hits[:5]]))
        elif b > max(a * tol.register_growth, a + tol.register_slack):
            add(Finding("register", "soft",
                        f"{name} {a} -> {b}, beyond the source's own rate"))

    # 6. Constructions. Source-relative like everything else here: a tic counts
    #    only when the rewrite INTRODUCED it. The source of the failed pass
    #    already used a colon hook in one heading, so flagging every colon would
    #    have reported the author's own habit back to him as a defect. Headings
    #    are paired by position, which is only meaningful when none were lost --
    #    and if any were, structure has already failed above.
    if len(s_head) == len(o_head):
        for (s_lv, s_t), (o_lv, o_t) in zip(s_head, o_head):
            for name, find in HEADING_TICS.items():
                if find(o_t) and not find(s_t):
                    add(Finding("construction", "hard",
                                f"{name} introduced into a {o_lv} heading",
                                [f"was: {s_t}", f"now: {o_t}"]))
    body_hits: list[str] = []
    for name, find in HEADING_TICS.items():
        if name == "contrarian colon":
            continue  # a colon mid-paragraph is punctuation, not a hook
        a, b = len(find(src_p)), len(find(out_p))
        if b > a:
            body_hits.append(f"{name} {a} -> {b}")
    if body_hits:
        add(Finding("construction", "soft",
                    "constructions the author rejects became more frequent",
                    body_hits))

    # 7. Claim strength. Soft by necessity: deciding whether a dropped hedge
    #    changed the claim needs a reader. Loud by design, because nothing else
    #    in the pipeline looks at meaning at all, and the observed failures were
    #    a hedge deletion ("no released cycle" -> "no cycle has ever") and an
    #    interpretation promoted to a finding.
    s_h, o_h = (len(re.findall(HEDGES, src_p, re.I)),
                len(re.findall(HEDGES, out_p, re.I)))
    s_u, o_u = (len(re.findall(UNIVERSALS, src_p, re.I)),
                len(re.findall(UNIVERSALS, out_p, re.I)))
    rep.metrics.update(hedges=(s_h, o_h), universals=(s_u, o_u))
    if o_h < s_h or o_u > s_u:
        new_u = [u for u in re.findall(UNIVERSALS, out_p, re.I)
                 if u.lower() not in {x.lower() for x in
                                      re.findall(UNIVERSALS, src_p, re.I)}]
        add(Finding("claim-strength", "soft",
                    f"hedges {s_h} -> {o_h}, universal claims {s_u} -> {o_u}; "
                    f"a dropped hedge is a strengthened claim and no other check "
                    f"can see it",
                    new_u[:6]))

    # 8. Length.
    sw, ow = len(source.split()), len(output.split())
    rep.metrics["words"] = (sw, ow)
    if sw and ow / sw - 1 > tol.max_growth:
        add(Finding("length", "hard",
                    f"grew {ow/sw - 1:+.0%} ({sw} -> {ow} words)"))
    return rep


def render(rep: FidelityReport) -> str:
    """Human-readable summary. Hard findings first; they are the blocking ones."""
    if not rep.findings:
        return "fidelity: clean (structure, rhythm, register, claims, style)"
    out = []
    for f in rep.hard + rep.soft:
        tag = "HARD" if f.severity == "hard" else "soft"
        out.append(f"[{tag}] {f.axis}: {f.detail}")
        for e in f.evidence[:4]:
            out.append(f"         - {e}")
    return "\n".join(out)
