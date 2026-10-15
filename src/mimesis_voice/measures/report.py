"""The structure block of the gate report: one line per measure, no total.

Six lines, in the order a reader needs them: coherence, openers, paragraphs,
cadence, vocabulary, then the four constructions. They are never combined. The
study spent weeks with three coherence features averaged inside a geometry index
and the looping they measured was invisible for all of it; the same averaging
would hide it here.

Every line is REPORTING, not a hard flag. The hard flags in `scrub.py` are
constructions the author rejected by name, measured at under 0.3 per 1,000 words
in his own corpus. These are bands fitted to a corpus, and a band at two
standard deviations flags a twentieth of the author's own writing by
construction. `DETECTORS.md`'s rule -- a check that fires on the author's real
prose is measuring the author -- is why they report instead of blocking, and the
measured self-flag rate is printed beside any line that fires so the reader can
see how often that line is wrong on this corpus.

Each line names the evidence rather than the number where it can: the echo line
quotes the sentences it counted, the opener line names the over-used class, the
construction line names the repeated openers.
"""
from __future__ import annotations

from . import construction, cpd, lexdiv, otd, per, psd
from .calibration import StructureCalibration
from .textbase import prose_paragraphs, sent_lengths, sentences

__all__ = ["Line", "analyse", "render", "repair_notes"]

# A construction z beyond this, in the machine direction, is quoted as a finding.
#
# Chosen by measurement, not by matching the band lines. Four candidate rules
# were scored on five of the study's authors: the share of the AUTHOR'S OWN
# held-out passages each rule flags, against the share of that author's base-model
# cells it catches (50 held-out passages and 60-180 machine cells per author,
# 2026-09-09; the table is in docs/STRUCTURE_GATES.md).
#
#   z >= 2   flags 6-28% of the authors' own prose      catches 17-83%
#   z >= 3   flags 0-8%                                 catches 13-73%
#   > the author's own corpus maximum   flags 0-4%      catches 6-56%
#
# DETECTORS.md's rule is that a check touching more than 15% of the author's own
# pieces is measuring the author. z >= 2 breaks that on three of the five; z >= 3
# clears it on all five and gives up 10-13 points of detection to do so. The
# corpus-maximum rule buys no false positives back and loses far more.
Z_FLAG = 3.0

_FLAG = "FLAG"
_OK = "ok"
_NA = "--"


class Line:
    """One report line. `status` is FLAG, ok, or -- for not measurable."""

    __slots__ = ("key", "status", "text", "detail")

    def __init__(self, key: str, status: str, text: str, detail: list[str] | None = None):
        self.key, self.status, self.text, self.detail = key, status, text, detail or []

    @property
    def flagged(self) -> bool:
        return self.status == _FLAG


def _scope(n_used: int, n_have: int, unit: str) -> str:
    """Says so when the comparison covers only the draft's opening.

    The band is matched to the draft's length; when the corpus has too few
    pieces that long, both sides are cut to the longest length it can match. The
    line must say that it read a prefix rather than the whole draft.
    """
    return f", first {n_used} {unit}" if n_used < n_have else ""


def _first_sentences(text: str, n: int) -> str:
    return " ".join([s for s in sentences(text) if s.strip()][:n])


def _first_paragraphs(text: str, n: int) -> str:
    return "\n\n".join([p for p in prose_paragraphs(text) if p.strip()][:n])


def _fp(cal: StructureCalibration, key: str) -> str:
    """The measured share of the author's own units this line flags, when it
    fires. A line that flags 18% of his real writing has said less than one
    that flags 3%, and the reader should be told which they are looking at."""
    r = cal.self_flag.get(key)
    return f"; flags {r:.0%} of the author's own pieces" if r is not None else ""


def _echo(text: str, cal: StructureCalibration) -> Line:
    v = per.worst_window(text)
    n_sent = int(v["per_n_sent"])
    if n_sent < 8:
        return Line("echo", _NA, f"coherence: {n_sent} scorable sentences, too few to rate")
    thr = cal.per_threshold()
    if thr is None:
        return Line("echo", _NA, "coherence: no calibration (too few corpus pieces)")
    limit, mu = thr
    where = (f" (worst of {v['n_windows']} 600-word windows, from word "
             f"{v['start_word']})") if v["n_windows"] > 1 else ""
    if v["per"] > limit:
        quoted = per.echo_sentences(v["text"])
        return Line("echo", _FLAG,
                    f"coherence: echo rate {v['per']:.3f} over {limit:.3f} "
                    f"(author {mu:.3f}){where}{_fp(cal, 'echo')}. "
                    f"{len(quoted)} sentence(s) restate an earlier one:",
                    [f'"{s[:160]}"' for s in quoted[:3]])
    return Line("echo", _OK,
                f"coherence: echo rate {v['per']:.3f} under {limit:.3f} "
                f"(author {mu:.3f}){where}")


def _openers(text: str, cal: StructureCalibration) -> Line:
    if not otd.available():
        return Line("openers", _NA, f"openers: not measurable, {otd.unavailable_reason()}")
    if not cal.openers:
        return Line("openers", _NA, "openers: no calibration (built without spaCy)")
    n_sent = len([s for s in sentences(text) if s.strip()])
    t = cal.otd_threshold(n_sent)
    if t is None:
        return Line("openers", _NA,
                    f"openers: {n_sent} sentences, no matched calibration at that length")
    limit, units, n_used = t
    prof = otd.profile(_first_sentences(text, n_used))
    ref = cal.opener_reference()
    d = otd.distance_to(prof, ref)
    cls, gap = otd.top_class(prof, ref)
    where = (f"; most over-used opener {cls} at {prof[f'share_{cls}']:.0%} "
             f"against the author's {ref[f'share_{cls}']:.0%}") if gap > 0 else ""
    scope = _scope(n_used, n_sent, "sentences")
    if d > limit:
        return Line("openers", _FLAG,
                    f"openers: class distance {d:.3f} over {limit:.3f} "
                    f"({units} matched pieces{scope}){where}{_fp(cal, 'openers')}")
    return Line("openers", _OK,
                f"openers: class distance {d:.3f} under {limit:.3f} "
                f"({units} matched pieces{scope})")


def _paragraphs(text: str, cal: StructureCalibration) -> Line:
    n_para = len([p for p in prose_paragraphs(text) if p.strip()])
    if n_para < psd.MIN_PARAS:
        return Line("paragraphs", _NA,
                    f"paragraphs: {n_para} paragraph(s), too few to rate")
    band = cal.para_cv_band(n_para)
    if band is None:
        return Line("paragraphs", _NA,
                    f"paragraphs: no matched calibration at {n_para} paragraphs")
    lo, hi, mu, units, n_used = band
    scope = _scope(n_used, n_para, "paragraphs")
    cv = psd.profile(_first_paragraphs(text, n_used))["para_cv_words"]
    if cv < lo:
        return Line("paragraphs", _FLAG,
                    f"paragraphs: length variation {cv:.2f} under the author's band "
                    f"[{lo:.2f}, {hi:.2f}] (author {mu:.2f}, {units} matched pieces{scope}); "
                    f"paragraphs are all one size{_fp(cal, 'paragraphs')}")
    if cv > hi:
        return Line("paragraphs", _FLAG,
                    f"paragraphs: length variation {cv:.2f} over the author's band "
                    f"[{lo:.2f}, {hi:.2f}] (author {mu:.2f}, {units} matched pieces{scope}); "
                    f"architecture, not quality -- the reader preferred an "
                    f"over-varied draft in the study{_fp(cal, 'paragraphs')}")
    return Line("paragraphs", _OK,
                f"paragraphs: length variation {cv:.2f} inside [{lo:.2f}, {hi:.2f}] "
                f"(author {mu:.2f}{scope})")


def _cadence(text: str, cal: StructureCalibration) -> Line:
    lengths = sent_lengths(text)
    n_sent = len(lengths)
    if n_sent < cpd.MIN_SENTS:
        return Line("cadence", _NA, f"cadence: {n_sent} sentences, too few to rate")
    band = cal.burst_band(n_sent)
    if band is None:
        return Line("cadence", _NA,
                    f"cadence: no matched calibration at {n_sent} sentences")
    lo, hi, mu, units, n_used = band
    scope = _scope(n_used, n_sent, "sentences")
    a = cpd.burstiness_an(lengths[:n_used])
    if a < lo:
        return Line("cadence", _FLAG,
                    f"cadence: burstiness {a:+.2f} under the author's band "
                    f"[{lo:+.2f}, {hi:+.2f}] (author {mu:+.2f}, {units} matched pieces{scope}); "
                    f"sentence lengths are metronomic{_fp(cal, 'cadence')}")
    if a > hi:
        return Line("cadence", _FLAG,
                    f"cadence: burstiness {a:+.2f} over the author's band "
                    f"[{lo:+.2f}, {hi:+.2f}] (author {mu:+.2f}, {units} matched "
                    f"pieces{scope}){_fp(cal, 'cadence')}")
    return Line("cadence", _OK,
                f"cadence: burstiness {a:+.2f} inside [{lo:+.2f}, {hi:+.2f}] "
                f"(author {mu:+.2f}{scope})")


def _vocabulary(text: str, cal: StructureCalibration) -> Line:
    lx = lexdiv.measure(text)
    if not lx:
        return Line("vocabulary", _NA,
                    f"vocabulary: under {lexdiv.MIN_TOKENS} tokens, no window to measure")
    band = cal.hapax_band()
    if band is None:
        return Line("vocabulary", _NA, "vocabulary: no calibration (too few corpus pieces)")
    lo, hi, mu = band
    h = lx["hapax_w300"]
    if h < lo or h > hi:
        side = "under" if h < lo else "over"
        tail = ("; a narrowing vocabulary is usually looping seen from the other side"
                if h < lo else "")
        return Line("vocabulary", _FLAG,
                    f"vocabulary: hapax {h:.3f} {side} the author's band "
                    f"[{lo:.3f}, {hi:.3f}] (author {mu:.3f}){tail}{_fp(cal, 'vocabulary')}")
    return Line("vocabulary", _OK,
                f"vocabulary: hapax {h:.3f} inside [{lo:.3f}, {hi:.3f}] (author {mu:.3f})")


def _resolvable(f: str, cal: StructureCalibration, n_sent: int, n_words: int) -> bool:
    """Can this feature say anything about a draft this short?

    Two ways it cannot, both derived from the author's own rates rather than
    typed in. For a feature whose machine direction is UP, a single instance
    must not be able to trip the cut on its own: in a five-sentence draft one
    long-then-short beat computes to 25 per 100 pairs, which was z +5.6 against
    an author whose mean is 1.0, and that is a quantisation artifact, not a
    finding. For the one feature whose machine direction is DOWN -- rhetorical
    questions, which every base model under-produces -- the author's own rate
    has to predict at least one instance in a draft this long, or its absence
    says nothing.
    """
    c = cal.constructions[f]
    cut = c["mu"] + Z_FLAG * max(c["sd"], construction.SD_FLOOR)
    if construction.MACHINE_DIRECTION[f] > 0:
        # copula density is per 100 words; the rest per 100 sentences or pairs
        units = n_words / 100.0 if f == "copula_density" else max(n_sent - 1, 1) / 100.0
        return units > 0 and (1.0 / units) < cut
    return n_sent * c["mu"] / 100.0 >= 1.0


def _constructions(text: str, cal: StructureCalibration) -> Line:
    if not cal.constructions:
        return Line("constructions", _NA, "constructions: no calibration")
    z = cal.construction_z(text)
    n_sent = len([s for s in sentences(text) if s.strip()])
    n_words = len(text.split())
    rows = []
    hits = []
    thin = []
    for f in construction.FEATURES:
        if f not in z:
            continue
        if not _resolvable(f, cal, n_sent, n_words):
            # No z at all rather than a z beside a caveat: on a four-sentence
            # draft one long-then-short beat computes to +20, and printing that
            # number invites someone to act on it.
            thin.append(f)
            rows.append(f"{f} n/a")
            continue
        rows.append(f"{f} {z[f]:+.1f}")
        if z[f] * construction.MACHINE_DIRECTION[f] >= Z_FLAG:
            hits.append(f)
    detail = []
    for f in hits:
        c = cal.constructions[f]
        line = (f"  {construction.LABELS[f]}: z {z[f]:+.1f} "
                f"(author {c['mu']:.2f}, sd {max(c['sd'], construction.SD_FLOOR):.2f})")
        if f == "anaphora2":
            reps = construction.anaphora_pairs(text)
            if reps:
                line += "; repeated openers: " + ", ".join(f'"{r}"' for r in reps[:5])
        if f == "rhetorical_question":
            line += "; under-produced, which is the machine direction for this one"
        detail.append(line)
    note = (f" (n/a: the draft is too short for {len(thin)} of them to resolve)"
            if thin else "")
    status = _FLAG if hits else (_NA if len(thin) == len(rows) else _OK)
    return Line("constructions", status,
                "constructions (signed z vs the author): " + ", ".join(rows) + note
                + (_fp(cal, "constructions") if hits else ""),
                detail)


def analyse(text: str, cal: StructureCalibration) -> list[Line]:
    """The six lines, in reading order. Never a total."""
    return [
        _echo(text, cal),
        _openers(text, cal),
        _paragraphs(text, cal),
        _cadence(text, cal),
        _vocabulary(text, cal),
        _constructions(text, cal),
    ]


def render(text: str, cal: StructureCalibration | None, author: str = "the author") -> str:
    """The block as it appears in a scrub or gate report."""
    if cal is None:
        return ("## Structure\n\nNo structure calibration for this voice. "
                "Run: mimesis calibrate <voice>\n")
    out = [f"## Structure (reporting only; bands measured on {author}'s own corpus, "
           f"{cal.n_units} pieces)", ""]
    if cal.thin:
        out.append(f"THIN CALIBRATION: {cal.n_units} pieces. Bands from this few "
                   "pieces are noise, and every line below inherits it.")
        out.append("")
    for line in analyse(text, cal):
        mark = {"FLAG": "FLAG", "ok": "ok  ", "--": "--  "}[line.status]
        out.append(f"- [{mark}] {line.text}")
        out.extend(f"    {d}" for d in line.detail)
    out.append("")
    out.append("Separate lines by design: coherence, openers, paragraphs, cadence and "
               "vocabulary measure different things and averaging them hides whichever "
               "one is failing.")
    return "\n".join(out) + "\n"


# --- repair -------------------------------------------------------------------

def repair_notes(text: str, cal: StructureCalibration | None) -> list[str]:
    """Actionable instructions for a rewrite pass, machine direction only.

    The report is two-sided because it is a diagnosis: paragraphs above the
    author's band are as unlike them as paragraphs below. A repair prompt is not
    a diagnosis, and the two sides are not symmetric in what they cost. The
    study's reader preferred a cell that over-varied its paragraphs (CV 0.79
    against the source's 0.34) and rejected the cells that under-varied them, so
    telling a model to make its paragraphs more uniform would optimise toward
    the failure the measure exists to catch. Only the direction the evidence
    calls machine is passed on: under-varied paragraphs, metronomic sentences,
    narrowed vocabulary, restatement, opener monotony, over-produced
    constructions.

    Each note names the evidence rather than the number. A model can act on
    "eleven sentences in a row open on a pronoun"; it cannot act on 0.502.
    """
    if cal is None:
        return []
    out: list[str] = []
    lines = {ln.key: ln for ln in analyse(text, cal)}

    if lines["echo"].flagged:
        quoted = per.echo_sentences(per.worst_window(text)["text"])
        out.append(
            "COHERENCE: the draft restates itself. These sentences repeat the content of "
            "an earlier one in fresh wording:\n"
            + "\n".join(f'    "{s[:200]}"' for s in quoted[:4])
            + "\n  Cut them or replace them with something that moves the piece on. Do not "
              "paraphrase them again."
        )

    if lines["openers"].flagged and otd.available():
        prof = otd.profile(text)
        ref = cal.opener_reference()
        cls, _ = otd.top_class(prof, ref)
        out.append(
            f"OPENERS: {prof[f'share_{cls}']:.0%} of sentences open "
            f"{_CLASS_PLAIN.get(cls, cls)}, against the author's "
            f"{ref[f'share_{cls}']:.0%}. Vary where sentences begin. That means "
            "restructuring them, not swapping the first word for a synonym."
        )

    if lines["paragraphs"].flagged and "under the author's band" in lines["paragraphs"].text:
        band = cal.para_cv_band(len([p for p in prose_paragraphs(text) if p.strip()]))
        mu = band[2] if band else 0.0
        out.append(
            "PARAGRAPHS: every paragraph is about the same length. The author's vary a "
            f"lot more (coefficient of variation {mu:.2f}). Let some run long and some "
            "stop after a line."
        )

    if lines["cadence"].flagged and "under the author's band" in lines["cadence"].text:
        out.append(
            "CADENCE: the sentences are all about the same length. Break the regularity "
            "by changing where sentences end, not by padding or trimming words."
        )

    if lines["vocabulary"].flagged and "under the author's band" in lines["vocabulary"].text:
        out.append(
            "VOCABULARY: the draft keeps reusing a small set of words. The problem is the "
            "repetition, not the register: say the new thing rather than reaching for a "
            "rarer word for the old one."
        )

    if lines["constructions"].flagged:
        for d in lines["constructions"].detail:
            f = d.strip().split(":")[0]
            if "repeated" in f and "openers" in f:
                reps = construction.anaphora_pairs(text)
                out.append(
                    "REPEATED OPENERS: consecutive sentences begin with the same two words"
                    + (" (" + ", ".join(f'"{r}"' for r in reps[:5]) + ")" if reps else "")
                    + ". This was the strongest machine tell in the study at every model "
                      "size. Rebuild those sentences so they do not start alike."
                )
            elif "copula" in f:
                out.append(
                    "COPULAS: too much of the draft is is/was/were. The author's prose "
                    "moves where this one describes states. Give the sentences verbs."
                )
            elif "long-then-short" in f:
                out.append(
                    "RHYTHM: the long-sentence-then-short-sentence beat is over-used. It "
                    "is a real move, used here as a reflex."
                )
    return out


_CLASS_PLAIN = {
    "PRON": "on a pronoun - he, she, it, they, I",
    "NP": "on a noun phrase - The, A, or a name",
    "CONJ": "on And, But or So",
    "DISC": "on a discourse marker - However, Then, Now, Finally",
    "SUBORD": "on a subordinator - When, If, Although, Because",
    "PP": "on a preposition",
    "ADV": "on an adverb",
    "PART": "on a participle, an -ing or -ed word",
    "QUOTE": "on dialogue",
}
