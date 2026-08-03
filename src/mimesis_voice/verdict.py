"""One place where every check is ordered, weighted, and allowed to disagree.

Before this, five subsystems voted independently with no arbitration:
``fingerprint`` (is it like the author), ``scrub`` (does it carry AI tells),
``presence`` (is anyone in it), ``rhetoric`` (is the sentence shape a machine
reflex) and now ``quality`` (is it worth reading). Each rendered its own advice
into one flat list in call order. Nothing decided what to do when they conflicted,
and they do conflict:

* the burstiness floor pushed toward sentence-length variance while the cadence
  evidence showed the cheapest way to satisfy it produced a metronome
* driving antithesis to zero satisfied the scrubber and moved the text further
  from the author, whose own rate is nonzero
* a draft can be maximally in-voice and maximally boring at once, and nothing
  ranked those against each other

Flat lists also mislead by volume. A draft with one fabricated statistic and six
style notes reads, in an unordered list, like a style problem.

## The ordering, and why it is this way

1. **CORRECTNESS** -- fabricated or dropped numbers, citations, task leakage.
   Never a matter of taste. A rewrite that invents a statistic is unusable no
   matter how well it scores everywhere else.
2. **AUTHENTICITY** -- off-voice on the fingerprint, or missing the author
   entirely. This is what the tool exists for.
3. **QUALITY** -- repetitive, padded, thin, unanchored. Ranked below authenticity
   deliberately: the author's own weaker pieces are still the author, and a system
   that "improves" prose past the point of recognition has stopped doing its job.
   But ranked above tics, because a boring draft is a worse outcome than a draft
   with one cleft in it.
4. **TICS** -- rhetorical reflexes and AI-tell vocabulary. Real, and the most
   over-weighted category in practice, because they are the easiest to count.
5. **ADVISORY** -- everything measured but not decided, including every
   rate-based check where the author's own floor is zero and under-use therefore
   cannot be judged.

Nothing here gates by itself. ``blocking`` names the tier at which a caller
should refuse to ship; the default is CORRECTNESS only, because that is the one
tier with no legitimate exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    CORRECTNESS = 1
    AUTHENTICITY = 2
    QUALITY = 3
    TICS = 4
    ADVISORY = 5

    @property
    def label(self) -> str:
        return self.name.capitalize()


@dataclass
class Finding:
    tier: Tier
    source: str          # which subsystem raised it
    message: str

    def __str__(self) -> str:
        return f"[{self.tier.label.upper()}/{self.source}] {self.message}"


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)

    def at(self, tier: Tier) -> list[Finding]:
        return [f for f in self.findings if f.tier == tier]

    @property
    def worst(self) -> Tier | None:
        return min((f.tier for f in self.findings), default=None)

    def blocks(self, threshold: Tier = Tier.CORRECTNESS) -> bool:
        return any(f.tier <= threshold for f in self.findings)

    def render(self, author: str = "the author") -> str:
        if not self.findings:
            return "CLEAN across correctness, authenticity, quality and tics."
        out: list[str] = []
        for tier in Tier:
            group = self.at(tier)
            if not group:
                continue
            out.append(f"## {tier.label.upper()}")
            out.extend(f"- ({f.source}) {f.message}" for f in group)
        w = self.worst
        out.append("")
        out.append(
            f"Worst tier: {w.label.upper()}. "
            + ("Do not ship: a correctness failure is not a stylistic judgement."
               if w == Tier.CORRECTNESS else
               "Fix in the order above; the tiers are ranked, the lists inside are not.")
        )
        return "\n".join(out)


def build(scrub_report, quality_report=None, author: str = "the author") -> Verdict:
    """Collect every subsystem's findings into one ranked verdict.

    Takes the already-computed reports rather than recomputing, so this adds no
    measurement cost and cannot disagree with what the individual renderers say.
    """
    v = Verdict()
    r = scrub_report

    # 1. correctness
    if getattr(r, "fidelity_added_numbers", None):
        v.findings.append(Finding(
            Tier.CORRECTNESS, "fidelity",
            f"numbers not present in the source: {', '.join(r.fidelity_added_numbers)}. "
            f"Verify each against the source or remove it."))
    if getattr(r, "fidelity_dropped_citations", None):
        v.findings.append(Finding(
            Tier.CORRECTNESS, "fidelity",
            f"citations/acronyms dropped from the source: "
            f"{', '.join(r.fidelity_dropped_citations)}."))
    rh = getattr(r, "rhetoric", None)
    if rh and getattr(rh, "leakage", None):
        v.findings.append(Finding(
            Tier.CORRECTNESS, "rhetoric",
            "the draft narrates its own reasoning about the prompt. Discard and "
            "regenerate; this is not a style issue."))

    # 2. authenticity
    if getattr(r, "fit_off", False):
        v.findings.append(Finding(
            Tier.AUTHENTICITY, "fingerprint",
            f"off-voice: distance {r.fp_distance:.2f} over threshold "
            f"{r.fp_threshold:.2f} ({author} baseline {r.fp_baseline:.2f}). "
            f"Furthest: {', '.join(f'{n} {z:+.1f}sd' for n, z in r.fp_worst)}."))
    if getattr(r, "presence_missing", False):
        v.findings.append(Finding(
            Tier.AUTHENTICITY, "presence",
            f"nobody is in this draft: no first-person reading and nothing running "
            f"against expectation. It can match {author}'s distribution and still "
            f"have no author in it."))

    # 3. quality
    if quality_report is not None:
        from . import quality as quality_mod
        for line in quality_mod.render_lines(quality_report):
            v.findings.append(Finding(
                Tier.QUALITY, "quality", line.split("] ", 1)[-1]))

    # 4. tics
    if getattr(r, "banned_words", None):
        v.findings.append(Finding(
            Tier.TICS, "scrub",
            f"AI-tell vocabulary outside {author}'s lexicon: "
            f"{', '.join(r.banned_words)}."))
    if getattr(r, "banned_phrases", None):
        v.findings.append(Finding(
            Tier.TICS, "scrub", f"AI cliche phrases: {'; '.join(r.banned_phrases)}."))
    if getattr(r, "emdash_count", 0):
        v.findings.append(Finding(
            Tier.TICS, "scrub", f"{r.emdash_count} em-dash(es); target is zero."))
    if rh:
        if getattr(rh, "closing_flourish", None):
            v.findings.append(Finding(
                Tier.TICS, "rhetoric",
                f"closing flourish \"{rh.closing_flourish[0]}...\": a move this "
                f"author never uses and generated text reaches for by default."))
        if getattr(rh, "cleft_over", False):
            v.findings.append(Finding(
                Tier.TICS, "rhetoric",
                f"wh-cleft rate {rh.cleft_rate:.1f}/1kw over {author}'s "
                f"{rh.cleft_p95:.1f}."))
        if getattr(rh, "antithesis_over", False):
            v.findings.append(Finding(
                Tier.TICS, "rhetoric",
                f"antithesis rate {rh.antithesis_rate:.1f}/1kw over {author}'s "
                f"{rh.antithesis_p95:.1f}."))
        for s in getattr(rh, "triads", [])[:2]:
            v.findings.append(Finding(
                Tier.TICS, "rhetoric", f"parallel-triad escalation: \"{s[:70]}\"."))

    # 5. advisory
    if getattr(r, "fit_drifting", False):
        v.findings.append(Finding(
            Tier.ADVISORY, "fingerprint",
            f"drifting: {r.fp_distance:.2f} vs baseline {r.fp_baseline:.2f}, under "
            f"the hard threshold. Worth a look if the draft reads flat."))
    if rh:
        for name, under, rate, floor in (
            ("wh-cleft", getattr(rh, "cleft_under", False), rh.cleft_rate, rh.cleft_p25),
            ("antithesis", getattr(rh, "antithesis_under", False),
             rh.antithesis_rate, rh.antithesis_p25),
        ):
            if under:
                v.findings.append(Finding(
                    Tier.ADVISORY, "rhetoric",
                    f"{name} rate {rate:.1f}/1kw is BELOW {author}'s floor "
                    f"{floor:.1f}. Scrubbing a construction past the author's own "
                    f"rate moves the draft away from them, not toward them."))
    if getattr(r, "burstiness", 0) and r.burstiness < getattr(r, "burstiness_floor", 0):
        v.findings.append(Finding(
            Tier.ADVISORY, "scrub",
            f"low burstiness {r.burstiness:.2f} < floor {r.burstiness_floor:.2f}. "
            f"Vary at passage scale, never line by line."))
    return v
