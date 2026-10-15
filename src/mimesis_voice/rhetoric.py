"""Rhetorical-tic detector: AI reflexes that survive every other check in this package.

Every existing check works at the lexical or statistical level: banned words, banned
phrases, sentence-length variance, hedge density, a 13-feature stylometric fingerprint.
None of them can see a pattern built entirely out of the author's own permitted
vocabulary. Found 2026-07-28 reading real generations against a live task: a draft can
score CLEAN on every check above and still read, to a human, as obviously machine-written,
because the tell is in the *shape of the sentence*, not its words.

Three shapes, found in the wild against this author's own corpus and generated text on
the same day, so these are not hypothetical:

1. **Anaphoric/parallel-triad escalation.** Three or more clauses in one sentence sharing
   a first word, or matching an ordinal/negation escalation pattern. This is one of the
   most recognizable LLM rhetorical reflexes there is, precisely because it produces
   instant gravitas from repetition rather than from information. Found live: "First the
   language goes, then the reasons, then the wanting" (flagged as "terrible" by the author
   on read), and independently, in a Qwen3-4B LoRA generation never touched by any prompt
   engineering: "a man in a play, or a woman in a ritual, or a master in a ceremony."

2. **Self-referential meta-commentary aside.** A relative clause that steps outside the
   scene to editorialise on the narrator's own reaction to what was just said: ", which is
   the part I like." A person narrating a memory does not usually pause to rate it in the
   middle of the sentence; an LLM asked to sound reflective often does exactly this, as a
   substitute for the reflection actually landing in what came before it.

2b. **Self-rating clause.** The expository sibling of (2). A finding is stated, then a clause
   rates how much the finding matters: "The raters did not keep the two constructs apart, and
   this is the strongest objection available to a reader of our released data." The first clause
   reports; the second tells the reader how to weigh it. Deleting the second costs no
   information, which is the test for whether a clause is doing this rather than adding a fact.
   Added 2026-08-13 after the author flagged it unprompted in a paper draft ("super AI-y"),
   where eleven instances had survived a full AI-tell scrub, a fingerprint check and two prose
   passes -- none of its words are on any banned list. Rule (2) misses it because that pattern
   requires a first-person reaction verb and this construction has none. Measured base rate on
   1.16M words of published papers: 0.032 per 1k. Calibration set and the informative-clause
   negatives that must NOT fire are in `tests/test_rhetoric.py`.

3. **Scene-then-flat-absence-punch.** A sentence or two of setup followed immediately by
   a short, flat declarative naming an ending or an absence: "That man is gone." This is a
   near relative of the AI-tell structure this project already bans in `CORE_VOICE.md`
   ("Not X. Y." two-word contrast fragments) but operates at a longer range and was not
   covered by that rule. Found live: "He had a list of them; he recited it to himself on
   the walk in, and for a while the reciting was enough. That man is gone." (flagged as
   "terrible" by the author on read).

A fourth class sits apart from the other three because it is not a rhetorical *tic* at
all -- it is the model failing to stay in character:

4. **Task/instruction leakage.** The model narrates its own reasoning about the prompt
   instead of executing it: "Okay, here's my task: write a passage in the style of the
   examples provided... Let me try to think through this." Found live in a real generation
   from this project's own pipeline (Qwen3-4B, RAG + style-spec prompt, 2026-07-28), where
   it silently passed every other check because it was long enough and did not copy
   verbatim from any reference text. This is the single most reliable tell in this file:
   nobody writing fiction produces these strings by accident, so the false-positive risk
   against real prose is close to zero, and it is checked first.

None of these are per-author calibrated. Unlike the burstiness floor or hedge ceiling,
which are properties of a specific corpus, these are properties of a specific writing
model's habits (or, for leakage, of the model breaking character), and apply the same way
regardless of whose voice is being targeted. Measured false-positive rate against the
reference creative corpus (111 works, 2026-07-28) is recorded in `tests/test_rhetoric.py`;
re-run that check before pointing this at a different author's corpus, since a very
short-clause writing style could in principle trip the triad check more often.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")
_CLAUSE_RE = re.compile(r"[,;]")

# Floor for the calibrated clause-chain length (see section 14 at the foot of this
# file). Declared here because RhetoricReport and RhetoricCalibration take it as a
# dataclass default, which is evaluated at class-definition time.
CHAIN_MIN_CLAUSES = 4
# Ceiling on the same value. An author whose corpus pushes the calibrated length
# past six has either a formatting problem or a sentence nobody can read; either
# way the check should not be allowed to calibrate itself into silence.
CHAIN_MAX_CLAUSES = 6
# Floor under the calibrated chain RATE, in over-length sentences per 100
# sentences, applied the way calibrate() already floors the cleft and antithesis
# rates: an author whose corpus happens to contain almost none still tolerates one
# incidental chain in a short passage. 4.0 is one chain per 25 sentences, which is
# 20-40x this author's own sentence-level rate for a chain of that length (0.176%
# creative, 0.118% research, 0.091% personal). Under it, a draft is doing what he
# does; over it, the draft has a habit he does not have.
CHAIN_RATE_FLOOR = 4.0


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return out


# --- 1. anaphoric / parallel-triad escalation ---------------------------------

# TWO OVERFIRING VERSIONS BEFORE THIS ONE, MEASURED 2026-07-28. v1 flagged any
# clause-initial word repeated 3+ times: fired on 56/111 (50.5%) of this author's real
# writing by catching genuine anaphoric intensification he uses on purpose --
# "the last few days, the last few hours, the last five seconds" repeats an article
# while escalating through increasingly specific CONTENT, a literary device rather than
# the AI tic. v2 restricted to a discourse-connective set that still included
# not/no/never/nor: fired on 26/111 (23.4%), because negation-parallelism ("No martyr am
# I, no Christ rose... no, I live as man") is itself a device this author's existentialist
# register uses constantly -- consistent with the corpus's own recorded themes
# (theology-without-a-god, nihilism).
#
# What survives is pure ORDINAL/TEMPORAL SEQUENCING -- first/second/third/then/next/
# finally, plus "or" for alternation-triads -- which is not a device found anywhere in
# 111 real works (0/111, see tests/test_rhetoric.py) but still catches both live examples
# that motivated this file: "First the language goes, then the reasons, then the
# wanting" (then x2) and "a man in a play, or a woman in a ritual, or a master in a
# ceremony" (or x2). Negation-triads are real for THIS author and are deliberately not
# flagged; re-measure before reusing this list for a different author's corpus.
_ESCALATION_OPENERS = ("first", "second", "third", "then", "next", "finally", "or")
_ESCALATION_SEQ = re.compile(
    r"^\s*first\b.*?[,;]\s*(second|then)\b.*?[,;]\s*(third|then|finally)\b",
    re.IGNORECASE,
)


def _clause_first_word(clause: str) -> str:
    m = _WORD_RE.search(clause)
    return m.group(0).lower() if m else ""


def find_triads(text: str) -> list[str]:
    """Sentences with 3+ clauses opening on a repeated ordinal/temporal connective
    (first/then/next/or/...), or matching an explicit 'first...then/second...third/
    then/finally' sequence. Deliberately does NOT flag repeated content words or
    negation ('not', 'no') -- see the note above the word list for why."""
    hits = []
    for sent in _split_sentences(text):
        if _ESCALATION_SEQ.search(sent):
            hits.append(sent)
            continue
        clauses = [c for c in _CLAUSE_RE.split(sent) if c.strip()]
        if len(clauses) < 3:
            continue
        firsts = [_clause_first_word(c) for c in clauses]
        counts: dict[str, int] = {}
        for w in firsts:
            if w in _ESCALATION_OPENERS:
                counts[w] = counts.get(w, 0) + 1
        if any(c >= 2 for c in counts.values()):
            hits.append(sent)
    return hits


# --- 2. self-referential meta-commentary aside --------------------------------

_META_ASIDE_RE = re.compile(
    r",\s*which (?:is|was)\s+(?:not\s+)?(?:the|a|exactly|precisely)?\s*[\w\s]{0,40}?"
    r"\b(?:i|we)\b\s+(?:like|find|found|love|loved|hate|hated|prefer|preferred|think|"
    r"thought|feel|felt|realize|realized|notice|noticed|want|wanted|enjoy|enjoyed)\b",
    re.IGNORECASE,
)


def find_meta_asides(text: str) -> list[str]:
    """A relative clause that steps outside the scene to rate the narrator's own
    reaction to what was just said, e.g. ', which is the part I like'."""
    return [m.group(0).strip(", ") for m in _META_ASIDE_RE.finditer(text)]


# --- 2b. self-rating clause (the expository sibling of the meta aside) ---------

# find_meta_asides above catches the narrative form, which needs an explicit first-person
# reaction verb: ", which is the part I like". Expository prose does the same move without one.
# It states a finding, then appends a clause rating how much the finding matters:
#
#   "The raters did not keep the two constructs apart, and this is the strongest objection
#    available to a reader of our released data."
#   "Its practical consequence is unchanged and is the point worth keeping: ..."
#   "One category is worth separating out because three measurements agree about it."
#
# The first clause reports; the second tells the reader how to weigh it. Deleting the second
# costs no information, which is the test for whether a clause is doing this. Flagged because
# it is dense in model output and near-absent in the authors' own corpora, and because it
# survives every vocabulary-level scrub: none of its words are on any banned list.
#
# Deliberately NOT matched: clauses that add a fact rather than a rating
# ("which is why the CI spans [0.42, 1.00]", "which is the figure carrying the parity claim").
# The discriminator is an evaluative head, not the back-reference itself.

_RATING_HEAD = (
    r"(?:strongest|weakest|biggest|largest|single most|most \w+|best|worst|"
    r"point worth \w+|thing worth \w+|worth (?:noting|stating|separating|keeping|emphasi[sz]ing)|"
    r"key (?:point|finding|result|insight)|central (?:point|claim|finding)|"
    r"crucial|essential|the whole point|precisely (?:why|what)|exactly (?:why|what)|"
    r"real (?:point|finding|question)|important (?:point|thing)|"
    r"the reason (?:to|why|this)|most (?:encouraging|troubling|striking|important|valuable)|"
    # "and this is the more consequential point" -- flagged 2026-08-28, not covered
    # by any head above.
    r"(?:more|less) \w+ point|consequential(?: point)?)"
)

# The first-person register of the same move, flagged 2026-08-28 as a "SUPER BAD
# Claude AI tell": the narrator announcing how the reader should take the next
# sentence. _SELF_RATING_RE above is expository and needs an evaluative head;
# these need none, because the announcement IS the tell.
_FIRST_PERSON_META_RE = re.compile(
    r"\b(?:I want to be clear|I should say|let me be clear|to be clear|"
    r"I'll be honest|I will be honest|I should note|I want to note|"
    r"I am not being clever|I mean this literally)\b",
    re.IGNORECASE,
)

_SELF_RATING_RE = re.compile(
    # ", and this is the strongest ..."  /  ", which is precisely why ..."
    rf"(?:,\s*(?:and\s+)?(?:this|that|it|which)\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+)?{_RATING_HEAD}"
    # "... and is the point worth keeping"
    rf"|\band\s+(?:is|was)\s+the\s+{_RATING_HEAD}"
    # sentence-initial "This is the single most ..." / "That is the whole point"
    rf"|(?:^|(?<=[.!?])\s+)(?:This|That|It)\s+(?:is|was)\s+(?:the\s+|a\s+|an\s+)?{_RATING_HEAD}"
    # "X is worth separating out because ..."
    rf"|\bis\s+worth\s+(?:separating|noting|stating|keeping|emphasi[sz]ing)\b"
    # "... are the most encouraging thing here" / "the single most valuable extension of this work"
    rf"|\b(?:is|are|was|were)\s+the\s+(?:single\s+)?most\s+\w+\s+\w+\s+"
    rf"(?:here|yet|available|of this (?:work|paper|study)|in this (?:work|paper|study))\b"
    # "This is the effect the paper is about" -- rating by pointing at the paper's own subject
    rf"|\b(?:this|that|it)\s+(?:is|was)\s+(?:the|what)\s+[\w\s]{{0,25}}?"
    rf"(?:this|the)\s+(?:paper|work|study|section)\s+(?:is|was)\s+about\b)",
    re.IGNORECASE,
)


def find_self_ratings(text: str) -> list[str]:
    """A clause that rates the importance of the claim it is attached to.

    Returns the offending span, not the whole sentence, so a caller can see exactly what to cut.
    """
    return ([m.group(0).strip(", ") for m in _SELF_RATING_RE.finditer(text)]
            + [m.group(0).strip() for m in _FIRST_PERSON_META_RE.finditer(text)])


# --- 3. scene-then-flat-absence-punch ------------------------------------------

_ABSENCE_PUNCH_RE = re.compile(
    r"^(?:that|this|it|he|she|they)\b[\w\s]{0,25}?\b(?:is|was|are|were)\s+"
    r"(?:gone|over|done|finished|no more|nothing|silent|still|empty)\b",
    re.IGNORECASE,
)
_SETUP_MIN_WORDS = 15
_PUNCH_MAX_WORDS = 8


def find_absence_punches(text: str) -> list[tuple[str, str]]:
    """A long setup sentence immediately followed by a short flat sentence naming an
    ending or absence, e.g. '...for a while the reciting was enough. That man is gone.'
    Returns (setup, punch) pairs. Generalises the existing 'Not X. Y.' ban in
    CORE_VOICE.md to a longer range; that rule alone does not catch this construction."""
    sents = _split_sentences(text)
    hits = []
    for i in range(1, len(sents)):
        punch = sents[i]
        setup = sents[i - 1]
        if (len(_WORD_RE.findall(punch)) <= _PUNCH_MAX_WORDS
                and len(_WORD_RE.findall(setup)) >= _SETUP_MIN_WORDS
                and _ABSENCE_PUNCH_RE.match(punch)):
            hits.append((setup, punch))
    return hits


# --- 4. task/instruction leakage (not a tic -- a character break) -------------

_LEAKAGE_RE = re.compile(
    r"\b(?:okay|alright|sure)[,.]?\s+here'?s (?:my|the|a)\s+(?:task|answer|response|draft)\b"
    r"|\blet me (?:try to |just |)?think (?:this |it )?through\b"
    r"|\bin example \d\b"
    r"|\bthe (?:topic|task|prompt|style)\s+(?:is|was)\s*[\"']"
    r"|\bwrite a passage in the style of\b"
    r"|^(?:okay|alright)\s*,\s*(?:so|here|let)",
    re.IGNORECASE | re.MULTILINE,
)


def find_leakage(text: str) -> list[str]:
    """The model narrating its own reasoning about the prompt instead of executing
    it. Near-zero false-positive risk against real prose; check this first."""
    return [m.group(0) for m in _LEAKAGE_RE.finditer(text)]


# --- report ---------------------------------------------------------------------

@dataclass
class RhetoricReport:
    leakage: list[str] = field(default_factory=list)
    triads: list[str] = field(default_factory=list)
    meta_asides: list[str] = field(default_factory=list)
    self_ratings: list[str] = field(default_factory=list)
    absence_punches: list[tuple[str, str]] = field(default_factory=list)
    clefts: list[str] = field(default_factory=list)
    antithesis: list[str] = field(default_factory=list)
    closing_flourish: list[str] = field(default_factory=list)
    reversals: list[str] = field(default_factory=list)
    self_explanations: list[str] = field(default_factory=list)
    way_similes: list[str] = field(default_factory=list)
    declarative_ratings: list[str] = field(default_factory=list)
    nominalization_pickups: list[str] = field(default_factory=list)
    it_is_openers: list[str] = field(default_factory=list)
    it_is_rate: float = 0.0
    it_is_p95: float = 0.0
    # Comma-spliced clause chains (section 14). chain_len is the calibrated
    # length in force for this report, carried so render() can name it; chain_max
    # is the longest chain the draft actually contains, which is the statistic
    # commas_per_sent averages away.
    comma_chains: list[str] = field(default_factory=list)
    chain_rate: float = 0.0
    chain_p95: float = 0.0
    chain_len: int = CHAIN_MIN_CLAUSES
    chain_max: int = 0
    cleft_rate: float = 0.0
    antithesis_rate: float = 0.0
    cleft_p95: float = 0.0
    antithesis_p95: float = 0.0
    cleft_p25: float = 0.0
    antithesis_p25: float = 0.0

    @property
    def cleft_over(self) -> bool:
        return self.cleft_p95 > 0 and self.cleft_rate > self.cleft_p95

    @property
    def antithesis_over(self) -> bool:
        return self.antithesis_p95 > 0 and self.antithesis_rate > self.antithesis_p95

    # UNDER-use is a real defect, not a clean result. See RhetoricCalibration.
    # Only meaningful when the author's own floor is nonzero: an author who never
    # uses a construction cannot under-use it.
    @property
    def cleft_under(self) -> bool:
        return self.cleft_p25 > 0.05 and self.cleft_rate < self.cleft_p25

    @property
    def antithesis_under(self) -> bool:
        return self.antithesis_p25 > 0.05 and self.antithesis_rate < self.antithesis_p25

    @property
    def it_is_over(self) -> bool:
        """p95 == 0 means the author does not open sentences this way at all, so
        any use is over his rate. Guarding on p95 > 0 the way the cleft and
        antithesis checks do would disable this exactly where it is most wanted:
        the research profile measures median 0.00 and p95 0.00, and that is the
        voice he objected to it in."""
        return self.it_is_rate > self.it_is_p95

    @property
    def chain_over(self) -> bool:
        """Comma-spliced clause chains, over this author's own rate.

        Two routes, because the complaint has two shapes. DENSITY is the main one
        -- "the OVER USE of commas" -- and it is a rate against his floored 95th
        percentile, so one chain inside a long piece is not overuse while one
        chain inside a twelve-sentence draft is. LENGTH short-circuits it: a chain
        two clauses past the calibrated length is the defect by itself, and making
        it wait for a second instance would let the worst single sentence through.
        Measured against his two blind drafts that name comma overuse in the
        rationale, both routes are needed -- one draft trips each.
        """
        return (self.chain_max >= self.chain_len + 2
                or self.chain_rate > self.chain_p95)

    @property
    def hard_flags(self) -> list[str]:
        """Leakage is unambiguous (the model breaking character) and gates like an
        em-dash. The three tics are real but each has a nonzero measured false-positive
        rate against real prose, so they are advisory -- see render()."""
        flags = ["task-leakage"] if self.leakage else []
        # Presence-based and HARD, unlike the rated tics. The author has never
        # asked for this one to be tolerated at any rate.
        if self.reversals:
            flags.append("unheeded-reversal")
        # Hard on presence at his measured rate of 0.14-0.41 per 1000 words. The
        # pickup form is advisory: loosest definition, highest corpus rate.
        if self.self_explanations:
            flags.append("self-explanation")
        # Both measured at or near zero in his own corpora, so there is no
        # rate of his to stay under and presence is the right gate.
        if self.way_similes:
            flags.append("way-simile")
        if self.declarative_ratings:
            flags.append("declarative-rating")
        return flags

    @property
    def is_clean(self) -> bool:
        return not (self.leakage or self.reversals or self.self_explanations
                    or self.way_similes or self.declarative_ratings
                    or self.nominalization_pickups or self.it_is_over
                    or self.triads or self.meta_asides
                    or self.self_ratings
                    or self.absence_punches or self.closing_flourish
                    or self.cleft_over or self.antithesis_over
                    or self.cleft_under or self.antithesis_under
                    or self.chain_over)


def analyze(text: str, cal: "RhetoricCalibration | None" = None) -> RhetoricReport:
    cal = cal or RhetoricCalibration()
    # A calibration written before section 14 existed carries chain_len 0. Fall
    # back to the floor rather than to zero, which would make every sentence a
    # chain; the same treatment the opener stats get in scrub.ScrubCalibration.
    chain_len = min(max(int(cal.chain_len or 0), CHAIN_MIN_CLAUSES), CHAIN_MAX_CLAUSES)
    return RhetoricReport(
        cleft_rate=_rate(text, find_clefts),
        antithesis_rate=_rate(text, find_antithesis),
        cleft_p95=cal.cleft_p95,
        antithesis_p95=cal.antithesis_p95,
        cleft_p25=cal.cleft_p25,
        antithesis_p25=cal.antithesis_p25,
        clefts=find_clefts(text),
        antithesis=find_antithesis(text),
        closing_flourish=find_closing_flourish(text),
        reversals=find_reversals(text),
        self_explanations=find_self_explanations(text),
        way_similes=find_way_similes(text),
        declarative_ratings=find_declarative_ratings(text),
        nominalization_pickups=find_nominalization_pickups(text),
        it_is_openers=find_it_is_openers(text),
        it_is_rate=it_is_opener_rate(text),
        it_is_p95=cal.it_is_p95,
        comma_chains=find_comma_chains(text, chain_len),
        chain_rate=comma_chain_rate(text, chain_len),
        # calibrate() already applies this floor, so for a calibrated profile the
        # max() is a no-op. It is here for the blank and pre-section-14
        # calibrations, where a stored 0.0 would otherwise flag a single chain in
        # any draft.
        chain_p95=max(cal.chain_p95, CHAIN_RATE_FLOOR),
        chain_len=chain_len,
        chain_max=longest_comma_chain(text),
        leakage=find_leakage(text),
        triads=find_triads(text),
        meta_asides=find_meta_asides(text),
        self_ratings=find_self_ratings(text),
        absence_punches=find_absence_punches(text),
    )


def _render_new(rep: "RhetoricReport") -> list[str]:
    """Lines for the rate-based checks. Framed as OVERUSE relative to the author,
    never as presence: he writes all three of these, just less often."""
    out: list[str] = []
    if rep.closing_flourish:
        out.append(
            f"- [MEDIUM] Closing flourish: ends on \"{rep.closing_flourish[0]}...\". "
            f"Measured across 157 of this author's real pieces, this closing move "
            f"appears 0 times, while it appeared in 2 of 3 generated abstracts from "
            f"three different voice profiles. It is the base model's ending, not his. "
            f"Cut the final intensifier clause or end on the concrete noun."
        )
    if rep.cleft_over:
        out.append(
            f"- [MEDIUM] Cleft overuse: {rep.cleft_rate:.1f} per 1000 words vs this "
            f"author's {rep.cleft_p95:.1f} (95th pct). e.g. "
            f"\"{rep.clefts[0][:60]}\". \"What X establishes is that Y\" defers the "
            f"claim behind a placeholder. Say Y."
        )
    if rep.antithesis_over:
        out.append(
            f"- [MEDIUM] Antithesis overuse: {rep.antithesis_rate:.1f} per 1000 words "
            f"vs this author's {rep.antithesis_p95:.1f} (95th pct). e.g. "
            f"\"{rep.antithesis[0][:60]}\". \"Not X but Y\" and \"rather than\" "
            f"manufacture precision from a contrast nobody raised. State the thing."
        )
    if rep.chain_over:
        out.append(
            f"- [MEDIUM] Comma-spliced clause chains: {rep.chain_rate:.1f} sentences "
            f"per 100 run {rep.chain_len}+ chained clauses (longest {rep.chain_max}), "
            f"against the {rep.chain_p95:.1f} allowed for this author. e.g. "
            f"\"{rep.comma_chains[0][:80]}\". His long sentences accumulate phrases; "
            f"this one accumulates predications. Break it, or cut the clauses that "
            f"only restate the one before."
        )
    elif rep.comma_chains:
        # Named under his rate rather than silently passed, on the same reasoning
        # as the contrastive-connective line below: "run on commas" is a standing
        # complaint, and an under-threshold instance was invisible until now. One
        # line, and a cheap edit becomes available.
        out.append(
            f"- [LOW] Clause chains, under your rate so not gated: "
            f"{'; '.join(repr(s[:70]) for s in rep.comma_chains[:2])}. Each is "
            f"{rep.chain_len}+ finite clauses strung on commas inside one sentence."
        )
    return out


def render_lines(rep: RhetoricReport) -> list[str]:
    out = _render_new(rep)
    for span in rep.way_similes:
        out.append(
            f"- [HIGH] Borrowed simile: \"{span}\". \"the way a X does\" is the "
            f"generating model's default lyric comparison, measured at up to 157x this "
            f"author's rate. Cut it or replace it with something seen rather than likened."
        )
    for span in rep.declarative_ratings:
        out.append(
            f"- [HIGH] Declarative rating: \"{span}\". A sentence that grades the "
            f"previous one by pointing at it. Delete it and let the previous sentence "
            f"carry its own weight."
        )
    for span in rep.self_explanations:
        out.append(
            f"- [HIGH] Self-explanation: \"{span}\". The clause narrates what the "
            f"sentence just did instead of letting it land. His most-repeated complaint, "
            f"named in every draft of the 2026-08-28 blind set. Delete the clause; the "
            f"sentence before it already said this."
        )
    for sent in rep.nominalization_pickups:
        out.append(
            f"- [MEDIUM] Nominalization pickup: \"{sent}\". A verb from the first half "
            f"returns as an abstract subject in the second. Keep the verb, cut the noun."
        )
    if rep.it_is_over:
        out.append(
            f"- [MEDIUM] \"It is/was\" sentence openers: {rep.it_is_rate:.1f} per 100 "
            f"sentences vs this author's {rep.it_is_p95:.1f} (95th pct). A placeholder "
            f"subject where a real one was available."
        )
    for pair in rep.reversals:
        out.append(
            f"- [HIGH] Unheeded reversal: \"{pair[:150]}\". A flat assertion taken back "
            f"and restated in the next breath. This is the author's single most-repeated "
            f"complaint (\"say the thing once\"). Delete one half: keep the claim, drop the "
            f"setup, or keep the setup and let it stand."
        )
    # Explicit contrastive connectives, listed even when the rate is under the
    # author's p95. He uses them himself, so a hard gate would fire on his own
    # prose and the rate stays the right gate -- but "the rather thans are also
    # annoying" is a standing complaint, and under-threshold instances were
    # invisible: the report said nothing at all until the rate was exceeded.
    # Naming them costs one line and makes a cheap edit available.
    if not rep.antithesis_over:
        explicit = [x for x in rep.antithesis
                    if re.search(r"\b(?:rather\s+than|instead\s+of)\b", x, re.IGNORECASE)]
        if explicit:
            out.append(
                f"- [LOW] Contrastive connectives, under your rate so not gated: "
                f"{'; '.join(repr(x) for x in explicit[:4])}. Each states a thing by "
                f"denying its neighbour. Usually cuttable without loss."
            )
    if rep.leakage:
        out.append(
            f"- [HIGH] Task/instruction leakage: {'; '.join(repr(x) for x in rep.leakage)}. "
            f"The draft is narrating its own reasoning about the prompt instead of writing "
            f"the piece. This is not a style issue: discard and regenerate."
        )
    for sent in rep.triads:
        out.append(
            f"- [MEDIUM] Anaphoric/parallel-triad escalation: \"{sent}\". Three "
            f"clauses building on repetition rather than information is a strong LLM "
            f"rhetorical reflex; break the parallelism or cut to two."
        )
    for aside in rep.meta_asides:
        out.append(
            f"- [MEDIUM] Self-referential meta-commentary aside: \"{aside}\". The "
            f"narrator is rating the sentence instead of the reflection landing in the "
            f"prose itself. Cut the aside or fold the reaction into action."
        )
    for span in rep.self_ratings:
        out.append(
            f"- [MEDIUM] Self-rating clause: \"{span}\". The claim is stated and then "
            f"rated, telling the reader how much to care instead of letting the claim "
            f"land. Delete the rating; if nothing is lost, it was doing no work."
        )
    for setup, punch in rep.absence_punches:
        out.append(
            f"- [MEDIUM] Scene-then-flat-absence-punch: \"...{setup[-60:]}\" then"
            f"\"{punch}\". A relative of the banned 'Not X. Y.' contrast fragment at "
            f"longer range; earn the ending or cut the punch."
        )
    return out


# --- 5. cleft constructions ----------------------------------------------------
#
# "What this study establishes is that...", "what this audit finds is that...".
# The wh-cleft front-loads a placeholder and defers the actual content, which buys
# a sentence the cadence of a thesis statement without requiring one. It is heavily
# represented in assistant-model output.
#
# Measured across six generated academic sections (~1,800 words): 5 instances. The
# author's own body text for the same paper contained the pattern at a much lower
# rate, so the generations were AMPLIFYING a construction he uses sparingly rather
# than importing a foreign one. That is why this is reported as a RATE against the
# author's own calibrated rate, not as a binary presence flag: the same construction
# is fine once and a signature five times.
# MEASURED AND NARROWED, 2026-07-30. A first version merged wh-clefts and
# it-clefts into one pattern and fired 113 times across this author's 313,820
# words (360 per million, against a published human academic baseline of ~145 per
# million). 105 of those 113 were the it-cleft branch, and sampling them showed
# two distinct problems:
#   - genuine it-clefts that are this author's own literary device, used on
#     purpose: "It was upon her that he...", "It was by her gait that I..."
#   - not clefts at all, but extraposition with a reporting verb: "it was found
#     that", "it was affirmed that", "it was likely a corrupted file that..."
# Merging them meant his deliberate style inflated the threshold and hid the
# construction actually complained about.
#
# So they are separate. The WH-CLEFT is the assistant tell -- "What this study
# establishes is that X" defers the claim behind a placeholder -- and this author
# uses it 8 times in 313,820 words (26 per million). The IT-CLEFT is his, tracked
# but not flagged, with reporting verbs excluded so extraposition stops counting.
_WH_CLEFT_PATTERNS = [
    r"\bWhat\s+\w+(?:\s+\w+){0,6}?\s+(?:is|was|does|did|shows?|showed|"
    r"establishes?|finds?|found|means?|matters?|describes?|tells?)\s+(?:is\b|that\b)",
    r"\bthe\s+\w+(?:\s+\w+){0,3}?\s+that\s+matters?(?:\s+most)?\s+is\b",
    r"\bThat\s+is\s+what\s+\w+",
]
_WH_CLEFT_RE = re.compile("|".join(_WH_CLEFT_PATTERNS), re.IGNORECASE)

# Reporting / evaluative verbs after "it was|is" mark extraposition, not a cleft.
_IT_CLEFT_RE = re.compile(
    r"\bIt\s+(?:is|was)\s+(?!clear\b|worth\b|possible\b|true\b|not\b|only\s+\w+\s+that\b"
    r"|found\b|affirmed\b|said\b|shown\b|noted\b|argued\b|likely\b|believed\b"
    r"|assumed\b|reported\b|thought\b|obvious\b|apparent\b|unclear\b)"
    r"[^.?!,;]{3,40}?\s+that\s+\w+",
    re.IGNORECASE,
)


def find_wh_clefts(text: str) -> list[str]:
    """The assistant tell: a placeholder subject deferring the actual claim."""
    return [m.group(0).strip() for m in _WH_CLEFT_RE.finditer(text)]


def find_it_clefts(text: str) -> list[str]:
    """This author's own device. Tracked for reporting, not flagged as a tic."""
    return [m.group(0).strip() for m in _IT_CLEFT_RE.finditer(text)]


def find_clefts(text: str) -> list[str]:
    """Wh-clefts only. See the note above: it-clefts are this author's own."""
    return find_wh_clefts(text)


# --- 6. x-not-y antithesis -----------------------------------------------------
#
# "not a benchmark report but a contribution", "saturates instead of compounding",
# "individually plausible, collectively wrong". The construction states a thing by
# denying its neighbour, and it is one of the most recognisable assistant reflexes
# there is because it manufactures the feel of precision from a contrast that was
# never in question.
#
# Same amplification finding as the cleft: the author's own abstract for the paper
# under test ran two instances; the generated arms ran three to five apiece, because
# the body uses "rather than" and "instead of" freely and the arms read that
# frequency as a target. Rate-based, not presence-based, for the same reason.
_ANTITHESIS_PATTERNS = [
    r"\bnot\s+[^.,;:]{2,50}?\s+but\s+(?:rather\s+)?\w+",   # not X but Y
    r"\brather\s+than\b",
    r"\binstead\s+of\b",
    r",\s*not\s+\w+",                                       # X, not Y
    r"\bis\s+not\s+[^.?!]{2,60}?\.\s*(?:It|That|This)\s+is\b",  # is not X. It is Y.
]
_ANTITHESIS_RE = re.compile("|".join(_ANTITHESIS_PATTERNS), re.IGNORECASE)


def find_antithesis(text: str) -> list[str]:
    return [m.group(0).strip() for m in _ANTITHESIS_RE.finditer(text)]


# --- 7. closing flourish -------------------------------------------------------
#
# The single clearest piece of evidence that stylometry misses rhetorical shape.
# Three voice profiles, three different corpora, one closing move:
#   "...exactly where they look most convincing"
#   "...exactly where neither a clinician nor a person would think to look"
#   "...precisely at the scale that no single encounter is built to expose"
# Every lexical and statistical check passed. The base model's prior for "how to
# end an abstract" survived all three voice profiles intact, which is what a
# fingerprint built from word and sentence statistics cannot see.
#
# Deliberately narrow: only the last two sentences, only the intensifier+deixis
# shape. A broad "ends on a flourish" detector would fire on any good closing line.
_CLOSING_FLOURISH_RE = re.compile(
    r"\b(?:exactly|precisely|only|entirely)\s+"
    r"(?:where|when|what|how|the\s+\w+\s+(?:that|where)|at\s+the\s+\w+\s+(?:that|where))\b",
    re.IGNORECASE,
)


def find_closing_flourish(text: str) -> list[str]:
    """The intensifier-plus-deixis closing move, in the final two sentences only."""
    sents = _split_sentences(text)
    if not sents:
        return []
    return [m.group(0).strip()
            for s in sents[-2:]
            for m in _CLOSING_FLOURISH_RE.finditer(s)]


# --- rate calibration ----------------------------------------------------------


@dataclass
class RhetoricCalibration:
    """Per-1000-word rates for constructions the author demonstrably uses.

    Presence means nothing for these three: an author who writes "rather than"
    is not writing like a machine, and flagging him for it produces a check he
    learns to ignore. What separates his prose from generated prose is RATE. So
    the calibration stores his own 95th-percentile rate and the report flags only
    output that exceeds it, which is the same author-relative logic the intake
    screen uses for lexical markers.
    """

    cleft_p95: float = 0.0
    antithesis_p95: float = 0.0
    # Sentence openers per 100 sentences. Sharply profile-dependent:
    # creative p95 5.36, personal 4.55, research 0.00.
    it_is_p95: float = 0.0
    # Lower edges of the author's own range. A rate check with only a ceiling is
    # half a check: measured on this project, driving antithesis to 0.00 scored as
    # "clean" while being exactly as unlike the author as 19.80, because his own
    # rate is 2.88-3.62. One-sided checks are how a voice system produces prose
    # more average than the author it is imitating.
    cleft_p25: float = 0.0
    antithesis_p25: float = 0.0
    # Comma-spliced clause chains, section 14. Two numbers because the check is
    # two-stage: chain_len is HOW LONG a chain has to be before it counts for this
    # author (his own 95th-percentile longest chain, floored at CHAIN_MIN_CLAUSES),
    # and chain_p95 is HOW OFTEN he writes one at that length, per 100 sentences.
    # A single number could not do both: a length alone flags his deliberate
    # accumulations, and a rate alone has no length to count.
    chain_len: int = 0
    chain_p95: float = 0.0
    n_pieces: int = 0

    def to_dict(self) -> dict:
        return {"cleft_p95": self.cleft_p95, "antithesis_p95": self.antithesis_p95,
                "cleft_p25": self.cleft_p25, "antithesis_p25": self.antithesis_p25,
                "chain_len": self.chain_len, "chain_p95": self.chain_p95,
                "n_pieces": self.n_pieces}

    @classmethod
    def from_dict(cls, d: dict) -> "RhetoricCalibration":
        return cls(cleft_p95=d.get("cleft_p95", 0.0),
                   antithesis_p95=d.get("antithesis_p95", 0.0),
                   cleft_p25=d.get("cleft_p25", 0.0),
                   antithesis_p25=d.get("antithesis_p25", 0.0),
                   chain_len=d.get("chain_len", 0),
                   chain_p95=d.get("chain_p95", 0.0),
                   n_pieces=d.get("n_pieces", 0))


def _rate(text: str, finder) -> float:
    n = len(_WORD_RE.findall(text))
    return (1000.0 * len(finder(text)) / n) if n else 0.0


def calibrate(texts: list[str], min_words: int = 200) -> RhetoricCalibration:
    """Author's own cleft and antithesis rates, at the 95th percentile.

    A floor is applied so an author whose corpus happens to contain none still
    tolerates one incidental use in a long passage rather than flagging it.
    """
    usable = [t for t in texts if len(_WORD_RE.findall(t)) >= min_words]
    if not usable:
        return RhetoricCalibration(cleft_p95=2.0, antithesis_p95=6.0)

    def p95(vals: list[float], floor: float) -> float:
        vals = sorted(vals)
        idx = int(0.95 * (len(vals) - 1))
        return max(vals[idx], floor)

    def pct(vals: list[float], q: float) -> float:
        vals = sorted(vals)
        return vals[int(q * (len(vals) - 1))]

    cl = [_rate(t, find_clefts) for t in usable]
    an = [_rate(t, find_antithesis) for t in usable]
    # Clause-chain length first, because the rate has to be counted at it. The
    # p95 of the longest chain per unit lands at 2-3 on all three of this author's
    # profiles, so the floor is what ships for him; the percentile is what would
    # carry an author who chains on purpose. Ceiling stops a damaged corpus from
    # calibrating the check into silence.
    chain_len = min(
        max(int(pct([longest_comma_chain(t) for t in usable], 0.95)), CHAIN_MIN_CLAUSES),
        CHAIN_MAX_CLAUSES,
    )
    return RhetoricCalibration(
        cleft_p95=p95(cl, 1.0),
        antithesis_p95=p95(an, 3.0),
        cleft_p25=pct(cl, 0.25),
        antithesis_p25=pct(an, 0.25),
        it_is_p95=p95([it_is_opener_rate(t) for t in usable], 0.0),
        chain_len=chain_len,
        chain_p95=p95([comma_chain_rate(t, chain_len) for t in usable],
                      CHAIN_RATE_FLOOR),
        n_pieces=len(usable),
    )


# --- 9. the unheeded reversal --------------------------------------------------
#
# The reference author's most-repeated complaint, on record from 2026-08-11 and
# reaffirmed under blind review 2026-08-28: a short flat assertion followed by a
# sentence that takes it back and restates it. His standing instruction is "say
# the thing once", and unlike every rated tic in this file he has never asked for
# it to be tolerated at any rate.
#
# The specimens, all of which the antithesis detector returned CLEAN on:
#   "This is not hope. Hope has weight, and this has none."
#   "The architecture is sound. What it was measuring was too narrow, and that is
#    a calibration problem, not a design one."
#   "What troubles me is not that I went quiet. It is that the quiet was comfortable."
#
# find_antithesis missed the first two because its is-not-X-It-is-Y pattern
# requires the second sentence to open with It/That/This followed by "is". A
# reversal that restates by repeating the noun ("Hope has weight") or by any
# other verb walks straight through. That is not a rare variant: repeating the
# noun is the more literary-sounding form, so it is the one a voice-cloning
# prompt selects for.
#
# Detected on SENTENCE PAIRS rather than on a span, because the construction is
# a relation between two sentences and no single-span regex can see it. Two
# shapes:
#
#   R1 negate-then-restate: S1 negates a predicate, S2 asserts about the same
#      subject. Requires a shared content word so that "It is not raining. We
#      should go." does not match.
#   R2 punchy-then-corrective: S1 is short (<= 9 words) and S2 opens with a
#      corrective move (But/Yet/What/The problem/Except/Actually/In fact).
#
# HARD, not advisory, and presence-based rather than rate-based. Every other tic
# here is rated because the author uses it at some nonzero rate himself. This one
# he has never asked to be tolerated at any rate: the instruction was "make sure
# it's not gonna happen".

# COPULAR negation only. An earlier draft matched every negation including "no
# money" and "did not answer", and fired on 71-88% of the author's own pieces:
# ordinary narrative negates constantly. The construction he objects to denies a
# PREDICATE and then restates it, so "is/was/has not" is the whole surface.
_NEG_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had)\s+not\b"
    r"|\b(?:isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)

# Openers that mark S2 as a correction of S1 rather than a new thought. But/Yet/
# Rather/Instead were in the first draft and are ordinary narrative prose: a
# sentence-initial "But" after a short line is a normal fiction rhythm, and on the
# reference corpus those four openers alone produced most of the false-positive
# rate. What remains is assistant-specific. Re-measure before trusting this list
# on a different author.
_CORRECTIVE_RE = re.compile(
    r"^(?:What\s+\w+"
    r"|The\s+(?:problem|point|question|issue|trouble|difficulty)\s+is"
    r"|It\s+is\s+that|It's\s+that|It\s+was\s+that|That\s+is\s+what|In\s+fact)\b",
    re.IGNORECASE,
)

# The setup half of the construction is short. A long negated sentence followed
# by a related one is ordinary argument, not the punchy-then-reversal move.
_REVERSAL_SETUP_MAX_WORDS = 14

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Function words carry no restatement signal: "this", "the", "is" recurring
# between two sentences means nothing. Only a repeated CONTENT word shows S2 is
# about S1's subject.
_STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her him his how i if in into is it its me my no nor not of on or our out
she should so some such than that the their them then there these they this those
to too was we were what when where which who whom why will with would you your
i'm it's that's there's what's dont don't isnt
""".split())


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", s.lower())
            if len(w) > 2 and w not in _STOP}


def find_reversals(text: str) -> list[str]:
    """Adjacent sentence pairs where the second takes back and restates the first.

    Returns the joined pair so the report can quote the whole construction; a
    half of it is not recognisable to a reader as the thing being flagged.
    """
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    out: list[str] = []
    for a, b in zip(sents, sents[1:]):
        pair = f"{a} {b}"
        # R1: S1 negates, S2 restates about the same thing. S2 is NOT required to
        # be free of negation: "This is not hope. Hope has weight, and this has
        # none." carries one in the restatement, and excluding those dropped the
        # specimen this detector was written for. The shared content word is what
        # keeps the pair a restatement rather than two adjacent thoughts.
        if (_NEG_RE.search(a)
                and len(a.split()) <= _REVERSAL_SETUP_MAX_WORDS
                and _content_words(a) & _content_words(b)):
            out.append(pair)
            continue
        # R2: short flat assertion, then a corrective turn.
        if len(a.split()) <= 9 and _CORRECTIVE_RE.match(b):
            out.append(pair)
    return out


# --- 10. self-explanation ------------------------------------------------------
#
# The single most-repeated complaint of the 2026-08-28 blind session: named in
# ALL TWELVE drafts across three voices and four generation arms. The author named
# it four different ways in one sitting: self-explaining the statement within
# itself; overexplaining and not giving the reader space; explaining and
# justifying; and too many "X is the Y that" constructions.
#
# That it appeared at equal strength in every arm is the finding. The arms varied
# the instruction and the retrieval; the defect did not move. It is the base
# model's prior for how a sentence ends, which is why no prompt change reached it
# and why it has to be caught after generation.
#
# One move in three grammatical disguises. A clause arrives at a point, then
# instead of stopping appends a clause that narrates what the point was:
#
#   HINGE           "Each attempt ends there, WHICH MEANS the phone has now asked
#                    me a genuine philosophical question three times"
#   IS-WHAT         "No threshold recovers it. THRESHOLDING IS WHAT discards it."
#   PICKUP          "an afternoon occurred and WAS WITNESSED and that THE
#                    WITNESSING was mine"
#
# Calibration, measured over his three live corpora: 9/9 specimens fire, 0/5
# controls fire, and his own rate is 0.14 (creative) / 0.28 (personal) / 0.41
# (research) per 1000 words. The hinge and is-what forms are HARD on presence at
# that rate. The pickup form carries the loosest definition and the highest
# corpus rate, so it stays advisory.

_HINGE_RE = re.compile(
    r",\s*which\s+(?:is|was|means|meant)\b"
    r"|(?:^|(?<=[.!?])\s+)Which\s+(?:is|was|means)\b",
    re.IGNORECASE,
)

# Reverse pseudo-cleft, restricted to a NOMINALIZED subject. The plain-noun form
# ("X was what killed it") is ordinary emphatic English and appears in the
# reference corpus; the nominalized form ("Thresholding is what discards it") is
# the tell. Requiring -ing/-ion/-ment/-ness separates the two without a rate gate,
# which matters because a rate gate would need a per-author calibration this check
# does not carry.
_IS_WHAT_RE = re.compile(
    r"\b(?:the\s+)?\w{3,}(?:ing|ion|ment|ness)\s+(?:\w+\s+){0,2}?(?:is|was)\s+what\b",
    re.IGNORECASE,
)

# \w{3,}? not \w{5,}?: the longer form needs an 8-character word and so missed
# "the costing nothing", which is the specimen this detector was written for.
# The len(stem) >= 4 check below is what actually guards against short junk.
_NOMINAL_RE = re.compile(r"\b(?:the|a|an)\s+(\w{3,}?(?:ing|ion|ment|ness))\b", re.IGNORECASE)


def _nom_stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ion", "ment", "ness"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def find_nominalization_pickups(text: str) -> list[str]:
    """A verb used in one clause returning as a nominalized subject in the next.

    "costs me nothing, and the costing nothing is what makes ..." / "was witnessed
    and that the witnessing was mine". The sentence performs the reader's
    inference and then hands back the result, which is what he means by a draft
    that does not give the reader space.
    """
    out: list[str] = []
    for sent in _SENT_SPLIT_RE.split(text):
        for m in _NOMINAL_RE.finditer(sent):
            stem = _nom_stem(m.group(1))
            if len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}\w*\b", sent[: m.start()], re.I):
                out.append(sent.strip()[:150])
                break
    return out


def find_self_explanations(text: str) -> list[str]:
    """Hinge and is-what forms. Hard-gated; see the calibration note above."""
    return ([m.group(0).strip(", ") for m in _HINGE_RE.finditer(text)]
            + [m.group(0).strip() for m in _IS_WHAT_RE.finditer(text)])


# --- 11. "It is" as a sentence-opening transition ------------------------------
#
# "It is carried by the items nobody was arguing about" -- flagged in the research
# packet, along with the note that "It is" is a transition he dislikes. Rate-gated
# rather than presence-gated, and profile-dependent, because the measurement says
# the answer differs sharply by voice. Sentence openers per 100 sentences over his
# own pieces of 10+ sentences:
#
#   creative  median 1.67   p95 5.36   max 6.85
#   personal  median 0.00   p95 4.55   max 7.02
#   research  median 0.00   p95 0.00   max 2.63
#
# He opens research sentences this way essentially never, and creative ones
# regularly. A single global threshold would either miss it in research or fire
# constantly on fiction.

_IT_IS_OPENER_RE = re.compile(r"(?:^|(?<=[.!?])\s+)It\s+(?:is|was)\b")


def find_it_is_openers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _IT_IS_OPENER_RE.finditer(text)]


def it_is_opener_rate(text: str) -> float:
    """Per 100 sentences."""
    n = len([s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()])
    return 100.0 * len(find_it_is_openers(text)) / n if n else 0.0


# --- 12. the borrowed simile ---------------------------------------------------
#
# "the way a machine refuses", "the way a stranger turns a wall that is not his",
# "the way anger does", "the way a tongue returns to a chipped tooth", "the way a
# patient thing gives when pressed long enough".
#
# Confirmed four separate times by two independent methods. A 2026-08-28
# stylometric pass over 28 prior generations measured it at up to 3.14 per 1000
# words against his corpus rate of 0.02 -- roughly 157x -- and he then flagged it
# unprompted in three different blind packets without seeing that measurement.
# It is the default lyric simile of the generating model, not his.
#
# Calibrated: 5/6 specimens fire, 0/5 controls, and his own rate is 0.023
# (creative) / 0.026 (personal) / 0.000 (research) per 1000 words, touching 2-3%
# of pieces. The lookahead exclusions carry the fixed idioms, which is where the
# false positives were: "that is the way it is", "the way back", "showed me the
# way", "the way he always walked" are all ordinary English and none of them is
# the simile.
_WAY_SIMILE_RE = re.compile(
    r"\bthe\s+way\s+"
    r"(?!it\s+is\b|back\b|home\b|to\b|out\b|in\b|things\s+are\b"
    r"|he\s+always\b|she\s+always\b|I\s+always\b)"
    r"(?:a|an|the|one|you|\w+)\s+[\w\s]{0,28}?"
    r"\b(?:does|do|did|is|was|are|were|would|will|goes|gets|turns|refuses|gives|"
    r"stops|falls|moves|works|breaks|holds|sits|looks|sounds|feels|returns|comes|"
    r"opens|closes|wears|settles|leaves|behaves|responds|answers|waits|bends|"
    r"slips|drifts|fades)\b",
    re.IGNORECASE,
)


def find_way_similes(text: str) -> list[str]:
    """"..., the way a machine refuses." Reaches for a comparison nobody asked for."""
    return [m.group(0).strip() for m in _WAY_SIMILE_RE.finditer(text)]


# --- 13. the declarative rating ------------------------------------------------
#
# "That is the part that sat with me." "That is a worse position than not
# knowing." "That is what tells me the drift is working."
#
# The sibling _SELF_RATING_RE cannot see these: it requires an evaluative head
# from a fixed list (strongest, most X, the whole point), and these rate by
# pointing rather than by naming a superlative, so no head is present to match.
# Rejected on sight by the reference author under blind review.
#
# Measured at exactly 0.000 per 1000 words across all three of his corpora, which
# is why this is hard rather than rated: there is no rate of his to stay under.
_DECLARATIVE_RATING_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:That|This|It)\s+(?:is|was)\s+"
    r"(?:the\s+(?:part|thing|bit|point)\s+(?:that|which)\b"
    r"|a\s+(?:worse|better|harder|stranger|smaller|larger|different|stronger|"
    r"weaker|cleaner|simpler)\s+\w+\s+than\b"
    r"|what\s+(?:tells|makes|matters|counts|does|separates|explains)\b)",
    re.IGNORECASE,
)


def find_declarative_ratings(text: str) -> list[str]:
    """A sentence that grades the previous one by pointing at it."""
    return [m.group(0).strip() for m in _DECLARATIVE_RATING_RE.finditer(text)]


# --- 14. comma-spliced clause chains -------------------------------------------
#
# Named by the reference author in every voice profile and never previously
# detectable: "the over use of commas, that I could, and I preferred, and I am";
# "run on commas and flowery vocab"; "really long overwrought commas". The reason
# no check existed is recorded in docs/AUTHOR_FEEDBACK.md section 7 -- the
# fingerprint carries `commas_per_sent`, which is a MEAN, and a mean cannot see
# accumulation. Six clauses chained inside one sentence and six clean sentences
# produce the same number. Nothing else in this package looks at the shape of the
# distribution, so the defect survived every gate.
#
# So this is measured as a MAXIMUM inside a sentence and a RATE across sentences,
# not as a comma count. A sentence is split on `,`, `;` and clause-level
# coordinators (and/but/or/nor/so/yet), and each fragment is asked whether it is a
# finite clause -- an explicit subject followed by a finite verb. The run of
# CONSECUTIVE finite clauses is the chain length. That distinction is the whole
# detector: this author's long sentences accumulate PHRASES, and the generated
# ones accumulate PREDICATIONS. His longest comma sentence in the creative corpus
# runs 18 comma-delimited fragments and scores a chain length of 1, because those
# fragments are appositives and participials rather than clauses. Counting commas
# would have flagged it; counting clauses does not.
#
# WHY THIS ONE IS PER-AUTHOR CALIBRATED, unlike sections 1-4. The model-habit
# detectors above are properties of the generating model and travel between
# authors. This one is not: the construction is a device this author uses on
# purpose, and some of his strongest lines are it: a four-clause first-person
# list in one creative piece, a four-clause "they had ..." run in another, and a
# second-person accumulation in a personal essay.
#
# A fixed presence threshold would gate those, which is the failure mode this file
# has already been through twice on the triad rule (50.5% then 23.4% of his own
# corpus). Measured over his three live corpora, share of sentences reaching each
# chain length:
#
#            n_sents   >=3      >=4      >=5      >=6
#   creative  18,709   1.155%   0.176%   0.027%   0.005%
#   research     850   0.118%   0.118%   0.000%   0.000%
#   personal   2,198   1.319%   0.091%   0.000%   0.000%
#
# Three chained clauses is ordinary English and 1.2% of his sentences, which is why
# CHAIN_MIN_CLAUSES floors the calibrated length at four rather than three. The
# calibrated length is the 95th percentile of the longest chain per calibration
# unit; on all three of his profiles that percentile lands at 2-3 and the floor
# carries it to 4, so the calibration is currently doing no visible work FOR HIM.
# It is here for the case the floor was written against -- an author who chains
# habitually, whose p95 lands at 5 or 6, and for whom a fixed 4 would repeat the
# triad rule's mistake a third time.
#
# The gate is two-sided. Density is the complaint ("over USE"), so the main route
# is a RATE against his own 95th percentile, floored at CHAIN_RATE_FLOOR; one
# chain inside a long piece is not overuse, one chain inside a twelve-sentence
# draft is. A chain two clauses past the calibrated length short-circuits the
# rate, because at that length the sentence is the defect on its own.
#
# GROUND TRUTH. Two drafts in profiles/*/judgments.jsonl carry a blind rationale
# that names comma overuse in his own words, and the rate route catches both:
#
#   ab2-2026-08-28-outoftouch-AvB / draftB-corpus-anchored, ranked 2 of 2:
#     "then the overuse of commas: 'that I could', 'and I preferred', 'and I am'"
#     -- the sentence he is quoting is "I found in it a version of our friendship
#     that made no demands, that I could hold at whatever distance suited me, and
#     I preferred it for a while, and I am ashamed of the preference." Chain 4,
#     rate 8.3 per 100 sentences. His verbatim complaint in the ledger is a
#     quotation of this line, which is why it is the calibration specimen.
#
#   arm4-2026-08-28-sibling / observe-wholepiece, ranked 4 of 4, "F minus tier":
#     "'May I was angry, and the anger was...' comma comma." Two chains of 4 in
#     fourteen sentences, rate 14.3.
#
# The length route is not idle either: it is what catches a single monster inside
# an otherwise clean draft, where the rate stays low by dilution. In this author's
# own corpus it is the only thing that reaches his longest chain, eight clauses.
#
# Swept over all 40 generated arms in the three judgment stores plus the one human
# arm: 5 arms fire, and all five are drafts he ranked at or near the bottom or
# whose rationale names an overexplaining tell. The human arm's longest chain is
# 2; every model arm runs 3 or 4. Nothing he approved fires.
#
# ADVISORY, not hard, and this is a considered split from the four hard flags. The
# hard flags are constructions he rejected by name under blind review AND that
# measure at or near 0.000 in his corpora -- there is no rate of his to stay under,
# so presence is the right gate. This one he has also named by name, repeatedly,
# but it measures 0.09-0.18% of his sentences rather than zero, and the three lines
# described above are his. The precedent for a named complaint that is nonetheless
# rate-gated is it_is_opener_rate in section 11, advisory for exactly this reason:
# the answer is profile-dependent, so a presence gate would be wrong somewhere.
# Making this hard would reject his own prose on 4-5% of pieces, and a gate that
# rejects the author is how a voice system produces prose more average than him.
#
# Measured false-positive rate, the share of his own pieces the gate flags with
# that voice's own calibration: creative 2/98 = 2.0%, research 0/21 = 0.0%,
# personal 1/38 = 2.6% over the raw source documents, and 1.5% / 0.0% / 2.7% over
# the shorter units cmd_calibrate actually builds. Against 6.3% of the generated
# text held in evals/, which has already passed the whole gate pipeline.
#
# The precedent this had to clear is in section 1: a triad rule that fired on 50.5%
# then 23.4% of his real writing was rewritten both times. The three surviving
# false positives are individually identifiable rather than a class of his prose
# being misread -- one creative piece (a chain of 8, the longest he has written)
# and one personal essay (a chain of 6) both come in on the length route, and a
# third piece is a rate hit, one chain inside a 24-sentence piece.
#
# The sweep is in tests/test_rhetoric.py. Re-run it before pointing this at a
# different author; a writer who chains habitually will calibrate to a longer
# CHAIN length and a higher rate, and that is the mechanism working, not a bug.

_CHAIN_COORD = r"(?:and|but|or|nor|so|yet)"
# Subordinators and coordinators both appear at the head of a chained clause
# ("..., THAT I could, AND I preferred"), so the clause test skips one of either
# before looking for the subject.
_CHAIN_SUBORD = (
    r"(?:that|which|who|when|while|because|if|though|although|since|where|"
    r"after|before|until|unless|whereas|as|and|but|or|nor|so|yet)"
)
_CHAIN_AUX = (
    r"am|is|are|was|were|be|been|being|have|has|had|do|does|did|can|could|"
    r"will|would|shall|should|may|might|must|ought|used"
)
# Irregular pasts carry no -ed to match on, and leaving them out cost real chains:
# "I told myself the room was wrong" scored as a non-clause and broke one run of
# four into two runs of two. This is the common irregular set, not a complete one;
# a verb missing from it makes the detector miss a chain, never invent one.
_CHAIN_IRREG = (
    r"said|went|made|took|came|saw|knew|thought|got|gave|found|told|felt|left|kept|"
    r"held|put|brought|began|sat|stood|lost|meant|met|ran|paid|heard|let|set|sent|"
    r"spoke|spent|understood|wore|wrote|drove|fell|drew|grew|broke|chose|ate|drank|"
    r"rose|threw|bought|caught|taught|fought|sold|built|slept|swept|read|led|fled|"
    r"hid|rode|woke|stole|shook|forgot|became|blew|flew|swam|sang|rang|hung|dug|won|"
    r"lay|laid|bore|tore|struck|stuck|shot|cut|hit|beat|bent|crept|dealt|knelt|lent|"
    r"slid|spat|split|spread|sprang|sank|shone|shrank|swore|swung|wept|withdrew|"
    r"arose|awoke|clung|strode"
)
_CHAIN_CONTR = r"(?:'s|'re|'ve|'ll|'d|'m)"
_CHAIN_FINITE = (
    rf"(?:(?:{_CHAIN_AUX})\b|(?:{_CHAIN_IRREG})\b|\w+(?:ed|es|s)\b|{_CHAIN_CONTR})"
)
# The -s/-es half of the heuristic above is a plural noun as often as it is a
# third-person verb, and after a DEMONSTRATIVE that ambiguity is the whole
# difference between a clause and a noun phrase. "He had kept them all, these
# useless receipts, these letters, these tickets" scored as a four-clause chain,
# because "useless", "letters" and "tickets" all end in -s. Demonstrative
# subjects therefore need a verb that is unambiguously finite.
_CHAIN_FINITE_STRICT = rf"(?:(?:{_CHAIN_AUX})\b|(?:{_CHAIN_IRREG})\b|\w+ed\b|{_CHAIN_CONTR})"
_CHAIN_ADV = (
    r"(?:not|never|still|also|already|just|only|then|now|again|once|always|often|\w+ly)"
)
# Personal pronouns are the one subject class that needs no verb list: at the head
# of a comma-delimited fragment "I", "he", "they" are subjects essentially always,
# and the next content word is the verb. Every other subject class has to show an
# explicit finite verb, because "the way it looked" and "in the morning light" are
# noun phrases rather than clauses, and matching those was the false-positive surface.
_CHAIN_PPRON = r"(?:I|you|he|she|we|they|it)"
_CHAIN_SUBJ = (
    r"(?:there|nothing|everything|something|anything|"
    r"nobody|everybody|everyone|someone|one|who)"
)
# Held apart from _CHAIN_SUBJ because a demonstrative is a determiner as often as
# it is a pronoun, and only the verb after it says which. Kept as a subject class
# rather than dropped, because a chain of relative clauses is the real thing:
# "that opened the door, that let him in, that kept him there, that sent him
# home" is four of them.
_CHAIN_DEM = r"(?:this|that|these|those)"
# One noun after the determiner, not two. With two, "the way it looked" matches
# ("the way it" + "looked") and it is a noun phrase; "the cat had gone" needs only one.
_CHAIN_DET = r"(?:the|a|an|his|her|its|their|my|our|your|this|that|these|those)"

# Commas and semicolons always split. A bare coordinator splits only when a
# SUBJECT follows it, and this lookahead is not decoration: splitting on every
# "and/or/so/yet" cut "she had said SO once out loud" into "she had said" +
# "once out loud" and turned a five-clause specimen into a three-clause one.
# "black and white", "two or three", "said so", "not yet" all stay whole now,
# while "I could and I preferred and I am" still splits. The {1,2} repetition is
# for stacked coordinators ("..., and yet she still found ..."), which otherwise
# leave a bare "and" fragment behind that breaks the run it belongs to.
_CHAIN_SUBJ_HEAD = rf"(?:{_CHAIN_PPRON}|{_CHAIN_SUBJ}|{_CHAIN_DEM}|{_CHAIN_DET})\b"
_CHAIN_SPLIT_RE = re.compile(
    rf"[,;]|\s+(?:{_CHAIN_COORD}\s+){{1,2}}(?={_CHAIN_SUBJ_HEAD})", re.IGNORECASE
)
_CHAIN_CLAUSE_RE = re.compile(
    rf"^\s*(?:{_CHAIN_SUBORD}\s+)?(?:"
    rf"{_CHAIN_PPRON}\b\s+(?:{_CHAIN_ADV}\s+){{0,2}}[a-z']+\b"
    rf"|{_CHAIN_SUBJ}\b\s+(?:{_CHAIN_ADV}\s+){{0,2}}{_CHAIN_FINITE}"
    rf"|{_CHAIN_DEM}\b\s+(?:{_CHAIN_ADV}\s+){{0,2}}{_CHAIN_FINITE_STRICT}"
    rf"|{_CHAIN_DET}\s+\w+\s+(?:{_CHAIN_ADV}\s+){{0,2}}{_CHAIN_FINITE}"
    rf")",
    re.IGNORECASE,
)


def _chain_length(sent: str) -> int:
    """Longest run of CONSECUTIVE finite clauses joined by commas, semicolons or
    coordinators inside one sentence.

    A fragment that is not a finite clause -- an appositive, a list item, a
    participial -- breaks the run rather than extending it. That is what keeps
    this author's phrase-accumulating long sentences out of the count.
    """
    run = best = 0
    for seg in _CHAIN_SPLIT_RE.split(sent):
        seg = seg.strip()
        if not seg:
            continue
        if _CHAIN_CLAUSE_RE.search(seg):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def find_comma_chains(text: str, min_clauses: int = CHAIN_MIN_CLAUSES) -> list[str]:
    """Sentences carrying ``min_clauses`` or more chained finite clauses.

    Returns whole sentences rather than spans: the construction IS the
    accumulation, and half of it is not recognisable to a reader as the thing
    being flagged.
    """
    return [s for s in _split_sentences(text) if _chain_length(s) >= min_clauses]


def longest_comma_chain(text: str) -> int:
    """The single longest clause chain anywhere in the text. This is the statistic
    the fingerprint's mean ``commas_per_sent`` cannot see."""
    return max((_chain_length(s) for s in _split_sentences(text)), default=0)


def comma_chain_rate(text: str, min_clauses: int = CHAIN_MIN_CLAUSES) -> float:
    """Over-length sentences per 100 sentences.

    Normalised by sentence count rather than word count, matching
    ``it_is_opener_rate``: this is a property of how sentences are built, and a
    per-1000-words rate would move with sentence length instead of with the habit.
    """
    sents = _split_sentences(text)
    if not sents:
        return 0.0
    return 100.0 * len(find_comma_chains(text, min_clauses)) / len(sents)
