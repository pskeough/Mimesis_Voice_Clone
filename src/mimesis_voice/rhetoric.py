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
    absence_punches: list[tuple[str, str]] = field(default_factory=list)
    clefts: list[str] = field(default_factory=list)
    antithesis: list[str] = field(default_factory=list)
    closing_flourish: list[str] = field(default_factory=list)
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
    def hard_flags(self) -> list[str]:
        """Leakage is unambiguous (the model breaking character) and gates like an
        em-dash. The three tics are real but each has a nonzero measured false-positive
        rate against real prose, so they are advisory -- see render()."""
        return ["task-leakage"] if self.leakage else []

    @property
    def is_clean(self) -> bool:
        return not (self.leakage or self.triads or self.meta_asides
                    or self.absence_punches or self.closing_flourish
                    or self.cleft_over or self.antithesis_over
                    or self.cleft_under or self.antithesis_under)


def analyze(text: str, cal: "RhetoricCalibration | None" = None) -> RhetoricReport:
    cal = cal or RhetoricCalibration()
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
        leakage=find_leakage(text),
        triads=find_triads(text),
        meta_asides=find_meta_asides(text),
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
    return out


def render_lines(rep: RhetoricReport) -> list[str]:
    out = _render_new(rep)
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
    for setup, punch in rep.absence_punches:
        out.append(
            f"- [MEDIUM] Scene-then-flat-absence-punch: \"...{setup[-60:]}\" then "
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
    # Lower edges of the author's own range. A rate check with only a ceiling is
    # half a check: measured on this project, driving antithesis to 0.00 scored as
    # "clean" while being exactly as unlike the author as 19.80, because his own
    # rate is 2.88-3.62. One-sided checks are how a voice system produces prose
    # more average than the author it is imitating.
    cleft_p25: float = 0.0
    antithesis_p25: float = 0.0
    n_pieces: int = 0

    def to_dict(self) -> dict:
        return {"cleft_p95": self.cleft_p95, "antithesis_p95": self.antithesis_p95,
                "cleft_p25": self.cleft_p25, "antithesis_p25": self.antithesis_p25,
                "n_pieces": self.n_pieces}

    @classmethod
    def from_dict(cls, d: dict) -> "RhetoricCalibration":
        return cls(cleft_p95=d.get("cleft_p95", 0.0),
                   antithesis_p95=d.get("antithesis_p95", 0.0),
                   cleft_p25=d.get("cleft_p25", 0.0),
                   antithesis_p25=d.get("antithesis_p25", 0.0),
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
    return RhetoricCalibration(
        cleft_p95=p95(cl, 1.0),
        antithesis_p95=p95(an, 3.0),
        cleft_p25=pct(cl, 0.25),
        antithesis_p25=pct(an, 0.25),
        n_pieces=len(usable),
    )
