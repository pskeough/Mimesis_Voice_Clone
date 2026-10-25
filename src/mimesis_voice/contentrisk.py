"""Content-type risk: which sentences carry the moves Pangram 4 reads as AI.

Reporting only. Nothing here gates, ranks or rewrites. It exists so a writer can
see which sentences of a research draft carry the content types that a detection
study (Oct 2026, about 800 Pangram 4.1 scans of research prose; docs/DETECTION.md)
found Pangram labels as AI whatever the wording, and rewrite those sentences
themselves.

The finding it encodes:

- Read AI: interpretation of the paper's own results, ranking the study against
  prior work, contribution and gap claims, roadmaps, glosses, obligation and
  consequence claims about the study's own design, abstract concept-equals-concept
  exposition, and stimulus or specification content (prompt templates, JSON
  formats, formal definitions, Example blocks).
- Read Human: reports of cited work ("X et al. found Y") on named systems,
  concrete run facts (counts, seeds, settings, named dictionaries), design
  contrasts ("In contrast with X, who ..., we use ..."), aims and hypothesis
  statements.
- Word-level levers did not move the label: pronoun, stance, sentence length,
  grammatical subject and word bans changed nothing or made it worse. The label
  follows the move, so this module tags moves, not words.
- Labels are window-level (512 tokens, stride 256). One caught sentence can flip
  about 280 surrounding words, so risky sentences are reported with their runs.

Every pattern is a heuristic over surface forms of those moves. The module's
agreement with real Pangram sentence labels is measured offline by
``evals/pangram_content_check.py`` against archived scans (within-document odds
ratio 1.63 [1.36, 1.95] over 8,685 labelled sentences); read that number before
trusting a tag. The tags are a reading aid. No offline detector tracked Pangram
forward in time in the study (a fine-tuned tagger, an LLM judge, Binoculars and
desklib all failed), and this one is not an exception until a scan says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import segment

# --- normalisation -------------------------------------------------------------

_CITE_CMD_RE = re.compile(r"\\(?:cite|citet|citep|citealt|citeauthor|parencite|textcite|autocite)\*?(?:\[[^\]]*\])*\{[^}]*\}")
_TEX_CMD_RE = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_TEX_BARE_RE = re.compile(r"\\[A-Za-z]+\*?")
_MATH_RE = re.compile(r"\$[^$]+\$")
CITE_TOKEN = "(Cited, 2000)"


def normalise(sentence: str) -> str:
    """Plain prose for pattern matching: LaTeX citations become an author-year
    token, other commands keep their argument text, inline math becomes a token."""
    s = _CITE_CMD_RE.sub(CITE_TOKEN, sentence)
    s = _MATH_RE.sub(" MATH ", s)
    for _ in range(3):
        s2 = _TEX_CMD_RE.sub(r"\1", s)
        if s2 == s:
            break
        s = s2
    s = _TEX_BARE_RE.sub("", s)
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


# --- safe markers ----------------------------------------------------------------

_CITATION_RE = re.compile(
    r"\b[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?,?\s*\(?(?:19|20)\d{2}[a-z]?\)?"
    # One pass over a parenthesis that holds a capital and a year. The earlier
    # repeated-group form split k citations about 2^k ways and took 30 s on six
    # citations in an unclosed parenthesis.
    r"|\((?=[^()]*[A-Z])[^()]*?\b(?:19|20)\d{2}[a-z]?\b[^()]*\)"
    r"|\[\d+(?:[,\u2013-]\s*\d+)*\]"
)
_REPORT_VERB_RE = re.compile(
    r"\b(?:found|find|finds|showed|shows|show|reported|reports|observed|observes|built|builds|"
    r"tested|tests|measured|measures|used|uses|proposed|proposes|introduced|introduces|trained|"
    r"trains|examined|examines|compared|compares|reviewed|reviews|described|describes|"
    r"demonstrated|demonstrates|evaluated|evaluates|ran|runs|applied|applies|identified|"
    r"identifies|estimated|estimates|recruited|collected|released|developed|develops)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?")
_NAMED_SYSTEM_RE = re.compile(r"\b(?:[A-Za-z]+-?\d[\w.\-]*|[A-Z][a-z]+[A-Z]\w*|[A-Z]{2,}\d*)\b")

# --- risk classes ----------------------------------------------------------------
#
# name -> (pattern, one-line evidence). Patterns are matched against the
# normalised sentence, case-insensitively.

RISK: dict[str, tuple[re.Pattern, str]] = {
    "interpretation": (re.compile(
        r"^(?:this|these|that|those|such|our|the (?:results?|findings?|pattern|data|effect|gap|difference))"
        r"\s+(?:\w+\s+){0,3}?(?:suggests?|indicates?|implies|imply|means?|reflects?|highlights?|"
        r"underscores?|reveals?|points?\s+to|speaks?\s+to|is\s+consistent\s+with|"
        r"is\s+evidence|shows?\s+that|demonstrates?\s+that)\b"
        r"|\bwe\s+(?:interpret|read|take)\s+(?:this|these|that|the)\b"
        r"|\b(?:can|could|should|may|might)\s+be\s+(?:understood|read|interpreted|seen|viewed)\s+as\b"
        r"|^(?:taken together|overall|in sum|in short|collectively|in summary|together),"
        r"|\bthis\s+(?:is\s+)?(?:likely|probably)\s+(?:because|due)\b",
        re.IGNORECASE), "own-result interpretation reads AI (a Discussion built on it scanned 100% AI; offline OR 1.73)"),
    "ranking": (re.compile(
        r"\b(?:unlike|beyond|extend(?:s|ing)?|improv(?:e|es|ing)\s+(?:up)?on|outperform\w*|go(?:es)?\s+further\s+than)\b"
        r"[^.]{0,60}\b(?:prior|previous|earlier|existing|past|most|other)\s+(?:work|studies|approaches|methods|research|literature|benchmarks?)\b"
        r"|\b(?:prior|previous|earlier|existing|past)\s+(?:work|studies|research|approaches|literature|benchmarks?)\b"
        r"[^.]{0,50}\b(?:has|have|had|did|do)\s+(?:not|only|largely|rarely|never|mostly)\b"
        r"|\bonly\s+qualitatively\b|\bfor\s+the\s+first\s+time\b|\bthe\s+first\s+(?:study|work|paper|benchmark|to)\b"
        r"|\bnovel\b",
        re.IGNORECASE), "ranking the study against prior work (one such sentence was caught in 14/19 configurations)"),
    "contribution": (re.compile(
        r"\b(?:our|the|its|three|two|main|key)\s+(?:main\s+|primary\s+|central\s+|key\s+)?contributions?\b"
        r"|\bwe\s+contribute\b|\bcontributes?\s+to\s+(?:the|a|our|this)\b"
        r"|\b(?:this|the\s+present|our)\s+(?:paper|work|study|article|framework|approach|benchmark|analysis)\s+"
        r"(?:provides|offers|introduces|presents|proposes|establishes|advances|makes|enables|fills|delivers|shows)\b"
        r"|\bwe\s+(?:introduce|propose|present)\s+(?:a|an|the)\s+(?:new|novel|first)\b",
        re.IGNORECASE), "contribution claims read AI (passing drafts had them removed or turned into reports)"),
    "gap": (re.compile(
        r"\b(?:remains?|remained)\s+(?:largely\s+)?(?:unclear|unknown|unexplored|understudied|open|untested|unaddressed|unexamined)\b"
        r"|\blittle\s+(?:is|was)\s+known\b"
        r"|\b(?:has|have)\s+(?:not|rarely|seldom|never)\s+(?:yet\s+)?(?:been\s+)?(?:\w+\s+)?(?:examined|studied|tested|investigated|addressed|explored|measured|evaluated|compared)\b"
        r"|\bno\s+(?:prior\s+|previous\s+|existing\s+)?(?:study|studies|work|paper|research)\s+(?:has|have)\b"
        r"|\bthis\s+gap\b|\b(?:a|the)\s+gap\s+in\b|\black\s+of\s+(?:research|studies|evidence|work)\b",
        re.IGNORECASE), "gap claims read AI ('Earlier work has not tested'); the same content stated as an aim passed"),
    "roadmap": (re.compile(
        r"\b(?:the\s+)?(?:rest|remainder)\s+of\s+(?:this|the)\s+(?:paper|article|section)\b"
        r"|\bsections?\s+\d"
        r"|\bwe\s+(?:will|then|next)\s+(?:show|describe|discuss|present|outline|turn|examine|review|investigate)\b"
        r"|\b(?:below|in\s+what\s+follows|in\s+the\s+next\s+section),?\s+we\b"
        r"|\bwill\s+(?:also\s+)?(?:show|be\s+(?:described|discussed|presented|shown))\b"
        r"|\bis\s+(?:organi[sz]ed|structured)\s+as\s+follows\b",
        re.IGNORECASE), "roadmaps and future-tense meta ending in an own claim caught whole runs (offline OR 1.63)"),
    "gloss": (re.compile(
        r"\bthat\s+is,|\bi\.e\.|\bin\s+other\s+words\b|\bput\s+(?:differently|simply|another\s+way)\b"
        r"|^this\s+means\b|\bthat\s+means\b|\bnamely\b|^specifically,|\b(?:is|are)\s+best\s+(?:understood|thought\s+of|read)\s+as\b"
        r"|^consider\b|\bin\s+(?:this|that)\s+sense\b",
        re.IGNORECASE), "glosses and restatements ('In other words', 'This means') concentrated in stable AI paragraphs"),
    "obligation": (re.compile(
        r"\bcan\s+only\s+be\s+\w+\s+(?:if|when)\b|\bwould\s+require\b|\bwould\s+be\s+confounded\b"
        r"|\bcannot\s+be\s+(?:isolated|separated|distinguished|ruled\s+out|attributed)\b"
        r"|\bfull\s+validation\b",
        re.IGNORECASE), "consequences drawn about the study's own design, caught 8/8 in every context tried"),
    "specification": (re.compile(
        r"[{}]|\bjson\b|\bsystem\s+prompt\b|\bprompt\s+(?:template|reads?|was|asked|instructed|stated|variants?)\b"
        r"|\byou\s+are\s+(?:a|an)\b|\bexample\s+\(\d+\)|\b(?:is|are)\s+defined\s+as\b|\blet\s+\w+\s+(?:denote|be)\b"
        r"|\bwhere\s+\w+\s+(?:is|denotes)\b|\btemplate\b|\bresponse\s+format\b",
        re.IGNORECASE), "stimulus building and formal definitions read AI even with citations (offline OR 2.21)"),
}

_CONCEPT_COPULA_RE = re.compile(r"\b(?:is|are)\s+(?:a|an|the|distinct|different|equivalent|separate)\b", re.IGNORECASE)
_CONCEPT_NOTE = "abstract 'X is a Y' exposition with no number, citation or named system (AI 7/8 draws; offline OR 1.63)"

SAFE: dict[str, str] = {
    "cited-report": "reports what a cited work did or found (stable Human paragraphs: cited-finding share 0.73 vs 0.15)",
    "run-fact": "concrete counts, settings or named systems (Human paragraphs carry more numbers per beat)",
    "design-contrast": "'In contrast with X, who ..., we use' design differences passed",
    "aims": "aims and hypothesis statements passed in every observed case (5/5)",
}
# The comparator must be a named work (capitalised author or a citation); "Unlike prior work"
# is ranking, not a design contrast.
_DESIGN_CONTRAST_RE = re.compile(r"^(?i:in\s+contrast\s+(?:with|to)|unlike|whereas)\s+(?:[A-Z][A-Za-z'\-]+|\()")
_AIMS_RE = re.compile(
    r"\b(?:the|our)\s+(?:aim|goal|purpose|objective)s?\s+of\s+(?:the\s+present|this)\s+(?:study|work|paper)\b"
    r"|\bwe\s+(?:expected|hypothesi[sz]ed|predicted)\b|^H\d+[:.]",
    re.IGNORECASE)
_OWN_SUBJECT_RE = re.compile(r"^(?:this|these|that|those|such|our|we|the\s+(?:results?|findings?|pattern|data))\b", re.IGNORECASE)


@dataclass
class SentenceRisk:
    index: int                 # prose-sentence number, 1-based
    text: str
    words: int
    risk: list[str] = field(default_factory=list)
    safe: list[str] = field(default_factory=list)
    citations: int = 0

    @property
    def is_risk(self) -> bool:
        return bool(self.risk)


def classify(sentence: str, index: int = 0) -> SentenceRisk:
    s = normalise(sentence)
    words = len(s.split())
    cites = len(_CITATION_RE.findall(s))
    has_num = bool(_NUMBER_RE.search(s.replace(CITE_TOKEN, "")))
    # Capitalised tokens after the first word count as named things; the first
    # word is capitalised by grammar.
    rest = " ".join(s.split()[1:]).replace(CITE_TOKEN, "")
    named = bool(_NAMED_SYSTEM_RE.search(rest)) or bool(re.search(r"\b[A-Z][a-z]{2,}\b", rest))
    out = SentenceRisk(index=index, text=sentence, words=words, citations=cites)

    if cites and _REPORT_VERB_RE.search(s):
        out.safe.append("cited-report")
    if _DESIGN_CONTRAST_RE.search(s) and re.search(r"\bwe\b", s, re.IGNORECASE):
        out.safe.append("design-contrast")
    if _AIMS_RE.search(s):
        out.safe.append("aims")
    if (has_num or _NAMED_SYSTEM_RE.search(rest)) and not out.safe:
        out.safe.append("run-fact")

    for name, (pat, _) in RISK.items():
        if not pat.search(s):
            continue
        # A cited work interpreting its own result is a report of cited work.
        if name == "interpretation" and "cited-report" in out.safe and not _OWN_SUBJECT_RE.search(s):
            continue
        # A design contrast with a named comparator passes; it is not ranking.
        if name == "ranking" and "design-contrast" in out.safe:
            continue
        # Aims statements carried gap content and passed.
        if name == "gap" and "aims" in out.safe:
            continue
        out.risk.append(name)
    if (not out.risk and _CONCEPT_COPULA_RE.search(s) and not cites and not has_num and not named
            and words >= 8):
        out.risk.append("concept-exposition")
    return out


def evidence(name: str) -> str:
    if name in RISK:
        return RISK[name][1]
    if name == "concept-exposition":
        return _CONCEPT_NOTE
    return SAFE.get(name, "")


# --- document report -------------------------------------------------------------

# Pangram scores 512-token windows; at ~0.75 words per token that is ~384 words.
WINDOW_WORDS = 384
# Realised-text citation density in stable Human vs stable AI paragraphs, per 100
# words, measured in the same study. Reported as reference points, not a gate.
CITES_PER100_HUMAN = 2.6
CITES_PER100_AI = 1.0


@dataclass
class RiskReport:
    sentences: list[SentenceRisk]
    words: int
    citations: int

    @property
    def risky(self) -> list[SentenceRisk]:
        return [s for s in self.sentences if s.is_risk]

    @property
    def risk_word_share(self) -> float:
        return sum(s.words for s in self.risky) / self.words if self.words else 0.0

    @property
    def cites_per100(self) -> float:
        return 100.0 * self.citations / self.words if self.words else 0.0

    def runs(self) -> list[tuple[int, int]]:
        """Maximal runs of consecutive risky sentences, as 1-based (first, last)."""
        out, start = [], None
        for s in self.sentences:
            if s.is_risk and start is None:
                start = s.index
            if not s.is_risk and start is not None:
                out.append((start, s.index - 1))
                start = None
        if start is not None:
            out.append((start, self.sentences[-1].index))
        return out

    def windows(self) -> list[tuple[int, int, float]]:
        """Pangram-sized windows (stride half a window) as (first, last, risky word share)."""
        out, n = [], len(self.sentences)
        if not n:
            return out
        starts = [0]
        acc = 0
        for i, s in enumerate(self.sentences):
            acc += s.words
            if acc >= WINDOW_WORDS // 2 and i + 1 < n:
                starts.append(i + 1)
                acc = 0
        for st in starts:
            wsum = rsum = 0
            j = st
            while j < n and wsum < WINDOW_WORDS:
                wsum += self.sentences[j].words
                rsum += self.sentences[j].words if self.sentences[j].is_risk else 0
                j += 1
            out.append((st + 1, j, rsum / wsum if wsum else 0.0))
            if j >= n:
                break
        return out


def analyze(text: str) -> RiskReport:
    sents = [classify(t, i) for i, t in enumerate(segment.sentences(text), 1)]
    return RiskReport(sentences=sents, words=sum(s.words for s in sents),
                      citations=sum(s.citations for s in sents))


def render(rep: RiskReport, max_listed: int = 25) -> str:
    """The DETECTION block for the scrub report and the edit tool."""
    lines = ["## DETECTION RISK (Pangram content types; reporting only, never gates)"]
    if not rep.sentences:
        return "\n".join(lines + ["- No prose sentences."])
    lines.append(
        f"- {len(rep.risky)}/{len(rep.sentences)} sentences carry a move Pangram labelled AI in "
        f"testing ({100 * rep.risk_word_share:.0f}% of words). Citations: {rep.cites_per100:.1f} per 100 "
        f"words (stable Human paragraphs ran {CITES_PER100_HUMAN}, stable AI {CITES_PER100_AI})."
    )
    for s in rep.risky[:max_listed]:
        head = s.text if len(s.text) <= 140 else s.text[:137] + "..."
        lines.append(f"  [{s.index}] {', '.join(s.risk)}: {head}")
    if len(rep.risky) > max_listed:
        lines.append(f"  ... {len(rep.risky) - max_listed} more")
    long_runs = [(a, b) for a, b in rep.runs() if b - a >= 2]
    if long_runs:
        lines.append("- Runs of 3+ risky sentences (one caught sentence can take ~280 words with it): "
                     + ", ".join(f"{a}-{b}" for a, b in long_runs))
    hot = [(a, b, sh) for a, b, sh in rep.windows() if sh >= 0.5]
    if hot:
        lines.append("- Windows (~384 words) over half risky: "
                     + ", ".join(f"sentences {a}-{b} ({100 * sh:.0f}%)" for a, b, sh in hot))
    lines.append(
        "- What to do: these are content types, not wording. Rewording them did not change the label "
        "in testing. Either the author writes or keeps these sentences in their own words, or the "
        "content is carried by what cited studies did (author-year citation beside the claim). "
        "Do not add hedges, stance phrases or synonyms to them."
    )
    return "\n".join(lines)
