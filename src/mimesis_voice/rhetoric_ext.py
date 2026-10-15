"""Second detector family: the tells that survived rhetoric.py.

Built 2026-09-02 from a real failure. Two announcement posts written in a
second author's voice by the shipped engine passed that author's scrubber
(the short one scored CLEAN outright, fingerprint fit 0.91 against a
self-baseline of 0.94) and were then rated 100% machine-written by the AI
detector a publishing platform runs on submissions. Every detector below was
written from a construction in those two posts, then measured against the
author's real corpus in three forms: short social posts (127 posts, 27,644
words), long-form professional documents (10 documents including a book,
94,354 words) and a novel manuscript (~100k words). The measured rates are in
the comment above each detector. The same DETECTORS.md rule applies: under
roughly 0.3 per 1,000 words and under 15% of pieces, the construction is the
generator's and a hard gate is safe; above that it is the author's and only a
rate gate against the author's own p95 is honest.

Why a second module rather than more functions in rhetoric.py: the first
family encodes one author's rejections (this repo's reference author) and its
hard flags were measured against his corpora. This module is calibrated with
its own file (`data/rhetoric_ext_calibration.json`) so nothing in the first
family changes.

A demotion mechanism exists here (`calibrate(..., demote=True)`) that would
gate a first-family flag at the author's own corpus rate when the corpus shows
the construction is theirs. It is OFF by default and was left off for the
author above: a large share of the short-post corpus was itself
machine-drafted, so the corpus rate for a reversal or a self-explanation is
partly the generator's rate, and the 4-of-8 real posts that fire
`unheeded-reversal` are not evidence that the author writes it. Hard on
presence. The same contamination inflates every other corpus-derived p95
(`dash_per1k_p95` read 42 on the short-post profile, so the scrubber treated
the author as an em-dash writer while their VOICE.md said none, ever).
`purity` below ranks the corpus pieces most likely to be the intruders.

Tells this module catches, with the specimen that motivated each:

  announcer          "Two frameworks hold it together." "One number sits under
                     all of it." "The argument runs in three movements."
                     "Inside, the argument travels." "Here is the map in one
                     breath."  A short sentence whose only job is to say what
                     the next stretch of prose is about to do.
  pointer-verdict    "That is precisely the point." "That is why it exists."
                     "and that image is the whole argument in miniature."
                     "That is the part the inevitability story conveniently
                     hides."  A sentence grading the previous one by pointing.
                     Extends rhetoric.find_declarative_ratings, whose pattern
                     needs "the part THAT" and missed "the part THE".
  participial-triad  "fear industrialised, silos automated, exhaustion
                     measured in real time."  Three noun+participle beats.
  colon-twist        ": distinct, formed by pressure and time, impossible to
                     replicate, and entirely capable of fading."  A colon list
                     that lands on an adverb-fronted reversal.
  parallel-address   "If you have been reading since June, thank you for
                     waiting. If you are new here, welcome."  Two audiences
                     addressed in matched conditionals.
  conversion         "Seventy sounded high to me the first time I read it.
                     Then I counted ... and I now read seventy as
                     conservative."  The narrator's staged change of mind.
  personification    "the question moved", "the old pattern broke", "the
                     argument travels", "one number sits under", "the book sat
                     quietly".  An abstraction doing a body's verb.
  tail-twist         "...into a world that was moving under our feet and
                     mostly telling us it was not."  A long sentence closing
                     on an adverb-fronted reversal clause.
  does-pair          "X does not choose between them. Y does."  A signature
                     line of the author's, reproduced verbatim by retrieval.
                     Once is theirs; the engine puts it in every piece.
  then-opener        Sentence-initial "Then" as a narrative pivot, three times
                     in one post. The author's long-form p95 is 0.12 per 100
                     sentences.
  intensifiers       precisely / entirely / quietly / mostly / genuinely /
                     exactly per 1,000 words. Post 10.4, long-form p95 2.2.
  generator-phrases  "at machine speed", "in real time", "in one breath",
                     "pull up a chair", "much rather", "what I call", "the
                     whole argument".  Zero in 220k words of the author's.

Added 2026-09-02 (second pass), the cheap and common social-platform tells,
all hard because the author's VOICE.md bans them and the target register has
none:

  negation-template  "The outcome is not fixed. It is chosen." "Not because
                     they have the smartest people."  The 'not A, it is B'
                     dramatic template and the dangling 'Not X. Y.' fragment,
                     both named in the author's VOICE.md as delete-on-sight.
  question-fragment  "The answers? ..." "Now? ..." "The real advantage now?"
                     A two-or-three-word noun-phrase question answered by the
                     next sentence. The author's own volley is full questions.
  emoji              any emoji in the body.
  hashtag-cluster    two or more hashtags.
  linkedin-phrases   "let that sink in", "read that again", "here's the
                     thing", "make no mistake", "the bottom line", "simply
                     put", "at its core", "unpopular opinion", "hot take" ...
  dash               any em-dash or en-dash, when the profile's calibration
                     sets dash_zero (the author's declared policy; see above
                     for why the corpus cannot be trusted to say so).

MEASURED ON THIS REPO'S REFERENCE AUTHOR, 2026-09-03, before wiring any of the
above into a hard gate. `measure` was run against the existing stores of
`creative` (116 pieces, 260,600 words), `research` (28 pieces, 17,305 words)
and `personal` (40 pieces, 39,280 words). Percentages are of pieces that would
HARD-flag, not of raw hits:

  detector            creative   research   personal   shipped
  announcer               0%         0%         0%     hard
  pointer-verdict         0%         0%         0%     hard
  colon-twist             0%         0%         0%     hard
  conversion              0%         0%         0%     hard
  emoji / hashtags        0%         0%         0%     hard
  personification (>=3)   0%         0%         0%     hard
  parallel-address        2%         0%         0%     hard
  participial-triad       3%         0%         2%     hard
  linkedin-phrase         0%         4%         0%     hard
  negation-template       9%         0%         5%     ADVISORY
  question-fragment       7%         0%         5%     ADVISORY
  then-opener             5%         0%         2%     ADVISORY

The three at the bottom fire on this author's real writing, so they ship
demoted (see DEFAULT_ADVISORY). The specimens are the reason, not the rate:
`negation-template` catches his fiction dialogue and his own notes to himself
("Not everyone has ability or talent.", "Not sure.", "Not really."), and
`question-fragment` catches ordinary spoken questions in the same dialogue
("It hurts?", "The next life?", "Getting comfy?"). `then-opener` is a rate gate
whose uncalibrated ceiling is 0.00 per 100 sentences, so it fires on any
uncalibrated profile whose author opens two sentences with "Then"; his measured
p95 is 1.76 (creative), 1.61 (personal), 0.00 (research).

Counted as whole pieces gaining a hard flag the first family did not already
raise, which is what a writer actually experiences: as shipped this family adds
one to 4.3% of his creative pieces, 3.6% of research and 2.5% of personal. With
the same three gating hard it would add one to 19.0% of the creative corpus.

That 19.0% is the reason for the split. rhetoric.py records rejecting and
rewriting a detector measured firing on 23.4% of this author's real corpus
rather than shipping it with a caveat, and a second family arriving at 19.0%
has reproduced the same mistake in a new file. Advisory is the weaker form of
the same discipline, for a construction worth reporting but not worth gating.

Three more things measured at the same time and deliberately NOT changed.
`intensifier_p95` defaults to 2.5 per 1kw against his measured p95 of 3.75
(creative), 4.15 (personal), 2.16 (research), so it over-reports on 11 of 116
creative and 11 of 40 personal pieces; it renders MEDIUM and never gates, so it
is recorded here rather than retuned. GENERATOR_PHRASES and LINKEDIN_PHRASES
were built from one other author's corpus and do hit his ("much rather", "pull
up a chair", "in real time", "here's the thing"); re-measure both lists before
extending either. And every rate above comes from stores built by the PRE-FIX
chunker, so a corpus containing a long book is measured short. Re-run `measure`
after any re-ingest.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")


def _sents(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip())
    return out


def _wc(s: str) -> int:
    return len(_WORD_RE.findall(s))


# --- 1. announcer --------------------------------------------------------------
#
# Measured: 0 hits in 127 short posts, 0 in 10 long-form documents, 0 in the
# novel. 4 hits in the 630-word generated post, 1 in the 300-word short
# version. The author's own structural moves name a stake rather than a table
# of contents, and do not match.
_ANNOUNCER_RE = re.compile(
    r"^(?:(?:One|Two|Three|Four|Five|Six)\s+\w+\s+(?:\w+\s+)?"
    r"(?:hold|holds|sit|sits|run|runs|carry|carries|frame|frames|anchor|anchors|underpin|underpins|tie|ties)\b"
    r"|(?:Inside|Here|Underneath|Beneath),?\s+the\s+(?:argument|book|story|idea|thesis|case)\s+"
    r"(?:travels|moves|runs|goes|turns|opens|unfolds)\b"
    r"|The\s+(?:argument|book|story|thesis|case|piece)\s+(?:runs|moves|travels|unfolds|comes|goes)\s+in\s+"
    r"(?:one|two|three|four|five|six)\s+\w+"
    r"|Here\s+is\s+the\s+\w+\s+in\s+one\s+(?:breath|line|sentence|paragraph|page))",
)


def find_announcers(text: str) -> list[str]:
    return [s for s in _sents(text) if _wc(s) <= 10 and _ANNOUNCER_RE.match(s)]


# --- 2. pointer verdict --------------------------------------------------------
#
# Measured: 0 in short posts, 0 in long-form, 1 in the novel (in dialogue).
# 3 in the generated long post. The v6 eval set adds "That is the
# part the inevitability story conveniently hides", which the first family's
# declarative-rating pattern missed because it requires "the part that".
_VERDICT_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:That|This|Which)\s+(?:is|was)\s+(?:precisely\s+|exactly\s+)?"
    r"(?:the\s+(?:point|whole\s+\w+|argument|answer)\b"
    r"|the\s+(?:part|thing|bit)\s+(?:the|that|which|this|\w+ing)\b"
    r"|why\s+(?:it|this|the\s+\w+|I|we)\s+(?:exists|matters|works|holds|is|wrote|built)\b)"
    r"|\b(?:is|was)\s+the\s+whole\s+(?:argument|point|story|book|case|question)"
    r"(?:\s+in\s+miniature|\s+about\s+\w+)?\b",
    re.IGNORECASE,
)


def find_pointer_verdicts(text: str) -> list[str]:
    return [m.group(0).strip() for m in _VERDICT_RE.finditer(text)]


# --- 3. participial triad ------------------------------------------------------
#
# Measured: 0 / 0 / 1 (the novel hit is a different shape caught by a loose
# pattern). Hard.
_PARTICIPIAL_TRIAD_RE = re.compile(
    r"\b[a-z]+\s+[a-z]+(?:ed|ised|ized)\s*,\s*[a-z]+\s+[a-z]+(?:ed|ised|ized)\s*,\s*(?:and\s+)?"
    r"[a-z]+\s+[a-z]+(?:ed|ised|ized)\b",
    re.IGNORECASE,
)


def find_participial_triads(text: str) -> list[str]:
    return [m.group(0) for m in _PARTICIPIAL_TRIAD_RE.finditer(text)]


# --- 4. colon twist ------------------------------------------------------------
#
# Measured: 0 / 0 / 0. Both posts carry the identical specimen. Hard.
_COLON_TWIST_RE = re.compile(
    r":\s*[^.:;!?]{0,140}?,\s*and\s+(?:entirely|utterly|wholly|quietly|completely|always|never|still|mostly|already|only)\s+"
    r"[^.!?]{2,40}[.!?]",
    re.IGNORECASE,
)


def find_colon_twists(text: str) -> list[str]:
    return [m.group(0).strip() for m in _COLON_TWIST_RE.finditer(text)]


# --- 5. parallel address -------------------------------------------------------
#
# Measured: 0 / 1 (a duplicated survey instruction line in a report) / 0. Hard.
def find_parallel_address(text: str) -> list[str]:
    s = _sents(text)
    out: list[str] = []
    for a, b in zip(s, s[1:]):
        if re.match(r"^If you\b", a) and re.match(r"^If you\b", b) and a != b:
            out.append(f"{a} {b}")
    return out


# --- 6. conversion narrative ---------------------------------------------------
#
# Measured: 0 / 0 / 0. Hard.
_CONV_SETUP_RE = re.compile(
    r"\b(?:sounded|seemed|felt|looked|struck me as)\s+\w+(?:\s+to\s+me)?\s+"
    r"(?:at\s+first|the\s+first\s+time|when\s+I\s+first)\b",
    re.IGNORECASE,
)
_CONV_TURN_RE = re.compile(r"^(?:Then\s+I\b|Now\s+I\b|I\s+now\b|And\s+I\s+now\b|Today\s+I\b)", re.IGNORECASE)


def find_conversions(text: str) -> list[str]:
    s = _sents(text)
    out: list[str] = []
    for i, a in enumerate(s):
        if _CONV_SETUP_RE.search(a):
            for b in s[i + 1:i + 3]:
                if _CONV_TURN_RE.match(b) or re.search(r"\band\s+I\s+now\b", b, re.I):
                    out.append(f"{a} {b}")
                    break
    return out


# --- 7. personification --------------------------------------------------------
#
# Measured: 0.18/1kw on short posts touching 2% of them, 0 in long-form, 0.02
# in the novel. 5 in the generated long post. The author does this once in a
# post at most; the generator does it five times. Hard at three or more in a
# piece, advisory below that.
_ABSTRACT_NOUNS = (
    r"question|argument|pattern|number|idea|book|title|answer|problem|word|sentence|story|"
    r"evidence|data|research|thesis|claim|case|map|figure|chapter|framework|principle|rule|"
    r"strategy|plan|logic|truth|premise|assumption|fingerprint|choice|season|silence"
)
_BODY_VERBS = (
    r"sits|sat|moves|moved|holds|held|travels|travelled|traveled|walks|walked|breaks|broke|"
    r"arrives|arrived|lands|landed|waits|waited|lives|lived|stands|stood|hides|hid|"
    r"swallowed|swallows|wears|wore|sleeps|slept|breathes|breathed|leans|leaned|rests|rested"
)
_PERSONIFY_RE = re.compile(
    rf"\b(?:the|this|that|one|every|each|its|my|our|a)\s+(?:\w+\s+)?(?:{_ABSTRACT_NOUNS})\s+(?:\w+\s+)?"
    rf"(?:{_BODY_VERBS})\b(?!\s+(?:for|on|in|by)\s+\w+ing)",
    re.IGNORECASE,
)
PERSONIFICATION_HARD_AT = 3


def find_personifications(text: str) -> list[str]:
    return [m.group(0) for m in _PERSONIFY_RE.finditer(text)]


# --- 8. tail twist -------------------------------------------------------------
#
# Measured: 0.036 / 0.011 / 0.039 per 1kw. The author's real hits are exactly
# the move, used rarely. Advisory.
_TAIL_TWIST_RE = re.compile(
    r",?\s+and\s+(?:mostly|quietly|largely|entirely|never|rarely|only|still|already|barely|utterly)\s+"
    r"[^,.!?]{2,45}[.!?]$",
    re.IGNORECASE,
)


def find_tail_twists(text: str) -> list[str]:
    return [s for s in _sents(text) if _wc(s) >= 14 and _TAIL_TWIST_RE.search(s)]


# --- 9. does-pair --------------------------------------------------------------
#
# "X does not choose between them. Y does." was a signature line of the
# author's (VOICE.md records it) and the retrieval anchors carry it, so the
# generator reproduces it as a closing move in piece after piece. Measured 0 in
# the corpus as a two-sentence pair. Advisory: once per piece is the author's,
# the second time in a month is the engine's.
def find_does_pairs(text: str) -> list[str]:
    s = _sents(text)
    out: list[str] = []
    for a, b in zip(s, s[1:]):
        if (re.search(r"\b(?:does|do)\s+not\s+\w+", a, re.I) and _wc(b) <= 4
                and re.search(r"\b(?:does|do|did|will|can)$", b.rstrip(".!?"))):
            out.append(f"{a} {b}")
    return out


# --- 10. then-opener rate ------------------------------------------------------
#
# Per 100 sentences. Measured p95: short posts 0.00, long-form 0.12, novel
# 1.27. Generated long post 6.25. Rate-gated against the profile's own p95.
def find_then_openers(text: str) -> list[str]:
    return [s for s in _sents(text) if re.match(r"^Then\b", s)]


def then_opener_rate(text: str) -> float:
    s = _sents(text)
    return 100.0 * len(find_then_openers(text)) / len(s) if s else 0.0


# --- 11. intensifier rate ------------------------------------------------------
#
# Per 1,000 words. Measured p95: short posts 7.14 (short pieces, noisy),
# long-form 2.24, novel 1.72. Generated long post 10.36, short version 6.78.
_INTENSIFIER_RE = re.compile(
    r"\b(?:precisely|exactly|entirely|quietly|mostly|genuinely|simply|merely|utterly|deeply|truly)\b",
    re.IGNORECASE,
)


def find_intensifiers(text: str) -> list[str]:
    return [m.group(0) for m in _INTENSIFIER_RE.finditer(text)]


def intensifier_rate(text: str) -> float:
    n = _wc(text)
    return 1000.0 * len(find_intensifiers(text)) / n if n else 0.0


# --- 12. generator phrases -----------------------------------------------------
#
# Each measured at 0.000 per 1kw across all three of the author's corpora and
# present in the generated post. Extend this list from every rejected draft; it
# is the
# cheapest detector in the file and the one most specific to the current
# generation of models.
GENERATOR_PHRASES = (
    "at machine speed", "in real time", "in one breath",
    "pull up a chair", "much rather", "what i call", "the whole argument",
    "in three movements", "a strange way to", "the walk starts", "held still",
    "wear it smooth", "swallowed the", "what books do",
)
# "in miniature" was on this list and came off it: the author used it twice in
# 27,644 words of posts. The list is only honest while every entry is at zero.
_GEN_PHRASE_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in GENERATOR_PHRASES) + r")\b", re.IGNORECASE)


def find_generator_phrases(text: str) -> list[str]:
    return [m.group(0) for m in _GEN_PHRASE_RE.finditer(text)]


# --- 13. fragment couplet (tracked, never gated) -------------------------------
#
# "One choice per chapter. One city at a time." Measured 0.69/1kw on short
# posts touching 9% of them: the author writes this shape and VOICE.md names it
# as a signature. Reported only when the rate exceeds the author's p95.
def find_fragment_couplets(text: str) -> list[str]:
    s = _sents(text)
    out: list[str] = []
    for a, b in zip(s, s[1:]):
        wa, wb = _WORD_RE.findall(a), _WORD_RE.findall(b)
        if 2 <= len(wa) <= 6 and 2 <= len(wb) <= 6 and wa[0].lower() == wb[0].lower():
            out.append(f"{a} {b}")
    return out


def couplet_rate(text: str) -> float:
    n = _wc(text)
    return 1000.0 * len(find_fragment_couplets(text)) / n if n else 0.0


# --- 14. negation template (hard) ------------------------------------------------
#
# Two shapes the author's VOICE.md names as anti-patterns to delete on sight:
#   "X is not A. It is B."      ("The outcome is not fixed. It is chosen.")
#   "Not A."  as a fragment     ("Not because they have the smartest people.")
# Measured 6 hits in 4 of 127 posts and 2 fragments in 2 posts. Those posts are
# suspected machine drafts and are not evidence of the author, so hard on
# presence.
_IS_NOT_IT_IS_RE = re.compile(
    r"\b(?:is|are|was|were|isn't|aren't|wasn't|weren't)\s+(?:not\s+)?(?:just\s+|only\s+|about\s+|merely\s+)?"
    r"[^.!?]{1,60}[.!?]\s+(?:It|That|This|They)\s*(?:'s|'re|\s+is|\s+are|\s+was|\s+were)\b",
    re.IGNORECASE,
)


def find_negation_templates(text: str) -> list[str]:
    out: list[str] = []
    for m in _IS_NOT_IT_IS_RE.finditer(text):
        span = m.group(0)
        first = re.split(r"[.!?]", span, maxsplit=1)[0]
        if re.search(r"\bnot\b|n't\b", first, re.I):
            out.append(span.strip())
    for s in _sents(text):
        if re.match(r"^Not\s+\w", s) and _wc(s) <= 8:
            out.append(s)
    return out


# --- 15. question fragment (hard) -----------------------------------------------
#
# "The answers?" "Now?" "The real advantage now?" "Shut down?" followed by the
# answer. A noun-phrase question of one to three words. The author's own move
# is a stacked volley of full questions, which starts with a wh-word and does
# not match.
_Q_OPENERS_EXCLUDED = re.compile(
    r"^(?:What|Who|Whom|Whose|Where|When|Why|How|Which|Is|Are|Was|Were|Do|Does|Did|Can|Could|"
    r"Will|Would|Should|Shall|Have|Has|Had|Am)\b", re.IGNORECASE,
)


def find_question_fragments(text: str) -> list[str]:
    s = _sents(text)
    out: list[str] = []
    for a, b in zip(s, s[1:]):
        if a.endswith("?") and 1 <= _wc(a) <= 3 and not b.endswith("?") and not _Q_OPENERS_EXCLUDED.match(a):
            out.append(f"{a} {b[:60]}")
    return out


# --- 16. emoji and hashtags (hard) ----------------------------------------------
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⬆✅❌✨‼⁉〰〽㊗㊙]")
_HASHTAG_RE = re.compile(r"(?:^|\s)#\w{2,}")


def find_emoji(text: str) -> list[str]:
    return _EMOJI_RE.findall(text)


def find_hashtags(text: str) -> list[str]:
    return [m.strip() for m in _HASHTAG_RE.findall(text)]


# --- 17. LinkedIn phrases (hard) -------------------------------------------------
#
# The platform's machine dialect. Each at zero in the reference corpus except
# where noted.
LINKEDIN_PHRASES = (
    "let that sink in", "read that again", "here's the thing", "here is the thing",
    "the uncomfortable truth", "the hard truth", "the harsh truth", "make no mistake",
    "let's be honest", "let's be real", "bottom line:", "put simply", "at its core",
    "the takeaway", "key takeaway", "food for thought", "plot twist", "spoiler alert",
    "hot take", "unpopular opinion", "this is your sign", "say it louder",
    "i'll say it again", "nobody talks about", "no one talks about",
    "no one is talking about", "let's talk about", "full stop.", "and honestly,",
    "here's why", "here's how", "here's what", "the answer is simple", "it's that simple",
    "that's the whole point", "and that's okay", "and that's the point", "game changer",
    "game-changer", "in today's fast-paced",
)
# Came off the list after measuring the author's long-form PDFs (2017 to 2024,
# written before the AI era): "the bottom line" (2), "the future of work is"
# (2), "in a world where" (2), "simply put" (1). The author's own, so not
# gated.
_LINKEDIN_RE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in LINKEDIN_PHRASES) + r")", re.IGNORECASE)


def find_linkedin_phrases(text: str) -> list[str]:
    return [m.group(0) for m in _LINKEDIN_RE.finditer(text)]


# --- 18. dashes under a zero policy (hard when dash_zero) ------------------------
#
# A line made only of dashes is a horizontal rule, not punctuation (social
# posts often separate the essay from the call to action with one), and counting it made
# every post in the corpus read as dashed. Rules are removed first, the same
# way scrub.py does it. A double hyphen counts only between word characters.
_RULE_LINE_RE = re.compile(r"(?m)^[ \t]*[-—–=_*]{3,}[ \t]*$")
# A YAML front-matter block flattened into prose ("type: post --- If talent
# was never...") is a separator, not a dash; a whole exported post corpus
# was once ingested with the header inline (see ingest._extract_text). Only a bare
# double hyphen between words counts, never a run of three or more.
_FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n.*?\n---\s*", re.DOTALL)
_INLINE_META_RE = re.compile(r"^\s*(?:date|url|likes|comments|type|title):[^\n]*?\btype:\s*\w+\s*---\s*", re.DOTALL)
_DASH_RE = re.compile(r"[—–]|(?<=\w)\s?--(?!-)\s?(?=\w)")


def find_dashes(text: str) -> list[str]:
    t = _FRONT_MATTER_RE.sub("", text, count=1)
    t = _INLINE_META_RE.sub("", t, count=1)
    return [m.group(0) for m in _DASH_RE.finditer(_RULE_LINE_RE.sub("", t))]


# --- calibration -----------------------------------------------------------------

# First-family hard flags that COULD be rate-gated when a clean corpus shows the
# construction is the author's. Off by default; see the module docstring.
DEMOTABLE = ("unheeded-reversal", "self-explanation")

# Second-family hard flags that report but do not gate unless a profile says
# otherwise. These are the three measured firing on this repo's reference author
# in the table above; that table is the justification, and re-measuring is the
# way to change this list, not a judgement about which construction sounds AI.
# A profile makes one of them hard again with
# `calibrate <voice> --hard negation-template,question-fragment`, which is what
# a corpus known to contain none of them wants: the second author's profiles
# this module was built for gate all three hard on presence, on the reasoning in
# the module docstring, and their calibration must pass --hard to keep doing so.
DEFAULT_ADVISORY = ("negation-template", "question-fragment", "then-opener")


@dataclass
class RhetoricExtCalibration:
    then_p95: float = 0.0          # per 100 sentences
    intensifier_p95: float = 2.5   # per 1,000 words
    couplet_p95: float = 0.0       # per 1,000 words
    # name -> the author's own p95 rate per 1,000 words. Present means demoted.
    # Empty unless calibrate(..., demote=True) was run on a corpus known clean.
    demote: dict[str, float] = field(default_factory=dict)
    # The author's declared policy: no em-dashes or en-dashes at all. Set from
    # the person, not the corpus, because the corpus can lie (see docstring).
    dash_zero: bool = False
    # Hard flags switched off for this profile: not gated, and not reported.
    # Fiction needs "negation-template" and "question-fragment" off: the novel
    # manuscript carries 148 "X is not A. It is B." pairs in dialogue, which is
    # a character talking, not a tell.
    disabled: list[str] = field(default_factory=list)
    # Hard flags demoted to advisory for this profile: still reported, at
    # MEDIUM, but not gated. The setting between hard and disabled, for a
    # construction the author demonstrably writes at a low rate. DEFAULT_ADVISORY
    # holds the measured defaults and the reason each one is there.
    advisory: list[str] = field(default_factory=lambda: list(DEFAULT_ADVISORY))
    n_pieces: int = 0

    def to_dict(self) -> dict:
        return {"then_p95": self.then_p95, "intensifier_p95": self.intensifier_p95,
                "couplet_p95": self.couplet_p95, "demote": dict(self.demote),
                "dash_zero": self.dash_zero, "disabled": list(self.disabled),
                "advisory": list(self.advisory), "n_pieces": self.n_pieces}

    @classmethod
    def from_dict(cls, d: dict) -> "RhetoricExtCalibration":
        return cls(then_p95=float(d.get("then_p95", 0.0)),
                   intensifier_p95=float(d.get("intensifier_p95", 2.5)),
                   couplet_p95=float(d.get("couplet_p95", 0.0)),
                   demote={k: float(v) for k, v in (d.get("demote") or {}).items()},
                   dash_zero=bool(d.get("dash_zero", False)),
                   disabled=[str(x) for x in (d.get("disabled") or [])],
                   # An absent key means a file written before advisory existed.
                   # Fall back to the measured default rather than to hard-on-
                   # presence, so an old calibration cannot silently gate a
                   # construction nobody measured on that author.
                   advisory=([str(x) for x in (d["advisory"] or [])]
                             if "advisory" in d else list(DEFAULT_ADVISORY)),
                   n_pieces=int(d.get("n_pieces", 0)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RhetoricExtCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "RhetoricExtCalibration":
        if path and Path(path).exists():
            return cls.load(path)
        return cls()


FILENAME = "rhetoric_ext_calibration.json"


def load_for(prof) -> RhetoricExtCalibration:
    """The profile's second-family calibration, or the uncalibrated default
    (no demotions, generic rate ceilings) when `calibrate` has not been run."""
    try:
        return RhetoricExtCalibration.load_or_default(Path(prof.scrub_path).parent / FILENAME)
    except Exception:
        return RhetoricExtCalibration()


def _p95(vals: list[float], floor: float = 0.0) -> float:
    if not vals:
        return floor
    vals = sorted(vals)
    return max(vals[int(0.95 * (len(vals) - 1))], floor)


def calibrate(texts: list[str], min_words: int = 150, demote: bool = False,
              dash_zero: bool = False,
              advisory: list[str] | None = None) -> RhetoricExtCalibration:
    """Author's own p95 for the rate-gated checks, and optionally the demotion table.

    ``demote`` measures the two first-family hard flags on the corpus and
    demotes each one whose rate is above 0.3 per 1kw or which touches more
    than 15% of pieces, the DETECTORS.md rule. Off by default: it is only
    honest on a corpus known to contain nothing machine-written, and on a
    contaminated corpus it licenses the generator's habits as the author's.

    ``dash_zero`` records the author's declared no-dash policy so `dash` gates
    hard regardless of what the corpus dash rate says.

    ``advisory`` is the list of second-family hard flags to demote to reporting
    only; it defaults to DEFAULT_ADVISORY rather than to empty, because a
    calibration run measures rates and does not read specimens, and the three
    defaults were set from specimens. Pass ``[]`` to gate all of them hard on a
    corpus known to contain none.
    """
    from . import rhetoric
    usable = [t for t in texts if _wc(t) >= min_words]
    cal = RhetoricExtCalibration(
        then_p95=_p95([then_opener_rate(t) for t in usable]),
        intensifier_p95=_p95([intensifier_rate(t) for t in usable], floor=2.0),
        couplet_p95=_p95([couplet_rate(t) for t in usable]),
        dash_zero=dash_zero,
        advisory=list(DEFAULT_ADVISORY if advisory is None else advisory),
        n_pieces=len(usable),
    )
    if demote and usable:
        words = sum(_wc(t) for t in usable)
        for name, fn in (("unheeded-reversal", rhetoric.find_reversals),
                         ("self-explanation", rhetoric.find_self_explanations)):
            hits = [len(fn(t)) for t in usable]
            rate = 1000.0 * sum(hits) / words if words else 0.0
            touched = sum(1 for h in hits if h) / len(usable)
            if rate > 0.3 or touched > 0.15:
                per_piece = [1000.0 * h / _wc(t) for h, t in zip(hits, usable)]
                cal.demote[name] = _p95(per_piece, floor=0.5)
    return cal


# --- report ---------------------------------------------------------------------


@dataclass
class RhetoricExtReport:
    announcers: list[str] = field(default_factory=list)
    pointer_verdicts: list[str] = field(default_factory=list)
    participial_triads: list[str] = field(default_factory=list)
    colon_twists: list[str] = field(default_factory=list)
    parallel_address: list[str] = field(default_factory=list)
    conversions: list[str] = field(default_factory=list)
    personifications: list[str] = field(default_factory=list)
    tail_twists: list[str] = field(default_factory=list)
    does_pairs: list[str] = field(default_factory=list)
    then_openers: list[str] = field(default_factory=list)
    then_rate: float = 0.0
    then_p95: float = 0.0
    intensifiers: list[str] = field(default_factory=list)
    intensifier_rate: float = 0.0
    intensifier_p95: float = 0.0
    generator_phrases: list[str] = field(default_factory=list)
    couplets: list[str] = field(default_factory=list)
    couplet_rate: float = 0.0
    couplet_p95: float = 0.0
    negation_templates: list[str] = field(default_factory=list)
    question_fragments: list[str] = field(default_factory=list)
    emoji: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    linkedin_phrases: list[str] = field(default_factory=list)
    dashes: list[str] = field(default_factory=list)
    dash_zero: bool = False
    disabled: list[str] = field(default_factory=list)
    # Second-family hard flags demoted to reporting-only for this profile.
    advisory: list[str] = field(default_factory=list)
    # First-family hard flags moved to advisory because the author uses the
    # construction at a measured rate. name -> (spans, rate, p95)
    demoted: dict[str, tuple[list[str], float, float]] = field(default_factory=dict)
    word_count: int = 0

    def allowed(self, p95_per_1kw: float) -> int:
        """How many instances the author's own p95 licenses in a text this
        long, never fewer than one. A rate gate alone is wrong on short pieces:
        one instance in a 280-word post computes to 3.6 per 1kw, above a p95
        of 2.9, and flagged three of eight of the author's real posts. The author writes the
        construction once; the engine writes it three times."""
        return max(1, int(p95_per_1kw * self.word_count / 1000.0))

    @property
    def then_over(self) -> bool:
        return self.then_rate > self.then_p95

    @property
    def intensifier_over(self) -> bool:
        return self.intensifier_rate > self.intensifier_p95

    @property
    def couplet_over(self) -> bool:
        return self.couplet_p95 > 0 and self.couplet_rate > self.couplet_p95

    @property
    def _fired(self) -> list[str]:
        """Every gating detector that found something, before this profile's
        advisory demotions are applied. `disabled` is honoured here because a
        disabled detector is off, not demoted."""
        flags: list[str] = []
        if self.announcers:
            flags.append("announcer")
        if self.pointer_verdicts:
            flags.append("pointer-verdict")
        if self.participial_triads:
            flags.append("participial-triad")
        if self.colon_twists:
            flags.append("colon-twist")
        if self.parallel_address:
            flags.append("parallel-address")
        if self.conversions:
            flags.append("conversion")
        if len(self.personifications) >= PERSONIFICATION_HARD_AT:
            flags.append("personification")
        if self.negation_templates:
            flags.append("negation-template")
        if self.question_fragments:
            flags.append("question-fragment")
        if self.emoji:
            flags.append("emoji")
        if len(self.hashtags) >= 2:
            flags.append("hashtag-cluster")
        if self.linkedin_phrases:
            flags.append("linkedin-phrase")
        if self.dash_zero and self.dashes:
            flags.append("dash")
        # Rate-gated AND at least two instances: one "Then" in a short post is
        # inside the author's habit even where the corpus p95 rounds to zero.
        if self.then_over and len(self.then_openers) >= 2:
            flags.append("then-opener")
        # A demoted construction is still hard when it exceeds what the author's
        # own p95 licenses for a text this long.
        for name, (spans, rate, p95) in self.demoted.items():
            if len(spans) > self.allowed(p95):
                flags.append(name)
        return [f for f in flags if f not in self.disabled]

    @property
    def hard_flags(self) -> list[str]:
        """What actually gates: fired, minus this profile's advisory demotions."""
        return [f for f in self._fired if f not in self.advisory]

    @property
    def advisory_flags(self) -> list[str]:
        """Detectors that fired and would gate on another profile, demoted to
        reporting here. Named so a caller can see what was demoted instead of
        inferring it from a MEDIUM line."""
        return [f for f in self._fired if f in self.advisory]

    @property
    def is_clean(self) -> bool:
        # advisory_flags is included on purpose: demoting a detector stops it
        # gating, and must not also make a draft carrying it read as clean.
        return not (self.hard_flags or self.advisory_flags
                    or self.personifications or self.tail_twists
                    or self.does_pairs or self.intensifier_over
                    or self.generator_phrases or self.couplet_over
                    or self.hashtags
                    or any(v[0] for v in self.demoted.values()))


def analyze(text: str, cal: RhetoricExtCalibration | None = None,
            first_family: "object | None" = None) -> RhetoricExtReport:
    """Run the second family. Pass the first family's RhetoricReport as
    ``first_family`` to apply the calibrated demotions: the demoted spans are
    moved off that report (so scrub's hard_flags stop seeing them) and carried
    here with the author's rate for advisory rendering."""
    cal = cal or RhetoricExtCalibration()
    rep = RhetoricExtReport(
        announcers=find_announcers(text),
        pointer_verdicts=find_pointer_verdicts(text),
        participial_triads=find_participial_triads(text),
        colon_twists=find_colon_twists(text),
        parallel_address=find_parallel_address(text),
        conversions=find_conversions(text),
        personifications=find_personifications(text),
        tail_twists=find_tail_twists(text),
        does_pairs=find_does_pairs(text),
        then_openers=find_then_openers(text),
        then_rate=then_opener_rate(text),
        then_p95=cal.then_p95,
        intensifiers=find_intensifiers(text),
        intensifier_rate=intensifier_rate(text),
        intensifier_p95=cal.intensifier_p95,
        generator_phrases=find_generator_phrases(text),
        couplets=find_fragment_couplets(text),
        couplet_rate=couplet_rate(text),
        couplet_p95=cal.couplet_p95,
        negation_templates=find_negation_templates(text),
        question_fragments=find_question_fragments(text),
        emoji=find_emoji(text),
        hashtags=find_hashtags(text),
        linkedin_phrases=find_linkedin_phrases(text),
        dashes=find_dashes(text),
        dash_zero=cal.dash_zero,
        disabled=list(cal.disabled),
        advisory=list(cal.advisory),
        word_count=_wc(text),
    )
    if first_family is not None and cal.demote:
        n = _wc(text) or 1
        attr = {"unheeded-reversal": "reversals", "self-explanation": "self_explanations"}
        for name, p95 in cal.demote.items():
            a = attr.get(name)
            spans = list(getattr(first_family, a, []) or [])
            if a and spans:
                rate = 1000.0 * len(spans) / n
                rep.demoted[name] = (spans, rate, p95)
                setattr(first_family, a, [])
    return rep


def level_for(rep: RhetoricExtReport, name: str) -> str | None:
    """Severity for one gating detector on this profile, or None when it is off.

    HIGH gates. MEDIUM means the detector is in the profile's `advisory` list:
    the construction was measured in this author's own corpus, so it is worth
    reporting and not worth failing a draft over. None means `disabled`, which
    suppresses the line entirely -- before this, a disabled detector dropped out
    of hard_flags but still rendered its [HIGH] line, so a switched-off check
    still read as a blocking one.
    """
    if name in rep.disabled:
        return None
    return "MEDIUM" if name in rep.advisory else "HIGH"


_ADVISORY_NOTE = (" Advisory here: this construction is measured in the author's own "
                  "corpus, so it is reported rather than gated.")


def render_lines(rep: RhetoricExtReport, author: str = "the author") -> list[str]:
    out: list[str] = []

    def add(name: str, body: str) -> None:
        level = level_for(rep, name)
        if level is None:
            return
        out.append(f"- [{level}] {body}" + (_ADVISORY_NOTE if level == "MEDIUM" else ""))

    if rep.dash_zero and rep.dashes:
        add("dash",
            f"Dashes: {len(rep.dashes)} em/en-dash(es). {author}'s policy is zero, "
            f"whatever the corpus rate says. Replace with a comma, a colon, or a full stop.")
    if rep.emoji:
        add("emoji",
            f"Emoji ({len(rep.emoji)}): {author}'s current register has none. Remove all.")
    if len(rep.hashtags) >= 2:
        add("hashtag-cluster", f"Hashtag cluster: {' '.join(rep.hashtags[:6])}. Remove.")
    for s in rep.linkedin_phrases:
        add("linkedin-phrase",
            f"LinkedIn phrase: \"{s}\". The platform's machine dialect; {author} does not "
            f"write it. Cut the sentence or say the thing plainly.")
    for s in rep.negation_templates:
        add("negation-template",
            f"Negation template: \"{s[:100]}\". \"X is not A. It is B.\" and dangling "
            f"\"Not A.\" fragments are anti-patterns in {author}'s own VOICE.md. Say the thing "
            f"once, as one sentence.")
    for s in rep.question_fragments:
        add("question-fragment",
            f"Question fragment: \"{s[:90]}\". A two-word question answered by the next "
            f"sentence. {author}'s questions are whole questions, stacked. Cut the fragment.")
    for s in rep.announcers:
        add("announcer",
            f"Announcer: \"{s}\". A sentence whose only job is to say what the "
            f"next stretch will do. Measured 0 times in {author}'s corpus. Delete it and "
            f"let the next sentence start.")
    for s in rep.pointer_verdicts:
        add("pointer-verdict",
            f"Pointer verdict: \"{s}\". Grades the previous sentence by pointing "
            f"at it. Measured 0 times in {author}'s corpus. Delete; the previous sentence "
            f"carries its own weight or it does not.")
    for s in rep.participial_triads:
        add("participial-triad",
            f"Participial triad: \"{s}\". Three noun-plus-participle beats in a "
            f"row is the generator's crescendo, not {author}'s. Keep one, make it a sentence.")
    for s in rep.colon_twists:
        add("colon-twist",
            f"Colon twist: \"{s[:110]}\". A colon list closing on an adverb-fronted "
            f"reversal. Cut the last item or end the list flat.")
    for s in rep.parallel_address:
        add("parallel-address",
            f"Parallel address: \"{s[:110]}\". Two audiences in matched "
            f"conditionals. Address one reader.")
    for s in rep.conversions:
        add("conversion",
            f"Conversion narrative: \"{s[:110]}\". The staged change of mind "
            f"(\"sounded X the first time... I now read it as Y\"). State the position.")
    if rep.personifications and level_for(rep, "personification") is not None:
        # Two reasons this can be MEDIUM: under the count that gates, or demoted
        # for this profile. Only the second earns the advisory note.
        demoted = "personification" in rep.advisory
        level = "HIGH" if len(rep.personifications) >= PERSONIFICATION_HARD_AT and not demoted else "MEDIUM"
        out.append(
            f"- [{level}] Abstractions doing a body's verb ({len(rep.personifications)}): "
            f"{'; '.join(repr(x) for x in rep.personifications[:5])}. {author} does this "
            f"at most once in a piece. Give the verb back to a person or cut it."
            + (_ADVISORY_NOTE if demoted else ""))
    for s in rep.tail_twists:
        out.append(
            f"- [MEDIUM] Tail twist: \"...{s[-90:]}\". A long sentence closing on "
            f"\"and mostly/never/entirely ...\". End on the fact.")
    for s in rep.does_pairs:
        out.append(
            f"- [MEDIUM] Recycled creed: \"{s[:110]}\". {author}'s own line, reproduced by "
            f"retrieval. Once is the author's; do not close every piece on it.")
    if rep.then_over and len(rep.then_openers) >= 2:
        add("then-opener",
            f"\"Then\" openers: {rep.then_rate:.1f} per 100 sentences vs {author}'s "
            f"p95 {rep.then_p95:.1f}. e.g. \"{rep.then_openers[0][:70]}\". Narrative pivots "
            f"the corpus does not make. Join the sentence to its neighbour or cut \"Then\".")
    if rep.intensifier_over:
        out.append(
            f"- [MEDIUM] Intensifiers: {rep.intensifier_rate:.1f} per 1000 words vs {author}'s "
            f"p95 {rep.intensifier_p95:.1f} ({', '.join(rep.intensifiers[:6])}). "
            f"Each one is the generator emphasising for the author. Cut all but one.")
    if rep.generator_phrases:
        out.append(
            f"- [MEDIUM] Generator phrases: {', '.join(repr(x) for x in rep.generator_phrases)}. "
            f"Each measured 0 times in {author}'s corpus. Rephrase.")
    if rep.couplet_over:
        out.append(
            f"- [LOW] Fragment couplets: {rep.couplet_rate:.1f} per 1000 words vs {author}'s "
            f"p95 {rep.couplet_p95:.1f}. The author's when it lands once; a tic when it recurs.")
    for name, (spans, rate, p95) in rep.demoted.items():
        if not spans:
            continue
        over = len(spans) > rep.allowed(p95)
        level = "HIGH" if over else "MEDIUM"
        out.append(
            f"- [{level}] {name}: {len(spans)} in {rep.word_count} words, {author}'s own p95 "
            f"({p95:.2f}/1kw) licenses {rep.allowed(p95)} here: "
            f"{'; '.join(repr(x[:80]) for x in spans[:3])}. {author} writes this construction, "
            f"so it is rate-gated, not banned. "
            f"{'Cut down to that rate.' if over else 'Under that rate; worth a look, not gated.'}")
    return out


# Detector names in `measure` below that can raise a hard flag, as opposed to
# the ones that only ever render at MEDIUM or LOW. Keeps the measure table
# honest about which rates are a gate's false-positive rate and which are noise.
_GATING = frozenset({
    "announcer", "pointer-verdict", "participial-triad", "colon-twist",
    "parallel-address", "conversion", "personification", "negation-template",
    "question-fragment", "emoji", "hashtags", "linkedin-phrases",
})
# `dashes` is the exception: it gates only where the profile has declared
# dash_zero, so its corpus rate is a false-positive rate only under that policy
# and is plain description otherwise. Reporting it as a hard detector on a
# profile without the policy warns about a check that cannot fire.


# --- CLI ----------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse
    from . import config, ingest
    ap = argparse.ArgumentParser(prog="python -m mimesis_voice.rhetoric_ext")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="measure the author's own rates and write data/" + FILENAME)
    c.add_argument("voice")
    c.add_argument("--demote", action="store_true",
                   help="rate-gate first-family flags the corpus shows are the author's (clean corpora only)")
    c.add_argument("--dash-zero", action="store_true", help="the author's policy is no dashes at all")
    c.add_argument("--disable", default="", help="comma-separated hard flags to switch off for this profile")
    c.add_argument("--hard", default="",
                   help="comma-separated flags to gate hard that are advisory by default "
                        f"(defaults: {','.join(DEFAULT_ADVISORY)}); use on a corpus measured "
                        "to contain none of them")
    k = sub.add_parser("check", help="run the second family on a file")
    k.add_argument("voice")
    k.add_argument("file")
    m = sub.add_parser("measure", help="false-positive rates of every detector on the author's corpus")
    m.add_argument("voice")
    u = sub.add_parser("purity", help="rank corpus pieces by how machine-written they look")
    u.add_argument("voice")
    u.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    prof = config.resolve_named(a.voice)
    if prof is None:
        print(f"no such voice: {a.voice}")
        return 2
    out_path = prof.scrub_path.parent / FILENAME
    if a.cmd == "calibrate":
        texts = [t for t in ingest.read_pieces(prof.db_path).values() if t.strip()]
        forced_hard = {x.strip() for x in a.hard.split(",") if x.strip()}
        cal = calibrate(texts, demote=a.demote, dash_zero=a.dash_zero,
                        advisory=[f for f in DEFAULT_ADVISORY if f not in forced_hard])
        cal.disabled = [x.strip() for x in a.disable.split(",") if x.strip()]
        cal.save(out_path)
        print(f"wrote {out_path}")
        print(json.dumps(cal.to_dict(), indent=2))
        return 0
    if a.cmd == "purity":
        from . import rhetoric
        rows = []
        for fn, t in ingest.read_pieces(prof.db_path).items():
            if not t.strip():
                continue
            ff = rhetoric.analyze(t)
            # advisory=[] and _fired, not hard_flags: purity is a diagnostic
            # ranking of which corpus pieces look machine-written, not a gate,
            # so a detector demoted for this author still counts as evidence
            # here. Demoting it is a statement about failing drafts, not about
            # which pieces to suspect.
            ex = analyze(t, RhetoricExtCalibration(dash_zero=True, advisory=[]))
            flags = ff.hard_flags + ex._fired
            n_dash = len(ex.dashes)
            score = len(flags) + (2 if ex.emoji else 0) + (1 if n_dash else 0)
            rows.append((score, fn, _wc(t), n_dash, len(ex.emoji), len(ex.hashtags), flags))
        rows.sort(key=lambda r: (-r[0], r[1]))
        print(f"{prof.slug}: {len(rows)} pieces. Score = hard flags (both families, hard on presence) "
              f"+2 for emoji +1 for dashes. Read the top of the list as candidates for exclusion, not verdicts.")
        print(f"{'score':>5s} {'piece':40s} {'words':>5s} {'dash':>4s} {'emoji':>5s} {'tags':>4s}  flags")
        for score, fn, wc, nd, ne, nh, flags in rows[:a.top]:
            print(f"{score:5d} {fn[:40]:40s} {wc:5d} {nd:4d} {ne:5d} {nh:4d}  {', '.join(flags)}")
        clean = sum(1 for r in rows if r[0] == 0)
        print(f"\n{clean} of {len(rows)} pieces score 0.")
        return 0
    if a.cmd == "check":
        cal = RhetoricExtCalibration.load_or_default(out_path)
        text = Path(a.file).read_text(encoding="utf-8")
        rep = analyze(text, cal)
        lines = render_lines(rep, prof.name)
        print("\n".join(lines) if lines else "second family: clean")
        print("hard:", rep.hard_flags)
        return 0
    if a.cmd == "measure":
        texts = [t for t in ingest.read_pieces(prof.db_path).values() if t.strip()]
        words = sum(_wc(t) for t in texts) or 1
        fns = {
            "announcer": find_announcers, "pointer-verdict": find_pointer_verdicts,
            "participial-triad": find_participial_triads, "colon-twist": find_colon_twists,
            "parallel-address": find_parallel_address, "conversion": find_conversions,
            "personification": find_personifications, "tail-twist": find_tail_twists,
            "does-pair": find_does_pairs, "generator-phrases": find_generator_phrases,
            "fragment-couplet": find_fragment_couplets,
            "negation-template": find_negation_templates, "question-fragment": find_question_fragments,
            "emoji": find_emoji, "hashtags": find_hashtags, "linkedin-phrases": find_linkedin_phrases,
            "dashes": find_dashes,
        }
        cal = RhetoricExtCalibration.load_or_default(out_path)
        print(f"{prof.slug}: {len(texts)} pieces, {words} words"
              f"{'' if out_path.exists() else '  (no calibration file; uncalibrated defaults)'}")
        # A rate here is a FALSE-POSITIVE rate: these are the author's own
        # pieces, so anything a gating detector finds is something a hard flag
        # would reject in their real writing. The DETECTORS.md rule is under
        # 0.3/1kw and under 15% of pieces before a hard gate is honest.
        # The percentage is pieces with ANY hit, which is an upper bound on the
        # pieces that would actually gate: `personification` needs
        # PERSONIFICATION_HARD_AT in one piece and `hashtags` needs two.
        for name, fn in fns.items():
            hits = sum(len(fn(t)) for t in texts)
            touched = sum(1 for t in texts if fn(t))
            rate, pct = 1000 * hits / words, 100 * touched / len(texts)
            if name in cal.disabled:
                gate = "off"
            elif name in cal.advisory:
                gate = "ADVISORY"
            elif name == "dashes":
                gate = "hard" if cal.dash_zero else "not gated (no dash_zero)"
            else:
                gate = "hard" if name in _GATING else "advisory-only"
            warn = "  <-- over the rule" if gate == "hard" and (rate > 0.3 or pct > 15) else ""
            print(f"  {name:20s} {rate:6.3f}/1kw  {pct:3.0f}% of pieces  [{gate}]{warn}")
        print(f"  then-opener p95 {_p95([then_opener_rate(t) for t in texts]):.2f}/100 sents; "
              f"intensifier p95 {_p95([intensifier_rate(t) for t in texts]):.2f}/1kw")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
