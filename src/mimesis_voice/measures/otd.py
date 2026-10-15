"""Opener-class distance: what every sentence in the passage starts as.

Surface first words are replaced by a fronting taxonomy (Biber 1988 dimensions
cut down to what a POS tagger labels reliably): QUOTE, CONJ, DISC (discourse
marker), SUBORD, PP (preposition-fronted), ADV, PART (participial), PRON, NP
(determiner / noun / proper / adjective / number), OTHER. The passage's class
distribution is compared to the author's reference distribution by the
Jensen-Shannon distance, base 2, so 0 is identical and 1 is disjoint.

This is what catches "every sentence starts with The or They" and the
synopsis-mode paragraph:

    Cray sat... He had woken... He had struck... Father Brown listened...

Chesterton's base model opened 40% of its sentences on a noun phrase against
his own 21%; the adapters ran 17-19%. Replicate floor 0.005, adapter effect
0.074, and the class order reproduced Patrick's blind style ranking on item 1
at rho +0.80. It correlates only 0.46 with the package's existing opener
monotony, so both are kept.

The tagger sees the first six words of a sentence, not the first token alone,
and the class often turns on what follows:

    "Left alone, she counted the tiles."      PART
    "Left hands are rarer than right ones."   NP
    "Past the church the road narrows."       PP
    "Past mistakes are not a plan."           NP

A word list cannot make those calls, which is why this measure needs spaCy.
Without spaCy and `en_core_web_sm` `available()` is False and the caller reports
the line as unavailable rather than substituting a cheaper classifier, which
would measure something the study never validated.

The taxonomy is not a linguistic claim and is not always right: a gerund subject
("Walking is good for the knees") tags VBG and lands in PART with a participial
opener. It is kept exactly as the study wrote it, because the distance is to the
author's own distribution under the same classifier, and a consistent mistake on
both sides cancels.
"""
from __future__ import annotations

import math

from .textbase import sentences

__all__ = ["CLASSES", "available", "js_distance", "opener_class", "profile", "unavailable_reason"]

CLASSES = ["QUOTE", "CONJ", "DISC", "SUBORD", "PP", "ADV", "PART", "PRON", "NP", "OTHER"]

DISC = set("""however then now still yet so meanwhile anyway besides instead nevertheless nonetheless
otherwise moreover indeed finally first second third next later afterwards perhaps maybe also again
thus hence therefore well today tonight yesterday tomorrow once sometimes suddenly""".split())
SUB = set("""when if although though because while as since until unless whereas whenever wherever
before after once whether even""".split())

_NLP = None
_LOAD_ERROR: str | None = None


def _nlp():
    """The tagger, loaded once. Parser, NER and lemmatizer are disabled: the
    measure reads `pos_` and `tag_` only, and the full pipeline is roughly ten
    times slower for no change in output."""
    global _NLP, _LOAD_ERROR
    if _NLP is None and _LOAD_ERROR is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        except Exception as exc:
            _LOAD_ERROR = str(exc)
    return _NLP


def available() -> bool:
    return _nlp() is not None


def unavailable_reason() -> str:
    _nlp()
    return (
        f"spaCy en_core_web_sm not loadable ({_LOAD_ERROR}). "
        "Install with: pip install 'mimesis-voice[structure]' && "
        "python -m spacy download en_core_web_sm"
    )


def opener_class(sent: str) -> str:
    s = sent.strip()
    if not s:
        return "OTHER"
    if s[0] in "\"'“‘":
        return "QUOTE"
    nlp = _nlp()
    if nlp is None:
        raise RuntimeError(unavailable_reason())
    doc = nlp(" ".join(s.split()[:6]))
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return "OTHER"
    t = toks[0]
    w = t.text.lower()
    if t.pos_ == "CCONJ" or w in ("and", "but", "or", "nor", "yet", "so"):
        return "CONJ" if w != "so" or t.pos_ == "CCONJ" else "DISC"
    if w in DISC:
        return "DISC"
    if t.pos_ == "SCONJ" or w in SUB:
        return "SUBORD"
    if t.pos_ == "ADP":
        return "PP"
    if t.pos_ == "ADV":
        return "ADV"
    if t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBG", "VBN"):
        return "PART"
    if t.pos_ == "PRON" or w in ("i", "you", "he", "she", "it", "we", "they",
                                 "there", "this", "that", "these", "those"):
        return "PRON"
    if t.pos_ in ("DET", "NOUN", "PROPN", "ADJ", "NUM"):
        return "NP"
    return "OTHER"


def profile(text: str) -> dict[str, float]:
    """Class shares plus the sentence count, keyed `share_<CLASS>`."""
    counts = {c: 0 for c in CLASSES}
    n = 0
    for s in sentences(text):
        if s.strip():
            counts[opener_class(s)] += 1
            n += 1
    out = {f"share_{c}": (counts[c] / n if n else 0.0) for c in CLASSES}
    out["n_sent"] = float(n)
    return out


def js_distance(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon distance, base 2. Bounded 0..1 by construction."""
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0 and b > 0)

    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def distance_to(text_profile: dict[str, float], reference: dict[str, float]) -> float:
    """JS distance from a passage's class profile to a reference distribution."""
    p = [text_profile.get(f"share_{c}", 0.0) for c in CLASSES]
    q = [reference.get(f"share_{c}", 0.0) for c in CLASSES]
    return js_distance(p, q)


def top_class(text_profile: dict[str, float], reference: dict[str, float]) -> tuple[str, float]:
    """The class furthest above the reference, and by how much.

    The distance says a passage's openers are wrong; this says which opener.
    "NP +0.19" is actionable, 0.074 is not.
    """
    best, gap = "", 0.0
    for c in CLASSES:
        d = text_profile.get(f"share_{c}", 0.0) - reference.get(f"share_{c}", 0.0)
        if d > gap:
            best, gap = c, d
    return best, gap
