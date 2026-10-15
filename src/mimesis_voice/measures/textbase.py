"""Text splitting and normalisation, ported verbatim from the study harness.

The four functions below are copies of `dr/core.py`'s `sentences`, `paragraphs`
and `normalise_punct` and of `eval_methodology/methods/common.py`'s
`fold_glyphs`, taken from the private decompose-regenerate study repo on
2026-09-09.
They are copied rather than imported: the study repo is a private research tree
that this package must run without, and a measure whose splitter drifts from the
one its thresholds were calibrated on is silently measuring something else.
`tests/test_measure_parity.py` reproduces the study's stored per-cell values to
three decimals and fails if either side moves.

Two properties matter downstream and are easy to get wrong:

`normalise_punct` folds `--` into an em dash. That is a *measurement* fold, so a
Gutenberg transcription and a model's output are counted the same way; 53-68% of
Wharton's real passages carry `--` and no model output does. It must never be
fed back into `scrub.py`, where the em dash is a hard flag: normalising first
would manufacture the flag.

`paragraphs` splits on blank lines only, which is the study's rule and stays
verbatim. Real corpora often separate paragraphs with a single newline, so the
gate path uses `prose_paragraphs` instead -- same function on the author's corpus
and on the draft. It falls back to newline splitting only when blank lines yield
one block, and it never fires on the study's own data (0 of 150 passages), so
parity is unaffected.
"""
from __future__ import annotations

import re

__all__ = [
    "fold_glyphs", "norm", "normalise_punct", "para_lengths", "paragraphs",
    "prose_paragraphs", "sent_lengths", "sentences", "truncate_words",
    "word_count", "words",
]

# Same splitter family as the study's instruments, so a sentence here is a
# sentence there.
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]]*\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
_PARA_SPLIT = re.compile(r"\n\s*\n")

_DDASH = re.compile(r"\s*--+\s*")
_UNDER = re.compile(r"(?<![A-Za-z0-9])_([^_\n]{1,80})_(?![A-Za-z0-9])")

# Curly quotes and apostrophes folded to straight, so no measure can read
# transcription provenance instead of style.
_GLYPHS = {
    "\u2019": "'", "\u2018": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u2033": '"',
    "\u2026": "...", "\u00a0": " ",
}


def sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def paragraphs(text: str) -> list[str]:
    ps = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    return ps or [text.strip()]


def normalise_punct(text: str) -> str:
    """Drop transcription conventions that give a source away.

    Gutenberg's `--` becomes an em dash with no spaces and `_italics_` loses its
    underscores. For measurement only; see the module docstring.
    """
    t = _UNDER.sub(r"\1", text)
    return _DDASH.sub("\u2014", t)


def fold_glyphs(text: str) -> str:
    for a, b in _GLYPHS.items():
        text = text.replace(a, b)
    return text


def norm(text: str) -> str:
    """The study's full pre-measurement normalisation: punctuation then glyphs."""
    return fold_glyphs(normalise_punct(text))


def truncate_words(text: str, n: int | None) -> str:
    """First `n` whitespace-separated words, keeping paragraph breaks.

    Splits on the original whitespace rather than on `words()`, because the cap
    exists to equalise length across passages and a regex that drops digits and
    symbols would cap two passages at different real lengths.
    """
    if n is None:
        return text
    out: list[str] = []
    count = 0
    for para in text.split("\n"):
        w = para.split()
        if count + len(w) <= n:
            out.append(para)
            count += len(w)
        else:
            out.append(" ".join(w[: n - count]))
            break
    return "\n".join(out)


def sent_lengths(text: str) -> list[int]:
    return [len(s.split()) for s in sentences(text) if s.strip()]


def para_lengths(text: str) -> list[int]:
    return [len(p.split()) for p in paragraphs(text) if p.strip()]


# Hard-wrapped prose looks like many short lines; single-newline paragraphing
# looks like fewer, longer ones. 25 words is the cut: a wrap at 80 columns runs
# 10-14 words a line, and a paragraph shorter than 25 words is rare enough that
# treating a corpus of them as unparagraphed costs nothing.
_WRAP_WORDS = 25


def prose_paragraphs(text: str) -> list[str]:
    """Paragraphs, tolerating a corpus that separates them with single newlines.

    `paragraphs` is the study's splitter and stays verbatim for parity. It splits
    on blank lines only, which is right for the study's data and wrong for a lot
    of real corpora: 33 of 40 documents in this repo's own creative profile are
    one blank-line block and forty-odd newline-separated paragraphs, so the
    paragraph-shape measure calibrated on them was measuring nothing.

    The fallback fires only when blank-line splitting yields a single block AND
    the newline blocks are long enough to be paragraphs rather than hard wrapping
    at a fixed column. Both the author's corpus and the draft go through this
    same function, which is the property that makes the comparison fair.
    """
    ps = paragraphs(text)
    if len(ps) > 1:
        return ps
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return ps
    lengths = sorted(len(ln.split()) for ln in lines)
    if lengths[len(lengths) // 2] < _WRAP_WORDS:
        return ps
    return lines
