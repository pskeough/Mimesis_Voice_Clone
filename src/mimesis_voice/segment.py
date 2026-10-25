"""Sentence segmentation that reconstructs the input byte for byte.

The scrubber's splitters return stripped sentence strings, which is enough for
counting and useless for editing: once a sentence is stripped there is no way to
put the text back together exactly. A minimal-diff edit has to leave every
untouched byte where it was, so this module returns segments that carry their
own trailing whitespace, and ``join(segments) == text`` always holds.

Two kinds of segment:

- ``prose``: one sentence, editable.
- ``fixed``: a heading, a LaTeX structure line (``\\section``, ``\\begin``,
  ``\\label``), a comment, or a blank-line run. Never edited, never counted as a
  sentence.

The sentence boundary is the scrubber's rule (terminal punctuation, whitespace,
then a capital, digit, quote, bracket or backslash) minus the abbreviations
that split research prose in the wrong place: "Smith et al. (2024) found" is one
sentence, and so is "see Fig. 2".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Tokens that end in a period without ending a sentence. "etc." is left out on
# purpose: it ends sentences as often as not.
# "No." is left out too: "Does it hold? No. The effect vanished." is three sentences.
_ABBREV = frozenset("""
al e.g i.e cf vs fig figs eq eqs sec secs tab ch nos vol pp approx ca resp
dr mr mrs ms prof st jr sr inc ltd co
""".split())

_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[\"'“‘(\[\\]?[A-Z0-9\\])")
_FIXED_LINE_RE = re.compile(
    r"^[ \t]*(?:%.*|#{1,6}[ \t].*|\\(?:section|subsection|subsubsection|paragraph|chapter|"
    r"begin|end|label|caption|maketitle|bibliography|bibliographystyle|input|include|"
    r"usepackage|documentclass)\b.*)$"
)
_BLANK_RUN_RE = re.compile(r"\r?\n[ \t]*\r?\n\s*")
# Environments whose lines are data, not sentences.
_FIXED_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "table", "tabular",
               "figure", "verbatim", "lstlisting", "minted", "algorithm", "algorithmic")
_ENV_BEGIN_RE = re.compile(r"\\begin\{(" + "|".join(_FIXED_ENVS) + r")\*?\}")
_ENV_END_RE = re.compile(r"\\end\{(" + "|".join(_FIXED_ENVS) + r")\*?\}")
_HEADING_LINE_RE = re.compile(r"^[ \t]*[A-Z0-9][^.!?:;]{0,80}[ \t\r]*$")


def _is_initial(block: str, dot: int) -> bool:
    """Is the single capital before ``block[dot]`` an initial ("J. Smith")?

    Only when a capitalised word follows and the letter is not preceded by a
    capitalised word: "by A. Smith" and "J. R. Tolkien" are initials, "Appendix A.
    We ran" and "Condition B. The rest" are sentence ends.
    """
    if not re.match(r"\.\s+[A-Z](?:[a-z]|\.)", block[dot:]):
        return False
    before = block[: dot - 1].rstrip()
    prev = before.split()[-1] if before.split() else ""
    return not prev or not prev[:1].isupper() or bool(re.fullmatch(r"[A-Z]\.", prev))


@dataclass
class Segment:
    text: str          # the segment without its trailing whitespace
    tail: str          # whitespace that followed it in the input
    kind: str          # "prose" or "fixed"

    @property
    def is_prose(self) -> bool:
        return self.kind == "prose"


def _word_before(s: str, end: int) -> str:
    m = re.search(r"([A-Za-z.]+)\.?$", s[:end])
    return (m.group(1) if m else "").rstrip(".").lower()


def _split_prose(block: str) -> list[tuple[str, str]]:
    """Split one prose block into (sentence, trailing whitespace) pairs."""
    out: list[tuple[str, str]] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(block):
        cut = m.start(1)
        if block[m.start()] == ".":
            w = _word_before(block, m.start())
            # "et al." / "e.g." / "Fig." and a single initial ("J. Smith").
            if w in _ABBREV or (len(w) == 1 and w.isalpha() and _is_initial(block, m.start())):
                continue
        sent = block[start:cut]
        if sent.strip():
            out.append((sent, m.group(1)))
        start = m.end(1)
    rest = block[start:]
    if rest.strip():
        body = rest.rstrip()
        out.append((body, rest[len(body):]))
    elif out:
        sent, tail = out[-1]
        out[-1] = (sent, tail + rest)
    return out


def split(text: str) -> list[Segment]:
    """Segments whose concatenation (text + tail) is exactly ``text``."""
    segs: list[Segment] = []
    lead = text[: len(text) - len(text.lstrip())]
    if lead:
        segs.append(Segment("", lead, "fixed"))
    body = text[len(lead):]
    pos = 0
    # Paragraphs first: a blank line always ends a sentence.
    pieces: list[tuple[str, str]] = []
    for m in _BLANK_RUN_RE.finditer(body):
        pieces.append((body[pos:m.start()], m.group(0)))
        pos = m.end()
    if pos < len(body) or not pieces:
        rest = body[pos:]
        stripped = rest.rstrip()
        pieces.append((stripped, rest[len(stripped):]))
    env_depth = [0]
    for para, gap in pieces:
        # Structure lines inside a paragraph are fixed; the prose between them is split.
        lines = para.split("\n")
        buf: list[str] = []

        def flush(tail: str) -> None:
            if not buf:
                return
            block = "\n".join(buf)
            parts = _split_prose(block)
            if not parts:
                segs.append(Segment(block, tail, "fixed"))
            else:
                consumed = sum(len(s) + len(t) for s, t in parts)
                extra = block[consumed:]
                for i, (s, t) in enumerate(parts):
                    segs.append(Segment(s, t + (extra + tail if i == len(parts) - 1 else ""), "prose"))
            buf.clear()

        for i, line in enumerate(lines):
            nl = "\n" if i < len(lines) - 1 else ""
            nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
            began = bool(_ENV_BEGIN_RE.search(line))
            if began:
                env_depth[0] += 1
            # A short unpunctuated line opening a run, followed by a line that
            # starts a sentence, is a plain-text heading ("Methods\nWe ran ...").
            # A one-line paragraph with no end punctuation is a heading too.
            heading = (not buf and (nxt[:1].isupper() or len(lines) == 1) and len(line.split()) <= 10
                       and bool(_HEADING_LINE_RE.match(line)))
            if env_depth[0] > 0 or _FIXED_LINE_RE.match(line) or heading:
                flush("\n")
                segs.append(Segment(line, nl, "fixed"))
            else:
                buf.append(line)
            if _ENV_END_RE.search(line):
                env_depth[0] = max(0, env_depth[0] - 1)
        flush("")
        if segs:
            segs[-1].tail += gap
        elif gap:
            segs.append(Segment("", gap, "fixed"))
    assert join(segs) == text, "segmentation must reconstruct the input exactly"
    return segs


def join(segs: list[Segment]) -> str:
    return "".join(s.text + s.tail for s in segs)


def sentences(text: str) -> list[str]:
    """The prose sentences only, as plain strings."""
    return [s.text for s in split(text) if s.is_prose]
