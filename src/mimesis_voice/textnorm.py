"""Markup -> prose, so stylometry measures writing instead of syntax.

Every feature in fingerprint.py, cadence.py, quality.py and scrub.py is defined
over prose: words per sentence, punctuation rates, type-token ratio, clause
rhythm. Feed those functions raw LaTeX and they measure the wrong object.
``\\subsection{Model identity dominates every design factor}`` is not a sentence,
``$d_{\\mathrm{pop}}$`` is not three words, and a tabular block is not a paragraph.

This mattered in practice. A LaTeX rewrite pass reported an RMS-z fingerprint
distance per subsection and the numbers looked healthy, but they were computed
over markup: backslash commands inflated the token count, ``$...$`` spans split
sentences at the wrong places, and table rows contributed hundreds of "words"
with no verbs. The gate was reading a thermometer held to the wrong patient.

The functions here are deliberately lossy and deliberately SYMMETRIC. A
comparison between a source and its rewrite is fair as long as both sides get
identical treatment, which matters more than either side being perfectly
recovered. Where a choice exists, structure that carries prose is kept (section
titles become sentences, ``\\emph{x}`` becomes ``x``) and structure that does not
is dropped entirely (tables, figures, equations, the preamble).
"""
from __future__ import annotations

import re

__all__ = ["to_prose", "detex", "demarkdown", "guess_format", "protected_spans"]

# --- LaTeX --------------------------------------------------------------------

# Whole environments whose contents are not prose. Dropped, not unwrapped: a
# tabular's cells are data, and letting them through adds hundreds of verbless
# "sentences" that flatten every rhythm measurement toward the mean.
_DROP_ENVS = (
    "table", "table*", "tabular", "tabularx", "figure", "figure*", "equation",
    "equation*", "align", "align*", "gather", "gather*", "verbatim", "lstlisting",
    "minted", "tikzpicture", "thebibliography",
)

_ENV_RE = re.compile(
    r"(?s)\\begin\{(" + "|".join(re.escape(e) for e in _DROP_ENVS) + r")\}.*?"
    r"\\end\{\1\}"
)

_COMMENT_RE = re.compile(r"(?m)(?<!\\)%.*$")
_MATH_RE = re.compile(r"(?s)\$\$.*?\$\$|(?<!\\)\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)")
_HEADING_RE = re.compile(r"\\(?:sub){0,2}section\*?\{([^{}]*)\}")
_REFCMD_RE = re.compile(r"\\(?:cite[tp]?|ref|eqref|label|autoref|nameref|input|"
                        r"include|bibliography|bibliographystyle)\s*\{[^{}]*\}")
# Commands that wrap prose: keep the argument, drop the wrapper. Nested braces
# are handled by repetition rather than a recursive pattern.
_WRAP_RE = re.compile(
    r"\\(?:textbf|textit|emph|texttt|textsc|underline|footnote|caption|"
    r"textnormal|mbox|text)\s*\{([^{}]*)\}"
)
_BARE_CMD_RE = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?")


def detex(text: str, math_token: str = "value") -> str:
    """LaTeX -> prose. Section titles survive as sentences; data does not.

    ``math_token`` stands in for every math span so a formula counts as one word
    rather than as its markup. It is a word, not a symbol, because punctuation
    rates are among the measured features and a placeholder full of braces would
    corrupt them.
    """
    t = _COMMENT_RE.sub("", text)
    # Environments first: a table can contain math, refs and wrapped commands,
    # and unwrapping those before dropping the table would leak its cells.
    prev = None
    while prev != t:
        prev, t = t, _ENV_RE.sub(" ", t)
    t = _MATH_RE.sub(f" {math_token} ", t)
    # Headings become sentences so their prose is measured and their rhythm
    # counts. They are prose the author wrote and the place tics are most
    # visible, so excluding them would hide exactly what needs checking.
    t = _HEADING_RE.sub(lambda m: " " + m.group(1).rstrip(".") + ". ", t)
    t = _REFCMD_RE.sub(" ", t)
    for _ in range(4):  # unwrap nested \textbf{\emph{...}}
        new = _WRAP_RE.sub(r"\1", t)
        if new == t:
            break
        t = new
    t = _BARE_CMD_RE.sub(" ", t)
    t = (t.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
          .replace("\\$", "$").replace("\\#", "#").replace("~", " "))
    # LaTeX dash notation -> the characters it renders as, so dash-rate features
    # see what a reader sees rather than the source encoding.
    t = t.replace("---", "\u2014").replace("--", "\u2013")
    t = re.sub(r"[{}]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# --- Markdown -----------------------------------------------------------------

_FENCE_RE = re.compile(r"(?sm)^```.*?^```", re.M)
_MD_TABLE_RE = re.compile(r"(?m)^\|.*\|[ \t]*$")


def demarkdown(text: str) -> str:
    """Markdown -> prose. Code fences and tables dropped, emphasis unwrapped."""
    t = _FENCE_RE.sub(" ", text)
    t = _MD_TABLE_RE.sub(" ", t)
    t = re.sub(r"(?m)^#{1,6}[ \t]+(.*)$", lambda m: m.group(1).rstrip(".") + ".", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", t)
    t = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "", t)
    t = re.sub(r"(?m)^[ \t]*>[ \t]?", "", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# --- dispatch -----------------------------------------------------------------

def guess_format(text: str) -> str:
    """Cheap sniff. Only used when a caller did not declare a format."""
    if re.search(r"\\(documentclass|begin\{document\}|section\*?\{|subsection\*?\{"
                 r"|cite[tp]?\{|label\{)", text):
        return "latex"
    if re.search(r"(?m)^#{1,6}[ \t]|^```|^\|.*\|[ \t]*$", text):
        return "markdown"
    return "plain"


def to_prose(text: str, fmt: str | None = None) -> str:
    """Normalise ``text`` for measurement. ``fmt`` None means sniff it."""
    kind = (fmt or guess_format(text)).lower()
    if kind in ("latex", "tex"):
        return detex(text)
    if kind in ("markdown", "md"):
        return demarkdown(text)
    return text


# --- protected spans ----------------------------------------------------------

# Regions where a surface edit is a data edit. The scalpel must not touch these,
# and the reason is concrete: a numeric range written ``0.17--0.21`` inside a
# table cell became ``0.17, 0.21``, turning one interval into two point estimates
# and contradicting the body text that described it as a range.
_PROTECT = {
    "latex": [
        _ENV_RE,
        re.compile(r"(?s)\$\$.*?\$\$|(?<!\\)\$[^$]*\$"),
        re.compile(r"(?s)\\\[.*?\\\]|\\\(.*?\\\)"),
        _REFCMD_RE,
        re.compile(r"\\[a-zA-Z@]+\*?\{[^{}]*\}"),
        # Numeric ranges anywhere: en-dash notation between digits is data.
        re.compile(r"\d[\d,.]*\s*-{2,3}\s*\d[\d,.]*"),
        re.compile(r"(?m)^\s*\\(?:usepackage|documentclass|newcommand|def)\b.*$"),
    ],
    "markdown": [
        _FENCE_RE,
        _MD_TABLE_RE,
        re.compile(r"`[^`]*`"),
        re.compile(r"\]\([^)]*\)"),
    ],
    "plain": [],
}


def protected_spans(text: str, fmt: str | None = None) -> list[tuple[int, int]]:
    """Character ranges a surface fix must leave alone, merged and sorted."""
    kind = (fmt or guess_format(text)).lower()
    kind = {"tex": "latex", "md": "markdown"}.get(kind, kind)
    spans: list[tuple[int, int]] = []
    for pat in _PROTECT.get(kind, []):
        spans.extend((m.start(), m.end()) for m in pat.finditer(text))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged
