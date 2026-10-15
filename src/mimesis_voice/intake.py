"""Look at a pile of writing before it becomes a voice.

``mimesis inspect PATH`` and the ``inspect_corpus`` MCP tool both run this.
It reads every supported document the way ingest will, counts what came out,
and names the problems that make a voice fail quietly later: too little
writing, short posts mixed with long chapters, one document large enough to
become the fingerprint on its own, and PDFs with no extractable text.

Nothing here writes anything. It is the look an operator takes before running
``mimesis new``, made cheap enough that an agent takes it every time.
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import config, ingest

# Piece-length bands, in words. A short social post sits well under SHORT and a
# chapter well over LONG; a corpus with real mass at both ends is two forms.
SHORT_WORDS = 400
LONG_WORDS = 3000

# Corpus-size guidance. RECOMMENDED_DOCS matches the CLAUDE.md advice (~40
# pieces); below MIN_WORDS the standard deviations calibrate estimates are noise.
RECOMMENDED_DOCS = 40
RECOMMENDED_WORDS = 20_000
MIN_WORDS = 5_000

# One document above this share of all words stops weighting the fingerprint
# and becomes it.
DOMINANT_SHARE = 0.40

# Share of documents at EACH end of the length range that marks mixed forms.
MIXED_SHARE = 0.15

CONVERT_HINTS = {
    ".html": "save as .txt or .md (or print to PDF)",
    ".htm": "save as .txt or .md (or print to PDF)",
    ".csv": "export each post as its own .txt; a LinkedIn or Twitter CSV puts every post in one cell",
    ".json": "export each piece as its own .txt",
    ".eml": "paste the body of each sent email into its own .txt",
    ".msg": "paste the body of each sent email into its own .txt",
    ".rtf": "save as .docx",
    ".odt": "save as .docx",
    ".pages": "export as .docx",
    ".doc": "re-save as .docx",
    ".epub": "convert to .txt or .pdf",
}


@dataclass
class DocInfo:
    path: Path
    words: int
    front_matter: bool = False


@dataclass
class IntakeReport:
    sources: list[str]
    docs: list[DocInfo] = field(default_factory=list)
    empty: list[Path] = field(default_factory=list)
    duplicates: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(d.words for d in self.docs)

    def warnings(self) -> list[tuple[str, str]]:
        """(severity, message) pairs, most serious first. Severity is STOP or WARN."""
        out: list[tuple[str, str]] = []
        n, words = len(self.docs), self.total_words
        if n == 0:
            out.append(("STOP", "No readable documents. Supported: "
                        + ", ".join(config.SUPPORTED_EXTS) + "."))
            return out
        if words < MIN_WORDS:
            out.append(("STOP", f"Only {words:,} words. Below about {MIN_WORDS:,} the calibration "
                        "is noise and the gate inherits it. Ask for more writing before building."))
        elif n < RECOMMENDED_DOCS and words < RECOMMENDED_WORDS:
            out.append(("WARN", f"Thin corpus: {n} pieces, {words:,} words. About {RECOMMENDED_DOCS} "
                        f"pieces or {RECOMMENDED_WORDS:,} words is the working target. It will build, "
                        "and calibrate will print THIN CALIBRATION."))
        lengths = [d.words for d in self.docs]
        n_short = sum(1 for w in lengths if w < SHORT_WORDS)
        n_long = sum(1 for w in lengths if w > LONG_WORDS)
        if n >= 6 and n_short / n >= MIXED_SHARE and n_long / n >= MIXED_SHARE:
            out.append(("WARN", f"Mixed forms: {n_short} pieces under {SHORT_WORDS} words and {n_long} "
                        f"over {LONG_WORDS:,}. One fingerprint across both measures format, not voice. "
                        "Split into one voice per form (e.g. NAME-posts, NAME-essays)."))
        top = max(self.docs, key=lambda d: d.words)
        if n > 1 and top.words / words > DOMINANT_SHARE:
            out.append(("WARN", f"'{top.path.name}' is {100 * top.words / words:.0f}% of all words. "
                        "A document that large becomes the fingerprint. Give it its own voice, or "
                        "add more of the other writing."))
        if self.empty:
            out.append(("WARN", f"{len(self.empty)} document(s) gave no text (scanned PDF or image-only "
                        "file; needs OCR first): " + ", ".join(p.name for p in self.empty[:5])
                        + (" ..." if len(self.empty) > 5 else "")))
        for ext, k in sorted(self.skipped.items()):
            hint = CONVERT_HINTS.get(ext)
            if hint:
                out.append(("WARN", f"{k} {ext} file(s) not read: {hint}."))
        return out


def collect(sources: list[str]) -> tuple[list[tuple[Path, Path]], dict[str, int], list[str]]:
    """Resolve paths to (file, walk-root) pairs, a tally of skipped extensions,
    and the paths that do not exist. Folders are walked recursively."""
    found: list[tuple[Path, Path]] = []
    skipped: dict[str, int] = {}
    missing: list[str] = []
    for raw in sources:
        src = Path(raw).expanduser()
        if not src.exists():
            missing.append(str(src))
            continue
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
        root = src.parent if src.is_file() else src
        for p in files:
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue  # .git, .obsidian, .DS_Store and friends are never writing
            ext = p.suffix.lower()
            if ext in config.SUPPORTED_EXTS:
                found.append((p, root))
            else:
                key = ext or "(no extension)"
                skipped[key] = skipped.get(key, 0) + 1
    return found, skipped, missing


def inspect(sources: list[str]) -> IntakeReport:
    found, skipped, missing = collect(sources)
    rep = IntakeReport(sources=list(sources), skipped=skipped, missing=missing)
    seen: set[str] = set()
    for path, _root in found:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            rep.empty.append(path)
            continue
        if digest in seen:
            rep.duplicates += 1
            continue
        seen.add(digest)
        try:
            paras = ingest.extract_paragraphs(path)
        except Exception:
            paras = []
        words = sum(len(p.split()) for p in paras)
        if words == 0:
            rep.empty.append(path)
            continue
        fm = False
        if path.suffix.lower() == ".md":
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                fm = ingest.strip_front_matter(raw) != raw
            except OSError:
                pass
        rep.docs.append(DocInfo(path=path, words=words, front_matter=fm))
    return rep


def render(rep: IntakeReport) -> str:
    lines = [f"# Corpus inspection: {', '.join(rep.sources)}", ""]
    for m in rep.missing:
        lines.append(f"  ! no such path: {m}")
    n, words = len(rep.docs), rep.total_words
    lines.append(f"readable documents: {n}   words: {words:,}")
    if rep.docs:
        lengths = sorted(d.words for d in rep.docs)
        lines.append(
            f"piece length: median {int(statistics.median(lengths)):,}, "
            f"shortest {lengths[0]:,}, longest {lengths[-1]:,} words"
        )
        by_ext: dict[str, int] = {}
        for d in rep.docs:
            by_ext[d.path.suffix.lower()] = by_ext.get(d.path.suffix.lower(), 0) + 1
        lines.append("by type: " + ", ".join(f"{k} {v}" for k, v in sorted(by_ext.items())))
        n_short = sum(1 for w in lengths if w < SHORT_WORDS)
        n_long = sum(1 for w in lengths if w > LONG_WORDS)
        lines.append(
            f"length bands: {n_short} under {SHORT_WORDS}, {n - n_short - n_long} between, "
            f"{n_long} over {LONG_WORDS:,}"
        )
        fm = sum(1 for d in rep.docs if d.front_matter)
        if fm:
            lines.append(f"{fm} .md file(s) carry YAML front matter; ingest strips it.")
    if rep.duplicates:
        lines.append(f"{rep.duplicates} duplicate file(s) by content; imported once.")
    other = {k: v for k, v in rep.skipped.items() if k not in CONVERT_HINTS}
    if other:
        lines.append("ignored: " + ", ".join(f"{v}x {k}" for k, v in sorted(other.items())))
    warns = rep.warnings()
    lines.append("")
    if warns:
        for sev, msg in warns:
            lines.append(f"[{sev}] {msg}")
    else:
        lines.append("No problems found. Ready to build.")
    lines.append("")
    lines.append(
        "Not checked, and it matters most: whether every piece is the person's own solo "
        "writing. Co-written, edited-by-others and AI-assisted pieces make a voice that is "
        "partly someone else's, and nothing downstream can tell. Ask them. After building, "
        "`mimesis purity NAME` ranks the pieces that read most machine-written."
    )
    return "\n".join(lines)


def verdict(rep: IntakeReport) -> str:
    """'stop', 'warn' or 'ok', for callers that branch on the inspection."""
    sev = {s for s, _ in rep.warnings()}
    return "stop" if "STOP" in sev else ("warn" if "WARN" in sev else "ok")
