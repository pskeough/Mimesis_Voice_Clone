"""Ingestion: documents -> ~180-word chunks -> SQLite (chunks + FTS5 + vectors).

Reads ``.docx`` / ``.txt`` / ``.md`` from a profile's ``source_documents/``,
packs paragraphs into ~180-word chunks with one-paragraph overlap, embeds them
with the profile's backend, and writes an SQLite store with a chunk table, an
FTS5 keyword index (skipped gracefully if the SQLite build lacks FTS5), and a
``meta`` table holding the corpus hash.

Ported from the v1 Claude_Style_Clone_MCP ingest, minus the PDF path (v2 targets
docx/txt/md per the design) and plus a corpus-hash short-circuit: an unchanged
corpus skips the whole embed pass (``hash-skip``).
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

from . import config, embed

# Make console output robust to non-cp1252 characters when piped on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUPPORTED_EXTS = config.SUPPORTED_EXTS

TARGET_WORDS = 190  # ~180-200 word chunks
MIN_WORDS = 40
HARD_CAP_WORDS = 350


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # strip control chars
    return text


def _finalize_paragraphs(paragraphs: list[str]) -> list[str]:
    """Normalize whitespace and drop fragments too short to carry voice."""
    out = []
    for p in paragraphs:
        p_clean = re.sub(r"\s+", " ", p).strip()
        if len(p_clean.split()) >= 8:
            out.append(p_clean)
    return out


def _extract_docx(path: Path) -> list[str]:
    from docx import Document  # lazy; python-docx is a base dep

    doc = Document(str(path))
    return _finalize_paragraphs([p.text for p in doc.paragraphs if p.text.strip()])


def _extract_text(path: Path) -> list[str]:
    raw = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    blocks = re.split(r"\n\s*\n", raw)
    return _finalize_paragraphs(blocks)


def extract_paragraphs(path: Path) -> list[str]:
    """Extract clean paragraphs from a supported document."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in (".txt", ".md"):
        return _extract_text(path)
    return []


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedy-pack paragraphs into ~TARGET_WORDS chunks with one-paragraph overlap."""
    chunks: list[str] = []
    i = 0
    while i < len(paragraphs):
        cur: list[str] = []
        cur_wc = 0
        j = i
        while j < len(paragraphs):
            p = paragraphs[j]
            p_wc = len(p.split())
            if p_wc > HARD_CAP_WORDS and not cur:
                # A single huge paragraph: sentence-split it into chunks.
                sents = re.split(r"(?<=[.!?])\s+", p)
                accum: list[str] = []
                accum_wc = 0
                for s in sents:
                    sw = len(s.split())
                    if accum and accum_wc + sw > TARGET_WORDS:
                        chunks.append(" ".join(accum))
                        accum, accum_wc = [s], sw
                    else:
                        accum.append(s)
                        accum_wc += sw
                if accum:
                    chunks.append(" ".join(accum))
                j += 1
                i = j
                break
            if cur and cur_wc + p_wc > TARGET_WORDS:
                break
            cur.append(p)
            cur_wc += p_wc
            j += 1
            if cur_wc >= TARGET_WORDS:
                break
        if cur_wc < MIN_WORDS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(cur)
            break
        if cur:
            chunks.append("\n\n".join(cur))
        if j >= len(paragraphs):
            break
        i = max(j - 1, i + 1)  # one-paragraph overlap
    return chunks


def find_documents(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    paths = [
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def corpus_hash(doc_paths: list[Path]) -> str:
    """Content hash over all source files, so an unchanged corpus can be skipped."""
    h = hashlib.sha256()
    for p in doc_paths:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def read_pieces(db_path: Path) -> dict[str, str]:
    """Reassemble full per-document texts from the store, grouped by filename.

    Chunks are stored with one-paragraph overlap, so joining them repeats a little
    text; that is harmless for stylometric calibration and eval, which care about
    distributional features, not exact reconstruction.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT filename, text FROM chunks ORDER BY filename, chunk_idx"
        ).fetchall()
    finally:
        conn.close()
    by_piece: dict[str, list[str]] = {}
    for filename, text in rows:
        by_piece.setdefault(filename, []).append(text)
    return {fn: "\n\n".join(chunks) for fn, chunks in by_piece.items()}


CALIBRATION_SEGMENT_WORDS = 1500
CALIBRATION_SPREAD_RATIO = 6.0
# Below this, feature values get unstable enough that extra units are worse than
# fewer good ones: type-token ratio, sentence-length stdev and the cadence
# autocorrelations all need a few dozen sentences to mean anything.
CALIBRATION_MIN_SEGMENT_WORDS = 400


def calibration_target(texts: list[str], n_features: int) -> int:
    """Segment size that yields roughly ``3 * n_features`` calibration units.

    A fixed 1500-word target serves a corpus of book chapters and starves a
    corpus of a few long documents. What calibration actually needs is *enough
    comparable units to estimate a mean and a standard deviation per feature*;
    the rule of thumb used throughout this codebase is ~3x the feature count.

    So aim the segment size at that unit count, clamped to
    [CALIBRATION_MIN_SEGMENT_WORDS, CALIBRATION_SEGMENT_WORDS]. A word-rich,
    document-poor corpus (24,309 words in 27 pieces) gets cut finer and reaches a
    usable unit count; a genuinely small corpus cannot be rescued by cutting it
    smaller, and the floor stops us trying.
    """
    total = sum(len(t.split()) for t in texts)
    want = max(1, 3 * n_features)
    ideal = total // want if want else CALIBRATION_SEGMENT_WORDS
    return max(CALIBRATION_MIN_SEGMENT_WORDS, min(CALIBRATION_SEGMENT_WORDS, ideal))


def segment_for_calibration(
    texts: list[str], target: int = CALIBRATION_SEGMENT_WORDS
) -> list[str]:
    """Cut over-long pieces into ~``target``-word units at paragraph boundaries.

    Calibration estimates a mean and standard deviation per feature across
    pieces. When the pieces themselves range from 353 to 17,717 words -- as a
    corpus of book chapters does -- those standard deviations are substantially a
    measure of document length, not of style, and the handful of very long
    documents dominate every estimate. Segmenting produces many comparable units:
    on that corpus it took the usable count from 15 to 60, which also matters
    because 26 features cannot be estimated from 15 samples.

    Only pieces above the target are cut, so a corpus of already-comparable
    pieces passes through untouched. Trailing scraps below a third of the target
    are dropped rather than admitted as short, noisy samples.
    """
    out: list[str] = []
    for t in texts:
        if len(t.split()) <= target:
            out.append(t)
            continue
        buf: list[str] = []
        n = 0
        for para in (p for p in t.split("\n\n") if p.strip()):
            buf.append(para)
            n += len(para.split())
            if n >= target:
                out.append("\n\n".join(buf))
                buf, n = [], 0
        if buf and n >= target // 3:
            out.append("\n\n".join(buf))
    return out


def needs_segmentation(texts: list[str], ratio: float = CALIBRATION_SPREAD_RATIO) -> bool:
    """True when piece lengths are spread widely enough to distort calibration."""
    lens = sorted(len(t.split()) for t in texts if t.strip())
    if len(lens) < 3:
        return False
    mid = lens[len(lens) // 2]
    return mid > 0 and (lens[-1] / mid) >= ratio


def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _read_meta(db_path: Path, key: str) -> str | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return None


def init_db(conn: sqlite3.Connection, use_fts: bool) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("DROP TABLE IF EXISTS fts_chunks")
    cur.execute("DROP TABLE IF EXISTS meta")
    cur.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, filename TEXT, text TEXT,
            word_count INTEGER, chunk_idx INTEGER, embedding BLOB
        )"""
    )
    if use_fts:
        cur.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(id UNINDEXED, text)")
    cur.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def run(profile: config.Profile | None = None, force: bool = False) -> bool:
    """Build the store for a profile (default: active). Returns True on success."""
    prof = profile or config.resolve_active()
    source_dir = prof.source_dir
    db_path = prof.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    doc_paths = find_documents(source_dir)
    if not doc_paths:
        print(f"ERROR: No documents found in {source_dir}")
        print(f"       Add samples ({', '.join(SUPPORTED_EXTS)}) there and re-run.")
        return False

    chash = corpus_hash(doc_paths)
    if not force and _read_meta(db_path, "corpus_hash") == chash:
        print(f"Corpus unchanged (hash {chash[:12]}); skipping re-embed. Use --force to rebuild.")
        return True

    print(f"Found {len(doc_paths)} document(s) in {source_dir}")
    all_chunks: list[dict] = []
    for doc_path in doc_paths:
        print(f"Processing: {doc_path.name}")
        try:
            paragraphs = extract_paragraphs(doc_path)
        except Exception as e:
            print(f"  Warning: could not read {doc_path.name}: {e}")
            continue
        if not paragraphs:
            print(f"  Warning: no usable text in {doc_path.name}")
            continue
        chunks = chunk_paragraphs(paragraphs)
        total_words = sum(len(p.split()) for p in paragraphs)
        print(f"  {len(paragraphs)} paragraphs ({total_words:,} words) -> {len(chunks)} chunks")
        for idx, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{doc_path.stem}::{idx:03d}",
                    "filename": doc_path.stem,
                    "text": ch,
                    "word_count": len(ch.split()),
                    "chunk_idx": idx,
                }
            )

    if not all_chunks:
        print("ERROR: No chunks parsed. Aborting.")
        return False

    conn = sqlite3.connect(str(db_path))
    use_fts = fts5_available(conn)
    if not use_fts:
        print("[INFO] SQLite build lacks FTS5; keyword index skipped (vector-only search).")
    init_db(conn, use_fts)

    print(f"Embedding {len(all_chunks)} chunks with backend '{prof.embed_backend}'...")
    texts = [c["text"] for c in all_chunks]
    vectors = embed.encode(texts, backend=prof.embed_backend)

    cur = conn.cursor()
    for i, ch in enumerate(all_chunks):
        blob = np.asarray(vectors[i], dtype=np.float32).tobytes()
        cur.execute(
            "INSERT INTO chunks (id, filename, text, word_count, chunk_idx, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ch["id"], ch["filename"], ch["text"], ch["word_count"], ch["chunk_idx"], blob),
        )
        if use_fts:
            cur.execute(
                "INSERT INTO fts_chunks (id, text) VALUES (?, ?)", (ch["id"], ch["text"])
            )
    cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('corpus_hash', ?)", (chash,))
    cur.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_backend', ?)",
        (prof.embed_backend,),
    )
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    print(f"Store built: {db_path} ({n} chunks, FTS5={'on' if use_fts else 'off'}).")
    return True
