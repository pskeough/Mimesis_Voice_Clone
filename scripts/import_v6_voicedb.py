"""Rebuild a Mimesis profile from a v6 Voice MCP ``voice_db.sqlite``.

The v6 engine (the predecessor shipped as a standalone Voice MCP) stored the
author's corpus *inside* its own SQLite store rather than as files on disk:
``chunks`` holds the corpus split into ~200-word pieces with the plain text
kept alongside the embedding, and ``voice_cookbook`` holds whole documents.

Mimesis expects the opposite: ``profiles/<slug>/source_documents/*.txt``, with
everything derived from those files by ``ingest`` and ``calibrate``. So an
upgrade from v6 is not a migration of derived artifacts (there is no schema
version and the feature set has changed, which is why UPGRADE.md says rebuild).
It is a recovery of the SOURCE from the only place it still exists, followed by
a normal fresh calibration.

Chunks are non-overlapping and carry ``chunk_idx``, so a document is recovered
by grouping on ``filename`` and joining in index order. The chunk table is
always the authority. ``voice_cookbook`` looks like it holds the originals but
its ``passage`` column is an EXCERPT: on a measured store, one document was 57
words in the cookbook against 53,197 words across its chunks. Preferring the
cookbook therefore drops most of a corpus without erroring, so it is used only
for documents that have no chunks at all.

Formats are kept apart by default. A single fingerprint computed across
LinkedIn posts and long-form documents together measures FORMAT far more than
it measures voice, which shows up as a self-baseline well above 1.0 and a gate
that cannot be satisfied. v6 knew this too and calibrated its gate per format.

Usage:
    python scripts/import_v6_voicedb.py <voice_db.sqlite> --slug <you> --out profiles
    python scripts/import_v6_voicedb.py <db> --slug <you> --out /tmp/x --no-split
    python scripts/import_v6_voicedb.py <db> --slug <you> --out profiles --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Written per generated profile. Deliberately minimal: every threshold that
# matters is MEASURED by `calibrate`, and hand-typing one here is exactly the
# failure scripts/portability_audit.py exists to catch.
CONFIG_STUB = {
    "embed_backend": "fast",
    "gate": {"slate_size": 4, "max_rewrites": 2},
    "anchors": {"exemplars": True, "transform_pairs": None},
    "whitelist": [],
}


def safe_name(name: str) -> str:
    """Filesystem-safe stem. v6 filenames are already tame (post_001), but the
    cookbook stores human titles with slashes and punctuation."""
    stem = re.sub(r"[^\w\s.-]", "", str(name)).strip().replace(" ", "_")
    return (stem or "untitled")[:120]


def load_documents(db_path: Path) -> list[dict]:
    """Recover whole documents by reassembling chunks; cookbook is fallback only."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "chunks" not in tables:
        sys.exit(f"error: {db_path} has no 'chunks' table; this is not a v6 voice_db.sqlite")

    # 1. Cookbook excerpts, kept only as a fallback for documents the chunk
    #    table does not cover at all (see the module docstring).
    cookbook: dict[str, dict] = {}
    if "voice_cookbook" in tables:
        for r in con.execute("SELECT filename, format, passage FROM voice_cookbook"):
            if r["passage"] and r["passage"].strip():
                cookbook[r["filename"]] = {
                    "filename": r["filename"],
                    "format": r["format"] or "general",
                    "domain": "",
                    "text": r["passage"].strip(),
                    "source": "cookbook excerpt (no chunks found)",
                }

    # 2. The real corpus: every document reassembled from its chunks in order.
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in con.execute(
        "SELECT filename, chunk_idx, text, format, domain FROM chunks ORDER BY filename, chunk_idx"
    ):
        grouped[r["filename"]].append(r)
    con.close()

    docs: list[dict] = []
    for filename, rows in grouped.items():
        cookbook.pop(filename, None)  # chunks win; the excerpt is redundant
        text = "\n\n".join((r["text"] or "").strip() for r in rows if (r["text"] or "").strip())
        if not text:
            continue
        docs.append({
            "filename": filename,
            "format": rows[0]["format"] or "general",
            "domain": rows[0]["domain"] or "",
            "text": text,
            "source": f"chunks x{len(rows)}",
        })

    # Cookbook entries with no matching chunks are still real documents.
    docs.extend(cookbook.values())
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild a Mimesis profile from a v6 voice_db.sqlite")
    ap.add_argument("db", type=Path, help="path to the v6 data/voice_db.sqlite")
    ap.add_argument("--slug", required=True, help="base profile slug, e.g. 'jane'")
    ap.add_argument("--out", type=Path, required=True, help="profiles/ root to write into")
    ap.add_argument("--no-split", action="store_true", help="one blended profile instead of one per format")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = ap.parse_args()

    if not args.db.exists():
        sys.exit(f"error: no such file: {args.db}")

    docs = load_documents(args.db)
    if not docs:
        sys.exit("error: recovered no documents; is this an empty store?")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        buckets[args.slug if args.no_split else f"{args.slug}-{d['format']}"].append(d)

    total_words = sum(len(d["text"].split()) for d in docs)
    print(f"recovered {len(docs)} documents, ~{total_words:,} words, from {args.db.name}")
    if not args.no_split:
        print("splitting by format (a fingerprint blended across forms measures format, not voice)")

    for slug, group in sorted(buckets.items()):
        ranked = sorted(group, key=lambda x: -len(x["text"].split()))
        words = sum(len(d["text"].split()) for d in group)
        note = "" if words >= 20_000 else "   <-- thin; a fingerprint from this will be noisy"
        print(f"\n  {slug}: {len(group)} docs, ~{words:,} words{note}")
        for d in ranked[:3]:
            print(f"      {d['filename']} ({len(d['text'].split()):,}w, {d['source']})")
        if len(group) > 3:
            print(f"      ... and {len(group) - 3} more")

        # One very large document inside a mixed bucket does not just weight the
        # fingerprint, it effectively becomes it. Worth splitting out by hand.
        top = len(ranked[0]["text"].split())
        if len(group) > 1 and top > 0.5 * words:
            print(f"      ! '{ranked[0]['filename']}' is {top / words:.0%} of this profile's words.")
            print(f"        Consider giving it its own profile; blended, it IS the fingerprint.")

        if args.dry_run:
            continue

        src_dir = args.out / slug / "source_documents"
        src_dir.mkdir(parents=True, exist_ok=True)
        for d in group:
            (src_dir / f"{safe_name(d['filename'])}.txt").write_text(d["text"], encoding="utf-8")

        cfg_path = args.out / slug / "config.json"
        if not cfg_path.exists():
            cfg = dict(CONFIG_STUB, author_name=args.slug)
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    print(f"\nwritten to {args.out}\nNext, per profile:")
    for slug in sorted(buckets):
        print(f"  mimesis ingest {slug} && mimesis calibrate {slug}")
    print("\nThen check each self-baseline, and run scripts/portability_audit.py <slug>.")


if __name__ == "__main__":
    main()
