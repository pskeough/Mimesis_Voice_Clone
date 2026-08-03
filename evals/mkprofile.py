"""Clone a voice profile with a different embed backend (or other overrides).

Used to stand up the Upgrade-1 comparison profiles (``research-style``,
``creative-style``) without disturbing the ``fast`` originals, which the brief
requires kept intact. The clone copies ``source_documents/``, any ``pairs/``,
``VOICE.md`` and ``config.json``, then rewrites ``embed_backend``. Ingest +
calibrate are left to the caller (``mimesis ingest`` / ``mimesis calibrate``),
so the store is (re-)embedded under the new backend.

    python evals/mkprofile.py --src research --dst research-style --backend style
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from mimesis_voice import config
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import config


def clone(src: str, dst: str, backend: str, author: str | None = None) -> Path:
    src_dir = config.profile_dir(src)
    dst_dir = config.profile_dir(dst)
    if not (src_dir / "config.json").exists():
        raise SystemExit(f"source profile '{src}' not found at {src_dir}")
    (dst_dir / "data").mkdir(parents=True, exist_ok=True)

    # source_documents (the corpus)
    if (src_dir / "source_documents").exists():
        shutil.copytree(
            src_dir / "source_documents", dst_dir / "source_documents",
            dirs_exist_ok=True,
        )
    # transform pairs, if any
    if (src_dir / "pairs").exists():
        shutil.copytree(src_dir / "pairs", dst_dir / "pairs", dirs_exist_ok=True)
    # voice guide
    if (src_dir / "VOICE.md").exists():
        shutil.copy2(src_dir / "VOICE.md", dst_dir / "VOICE.md")

    cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    cfg["embed_backend"] = backend
    cfg.pop("_embed_backend_intended", None)
    cfg["_cloned_from"] = src
    if author:
        cfg["author_name"] = author
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return dst_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--backend", default="style", choices=["fast", "style"])
    ap.add_argument("--author", default=None)
    args = ap.parse_args()
    d = clone(args.src, args.dst, args.backend, args.author)
    print(f"cloned '{args.src}' -> '{args.dst}' (backend={args.backend}) at {d}")
    print(f"next: mimesis ingest {args.dst} && mimesis calibrate {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
