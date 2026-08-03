#!/usr/bin/env python3
"""
Add `fit_threshold` to existing fingerprints without recalibrating them.

The two-sided fit check in scrub.analyze needs a per-voice threshold. New
fingerprints get one from calibrate(); fingerprints written before that field
existed fall back to a fixed 1.6x-baseline rule, which on the creative corpus
sits on the pieces' p90 and flags 11% of the author's own writing.

This backfills only that one field. Means, stds, self_baseline, n_pieces and meta
are read and written back unchanged, so a weighted (recalibrated) fingerprint
keeps its weighting. The threshold is the p95 of leave-one-out distances over the
same corpus, computed by the same code path as calibrate().

Idempotent. Run with --force to recompute an existing value.

    python scripts/backfill_fit_threshold.py [--force] [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import config, ingest  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint, calibrate  # noqa: E402

MIN_WORDS = 120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    changed = 0
    for slug in config.list_profiles():
        prof = config._build_profile(slug)
        if not prof.fingerprint_path.exists():
            print(f"{slug:<22} skip (no fingerprint)")
            continue

        fp = Fingerprint.load(prof.fingerprint_path)
        if fp.fit_threshold > 0 and not a.force:
            print(f"{slug:<22} already set ({fp.fit_threshold:.3f})")
            continue

        try:
            texts = [
                t for t in ingest.read_pieces(prof.db_path).values()
                if len(re.findall(r"[A-Za-z']+", t)) >= MIN_WORDS
            ]
        except Exception as e:  # noqa: BLE001
            print(f"{slug:<22} skip (corpus unreadable: {e})")
            continue

        if len(texts) < 5:
            print(f"{slug:<22} skip (only {len(texts)} usable pieces)")
            continue

        try:
            probe = calibrate(texts, min_words=MIN_WORDS)
        except ValueError as e:
            print(f"{slug:<22} skip ({e})")
            continue

        old_rule = fp.self_baseline * 1.6
        fp.fit_threshold = probe.fit_threshold
        print(f"{slug:<22} n={len(texts):<4} baseline={fp.self_baseline:.3f}  "
              f"old rule={old_rule:.3f} -> p95={fp.fit_threshold:.3f}")

        if not a.dry_run:
            fp.save(prof.fingerprint_path)
            changed += 1

    print(f"\n{'(dry run) ' if a.dry_run else ''}{changed} fingerprint(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
