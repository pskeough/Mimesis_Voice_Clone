#!/usr/bin/env python3
"""
Regression check for the two-sided fit gate in scrub.analyze.

Fixes a specific, reproduced failure: a voiced draft that carried no AI tells but
sat at roughly half the corpus lexical diversity was reported "CLEAN" by
scrub_ai_footprint, because that path never consulted the fingerprint. The CLI
compose gate did; the MCP path did not.

Asserts, for every calibrated voice:
  1. real corpus pieces still pass (no false alarms introduced), and
  2. a flat off-voice draft is now flagged.

    python scripts/check_fit_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import re  # noqa: E402

from mimesis_voice import config, ingest, scrub as scrub_mod  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402
from mimesis_voice.scrub import FIT_RATIO, SINGLE_FEATURE_Z, ScrubCalibration  # noqa: E402

# Deliberately flat: no AI tells, no em-dashes, short words, thin vocabulary,
# low reading level. This is the shape that used to score CLEAN.
FLAT = """
The man went back to the house. It was the same house. He had not been there
in a long time. The door was not locked so he went in and stood in the hall.

There was a room off the kitchen where his father had worked. The tools were
still out on the bench. A chair sat on the bench with three legs on it. One leg
was not on yet. There was a line drawn on the wood where the cut had to go.

He looked at it for a while. Then he took off his coat and hung it up. He picked
up the tool and made the cut. It was not a good cut but it was close enough.

He put the leg on and turned the chair over and set it down. It stood up. It was
not straight but it stood. He sat down on it and put his hands on his knees and
waited. Nothing came. He sat there until it got dark and he could not see.
""".strip()


def main() -> int:
    slugs = config.list_profiles()
    if not slugs:
        print("no profiles configured")
        return 1

    failures = 0
    for slug in slugs:
        prof = config._build_profile(slug)
        if not prof.scrub_path.exists() or not prof.fingerprint_path.exists():
            print(f"{prof.slug:<14} SKIP (not calibrated)")
            continue

        cal = ScrubCalibration.load(prof.scrub_path)
        fp = Fingerprint.load(prof.fingerprint_path)

        rep = scrub_mod.analyze(FLAT, cal, fp=fp)
        old_would_say_clean = rep.is_clean and not rep.emdash_count and not rep.banned_words
        status = ("HARD" if rep.fit_off
                  else "advisory" if rep.fit_drifting
                  else "MISSED")
        print(f"{prof.slug:<14} flat draft: {status:<8} "
              f"dist={rep.fp_distance:.2f} baseline={rep.fp_baseline:.2f} "
              f"ratio={rep.fp_distance / rep.fp_baseline:.2f}x"
              if rep.fp_baseline else f"{prof.slug:<14} no baseline")
        if rep.fp_worst:
            print("               worst: " + ", ".join(
                f"{n} {z:+.1f}" for n, z in rep.fp_worst))
        if not (rep.fit_off or rep.fit_drifting):
            print(f"               WARNING: flat draft not caught at all for "
                  f"{prof.slug}"
                  f"{' (and old path would have called it CLEAN)' if old_would_say_clean else ''}")
            failures += 1
        elif not rep.fit_off:
            # Not a failure, but worth naming: the corpus spread is wide enough
            # that only the advisory tier fires. Per-form fingerprints would
            # tighten this.
            print(f"               note: advisory tier only; {prof.slug} corpus "
                  f"spread (p95 {rep.fp_threshold:.2f}) is too wide for a hard flag")

        # False-alarm rate. A gate that flags the author's own writing is worse
        # than no gate: it trains the user to ignore it.
        try:
            pieces = [
                t for t in ingest.read_pieces(prof.db_path).values()
                if len(re.findall(r"[A-Za-z']+", t)) >= 120
            ]
        except Exception as e:  # noqa: BLE001 - diagnostic script
            print(f"               (corpus unreadable: {e})")
            continue
        if not pieces:
            print("               (no corpus pieces to check false alarms)")
            continue
        # Both tiers must be measured. An advisory that fires on half the
        # author's own pieces is noise, and noise is what gets ignored.
        hard = soft = 0
        dists = []
        for t in pieces:
            r = scrub_mod.analyze(t, cal, fp=fp)
            dists.append(r.fp_distance)
            if r.fit_off:
                hard += 1
            elif r.fit_drifting:
                soft += 1
        dists.sort()
        n = len(dists)
        hp, sp = 100.0 * hard / n, 100.0 * soft / n
        print(f"               real pieces: n={n} "
              f"median={dists[n//2]:.2f} p90={dists[int(n*0.9)]:.2f} "
              f"max={dists[-1]:.2f} | threshold={rep.fp_threshold:.2f}")
        print(f"               false alarms: HARD {hard} ({hp:.0f}%)  "
              f"advisory {soft} ({sp:.0f}%)")
        if hp > 8.0:
            print(f"               WARNING: hard tier fires on {hp:.0f}% of real "
                  f"pieces for {prof.slug}")
            failures += 1
        if hp + sp > 35.0:
            print(f"               WARNING: some tier fires on {hp + sp:.0f}% of real "
                  f"pieces for {prof.slug}; SINGLE_FEATURE_Z={SINGLE_FEATURE_Z} "
                  f"is too loose")
            failures += 1

    print()
    print("FAIL" if failures else
          "OK -- flat drafts flagged, real writing passes, for every calibrated voice")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

