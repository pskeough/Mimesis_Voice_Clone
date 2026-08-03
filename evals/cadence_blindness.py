"""Does the 13-feature fingerprint see cadence at all?

Decisive offline test. Take real corpus pieces and destroy their *rhythm* while
holding their *content and sentence inventory* exactly constant:

  - shuffle:  reorder the sentences at random (same sentences, new order)
  - sort:     sentences ordered shortest->longest (maximally un-humanlike rhythm,
              a monotone ramp no writer produces)

Every one of the 13 features is an order-invariant aggregate (mean, stdev,
percentage, per-100w rate, type-token ratio). If the fingerprint is blind to
cadence, RMS-z will not move under either transform. Sorting is the stronger
probe: a text whose sentences climb monotonically from 3 words to 40 is
rhythmically absurd, and a cadence-aware metric must reject it.

Run:  .venv/Scripts/python.exe evals/cadence_blindness.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice.fingerprint import Fingerprint, _sentences, _WORD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260729


def rebuild(sents: list[str]) -> str:
    """Reassemble sentences into a single paragraph-per-5 text."""
    out, buf = [], []
    for i, s in enumerate(sents, 1):
        buf.append(s)
        if i % 5 == 0:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


def main() -> None:
    rng = random.Random(SEED)
    voice = sys.argv[1] if len(sys.argv) > 1 else "creative"
    fp = Fingerprint.load(ROOT / "profiles" / voice / "data" / "fingerprint.json")
    docs = sorted((ROOT / "profiles" / voice / "source_documents").glob("*.txt"))

    rows = []
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="ignore")
        sents = _sentences(text)
        if len(sents) < 12 or len(_WORD.findall(text)) < 200:
            continue
        # Control: the same reassembly, original order. Isolates the reorder
        # effect from the paragraph-rebuild effect.
        control = rebuild(sents)
        shuffled = sents[:]
        rng.shuffle(shuffled)
        ramp = sorted(sents, key=lambda s: len(_WORD.findall(s)))
        rows.append(
            (
                p.name,
                fp.distance(control),
                fp.distance(rebuild(shuffled)),
                fp.distance(rebuild(ramp)),
            )
        )

    if not rows:
        print("no usable documents")
        return

    ctrl = [r[1] for r in rows]
    shuf = [r[2] for r in rows]
    ramp = [r[3] for r in rows]

    print(f"voice={voice}  n={len(rows)}  self_baseline={fp.self_baseline:.4f}  "
          f"fit_threshold={fp.fit_threshold:.4f}")
    print()
    print(f"{'condition':<28}{'mean RMS-z':>12}{'sd':>10}{'delta vs control':>20}")
    print("-" * 70)
    print(f"{'original order (control)':<28}{statistics.mean(ctrl):>12.4f}"
          f"{statistics.pstdev(ctrl):>10.4f}{'--':>20}")
    print(f"{'sentences SHUFFLED':<28}{statistics.mean(shuf):>12.4f}"
          f"{statistics.pstdev(shuf):>10.4f}"
          f"{statistics.mean(shuf) - statistics.mean(ctrl):>+20.6f}")
    print(f"{'sentences SORTED (ramp)':<28}{statistics.mean(ramp):>12.4f}"
          f"{statistics.pstdev(ramp):>10.4f}"
          f"{statistics.mean(ramp) - statistics.mean(ctrl):>+20.6f}")
    print()

    max_shuf = max(abs(r[2] - r[1]) for r in rows)
    max_ramp = max(abs(r[3] - r[1]) for r in rows)
    n_flag_ramp = sum(1 for r in rows if r[3] > fp.fit_threshold)
    n_flag_ctrl = sum(1 for r in rows if r[1] > fp.fit_threshold)
    print(f"max |delta| any single doc, shuffle : {max_shuf:.8f}")
    print(f"max |delta| any single doc, ramp    : {max_ramp:.8f}")
    print(f"docs flagged off-voice, control     : {n_flag_ctrl}/{len(rows)}")
    print(f"docs flagged off-voice, sorted ramp : {n_flag_ramp}/{len(rows)}")
    print()
    if max_ramp < 1e-6:
        print("VERDICT: the fingerprint is EXACTLY blind to sentence order.")
        print("A monotone shortest-to-longest ramp scores identically to the")
        print("author's real prose. Cadence is invisible to all 13 features.")


if __name__ == "__main__":
    main()
