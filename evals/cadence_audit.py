"""Do the shipped v1-engine generations have the author's cadence?

No generation needed: the previous upgrade campaign saved 37 real outputs across
five engine configs under ``evals/<config>/<voice>/``. Every one was selected by
a gate optimising the 13 order-invariant features -- which cannot see rhythm. So
this asks the direct question:

    on the 13 order-AWARE features, how far from the author is the engine's
    output, compared with the author's own writing?

Scoring is on a fingerprint calibrated from the corpus, and the human reference
is leave-one-out, so real pieces are never scored against a ruler containing
themselves. Generations were never in the calibration at all.

Reported per config:
  * cadence RMS-z   -- the 13 rhythm features only
  * surface RMS-z   -- the 13 original features only
  * AUROC vs human  -- can the cadence block tell generations from real writing?
                       0.5 = indistinguishable, 1.0 = perfectly separable.

The last column is the headline. If the engine reproduced the author's rhythm,
its outputs would be as hard to separate from real prose as real prose is from
itself, and AUROC would sit near 0.5.

Run:  .venv/Scripts/python.exe evals/cadence_audit.py [voice]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import cadence as cadmod  # noqa: E402
from mimesis_voice import config, ingest  # noqa: E402
from mimesis_voice import fingerprint as fpmod  # noqa: E402

CONFIGS = ("baseline", "style", "detect", "recalibrate", "combined")
EVALS = Path(__file__).resolve().parent


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def block_rmsz(fp: fpmod.Fingerprint, text: str, names: tuple[str, ...]) -> float:
    _, zs = fp.distance_detail(text)
    return math.sqrt(sum(zs[f] ** 2 for f in names) / len(names))


def main() -> None:
    voice = sys.argv[1] if len(sys.argv) > 1 else "creative"
    prof = config.resolve(voice)
    pieces = list(ingest.read_pieces(prof.db_path).values())
    fp = fpmod.calibrate(pieces, feature_set="v2")

    # Human reference, leave-one-out: never score a piece against a ruler that
    # contains it. Generations face the full-corpus ruler, which is the harder
    # (more tightly estimated) target -- the comparison is not tilted their way.
    human_cad, human_surf = [], []
    rows = [fpmod.extract_all(t, "v2", fp.cuts) for t in pieces]
    names = fp.features
    for i, t in enumerate(pieces):
        rest = rows[:i] + rows[i + 1:]
        loo = fpmod.Fingerprint(
            means={f: statistics.mean(r[f] for r in rest) for f in names},
            stds={f: statistics.stdev([r[f] for r in rest]) for f in names},
            feature_set="v2", cuts=fp.cuts,
        )
        human_cad.append(block_rmsz(loo, t, cadmod.CADENCE_FEATURES))
        human_surf.append(block_rmsz(loo, t, fpmod.FEATURES))

    print(f"voice={voice}  corpus n={len(pieces)}  (human reference is leave-one-out)\n")
    print(f"{'config':<14}{'n':>4}{'cadence z':>12}{'surface z':>12}"
          f"{'AUROC cad':>12}{'AUROC surf':>12}")
    print("-" * 66)
    print(f"{'HUMAN':<14}{len(pieces):>4}{statistics.mean(human_cad):>12.3f}"
          f"{statistics.mean(human_surf):>12.3f}{0.5:>12.3f}{0.5:>12.3f}")

    summary = {}
    for cfg in CONFIGS:
        d = EVALS / cfg / voice
        if not d.exists():
            continue
        gen_cad, gen_surf = [], []
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            text = rec.get("output") or rec.get("text") or ""
            if len(text.split()) < 80:
                continue
            gen_cad.append(block_rmsz(fp, text, cadmod.CADENCE_FEATURES))
            gen_surf.append(block_rmsz(fp, text, fpmod.FEATURES))
        if not gen_cad:
            continue
        a_cad = auroc(gen_cad, human_cad)
        a_surf = auroc(gen_surf, human_surf)
        summary[cfg] = {
            "n": len(gen_cad),
            "cadence": statistics.mean(gen_cad),
            "surface": statistics.mean(gen_surf),
            "auroc_cadence": a_cad,
            "auroc_surface": a_surf,
        }
        print(f"{cfg:<14}{len(gen_cad):>4}{statistics.mean(gen_cad):>12.3f}"
              f"{statistics.mean(gen_surf):>12.3f}{a_cad:>12.3f}{a_surf:>12.3f}")

    print("-" * 66)
    print("AUROC = P(a generation scores further from the author than a real piece).")
    print("0.5 means indistinguishable. Higher means the block separates them.\n")

    # Which cadence features are furthest off, pooled across all configs?
    allgen = []
    for cfg in CONFIGS:
        d = EVALS / cfg / voice
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            rec = json.loads(p.read_text(encoding="utf-8"))
            text = rec.get("output") or rec.get("text") or ""
            if len(text.split()) >= 80:
                allgen.append(text)
    if allgen:
        print(f"Worst cadence features across all {len(allgen)} generations:")
        acc: dict[str, list[float]] = {f: [] for f in cadmod.CADENCE_FEATURES}
        for t in allgen:
            _, zs = fp.distance_detail(t)
            for f in cadmod.CADENCE_FEATURES:
                acc[f].append(zs[f])
        ranked = sorted(acc.items(), key=lambda kv: abs(statistics.mean(kv[1])), reverse=True)
        for f, vals in ranked[:6]:
            mz = statistics.mean(vals)
            h = cadmod.hint(f, mz)
            print(f"  {f:<22}{mz:>+7.2f} SD   {h}")

    (EVALS / "cadence_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
