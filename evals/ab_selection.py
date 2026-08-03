"""Does aiming at the author's baseline beat aiming at their centroid?

The gate has always ranked candidates by ``min(RMS-z)``, which targets the corpus
centroid. Nothing the author wrote lives there: their own pieces score at the
self-baseline. Measured consequence, which is what motivated this experiment: six
generations from three different voice profiles had a mean pairwise distance of
0.657 while seven of the author's own comparable pieces had 1.065. The outputs
were 38% tighter than the writer they imitate. That is what optimising toward a
mean produces, and it is the same collapse that showed up on an external author
as generations sitting closer to his mean than his own writing did.

``select: "band"`` ranks by ``|RMS-z - self_baseline|`` instead. Same slate, same
gate, same everything else; only the ranking rule changes.

The outcome measure is NOT mean RMS-z. Under band selection RMS-z would move
toward the baseline by construction, which proves nothing. What matters is
whether the outputs recover the author's own variety:

    spread ratio = mean pairwise distance among generations
                 / mean pairwise distance among the author's comparable pieces

1.0 means the engine produces as much variety as the author does. The current
engine sits near 0.6. Quality and fidelity are reported alongside, because a
selection rule that buys variety by shipping worse prose has not helped.

Both arms are scored on the same profile calibration, so the comparison is
internal and does not depend on any held-out ruler.

Usage:
  python evals/ab_selection.py --profile research-blend --briefs 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import config, gate, ingest, quality, scrub  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402

EVALS = Path(__file__).resolve().parent


def zvec(fp: Fingerprint, text: str) -> list[float]:
    _, zs = fp.distance_detail(text)
    return [zs[k] for k in fp.features]


def pairwise(fp: Fingerprint, texts: list[str]) -> float:
    vs = [zvec(fp, t) for t in texts]
    if len(vs) < 2:
        return 0.0
    d = [math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
         for a, b in itertools.combinations(vs, 2)]
    return statistics.mean(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--brief-file", default="briefs.json")
    ap.add_argument("--voice", help="brief 'voice' tag; defaults to --profile")
    ap.add_argument("--briefs", type=int, default=6)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", default=str(EVALS / "selection_ab"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((EVALS / args.brief_file).read_text(encoding="utf-8"))
    tag = args.voice or args.profile
    briefs = [(b["id"], b["task"]) for b in doc["briefs"]
              if b.get("voice") == tag][: args.briefs]
    if not briefs:
        print(f"no briefs tagged voice={tag!r} in {args.brief_file}")
        return 1

    prof = config.resolve(args.profile)
    fp = Fingerprint.load(prof.fingerprint_path)
    cal = scrub.ScrubCalibration.load(prof.scrub_path)
    qcal = quality.QualityCalibration(
        density_p25=cal.density_p25, specificity_p25=cal.specificity_p25)

    # The author's own spread, on pieces of comparable length to what we generate.
    corpus = [t for t in ingest.read_pieces(prof.db_path).values()
              if 180 <= len(t.split()) <= 500]
    if len(corpus) < 4:
        corpus = [t for t in ingest.read_pieces(prof.db_path).values()
                  if len(t.split()) >= 150]
    author_spread = pairwise(fp, corpus)
    print(f"{args.profile}: baseline {fp.self_baseline:.3f}, gate {fp.fit_threshold:.3f}")
    print(f"author spread over {len(corpus)} comparable pieces: {author_spread:.3f}\n")

    results = {}
    for mode in ("minimize", "band"):
        # format_config merges over profile defaults, so a format override is the
        # least invasive way to flip one knob without editing the profile on disk.
        prof_mode = config.resolve(args.profile)
        prof_mode.gate["select"] = mode
        texts, rows = [], []
        for bid, task in briefs:
            t0 = time.time()
            try:
                res = gate.compose(task, prof_mode, model=args.model)
                o = (res.output or "").strip()
            except Exception as e:
                print(f"  [{mode}/{bid}] FAILED {e}")
                continue
            if not o:
                continue
            texts.append(o)
            r = scrub.analyze(o, cal, fp=fp)
            q = quality.measure(o, qcal)
            slate = next((n for n in res.notes if n.startswith("slate spread")), "")
            rows.append({"brief": bid, "rmsz": r.fp_distance, "words": len(o.split()),
                         "quality_flags": q.flags, "redundancy": q.redundancy,
                         "density": q.density, "specificity": q.specificity,
                         "slate_note": slate, "secs": round(time.time() - t0, 1),
                         "text": o})
            print(f"  [{mode}/{bid}] rmsz {r.fp_distance:.3f}  "
                  f"q={','.join(q.flags) or 'clean'}  ({rows[-1]['secs']}s)")
        sp = pairwise(fp, texts)
        results[mode] = {
            "n": len(texts),
            "mean_rmsz": statistics.mean(r["rmsz"] for r in rows) if rows else None,
            "mean_abs_dev_from_baseline":
                statistics.mean(abs(r["rmsz"] - fp.self_baseline) for r in rows) if rows else None,
            "spread": sp,
            "spread_ratio": (sp / author_spread) if author_spread else 0.0,
            "quality_flagged": sum(1 for r in rows if r["quality_flags"]),
            "rows": rows,
        }
        print(f"  -> spread {sp:.3f} = {results[mode]['spread_ratio']:.0%} of author\n")

    (out / f"{args.profile}_selection_ab.json").write_text(
        json.dumps({"profile": args.profile, "author_spread": author_spread,
                    "self_baseline": fp.self_baseline, "arms": results}, indent=2),
        encoding="utf-8")

    print("=" * 66)
    print(f"{'arm':<10}{'n':>3}{'mean RMS-z':>12}{'|dev| from base':>17}"
          f"{'spread':>9}{'% of author':>13}{'q-flagged':>11}")
    for mode in ("minimize", "band"):
        r = results.get(mode)
        if not r or not r["n"]:
            continue
        print(f"{mode:<10}{r['n']:>3}{r['mean_rmsz']:>12.3f}"
              f"{r['mean_abs_dev_from_baseline']:>17.3f}{r['spread']:>9.3f}"
              f"{r['spread_ratio']:>12.0%}{r['quality_flagged']:>11}")
    print("=" * 66)
    print("Spread ratio is the outcome that matters: 100% means the engine produces")
    print("as much variety as the author. Mean RMS-z moving toward the baseline")
    print("under 'band' is tautological and is reported only for completeness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
