"""End-to-end A/B: does gating on cadence produce more author-like prose?

The offline test (``cadence_ab.py``) proves the v2 metric can *see* rhythm. This
one asks the question that actually matters: if you put v2 in the compose loop,
do the generations come out closer to the author?

Non-circularity is the whole design here. Gating on a metric and then reporting
that metric is training on the test set -- the v1 engine's headline "RMS-z 0.78x
self-baseline" is exactly that, since the gate selects candidates to minimise
RMS-z. So:

  * the corpus is split 70/30 by document;
  * BOTH arms are calibrated on split A only, and generate under that gate;
  * every output is scored on a v2 ruler calibrated on split B, which neither
    arm has ever seen.

Both arms retrieve anchors from the full store (identical retrieval, so no
differential advantage) and share the frozen brief set, the generator model, and
the seed. The only difference is which feature set the gate optimises.

Run:  .venv/Scripts/python.exe evals/ab_cadence_e2e.py [--voice creative] [--briefs 7]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import config, gate, ingest  # noqa: E402
from mimesis_voice import fingerprint as fpmod  # noqa: E402

SEED = 20260729
OUT = Path(__file__).resolve().parent / "cadence_e2e"


def segment(texts: list[str], target: int) -> list[str]:
    """Cut long pieces into ~``target``-word segments at paragraph boundaries.

    A corpus of book chapters runs 353 to 17,717 words per file. Calibrating
    across that range makes the feature standard deviations a measure of
    document length rather than of style, and leaves too few pieces to estimate
    26 features from. Segmenting gives many comparable units. Off by default
    (target=0), since a corpus of already-comparable pieces needs none.
    """
    if target <= 0:
        return texts
    out: list[str] = []
    for t in texts:
        paras = [p for p in t.split("\n\n") if p.strip()]
        buf: list[str] = []
        n = 0
        for p in paras:
            buf.append(p)
            n += len(p.split())
            if n >= target:
                out.append("\n\n".join(buf))
                buf, n = [], 0
        if n >= target // 3 and buf:  # keep a substantial tail, drop scraps
            out.append("\n\n".join(buf))
    return out


def split_corpus(pieces: dict[str, str], frac: float = 0.7):
    keys = sorted(pieces)
    random.Random(SEED).shuffle(keys)
    cut = int(len(keys) * frac)
    return (
        [pieces[k] for k in keys[:cut]],
        [pieces[k] for k in keys[cut:]],
        set(keys[cut:]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="creative")
    ap.add_argument("--briefs", type=int, default=7)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--brief-file", default="briefs.json")
    ap.add_argument("--v1-profile", help="defaults to --voice")
    ap.add_argument("--v2-profile", help="defaults to <voice>-v2")
    ap.add_argument("--out", help="defaults to evals/cadence_e2e")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--segment-words", type=int, default=0,
                    help="cut pieces into ~N-word segments; 0 = off")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((Path(__file__).resolve().parent / args.brief_file).read_text())
    items = [
        (b["id"], b["task"])
        for b in doc["briefs"]
        if b.get("voice") == args.voice
    ][: args.briefs]
    if not items:
        print(f"no briefs for voice '{args.voice}' in {args.brief_file}")
        return

    v1 = config.resolve(args.v1_profile or args.voice)
    v2 = config.resolve(args.v2_profile or f"{args.voice}-v2")

    pieces = ingest.read_pieces(v1.db_path)
    train, test, test_keys = split_corpus(pieces, args.split)
    if args.segment_words:
        train = segment(train, args.segment_words)
        test = segment(test, args.segment_words)
    print(f"corpus {len(pieces)} pieces -> calibrate {len(train)} / eval-ruler {len(test)}"
          f"{f' (segmented at ~{args.segment_words}w)' if args.segment_words else ''}")

    # Both arms gate on a split-A calibration. Overwrite the live fingerprints.
    for prof, fs in ((v1, "v1"), (v2, "v2")):
        fp = fpmod.calibrate(train, feature_set=fs)
        fp.save(prof.fingerprint_path)
        print(f"  {prof.slug:<16} gate={fs} self-baseline {fp.self_baseline:.3f} "
              f"fit_threshold {fp.fit_threshold:.3f}")

    # The judge ruler: v2, split B, never seen by either gate.
    ruler = fpmod.calibrate(test, feature_set="v2")
    ruler.save(out / "ruler_splitB_v2.json")
    print(f"  ruler(split B, v2)  self-baseline {ruler.self_baseline:.3f}\n")

    rows = []
    for arm, prof in (("v1", v1), ("v2", v2)):
        for i, (bid, brief) in enumerate(items):
            t0 = time.time()
            try:
                res = gate.compose(
                    brief, prof, model=args.model, exclude_files=test_keys
                )
            except Exception as exc:  # keep the sweep alive
                print(f"[{arm}/{bid}] FAILED: {exc}")
                continue
            if not res.output:
                print(f"[{arm}/{bid}] no output")
                continue
            rec = {
                "arm": arm,
                "brief_i": i,
                "brief_id": bid,
                "text": res.output,
                "gate_rmsz": res.chosen.rmsz if res.chosen else None,
                "notes": res.notes,
                "ruler_rmsz": ruler.distance(res.output),
                "ruler_cadence": ruler.cadence_distance(res.output),
                "secs": round(time.time() - t0, 1),
            }
            rows.append(rec)
            (out / f"{arm}_{bid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[{arm}/{bid}] ruler_rmsz={rec['ruler_rmsz']:.3f} "
                  f"cadence={rec['ruler_cadence']:.3f} ({rec['secs']}s)")

    (out / "all.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Human reference: split-B pieces scored on the split-B ruler is circular, so
    # score split-A real pieces on the split-B ruler. That is the number a
    # generation would have to reach to be indistinguishable from real writing.
    ref_rmsz = [ruler.distance(t) for t in train]
    ref_cad = [ruler.cadence_distance(t) for t in train]

    print("\n" + "=" * 62)
    print(f"{'arm':<10}{'n':>4}{'ruler RMS-z':>14}{'cadence':>12}{'vs human':>14}")
    print("-" * 62)
    hr, hc = statistics.mean(ref_rmsz), statistics.mean(ref_cad)
    print(f"{'HUMAN':<10}{len(train):>4}{hr:>14.3f}{hc:>12.3f}{'--':>14}")
    for arm in ("v1", "v2"):
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        m = statistics.mean(r["ruler_rmsz"] for r in sel)
        c = statistics.mean(r["ruler_cadence"] for r in sel)
        print(f"{arm:<10}{len(sel):>4}{m:>14.3f}{c:>12.3f}{c - hc:>+14.3f}")
    print("=" * 62)
    print("cadence column is the 13 order-aware features only; 'vs human' is how")
    print("far the arm's rhythm sits from real writing on the same unseen ruler.")


if __name__ == "__main__":
    main()
