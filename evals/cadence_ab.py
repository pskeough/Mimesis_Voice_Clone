"""v1 vs v2 fingerprint: does adding cadence features buy discrimination?

Three questions, all offline and judge-free:

1. **Rhythm sensitivity.** Sorting a real piece's sentences shortest-to-longest
   destroys its cadence while holding every word constant. v1 is provably blind
   to this. How far does v2 move, in units of the author's own self-baseline?

2. **Does it still fit the author?** A metric that flags rhythm-destroyed text
   is worthless if it also flags the author's real writing. Reported as the
   leave-one-out self-baseline and the p95 fit threshold for each version.

3. **Separation.** The number that matters: distance-to-author for real pieces
   vs for rhythm-destroyed pieces, expressed as AUROC. A metric that cannot
   separate them scores 0.5. This is computed on held-out pieces against a
   fingerprint calibrated WITHOUT them, so it is not circular.

Run:  .venv/Scripts/python.exe evals/cadence_ab.py [voice]
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import fingerprint as fpmod  # noqa: E402
from mimesis_voice.fingerprint import _sentences, _WORD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260729


def rebuild(sents: list[str]) -> str:
    out, buf = [], []
    for i, s in enumerate(sents, 1):
        buf.append(s)
        if i % 5 == 0:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a random positive scores higher than a random negative). Ties = 0.5."""
    if not pos or not neg:
        return 0.5
    wins = sum(
        1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg))


def load_docs(voice: str) -> list[str]:
    docs = []
    for p in sorted((ROOT / "profiles" / voice / "source_documents").glob("*.txt")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if len(_WORD.findall(t)) >= 200 and len(_sentences(t)) >= 12:
            docs.append(t)
    return docs


def main() -> None:
    rng = random.Random(SEED)
    voice = sys.argv[1] if len(sys.argv) > 1 else "creative"
    docs = load_docs(voice)
    if len(docs) < 20:
        print(f"{voice}: only {len(docs)} usable docs; need >=20 for a split")
        return

    idx = list(range(len(docs)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    train = [docs[i] for i in idx[:cut]]
    test = [docs[i] for i in idx[cut:]]

    print(f"voice={voice}  calibrate n={len(train)}  held-out n={len(test)}")
    print()

    for fs in ("v1", "v2"):
        fp = fpmod.calibrate(train, feature_set=fs)
        # Every condition goes through rebuild(), including the control. rebuild
        # regroups paragraphs to a fixed 5 sentences, which shifts para_len_mean;
        # scoring raw text against rebuilt variants would credit the paragraph
        # reformat to cadence. Order is the only thing that differs here.
        real, ramp, shuf = [], [], []
        for t in test:
            s = _sentences(t)
            real.append(fp.distance(rebuild(s)))
            ramp.append(fp.distance(rebuild(sorted(s, key=lambda x: len(_WORD.findall(x))))))
            sh = s[:]
            rng.shuffle(sh)
            shuf.append(fp.distance(rebuild(sh)))

        print(f"--- {fs}  ({len(fp.features)} features)")
        print(f"    self_baseline {fp.self_baseline:.4f}   fit_threshold {fp.fit_threshold:.4f}")
        print(f"    held-out real       mean {statistics.mean(real):.4f}")
        print(f"    shuffled sentences  mean {statistics.mean(shuf):.4f}"
              f"   (+{statistics.mean(shuf)-statistics.mean(real):.4f})")
        print(f"    sorted ramp         mean {statistics.mean(ramp):.4f}"
              f"   (+{statistics.mean(ramp)-statistics.mean(real):.4f})")
        print(f"    AUROC real vs shuffled : {auroc(shuf, real):.3f}")
        print(f"    AUROC real vs ramp     : {auroc(ramp, real):.3f}")
        flagged = sum(1 for d in ramp if d > fp.fit_threshold)
        fp_real = sum(1 for d in real if d > fp.fit_threshold)
        print(f"    ramp flagged off-voice : {flagged}/{len(ramp)}"
              f"    real false-flagged: {fp_real}/{len(real)}")
        if fs == "v2":
            cad_real = [fp.cadence_distance(rebuild(_sentences(t))) for t in test]
            cad_ramp = [
                fp.cadence_distance(
                    rebuild(sorted(_sentences(t), key=lambda x: len(_WORD.findall(x))))
                )
                for t in test
            ]
            print(f"    cadence-block only: real {statistics.mean(cad_real):.4f} "
                  f"vs ramp {statistics.mean(cad_ramp):.4f}  "
                  f"AUROC {auroc(cad_ramp, cad_real):.3f}")
        print()


if __name__ == "__main__":
    main()
