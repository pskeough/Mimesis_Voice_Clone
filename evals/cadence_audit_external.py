"""Cadence audit for an ARBITRARY corpus + generation set, outside this repo.

``cadence_audit.py`` is wired to this repo's own profiles. This one takes paths,
so the same instrument can be pointed at any author's corpus and any engine's
outputs -- which is how the finding gets tested for external validity instead of
resting on a single author.

Nothing is written inside this repository: ``--out`` is required and should live
next to the corpus being audited. Only aggregate numbers are emitted, never
corpus or generation text.

Usage:
  python evals/cadence_audit_external.py \
      --corpus  <dir with .txt/.md files>  [--corpus ...] \
      --gen-jsonl <file.jsonl> --gen-field output \
      --out <dir outside this repo> --label <name>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import cadence as cadmod  # noqa: E402
from mimesis_voice import fingerprint as fpmod  # noqa: E402

MIN_WORDS = 120


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    return sum(
        1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg
    ) / (len(pos) * len(neg))


def block(fp: fpmod.Fingerprint, text: str, names: tuple[str, ...]) -> float:
    _, zs = fp.distance_detail(text)
    return math.sqrt(sum(zs[f] ** 2 for f in names) / len(names))


def read_corpus(dirs: list[str]) -> list[str]:
    """Read .txt/.md/.docx. Reusing ingest's extractor matters: a first pass here
    skipped .docx and silently audited whatever stray text files were nearby,
    producing a plausible-looking table built on the wrong corpus."""
    from mimesis_voice import ingest

    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".txt", ".md", ".docx"}:
                continue
            try:
                if p.suffix.lower() == ".docx":
                    t = "\n\n".join(ingest._extract_docx(p))
                else:
                    t = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                print(f"  skip {p.name}: {exc}")
                continue
            if len(t.split()) >= MIN_WORDS:
                out.append(t)
    return out


def dig(obj, field: str):
    """Pull ``field`` out of a nested record, first match wins."""
    if isinstance(obj, dict):
        if field in obj and isinstance(obj[field], str):
            return obj[field]
        for v in obj.values():
            r = dig(v, field)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = dig(v, field)
            if r:
                return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--gen-jsonl")
    ap.add_argument("--gen-field", default="output")
    ap.add_argument("--gen-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="external")
    args = ap.parse_args()

    outdir = Path(args.out)
    if ROOT in outdir.resolve().parents or outdir.resolve() == ROOT:
        print("refusing to write audit output inside the Mimesis repo; pick --out elsewhere")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = read_corpus(args.corpus)
    if len(corpus) < 10:
        print(f"only {len(corpus)} usable pieces (>= {MIN_WORDS} words); need >= 10")
        return

    fp1 = fpmod.calibrate(corpus, feature_set="v1")
    fp2 = fpmod.calibrate(corpus, feature_set="v2")
    print(f"[{args.label}] corpus n={len(corpus)}  "
          f"v1 self-baseline {fp1.self_baseline:.3f}  v2 {fp2.self_baseline:.3f}")
    print(f"           sentence cuts (author's own short/mid/long): "
          f"{fp2.cuts[0]:.1f} / {fp2.cuts[1]:.1f} words")

    # --- 1. Blindness replication: does v1 see a destroyed rhythm on THIS author?
    def rebuild(sents):
        out, buf = [], []
        for i, s in enumerate(sents, 1):
            buf.append(s)
            if i % 5 == 0:
                out.append(" ".join(buf)); buf = []
        if buf:
            out.append(" ".join(buf))
        return "\n\n".join(out)

    real1, ramp1, real2, ramp2 = [], [], [], []
    for t in corpus:
        s = fpmod._sentences(t)
        if len(s) < 12:
            continue
        ctrl = rebuild(s)
        rmp = rebuild(sorted(s, key=lambda x: len(fpmod._WORD.findall(x))))
        real1.append(fp1.distance(ctrl)); ramp1.append(fp1.distance(rmp))
        real2.append(fp2.distance(ctrl)); ramp2.append(fp2.distance(rmp))

    a1, a2 = auroc(ramp1, real1), auroc(ramp2, real2)
    print(f"\n  rhythm-destruction AUROC   v1 {a1:.3f}   v2 {a2:.3f}   (n={len(real1)})")

    # --- 2. Human reference, leave-one-out on the cadence + surface blocks.
    names = fp2.features
    rows = [fpmod.extract_all(t, "v2", fp2.cuts) for t in corpus]
    h_cad, h_surf = [], []
    for i, t in enumerate(corpus):
        rest = rows[:i] + rows[i + 1:]
        loo = fpmod.Fingerprint(
            means={f: statistics.mean(r[f] for r in rest) for f in names},
            stds={f: statistics.stdev([r[f] for r in rest]) for f in names},
            feature_set="v2", cuts=fp2.cuts,
        )
        h_cad.append(block(loo, t, cadmod.CADENCE_FEATURES))
        h_surf.append(block(loo, t, fpmod.FEATURES))

    # --- 3. Generations.
    gens: list[str] = []
    if args.gen_jsonl:
        for line in Path(args.gen_jsonl).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = dig(rec, args.gen_field)
            if t and len(t.split()) >= 80:
                gens.append(t)
    if args.gen_dir:
        for p in sorted(Path(args.gen_dir).rglob("*")):
            if p.suffix.lower() in {".txt", ".md"} and p.is_file():
                t = p.read_text(encoding="utf-8", errors="ignore")
                if len(t.split()) >= 80:
                    gens.append(t)

    summary = {
        "label": args.label,
        "corpus_n": len(corpus),
        "v1_self_baseline": fp1.self_baseline,
        "v2_self_baseline": fp2.self_baseline,
        "cuts": list(fp2.cuts),
        "auroc_rhythm_v1": a1,
        "auroc_rhythm_v2": a2,
        "human_cadence": statistics.mean(h_cad),
        "human_surface": statistics.mean(h_surf),
    }

    print(f"\n{'set':<14}{'n':>5}{'cadence z':>12}{'surface z':>12}"
          f"{'AUROC cad':>12}{'AUROC surf':>12}")
    print("-" * 67)
    print(f"{'HUMAN':<14}{len(corpus):>5}{statistics.mean(h_cad):>12.3f}"
          f"{statistics.mean(h_surf):>12.3f}{0.5:>12.3f}{0.5:>12.3f}")

    if gens:
        g_cad = [block(fp2, t, cadmod.CADENCE_FEATURES) for t in gens]
        g_surf = [block(fp2, t, fpmod.FEATURES) for t in gens]
        ac, as_ = auroc(g_cad, h_cad), auroc(g_surf, h_surf)
        summary.update({
            "gen_n": len(gens),
            "gen_cadence": statistics.mean(g_cad),
            "gen_surface": statistics.mean(g_surf),
            "auroc_cadence": ac,
            "auroc_surface": as_,
        })
        print(f"{'GENERATED':<14}{len(gens):>5}{statistics.mean(g_cad):>12.3f}"
              f"{statistics.mean(g_surf):>12.3f}{ac:>12.3f}{as_:>12.3f}")

        acc = {f: [] for f in cadmod.CADENCE_FEATURES}
        for t in gens:
            _, zs = fp2.distance_detail(t)
            for f in cadmod.CADENCE_FEATURES:
                acc[f].append(zs[f])
        ranked = sorted(acc.items(), key=lambda kv: abs(statistics.mean(kv[1])), reverse=True)
        summary["worst_cadence"] = [
            {"feature": f, "mean_z": statistics.mean(v)} for f, v in ranked[:8]
        ]
        print(f"\nWorst cadence features across {len(gens)} generations:")
        for f, v in ranked[:6]:
            mz = statistics.mean(v)
            print(f"  {f:<22}{mz:>+7.2f} SD   {cadmod.hint(f, mz)}")
    else:
        print("(no generations supplied)")

    (outdir / f"{args.label}_cadence_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {outdir / (args.label + '_cadence_audit.json')}")


if __name__ == "__main__":
    main()
