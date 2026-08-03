"""Find machine-written text already sitting inside a voice corpus.

Motivation, concrete: a harvest agent found `claudeSOPPSK.pdf` in the author's
Drive -- a 1,200-word first-person Statement of Purpose, AI-drafted, with unfilled
template placeholders still in the body. It was rejected at intake. But if one
such document exists outside the corpus, another may already be inside it, and a
contaminated corpus does not announce itself: it quietly teaches the voice model
to write like the machine it was supposed to be distinguished from.

Lexical screening will not find these. That document contains no "delve" or
"plethora"; its tells are structural. So this audits STRUCTURE, and it does so
against the author's own distribution rather than a universal threshold:

* **em-dash rate** -- already a fingerprint feature. This author's corpus averages
  ~0.004 per 100 words; LLM prose runs orders of magnitude higher. A high z here
  is the single strongest signal available.
* **template placeholders** -- `[University Name]`, `[insert X]`, `[Word count: ...]`.
  Decisive when present. Humans do not leave these in finished prose.
* **paragraph-length uniformity** -- machine prose produces suspiciously even
  paragraphs. Measured as the coefficient of variation of paragraph lengths;
  unusually LOW is the anomaly.
* **sentence-length uniformity** -- same logic, at sentence scale.
* **markdown structure in prose** -- bold headers and bullet scaffolding inside
  what should be continuous writing.
* **tricolon density** -- "x, y, and z" triples per 1000 words.

Every signal is reported as a z-score against the corpus's own mean, so the output
says "unlike the rest of this author's writing", not "unlike writing in general".
That framing matters: the goal is to find the intruder, and an intruder is defined
relative to the population it hides in.

Nothing is deleted. This prints a ranked suspicion table for a human to read.

Usage:
    python scripts/audit_corpus_purity.py                 # every profile
    python scripts/audit_corpus_purity.py personal creative
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import config, ingest  # noqa: E402

PLACEHOLDER = re.compile(r"\[(?:[A-Z][a-z]+ ){0,3}(?:Name|Title|Insert|insert|Word count|"
                         r"University|Company|Program|Date|X)\b[^\]]{0,60}\]")
BOLD_HEADER = re.compile(r"^\s*(\*\*.+\*\*|#{1,4}\s+\S)", re.MULTILINE)
BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
TRICOLON = re.compile(r"\b[\w']+,\s+[\w']+,\s+and\s+[\w']+\b")
WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")


def metrics(t: str) -> dict:
    w = WORD.findall(t)
    n = len(w) or 1
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    plens = [len(WORD.findall(p)) for p in paras] or [n]
    sents = [s for s in SENT.split(t) if s.strip()]
    slens = [len(WORD.findall(s)) for s in sents if WORD.findall(s)] or [n]

    def cv(xs):
        m = statistics.mean(xs)
        return (statistics.stdev(xs) / m) if len(xs) > 1 and m else 0.0

    return {
        "words": n,
        "emdash_per_100w": 100.0 * (t.count("—") + t.count("--")) / n,
        "placeholders": len(PLACEHOLDER.findall(t)),
        "para_cv": cv(plens),
        "sent_cv": cv(slens),
        "md_structure_per_1kw": 1000.0 * (len(BOLD_HEADER.findall(t)) + len(BULLET.findall(t))) / n,
        "tricolon_per_1kw": 1000.0 * len(TRICOLON.findall(t)) / n,
    }


# (metric, direction) -- direction +1 means HIGH is suspicious, -1 means LOW is.
SIGNALS = [
    ("emdash_per_100w", +1),
    ("md_structure_per_1kw", +1),
    ("tricolon_per_1kw", +1),
    ("para_cv", -1),
    ("sent_cv", -1),
]


def audit(slug: str) -> None:
    try:
        prof = config.resolve(slug)
    except Exception as e:
        print(f"{slug}: cannot resolve ({e})")
        return
    pieces = {k: v for k, v in ingest.read_pieces(prof.db_path).items()
              if len(WORD.findall(v)) >= 200}
    if len(pieces) < 8:
        print(f"{slug}: only {len(pieces)} pieces >=200w; too few to find an outlier "
              f"against. An intruder is only detectable relative to a population.")
        return

    rows = {k: metrics(v) for k, v in pieces.items()}
    stats = {}
    for m, _ in SIGNALS:
        vals = [r[m] for r in rows.values()]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[m] = (mu, sd)

    scored = []
    for name, r in rows.items():
        zs = {}
        for m, direction in SIGNALS:
            mu, sd = stats[m]
            z = ((r[m] - mu) / sd) if sd > 1e-9 else 0.0
            zs[m] = z * direction
        # Suspicion = sum of only the signals pointing the wrong way. Summing
        # signed values would let a normal-looking feature cancel a damning one.
        susp = sum(max(0.0, v) for v in zs.values()) + 10.0 * r["placeholders"]
        scored.append((susp, name, r, zs))
    scored.sort(reverse=True)

    print(f"\n=== {slug}: {len(pieces)} pieces, corpus means "
          f"em-dash {stats['emdash_per_100w'][0]:.3f}/100w, "
          f"para-CV {stats['para_cv'][0]:.2f}, sent-CV {stats['sent_cv'][0]:.2f}")
    print(f"{'piece':<34}{'susp':>6}{'words':>7}{'emdash':>8}{'md':>7}{'tri':>6}"
          f"{'pCV':>6}{'sCV':>6}  flags")
    for susp, name, r, zs in scored[:6]:
        flags = []
        if r["placeholders"]:
            flags.append(f"{r['placeholders']} TEMPLATE PLACEHOLDER(S)")
        for m, _ in SIGNALS:
            if zs[m] >= 2.0:
                flags.append(f"{m} {zs[m]:+.1f}sd")
        print(f"{name[:33]:<34}{susp:>6.1f}{r['words']:>7}"
              f"{r['emdash_per_100w']:>8.3f}{r['md_structure_per_1kw']:>7.1f}"
              f"{r['tricolon_per_1kw']:>6.1f}{r['para_cv']:>6.2f}{r['sent_cv']:>6.2f}"
              f"  {'; '.join(flags)}")
    worst = scored[0][0]
    if worst < 3.0:
        print("  -> nothing stands out. No piece is structurally unlike the rest.")
    else:
        print("  -> review the top rows by hand. High suspicion is an outlier, not a "
              "verdict: a genuinely unusual piece by the author scores the same way.")


def main() -> int:
    slugs = sys.argv[1:] or [
        s for s in config.list_profiles()
        if not (config.load_profile_config(s).get("compose_from"))
    ]
    for s in slugs:
        audit(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
