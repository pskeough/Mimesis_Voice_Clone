"""Which candidate scorer actually predicts the author's ranking?

Baseline from seed_t1_judgments.py: the shipped fingerprint scores 60% pairwise,
identical to a word-count control. That is the number to beat.

Every scorer here is computed inline rather than imported, so nothing gets credit
for being part of the existing engine. A feature earns selection by predicting
verdicts; that is the only defence against the Goodhart failure CADENCE_FINDINGS
already documented, where 26 features detected synthetic damage at AUROC 1.000
and still lost blind discrimination 25% to 25%.

n is small. One judgment, 10 pairs. Read this as "which hypotheses survive a first
contact with real data", not as a ranking with confidence behind it.
"""
import re
import sys
from collections import Counter

sys.path.insert(0, r"C:\AI Coding Projects\Online_AI\Mimesis\src")

from mimesis_voice import cli as up_cli, preference as pref  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402

_SENT = re.compile(r"[^.!?]+[.!?]+[\s\"']*", re.S)
_WORDS = re.compile(r"[A-Za-z']+")


def _sentences(t):
    return [s.strip() for s in _SENT.findall(t or "") if s.strip()]


def _openers(t, n=1):
    out = []
    for s in _sentences(t):
        w = _WORDS.findall(s)
        if w:
            out.append(" ".join(x.lower() for x in w[:n]))
    return out


def opener_top_share(t):
    ops = _openers(t)
    if not ops:
        return 1.0
    return Counter(ops).most_common(1)[0][1] / len(ops)


def opener_distinct(t):
    ops = _openers(t)
    return len(set(ops)) / len(ops) if ops else 0.0


def longest_opener_run(t):
    ops = _openers(t)
    best = run = 1
    for a, b in zip(ops, ops[1:]):
        run = run + 1 if a == b and a else 1
        best = max(best, run)
    return float(best)


def commas_per_sentence(t):
    s = _sentences(t)
    return (t.count(",") / len(s)) if s else 0.0


def or_restatements(t):
    return float(len(re.findall(r",\s+or\s+(?:he|she|it|they|the|a|an)\b", t or "", re.I)))


def main():
    prof = up_cli._resolve("creative")
    fp = Fingerprint.load(prof.fingerprint_path)

    scorers = {
        "fingerprint (shipped)":      fp.distance,
        "word count (control)":       lambda t: len(t.split()),
        "opener top-share":           opener_top_share,
        "opener distinct":            opener_distinct,
        "longest opener run":         longest_opener_run,
        "commas per sentence":        commas_per_sentence,
        "'X, or Y' count":            or_restatements,
    }
    # distinct openers: HIGHER is better. everything else: lower.
    direction = {n: True for n in scorers}
    direction["opener distinct"] = False

    print("Pairwise agreement with the author's ranking (50% = chance, n=10 pairs)\n")
    for r in pref.compare(prof, scorers, lower_is_better=direction):
        bar = "#" * int(r.accuracy * 20)
        print(f"  {r.name:24} {r.accuracy*100:5.0f}%  tau={r.kendall_tau:+.2f}  {bar}")

    print("\nn is 10 pairs from one judgment. Nothing here is significant; this "
          "separates live hypotheses from dead ones.")


if __name__ == "__main__":
    main()
