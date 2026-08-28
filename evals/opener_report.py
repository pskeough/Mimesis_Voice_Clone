"""One table: the author's corpus, every prior generated arm, and the A/B arms.

Prior arms carry the author's actual blind grades where he gave them, so the
metric can be read against human judgment rather than against itself.

Excerpts are the first two sentences of each text. A table of ratios is not
evidence about prose; the excerpt is there so the numbers can be checked against
something readable.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

M = r"C:\AI Coding Projects\Online_AI\Mimesis"
sys.path.insert(0, os.path.join(M, "src"))
from mimesis_voice import cli as up_cli, scrub as S  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402

BAS = r"C:\LocalAI\LLM\eval\basilisk"
AB = os.path.join(BAS, "ab_opener", "ab_results.json")

_SENT = re.compile(r"[^.!?]+[.!?]+[\s\"']*", re.S)
_W = re.compile(r"[A-Za-z']+")

PRIOR = [
    ("your Day 1 chapter",      "T1__HUMAN.json",        "A"),
    ("Opus + mimesis (T1)",     "T1__OPUS_mimesis.json", "B+"),
    ("Qwen + mimesis (T1)",     "T1__QWEN_mimesis.json", "B"),
    ("Qwen bare (T1)",          "T1__QWEN_bare.json",    "B-"),
    ("Opus bare (T1)",          "T1__OPUS_bare.json",    "F"),
    ("Qwen bare (T3)",          "T3__QWEN_bare.json",    "-"),
    ("Qwen + mimesis (T3)",     "T3__QWEN_mimesis.json", "-"),
]


def sentences(t):
    return [s.strip() for s in _SENT.findall(t or "") if s.strip()]


def measure(t, fp):
    sents = sentences(t)
    ops = []
    for s in sents:
        w = _W.findall(s)
        if w:
            ops.append(w[0].lower())
    if not ops:
        return None
    c = Counter(ops)
    fronted = sum(1 for s in sents if S._is_fronted(s)) / len(sents)
    try:
        d = float(fp.distance(t))
    except Exception:
        d = float("nan")
    return {"distinct": len(c) / len(ops), "top": c.most_common(1)[0][1] / len(ops),
            "top_word": c.most_common(1)[0][0], "fronted": fronted,
            "words": len(t.split()), "fp": d,
            "excerpt": " ".join(sents[:2])[:190]}


def main():
    prof = up_cli._resolve("creative")
    fp = Fingerprint.load(prof.fingerprint_path)

    docs = glob.glob(os.path.join(prof.root, "source_documents", "*"))
    texts = []
    for f in docs[:400]:
        try:
            texts.append(open(f, encoding="utf-8-sig", errors="replace").read())
        except Exception:
            pass
    cal = S.calibrate(texts)

    rows = []
    for label, fname, grade in PRIOR:
        p = os.path.join(BAS, fname)
        if not os.path.exists(p):
            continue
        t = json.load(open(p, encoding="utf-8-sig")).get("content", "")
        m = measure(t, fp)
        if m:
            m["label"], m["grade"] = label, grade
            rows.append(m)

    if os.path.exists(AB):
        ab = json.load(open(AB, encoding="utf-8"))
        for arm, cands in ab.items():
            for c in cands:
                m = measure(c["text"], fp)
                if m:
                    m["label"] = f"{arm} #{c['i']}"
                    m["grade"] = "new"
                    rows.append(m)

    print("AUTHOR CORPUS BASELINE  "
          f"(n={cal.n_opener_pieces} pieces)   distinct p25={cal.opener_distinct_p25*100:.0f}%   "
          f"top p75={cal.opener_top_p75*100:.0f}%   fronted median={cal.opener_fronted_p50*100:.0f}%")
    print()
    print(f"{'source':26} {'grade':>5} {'words':>6} {'distinct':>9} {'top':>12} "
          f"{'fronted':>8} {'fp':>6}")
    print("-" * 82)
    for r in rows:
        top = f"{r['top_word']} {r['top']*100:.0f}%"
        print(f"{r['label']:26} {r['grade']:>5} {r['words']:>6} "
              f"{r['distinct']*100:>8.0f}% {top:>12} {r['fronted']*100:>7.0f}% {r['fp']:>6.2f}")

    print("\nEXCERPTS (first two sentences)\n")
    for r in rows:
        print(f"--- {r['label']} [{r['grade']}] ---")
        print(f"    {r['excerpt']}\n")


if __name__ == "__main__":
    main()
