"""A/B the opener rule on the PRODUCTION path (claude -p), not the local model.

Same design as ab_opener_rule.py but generation goes through gate.claude_generate,
which is what the shipped engine actually calls. The local variant is left in
place; its slate calls returned zero candidates in 16 minutes and that is a
separate problem belonging to the local backend.

Nothing is written back to the profile. Arm A reads the calibration on disk
verbatim; arm B uses a freshly computed one that includes opener statistics.

Every candidate is reported, not just the best, because the question is whether
the rule moves the distribution the gate must choose from. A rule that yields one
good draft among four has not helped: the gate could already pick a good one when
it happened to exist.
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter

M = r"C:\AI Coding Projects\Online_AI\Mimesis"
sys.path.insert(0, os.path.join(M, "src"))

from mimesis_voice import cli as up_cli, gate, scrub as S  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402
from mimesis_voice.scrub import ScrubCalibration  # noqa: E402

N_SLATE = 4
VOICE = sys.argv[1] if len(sys.argv) > 1 else "creative"
OUT = os.path.join(r"C:\LocalAI\LLM\eval\basilisk\ab_cloud", VOICE)

TASKS = {
    "creative": (
        "Write the opening of a chapter in a literary post-apocalyptic novel. "
        "Third person past, welded close to one character's perception. A man wakes "
        "in a cold room in a town where everyone is dead, and goes outside. "
        "About 450 words. Return only the prose."),
    "research": (
        "Write the Methods subsection describing an LLM-as-judge protocol, about 280 "
        "words. State only these facts: three votes per item from a single judge model "
        "at temperature 0.7 resolved by majority; a documented fallback judge on quota "
        "exhaustion; a 400-item stratified sample re-judged by two independent model "
        "families on a byte-identical prompt, reporting verdict agreement percentage "
        "and Pearson r; 186 hand-labelled items for human calibration, reporting "
        "Cohen's kappa and a sensitivity table stratified by severity band; pairwise "
        "comparisons run in both orders with agreement required; a de-verbosity "
        "correction with length-driven tie-flips counted. Return only the prose."),
}
TASK = TASKS.get(VOICE, TASKS["creative"])

_SENT = re.compile(r"[^.!?]+[.!?]+[\s\"']*", re.S)
_W = re.compile(r"[A-Za-z']+")


def sentences(t):
    return [s.strip() for s in _SENT.findall(t or "") if s.strip()]


def measure(t, fp):
    sents = sentences(t)
    ops = []
    for s in sents:
        w = _W.findall(s)
        if w:
            ops.append(w[0].lower())
    if len(ops) < 5:
        return None
    c = Counter(ops)
    run = best = 1
    for a, b in zip(ops, ops[1:]):
        run = run + 1 if a == b and a else 1
        best = max(best, run)
    try:
        d = float(fp.distance(t))
    except Exception:
        d = float("nan")
    return {"words": len(t.split()), "sentences": len(ops),
            "distinct": len(c) / len(ops),
            "top": c.most_common(1)[0][1] / len(ops),
            "top_word": c.most_common(1)[0][0],
            "fronted": sum(1 for s in sents if S._is_fronted(s)) / len(sents),
            "max_run": best, "fp": d,
            "excerpt": " ".join(sents[:2])[:200], "text": t}


def main():
    os.makedirs(OUT, exist_ok=True)
    prof = up_cli._resolve(VOICE)
    fp = Fingerprint.load(prof.fingerprint_path)

    cal_old = ScrubCalibration.load(prof.scrub_path)
    docs = glob.glob(os.path.join(prof.root, "source_documents", "*"))
    texts = []
    for f in docs[:400]:
        try:
            texts.append(open(f, encoding="utf-8-sig", errors="replace").read())
        except Exception:
            pass
    cal_new = S.calibrate(texts)

    print(f"[{VOICE}] corpus: {cal_new.n_opener_pieces} pieces measured")
    print(f"[{VOICE}] author: distinct_p25={cal_new.opener_distinct_p25*100:.0f}%  "
          f"top_p75={cal_new.opener_top_p75*100:.0f}%  "
          f"fronted_p50={cal_new.opener_fronted_p50*100:.0f}%\n", flush=True)

    arms = {}
    for name, cal in (("A_no_rule", cal_old), ("B_opener_rule", cal_new)):
        kit = gate.build_kit(TASK, prof, cal, n_examples=5)
        has = "put something before the" in kit
        print(f"{name}: rule in kit = {has}", flush=True)
        t0 = time.time()
        try:
            raw = gate.claude_generate(gate._slate_prompt(kit, N_SLATE), model="sonnet")
        except Exception as e:
            print(f"  generation FAILED: {e}", flush=True)
            arms[name] = []
            continue
        dt = time.time() - t0
        open(os.path.join(OUT, f"raw_{name}.txt"), "w", encoding="utf-8").write(raw)
        cands = gate._parse_slate(raw, N_SLATE)
        rows = []
        for i, t in enumerate(cands):
            m = measure(t, fp)
            if m:
                m["i"] = i
                rows.append(m)
        arms[name] = rows
        print(f"  {len(rows)}/{len(cands)} usable in {dt:.0f}s "
              f"(raw {len(raw)} chars, separator present={gate._SLATE_MARK in raw})",
              flush=True)

    print(f"\n{'arm':16} {'#':>2} {'words':>6} {'distinct':>9} {'top':>12} "
          f"{'fronted':>8} {'run':>4} {'fp':>6}")
    print("-" * 74)
    for name, rows in arms.items():
        for r in rows:
            print(f"{name:16} {r['i']:>2} {r['words']:>6} {r['distinct']*100:>8.0f}% "
                  f"{r['top_word'] + ' ' + str(round(r['top']*100)) + '%':>12} "
                  f"{r['fronted']*100:>7.0f}% {r['max_run']:>4} {r['fp']:>6.2f}")

    print()
    for name, rows in arms.items():
        if not rows:
            print(f"{name}: no candidates")
            continue
        n = len(rows)
        print(f"{name:16} MEAN distinct={sum(r['distinct'] for r in rows)/n*100:.0f}%  "
              f"top={sum(r['top'] for r in rows)/n*100:.0f}%  "
              f"fronted={sum(r['fronted'] for r in rows)/n*100:.0f}%  "
              f"fp={sum(r['fp'] for r in rows)/n:.2f}")

    print(f"\nauthor reference: distinct {cal_new.opener_distinct_p25*100:.0f}%  "
          f"top {cal_new.opener_top_p75*100:.0f}%  fronted {cal_new.opener_fronted_p50*100:.0f}%")

    print("\nEXCERPTS\n")
    for name, rows in arms.items():
        for r in rows:
            print(f"--- {name} #{r['i']} ---\n    {r['excerpt']}\n")

    json.dump(arms, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
