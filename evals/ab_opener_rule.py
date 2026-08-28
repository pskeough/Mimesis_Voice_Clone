"""A/B the opener rule: same task, same model, same slate size, kit differs only
in whether the opener rule is present.

The old calibration is read from disk and used verbatim for arm A. Arm B uses a
freshly computed calibration with opener statistics. Nothing is written back:
the live profile keeps its current kit either way.

Reports every candidate, not just the winner, because the question is whether the
rule shifts the DISTRIBUTION the gate has to choose from. A rule that produces one
good candidate and three bad ones has not helped; the gate could already pick a
good one when it existed.
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter

M = r"C:\AI Coding Projects\Online_AI\Mimesis"
sys.path.insert(0, os.path.join(M, "src"))

from mimesis_voice import cli as up_cli, gate, scrub as S  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402
from mimesis_voice.scrub import ScrubCalibration  # noqa: E402

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
N_SLATE = 4
VOICE = sys.argv[1] if len(sys.argv) > 1 else "creative"
OUT = os.path.join(r"C:\LocalAI\LLM\eval\basilisk\ab_opener", VOICE)

# One task per voice. The research task is deliberately a real methods paragraph:
# academic prose has less room to front phrases than fiction does, so if the rule
# is going to over-apply and read as mannered, it will show up here first.
TASKS = {
    "creative": (
        "Write the opening of a chapter in a literary post-apocalyptic novel. "
        "Third person past, welded close to one character's perception. A man wakes "
        "in a cold room in a town where everyone is dead, and goes outside. "
        "About 500 words. Return only the prose."),
    "research": (
        "Write the Methods subsection describing an LLM-as-judge protocol, about 300 "
        "words. Facts to state and nothing beyond them: three votes per item from a "
        "single judge model at temperature 0.7, resolved by majority; a documented "
        "fallback judge on quota exhaustion; a 400-item stratified sample re-judged by "
        "two independent model families on a byte-identical prompt, reporting verdict "
        "agreement percentage and Pearson r; 186 hand-labelled items for human "
        "calibration, reporting Cohen's kappa and a sensitivity table stratified by "
        "severity band; pairwise comparisons run in both orders with agreement "
        "required; a de-verbosity correction with length-driven tie-flips counted. "
        "Return only the prose."),
}
TASK = TASKS.get(VOICE, TASKS["creative"])

_SENT = re.compile(r"[^.!?]+[.!?]+[\s\"']*", re.S)
_W = re.compile(r"[A-Za-z']+")


def sentences(t):
    return [s.strip() for s in _SENT.findall(t or "") if s.strip()]


def openers(t):
    out = []
    for s in sentences(t):
        w = _W.findall(s)
        if w:
            out.append(w[0].lower())
    return out


def metrics(t):
    ops = openers(t)
    if not ops:
        return None
    c = Counter(ops)
    sents = sentences(t)
    fronted = sum(1 for s in sents if S._is_fronted(s)) / len(sents) if sents else 0
    run = best = 1
    for a, b in zip(ops, ops[1:]):
        run = run + 1 if a == b and a else 1
        best = max(best, run)
    return {"words": len(t.split()), "sentences": len(ops),
            "distinct": len(c) / len(ops), "top_share": c.most_common(1)[0][1] / len(ops),
            "top_word": c.most_common(1)[0][0], "fronted": fronted, "max_run": best}


def generate(kit, n, tag=""):
    """One slate call. Saves the raw response: a run that yields zero candidates
    is a fact about the harness, and discarding the body makes it undiagnosable.
    The first attempt returned 0 usable candidates in 482s with no record of why.
    """
    body = json.dumps({"messages": [{"role": "user", "content": gate._slate_prompt(kit, n)}],
                       "temperature": 1.0, "top_p": 0.95, "max_tokens": 12000,
                       "reasoning_effort": "low"}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    raw = msg.get("content", "") or ""
    think = msg.get("reasoning_content", "") or ""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"raw_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(f"finish={d['choices'][0].get('finish_reason')}\n"
                f"content_chars={len(raw)} thinking_chars={len(think)}\n"
                f"{'='*60}\nCONTENT:\n{raw}\n{'='*60}\nTHINKING (first 2000):\n{think[:2000]}")
    print(f"    raw: content={len(raw)} chars, thinking={len(think)} chars, "
          f"finish={d['choices'][0].get('finish_reason')}", flush=True)
    return gate._parse_slate(raw, n), time.time() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    prof = up_cli._resolve(VOICE)
    fp = Fingerprint.load(prof.fingerprint_path)

    cal_old = ScrubCalibration.load(prof.scrub_path)          # exactly what ships today
    docs = glob.glob(os.path.join(prof.root, "source_documents", "*"))
    texts = []
    for f in docs[:400]:
        try:
            texts.append(open(f, encoding="utf-8-sig", errors="replace").read())
        except Exception:
            pass
    cal_new = S.calibrate(texts)                              # adds opener stats

    print(f"corpus: {cal_new.n_opener_pieces} pieces measured for openers")
    print(f"author floor: distinct_p25={cal_new.opener_distinct_p25:.2f} "
          f"top_p75={cal_new.opener_top_p75:.2f} fronted_p50={cal_new.opener_fronted_p50:.2f}\n")

    arms = {}
    for name, cal in (("A_no_rule", cal_old), ("B_opener_rule", cal_new)):
        kit = gate.build_kit(TASK, prof, cal, n_examples=5)
        has = "put something before the" in kit
        print(f"{name}: kit has opener rule = {has}, generating {N_SLATE}...", flush=True)
        cands, dt = generate(kit, N_SLATE, tag=name)
        print(f"    parsed {len(cands)} candidate(s) from the slate", flush=True)
        rows = []
        for i, t in enumerate(cands):
            # 60 words, not 120. The opener metrics need ~6 sentences to mean
            # anything, which is well under 120 words, and the stricter filter
            # was silently discarding usable candidates.
            if not t or len(t.split()) < 60:
                continue
            m = metrics(t)
            if not m:
                continue
            try:
                m["fingerprint"] = float(fp.distance(t))
            except Exception:
                m["fingerprint"] = float("nan")
            m["i"] = i
            m["text"] = t
            rows.append(m)
        arms[name] = rows
        print(f"  {len(rows)} usable candidates in {dt:.0f}s", flush=True)

    print(f"\n{'arm':16} {'cand':>4} {'words':>6} {'distinct':>9} {'top':>13} "
          f"{'fronted':>8} {'run':>4} {'fp':>6}")
    for name, rows in arms.items():
        for r in rows:
            print(f"{name:16} {r['i']:>4} {r['words']:>6} {r['distinct']*100:>8.0f}% "
                  f"{r['top_word']+' '+str(round(r['top_share']*100))+'%':>13} "
                  f"{r['fronted']*100:>7.0f}% {r['max_run']:>4} {r['fingerprint']:>6.2f}")

    print()
    for name, rows in arms.items():
        if not rows:
            continue
        n = len(rows)
        print(f"{name:16} mean distinct={sum(r['distinct'] for r in rows)/n*100:.0f}%  "
              f"mean top={sum(r['top_share'] for r in rows)/n*100:.0f}%  "
              f"mean fronted={sum(r['fronted'] for r in rows)/n*100:.0f}%  "
              f"mean fp={sum(r['fingerprint'] for r in rows)/n:.2f}")

    json.dump({k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in arms.items()},
              open(os.path.join(OUT, "ab_results.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}\\ab_results.json")


if __name__ == "__main__":
    main()
