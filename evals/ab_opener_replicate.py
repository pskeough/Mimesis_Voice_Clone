"""Replicate the opener-rule A/B across several slates per arm.

The single-slate run gave A mean 39% fronted / fp 1.05 against B 58% / 1.00 on
n=3 and n=4. Directionally good, and the fingerprint moving the right way was a
genuine surprise, but B's fingerprint spread was 0.87-1.38 against A's 1.01-1.09.
A mean difference sitting inside that much variance is a hypothesis, not a result.

Cloud generation runs ~60-90s per slate, so replication is cheap: R slates per
arm, all candidates pooled, with the spread reported alongside the mean.

Reports a bootstrap interval on the difference rather than a p-value. With tens of
candidates from a handful of slates the samples are not independent (candidates
within a slate share a call), so a t-test would overstate confidence. The interval
is descriptive and honest about that.
"""
import glob
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter

M = r"C:\AI Coding Projects\Online_AI\Mimesis"
sys.path.insert(0, os.path.join(M, "src"))

from mimesis_voice import cli as up_cli, gate, scrub as S  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402
from mimesis_voice.scrub import ScrubCalibration  # noqa: E402

VOICE = sys.argv[1] if len(sys.argv) > 1 else "creative"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N_SLATE = 4
OUT = os.path.join(r"C:\LocalAI\LLM\eval\basilisk\ab_replicate", VOICE)

TASKS = {
    "creative": [
        "Write the opening of a chapter in a literary post-apocalyptic novel. Third "
        "person past. A man wakes in a cold room in a town where everyone is dead, "
        "and goes outside. About 450 words. Return only the prose.",
        "Write a passage of literary fiction, third person past: a man walks through "
        "an emptied town at dusk and finds a dog that has learned to hunt. About 450 "
        "words. Return only the prose.",
    ],
    "research": [
        "Write the Methods subsection describing an LLM-as-judge protocol, about 280 "
        "words. State only these facts: three votes per item at temperature 0.7 "
        "resolved by majority; a documented fallback judge on quota exhaustion; a "
        "400-item stratified sample re-judged by two independent model families on a "
        "byte-identical prompt, reporting verdict agreement percentage and Pearson r; "
        "186 hand-labelled items reporting Cohen's kappa and a sensitivity table "
        "stratified by severity band; pairwise comparisons in both orders with "
        "agreement required. Return only the prose.",
        "Write a Limitations subsection, about 250 words. State only these facts: the "
        "sample is 522 responses from a single pilot run; one judge model with a "
        "documented fallback; no human validation beyond 186 hand-labelled items; "
        "results are directionally reliable but magnitudes will shift at production "
        "scale; the corpus is one domain so generalisation is untested. Return only "
        "the prose.",
    ],
}

_SENT = re.compile(r"[^.!?]+[.!?]+[\s\"']*", re.S)
_W = re.compile(r"[A-Za-z']+")


def measure(t, fp):
    sents = [s.strip() for s in _SENT.findall(t or "") if s.strip()]
    ops = []
    for s in sents:
        w = _W.findall(s)
        if w:
            ops.append(w[0].lower())
    if len(ops) < 5:
        return None
    c = Counter(ops)
    try:
        d = float(fp.distance(t))
    except Exception:
        return None
    return {"distinct": len(c) / len(ops), "top": c.most_common(1)[0][1] / len(ops),
            "fronted": sum(1 for s in sents if S._is_fronted(s)) / len(sents),
            "fp": d, "words": len(t.split()), "text": t}


def boot_diff(a, b, n=4000):
    """Bootstrap CI on mean(b) - mean(a). Descriptive, not a significance test."""
    if not a or not b:
        return (float("nan"),) * 3
    rng = random.Random(11)
    diffs = []
    for _ in range(n):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        diffs.append(statistics.mean(rb) - statistics.mean(ra))
    diffs.sort()
    return (statistics.mean(diffs), diffs[int(0.025 * n)], diffs[int(0.975 * n)])


def main():
    os.makedirs(OUT, exist_ok=True)
    prof = up_cli._resolve(VOICE)
    fp = Fingerprint.load(prof.fingerprint_path)
    cal_old = ScrubCalibration.load(prof.scrub_path)

    texts = []
    for f in glob.glob(os.path.join(prof.root, "source_documents", "*"))[:400]:
        try:
            texts.append(open(f, encoding="utf-8-sig", errors="replace").read())
        except Exception:
            pass
    cal_new = S.calibrate(texts)

    tasks = TASKS.get(VOICE, TASKS["creative"])
    print(f"[{VOICE}] author: distinct {cal_new.opener_distinct_p25*100:.0f}%  "
          f"top {cal_new.opener_top_p75*100:.0f}%  fronted {cal_new.opener_fronted_p50*100:.0f}%")
    print(f"{REPS} slates per arm, {N_SLATE} candidates each, {len(tasks)} task(s)\n", flush=True)

    # Write after EVERY slate, and resume from what is already on disk.
    # The first attempt held 16 candidates in memory and wrote only at the end;
    # the process died silently between reps and every one of them was lost.
    # Partial results are worth more than a tidy single write.
    pools_path = os.path.join(OUT, "pools.json")
    pools = {"A_no_rule": [], "B_opener_rule": []}
    if os.path.exists(pools_path):
        try:
            prior = json.load(open(pools_path, encoding="utf-8"))
            for k in pools:
                pools[k] = prior.get(k, [])
            print(f"resuming: {len(pools['A_no_rule'])} A + "
                  f"{len(pools['B_opener_rule'])} B already on disk", flush=True)
        except Exception:
            pass

    def flush():
        json.dump(pools, open(pools_path, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    done_reps = {k: {r.get("rep") for r in v} for k, v in pools.items()}
    for rep in range(REPS):
        task = tasks[rep % len(tasks)]
        for name, cal in (("A_no_rule", cal_old), ("B_opener_rule", cal_new)):
            if rep in done_reps.get(name, set()):
                print(f"  rep{rep} {name}: already have it, skipping", flush=True)
                continue
            kit = gate.build_kit(task, prof, cal, n_examples=5)
            t0 = time.time()
            try:
                raw = gate.claude_generate(gate._slate_prompt(kit, N_SLATE), model="sonnet")
            except Exception as e:
                print(f"  rep{rep} {name}: FAILED {e}", flush=True)
                continue
            got = 0
            for t in gate._parse_slate(raw, N_SLATE):
                m = measure(t, fp)
                if m:
                    m["rep"] = rep
                    pools[name].append(m)
                    got += 1
            flush()
            print(f"  rep{rep} {name}: +{got} candidates ({time.time()-t0:.0f}s), "
                  f"saved (A={len(pools['A_no_rule'])} B={len(pools['B_opener_rule'])})",
                  flush=True)

    print()
    for name, rows in pools.items():
        if not rows:
            print(f"{name}: EMPTY")
            continue
        n = len(rows)
        def ms(k):
            vals = [r[k] for r in rows]
            sd = statistics.stdev(vals) if n > 1 else 0.0
            return statistics.mean(vals), sd
        d, dsd = ms("distinct"); t, tsd = ms("top")
        f, fsd = ms("fronted");  p, psd = ms("fp")
        print(f"{name:15} n={n:3}  distinct {d*100:.0f}%±{dsd*100:.0f}  "
              f"top {t*100:.0f}%±{tsd*100:.0f}  fronted {f*100:.0f}%±{fsd*100:.0f}  "
              f"fp {p:.2f}±{psd:.2f}")

    A, B = pools["A_no_rule"], pools["B_opener_rule"]
    print("\nDifference B - A, bootstrap 95% interval:")
    for key, label in (("fronted", "fronted rate"), ("distinct", "distinct openers"),
                       ("top", "top-opener share"), ("fp", "fingerprint (lower better)")):
        m, lo, hi = boot_diff([r[key] for r in A], [r[key] for r in B])
        crosses = "  (interval crosses zero)" if lo <= 0 <= hi else ""
        print(f"  {label:26} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{crosses}")

    flush()
    print(f"\nwrote {pools_path}")


if __name__ == "__main__":
    main()
