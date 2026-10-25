"""Offline agreement of contentrisk tags with real Pangram sentence labels.

No Pangram call and no LLM. Reads a sentence archive of your own scans
(archive_sentences.jsonl: one row per scanned sentence, y = 1 when Pangram
labelled it AI; schema in docs/DETECTION.md) and asks one question:
inside the same scanned document, are sentences that contentrisk tags as risky
labelled AI more often than the untagged ones?

Within-document is the only fair comparison. Across documents the archive mixes
pre-2022 human papers with model drafts, and any tag that separates sources would
look like a detector. Only documents with at least one Human and one AI sentence
enter. Pangram labels windows, so neighbouring sentences share labels and the
sentence-level association is diluted by construction; read it as a lower bound.

    python evals/pangram_content_check.py PATH/TO/archive_sentences.jsonl [--out evals/pangram_content/RESULTS.md]

Writes aggregates only. The archive holds unpublished text, so no sentence is
copied into the output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimesis_voice import contentrisk  # noqa: E402


def mh_odds_ratio(strata: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    """Mantel-Haenszel OR over 2x2 strata (a=risk&AI, b=risk&Human, c=safe&AI, d=safe&Human),
    with the Robins-Breslow-Greenland 95% CI."""
    num = den = 0.0
    p_r = pq_s = q_s = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        r, s = a * d / n, b * c / n
        num += r
        den += s
        p, q = (a + d) / n, (b + c) / n
        p_r += p * r
        pq_s += p * s + q * r
        q_s += q * s
    if den == 0 or num == 0:
        return float("nan"), float("nan"), float("nan")
    orr = num / den
    var = p_r / (2 * num ** 2) + pq_s / (2 * num * den) + q_s / (2 * den ** 2)
    se = math.sqrt(var)
    return orr, math.exp(math.log(orr) - 1.96 * se), math.exp(math.log(orr) + 1.96 * se)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", help="archive_sentences.jsonl, or the dataset export scans.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-round", type=int, default=None,
                    help="per-scan format only: keep rows whose round is >= N")
    a = ap.parse_args()
    by_item: dict[str, list[dict]] = defaultdict(list)
    with open(a.archive, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "sentence_labels" in r:
                # Dataset export: one row per scan, sentence labels as char spans of `text`.
                if a.min_round is not None and (r.get("round") or 0) < a.min_round:
                    continue
                for s in r.get("sentence_labels") or []:
                    lab = s.get("label")
                    if lab not in ("human", "ai_generated", "ai_assisted"):
                        continue
                    by_item[r["scan_id"]].append(
                        {"text": r["text"][s["start"]:s["end"]], "y": 0 if lab == "human" else 1})
            elif r.get("y") in (0, 1) and r.get("text"):
                by_item[r["item"]].append(r)
    mixed = {k: v for k, v in by_item.items() if len({r["y"] for r in v}) == 2}

    overall: list[tuple[int, int, int, int]] = []
    per_class: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    per_safe: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    counts = defaultdict(lambda: [0, 0])   # class -> [n tagged, n tagged AI]
    n_sent = 0
    for item, rows in mixed.items():
        tags = [contentrisk.classify(r["text"]) for r in rows]
        ys = [r["y"] for r in rows]
        n_sent += len(rows)

        def table(pred) -> tuple[int, int, int, int]:
            a_ = sum(1 for t, y in zip(tags, ys) if pred(t) and y == 1)
            b_ = sum(1 for t, y in zip(tags, ys) if pred(t) and y == 0)
            c_ = sum(1 for t, y in zip(tags, ys) if not pred(t) and y == 1)
            d_ = sum(1 for t, y in zip(tags, ys) if not pred(t) and y == 0)
            return a_, b_, c_, d_

        overall.append(table(lambda t: t.is_risk))
        names = set(contentrisk.RISK) | {"concept-exposition"}
        for name in names:
            per_class[name].append(table(lambda t, n=name: n in t.risk))
        for name in contentrisk.SAFE:
            per_safe[name].append(table(lambda t, n=name: n in t.safe and not t.is_risk))
        for t, y in zip(tags, ys):
            for name in t.risk:
                counts[name][0] += 1
                counts[name][1] += y
            for name in t.safe:
                if not t.is_risk:
                    counts["safe:" + name][0] += 1
                    counts["safe:" + name][1] += y

    def fmt(strata, label):
        a_ = sum(s[0] for s in strata); b_ = sum(s[1] for s in strata)
        c_ = sum(s[2] for s in strata); d_ = sum(s[3] for s in strata)
        orr, lo, hi = mh_odds_ratio(strata)
        rate_t = a_ / (a_ + b_) if a_ + b_ else float("nan")
        rate_u = c_ / (c_ + d_) if c_ + d_ else float("nan")
        return f"| {label} | {a_ + b_} | {rate_t:.2f} | {rate_u:.2f} | {orr:.2f} [{lo:.2f}, {hi:.2f}] |"

    out = ["# contentrisk vs Pangram sentence labels (offline, within-document)", "",
           f"Archive: `{Path(a.archive).name}`. Documents with both labels: {len(mixed)} of {len(by_item)}; "
           f"sentences: {n_sent}. AI rate = share of tagged (or untagged) sentences Pangram labelled AI. "
           "OR = Mantel-Haenszel odds ratio stratified by document, 95% CI; above 1 means the tag goes with AI.", "",
           "| tag | n tagged | AI rate, tagged | AI rate, rest | MH OR [95% CI] |",
           "|---|---|---|---|---|",
           fmt(overall, "any risk class")]
    for name in sorted(per_class, key=lambda n: -counts[n][0]):
        if counts[name][0]:
            out.append(fmt(per_class[name], name))
    out += ["", "Safe markers (sentences with the marker and no risk class):", "",
            "| marker | n | AI rate, marked | AI rate, rest | MH OR [95% CI] |", "|---|---|---|---|---|"]
    for name in contentrisk.SAFE:
        out.append(fmt(per_safe[name], name))
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
