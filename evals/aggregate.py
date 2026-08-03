"""Aggregate saved eval artifacts into the three A/B axes.

Reads every generation JSON and discrimination JSON under ``evals/<config>/<voice>/``
and emits per-(config, voice) rows on:

* **fingerprint** -- mean reference RMS-z of curated-brief generations (scored on
  the frozen baseline ruler), plus the ratio to the corpus self-baseline;
* **binoculars** -- mean detector score and the fraction of generations that read
  "machine" against the calibrated threshold;
* **discrimination** -- fool-rate and generated-vs-real RMS-z from the held-out
  discrimination eval.

Also computes ``distance_to_accepted`` for any config when an accepted-set target
is present (Upgrade 3). Output is a JSON summary (machine-readable) and a Markdown
block; the narrative in REPORT.md is written by hand on top of these numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    from mimesis_voice.fingerprint import AcceptedTarget
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice.fingerprint import AcceptedTarget

EVALS_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = EVALS_DIR / "reference"
CONFIGS = ["baseline", "style", "detect", "recalibrate", "combined"]
VOICES = ["research", "creative"]


def _mean(xs):
    return statistics.mean(xs) if xs else None


def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _load_gens(config: str, voice: str) -> list[dict]:
    d = EVALS_DIR / config / voice
    if not d.exists():
        return []
    return [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(d.glob("*.json"))
        if f.name != "discrimination.json"
    ]


def _load_discrim(config: str, voice: str) -> dict | None:
    p = EVALS_DIR / config / voice / "discrimination.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _accepted_target(voice: str) -> AcceptedTarget | None:
    p = REFERENCE_DIR / f"{voice}.accepted_target.json"
    return AcceptedTarget.load(p) if p.exists() else None


def summarize(config: str, voice: str) -> dict | None:
    gens = _load_gens(config, voice)
    discrim = _load_discrim(config, voice)
    if not gens and not discrim:
        return None
    rmsz = [g["reference_rmsz"] for g in gens if g.get("reference_rmsz") is not None]
    self_baseline = gens[0]["reference_self_baseline"] if gens else (discrim or {}).get("self_baseline")
    bino = [g["binoculars"]["score"] for g in gens
            if g.get("binoculars") and g["binoculars"].get("score") is not None]
    machine = sum(1 for g in gens if g.get("binoculars") and g["binoculars"].get("label") == "machine")
    labeled = sum(1 for g in gens if g.get("binoculars") and g["binoculars"].get("label") in ("machine", "human"))
    tgt = _accepted_target(voice)
    d2a = [tgt.distance(g["output"]) for g in gens if g.get("output")] if tgt else []
    row = {
        "config": config,
        "voice": voice,
        "n_gens": len(gens),
        "rmsz_mean": round(_mean(rmsz), 4) if rmsz else None,
        "rmsz_sd": round(_sd(rmsz), 4) if rmsz else None,
        "self_baseline": round(self_baseline, 4) if self_baseline else None,
        "rmsz_ratio": round(_mean(rmsz) / self_baseline, 3) if (rmsz and self_baseline) else None,
        "bino_mean": round(_mean(bino), 4) if bino else None,
        "bino_machine": machine,
        "bino_labeled": labeled,
        "bino_machine_frac": round(machine / labeled, 3) if labeled else None,
        "d2a_mean": round(_mean(d2a), 4) if d2a else None,
        "mean_wc": round(_mean([g["word_count"] for g in gens])) if gens else None,
        "iters_mean": round(_mean([g["gate"]["iterations"] for g in gens]), 2) if gens else None,
    }
    if discrim:
        row["fool_rate"] = discrim.get("fool_rate")
        row["n_trials"] = discrim.get("n_trials")
        row["fooled"] = discrim.get("fooled")
        row["gen_rmsz_discrim"] = round(_mean(discrim.get("generated_rmsz", [])), 4) if discrim.get("generated_rmsz") else None
        row["real_rmsz_discrim"] = round(_mean(discrim.get("real_rmsz", [])), 4) if discrim.get("real_rmsz") else None
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit only the JSON summary")
    args = ap.parse_args()

    rows = []
    for voice in VOICES:
        for config in CONFIGS:
            r = summarize(config, voice)
            if r:
                rows.append(r)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    def cell(v, fmt="{}"):
        return fmt.format(v) if v is not None else "-"

    lines = ["## Fingerprint (curated briefs) + Binoculars + accepted-distance", ""]
    lines.append("| voice | config | n | RMS-z mean±sd | x self-base | Bino mean | machine% | dist-to-accepted | mean wc |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        rmsz = f"{cell(r['rmsz_mean'])}±{cell(r['rmsz_sd'])}" if r["rmsz_mean"] is not None else "-"
        mf = f"{int(r['bino_machine_frac']*100)}% ({r['bino_machine']}/{r['bino_labeled']})" if r["bino_machine_frac"] is not None else "-"
        lines.append(
            f"| {r['voice']} | {r['config']} | {r['n_gens']} | {rmsz} | "
            f"{cell(r['rmsz_ratio'],'{:.2f}×')} | {cell(r['bino_mean'])} | {mf} | "
            f"{cell(r['d2a_mean'])} | {cell(r['mean_wc'])} |"
        )
    lines.append("")
    lines.append("## Discrimination (held-out real vs generated)")
    lines.append("")
    lines.append("| voice | config | fool-rate | trials | gen RMS-z | real RMS-z | self-base |")
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    for r in rows:
        if "fool_rate" not in r:
            continue
        fr = f"{int(r['fool_rate']*100)}% ({r['fooled']}/{r['n_trials']})" if r.get("fool_rate") is not None else "-"
        lines.append(
            f"| {r['voice']} | {r['config']} | {fr} | {cell(r.get('n_trials'))} | "
            f"{cell(r.get('gen_rmsz_discrim'))} | {cell(r.get('real_rmsz_discrim'))} | {cell(r['self_baseline'])} |"
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
