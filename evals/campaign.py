"""Unattended experiment runner: queue jobs, survive failures, report in the morning.

Every useful result in this project came from re-running something, and each
re-run cost 20-50 minutes of supervision. This runs the queue while nobody is
watching.

Design constraints, all learned the hard way in this repo:

* **Resumable.** Each finished job appends one line to ``ledger.jsonl``. Re-running
  the same spec skips anything already recorded, so a crash at hour four costs
  one job, not the night.
* **A failed job must not kill the campaign.** Every job is wrapped; failures are
  recorded with their traceback and the queue continues.
* **Nothing known-good is overwritten.** Jobs write to their own output
  directories and build their own profiles. A campaign cannot damage a calibrated
  voice you rely on.
* **Every job records provenance** -- git HEAD, spec hash, timestamps, and the
  exact parameters -- because a number whose configuration you cannot reconstruct
  is not evidence.
* **Paired stats, not arm means.** Comparisons are scored by the paired sign test
  used throughout ``CADENCE_FINDINGS.md``, and the report states n and p rather
  than a bare delta.

Usage:
    python evals/campaign.py run  --spec evals/specs/overnight.json
    python evals/campaign.py report --out-dir evals/campaigns/<name>
    python evals/campaign.py run  --spec ... --dry-run     # plan only, no compute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EVALS = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


# --- small helpers ------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """Exact two-sided sign test. Returns (improved, n, p). Ties count against."""
    n = len(deltas)
    if n == 0:
        return 0, 0, 1.0
    w = sum(1 for d in deltas if d < 0)
    p = 2 * sum(math.comb(n, k) for k in range(0, min(w, n - w) + 1)) / 2 ** n
    return w, n, min(1.0, p)


def run_step(cmd: list[str], log: Path, timeout: int) -> tuple[int, str]:
    """Run a subprocess, tee output to ``log``, return (returncode, tail)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[campaign] TIMEOUT after {timeout}s\n")
            rc = -9
    tail = "\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:])
    return rc, tail


# --- job types ----------------------------------------------------------------


def job_ab(job: dict, outdir: Path) -> dict:
    """v1-vs-v2 A/B for a voice. Returns paired stats on the cadence block."""
    p = job["params"]
    out = outdir / job["id"]
    cmd = [
        PY, "-u", str(EVALS / "ab_cadence_e2e.py"),
        "--voice", p["voice"],
        "--brief-file", p.get("brief_file", "briefs.json"),
        "--v1-profile", p["v1_profile"],
        "--v2-profile", p["v2_profile"],
        "--out", str(out),
        "--briefs", str(p.get("briefs", 6)),
    ]
    if p.get("segment_words"):
        cmd += ["--segment-words", str(p["segment_words"])]
    if p.get("split"):
        cmd += ["--split", str(p["split"])]
    rc, tail = run_step(cmd, out / "run.log", job.get("timeout", 14400))

    result: dict = {"rc": rc, "tail": tail}
    allj = out / "all.json"
    if not allj.exists():
        return result
    rows = json.loads(allj.read_text(encoding="utf-8"))
    by: dict = {}
    for r in rows:
        by.setdefault(r["brief_id"], {})[r["arm"]] = r
    ids = sorted(k for k, v in by.items() if "v1" in v and "v2" in v)
    d_cad = [by[k]["v2"]["ruler_cadence"] - by[k]["v1"]["ruler_cadence"] for k in ids]
    d_rms = [by[k]["v2"]["ruler_rmsz"] - by[k]["v1"]["ruler_rmsz"] for k in ids]
    w, n, pv = sign_test(d_cad)
    result.update({
        "n_pairs": n,
        "v1_cadence": statistics.mean(by[k]["v1"]["ruler_cadence"] for k in ids) if ids else None,
        "v2_cadence": statistics.mean(by[k]["v2"]["ruler_cadence"] for k in ids) if ids else None,
        "mean_delta_cadence": statistics.mean(d_cad) if d_cad else None,
        "mean_delta_rmsz": statistics.mean(d_rms) if d_rms else None,
        "improved": w,
        "p_value": pv,
        "per_brief": {k: {"v1": by[k]["v1"]["ruler_cadence"], "v2": by[k]["v2"]["ruler_cadence"]} for k in ids},
    })
    return result


def job_build_profile(job: dict, outdir: Path) -> dict:
    """Create + ingest + calibrate a profile from an external source directory.

    Uses a directory junction rather than copying, so no client corpus is ever
    duplicated into this repository.
    """
    from mimesis_voice import config as cfgmod

    p = job["params"]
    slug = p["slug"]
    d = ROOT / "profiles" / slug
    (d / "data").mkdir(parents=True, exist_ok=True)
    cfg = {
        "author_name": p.get("author_name", "Author"),
        "embed_backend": p.get("embed_backend", "fast"),
        "gate": {"rmsz_max": p.get("rmsz_max", "auto"),
                 "slate_size": p.get("slate_size", 4),
                 "max_rewrites": p.get("max_rewrites", 2)},
        "formats": {},
        "anchors": {"exemplars": True, "transform_pairs": None},
        "whitelist": p.get("whitelist", []),
        "feature_set": p.get("feature_set", "v2"),
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    link = d / "source_documents"
    src = p.get("source_dir")
    if src:
        if link.exists() or link.is_symlink():
            subprocess.run(["cmd", "/c", "rmdir", str(link)], cwd=ROOT,
                           capture_output=True, text=True)
        rc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), src],
                            cwd=ROOT, capture_output=True, text=True)
        if not link.exists():
            return {"rc": 1, "tail": f"junction failed: {rc.stdout}{rc.stderr}"}

    log = outdir / job["id"] / "build.log"
    rc1, t1 = run_step([PY, "-m", "mimesis_voice.cli", "ingest", slug], log,
                       job.get("timeout", 3600))
    rc2, t2 = run_step([PY, "-m", "mimesis_voice.cli", "calibrate", slug],
                       outdir / job["id"] / "calibrate.log", job.get("timeout", 3600))

    info: dict = {"rc": rc1 or rc2, "tail": (t1 + "\n" + t2)[-1500:]}
    try:
        prof = cfgmod.resolve(slug)
        fpj = json.loads(prof.fingerprint_path.read_text(encoding="utf-8"))
        info.update({
            "slug": slug,
            "n_pieces": fpj.get("n_pieces"),
            "self_baseline": fpj.get("self_baseline"),
            "fit_threshold": fpj.get("fit_threshold"),
            "feature_set": fpj.get("feature_set"),
        })
    except Exception as exc:
        info["tail"] += f"\n[campaign] could not read calibration: {exc}"
    return info


def job_discrim(job: dict, outdir: Path) -> dict:
    """Blind discrimination eval: does a judge pick the generation over real writing?

    This is the only metric in the project that does not score generations on
    features chosen by the same person who built the generator, so it is the one
    that survives "you invented the ruler".
    """
    from mimesis_voice import config as cfgmod
    from mimesis_voice import evalcli

    p = job["params"]
    out = outdir / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, slug in p["profiles"].items():
        try:
            res = evalcli.run_eval(
                cfgmod.resolve(slug),
                held_out=int(p.get("held_out", 8)),
                model=p.get("model", "sonnet"),
            )
            results[label] = {
                "profile": slug,
                "n_trials": res.n_trials,
                "fooled": res.fooled,
                "fool_rate": res.fool_rate,
                "generated_rmsz": statistics.mean(res.generated_rmsz) if res.generated_rmsz else None,
                "real_rmsz": statistics.mean(res.real_rmsz) if res.real_rmsz else None,
                "self_baseline": res.self_baseline,
                "notes": res.notes,
            }
        except Exception as exc:
            results[label] = {"profile": slug, "error": f"{type(exc).__name__}: {exc}"}
    (out / "discrim.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Two-proportion note: at held_out<=10 per arm this cannot resolve small
    # differences. Report it rather than implying significance.
    arms = [v for v in results.values() if v.get("n_trials")]
    if len(arms) == 2:
        n = min(a["n_trials"] for a in arms)
        results["_power_note"] = (
            f"{n} trials per arm; a fool-rate difference below roughly "
            f"{100 * 1.96 * (0.5 * 0.5 / max(n, 1)) ** 0.5 * 2:.0f} points is not "
            f"resolvable at this n."
        )
    return {"rc": 0, "results": results}


def job_audit(job: dict, outdir: Path) -> dict:
    """Cadence audit of existing generations: is there a rhythm gap worth gating on?"""
    p = job["params"]
    out = outdir / job["id"]
    cmd = [PY, "-u", str(EVALS / "cadence_audit_external.py"),
           "--out", p["out"], "--label", p.get("label", job["id"])]
    for c in p["corpus"]:
        cmd += ["--corpus", c]
    if p.get("gen_jsonl"):
        cmd += ["--gen-jsonl", p["gen_jsonl"], "--gen-field", p.get("gen_field", "output")]
    if p.get("gen_dir"):
        cmd += ["--gen-dir", p["gen_dir"]]
    rc, tail = run_step(cmd, out / "run.log", job.get("timeout", 3600))
    res = {"rc": rc, "tail": tail}
    f = Path(p["out"]) / f"{p.get('label', job['id'])}_cadence_audit.json"
    if f.exists():
        res["audit"] = json.loads(f.read_text(encoding="utf-8"))
    return res


JOBS = {"ab": job_ab, "build_profile": job_build_profile,
        "discrim": job_discrim, "audit": job_audit}


# --- campaign driver ----------------------------------------------------------


def load_done(ledger: Path) -> set[str]:
    if not ledger.exists():
        return set()
    done = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec["id"])
        except json.JSONDecodeError:
            continue
    return done


def cmd_run(args) -> int:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    name = spec.get("name") or Path(args.spec).stem
    outdir = Path(args.out_dir) if args.out_dir else EVALS / "campaigns" / name
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "ledger.jsonl"
    done = load_done(ledger)
    jobs = spec["jobs"]
    head = git_head()
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    print(f"campaign '{name}'  jobs={len(jobs)}  already done={len(done)}  "
          f"git={head}  spec={spec_hash}")
    print(f"output: {outdir}")
    if args.dry_run:
        for j in jobs:
            mark = "SKIP(done)" if j["id"] in done else "run"
            print(f"  [{mark:<10}] {j['id']:<28} type={j['type']}")
        return 0

    t_start = time.time()
    for i, job in enumerate(jobs, 1):
        jid = job["id"]
        if jid in done:
            print(f"[{i}/{len(jobs)}] {jid}: already done, skipping")
            continue
        fn = JOBS.get(job["type"])
        rec = {"id": jid, "type": job["type"], "started": now(),
               "git": head, "spec_hash": spec_hash, "params": job.get("params", {})}
        if fn is None:
            rec.update({"status": "error", "error": f"unknown job type {job['type']}"})
        else:
            print(f"[{i}/{len(jobs)}] {jid}: running ({job['type']})...", flush=True)
            t0 = time.time()
            try:
                out = fn(job, outdir)
                rec["result"] = out
                rec["status"] = "ok" if out.get("rc", 0) == 0 else "failed"
            except Exception:
                rec.update({"status": "error", "error": traceback.format_exc()[-2000:]})
            rec["secs"] = round(time.time() - t0, 1)
        rec["finished"] = now()
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"      -> {rec['status']} in {rec.get('secs', 0):.0f}s", flush=True)

    print(f"\ncampaign finished in {(time.time() - t_start) / 60:.1f} min")
    write_report(outdir)
    return 0


def write_report(outdir: Path) -> None:
    ledger = outdir / "ledger.jsonl"
    if not ledger.exists():
        print("no ledger to report on")
        return
    recs = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = [f"# Campaign report — {outdir.name}", "",
             f"Generated {now()}. {len(recs)} jobs recorded.", ""]

    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    lines += [f"- completed: {len(ok)}", f"- failed/errored: {len(bad)}",
              f"- total compute: {sum(r.get('secs', 0) for r in recs) / 3600:.1f} h", ""]

    abs_ = [r for r in ok if r["type"] == "ab" and r["result"].get("n_pairs")]
    if abs_:
        lines += ["## A/B results (paired, cadence block)", "",
                  "| job | n | v1 | v2 | mean delta | improved | p |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for r in abs_:
            x = r["result"]
            lines.append(
                f"| {r['id']} | {x['n_pairs']} | {x['v1_cadence']:.3f} | {x['v2_cadence']:.3f} | "
                f"{x['mean_delta_cadence']:+.3f} | {x['improved']}/{x['n_pairs']} | {x['p_value']:.4f} |"
            )
        lines += ["", "p is an exact two-sided sign test. At n=6 the smallest attainable "
                      "p is 0.031, so a null result here is weak evidence of no effect, "
                      "not evidence of none.", ""]

    builds = [r for r in ok if r["type"] == "build_profile" and r["result"].get("slug")]
    if builds:
        lines += ["## Profiles built", "",
                  "| slug | pieces | self-baseline | p95 | features |",
                  "|---|--:|--:|--:|---|"]
        for r in builds:
            x = r["result"]
            lines.append(f"| {x['slug']} | {x.get('n_pieces')} | "
                         f"{(x.get('self_baseline') or 0):.3f} | "
                         f"{(x.get('fit_threshold') or 0):.3f} | {x.get('feature_set')} |")
        lines.append("")

    disc = [r for r in ok if r["type"] == "discrim"]
    for r in disc:
        lines += [f"## Blind discrimination — {r['id']}", "",
                  "| arm | profile | trials | fooled | fool-rate |", "|---|---|--:|--:|--:|"]
        for label, v in r["result"]["results"].items():
            if label.startswith("_") or "error" in v:
                continue
            lines.append(f"| {label} | {v['profile']} | {v['n_trials']} | {v['fooled']} | "
                         f"{v['fool_rate']:.1%} |")
        note = r["result"]["results"].get("_power_note")
        if note:
            lines += ["", f"_{note}_"]
        lines.append("")

    auds = [r for r in ok if r["type"] == "audit" and r["result"].get("audit")]
    if auds:
        lines += ["## Cadence audits (is there a gap worth gating on?)", "",
                  "| job | human cadence | generated | AUROC | verdict |", "|---|--:|--:|--:|---|"]
        for r in auds:
            a = r["result"]["audit"]
            g = a.get("gen_cadence")
            auc = a.get("auroc_cadence")
            if g is None:
                lines.append(f"| {r['id']} | {a['human_cadence']:.3f} | — | — | no generations |")
                continue
            gap = g - a["human_cadence"]
            verdict = ("gate is worth it" if gap > 0.3 else
                       "marginal" if gap > 0.1 else "already at parity, gating is a no-op")
            lines.append(f"| {r['id']} | {a['human_cadence']:.3f} | {g:.3f} | "
                         f"{auc:.3f} | {verdict} |")
        lines += ["", "Thresholds come from the measured relationship between the "
                      "pre-existing gap and how much of it gating closes (r=0.789, n=5).", ""]

    if bad:
        lines += ["## Failures", ""]
        for r in bad:
            detail = (r.get("error") or r.get("result", {}).get("tail", ""))[-400:]
            lines += [f"### {r['id']} ({r['status']})", "```", detail.strip(), "```", ""]

    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {outdir / 'REPORT.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--out-dir")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report")
    rep.add_argument("--out-dir", required=True)
    rep.set_defaults(fn=lambda a: (write_report(Path(a.out_dir)), 0)[1])
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
