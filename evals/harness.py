"""A/B eval harness for the Mimesis upgrade study.

One instrument, reused across every engine config (baseline, style, detect,
recalibrate, combined). It does three things and nothing else:

* ``generate`` -- run the compose loop over the frozen brief set for a voice,
  capture the output plus full gate metadata, score it against a *frozen
  reference fingerprint*, and save one JSON per (config, voice, brief).
* ``discrim`` -- run the discrimination eval (held-out real pieces -> neutral
  brief -> generate -> claude judge) for a voice/profile and save the raw
  per-trial results, so the fool-rate axis is reproducible.
* ``freeze-ref`` -- snapshot a voice's calibrated fingerprint into
  ``evals/reference/`` so every config is scored against the same yardstick,
  even after an upgrade recalibrates the live fingerprint.

Why a frozen reference fingerprint: the RMS-z axis must answer "did the output
move toward the author?" on a fixed ruler. Upgrade 1 (style) does not touch the
fingerprint (it is surface-feature based, independent of the embed backend), but
Upgrade 3 (recalibrate) does. Scoring every generation against the baseline
fingerprint keeps the cross-config delta attributable to the engine, not to a
shifted ruler. Binoculars is scored later, offline, over the saved texts.

Run with the repo venv python, e.g.::

    python evals/harness.py generate --config baseline --voice research --profile research
    python evals/harness.py discrim  --config baseline --voice creative --profile creative --held-out 6
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

# The package is installed editable, so a plain import works from anywhere; fall
# back to adding src/ for a bare checkout.
try:
    from mimesis_voice import config, evalcli, gate, ingest
    from mimesis_voice.fingerprint import FEATURES, Fingerprint
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import config, evalcli, gate, ingest
    from mimesis_voice.fingerprint import FEATURES, Fingerprint

EVALS_DIR = Path(__file__).resolve().parent
BRIEFS_PATH = EVALS_DIR / "briefs.json"
REFERENCE_DIR = EVALS_DIR / "reference"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def load_briefs(voice: str) -> list[dict]:
    data = json.loads(BRIEFS_PATH.read_text(encoding="utf-8"))
    return [b for b in data["briefs"] if b["voice"] == voice]


def reference_fingerprint(voice: str) -> Fingerprint:
    path = REFERENCE_DIR / f"{voice}.fingerprint.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen reference fingerprint for '{voice}'. Run: "
            f"python evals/harness.py freeze-ref --voice {voice} --profile {voice}"
        )
    return Fingerprint.load(path)


# --- freeze-ref ---------------------------------------------------------------


def cmd_freeze_ref(args) -> int:
    prof = config.resolve_named(args.profile)
    if prof is None or not prof.fingerprint_path.exists():
        print(f"profile '{args.profile}' has no calibrated fingerprint; calibrate it first")
        return 1
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    dst = REFERENCE_DIR / f"{args.voice}.fingerprint.json"
    fp = Fingerprint.load(prof.fingerprint_path)
    fp.save(dst)
    print(f"froze reference fingerprint for '{args.voice}' from '{args.profile}' "
          f"(self-baseline {fp.self_baseline:.3f}, {fp.n_pieces} pieces) -> {dst}")
    return 0


# --- generate -----------------------------------------------------------------


def _gate_meta(res) -> dict:
    c = res.chosen
    return {
        "n_candidates": len(res.candidates),
        "n_survivors": len(res.survivors),
        "iterations": res.iterations,
        "chosen_rmsz_live": round(c.rmsz, 4) if c else None,
        "emdash_fixed": c.emdash_fixed if c else 0,
        "scrub_hard_flags": list(c.scrub.hard_flags) if (c and c.scrub) else [],
        "scrub_banned_words": list(c.scrub.banned_words) if (c and c.scrub) else [],
        "notes": list(res.notes),
    }


def cmd_generate(args) -> int:
    briefs = load_briefs(args.voice)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        briefs = [b for b in briefs if b["id"] in wanted]
    if not briefs:
        print(f"no briefs for voice '{args.voice}' in {BRIEFS_PATH}")
        return 1
    prof = config.resolve_named(args.profile)
    if prof is None:
        print(f"no profile '{args.profile}'")
        return 1
    ref_fp = reference_fingerprint(args.voice)
    out_dir = EVALS_DIR / args.config / args.voice
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detector-in-loop (Upgrade 2): load the voice's calibrated threshold + direction.
    det_threshold = None
    det_direction = "low_is_machine"
    if args.detector:
        cal_path = REFERENCE_DIR / f"{args.voice}.detector.json"
        if not cal_path.exists():
            print(f"--detector set but no calibration at {cal_path}; run binoculars.py calibrate first")
            return 1
        _cal = json.loads(cal_path.read_text(encoding="utf-8"))
        det_threshold = _cal["threshold"]
        det_direction = _cal.get("direction", "low_is_machine")

    print(f"[generate] config={args.config} voice={args.voice} profile={args.profile} "
          f"model={args.model} briefs={len(briefs)} detector={'on@'+str(det_threshold) if args.detector else 'off'}")
    done = fail = skip = 0
    for b in briefs:
        dst = out_dir / f"{b['id']}.json"
        if dst.exists() and not args.force:
            skip += 1
            print(f"  = {b['id']}: exists, skip")
            continue
        try:
            res = gate.compose(
                b["task"], prof, model=args.model, n_examples=args.examples,
                use_detector=args.detector, detector_threshold=det_threshold,
                detector_calibrated=args.detector, detector_direction=det_direction,
            )
        except Exception as exc:
            fail += 1
            print(f"  ! {b['id']}: compose failed: {exc}")
            continue
        output = res.output or ""
        if not output:
            fail += 1
            print(f"  ! {b['id']}: empty output")
            continue
        ref_rmsz, ref_zs = ref_fp.distance_detail(output)
        record = {
            "config": args.config,
            "voice": args.voice,
            "compose_profile": args.profile,
            "brief_id": b["id"],
            "genre": b.get("genre"),
            "task": b["task"],
            "model": args.model,
            "output": output,
            "word_count": _word_count(output),
            "reference_rmsz": round(ref_rmsz, 4),
            "reference_self_baseline": round(ref_fp.self_baseline, 4),
            "reference_zs": {f: round(ref_zs[f], 4) for f in FEATURES},
            "gate": _gate_meta(res),
            # If the detector ran in-loop, keep its post-loop reading here; else
            # leave null for the offline binoculars.py score pass to fill.
            "binoculars": (res.chosen.detector or None) if (args.detector and res.chosen) else None,
            "timestamp": _now(),
        }
        dst.write_text(json.dumps(record, indent=2), encoding="utf-8")
        done += 1
        print(f"  + {b['id']}: rmsz={ref_rmsz:.3f} (baseline {ref_fp.self_baseline:.3f}), "
              f"wc={record['word_count']}, flags={record['gate']['scrub_hard_flags']}")
    print(f"[generate] done={done} skip={skip} fail={fail}")
    return 0 if fail == 0 else 2


# --- discrim ------------------------------------------------------------------


def cmd_discrim(args) -> int:
    prof = config.resolve_named(args.profile)
    if prof is None:
        print(f"no profile '{args.profile}'")
        return 1
    print(f"[discrim] config={args.config} voice={args.voice} profile={args.profile} "
          f"held_out={args.held_out} model={args.model}")
    res = evalcli.run_eval(
        prof, held_out=args.held_out, model=args.model,
        fingerprint_only=args.fingerprint_only,
    )
    out_dir = EVALS_DIR / args.config / args.voice
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "discrimination.json"
    record = {
        "config": args.config,
        "voice": args.voice,
        "compose_profile": args.profile,
        "model": args.model,
        "held_out": args.held_out,
        "fingerprint_only": res.fingerprint_only,
        "n_trials": res.n_trials,
        "fooled": res.fooled,
        "fool_rate": round(res.fool_rate, 4),
        "self_baseline": round(res.self_baseline, 4),
        "real_rmsz": [round(x, 4) for x in res.real_rmsz],
        "generated_rmsz": [round(x, 4) for x in res.generated_rmsz],
        "per_feature_absz": {k: round(v, 4) for k, v in res.per_feature_absz.items()},
        "notes": res.notes,
        "timestamp": _now(),
    }
    dst.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(evalcli.render(res, prof))
    print(f"[discrim] saved -> {dst}")
    return 0


# --- args ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="harness", description="Mimesis A/B eval harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fr = sub.add_parser("freeze-ref", help="snapshot a voice's fingerprint as the scoring reference")
    p_fr.add_argument("--voice", required=True)
    p_fr.add_argument("--profile", required=True)

    p_gen = sub.add_parser("generate", help="generate + score over the frozen brief set")
    p_gen.add_argument("--config", required=True, help="engine config name (output dir)")
    p_gen.add_argument("--voice", required=True, help="logical voice (briefs + reference fp)")
    p_gen.add_argument("--profile", required=True, help="profile slug to compose with")
    p_gen.add_argument("--model", default="sonnet")
    p_gen.add_argument("--examples", type=int, default=5)
    p_gen.add_argument("--only", default=None, help="comma-separated brief ids to shard a run")
    p_gen.add_argument("--detector", action="store_true", help="enable the Binoculars detector-in-loop")
    p_gen.add_argument("--force", action="store_true")

    p_dis = sub.add_parser("discrim", help="discrimination eval (fool-rate axis)")
    p_dis.add_argument("--config", required=True)
    p_dis.add_argument("--voice", required=True)
    p_dis.add_argument("--profile", required=True)
    p_dis.add_argument("--held-out", type=int, default=6)
    p_dis.add_argument("--model", default="sonnet")
    p_dis.add_argument("--fingerprint-only", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "freeze-ref":
        return cmd_freeze_ref(args)
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "discrim":
        return cmd_discrim(args)
    return 1


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
