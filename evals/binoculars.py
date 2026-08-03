"""Binoculars axis for the A/B study: calibrate a threshold, then score saved text.

Three steps, kept honest:

* ``gen-ai`` -- generate a plain, non-voice ``claude -p`` answer for each brief.
  This is the *known-machine* negative class: generic model prose, topically
  matched to the study, produced WITHOUT the voice engine. Calibrating on our own
  voice-clone drafts would be circular (it would bake "the clones are human" into
  the ruler); calibrating real-human corpus vs generic-AI, then asking where the
  clones land, is the honest test.
* ``calibrate`` -- score the author's corpus (human) and the AI decoys (machine),
  fit a low-false-positive threshold + AUROC for the configured model pair, and
  save it per profile and into ``evals/reference/<voice>.detector.json``.
* ``score`` -- offline pass that fills the ``binoculars`` field of every saved
  generation JSON under a config/voice, using the calibrated threshold. Cheap and
  decoupled from generation, so it can run over baseline/style/detect/... alike.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from mimesis_voice import config, detect, gate, ingest
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import config, detect, gate, ingest

EVALS_DIR = Path(__file__).resolve().parent
BRIEFS_PATH = EVALS_DIR / "briefs.json"
REFERENCE_DIR = EVALS_DIR / "reference"
AIDECOY_DIR = EVALS_DIR / "scratch" / "aidecoy"
_MIN_WORDS = 120


def _briefs(voice: str) -> list[dict]:
    data = json.loads(BRIEFS_PATH.read_text(encoding="utf-8"))
    return [b for b in data["briefs"] if b["voice"] == voice]


def _corpus_human(profile: config.Profile) -> list[str]:
    pieces = ingest.read_pieces(profile.db_path)
    return [t for t in pieces.values() if len(re.findall(r"[A-Za-z']+", t)) >= _MIN_WORDS]


# --- gen-ai -------------------------------------------------------------------


def cmd_gen_ai(args) -> int:
    briefs = _briefs(args.voice)
    out = AIDECOY_DIR / args.voice
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for b in briefs:
        dst = out / f"{b['id']}.txt"
        if dst.exists() and not args.force:
            print(f"  = {b['id']}: exists")
            continue
        prompt = (
            f"{b['task']}\n\nWrite {b.get('target_words', [160, 220])[0]}-"
            f"{b.get('target_words', [160, 220])[1]} words. Output only the piece."
        )
        try:
            text = gate.claude_generate(prompt, model=args.model)
        except Exception as exc:
            print(f"  ! {b['id']}: {exc}")
            continue
        dst.write_text(text.strip(), encoding="utf-8")
        done += 1
        print(f"  + {b['id']}: {len(text.split())} words")
    print(f"[gen-ai] {done} AI decoys -> {out}")
    return 0


# --- calibrate ----------------------------------------------------------------


def cmd_calibrate(args) -> int:
    prof = config.resolve_named(args.profile)
    if prof is None:
        print(f"no profile '{args.profile}'")
        return 1
    human = _corpus_human(prof)
    ai_dir = AIDECOY_DIR / args.voice
    ai = [p.read_text(encoding="utf-8") for p in sorted(ai_dir.glob("*.txt"))] if ai_dir.exists() else []
    if len(ai) < 3:
        print(f"need >=3 AI decoys for '{args.voice}'; run: python evals/binoculars.py gen-ai --voice {args.voice}")
        return 1
    print(f"[calibrate] voice={args.voice} human={len(human)} ai={len(ai)} pair={detect.model_pair()}")
    cal = detect.calibrate_threshold(human, ai, target_fpr=args.target_fpr)
    cal["voice"] = args.voice
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if args.tag:
        # comparison run (e.g. a different model pair): write only to a tagged file.
        (REFERENCE_DIR / f"{args.voice}.detector.{args.tag}.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    else:
        (prof.data_dir).mkdir(parents=True, exist_ok=True)
        (prof.data_dir / "detector_calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        (REFERENCE_DIR / f"{args.voice}.detector.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps(cal, indent=2))
    return 0


def load_threshold(voice: str) -> dict | None:
    p = REFERENCE_DIR / f"{voice}.detector.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- score --------------------------------------------------------------------


def cmd_score(args) -> int:
    cal = load_threshold(args.voice)
    if cal is None:
        print(f"no detector calibration for '{args.voice}'. Run calibrate first.")
        return 1
    threshold = cal["threshold"]
    direction = cal.get("direction", "low_is_machine")
    d = EVALS_DIR / args.config / args.voice
    files = sorted(f for f in d.glob("*.json") if f.name != "discrimination.json")
    if not files:
        print(f"no generation JSONs under {d}")
        return 1
    machine = scored = 0
    vals = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        text = rec.get("output", "")
        if not text:
            continue
        sig = detect.score(text, threshold=threshold, calibrated=True, direction=direction)
        rec["binoculars"] = sig
        f.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        scored += 1
        if sig.get("score") is not None:
            vals.append(sig["score"])
        if sig.get("label") == "machine":
            machine += 1
        print(f"  {rec['brief_id']}: score={sig.get('score')} label={sig.get('label')}")
    import statistics
    mean = round(statistics.mean(vals), 4) if vals else None
    print(f"[score] config={args.config} voice={args.voice}: {scored} scored, "
          f"{machine} read machine (threshold {threshold}); mean score {mean}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="binoculars")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_g = sub.add_parser("gen-ai")
    p_g.add_argument("--voice", required=True)
    p_g.add_argument("--model", default="sonnet")
    p_g.add_argument("--force", action="store_true")
    p_c = sub.add_parser("calibrate")
    p_c.add_argument("--voice", required=True)
    p_c.add_argument("--profile", required=True)
    p_c.add_argument("--target-fpr", type=float, default=0.05)
    p_c.add_argument("--tag", default=None, help="write to a tagged reference file (comparison run)")
    p_s = sub.add_parser("score")
    p_s.add_argument("--config", required=True)
    p_s.add_argument("--voice", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "gen-ai":
        return cmd_gen_ai(args)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "score":
        return cmd_score(args)
    return 1


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
