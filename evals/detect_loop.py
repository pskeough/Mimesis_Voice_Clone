"""Upgrade-2 arm: apply the detector-in-loop to the FIXED baseline drafts.

Isolating the loop's marginal effect means holding the base generation constant
and running only the final detector step on it: for each baseline draft that reads
"machine" against the calibrated threshold, do one bounded de-machine rewrite
(the exact gate prompt + accept-guard: keep it only if the detector score improves
and the fingerprint does not regress). Drafts that already read human are copied
through unchanged (the loop is a no-op on them). This measures the loop, not
generation noise, and is cheap because the weak small-pair detector fires rarely.

Writes the resulting arm to ``evals/detect/<voice>/`` with updated output +
binoculars + a per-draft note on what the loop did.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from mimesis_voice import config, detect, gate
    from mimesis_voice import scrub as scrub_mod
    from mimesis_voice.fingerprint import FEATURES, Fingerprint
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import config, detect, gate
    from mimesis_voice import scrub as scrub_mod
    from mimesis_voice.fingerprint import FEATURES, Fingerprint

EVALS_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = EVALS_DIR / "reference"


def _wc(t: str) -> int:
    return len(re.findall(r"[A-Za-z']+", t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--rmsz-max", type=float, default=1.1)
    args = ap.parse_args()

    cal_j = json.loads((REFERENCE_DIR / f"{args.voice}.detector.json").read_text(encoding="utf-8"))
    threshold, direction = cal_j["threshold"], cal_j.get("direction", "low_is_machine")
    ref_fp = Fingerprint.load(REFERENCE_DIR / f"{args.voice}.fingerprint.json")
    prof = config.resolve_named(args.profile)
    cal = scrub_mod.ScrubCalibration.load(prof.scrub_path)

    src = EVALS_DIR / "baseline" / args.voice
    dst = EVALS_DIR / "detect" / args.voice
    dst.mkdir(parents=True, exist_ok=True)

    fired = flagged = 0
    for f in sorted(src.glob("*.json")):
        if f.name == "discrimination.json":
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        text = rec["output"]
        # (re)score the baseline draft with the detector
        sig = detect.score(text, threshold=threshold, calibrated=True, direction=direction)
        rec["binoculars_baseline"] = sig
        note = "loop no-op (reads human)"
        if sig.get("available") and sig.get("label") == "machine":
            flagged += 1
            kit = gate.build_kit(rec["task"], prof, cal, n_examples=5)
            try:
                rewrite = gate.claude_generate(
                    gate._detector_rewrite_prompt(kit, text, prof.name), model=args.model
                )
                fixed, _ = gate.scalpel(rewrite)
                new_sig = detect.score(fixed, threshold=threshold, calibrated=True, direction=direction)
                new_rmsz = ref_fp.distance(fixed)
                improved = (new_sig.get("score") is not None and sig.get("score") is not None
                            and _better(new_sig["score"], sig["score"], direction))
                fp_ok = new_rmsz <= args.rmsz_max or new_rmsz <= rec["reference_rmsz"] + 1e-6
                if improved and fp_ok:
                    text = fixed
                    sig = new_sig
                    rec["reference_rmsz"] = round(new_rmsz, 4)
                    fired += 1
                    note = f"de-machine accepted (label {new_sig.get('label')})"
                else:
                    note = "de-machine rejected (no score gain or fingerprint regressed)"
            except Exception as exc:
                note = f"de-machine call failed: {exc}"
        rec["output"] = text
        rec["word_count"] = _wc(text)
        rec["binoculars"] = sig
        rec["config"] = "detect"
        rec["detect_note"] = note
        (dst / f.name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  {rec['brief_id']}: {note}  (score {sig.get('score')} / thr {threshold})")

    print(f"[detect] voice={args.voice}: {flagged} flagged, {fired} rewrites accepted")
    return 0


def _better(new: float, old: float, direction: str) -> bool:
    # lower score is more human when high_is_machine; higher is more human when low_is_machine
    return new < old if direction == "high_is_machine" else new > old


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
