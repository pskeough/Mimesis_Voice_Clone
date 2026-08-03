"""Would this system work for an author it was never tuned on?

Every threshold in a voice engine is either MEASURED from the author's corpus or
TYPED BY HAND by whoever built it. The second kind is invisible until a second
author arrives, and then it fails silently rather than loudly: a hardcoded
``rmsz_max`` of 1.1 did not raise an error on the external long-form corpus, it
just rejected every candidate on every brief and fell back to a default path at
five times the compute cost.

This audits which is which. It reports, per profile:

* every calibrated artifact present, and whether it was estimated from enough
  data to mean anything
* every constant still hardcoded in the engine that describes THIS author rather
  than authors in general
* whether a new profile could be stood up without editing code

The bar to clear: adding a new author should require corpus and zero code edits.

Usage:  python scripts/portability_audit.py [profile ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimesis_voice import config, ingest, quality, rhetoric, scrub  # noqa: E402
from mimesis_voice.fingerprint import FEATURE_SETS, Fingerprint  # noqa: E402

# Constants in the engine that encode one author's habits rather than a general
# property of prose. Each is listed with what would happen to a different author.
AUTHOR_SHAPED_CONSTANTS = [
    ("scrub.AI_TELL_WORDS", len(scrub.AI_TELL_WORDS),
     "static 2023-vintage LLM vocabulary. Filtered per author by the >=3-uses "
     "whitelist rule, so it adapts, but the LIST itself was never re-derived and "
     "no newer marker set exists in the literature."),
    ("scrub._HEDGE_PATTERNS", len(scrub._HEDGE_PATTERNS),
     "hedge inventory. Already split once because penalising 'seems to' taught "
     "writers to strip the markers that make prose read as authored."),
    ("rhetoric._ESCALATION_OPENERS", len(rhetoric._ESCALATION_OPENERS),
     "narrowed twice against THIS author's corpus (50.5% -> 23.4% -> 0/111 false "
     "positives). rhetoric.py's own docstring says re-measure before pointing it "
     "at another author. Nothing enforces that."),
    ("rhetoric._CLOSING_FLOURISH_RE", 1,
     "measured 0/157 on this author. A different author may genuinely close that "
     "way, in which case this fires on their real writing."),
    ("quality._FUNCTION", len(quality._FUNCTION),
     "closed-class English words. Language-bound, not author-bound: safe across "
     "authors, would need replacing for another language."),
    ("quality.is_repetitive / is_padded thresholds", 2,
     "absolute by design (0.60 overlap, 6.0 filler/1kw). Defensible for any "
     "author, but never validated against a second corpus."),
]

CALIBRATED = [
    ("fingerprint means/stds", "fingerprint.json", "per-author"),
    ("self_baseline + p95 gate", "fingerprint.json", "per-author"),
    ("corpus_spread", "fingerprint.json", "per-author"),
    ("sentence-class terciles (v2)", "fingerprint.json", "per-author"),
    ("banlist minus whitelist", "scrub_calibration.json", "per-author"),
    ("burstiness floor / hedge ceiling", "scrub_calibration.json", "per-author"),
    ("cleft + antithesis bands", "scrub_calibration.json", "per-author"),
    ("density + specificity floors", "scrub_calibration.json", "per-author"),
    ("presence floors", "presence_calibration.json", "per-author"),
]


def audit_profile(slug: str) -> dict:
    prof = config.resolve(slug)
    out = {"slug": slug, "problems": [], "warnings": []}
    if not prof.fingerprint_path.exists():
        out["problems"].append("not calibrated")
        return out

    fp = Fingerprint.load(prof.fingerprint_path)
    cal = scrub.ScrubCalibration.load(prof.scrub_path)
    n_feat = len(FEATURE_SETS.get(fp.feature_set, ()))
    out["n_pieces"] = fp.n_pieces
    out["feature_set"] = fp.feature_set

    if fp.n_pieces < 3 * n_feat:
        out["warnings"].append(
            f"{fp.n_pieces} units for {n_feat} features (want ~{3*n_feat})")

    rz = prof.gate.get("rmsz_max", 1.1)
    if isinstance(rz, str) and rz.lower() == "auto":
        out["gate"] = f"auto -> {fp.fit_threshold:.3f}"
    else:
        out["gate"] = f"HARDCODED {rz}"
        if float(rz) < fp.self_baseline:
            out["problems"].append(
                f"rmsz_max {rz} is below this voice's own self-baseline "
                f"{fp.self_baseline:.3f}: the author's own prose fails this gate")
        else:
            out["warnings"].append(f"rmsz_max is a typed constant, not this voice's p95")

    if not fp.meta.get("corpus_spread"):
        out["warnings"].append("no corpus_spread; slate-collapse detection disabled")

    # Calibrations that came back empty are silently-disabled checks.
    empty = []
    if cal.density_p25 <= 0:
        empty.append("quality density floor")
    if cal.cleft_p95 <= 0:
        empty.append("cleft band")
    if cal.antithesis_p95 <= 0:
        empty.append("antithesis band")
    if empty:
        out["warnings"].append("checks disabled by empty calibration: " + ", ".join(empty))
    return out


def main() -> int:
    slugs = sys.argv[1:] or config.list_profiles()
    print("=" * 74)
    print("PER-PROFILE: is this voice standing on measured ground?")
    print("=" * 74)
    for slug in slugs:
        try:
            r = audit_profile(slug)
        except Exception as e:
            print(f"\n{slug}: audit failed ({e})")
            continue
        head = f"\n{slug}"
        if "n_pieces" in r:
            head += f"  [{r['n_pieces']} units, {r['feature_set']}, gate {r['gate']}]"
        print(head)
        for p in r["problems"]:
            print(f"   FAIL  {p}")
        for w in r["warnings"]:
            print(f"   warn  {w}")
        if not r["problems"] and not r["warnings"]:
            print("   ok    fully calibrated, nothing hardcoded")

    print("\n" + "=" * 74)
    print("ENGINE CONSTANTS THAT DESCRIBE AN AUTHOR RATHER THAN AUTHORS")
    print("=" * 74)
    for name, n, why in AUTHOR_SHAPED_CONSTANTS:
        print(f"\n  {name}  ({n} entries)")
        for line in (why[i:i + 68] for i in range(0, len(why), 68)):
            print(f"      {line}")

    print("\n" + "=" * 74)
    print("CALIBRATED PER AUTHOR (portable by construction)")
    print("=" * 74)
    for what, where, scope in CALIBRATED:
        print(f"  {what:<38} {where:<28} {scope}")

    print("\n" + "=" * 74)
    print("PORTABILITY BAR: a new author should need corpus and zero code edits.")
    print("Standing up a new voice today:")
    print("  mimesis profile new NAME      -> auto gate, per-author everything")
    print("  <drop corpus in source_documents/>")
    print("  mimesis ingest NAME && mimesis calibrate NAME")
    print("  mimesis doctor NAME           -> flags thin calibration and bad gates")
    print("The listed engine constants are the residual risk: they were narrowed")
    print("against one corpus and are applied to all. Re-measure them on a second")
    print("author before trusting the tic checks on that author's prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
