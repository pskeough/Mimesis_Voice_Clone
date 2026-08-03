"""``mimesis`` command-line interface.

Subcommands::

    mimesis profile list|use NAME|new NAME [--backend fast|style]
    mimesis ingest [NAME] [--force]
    mimesis calibrate [NAME]
    mimesis compose NAME "task" [--format F] [--dry-run] [--model M] [--examples N]
    mimesis scrub [NAME] [--source FILE] ["text" | -]
    mimesis eval [NAME] [--held-out N] [--fingerprint-only] [--model M]
    mimesis audit [NAME]          # cadence diagnostic: does this voice have a rhythm gap?
    mimesis doctor

The CLI is the human entry point; ``mimesis_voice.server`` is the MCP entry point.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from . import accepted as accepted_mod
from . import composite as composite_mod
from . import config, evalcli, gate, ingest
from . import scrub as scrub_mod
from . import presence as presence_mod
from . import fingerprint as fingerprint_mod
from .fingerprint import calibrate as fp_calibrate
from .fingerprint import calibrate_weighted, recency_weights

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _resolve(name: str | None) -> config.Profile:
    return config.resolve(name) if name else config.resolve_active()


# --- profile ------------------------------------------------------------------


def cmd_profile_list() -> int:
    config.ensure_profiles()
    st = config.read_state()
    profs = config.list_profiles()
    print(f"Voice mode: {'ON' if st['enabled'] else 'OFF'}")
    if not profs:
        print("No profiles yet. Create one: mimesis profile new NAME")
        return 0
    active = config.resolve_active().slug
    for slug in profs:
        cfg = config.load_profile_config(slug)
        mark = "*" if slug == active else " "
        prof = config.resolve(slug)
        n_docs = len(ingest.find_documents(prof.source_dir))
        calibrated = "calibrated" if prof.fingerprint_path.exists() else "uncalibrated"
        comp = cfg.get("compose_from") or []
        detail = (
            "composite of " + ", ".join(
                f"{c.get('profile')}x{c.get('weight', 1)}" for c in comp)
            if comp else f"{n_docs} docs"
        )
        print(
            f"  {mark} {slug:<16} {cfg['author_name']:<12} "
            f"backend={cfg['embed_backend']:<5} {detail}, {calibrated}"
        )
    return 0


def cmd_profile_use(name: str) -> int:
    slug = config.slugify(name)
    config.ensure_profiles()
    if slug not in config.list_profiles():
        print(f"No profile '{slug}'. Available: {', '.join(config.list_profiles()) or '(none)'}")
        return 1
    config.write_state(active=slug)
    print(f"Active voice: {slug}")
    return 0


def cmd_profile_new(name: str, backend: str) -> int:
    prof = config.create_profile(name, name.replace("-", " ").title(), backend)
    print(f"Created profile '{prof.slug}' (backend={prof.embed_backend}).")
    print(f"Add documents to: {prof.source_dir}")
    print(f"Then: mimesis ingest {prof.slug} && mimesis calibrate {prof.slug}")
    return 0


# --- ingest / calibrate -------------------------------------------------------


def cmd_ingest(name: str | None, force: bool) -> int:
    prof = _resolve(name)
    return 0 if ingest.run(prof, force=force) else 1


def cmd_audit(name: str | None) -> int:
    """Report a voice's cadence gap: is its rhythm actually unlike the author?

    This is the validated half of the cadence work. The order-aware feature set
    is the only thing in the system that can see rhythm at all -- every feature in
    the original fingerprint is order-invariant, so a text whose sentences are
    sorted shortest-to-longest scores identically to real prose. As a *generation
    gate* the order-aware set did not earn its place (see evals/CADENCE_FINDINGS.md
    section 7d); as a *diagnostic* it locates a real defect.

    Reports, for the corpus itself and for any accepted drafts:
      - cadence RMS-z, leave-one-out, so nothing is scored against a ruler
        containing itself
      - the worst-deviating cadence features, in plain language
    """
    from . import cadence as cadence_mod

    prof = _resolve(name)
    pieces = list(ingest.read_pieces(prof.db_path).values())
    if len(pieces) < 10:
        print(f"'{prof.slug}': need >=10 corpus pieces to audit, got {len(pieces)}.")
        return 1
    if ingest.needs_segmentation(pieces):
        pieces = ingest.segment_for_calibration(pieces)
    fp = fingerprint_mod.calibrate(pieces, feature_set="v2")

    import math
    import statistics

    names = cadence_mod.CADENCE_FEATURES
    rows = [fingerprint_mod.extract_all(t, "v2", fp.cuts) for t in pieces]

    def block(f: "fingerprint_mod.Fingerprint", text: str) -> float:
        _, zs = f.distance_detail(text)
        return math.sqrt(sum(zs[k] ** 2 for k in names) / len(names))

    human = []
    for i, t in enumerate(pieces):
        rest = rows[:i] + rows[i + 1:]
        loo = fingerprint_mod.Fingerprint(
            means={k: statistics.mean(r[k] for r in rest) for k in fp.features},
            stds={k: statistics.stdev([r[k] for r in rest]) for k in fp.features},
            feature_set="v2", cuts=fp.cuts,
        )
        human.append(block(loo, t))
    h = statistics.mean(human)
    print(f"'{prof.slug}': {len(pieces)} units. Author's own cadence distance: {h:.3f}")
    print(f"  sentence classes (this author's terciles): short <={fp.cuts[0]:.0f}w, "
          f"long >{fp.cuts[1]:.0f}w")

    drafts = list(accepted_mod.accepted_texts(prof))
    if not drafts:
        print("\n  No accepted drafts to compare. Generate and `mimesis accept` some,")
        print("  or audit saved generations with evals/cadence_audit_external.py.")
        return 0

    gen = [block(fp, t) for t in drafts if len(t.split()) >= 80]
    if not gen:
        print("\n  Accepted drafts are all too short to score (need >=80 words).")
        return 0
    g = statistics.mean(gen)
    gap = g - h
    verdict = ("a real cadence gap; worth trying feature_set v2 on this voice"
               if gap > 0.3 else
               "marginal" if gap > 0.1 else
               "already at parity; cadence gating would be a no-op here")
    print(f"  Accepted drafts ({len(gen)}): {g:.3f}   gap {gap:+.3f}  -> {verdict}")

    acc: dict[str, list[float]] = {k: [] for k in names}
    for t in drafts:
        _, zs = fp.distance_detail(t)
        for k in names:
            acc[k].append(zs[k])
    ranked = sorted(acc.items(), key=lambda kv: abs(statistics.mean(kv[1])), reverse=True)
    print("\n  Worst cadence features:")
    for k, v in ranked[:5]:
        mz = statistics.mean(v)
        print(f"    {k:<22}{mz:>+6.2f} SD  {cadence_mod.hint(k, mz)}")
    return 0


def _report_calibration_confidence(fp, feature_set: str) -> None:
    """Warn when there are too few pieces to estimate the feature set from.

    Calibration succeeds at n=5 and says nothing about how well-determined the
    result is. It should: a standard deviation estimated from a handful of
    pieces can collapse toward zero on a feature the author rarely uses, and then
    every z-score on that feature explodes. Measured case: emdash_per_100w
    calibrated on 34 pieces had std 0.0117, and the author's own held-out prose
    averaged |z| = 12 on it, swamping the other 25 features in the aggregate.
    Winsorization (fingerprint.Z_CLIP) bounds the damage; it does not make the
    estimate trustworthy.

    The 3x-features heuristic is a rule of thumb, not a theorem, and is labelled
    as such: the point is to stop the tool reporting a p95 threshold with the
    same confidence at n=11 as at n=111.
    """
    n_feat = len(fingerprint_mod.FEATURE_SETS.get(feature_set, ()))
    comfortable = 3 * n_feat
    if fp.n_pieces >= comfortable:
        return
    tier = "THIN" if fp.n_pieces >= n_feat else "VERY THIN"
    print(
        f"  [{tier} CALIBRATION] {fp.n_pieces} pieces for {n_feat} features "
        f"({feature_set}). Feature standard deviations, the self-baseline "
        f"({fp.self_baseline:.3f}) and the p95 fit threshold "
        f"({fp.fit_threshold:.3f}) are all weakly determined at this n; treat "
        f"gate decisions as provisional. Rule of thumb: ~{comfortable} pieces. "
        f"Add corpus, or lower the ingest chunk size to yield more units."
    )



def _calibrate_composite(prof: config.Profile) -> int:
    """Calibrate a voice blended from several corpora.

    The fingerprint uses weighted means and stds, so the dominant source sets the
    central tendency while the others stabilise the variance -- which is the whole
    point for a corpus too thin to estimate 13 features from. The scrubber
    intersects banlists and unions whitelists (see composite.blend_scrub).
    """
    srcs = composite_mod.load_sources(prof, min_words=120)
    if not srcs:
        print(f"'{prof.slug}': no source corpus is usable. Ingest the sources first.")
        return 1
    print(composite_mod.describe(prof))

    texts, weights = composite_mod.weighted_corpus(srcs)
    try:
        fp = calibrate_weighted(texts, weights, feature_set=prof.feature_set)
    except ValueError as e:
        print(f"Fingerprint calibration failed: {e}")
        return 1
    fp.meta = dict(fp.meta or {})
    fp.meta["composite"] = {s.slug: s.weight for s in srcs}
    fp.meta["shares"] = composite_mod.shares(srcs)
    fp.save(prof.fingerprint_path)
    _report_calibration_confidence(fp, prof.feature_set)

    cal = composite_mod.blend_scrub(srcs, prof.whitelist)
    cal.save(prof.scrub_path)
    pres = presence_mod.calibrate(texts)
    pres.save(prof.presence_path)
    print(
        f"Calibrated composite '{prof.slug}': fingerprint over {fp.n_pieces} weighted "
        f"units (self-baseline RMS-z {fp.self_baseline:.3f}, p95 {fp.fit_threshold:.3f}); "
        f"scrub banlist {len(cal.banned_words)} words (intersection), whitelist "
        f"{len(cal.whitelist)} (union), burstiness floor {cal.burstiness_floor:.2f}."
    )
    return 0


def cmd_calibrate(name: str | None) -> int:
    prof = _resolve(name)

    # A composite voice calibrates from its declared sources, with each source's
    # influence normalized to its weight share rather than its piece count.
    if composite_mod.is_composite(prof):
        problems = composite_mod.validate(prof)
        if problems:
            print(f"Cannot calibrate composite '{prof.slug}':")
            for x in problems:
                print(f"  - {x}")
            return 1
        return _calibrate_composite(prof)

    pieces = ingest.read_pieces(prof.db_path)
    if not pieces:
        print(f"No store for '{prof.slug}'. Run: mimesis ingest {prof.slug}")
        return 1
    texts = list(pieces.values())
    # Segment only when the corpus is lopsided enough for length to contaminate
    # the feature standard deviations. A corpus of comparable pieces is left
    # alone, so this changes nothing for profiles that were already well-formed.
    # Segment when the corpus is lopsided enough for length to contaminate the
    # feature stds, OR when it is word-rich but document-poor and cutting finer
    # buys the unit count calibration needs. The target adapts to both.
    n_feat = len(fingerprint_mod.FEATURE_SETS.get(prof.feature_set, ()))
    target = ingest.calibration_target(texts, n_feat)
    candidate = ingest.segment_for_calibration(texts, target=target)
    lopsided = ingest.needs_segmentation(texts)
    thin = len(texts) < 3 * n_feat and len(candidate) > len(texts)
    if lopsided or thin:
        reason = ("piece lengths are uneven" if lopsided
                  else "too few pieces for the feature count")
        print(
            f"Corpus {reason}; segmenting {len(texts)} pieces into "
            f"{len(candidate)} units of ~{target} words for calibration."
        )
        texts = candidate
    try:
        fp = fp_calibrate(texts, feature_set=prof.feature_set)
    except ValueError as e:
        print(f"Fingerprint calibration failed: {e}")
        return 1
    fp.save(prof.fingerprint_path)
    _report_calibration_confidence(fp, prof.feature_set)
    cal = scrub_mod.calibrate(texts, whitelist=prof.whitelist)
    cal.save(prof.scrub_path)
    # Presence floors are the corpus 25th percentile or the published-field floor, whichever is
    # higher. A terse corpus must not license a draft with nobody in it, so the field floor is a
    # backstop rather than a starting point.
    pres = presence_mod.calibrate(texts)
    pres.save(prof.presence_path)
    print(
        f"Calibrated '{prof.slug}': fingerprint over {fp.n_pieces} pieces "
        f"(self-baseline RMS-z {fp.self_baseline:.3f}); "
        f"scrub banlist {len(cal.banned_words)} words, whitelist {len(cal.whitelist)}, "
        f"burstiness floor {cal.burstiness_floor:.2f}; "
        f"presence floors {pres.floors} over {pres.n_pieces} pieces."
    )
    return 0


# --- accept / recalibrate (Upgrade 3: learn from accept/edit) ------------------


def cmd_accept(args) -> int:
    prof = _resolve(args.name)
    after = Path(args.file).read_text(encoding="utf-8").strip()
    if not after:
        print(f"'{args.file}' is empty; nothing to accept.")
        return 1
    if args.from_file:
        before = Path(args.from_file).read_text(encoding="utf-8")
        res = accepted_mod.record_edit(prof, before, after, task=args.task)
        acc = res["accepted"]
        print(
            f"Recorded edit for '{prof.slug}': {res['pairs_added']} contrastive pair(s) "
            f"mined (pre-edit=avoid, post-edit=target), accepted as {acc['id'] if acc else '?'}."
        )
    else:
        rec = accepted_mod.record_accept(prof, after, task=args.task, source=args.file)
        print(f"Accepted draft for '{prof.slug}' as {rec['id']} "
              f"({len(after.split())} words). Total accepted: {len(accepted_mod.load_accepted(prof))}.")
    print(f"Next: mimesis recalibrate {prof.slug}   # fold the accepted set into the fingerprint")
    return 0


def cmd_recalibrate(args) -> int:
    prof = _resolve(args.name)
    pieces = ingest.read_pieces(prof.db_path)
    if not pieces:
        print(f"No store for '{prof.slug}'. Run: mimesis ingest {prof.slug}")
        return 1
    acc_texts = accepted_mod.accepted_texts(prof)  # oldest first
    if not acc_texts:
        print(f"No accepted set for '{prof.slug}'. Add some with: mimesis accept {prof.slug} <file>")
        return 1
    base_texts = list(pieces.values())
    weights = [1.0] * len(base_texts) + recency_weights(
        len(acc_texts), base_weight=args.base_weight, half_life=args.half_life
    )
    try:
        fp = calibrate_weighted(base_texts + acc_texts, weights)
    except ValueError as e:
        print(f"Recalibration failed: {e}")
        return 1
    fp.meta = {
        "recalibrated_from_accepted": True,
        "n_accepted": len(acc_texts),
        "base_weight": args.base_weight,
        "half_life": args.half_life,
    }
    # Preserve the pre-accept fingerprint once, so the fold-in is reversible.
    backup = prof.fingerprint_path.with_name("fingerprint.base.json")
    if prof.fingerprint_path.exists() and not backup.exists():
        backup.write_text(prof.fingerprint_path.read_text(encoding="utf-8"), encoding="utf-8")
    fp.save(prof.fingerprint_path)
    print(
        f"Recalibrated '{prof.slug}' with {len(acc_texts)} accepted sample(s) "
        f"(base_weight={args.base_weight}, half_life={args.half_life}); "
        f"self-baseline RMS-z {fp.self_baseline:.3f}. Base fingerprint saved to {backup.name}."
    )
    return 0


# --- compose / scrub / eval ---------------------------------------------------


def cmd_compose(args) -> int:
    prof = _resolve(args.name)
    try:
        res = gate.compose(
            args.task,
            prof,
            fmt=args.format,
            dry_run=args.dry_run,
            n_examples=args.examples,
            model=args.model,
        )
    except FileNotFoundError as e:
        print(str(e))
        return 1
    except RuntimeError as e:
        print(f"Generation failed: {e}")
        print("Tip: re-run with --dry-run to exercise the pipeline without the model.")
        return 1

    if res.dry_run:
        print("=== DRY RUN (no generation) ===")
        print(res.kit)
        print("\n=== NOTES ===")
        for n in res.notes:
            print(f"- {n}")
        return 0

    print("=== OUTPUT ===")
    print(res.output)
    c = res.chosen
    print("\n=== GATE ===")
    print(
        f"candidates={len(res.candidates)} survivors={len(res.survivors)} "
        f"rewrites={res.iterations} chosen RMS-z={c.rmsz:.3f}"
    )
    print("\n=== SCRUB ===")
    print(scrub_mod.render(c.scrub, prof.name) if c.scrub else "(no scrub report)")
    for n in res.notes:
        print(f"- {n}")
    return 0


def _read_text_arg(text: str | None) -> str:
    if text in (None, "-"):
        return sys.stdin.read()
    return text


def cmd_scrub(args) -> int:
    prof = _resolve(args.name)
    if not prof.scrub_path.exists():
        print(f"'{prof.slug}' is not calibrated. Run: mimesis calibrate {prof.slug}")
        return 1
    cal = scrub_mod.ScrubCalibration.load(prof.scrub_path)
    text = _read_text_arg(args.text)
    source = Path(args.source).read_text(encoding="utf-8") if args.source else None
    rep = scrub_mod.analyze(text, cal, source=source, detect=args.detect)
    print(scrub_mod.render(rep, prof.name))
    return 0


def cmd_eval(args) -> int:
    prof = _resolve(args.name)
    try:
        res = evalcli.run_eval(
            prof,
            held_out=args.held_out,
            model=args.model,
            fingerprint_only=args.fingerprint_only,
        )
    except FileNotFoundError as e:
        print(str(e))
        return 1
    print(evalcli.render(res, prof))
    return 0


# --- doctor -------------------------------------------------------------------

_ICON = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


def cmd_doctor(name: str | None = None) -> int:
    results: list[str] = []

    def record(level: str, label: str, detail: str = "") -> None:
        results.append(level)
        line = f"{_ICON[level]} {label}"
        if detail:
            line += f"\n        {detail}"
        print(line)

    print("=" * 68)
    print("  MIMESIS v2 — DOCTOR")
    print("=" * 68)

    missing = []
    for mod, pip_name in [
        ("fastmcp", "fastmcp"),
        ("fastembed", "fastembed"),
        ("numpy", "numpy"),
        ("docx", "python-docx"),
    ]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    deps_ok = not missing
    record("OK" if deps_ok else "FAIL", "Dependencies",
           "fastmcp, fastembed, numpy, python-docx import"
           if deps_ok else f"missing: {', '.join(missing)}")

    config.ensure_profiles()
    st = config.read_state()
    profs = config.list_profiles()
    prof = _resolve(name)
    record("OK" if profs else "WARN", "Profiles",
           f"active={prof.slug} ({prof.name}); {len(profs)} profile(s); voice mode "
           f"{'ON' if st['enabled'] else 'OFF'}")

    docs = ingest.find_documents(prof.source_dir)
    record("OK" if docs else "WARN", "Source documents",
           f"{len(docs)} file(s) in {prof.source_dir}" if docs
           else f"none in {prof.source_dir}")

    db_ok = prof.db_path.exists()
    if db_ok:
        try:
            conn = sqlite3.connect(str(prof.db_path))
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            has_fts = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'"
            ).fetchone() is not None
            conn.close()
            record("OK" if n else "FAIL", "Store",
                   f"{n} chunks, FTS5={'on' if has_fts else 'off'}")
            db_ok = n > 0
        except Exception as e:
            record("FAIL", "Store", f"unreadable: {e}")
            db_ok = False
    else:
        record("FAIL", "Store", f"{prof.db_path} not found. Run: mimesis ingest {prof.slug}")

    if composite_mod.is_composite(prof):
        problems = composite_mod.validate(prof)
        record("OK" if not problems else "FAIL", "Composite",
               composite_mod.describe(prof) if not problems else "; ".join(problems))

    cal_ok = prof.fingerprint_path.exists() and prof.scrub_path.exists()
    record("OK" if cal_ok else "FAIL", "Calibration",
           "fingerprint.json + scrub_calibration.json present" if cal_ok
           else f"missing. Run: mimesis calibrate {prof.slug}")

    # Calibration density and threshold mode. Both failed silently on a second
    # author: a hardcoded rmsz_max of 1.1 sat below what that corpus's own prose
    # scores, so no candidate ever passed the gate, and a 26-feature fingerprint
    # was being estimated from 15 pieces.
    if cal_ok:
        try:
            fp = fingerprint_mod.Fingerprint.load(prof.fingerprint_path)
            n_feat = len(fp.features)
            want = 3 * n_feat
            dens = "OK" if fp.n_pieces >= want else "WARN"
            record(dens, "Calibration density",
                   f"{fp.n_pieces} units for {n_feat} features ({fp.feature_set}); "
                   f"rule of thumb ~{want}. self-baseline {fp.self_baseline:.3f}, "
                   f"p95 {fp.fit_threshold:.3f}")
            rz = prof.gate.get("rmsz_max", 1.1)
            if isinstance(rz, str) and rz.lower() == "auto":
                record("OK", "Gate threshold",
                       f"auto -> {fp.fit_threshold:.3f} (this profile's own p95)")
            elif fp.fit_threshold > 0 and float(rz) < fp.self_baseline:
                record("FAIL", "Gate threshold",
                       f"rmsz_max={rz} is BELOW this voice's self-baseline "
                       f"({fp.self_baseline:.3f}): the author's own writing would "
                       f"fail this gate. Set \"rmsz_max\": \"auto\".")
            elif fp.fit_threshold > 0 and float(rz) < fp.fit_threshold * 0.75:
                record("WARN", "Gate threshold",
                       f"rmsz_max={rz} is well under the calibrated p95 "
                       f"({fp.fit_threshold:.3f}); the gate will reject often and "
                       f"burn rewrites. Consider \"rmsz_max\": \"auto\".")
            else:
                record("OK", "Gate threshold", f"rmsz_max={rz} (p95 {fp.fit_threshold:.3f})")
        except Exception as e:
            record("WARN", "Calibration density", f"unreadable: {e}")

    emb_ok = False
    if deps_ok:
        try:
            from . import embed
            import numpy as np

            v = embed.embed_one("a short probe sentence for the doctor.", backend="fast")
            if np.asarray(v).shape[0] == 384:
                record("OK", "Embedding (fast)", "bge-small produced a 384-dim vector")
                emb_ok = True
            else:
                record("FAIL", "Embedding (fast)", f"unexpected dim {np.asarray(v).shape[0]}")
        except Exception as e:
            record("FAIL", "Embedding (fast)", f"failed: {e}")
    else:
        record("SKIP", "Embedding (fast)", "dependencies missing")

    if db_ok and emb_ok and cal_ok:
        try:
            res = gate.compose("write a short note about morning routines", prof, dry_run=True)
            ok = res.dry_run and res.kit and "COMPOSITION KIT" in res.kit
            record("OK" if ok else "FAIL", "Compose (dry-run)",
                   "; ".join(res.notes) if ok else "unexpected dry-run output")
        except Exception as e:
            record("FAIL", "Compose (dry-run)", f"failed: {e}")
    else:
        record("SKIP", "Compose (dry-run)", "needs store + embedding + calibration")

    import shutil

    record("OK" if shutil.which("claude") else "SKIP", "claude CLI",
           "found on PATH (generation available)" if shutil.which("claude")
           else "not on PATH; compose --dry-run and fingerprint-only eval still work")

    fails = results.count("FAIL")
    warns = results.count("WARN")
    print("=" * 68)
    if fails:
        print(f"  RESULT: {fails} failure(s), {warns} warning(s).")
    elif warns:
        print(f"  RESULT: healthy, {warns} warning(s).")
    else:
        print("  RESULT: all checks passed.")
    print("=" * 68)
    return 1 if fails else 0


# --- arg parsing --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mimesis", description="Mimesis v2 voice engine.")
    sub = ap.add_subparsers(dest="cmd")

    p_prof = sub.add_parser("profile", help="list/use/new profiles")
    psub = p_prof.add_subparsers(dest="pcmd")
    psub.add_parser("list")
    p_use = psub.add_parser("use")
    p_use.add_argument("name")
    p_new = psub.add_parser("new")
    p_new.add_argument("name")
    p_new.add_argument("--backend", choices=["fast", "style"], default="fast")

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("name", nargs="?", default=None)
    p_ing.add_argument("--force", action="store_true")

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("name", nargs="?", default=None)

    p_acc = sub.add_parser("accept", help="record an accepted draft (optionally an edit)")
    p_acc.add_argument("name")
    p_acc.add_argument("file", help="the accepted / post-edit text file")
    p_acc.add_argument("--from", dest="from_file", default=None,
                       help="the pre-edit file; mines contrastive pairs from the diff")
    p_acc.add_argument("--task", default=None, help="the brief this draft answered")

    p_recal = sub.add_parser("recalibrate", help="fold the accepted set into the fingerprint")
    p_recal.add_argument("name", nargs="?", default=None)
    p_recal.add_argument("--base-weight", dest="base_weight", type=float, default=4.0)
    p_recal.add_argument("--half-life", dest="half_life", type=float, default=3.0)

    p_comp = sub.add_parser("compose")
    p_comp.add_argument("name")
    p_comp.add_argument("task")
    p_comp.add_argument("--format", default=None)
    p_comp.add_argument("--dry-run", action="store_true")
    p_comp.add_argument("--model", default="sonnet")
    p_comp.add_argument("--examples", type=int, default=5)

    p_scrub = sub.add_parser("scrub")
    p_scrub.add_argument("name", nargs="?", default=None)
    p_scrub.add_argument("text", nargs="?", default=None, help="text, or - for stdin")
    p_scrub.add_argument("--source", default=None, help="source file for fidelity audit")
    p_scrub.add_argument("--detect", action="store_true")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("name", nargs="?", default=None)
    p_eval.add_argument("--held-out", type=int, default=5)
    p_eval.add_argument("--fingerprint-only", action="store_true")
    p_eval.add_argument("--model", default="sonnet")

    p_aud = sub.add_parser("audit", help="report a voice's cadence gap (rhythm diagnostic)")
    p_aud.add_argument("name", nargs="?")
    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("name", nargs="?")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "profile":
        if args.pcmd == "use":
            return cmd_profile_use(args.name)
        if args.pcmd == "new":
            return cmd_profile_new(args.name, args.backend)
        return cmd_profile_list()
    if args.cmd == "ingest":
        return cmd_ingest(args.name, args.force)
    if args.cmd == "calibrate":
        return cmd_calibrate(args.name)
    if args.cmd == "accept":
        return cmd_accept(args)
    if args.cmd == "recalibrate":
        return cmd_recalibrate(args)
    if args.cmd == "compose":
        return cmd_compose(args)
    if args.cmd == "scrub":
        return cmd_scrub(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "audit":
        return cmd_audit(args.name)
    if args.cmd == "doctor":
        return cmd_doctor(args.name)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
