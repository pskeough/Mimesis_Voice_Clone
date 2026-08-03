"""Upgrade-3 experiment: does learning from accepted/edited samples move new
outputs toward what the author keeps?

Method (honest, self-contained):

1. ``select`` -- cluster the voice's real corpus in stylometric z-space and take a
   coherent, *distinctive* cluster (the one whose centroid sits furthest from the
   corpus mean) as the stand-in "accepted set" A: genuinely the author's writing,
   with a style signature distinct enough that convergence is measurable. The
   selection is deterministic and printed in full, no cherry-picking by hand.
2. ``setup`` -- clone the base profile, ``record_accept`` the slice, optionally
   mine a few genuine edit pairs (plain-AI draft on the piece's own topic = the
   pre-edit "avoid"; the real piece = the post-edit "target"), then recalibrate
   the fingerprint with the recency-weighted fold-in.
3. ``target`` -- freeze an ``AcceptedTarget`` (A's centroid, z-scaled by the stable
   base reference fingerprint) as a fixed ruler.

The test: generate the same curated briefs with the base engine (baseline arm)
and the recalibrated engine (recalibrate arm), then compare mean distance to A.
If the loop works, the recalibrate arm sits closer to A while keeping base-voice
RMS-z fidelity. Measurement is done by aggregate.py (``dist->accepted``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from mimesis_voice import accepted as accepted_mod
    from mimesis_voice import config, gate, ingest
    from mimesis_voice.evalcli import _neutral_brief
    from mimesis_voice.fingerprint import FEATURES, Fingerprint, accepted_target, extract_features
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import accepted as accepted_mod
    from mimesis_voice import config, gate, ingest
    from mimesis_voice.evalcli import _neutral_brief
    from mimesis_voice.fingerprint import FEATURES, Fingerprint, accepted_target, extract_features

EVALS_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = EVALS_DIR / "reference"
_MIN_WORDS = 120


def _pieces(profile: config.Profile) -> dict[str, str]:
    return {
        fn: t for fn, t in ingest.read_pieces(profile.db_path).items()
        if len(re.findall(r"[A-Za-z']+", t)) >= _MIN_WORDS
    }


def _zmatrix(names: list[str], pieces: dict[str, str], fp: Fingerprint):
    rows = []
    for n in names:
        feats = extract_features(pieces[n])
        z = [((feats[f] - fp.means[f]) / fp.stds[f]) if fp.stds.get(f, 0) > 1e-9 else 0.0 for f in FEATURES]
        rows.append(z)
    return np.array(rows, dtype=np.float64)


def cmd_select(args) -> int:
    prof = config.resolve_named(args.voice)
    fp = Fingerprint.load(REFERENCE_DIR / f"{args.voice}.fingerprint.json")
    pieces = _pieces(prof)
    names = sorted(pieces)
    Z = _zmatrix(names, pieces, fp)
    n = len(names)
    k = min(args.k, n // 2)

    from sklearn.cluster import KMeans

    n_clusters = max(2, n // args.k)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(Z)
    # pick the cluster whose centroid is furthest from the corpus mean (origin in
    # z-space) and that has at least a few members -> distinctive + coherent.
    best_c, best_score = None, -1.0
    for c in range(n_clusters):
        members = [i for i in range(n) if km.labels_[i] == c]
        if len(members) < 3:
            continue
        centroid = Z[members].mean(axis=0)
        dist = float(np.linalg.norm(centroid) / np.sqrt(len(FEATURES)))  # RMS-z of centroid
        if dist > best_score:
            best_c, best_score = c, dist
    members = [i for i in range(n) if km.labels_[i] == best_c]
    # anchor on the distinctive cluster's centroid, then take the k nearest pieces
    # from the whole corpus to it: coherent (all near one style target) and always
    # exactly k, even when the raw cluster is small.
    centroid = Z[members].mean(axis=0)
    order = sorted(range(n), key=lambda i: float(np.linalg.norm(Z[i] - centroid)))
    chosen = [names[i] for i in order[:k]]
    print(f"[select] voice={args.voice} corpus={n} clusters={n_clusters} "
          f"picked cluster {best_c} (centroid RMS-z {best_score:.3f}), taking {len(chosen)} pieces:")
    for nm in chosen:
        print(f"  {nm}")
    print("SLICE=" + ",".join(chosen))
    return 0


def cmd_setup(args) -> int:
    src = config.resolve_named(args.profile)
    slice_names = [s.strip() for s in args.slice.split(",") if s.strip()]
    pieces = _pieces(src)
    missing = [s for s in slice_names if s not in pieces]
    if missing:
        print(f"slice names not in corpus: {missing}")
        return 1

    # clone base -> dst (keep the base backend to isolate the recalibrate effect)
    sys.path.insert(0, str(EVALS_DIR))
    from mkprofile import clone  # type: ignore
    clone(args.profile, args.dst, backend=src.embed_backend)
    dst = config.resolve_named(args.dst)
    # copy the built store + calibration so the clone is immediately usable (same
    # corpus, same backend -> identical embeddings; no need to re-ingest).
    import shutil
    dst.data_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("store.sqlite", "fingerprint.json", "scrub_calibration.json"):
        srcf = src.data_dir / fn
        if srcf.exists():
            shutil.copy2(srcf, dst.data_dir / fn)

    # accept the slice pieces
    for nm in slice_names:
        accepted_mod.record_accept(dst, pieces[nm], task=None, source=f"corpus:{nm}")
    print(f"[setup] accepted {len(slice_names)} pieces into '{args.dst}'")

    # mine a few genuine edit pairs: plain-AI draft on the piece's own topic (avoid)
    # -> the real piece (target). Same-topic, so the pair teaches a real rewrite move.
    n_edits = min(args.edits, len(slice_names))
    for nm in slice_names[:n_edits]:
        real = pieces[nm]
        brief = _neutral_brief(real)
        try:
            before = gate.claude_generate(
                f"{brief}\n\nWrite 150-230 words. Output only the piece.", model=args.model
            )
        except Exception as exc:
            print(f"  edit skip {nm}: {exc}")
            continue
        res = accepted_mod.record_edit(dst, before, real, task=brief, accept_after=False)
        print(f"  edit {nm}: +{res['pairs_added']} contrastive pair(s)")

    # recalibrate the dst fingerprint with the recency-weighted fold-in
    import subprocess
    py = sys.executable
    subprocess.run([py, "-m", "mimesis_voice.cli", "recalibrate", args.dst,
                    "--base-weight", str(args.base_weight), "--half-life", str(args.half_life)],
                   check=False)
    return 0


def cmd_target(args) -> int:
    src = config.resolve_named(args.profile)
    slice_names = [s.strip() for s in args.slice.split(",") if s.strip()]
    pieces = _pieces(src)
    base_fp = Fingerprint.load(REFERENCE_DIR / f"{args.voice}.fingerprint.json")
    tgt = accepted_target([pieces[n] for n in slice_names if n in pieces], scale_from=base_fp)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / f"{args.voice}.accepted_target.json"
    tgt.save(out)
    # sanity: distance of the slice itself (should be small) vs corpus mean pieces
    slice_d = [tgt.distance(pieces[n]) for n in slice_names if n in pieces]
    import statistics
    print(f"[target] voice={args.voice} accepted n={tgt.n} -> {out}")
    print(f"  mean distance of the accepted slice to its own centroid: {statistics.mean(slice_d):.3f}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="recal_experiment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sel = sub.add_parser("select")
    p_sel.add_argument("--voice", required=True)
    p_sel.add_argument("--k", type=int, default=8)
    p_set = sub.add_parser("setup")
    p_set.add_argument("--voice", required=True)
    p_set.add_argument("--profile", required=True)
    p_set.add_argument("--dst", required=True)
    p_set.add_argument("--slice", required=True)
    p_set.add_argument("--edits", type=int, default=3)
    p_set.add_argument("--model", default="sonnet")
    p_set.add_argument("--base-weight", dest="base_weight", type=float, default=4.0)
    p_set.add_argument("--half-life", dest="half_life", type=float, default=3.0)
    p_tgt = sub.add_parser("target")
    p_tgt.add_argument("--voice", required=True)
    p_tgt.add_argument("--profile", required=True)
    p_tgt.add_argument("--slice", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "setup":
        return cmd_setup(args)
    if args.cmd == "target":
        return cmd_target(args)
    return 1


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
