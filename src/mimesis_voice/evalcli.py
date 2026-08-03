"""Discrimination eval: can a judge tell the author's real writing from a generation?

For each held-out real piece we derive a neutral brief, generate a piece in the
voice while excluding that piece and its three nearest neighbours from the anchors
(the leak controls), then ask a ``claude -p`` judge which of the two texts is the
real author. We report the fool-rate (judge picks the generation), the fingerprint
RMS-z distribution of the generations against the corpus self-baseline, and the
per-feature mean absolute z. A ``fingerprint_only`` mode runs the whole scoring
path with no generation, so the eval is exercisable without the CLI.
"""
from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from . import config, embed, gate, ingest
from .fingerprint import FEATURES, Fingerprint

_MIN_WORDS = 120


@dataclass
class EvalResult:
    n_trials: int = 0
    fooled: int = 0
    generated_rmsz: list[float] = field(default_factory=list)
    real_rmsz: list[float] = field(default_factory=list)
    self_baseline: float = 0.0
    per_feature_absz: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    fingerprint_only: bool = False

    @property
    def fool_rate(self) -> float:
        return self.fooled / self.n_trials if self.n_trials else 0.0


def _neutral_brief(piece: str) -> str:
    """A deliberately naive, content-neutral brief seeded from the piece's opening.

    Kept heuristic (no model call) so the leak control is transparent: the brief
    carries the topic, not the author's sentences.
    """
    first = re.split(r"(?<=[.!?])\s+", piece.strip(), maxsplit=1)[0]
    words = first.split()[:15]
    return "Write a short piece on this subject: " + " ".join(words)


def _nearest(pieces: dict[str, str], target: str, backend: str, k: int = 3) -> set[str]:
    names = list(pieces)
    vecs = embed.encode([pieces[n] for n in names], backend=backend)
    tv = embed.embed_one(pieces[target], backend=backend)
    sims = vecs @ tv
    order = np.argsort(-sims)
    out: set[str] = set()
    for i in order:
        if names[i] == target:
            continue
        out.add(names[i])
        if len(out) >= k:
            break
    return out


def _judge(real: str, generated: str, model: str) -> bool:
    """Return True if the judge is fooled (picks the generation as the real author)."""
    texts = [("real", real), ("gen", generated)]
    random.shuffle(texts)
    listing = "\n\n".join(f"[{i}]\n{t}" for i, (_, t) in enumerate(texts))
    prompt = (
        "Two texts follow. Exactly one was written by the real author; the other was "
        "generated to imitate them. Reply with ONLY the number of the text you believe "
        "the real author wrote.\n\n" + listing
    )
    raw = gate.claude_generate(prompt, model=model)
    m = re.search(r"\d+", raw)
    pick = int(m.group()) if m else 0
    pick = pick if 0 <= pick < len(texts) else 0
    return texts[pick][0] == "gen"


def run_eval(
    profile: config.Profile,
    held_out: int = 5,
    model: str = "sonnet",
    fingerprint_only: bool = False,
) -> EvalResult:
    """Run the discrimination eval and return structured results."""
    if not profile.fingerprint_path.exists():
        raise FileNotFoundError(
            f"'{profile.slug}' is not calibrated. Run: mimesis calibrate {profile.slug}"
        )
    fp = Fingerprint.load(profile.fingerprint_path)
    pieces = {
        fn: t
        for fn, t in ingest.read_pieces(profile.db_path).items()
        if len(re.findall(r"[A-Za-z']+", t)) >= _MIN_WORDS
    }
    res = EvalResult(self_baseline=fp.self_baseline, fingerprint_only=fingerprint_only)
    if len(pieces) < held_out:
        res.notes.append(
            f"only {len(pieces)} pieces >= {_MIN_WORDS} words; using all of them."
        )
    names = sorted(pieces, key=lambda n: len(pieces[n]), reverse=True)[: max(1, held_out)]

    absz_acc: dict[str, list[float]] = {f: [] for f in FEATURES}
    for name in names:
        real = pieces[name]
        r_rmsz, _ = fp.distance_detail(real)
        res.real_rmsz.append(r_rmsz)

        if fingerprint_only:
            continue

        brief = _neutral_brief(real)
        exclude = {name} | _nearest(pieces, name, profile.embed_backend, k=3)
        try:
            comp = gate.compose(brief, profile, model=model, exclude_files=exclude)
        except Exception as exc:  # generation unavailable, etc.
            res.notes.append(f"trial '{name}' skipped: {exc}")
            continue
        gen = comp.output or ""
        if not gen:
            res.notes.append(f"trial '{name}' skipped: empty generation")
            continue
        g_rmsz, g_zs = fp.distance_detail(gen)
        res.generated_rmsz.append(g_rmsz)
        for f in FEATURES:
            absz_acc[f].append(abs(g_zs[f]))
        res.n_trials += 1
        try:
            if _judge(real, gen, model):
                res.fooled += 1
        except Exception as exc:
            res.notes.append(f"judge for '{name}' failed: {exc}")

    res.per_feature_absz = {
        f: (statistics.mean(v) if v else 0.0) for f, v in absz_acc.items()
    }
    return res


def render(res: EvalResult, profile: config.Profile) -> str:
    lines = [f"# Discrimination eval — {profile.name} ({profile.slug})"]
    real_mean = statistics.mean(res.real_rmsz) if res.real_rmsz else 0.0
    lines.append(
        f"corpus self-baseline RMS-z: {res.self_baseline:.3f}  |  "
        f"held-out real pieces mean RMS-z: {real_mean:.3f} (n={len(res.real_rmsz)})"
    )
    if res.fingerprint_only:
        lines.append("(fingerprint-only mode: no generation, no judging)")
    else:
        gen_mean = statistics.mean(res.generated_rmsz) if res.generated_rmsz else 0.0
        gen_sd = statistics.stdev(res.generated_rmsz) if len(res.generated_rmsz) > 1 else 0.0
        ratio = gen_mean / res.self_baseline if res.self_baseline else float("inf")
        lines.append(
            f"generations: mean RMS-z {gen_mean:.3f} (stdev {gen_sd:.3f}, n={len(res.generated_rmsz)}); "
            f"{ratio:.2f}x the self-baseline (acceptance: <= 2x)"
        )
        lines.append(f"fool-rate: {res.fool_rate:.0%} ({res.fooled}/{res.n_trials} trials)")
        worst = sorted(res.per_feature_absz.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("top deviating features (mean |z|): " + ", ".join(f"{f}={v:.2f}" for f, v in worst))
    for n in res.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)
