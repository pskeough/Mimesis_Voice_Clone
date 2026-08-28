"""Supervised preference capture: store what the author actually judged.

Every calibration signal in the engine is unsupervised. The fingerprint measures
distance to the author's corpus; the scrub measures distance to a banlist. Neither
has ever seen the author say "this draft is better than that one".

``evals/CADENCE_FINDINGS.md`` reaches the same conclusion from the other
direction. Twenty-six order-aware features detected synthetic rhythm damage at
AUROC 1.000 and still failed end to end, because minimizing distance to aggregate
statistics does not produce better-reading text: blind fool-rate was 25.0% vs
25.0% on creative prose, and v1 beat v2 on short-form 87.5% to 37.5%. Its closing
recommendation is "measure human reader judgment rather than relying solely on
distance-to-author statistical metrics."

This module is that missing half. It does two things:

1. **Records** ranked judgments over candidate texts, blind where possible, with
   the author's own rationale kept verbatim.
2. **Turns them into a benchmark.** ``benchmark()`` takes any candidate scorer and
   measures how well it reproduces the author's ordering. That converts "does
   this feature help?" from an argument into a number, and it is the guard
   against the Goodhart failure the cadence work already hit: a feature can only
   earn its way into selection by predicting real verdicts.

Author-agnostic by construction. Nothing here knows whose voice it is; a profile
slug is the only key. That is what makes it usable in a shipped build where the
author's corpus is someone else's.

Storage is JSONL at ``profiles/<voice>/judgments.jsonl``, append-only, one record
per rated group. Append-only matters: a judgment is evidence about a moment, and
rewriting history would let a later opinion silently overwrite the data an earlier
experiment was validated against.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

SCHEMA_VERSION = 1


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RatedItem:
    """One candidate the author ranked.

    ``rank`` is 1-based, 1 = best. Ties share a rank, which is why pairwise
    comparison skips equal ranks rather than guessing an order.

    ``arm`` is whatever produced it ("opus_bare", "qwen_mimesis", "human").
    Kept so provenance survives, but never used in scoring: a benchmark that
    could see the arm label would be measuring recall of the experiment, not
    prediction of the judgment.
    """
    arm: str
    text: str
    rank: int
    grade: str | None = None
    rationale: str | None = None
    text_sha: str = ""

    def __post_init__(self):
        if not self.text_sha:
            self.text_sha = _sha(self.text)


@dataclass
class Judgment:
    """One rating session over a set of candidates for a single task."""
    task_id: str
    voice: str
    items: list[RatedItem]
    blind: bool = True
    task_prompt: str | None = None
    note: str | None = None
    ts: str = field(default_factory=_now)
    schema: int = SCHEMA_VERSION

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "Judgment":
        items = [RatedItem(**i) for i in d.get("items", [])]
        return Judgment(
            task_id=d["task_id"], voice=d["voice"], items=items,
            blind=d.get("blind", True), task_prompt=d.get("task_prompt"),
            note=d.get("note"), ts=d.get("ts", ""), schema=d.get("schema", 1))

    def pairs(self) -> list[tuple[RatedItem, RatedItem]]:
        """Ordered pairs (better, worse). Ties are dropped, not guessed."""
        out = []
        for a, b in itertools.combinations(self.items, 2):
            if a.rank == b.rank:
                continue
            out.append((a, b) if a.rank < b.rank else (b, a))
        return out


def store_path(profile) -> Path:
    return Path(profile.root) / "judgments.jsonl"


def record(profile, judgment: Judgment) -> Path:
    """Append one judgment. Creates the file if absent."""
    p = store_path(profile)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(judgment.to_json() + "\n")
    return p


def load(profile) -> list[Judgment]:
    p = store_path(profile)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Judgment.from_dict(json.loads(line)))
        except Exception:
            continue  # a malformed line must not sink the whole store
    return out


@dataclass
class BenchmarkResult:
    name: str
    pairs: int
    correct: int
    ties_skipped: int = 0
    errors: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.pairs if self.pairs else 0.0

    @property
    def kendall_tau(self) -> float:
        """Rank correlation over the pairs actually compared. -1..+1, 0 = chance."""
        return (2.0 * self.accuracy) - 1.0 if self.pairs else 0.0

    def render(self) -> str:
        if not self.pairs:
            return f"{self.name}: no comparable pairs"
        return (f"{self.name}: {self.correct}/{self.pairs} pairs "
                f"({self.accuracy*100:.0f}%), tau={self.kendall_tau:+.2f}"
                + (f", {self.errors} scoring errors" if self.errors else ""))


def benchmark(profile, scorer: Callable[[str], float], name: str = "scorer",
              lower_is_better: bool = True,
              judgments: Iterable[Judgment] | None = None) -> BenchmarkResult:
    """How often does ``scorer`` order two texts the way the author did?

    50% is chance. The existing fingerprint should be run through this first, so
    every later proposal has a real baseline to beat rather than an assumed one.
    """
    js = list(judgments) if judgments is not None else load(profile)
    correct = total = errors = ties = 0
    for j in js:
        ties += sum(1 for a, b in itertools.combinations(j.items, 2)
                    if a.rank == b.rank)
        for better, worse in j.pairs():
            try:
                sb, sw = scorer(better.text), scorer(worse.text)
            except Exception:
                errors += 1
                continue
            total += 1
            if (sb < sw) if lower_is_better else (sb > sw):
                correct += 1
    return BenchmarkResult(name=name, pairs=total, correct=correct,
                           ties_skipped=ties, errors=errors)


def compare(profile, scorers: dict[str, Callable[[str], float]],
            lower_is_better: dict[str, bool] | None = None) -> list[BenchmarkResult]:
    """Benchmark several scorers over the same judgments. Sorted best first."""
    lib = lower_is_better or {}
    out = [benchmark(profile, fn, name=n, lower_is_better=lib.get(n, True))
           for n, fn in scorers.items()]
    return sorted(out, key=lambda r: r.accuracy, reverse=True)


def summary(profile) -> str:
    js = load(profile)
    if not js:
        return "no judgments recorded"
    items = sum(len(j.items) for j in js)
    pairs = sum(len(j.pairs()) for j in js)
    blind = sum(1 for j in js if j.blind)
    arms: dict[str, int] = {}
    for j in js:
        for i in j.items:
            arms[i.arm] = arms.get(i.arm, 0) + 1
    arm_s = ", ".join(f"{k} x{v}" for k, v in sorted(arms.items(), key=lambda kv: -kv[1]))
    return (f"{len(js)} judgments ({blind} blind), {items} rated texts, "
            f"{pairs} comparable pairs\narms: {arm_s}")
