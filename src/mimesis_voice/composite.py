"""Composite voices: build one voice from several corpora with declared weights.

The problem this solves is a real one, measured in this repo. The research voice
has 11 pieces and 2,441 words total. Thirteen features cannot be estimated from
that: `mimesis doctor` reports VERY THIN, the feature standard deviations are
barely determined, and the p95 gate threshold derived from them is close to
noise. No algorithm fixes a corpus that small.

But the author has other corpora, and they are not unrelated. Academic prose and
personal essayistic prose share an idiolect: the same function-word habits, the
same clause rhythms, the same reach for a particular kind of qualification. What
differs is register and subject, not the hand. So a thin voice can borrow
*variance* from thicker neighbours while keeping its own *central tendency*.

That is what a composite does. Declare it in a profile's config.json:

    {
      "author_name": "Example Author",
      "compose_from": [
        {"profile": "research",  "weight": 3.0},
        {"profile": "personal",  "weight": 1.0},
        {"profile": "creative",  "weight": 0.5}
      ]
    }

## Weights are shares of influence, not per-piece multipliers

This is the part that has to be right or the feature is worse than useless.

Naively, weighting each *piece* by its profile's weight makes the blend a
function of corpus size: research at weight 3 with 11 pieces contributes 33
weight-units, personal at weight 1 with 41 pieces contributes 41, and the voice
you were trying to preserve is outvoted by the one you meant to borrow from.

So weights here are normalized per source: every piece from profile P carries
``w_P / n_P``. Profile P then contributes exactly ``w_P / sum(w)`` of the total
influence on every mean and standard deviation, whatever its piece count. In the
example above research holds 67% of the voice, personal 22%, creative 11% —
independent of how the corpora happen to be sized, today or after an ingest.

## What each layer does with the blend

* **Fingerprint** — weighted means and stds via ``fingerprint.calibrate_weighted``.
  The dominant source pulls the means; the others stabilise the stds, which is
  precisely what a thin corpus lacks.
* **Retrieval** — every source store is queried and the results interleaved by
  weight, so a composite gets anchors it would not otherwise have. Anchors carry
  their originating profile so the compose kit can label them.
* **Scrubber** — the banlist is the INTERSECTION of the sources' banlists and the
  whitelist is the UNION of their whitelists. A word the author demonstrably uses
  in any register is their word, and banning it in one voice because a small
  sample missed it is the failure mode to avoid. Numeric floors and ceilings are
  blended by the same normalized weights.
* **Presence floors** — weighted, same normalization.

## What a composite is not

It is not a way to write in someone else's voice, and it is not a way to fake a
corpus you do not have. A composite of one author's registers is a defensible
estimate of that author's idiolect. Mixing two different people would produce a
fingerprint belonging to nobody; ``validate`` refuses a composite whose sources
disagree on ``author_name`` unless ``allow_mixed_authors`` is set explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, ingest
from . import quality as quality_mod
from . import rhetoric as rhetoric_mod


@dataclass
class Source:
    slug: str
    weight: float
    profile: "config.Profile | None" = None
    texts: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.texts)


def declared(profile: config.Profile) -> list[dict]:
    """The raw ``compose_from`` list, or [] for an ordinary single-corpus voice."""
    return list(profile.compose_from or [])


def is_composite(profile: config.Profile) -> bool:
    return bool(declared(profile))


def validate(profile: config.Profile) -> list[str]:
    """Problems that would make a composite meaningless. Empty list = fine."""
    problems: list[str] = []
    decls = declared(profile)
    if not decls:
        return problems
    names = {profile.name}
    seen = set()
    for d in decls:
        slug = d.get("profile")
        if not slug:
            problems.append("a compose_from entry has no 'profile'")
            continue
        if slug == profile.slug:
            problems.append(f"'{slug}' lists itself in compose_from")
        if slug in seen:
            problems.append(f"'{slug}' appears twice in compose_from")
        seen.add(slug)
        try:
            src = config.resolve(slug)
        except Exception:
            problems.append(f"compose_from references unknown profile '{slug}'")
            continue
        if not src.db_path.exists():
            problems.append(f"'{slug}' has no store; run: mimesis ingest {slug}")
        names.add(src.name)
        if float(d.get("weight", 1.0)) <= 0:
            problems.append(f"'{slug}' has a non-positive weight")
    if len(names) > 1 and not profile.allow_mixed_authors:
        problems.append(
            "compose_from mixes different author_names "
            f"({', '.join(sorted(names))}). A fingerprint blended across two people "
            "describes neither. Set \"allow_mixed_authors\": true only if that is "
            "genuinely what you want."
        )
    return problems


def load_sources(profile: config.Profile, min_words: int = 120) -> list[Source]:
    """Resolve every source, read its corpus, and drop unusable pieces.

    Segmentation is applied per source, using each source's own shape, so a
    composite of book chapters and short posts does not inherit one corpus's
    length distribution as if it were style.
    """
    out: list[Source] = []
    n_feat = len(_features_for(profile))
    for d in declared(profile):
        slug = d["profile"]
        try:
            src = config.resolve(slug)
        except Exception:
            continue
        texts = [t for t in ingest.read_pieces(src.db_path).values() if t.strip()]
        if not texts:
            continue
        target = ingest.calibration_target(texts, n_feat)
        if ingest.needs_segmentation(texts) or len(texts) < 3 * n_feat:
            texts = ingest.segment_for_calibration(texts, target=target)
        texts = [t for t in texts if len(t.split()) >= min_words]
        if texts:
            out.append(Source(slug=slug, weight=float(d.get("weight", 1.0)),
                              profile=src, texts=texts))
    return out


def _features_for(profile: config.Profile) -> tuple:
    from . import fingerprint as fp_mod
    return fp_mod.FEATURE_SETS.get(profile.feature_set, fp_mod.FEATURES)


def weighted_corpus(sources: list[Source]) -> tuple[list[str], list[float]]:
    """Flatten sources into (texts, per-piece weights) with shares normalized.

    Each piece from source P carries ``w_P / n_P``, so P's total influence is
    ``w_P`` regardless of how many pieces it has. See the module docstring: doing
    this per-piece instead lets the largest corpus quietly win.
    """
    texts: list[str] = []
    weights: list[float] = []
    for s in sources:
        if s.n == 0:
            continue
        per_piece = s.weight / s.n
        for t in s.texts:
            texts.append(t)
            weights.append(per_piece)
    return texts, weights


def shares(sources: list[Source]) -> dict[str, float]:
    """Fraction of total influence held by each source. Sums to 1.0."""
    total = sum(s.weight for s in sources if s.n) or 1.0
    return {s.slug: s.weight / total for s in sources if s.n}


def retrieve_blended(
    query_text: str,
    limit: int,
    profile: config.Profile,
    exclude_files: set[str] | None = None,
) -> list[dict]:
    """Anchors drawn from every source, allocated by influence share.

    Each source is queried independently against its own store, then the slates
    are interleaved so the returned list is ordered best-first *within* a
    weight-proportional allocation. Every hit is tagged with ``_source`` so the
    compose kit can say which register an example came from -- a reader of the kit
    should be able to tell that an anchor is essayistic rather than academic.
    """
    from . import retrieve as retrieve_mod

    srcs = load_sources(profile)
    if not srcs:
        return []
    sh = shares(srcs)

    # Allocate slots by share, guaranteeing at least one to any source with a
    # non-trivial share so a small-but-deliberate contributor is never silent.
    alloc: dict[str, int] = {}
    for s in srcs:
        alloc[s.slug] = max(1 if sh.get(s.slug, 0) >= 0.05 else 0,
                            int(round(sh.get(s.slug, 0) * limit)))
    slates: dict[str, list[dict]] = {}
    for s in srcs:
        want = alloc.get(s.slug, 0)
        if want <= 0 or s.profile is None:
            continue
        try:
            hits = retrieve_mod.retrieve(
                query_text, want, s.profile, exclude_files=exclude_files
            )
        except Exception:
            hits = []
        for h in hits:
            h["_source"] = s.slug
        slates[s.slug] = hits

    # Interleave, strongest share first, so the dominant register leads the kit.
    order = sorted(slates, key=lambda k: sh.get(k, 0), reverse=True)
    merged: list[dict] = []
    i = 0
    while len(merged) < limit and any(len(slates[k]) > i for k in order):
        for k in order:
            if len(slates[k]) > i and len(merged) < limit:
                merged.append(slates[k][i])
        i += 1
    return merged


def blend_scrub(sources: list[Source], base_whitelist: list[str]):
    """Scrub calibration for a composite.

    Banlist is the INTERSECTION across sources and whitelist is the UNION.
    Rationale: the banlist is a list of words the author does NOT use, and that
    claim only survives if every corpus agrees. A word appearing in any register
    is evidence the author uses it, and a thin corpus failing to contain it is
    absence of evidence. Getting this backwards would have the composite ban
    vocabulary the author demonstrably writes.

    Numeric floors/ceilings are blended by normalized influence share.
    """
    from . import scrub as scrub_mod

    cals = []
    for s in sources:
        if s.n:
            cals.append((s, scrub_mod.calibrate(s.texts, whitelist=base_whitelist)))
    if not cals:
        raise ValueError("composite has no usable sources to calibrate the scrubber from")

    banned_sets = [set(c.banned_words) for _, c in cals]
    banned = set.intersection(*banned_sets) if banned_sets else set()
    white = set(base_whitelist or [])
    for _, c in cals:
        white |= set(c.whitelist)
    banned -= white

    sh = shares([s for s, _ in cals])
    def blend(attr: str) -> float:
        return sum(getattr(c, attr) * sh.get(s.slug, 0) for s, c in cals)

    # Rhetoric rates are calibrated over the POOLED text rather than blended from
    # per-source rates. A composite's whole claim is that these registers are one
    # idiolect, so the author's cleft and antithesis habits should be measured
    # across all of it. Blending per-source p95s would also have produced 0.00
    # here and silently disabled both checks for every composite voice.
    pooled = [t for s, _ in cals for t in s.texts]
    rhet = rhetoric_mod.calibrate(pooled)
    qual = quality_mod.calibrate(pooled)

    return scrub_mod.ScrubCalibration(
        banned_words=sorted(banned),
        banned_phrases=list(scrub_mod.AI_TELL_PHRASES),
        whitelist=sorted(white),
        burstiness_floor=blend("burstiness_floor"),
        hedge_ceiling=blend("hedge_ceiling"),
        mean_sentence_len=blend("mean_sentence_len"),
        n_pieces=sum(s.n for s, _ in cals),
        cleft_p95=rhet.cleft_p95,
        antithesis_p95=rhet.antithesis_p95,
        cleft_p25=rhet.cleft_p25,
        antithesis_p25=rhet.antithesis_p25,
        density_p25=qual.density_p25,
        specificity_p25=qual.specificity_p25,
    )


def describe(profile: config.Profile) -> str:
    """One-paragraph human summary of what a composite is actually made of."""
    srcs = load_sources(profile)
    if not srcs:
        return f"'{profile.slug}' declares compose_from but no source has a usable corpus."
    sh = shares(srcs)
    parts = [
        f"{s.slug} {sh.get(s.slug, 0):.0%} (weight {s.weight:g}, {s.n} units)"
        for s in sorted(srcs, key=lambda x: sh.get(x.slug, 0), reverse=True)
    ]
    total = sum(s.n for s in srcs)
    return (f"'{profile.slug}' is a composite of {len(srcs)} corpora, "
            f"{total} units total: " + "; ".join(parts))
