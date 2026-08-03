"""Composite voices: weights must mean shares of influence, not per-piece multipliers."""
from __future__ import annotations

from mimesis_voice import composite


def _src(slug, weight, n):
    return composite.Source(slug=slug, weight=weight,
                            texts=[f"{slug} piece {i} " * 60 for i in range(n)])


def test_weight_is_a_share_not_a_per_piece_multiplier():
    """The whole feature turns on this.

    A thin source with a high weight must not be outvoted by a fat source with a
    low weight. Naive per-piece weighting does exactly that: 11 pieces at weight 3
    is 33 weight-units against 41 pieces at weight 1.
    """
    thin = _src("research", 3.0, 11)
    fat = _src("creative", 0.5, 193)
    texts, weights = composite.weighted_corpus([thin, fat])
    assert len(texts) == len(weights) == 11 + 193

    total_thin = sum(w for t, w in zip(texts, weights) if t.startswith("research"))
    total_fat = sum(w for t, w in zip(texts, weights) if t.startswith("creative"))
    assert abs(total_thin - 3.0) < 1e-9
    assert abs(total_fat - 0.5) < 1e-9
    # The thin, heavily-weighted source dominates despite having 17x fewer pieces.
    assert total_thin > total_fat * 5


def test_shares_sum_to_one_and_ignore_piece_count():
    srcs = [_src("a", 3.0, 5), _src("b", 1.5, 500), _src("c", 0.5, 50)]
    sh = composite.shares(srcs)
    assert abs(sum(sh.values()) - 1.0) < 1e-9
    assert abs(sh["a"] - 3.0 / 5.0) < 1e-9
    assert abs(sh["b"] - 1.5 / 5.0) < 1e-9
    assert abs(sh["c"] - 0.5 / 5.0) < 1e-9


def test_empty_sources_are_ignored_not_divided_by_zero():
    srcs = [_src("a", 1.0, 3), composite.Source(slug="empty", weight=9.0, texts=[])]
    texts, weights = composite.weighted_corpus(srcs)
    assert len(texts) == 3
    assert "empty" not in composite.shares(srcs)


def test_blend_scrub_intersects_banlists_and_unions_whitelists():
    """A word the author uses in ANY register is theirs.

    Getting this backwards would have a composite ban vocabulary the author
    demonstrably writes, because one thin corpus happened not to contain it.
    """
    from mimesis_voice import scrub as scrub_mod

    # scrub.calibrate skips pieces with <3 sentences before counting vocabulary,
    # so the fixture needs real sentence punctuation or nothing is whitelisted.
    sent = "Ordinary words here for the counter. "
    a = composite.Source(slug="a", weight=1.0,
                         texts=[("The realm is a realm of realm and realm. " + sent * 20)
                                for _ in range(6)])
    b = composite.Source(slug="b", weight=1.0, texts=[(sent * 24) for _ in range(6)])
    cal = composite.blend_scrub([a, b], base_whitelist=[])
    assert "realm" in cal.whitelist
    assert "realm" not in cal.banned_words
    # A word neither source uses stays banned (intersection keeps it).
    assert "tapestry" in cal.banned_words
    assert isinstance(cal, scrub_mod.ScrubCalibration)


def test_ordinary_profiles_are_not_composites():
    from mimesis_voice import config

    p = config.resolve("example")
    assert not composite.is_composite(p)
    assert composite.declared(p) == []
