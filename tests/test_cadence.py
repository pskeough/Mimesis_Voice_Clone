"""Cadence features must see sentence ORDER, which the v1 features cannot."""
from __future__ import annotations

import random

from mimesis_voice import cadence, fingerprint


def _text(lengths: list[int]) -> str:
    """Build a text whose sentences have exactly the given word counts."""
    sents = [" ".join(["word"] * (n - 1) + ["end."]) for n in lengths]
    out, buf = [], []
    for i, s in enumerate(sents, 1):
        buf.append(s.capitalize())
        if i % 5 == 0:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


ALTERNATING = [4, 30, 4, 30, 4, 30, 4, 30, 4, 30, 4, 30]
RAMP = sorted(ALTERNATING)


def test_v1_is_blind_to_order():
    """The documented failure: identical v1 features for opposite rhythms."""
    a = fingerprint.extract_features(_text(ALTERNATING))
    b = fingerprint.extract_features(_text(RAMP))
    for f in fingerprint.FEATURES:
        assert abs(a[f] - b[f]) < 1e-9, f"{f} differed; v1 was expected to be blind"


def test_cadence_separates_order():
    a = cadence.extract_features(_text(ALTERNATING))
    b = cadence.extract_features(_text(RAMP))
    # Perfect alternation reverses direction every step; a ramp never does.
    assert a["sl_dir_changes"] > 0.9
    assert b["sl_dir_changes"] == 0.0
    # Alternation is strongly anti-correlated at lag 1; a ramp is positive.
    assert a["sl_autocorr1"] < -0.5
    assert b["sl_autocorr1"] > 0.5


def test_shuffle_changes_cadence_but_not_surface():
    rng = random.Random(7)
    lens = [rng.randint(3, 35) for _ in range(40)]
    shuffled = lens[:]
    rng.shuffle(shuffled)
    while shuffled == lens:  # pragma: no cover - vanishingly unlikely
        rng.shuffle(shuffled)
    s1 = fingerprint.extract_features(_text(lens))
    s2 = fingerprint.extract_features(_text(shuffled))
    for f in ("sent_len_mean", "sent_len_stdev", "pct_short_sents", "pct_long_sents"):
        assert abs(s1[f] - s2[f]) < 1e-9


def test_feature_sets_and_backcompat():
    assert len(fingerprint.FEATURE_SETS["v1"]) == 13
    assert len(fingerprint.FEATURE_SETS["v2"]) == 26
    # A fingerprint with no explicit feature_set must behave as v1.
    fp = fingerprint.Fingerprint(means={}, stds={})
    assert fp.feature_set == "v1"
    assert fp.cadence_distance("anything at all here") == 0.0


def test_every_cadence_feature_has_a_hint():
    for f in cadence.CADENCE_FEATURES:
        assert cadence.hint(f, 1.0)
        assert cadence.hint(f, -1.0)
        assert cadence.hint(f, 1.0) != cadence.hint(f, -1.0)


def test_z_clip_bounds_a_degenerate_feature():
    """One near-zero-variance feature must not dominate the aggregate.

    Mirrors the measured failure: emdash_per_100w had std 0.0117 on a corpus
    that barely uses them, and real prose averaged |z| = 12 on it.
    """
    means = {f: 0.0 for f in fingerprint.FEATURES}
    stds = {f: 1.0 for f in fingerprint.FEATURES}
    stds["emdash_per_100w"] = 1e-6
    fp = fingerprint.Fingerprint(means=means, stds=stds)
    _, zs = fp.distance_detail("Some text with an em-dash — right here, plus more words to pass the floor. " * 6)
    assert abs(zs["emdash_per_100w"]) <= fingerprint.Z_CLIP


def test_segmentation_only_fires_on_lopsided_corpora():
    from mimesis_voice import ingest

    even = ["word " * 800 for _ in range(6)]
    assert not ingest.needs_segmentation(even)
    assert ingest.segment_for_calibration(even) == even

    lopsided = ["word " * 800 for _ in range(5)] + ["para\n\n" * 1 + "word " * 20000]
    assert ingest.needs_segmentation(lopsided)


def test_segmentation_splits_long_pieces_at_paragraphs():
    from mimesis_voice import ingest

    para = " ".join(["word"] * 500)
    long_piece = "\n\n".join([para] * 10)  # ~5000 words
    out = ingest.segment_for_calibration([long_piece], target=1500)
    assert len(out) > 1
    # No segment may exceed the target by more than one paragraph's worth.
    assert all(len(s.split()) <= 1500 + 500 for s in out)
    # Short pieces pass through untouched.
    assert ingest.segment_for_calibration(["short text here"], target=1500) == ["short text here"]


def test_new_profiles_get_auto_gate_and_v2():
    from mimesis_voice import config

    assert config.NEW_PROFILE_GATE["rmsz_max"] == "auto"
    # v2 is a diagnostic, not a default gate: see config.create_profile.
    # The legacy default stays numeric so existing profiles keep their thresholds.
    assert config.DEFAULT_GATE["rmsz_max"] == 1.1


def test_quality_flags_are_absolute_or_banded_correctly():
    """Repetition and padding are defects for anyone; thinness is author-relative."""
    from mimesis_voice import quality as Q

    dup = ("The models overestimate depression in every benchmarked cohort by a wide "
           "margin. " * 2) + ("Separate content about anchors and survey weighting "
                              "appears here to pad the length past the floor. " * 3)
    r = Q.measure(dup)
    assert r.is_repetitive, "identical sentences must flag regardless of author"

    # Thinness needs a calibration; with none, it must not fire.
    assert not Q.measure(dup).is_thin
    cal = Q.QualityCalibration(density_p25=90.0, specificity_p25=0.9)
    assert Q.measure(dup, cal).is_thin


def test_verdict_orders_correctness_above_style():
    from mimesis_voice import verdict as V

    class R:  # minimal stand-in for a ScrubReport
        fidelity_added_numbers = ["9.99"]
        fidelity_dropped_citations = []
        banned_words = ["delve"]
        banned_phrases = []
        emdash_count = 3
        rhetoric = None
        fit_off = False
        presence_missing = False
        fit_drifting = False
        burstiness = 0
        burstiness_floor = 0

    v = V.build(R())
    assert v.worst == V.Tier.CORRECTNESS
    assert v.blocks()                      # invented number blocks by default
    assert v.at(V.Tier.CORRECTNESS)
    assert v.at(V.Tier.TICS)               # style findings survive, ranked below
    assert v.render().index("CORRECTNESS") < v.render().index("TICS")


def test_slate_spread_detects_collapse():
    from mimesis_voice import gate as G
    from mimesis_voice.fingerprint import calibrate

    # calibrate() needs >=5 pieces of >=120 words each.
    varied = [
        "Short lines here. " * 70,
        "A considerably longer sentence structure runs through this piece, with "
        "several clauses stacked before it closes. " * 12,
        "Mid length sentences appear throughout this one, evenly. " * 20,
    ]
    fp = calibrate(varied + [
        "Another piece with its own shape entirely, and its own rhythm. " * 16,
        "And one more, different again in phrasing and in pace. " * 18,
    ])
    same = [G.Candidate(text=varied[0], rmsz=0.0) for _ in range(4)]
    assert G.slate_spread(same, fp) == 0.0, "identical candidates have zero spread"
    diff = [G.Candidate(text=t, rmsz=0.0) for t in varied]
    assert G.slate_spread(diff, fp) > 0.0
