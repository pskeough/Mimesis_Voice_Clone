"""Tests for supervised preference capture.

The benchmark is the load-bearing piece: it decides whether a proposed feature
earns its way into selection. So it is tested against scorers whose behaviour is
known by construction (perfect, inverted, constant) rather than only on real data,
where a bug would look like a finding.
"""
from __future__ import annotations

import types

import pytest

from mimesis_voice import preference as pref


@pytest.fixture
def profile(tmp_path):
    return types.SimpleNamespace(root=str(tmp_path), slug="testvoice")


def _judgment(voice="testvoice", task="t1"):
    return pref.Judgment(
        task_id=task, voice=voice, blind=True,
        items=[
            pref.RatedItem(arm="best", text="aaa", rank=1, grade="A"),
            pref.RatedItem(arm="mid", text="bbbb", rank=2, grade="B"),
            pref.RatedItem(arm="worst", text="ccccc", rank=3, grade="F"),
        ])


def test_roundtrip(profile):
    pref.record(profile, _judgment())
    got = pref.load(profile)
    assert len(got) == 1
    assert [i.arm for i in got[0].items] == ["best", "mid", "worst"]
    assert got[0].items[0].grade == "A"


def test_sha_autofilled(profile):
    item = pref.RatedItem(arm="a", text="hello", rank=1)
    assert len(item.text_sha) == 12


def test_pairs_exclude_ties():
    j = pref.Judgment(task_id="t", voice="v", items=[
        pref.RatedItem(arm="a", text="x", rank=1),
        pref.RatedItem(arm="b", text="y", rank=1),   # tie
        pref.RatedItem(arm="c", text="z", rank=2),
    ])
    pairs = j.pairs()
    assert len(pairs) == 2                      # a>c and b>c, not a vs b
    assert all(better.rank < worse.rank for better, worse in pairs)


def test_benchmark_perfect_and_inverted(profile):
    pref.record(profile, _judgment())
    # length ascending matches the recorded ranking exactly
    perfect = pref.benchmark(profile, len, name="perfect")
    assert perfect.pairs == 3
    assert perfect.accuracy == 1.0
    assert perfect.kendall_tau == pytest.approx(1.0)

    inverted = pref.benchmark(profile, lambda t: -len(t), name="inverted")
    assert inverted.accuracy == 0.0
    assert inverted.kendall_tau == pytest.approx(-1.0)


def test_constant_scorer_is_chance_not_credit(profile):
    """A scorer with no signal must not score above zero.

    Guards the comparison direction: with lower_is_better a constant scorer
    never satisfies sb < sw, so it lands at 0.0 rather than being handed
    free wins on ties.
    """
    pref.record(profile, _judgment())
    flat = pref.benchmark(profile, lambda t: 1.0, name="flat")
    assert flat.accuracy == 0.0


def test_higher_is_better_direction(profile):
    pref.record(profile, _judgment())
    r = pref.benchmark(profile, lambda t: -len(t), name="hib", lower_is_better=False)
    assert r.accuracy == 1.0


def test_scoring_errors_are_counted_not_raised(profile):
    pref.record(profile, _judgment())

    def boom(_):
        raise ValueError("nope")

    r = pref.benchmark(profile, boom, name="boom")
    assert r.errors == 3
    assert r.pairs == 0
    assert "no comparable pairs" in r.render()


def test_malformed_line_does_not_sink_store(profile):
    pref.record(profile, _judgment())
    p = pref.store_path(profile)
    with p.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
    pref.record(profile, _judgment(task="t2"))
    assert len(pref.load(profile)) == 2


def test_load_missing_file_is_empty(profile):
    assert pref.load(profile) == []
    assert "no judgments" in pref.summary(profile)


def test_compare_sorts_best_first(profile):
    pref.record(profile, _judgment())
    res = pref.compare(profile, {"good": len, "bad": lambda t: -len(t)})
    assert res[0].name == "good"
    assert res[0].accuracy > res[-1].accuracy


def test_summary_counts(profile):
    pref.record(profile, _judgment())
    pref.record(profile, _judgment(task="t2"))
    s = pref.summary(profile)
    assert "2 judgments" in s
    assert "6 rated texts" in s
    assert "best x2" in s
