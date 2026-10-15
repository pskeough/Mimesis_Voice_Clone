"""Parity against the decompose-regenerate study's stored per-cell values.

The measures in `mimesis_voice.measures` are copies, not imports: the study repo
is a private research tree this package must run without. A copy drifts, and a
splitter or a stopword list that drifts silently rescales every threshold
calibrated from the study's tables. This test is what makes the copy safe.

It loads the study's cached per-cell values from
`eval_methodology/results/cache/*.json`, loads the same generations out of
`decompose_regenerate/runs/<run>/gens.jsonl`, recomputes each measure here, and
requires agreement to three decimals. Nothing is written into the study tree.

Skipped when the study tree is absent, which is every machine but the one it was
run on. The recorded result is in `docs/STRUCTURE_GATES.md`; re-run it there
before trusting a change to `measures/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mimesis_voice.measures import construction, cpd, lexdiv, otd, per, psd
from mimesis_voice.measures.textbase import norm, truncate_words

# The study tree is private and lives outside this repo. Point
# MIMESIS_STUDY_ROOT at it to run the parity check; without it the file skips.
STUDY = Path(os.environ.get("MIMESIS_STUDY_ROOT", "") or "/nonexistent-study-root")
CACHE = STUDY / "eval_methodology" / "results" / "cache"
RUNS = STUDY / "decompose_regenerate" / "runs"

pytestmark = pytest.mark.skipif(
    not (CACHE.is_dir() and RUNS.is_dir()),
    reason=f"study tree not present at {STUDY}",
)

N_CELLS = 10
PLACES = 3

# cache file -> (measure callable, keys to compare, word cap)
CASES = {
    "per_A0_cap600.json": (per.per, ["per", "per_n_sent"], 600),
    "otd_raw_A0.json": (None, [f"share_{c}" for c in otd.CLASSES] + ["n_sent"], None),
    "psd_raw_A0.json": (psd.profile, ["para_n", "para_mean_words", "para_mean_sents",
                                      "para_cv_words", "para_cv_sents", "first_sent_ratio"], None),
    "cpd_raw_A0.json": (cpd.profile, ["burst_an", "burst_b", "q90_q50", "q50_q10",
                                      "sent_cv", "n_sent"], None),
    "lexdiv_A0.json": (lexdiv.measure, ["hapax_w300", "ttr_w300", "mattr50_w300",
                                        "mtld_w300", "n_tok"], None),
}


def _gen_texts(run: str) -> dict[tuple[str, int], str]:
    """{(id, sample): raw text} for arm A0 of one run, normalised as the study
    normalises: punctuation fold then glyph fold."""
    path = RUNS / run / "gens.jsonl"
    out: dict[tuple[str, int], str] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            if g.get("arm") != "A0":
                continue
            out[(g["id"], int(g.get("sample", 0)))] = norm(g["raw"])
    return out


def _cells(name: str, n: int) -> list[dict]:
    """`n` cells spread across the cache rather than the first `n`, so a single
    run's quirks cannot carry the test."""
    data = json.loads((CACHE / name).read_text(encoding="utf-8"))
    cells = data["cells"]
    step = max(1, len(cells) // n)
    picked = [cells[i * step] for i in range(n) if i * step < len(cells)]
    assert len(picked) == n, f"{name}: only {len(picked)} cells available"
    return picked


@pytest.mark.parametrize("cache_name", sorted(CASES))
def test_parity_with_study_cache(cache_name):
    fn, keys, cap = CASES[cache_name]
    if cache_name.startswith("otd") and not otd.available():
        pytest.skip("spaCy en_core_web_sm not installed")
    fn = fn or otd.profile
    texts: dict[str, dict] = {}
    checked = 0
    for cell in _cells(cache_name, N_CELLS):
        run = cell["run"]
        if run not in texts:
            texts[run] = _gen_texts(run)
        text = texts[run].get((cell["id"], int(cell["sample"])))
        assert text is not None, f"{run}/{cell['id']}#{cell['sample']} not in gens.jsonl"
        got = fn(truncate_words(text, cap))
        for k in keys:
            if k not in cell:  # lexdiv returns {} under MIN_TOKENS
                continue
            assert k in got, f"{cache_name} {run}/{cell['id']}: missing {k}"
            assert round(float(got[k]), PLACES) == round(float(cell[k]), PLACES), (
                f"{cache_name} {run}/{cell['id']}#{cell['sample']} {k}: "
                f"got {got[k]!r}, study {cell[k]!r}"
            )
            checked += 1
    assert checked > 0, f"{cache_name}: nothing compared"


def test_parity_constructional_four_new_features():
    """The four features added to this package, against the study's z cache.

    Compared as rates rather than z-scores: the cache stores z against the
    study's own REF+TRAIN calibration, which this package does not have. The
    rate is the thing that was ported; the calibration is rebuilt per voice.
    """
    data = json.loads((CACHE / "constructional_A0.json").read_text(encoding="utf-8"))
    cal = data["calibration"]
    texts: dict[str, dict] = {}
    checked = 0
    cells = data["cells"]
    step = max(1, len(cells) // N_CELLS)
    for i in range(N_CELLS):
        cell = cells[i * step]
        run, author = cell["run"], cell["author"]
        if run not in texts:
            texts[run] = _gen_texts(run)
        text = texts[run][(cell["id"], int(cell["sample"]))]
        got = construction.rates(text)
        for f in construction.FEATURES:
            c = cal[author][f]
            want_rate = cell["z_" + f] * c["sd"] + c["mu"]
            assert abs(got[f] - want_rate) < 1e-6, (
                f"{run}/{cell['id']}#{cell['sample']} {f}: got {got[f]!r}, "
                f"study rate {want_rate!r}"
            )
            checked += 1
    assert checked == N_CELLS * len(construction.FEATURES)
