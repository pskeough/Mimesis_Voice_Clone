"""Structural and coherence measures adopted from the decompose-regenerate study.

Every measure here was validated on 2026-09-09 against 4,392 scored generation
cells across nine authors and four model sizes, then adopted into this package.
The validation protocol and the numbers each measure had to clear are in
`docs/STRUCTURE_GATES.md`; the study's own tables live in its private
companion repo (`eval_methodology/results/`).

What this layer adds that the rest of the package cannot see:

- `fingerprint.py` measures thirteen order-invariant surface aggregates. Shuffle
  a passage's sentences and it scores identically. Cadence, paragraph shape and
  opener classes are order- and architecture-sensitive, so they see the
  synopsis-mode prose the fingerprint passes.
- `rhetoric.py` and `rhetoric_ext.py` find named constructions. Neither can see
  a passage restating its own propositions in fresh wording, which is what the
  proposition-echo rate catches.

Nothing here is combined into a single score. Three coherence features spent
weeks hidden inside the study's averaged geometry index and the looping they
measured was invisible for all of it. Each measure reports on its own line.

Import cost: `otd` needs spaCy and the `en_core_web_sm` model; it reports
unavailable without them and every other measure works regardless.
"""
from __future__ import annotations

from . import calibration, construction, cpd, lexdiv, otd, per, psd, report, textbase
from .calibration import StructureCalibration, load_for

__all__ = [
    "StructureCalibration", "calibration", "construction", "cpd", "lexdiv",
    "load_for", "otd", "per", "psd", "report", "textbase",
]
