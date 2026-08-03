# Authorial presence

## Why this exists

The scrubber's other checks all ask the same question in different ways: does this prose *match a
distribution*? Em-dash count, banned vocabulary, sentence-length variance, fingerprint RMS-z. A draft
can satisfy every one of them and still read as machine-written, because what makes writing feel
authored is not its variance. It is that somebody in the text notices, expects, doubts, concedes,
and is occasionally wrong.

The case that prompted this: a 21,000-word research manuscript scored CLEAN by the fingerprint, which
its author read and immediately called machine-written. Measuring it against two peer-reviewed papers
in the same field:

| per 1,000 words | CoMPosT (EMNLP'23) | Marked Personas (ACL'23) | the manuscript |
|---|---|---|---|
| epistemic verbs (find / suspect / suggests) | 0.32 | 0.31 | **0.00** |
| expectation violation (surprised / we had expected) | 0.16 | 0.16 | **0.00** |

Zero of each, across the whole paper. Nobody ever found anything, expected anything, or was
surprised by anything. Every sentence was a verified assertion.

Notably the manuscript's sentence statistics were *better* than both human papers: mean 25.2 words
against 30.2 and 29.7, fewer long sentences, a quarter as many multi-semicolon sentences, identical
colon density. Sentence shape was never the problem. The fingerprint could not see the real problem
because assertion density is not one of its features, and no wordlist could see it because the tell
is an **absence**.

## What it measures

Five classes, counted per 1,000 words. They are tracked separately because they fail independently:
a draft can narrate its own method ("we ran", "we chose") while never conceding a doubt, and that
draft reads like a lab notebook rather than a paper.

| class | what it catches |
|---|---|
| `epistemic` | a first-person reading of evidence: *we find*, *we take it to mean*, *suggests*, *indicates* |
| `expectation` | results that ran against expectation: *surprised*, *we had expected*, *contrary to* |
| `stance` | claims held at a stated strength: *seems to*, *arguably*, *we cannot settle* |
| `signpost` | the author walking a reader through: *First*, *Regarding*, *In summary*, *Taken together* |
| `agency` | decisions someone made: *we ran*, *we chose*, *we dropped*, *we withdrew* |

## How it fires

Two rules, because absence and density need different amounts of text.

**Absence** (from 220 words). Total presence across all five classes below `ABSENCE_TOTAL = 1.5` per
1k. This is the headline flag, and it is deliberately keyed on the total rather than on any single
class. Presence can show up as interpretation, surprise, stance, organisation or agency, and a writer
need not use all five.

**Density floors** (from 1,500 words). Only `epistemic` and `expectation` are gated, against
`max(corpus 25th percentile, published-field floor)`.

Two thresholds that were wrong in earlier versions, kept here because the failures are instructive:

- A 400-word minimum meant the check never fired on the sections a scrubber actually receives. The
  known-flat draft passed because it was 371 words long.
- Gating any single class false-positived on a real EMNLP limitations section, which legitimately
  carries no expectation markers. Requiring `epistemic`, `expectation` and `stance` all to be zero
  still flagged **2 of 4** of the author's own pieces. The total rule flags **0 of 4** and still
  catches the flat draft, which scores 0.0 against their 3.9 to 25.0.

## The reference is not only the corpus

Every other calibration in this package derives from the author's own writing. Presence does not,
entirely. The author's framing: *"i dont care too much if it perfectly reflects my database, that
doesn't matter."*

A short-form corpus of technical reports will not contain the epistemic density of a full research
paper, so calibrating presence purely against it would enforce the flatness we are detecting. The
floor is `max(corpus p25, FIELD_FLOOR)`, so a thin corpus cannot drag the target below what real
published prose carries. Concretely, the research corpus medians are `epistemic 1.13`,
`signpost 7.76`, but `expectation 0.00` and `stance 0.00` — those two classes come entirely from the
field floor.

Corpus values may **raise** a field floor, never introduce a new gated class. The research corpus
signposts at 7.8 per 1k because its pieces are 200-word notes; carrying that over as a floor would
flag every long-form draft for insufficient *First* and *Finally*.

## The hedge fix that came with it

The scrubber's hedge list conflated two different things under one ceiling:

| | examples | verdict |
|---|---|---|
| softening | *sort of*, *kind of*, *somehow*, *in a sense* | weak writing, penalize |
| epistemic stance | *suggests*, *indicates*, *seems to*, *we suspect* | a thinking author, reward |

Because both were penalized together, a corpus that avoids the first produced a hedge ceiling of
**0.00**, which then flagged the second. In practice that instructed a writer to strip the exact
markers that make prose read as authored, which is what happened on the motivating manuscript.

`seems to` and `appears to` have moved out of `_HEDGE_PATTERNS` into `presence.STANCE_MARKERS`. They
are counted as presence and never against the hedge ceiling. `WEAK_HEDGES` keeps the old behaviour
for genuine mush.

## Files

- `src/mimesis_voice/presence.py` — the tier
- `src/mimesis_voice/scrub.py` — calls it; `ScrubReport.presence_missing` participates in `is_clean`
- `tests/test_presence.py` — the three real cases that broke earlier designs
- `profiles/<voice>/data/presence_calibration.json` — written by `mimesis calibrate`

Voices calibrated before this existed fall back to the published-field floors, so the check works on
every profile without recalibration.
