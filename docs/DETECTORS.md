# Rhetorical detectors

Every other check in this package works at the lexical or statistical level:
banned words, banned phrases, sentence-length variance, hedge density, a
13-feature stylometric fingerprint. None of them can see a construction built
entirely from vocabulary the author uses.

This is the layer that can. Each detector here was added after a specific draft
passed every other check and was still rejected on sight by the author it was
imitating.

If you are pointing this package at your own corpus, read
[Calibrating for a different author](#calibrating-for-a-different-author) before
trusting any of the hard flags. The rates below were measured on one author.

---

## Why this layer exists

The short version: surface features are the cheap tell, and removing them does
not make prose human.

Three findings from this repo's own evaluation drove it.

**A blocklist cannot see sentence shape.** Generated text was measured at surface
AUROC 0.691 against cadence AUROC 0.884. The features a word-level scrubber can
see were already nearly matched; what separated the prose was the part it could
not see.

**Optimising toward a corpus mean produces prose more average than the author.**
Measured surface AUROC 0.316 on an external author with the shipped engine.
Below 0.5 means the generations sit *closer to the author's centre* than the
author's own writing does. `gate.py` carries a `select: "band"` mode that aims at
the author's self-baseline instead; see the note there.

**The fingerprint alone tracked a human verdict no better than counting words.**
60% pairwise agreement with the author's blind ranking, identical to a word-count
control. That result is why `blind.py` exists.

A later blind round with four detectors in place scored 0.72 pairwise against the
author's ranking, with the fingerprint at 0.78 and a word-count control at 0.14 on
the same set. Both real signals sit far above the control. Small n, one judge, and
reported here because the earlier 0.60 result is also in this document.

---

## The detectors

Ordered by how much evidence stands behind each.

### `self-explanation` — hard

A clause arrives at a point, then instead of stopping appends a clause that
narrates what the point was. Named by the reference author in **all twelve drafts**
of a four-arm blind comparison, at equal strength in every arm. That it did not
vary with the generation strategy is the reason it is caught after the fact rather
than prompted away.

Two forms:

- **hinge** — `, which is` / `, which means` attached to a clause rather than a
  noun. *"Each attempt ends there, which means the phone has now asked me a
  question three times."*
- **is-what** — a reverse pseudo-cleft with a **nominalized** subject.
  *"Thresholding is what discards it."* The plain-noun form (*"rust was what
  stopped the engine"*) is ordinary emphatic English and is deliberately not
  matched.

Reference rate: 0.09–0.12 per 1000 words.
`find_self_explanations()`

### `nominalization-pickup` — advisory

A verb used in one clause returning as an abstract subject in the next.
*"...occurred and was witnessed and that the witnessing was mine."* Loosest
definition of the family and the highest reference rate (0.10–0.41/1kw), so it
reports rather than gates.
`find_nominalization_pickups()`

### `unheeded-reversal` — hard

A short flat assertion taken back and restated in the next breath.
*"This is not hope. Hope has weight, and this has none."*

Two shapes: a copular negation followed by a restatement sharing a content word,
and a short assertion followed by an assistant-specific corrective opener
(`What…`, `The problem is`, `It is that`, `In fact`).

The tuning history is the useful part. A first version matched any negation and
any `But`-opener and fired on **71–88%** of the reference author's own pieces,
because ordinary narrative negates constantly. Restricting to copular negation,
capping the setup sentence at 14 words, and dropping `But`/`Yet`/`Rather`/`Instead`
from the corrective set brought it to 0.10–0.19/1kw with all specimens still
firing.

`find_reversals()`

### `way-simile` — hard

`the way a X does`. *"...the way a machine refuses."* Confirmed independently by
two methods: a stylometric pass measured it at up to 3.14 per 1000 words in
generated text against a reference rate of 0.02 (roughly 157x), and the author
then flagged it unprompted in three separate blind packets without having seen
that measurement.

The lookahead exclusions carry the ordinary idioms — *the way back*, *that is the
way it is*, *the way he always walked*, *showed me the way* — which were the entire
false-positive surface.

Reference rate: 0.023/1kw. `find_way_similes()`

### `declarative-rating` — hard

A sentence that grades the previous one by pointing at it. *"That is the part that
sat with me."* *"That is a worse position than not knowing."*

`_SELF_RATING_RE` cannot see these because it requires an evaluative head from a
fixed list (*strongest*, *most X*, *the whole point*), and these rate by pointing
instead. Measured at exactly **0.000/1kw** across all three reference corpora,
which is why it gates on presence.

`find_declarative_ratings()`

### `it-is-opener` — advisory, per-author calibrated

`It is` / `It was` opening a sentence, as a transition. This one **is** calibrated
per profile, and it has to be: on the reference author the p95 is 5.45 for
creative prose, 4.55 for personal, and **0.00** for academic writing. A single
global threshold would either miss it entirely in one register or fire constantly
in another.

Stored as `it_is_p95` on the scrub calibration and populated by `calibrate()`.

### Earlier detectors

`find_triads`, `find_meta_asides`, `find_self_ratings`, `find_absence_punches`,
`find_closing_flourish`, `find_leakage`, plus rate-gated cleft and antithesis
checks. See the module docstring in `rhetoric.py` for the evidence behind each.

The closing-flourish check is worth singling out: three different voice profiles,
three different corpora, one closing move — an intensifier plus a deictic
(*"exactly where they look most convincing"*). Every lexical and statistical check
returned clean on all three. That is the clearest single piece of evidence in this
repo that stylometry misses rhetorical shape.

---

## Calibrating for a different author

**The hard detectors are not per-author calibrated.** Like the earlier three, they
encode habits of a *generating model*, not properties of a specific corpus, so
they apply the same way regardless of whose voice you target. That is the design
claim, and it is exactly the kind of claim that should be checked rather than
assumed.

Before you trust them on a new corpus, measure the false-positive rate:

```python
from mimesis_voice import config, ingest, rhetoric

profile = config.resolve_named("your-voice")
pieces = [t for t in ingest.read_pieces(profile.db_path).values() if t.strip()]
words = sum(len(t.split()) for t in pieces)

for name, fn in (("self-explanation", rhetoric.find_self_explanations),
                 ("reversal", rhetoric.find_reversals),
                 ("way-simile", rhetoric.find_way_similes),
                 ("declarative-rating", rhetoric.find_declarative_ratings)):
    hits = sum(len(fn(t)) for t in pieces)
    touched = sum(1 for t in pieces if fn(t))
    print(f"{name:20s} {1000 * hits / words:.3f}/1kw  {100 * touched / len(pieces):.0f}% of pieces")
```

Read the result like this. Under roughly **0.3 per 1000 words** and under **15% of
pieces**, the detector is measuring the generator rather than your author and the
hard gate is safe. Substantially above that, the construction is part of how your
author actually writes, and a hard flag will produce a check they learn to ignore
— which is worse than no check. Demote it to advisory or convert it to a rate gate
against the measured p95, the way the cleft and antithesis checks already work.

Two constructions in this repo were caught doing exactly that during development:
negation-parallelism and anaphoric intensification both read as AI tics until they
were measured, and both turned out to be deliberate devices of the reference
author. The triad detector was narrowed twice as a result.

`it_is_p95` needs no manual work — run `mimesis calibrate <voice>` and it is
populated from your corpus.

---

## Reporting

`scrub.render()` prints hard flags as `[HIGH]` with the offending span quoted and
a specific repair, not a category name. A flag that cannot tell you what to delete
is not actionable, and every detector here returns the span rather than the
sentence for that reason.

Hard flags gate. Advisory findings do not, and the distinction is deliberate: a
verifier that rejects everything is not a verifier, which is why
`tests/test_rhetoric.py` carries negative cases for every detector alongside the
positive ones.
