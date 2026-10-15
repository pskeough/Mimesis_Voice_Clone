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

### `comma-chain` — advisory, per-author calibrated

Comma-spliced clause chains. Named by the reference author in every voice profile
— *"the over use of commas, that I could, and I preferred, and I am"*, *"run on
commas and flowery vocab"*, *"really long overwrought commas"* — and undetectable
until 2026-09-03, because the fingerprint's `commas_per_sent` is a **mean**: six
clauses chained in one sentence and six clean sentences give the same number.

Distributional, not central-tendency. A sentence is split on `,`, `;` and
clause-level coordinators, each fragment is asked whether it is a finite clause
(explicit subject plus finite verb), and the measure is the longest run of
**consecutive** finite clauses, plus the rate of over-length sentences per 100.

Counting clauses rather than commas is the whole detector. The reference author's
longest comma sentence runs 18 comma-delimited fragments and scores a chain of
**1**, because those fragments are appositives and participials. His long
sentences accumulate phrases; the generated ones accumulate predications.

Per-author calibrated, and unlike the four hard flags it has to be: this is a
device he uses on purpose (a short list of first-person clauses, for
emphasis) at 0.09–0.18% of his sentences. Two numbers are stored, `chain_len` (his
p95 longest chain, floored at 4 and capped at 6) and `chain_p95` (his rate at that
length, floored at 4 per 100 sentences). A single number cannot do both jobs: a
length alone flags his own accumulations, and a rate alone has no length to count.

Ground truth. Two drafts in the judgment stores carry a blind rationale naming
comma overuse in his own words, and the detector fires on both — including the
sentence his complaint is a verbatim quotation of. Across all 40 generated arms it
fires on 5, all of them drafts he ranked at or near the bottom; the one human arm
in the same comparison has a longest chain of 2 and does not fire.

False positives against his own corpora, each voice on its own calibration:
creative **2.0%** of pieces, research **0.0%**, personal **2.6%**, against
**6.3%** of the generated text in `evals/` that has already passed the whole gate
pipeline.

`find_comma_chains()`, `comma_chain_rate()`, `longest_comma_chain()`

### The four rate constructions — reporting only, per-author calibrated

Added 2026-09-09 from the decompose-regenerate study, which measured all
twenty-nine detectors above as rates per 1,000 words against nine authors and
added these four of its own. They live in `measures/construction.py`, not in
`rhetoric.py`, because none of them is a span you can quote and delete: each is
a property of a sequence of sentences, so the report gives a signed z against
the author's own corpus and the sentences that produced it.

| construction | unit | what it is | the study's finding |
|---|---|---|---|
| `anaphora2` | per 100 sentence pairs | consecutive sentences opening on the same two words | the dominant machine tell at every model size, z +5 to +7 in the 4B bases. *"I remember the way she had..."* fifteen times in one cell |
| `copula_density` | per 100 words | is / was / are / were / be / been / being / am | above the author in every adapter at every size: the passage describes states where the author moves |
| `rhetorical_question` | per 100 sentences | sentences ending in a question mark | **under**-produced by every base and restored by the adapters. The only one of the four whose machine direction is down |
| `long_short` | per 100 sentence pairs | a sentence over 25 words followed by one of six or fewer | clears the replicate floor at ratio 2.0; **and see below** |

All four clear the study's replicate floor individually (ratio 2.0-4.8). The
aggregates over all thirty-three constructions do not once winsorised (1.3-1.4)
and are deliberately not implemented.

#### `long_short` is the mechanical form of a rule the author had already stated

Measured 2026-09-09 on ten fresh Sonnet drafts written in the reference author's
creative voice through the real kit. Every draft passed the fingerprint gate
(0 of 10 above threshold, RMS-z 0.73-1.34 against 1.57), and `long_short` fired
on **8 of 10**, at z +2.2 to +31.9 against an author mean of 0.93 per 100 pairs.
The same line touches **1%** of that author's own corpus.

What it is catching, from the worst draft — eleven beats in twenty-nine
sentences:

> ...and he knew her before he had finished admitting he was looking.
> **"Ten years, maybe more."**
>
> ...some year that had run into the ones beside it without leaving a mark.
> **"She had cut her hair."**
>
> ...the direction the bus would come from if it was coming.
> **"None of it was for him."**

That draft carried one scrub flag (`pointer-verdict`) and sat at RMS-z 1.27.
Neither the thirteen surface features nor any of the twenty-nine lexical
detectors could see the pattern, because there is no word to point at.

The author's own recorded rule, from 2026-08-11, months before this measure
existed: *"Say the thing once. No antithesis, no comma-spliced clause chains,
no punchy declarative then reversal, no aphorism closers."* `long_short` is that
rule with a number on it. `find_closing_flourish` catches one lexical shape of
it; this catches the rhythm regardless of wording.

**None of them gates.** The cut is a signed z of 3 in the machine direction,
which was chosen against this document's own rule rather than assumed. Four
candidate rules were scored on five of the study's authors, counting the share of
each author's **own held-out passages** the rule flags against the share of that
author's base-model cells it catches:

| rule | flags of the authors' own prose | catches of base-model cells |
|---|---|---|
| z ≥ 2 | 6-28% | 17-83% |
| **z ≥ 3** | **0-8%** | **13-73%** |
| above the author's corpus maximum | 0-4% | 6-56% |

z ≥ 2 breaks the 15% rule on three of the five authors. z ≥ 3 clears it on all
five. The full table, per author, is in `docs/STRUCTURE_GATES.md`.

A feature is skipped rather than scored when the draft is too short for it to
resolve — when one instance would by itself exceed the cut (a single
long-then-short beat in a five-sentence draft computes to 25 per 100 pairs), or,
for rhetorical questions, when the author's own rate would not predict even one.
Both floors come from the author's numbers, not from a constant.

`measures/construction.rates()`, `measures/report._constructions()`

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
populated from your corpus. So do `chain_len` and `chain_p95`; until you run it,
the comma-chain check falls back to module floors rather than to zero, and
`scripts/portability_audit.py` warns that it is standing on a constant instead of
on your voice. Check that one the same way as the rest, on pieces rather than
words:

```python
cal = rhetoric.calibrate(pieces)
flagged = sum(1 for t in pieces if rhetoric.analyze(t, cal).chain_over)
print(f"comma-chain  len={cal.chain_len} p95={cal.chain_p95:.1f}  "
      f"{100 * flagged / len(pieces):.1f}% of pieces")
```

Under about **5% of pieces** is the working ceiling; the reference author sits at
0.0–2.6%. Above it, either the clause test is wrong for your author's syntax or
the corpus needs the length raising, and `chain_len` is the knob — it is capped at
6 so the check cannot calibrate itself into silence.

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
