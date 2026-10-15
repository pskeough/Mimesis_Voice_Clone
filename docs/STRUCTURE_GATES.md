# Structural measures

Adopted 2026-09-09 from the decompose-regenerate study (a private companion
repo, `eval_methodology/`), which built and
validated them on 4,392 scored generation cells: nine authors, LoRA adapters
and prompting arms, base models at 1.7B, 4B, 8B and 14B. This file records what
was adopted, what was left behind, the thresholds and where each number came
from.

The reason the layer exists: every feature in `fingerprint.py` is an
order-invariant aggregate. Shuffle a passage's sentences and all thirteen score
identically. `evals/CADENCE_FINDINGS.md` already documents that blind spot; this
is the part that fills it. The five measures here are sensitive to order,
architecture and repetition, which is what separates prose that matches an
author's distribution from prose that reads like them.

The organising finding of the study, and the reason nothing here gates: **under
a fine-tune, voice and construction move in opposite directions.** Every voice
measure moved the adapter onto the author; every construction measure charged
for it. A single number over both is uninterpretable, and a gate built on either
would be optimising against the other.

---

## What was adopted

| line | measure | source module | direction |
|---|---|---|---|
| coherence | proposition-echo rate over 600-word windows | `measures/per.py` | one-sided, above |
| openers | opener-class Jensen-Shannon distance | `measures/otd.py` | one-sided, above |
| paragraphs | coefficient of variation of paragraph length | `measures/psd.py` | two-sided band |
| cadence | finite-size-corrected burstiness A_n | `measures/cpd.py` | two-sided band |
| vocabulary | hapax proportion over a 300-token window | `measures/lexdiv.py` | two-sided band |
| constructions | four constructions, signed z | `measures/construction.py` | per feature |

Nothing here **gates**. No candidate is rejected on a structural finding and no
winner is chosen on one, because voice and construction move in opposite
directions under optimisation and a gate on both optimises against one.
`scrub.py`'s hard flags are unchanged, and `gate_candidates` prints which lines
fired per candidate without scoring them into the ranking or the Pareto front.

What the findings **do** feed is the rewrite that was going to happen anyway.
`gate.compose` hands `measures.report.repair_notes` to the fingerprint rewrite
prompt, and runs one bounded structural repair pass afterwards when nothing
earlier has already rewritten the draft. Both are guarded: a repair is kept only
if it removed findings, did not push the fingerprint out of the gate, did not add
a hard flag, and did not break source fidelity. Structure loses every tie,
because the author's rulings outrank it.

### Does feeding the findings to a rewrite actually help? Not proven.

Measured 2026-09-09 on ten fresh Sonnet drafts written in the reference author's
creative voice through the real kit, all ten carrying at least one structural
finding (seventeen findings in total). Four rewrite arms, same kit, same model,
same drafts:

| arm | findings left | RMS-z | hard flags | cleared | fixed the named line | new findings introduced |
|---|---:|---:|---:|---:|---:|---:|
| *before* | 1.50 | 1.104 | 1.10 | — | — | — |
| A status quo (fingerprint features only) | 0.80 | 0.909 | 0.30 | 3/10 | 10/17 | 2 |
| B measured notes (**shipped**) | 0.80 | 0.770 | 0.30 | 4/10 | 12/17 | 3 |
| C generic structural advice | 0.40 | 0.744 | 0.60 | 6/10 | 12/17 | 0 |
| D evidence + full coverage | 0.50 | 0.882 | 0.90 | 6/10 | 14/17 | 3 |

**The honest reading. The rewrite is the win, not the instructions.** Every arm
cut the fingerprint distance by roughly a fifth and hard flags by two thirds or
more. Between arms, differences of two or three out of seventeen at n=10 are
noise; separating a twenty-point difference in these rates would need on the
order of a hundred drafts per arm.

**The measured evidence did not beat generic advice on structure.** B and C fix
the same 12 of 17. C — four generic instructions carrying no measurement at all —
cleared more drafts, and did it by introducing zero new findings where B
introduced three. C was built as a placebo and turned out not to be one: it names
all four structural axes every time, where B names only the ones that fired, so it
is a broader intervention rather than a vaguer one. That is a flaw in the design,
not a result.

**Adding coverage buys structure and costs the scrubber.** D, which pairs the
measured evidence with a list of the axes to preserve, fixed the most named lines
(14/17) and finished with three times B's hard-flag count. The pattern across
A → B → C → D is that structural instruction and scrub cleanliness trade against
each other, which is the same voice-versus-construction tension the study found
in the adapters, reappearing inside a single prompt.

B ships because it is the arm that improves structure without costing anything
the author's own rulings care about: joint-best hard flags, second-best
fingerprint. It is the conservative choice, not the demonstrated winner.

**What this does and does not license.** The measures' demonstrated value is in
*detection*: 18% of machine cells that the shipped gate passes clean, and
`long_short` firing on 8 of 10 real drafts against 1% of the author's corpus.
For repair, see the corrected experiment below.

### The corrected experiment, n=40 paired (2026-09-09)

Coverage held constant. Both arms name all six structural axes, in the same
order, at the same length; they differ only in whether the axes that actually
failed carry their evidence. Ninety-one fresh Opus drafts in the creative voice
through the real kit, every one carrying a finding; forty completed both arms
before the CLI's concurrency limits made further calls unproductive.

| axis | n flagged | G generic fixed | M measured fixed | M-only | G-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| openers | 20 | 11 | **19** | 9 | 1 | **0.021** |
| rhythm (`long_short`) | 35 | 5 | 3 | 1 | 3 | 0.625 |
| echo | 3 | 3 | 3 | 0 | 0 | 1.000 |
| paragraphs | 1 | 1 | 0 | 0 | 1 | 1.000 |
| **total** | **59** | **20** | **25** | 10 | 5 | 0.302 |

**Where the measurement works: openers.** 19 of 20 against 11 of 20, paired
p = 0.021. It is the one axis whose failure is a countable distribution, and
naming the count is what a model can act on. Generic advice can make it *worse*:
on one draft, "vary how sentences begin" took noun-phrase openers from 49% to
**59%**, while "49% of sentences open on a noun phrase, against the author's 25%"
took it to 43%.

**Where nothing works: the rhythm tell.** `long_short` is 35 of the 59 findings
and both arms fail it — 5 of 35 and 3 of 35. It is not that the instruction is
ignored: the mean rate falls from 24 to 12 (generic) and 10 (measured) on the
worst draft. It is that the author's own rate is 0.9 per 100 pairs and the
threshold is 4.5, so halving 24 is still eleven times him. This is a ceiling, not
a prompt problem, and the next thing to try is the kit rather than the repair:
prevent the beat while writing instead of asking for it back afterwards.

**Overall, the two arms are indistinguishable.** Across all 40 pairs the measured
arm is closer to the author on opener distance in 22 of 40 and on the rhythm rate
in 19 of 40 — coin flips. It fixed five more findings and introduced four more
(5 against 1), so net structural improvement is **+20 for measured against +19
for generic**. Hard flags came out marginally worse under the measured arm
(self-explanation 33 against 29).

**The reading.** The measurement earns its place on one axis of six and on 20 of
59 findings, and pays for it with a few more introduced findings elsewhere. That
is enough to keep the wiring — it is free, and openers is a real, significant win
— and not enough to claim the block improves generation broadly. The dominant
tell is not repairable by prompting at all.

Still missing: the author's own read. The measures scoring the arms is circular,
since they wrote the instructions. A blind sheet of twelve pairs was built for
that (`evals/blind_structure_2026-09-09/`) and has not been ruled on.

### Two mechanisms already in this repo are switched off, and both address these failures

Found 2026-09-09 while asking whether the corpus statistics reach the model
*before* drafting. They do — `_rules_block` ships mean sentence length, the
burstiness floor, the hedge ceiling, the banlist and the whitelist in every kit.
Two of the most relevant do not.

**1. The opener rule never fires on any of this repo's voices.** `_rules_block`
gates it on `n_opener_pieces >= 10 and opener_fronted_p50 >= 0.60`. Every stored
scrub calibration has those fields at **0.0** — written before they were
populated. Recomputed from the same corpus, `creative` gives
`n_opener_pieces=220, opener_fronted_p50=0.648`, which clears the gate. The rule
carries its own A/B in the source comment (arm A 39% fronted → arm B 58%,
fingerprint 1.05 → 1.00) and has not run since. This is the likeliest reason
openers flagged on half the drafts in the experiment above, and why naming the
opener count at *repair* time helped so much: it was the first time the model
heard about openers at all. **`mimesis calibrate <voice>` fixes it.** Not run
here, because it also rewrites the fingerprint and that is the author's call.

**2. No voice scores the cadence features, and they are what catches the tell.**
`cadence.py` has held `sl_short_after_long` — *"P(short | previous long), the
punch after a build"* — with a written rewrite hint since before this work.
It lives in feature set **v2**, and `creative`, `research` and `personal` are all
on **v1**, so it has never been scored. Building a v2 fingerprint from the same
corpus and scoring the 91 generated drafts:

| | v1 (shipped) | v2 (dormant) |
|---|---:|---:|
| mean distance | 1.160 | 1.724 |
| drafts over the voice's own p95 gate | **1 / 91** | **79 / 91** |

`sl_short_after_long` sits above the author on **91 of 91** drafts, mean z +2.79.
The worst two features are `sl_delta_abs` (3.94) and `sl_delta_sd` (3.87) —
sentence-to-sentence length jumps, which is the long/short alternation seen from
another angle.

Both gates are calibrated the same way, as the leave-one-out p95 of the author's
own pieces, so 5% of his real writing exceeds either by construction. A gate that
5% of his writing trips and 87% of generated drafts trip is separating; the v1
number is not evidence the drafts are fine, it is evidence v1 cannot see this.

That is a larger lever than anything in this document. The structural measures
adopted here were built because the fingerprint is order-invariant — and this
repo already had thirteen order-aware features sitting unused behind a config
key.

### The repair notes are one-sided on purpose

The report is two-sided because it is a diagnosis: paragraphs above the author's
band are as unlike them as paragraphs below. A repair prompt is not a diagnosis,
and the two sides do not cost the same. The study's reader preferred the cell
that over-varied its paragraphs (CV 0.79 against the source's 0.34) and rejected
the ones that under-varied them, so a prompt telling a model to even its
paragraphs out would optimise straight toward the failure the measure exists to
catch. `repair_notes` passes on only the machine direction — restatement,
opener monotony, one-size paragraphs, metronomic sentences, narrowed vocabulary,
over-produced constructions — and `tests/test_measures.py` asserts the over side
never reaches a prompt.

Each note names the evidence rather than the number, because a model can act on
*"eleven sentences in a row open on a pronoun"* and cannot act on 0.502.

### What was deliberately not ported

- **Lag-1 and lag-3 autocorrelation of sentence length.** Adapter effects 0.7x
  and 0.5x their own replicate floors, which is inside the noise.
- **The constructional aggregates** (`rfpd`, `arti`, mean |z| and mean positive
  z over all 33 constructions). Winsorised at |z| = 5 their ratio falls to
  1.3-1.4. Unwinsorised the ratio of 2.4 is carried entirely by extreme cells,
  which is a list of extreme cells with the labels removed, not a gate.
- **The judge rubric's craft block.** Both judges ranked the 14B bare model
  above the private author's own prose and above the cell he chose blind. As a
  construction instrument the rubric passes; as a voice or quality criterion it
  fails on exactly the case this package exists for.
- **Detector-tell dissociation.** The study's own prediction failed: adapters
  did not dissociate more than bases (+0.34 vs +0.32, +0.33 vs +0.40, +0.29 vs
  +0.10).
- **Calibrated embedding anchors.** Only relevant if this package embeds for
  style, which it does not by default.
- **Event order and terminal event.** Needs a brief carrying numbered events.
  Mimesis task specs do not supply one, and the keyword matcher's ceiling on the
  authors' own passages is only tau 0.75. Left for the day a task spec does.

---

## Parity with the study

The measures are **copies, not imports**. The study repo is a private research
tree this package must run without, and an import would make every Mimesis
install depend on it. A copy drifts, so `tests/test_measure_parity.py` loads the
study's stored per-cell values and requires agreement to three decimals.

Run on 2026-09-09 against `eval_methodology/results/cache/`:

| cache | cells | values compared | mismatches |
|---|---:|---:|---:|
| `per_A0_cap600.json` | 300 | 600 | 0 |
| `otd_raw_A0.json` | 300 | 3,300 | 0 |
| `psd_raw_A0.json` | 300 | 1,800 | 0 |
| `cpd_raw_A0.json` | 300 | 1,800 | 0 |
| `lexdiv_A0.json` | 300 | 1,490 | 0 |
| `constructional_A0.json` (four new features) | 300 | 1,200 | 0 |

10,190 values, zero mismatches. The committed test checks ten cells per measure
so it runs in seconds; the 300-cell sweep is the one-off recorded here. It is
skipped automatically when the study tree is absent, which is every machine but
the one it was built on.

`textbase.py` carries verbatim copies of `dr/core.py`'s `sentences`,
`paragraphs` and `normalise_punct` and of `methods/common.py`'s `fold_glyphs`.
Two properties matter:

- `normalise_punct` folds `--` into an em dash. That is a *measurement* fold and
  must never reach `scrub.py`, where the em dash is a hard flag — normalising
  first would manufacture the flag.
- `paragraphs` splits on blank lines only. A draft written with single newlines
  reads as one paragraph and the paragraph line reports "too few to rate"
  rather than treating it as an author writing one block.

---

## Thresholds, and where each came from

Every threshold is measured from the author's own corpus at
`mimesis calibrate` time and stored in
`profiles/<voice>/data/structure_calibration.json`.

The corpus read is the author's **source documents**, not the store. That is not
a detail: `ingest.read_pieces` reassembles a document out of ~180-word retrieval
chunks joined by blank lines, so every "paragraph" in a store text is one chunk.
Calibrated that way this repo's creative voice reported an author paragraph CV of
**0.06**; read from the documents it is a median of **0.58**, against the study's
nine authors at 0.62-0.95. The store was measuring the chunker.

A second splitting problem sat under that one. `dr/core.py`'s `paragraphs`
splits on blank lines only, and 33 of the first 40 documents in that same corpus
are a single blank-line block containing forty-odd newline-separated paragraphs.
`textbase.prose_paragraphs` falls back to newline splitting when blank lines
yield one block and the blocks are long enough not to be hard wrapping at a fixed
column. The author's corpus and the draft go through the same function, and the
fallback never fires on the study's own data (0 of 150 passages), so parity is
untouched.

### Length is the standing confound, and matching is the fix

A 250-word draft has fifteen sentences; a 1,200-word calibration unit has
seventy. The opener distribution of fifteen sentences sits further from any
reference than the distribution of seventy, for reasons that have nothing to do
with who wrote it. The same holds for a coefficient of variation over four
paragraphs against one over fifteen.

So the calibration stores the author's own **series** — the opener class of
every sentence, the word length of every paragraph and of every sentence — and
the threshold is computed at scoring time from the author's own units truncated
to the draft's own sentence or paragraph count. Every reference statistic is
leave-one-out where the unit contributes to it; without that a unit is compared
to a reference it helped build and the self-distances come out too small.

The cost of matching is real and is reported in the line: at a high sentence
count only the author's longer pieces qualify, so the line prints how many
matched pieces the band came from, and says "no matched calibration at that
length" rather than guessing when too few do.

| line | rule | why |
|---|---|---|
| coherence | `max(mu + 3 sd, 0.08)` over the author's own 600-word windows | the study's rule; 0.08 is an absolute floor because authors run 0.01-0.04 |
| openers | `mu + 2 sd` of the author's leave-one-out distances at the draft's sentence count | distance has no informative low end |
| paragraphs | `mu ± 2 sd` at the draft's paragraph count | two-sided: under-varied is the machine failure, over-varied is architecture |
| cadence | `mu ± 2 sd` at the draft's sentence count | as above; A_n is n-independent by construction |
| vocabulary | `mu ± 2 sd` on hapax over a 300-token window | vocabulary above the author is as unlike them as below |
| constructions | per feature, signed z ≥ 3 in the machine direction | chosen by measurement, see below |

### The one change from the study's implementation: windowed echo

The study took the first 600 words and stopped. That is right for ranking
thousands of cells against each other — uncapped, the echo rate correlates +0.41
with word count and up to +0.57 on the 1.7B runs — and wrong for gating one
draft, because a model that loops does it late and the truncation discards the
half of the draft where the looping is.

On the blind item, the candidate Patrick said lost the story scores:

| | whole draft | first 600 words | worst 600-word window |
|---|---:|---:|---:|
| B, the story-losing candidate | 0.167 | **0.000** | **0.200** |

(The study's `GATES.md` prints 0.167 under both `per_A0` and `per_A0_cap600`
because its item-1 block recomputes the four candidates without applying the
cap. The 0.000 above is what the capped measure actually returns on B, which is
how the gap went unnoticed there.)

So the draft is scored in 600-word windows at a 300-word stride and the worst
window is reported. Every window is exactly the object the study validated — a
passage of at most 600 words scored by `per` — and the overlap catches an echo
straddling a boundary. Echoes spanning more than 600 words are missed, which
under-reports rather than over-reports. The author's threshold is calibrated
over their own windows, so a long draft's extra windows are extra draws from a
distribution measured the same way, and the self-flag audit pays the same
multiplicity by scoring each calibration unit on its own worst window.

### The construction cut was selected, not assumed

Four candidate rules, scored on five of the study's authors: the share of the
**author's own held-out passages** each rule flags, against the share of that
author's base-model cells it catches. 50 held-out passages and 60-180 machine
cells per author, 2026-09-09.

| author | cal units | z ≥ 2 (FP/catch) | z ≥ 3 (FP/catch) | > corpus max (FP/catch) |
|---|---:|---|---|---|
| patrick_v2 | 150 | 22% / 67% | **2% / 54%** | 2% / 31% |
| chesterton | 318 | 16% / 17% | **6% / 13%** | 2% / 6% |
| mansfield | 151 | 6% / 60% | **0% / 50%** | 0% / 40% |
| wharton | 234 | 12% / 83% | **0% / 73%** | 0% / 51% |
| chopin | 97 | 28% / 74% | **8% / 60%** | 4% / 56% |

A rule at the author's own 99th percentile was also scored on patrick_v2: 8%
false positives for the same 54% catch rate as z ≥ 3, so it is dominated.

`DETECTORS.md`'s rule is that a check touching more than 15% of an author's own
pieces is measuring the author. **z ≥ 2 breaks that on three of the five.**
z ≥ 3 clears it on all five and gives up 10-13 points of detection to do so; the
corpus-maximum rule buys back no false positives and loses far more catch. So
`Z_FLAG = 3.0`.

Chesterton is the weak case at 13% catch: his base model does not over-produce
these four constructions much. That is a limit of the four features on that
author, not a reason to loosen the cut.

---

## Agreement with the author's blind ruling

The blind item Patrick ruled on, four candidates, key withheld until he ruled:
**D best; B lost the story; A and C regressive** (A = 1.7B base, B = 1.7B
adapter, C = 4B base, D = 4B adapter). Scored with a calibration built by this
package from that author's own REF+TRAIN pieces:

| | coherence | openers | lines flagged |
|---|---|---|---|
| the author's own passage | 0.000, under 0.082 | 0.399, under 0.496 | — |
| A (1.7B base) | 0.000 | **0.502 over 0.464**, PRON at 47% vs his 26% | openers, paragraphs, vocabulary |
| B (1.7B adapter) | **0.200 over 0.082**, worst window from word 900 | 0.281, under 0.501 (first 57 sentences) | coherence, vocabulary |
| C (4B base) | 0.077 | **0.559 over 0.440**, PRON at 58% vs his 26% | openers, vocabulary |
| D (4B adapter) | 0.000 | 0.391, under 0.461 | cadence |

B is the only coherence flag. A and C are the only opener flags. D is clean on
both, and the author's own passage is clean on all six lines. That is the
required result. D's cadence line fires at A_n -0.48 against a band of
[-0.47, -0.16] built from the 29 pieces long enough to match its 48 sentences;
the miss is 0.01 and the author's own passage in the same item sits at -0.54, so
that line is at the edge of what 29 matched pieces can resolve.

The burstiness values reproduce the study's item-1 row exactly (source -0.536,
A -0.464, B -0.263, C -0.507, D -0.476), as do the opener class shares.

Scored instead against this repo's `creative` voice — a different calibration,
the same four candidates — `gate_candidates` puts the structure flags on
candidates 1, 2 and 3 (A, B, C) and leaves candidate 4 (D) clean, and the
existing fingerprint gate independently picks D as the winner. The structure
column does not enter that ranking.

---

## False positives on the authors' own writing

`DETECTORS.md`'s rule: a check that touches more than 15% of an author's own
pieces is measuring the author. Two audits, both 2026-09-09.

**Five of the study's authors.** Calibrated on each author's REF+TRAIN, scored on
their 50 held-out passages, and on that author's base-model cells for contrast.

| author | | echo | openers | paragraphs | cadence | vocabulary | constructions |
|---|---|---:|---:|---:|---:|---:|---:|
| patrick_v2 | own prose | 2% | 2% | 2% | 10% | 8% | 2% |
| | base cells | 48% | 66% | 47% | 17% | 47% | 54% |
| chesterton | own prose | 2% | 2% | 4% | 6% | 4% | 6% |
| | base cells | 13% | 44% | 27% | 18% | 10% | 13% |
| mansfield | own prose | 0% | 4% | 6% | 2% | 12% | 0% |
| | base cells | 21% | 82% | 4% | 55% | 12% | 50% |
| wharton | own prose | 4% | 4% | 6% | 4% | 4% | 0% |
| | base cells | 25% | 35% | 24% | 51% | 42% | 73% |
| chopin | own prose | 2% | 0% | 4% | 6% | 4% | 8% |
| | base cells | 29% | 37% | 13% | 46% | 38% | 60% |

Every line on every author is under the bar on their own prose, and every line
separates on every author.

**This repo's own voices**, five-fold: calibrate on 80% of the voice's source
documents, score the other 20%, so every document is held out once.

| voice | units | echo | openers | paragraphs | cadence | vocabulary | constructions |
|---|---:|---:|---:|---:|---:|---:|---:|
| creative | 221 | 2% | 3% | 7% | 13% | 6% | 3% |
| research | 31 | 3% | 3% | 6% | 6% | 6% | 6% |
| personal | 42 | 0% | 5% | 2% | 10% | 2% | 10% |

Cadence is the loosest line at 13% on the creative voice, still under the bar and
the one to watch if a voice is added whose pieces vary in form.

Each voice's own numbers are recomputed at calibration time and stored in its
`structure_calibration.json`; when a line fires, it prints its rate for that
voice, so the reader can see how often it is wrong before acting on it.

### Two known traps

**Text that came out of the store carries the chunker's paragraphing.**
`ingest.read_pieces` rejoins ~180-word retrieval chunks with blank lines. Any
such text — including `accepted.jsonl` entries seeded with `source: corpus:*` —
has paragraphs of near-identical length and the paragraph line will report
"paragraphs are all one size" about the chunker. Four of the eight entries in
`creative-recal`'s accepted set are exactly this. Calibration reads source
documents to avoid it; a draft pasted from the store cannot be protected and the
paragraph line should be ignored for it.

**Composites have no documents of their own.** A composite voice draws on other
profiles, so its structure calibration reads *their* documents, sampled in
proportion to the declared shares. Without the proportional sample the biggest
corpus wins: `research-blend` declares research 60% / personal 30% / creative
10%, and creative alone supplies four times the units of the other two together.

### The measure is self-correcting at short lengths

The opener band widens as the compared stretch shortens, which is what a matched
threshold is supposed to do. Measured on the creative voice, 203 units:

| sentences compared | threshold | flags of his own units |
|---:|---:|---:|
| 5 | 0.739 | 2% |
| 10 | 0.631 | 3% |
| 20 | 0.520 | 3% |
| 40 | 0.442 | 4% |
| 60 | 0.406 | 3% |

So no arbitrary minimum length is imposed on the opener line: the band handles
it, and the false-positive rate is flat across the range. The one floor that IS
imposed is on the construction line, and it too is derived from the author's own
rates rather than typed in — a feature is skipped when a single instance would
by itself exceed the cut, or, for rhetorical questions, when the author's own
rate would not predict even one instance in a draft that long.

### When the corpus has no pieces as long as the draft

The band is matched to the draft's sentence or paragraph count, and a draft
longer than most of the corpus has nothing to match. Rather than report nothing,
both sides are cut to the longest length at least twelve of the author's units
can match, and the line says so: *"13 matched pieces, first 57 sentences"*. On
the repo's own voices this took the not-measurable rate for openers from 16% to
8% and for cadence from 15% to 7%, with the flag rates unchanged.

---

## Running it

```bash
mimesis calibrate <voice> --structure-only
```

Rebuilds only the structural bands, leaving the fingerprint, scrubber and
presence floors untouched, so an existing voice gains the block without any
other threshold moving under it. A full `mimesis calibrate` builds it too.

The opener line needs spaCy and `en_core_web_sm`:

```bash
pip install "mimesis-voice[structure]" && python -m spacy download en_core_web_sm
```

Without them the opener line reports unavailable and every other line works.
A cheaper word-list classifier was considered and rejected: the tagger reads the
first six words of a sentence, not the first token, and the class turns on what
follows — *"Left alone, she counted the tiles"* is PART where *"Left hands are
rarer"* is NP, *"Past the church"* is PP where *"Past mistakes"* is NP. A
substitute classifier would measure something the study never validated.

The taxonomy is not always right and is kept wrong on purpose: a gerund subject
("Walking is good for the knees") tags VBG and lands in PART. The distance is to
the author's own distribution under the same classifier, so a consistent mistake
on both sides cancels, and changing it would break parity with the tables above.

### Where the code is

```
src/mimesis_voice/measures/
  textbase.py     splitters and normalisation, copied verbatim from the study
  per.py          proposition-echo rate, windowed
  otd.py          opener-class distance (spaCy)
  psd.py          paragraph shape
  cpd.py          corrected burstiness
  lexdiv.py       windowed hapax / MATTR / MTLD
  construction.py the four new constructions
  calibration.py  per-voice bands, stored beside rhetoric_ext_calibration.json
  report.py       the six lines
tests/test_measure_parity.py   against the study's cache (skipped without it)
tests/test_measures.py         behaviour, runs anywhere
```

Wired into `scrub_ai_footprint`, `mimesis scrub` and `gate_candidates`. Nothing
in `gate.py`'s scoring, the Pareto front or `scrub.py`'s hard flags changed.
