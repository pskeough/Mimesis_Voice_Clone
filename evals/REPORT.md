# Mimesis upgrade study — before/after A/B

Three upgrades to the Mimesis v2 engine, each measured against the current
`fast`-backend baseline on a frozen brief set, across three axes:

1. **Fingerprint RMS-z** — the 13-feature stylometric distance of a generation to
   the author, on a *frozen reference ruler* (the baseline calibration), so the
   number is comparable across every config. Lower is closer to the author.
2. **Binoculars** — a perplexity-ratio machine-text detector; the fraction of
   generations that read "machine" against an empirically calibrated threshold.
3. **Blind discrimination fool-rate** — a `claude -p` judge, shown a held-out real
   piece and a generation under leak controls, guesses which is real; fool-rate is
   how often it picks the generation.

Generator/judge model for every engine `claude -p` call: `sonnet` (held constant
across all arms, matching the README baseline). Coding/analysis model: Opus 4.8.
This is R&D on Patrick's own voices; no corpus text is reproduced here — only
numbers.

## Method notes (honesty controls)

- **Frozen brief set**: 14 prompts (7 research, 7 creative) in `evals/briefs.json`,
  genre-matched per voice, content-generic to avoid leaking real pieces. Frozen at
  baseline, reused unchanged by every config.
- **Frozen reference fingerprint**: each voice's baseline calibration is snapshot
  to `evals/reference/` and used to score *all* configs, so Upgrade 3 (which
  shifts the live fingerprint) is still measured on the baseline ruler.
- **The fingerprint is backend-independent** (surface features of text, not
  embeddings), so Upgrade 1 changes *retrieval only*, not the RMS-z ruler — a
  clean isolation of the embedder's effect.
- **Style profiles are separate** (`research-style`, `creative-style`): the `fast`
  originals are kept intact for comparison, as required.
- **Detector threshold is calibrated per voice** on real corpus (human) vs generic
  plain-`claude` text (machine) — NOT on our own clones, which would be circular.
- **Coverage**: the stable axes (RMS-z, Binoculars) are run for all configs, both
  voices. Fool-rate (expensive, ±20% granularity at these n) is run for the arms
  where "does it fool a judge" is the headline question. Cells not yet run are
  marked pending; this document is updated as the background campaign completes.

---

## Phase 0 — Baseline (control)

Current `fast`-backend engine, curated brief set, RMS-z vs the frozen reference.

| voice | n | RMS-z mean±sd | ×self-baseline | Bino mean | machine% | mean words |
|---|--:|--:|--:|--:|--:|--:|
| research | 7 | 0.794 ± 0.131 | 0.78× | 1.114 | 29% (2/7) | 158 |
| creative | 7 | 0.869 ± 0.177 | 0.93× | 1.024 | 14% (1/7) | 258 |

Self-baselines: research 1.017, creative 0.934. Both arms sit comfortably under
the 2× acceptance bar (0.78× and 0.93×), i.e. the baseline engine already produces
generations closer to the author-mean than a held-out real piece is, on average.
Binoculars machine% is against each voice's calibrated threshold (human-corpus
95th percentile); most baseline drafts already read as human, so the detect-loop
has little to fix. This is the control every upgrade is measured against.

_(Fool-rate for baseline: pending the discrimination pass.)_

---

## Upgrade 1 — StyleDistance backend

**Implemented.** `StyleDistance/styledistance` (arXiv:2410.12757, MIT), 768-dim
RoBERTa-base, mean-pooled, wired into `embed.py` as the `style` backend (CUDA on
the Blackwell GPU, cosine, L2-normalized). Both voices re-embedded and
re-calibrated on `style` into separate profiles; `fast` kept intact.

### Does style retrieval pull *different* anchors than bge-small? (offline, decisive)

For each brief, the top-5 retrieved anchors under `fast` vs `style`:

| voice | mean shared top-5 | mean Jaccard |
|---|--:|--:|
| creative | **0.43 / 5** | **0.056** |
| research | 2.43 / 5 | 0.382 |

**Creative: 91% of the retrieved anchors change** when you swap the topic-leaning
embedder for the style embedder. The `fast` backend matches *subject* (a myth
brief pulls "revolt/revolution/light"; an entropy brief pulls "heat-death");
`style` matches *voice* — the same handful of stylistically-central pieces recur
across unrelated briefs. This is the literature claim made concrete: a style
critic anchors on how the author writes, not what about.

**Research: only muted divergence** (Jaccard 0.38) — the research corpus is
topically homogeneous (all LLM/RAG reports), so a topic embedder and a style
embedder largely agree. The style backend has the most to offer a *topically
diverse* corpus.

### Does it improve fingerprint distance / fool-rate?

| voice | RMS-z base → style | Δ | Bino machine% base → style | fool-rate base → style |
|---|--:|--:|--:|--:|
| creative | 0.869 → 0.864 | −0.006 | 14% → 0% | **50% → 25%** (worse) |
| research | 0.794 → 0.801 | +0.006 | 28% → 57% (worse) | 67% → 67% (same) |

(Discrimination-set gen RMS-z: creative 0.925 → 1.008 worse; research 0.945 → 0.888
slightly better. So research is a genuine wash-to-slightly-positive; creative is the
clear loser — the two voices differ exactly as the anchor-divergence predicts:
style hurts most where it changes the anchors most.)

**Verdict: style retrieval is not a win for these voices — and on creative it
mildly hurts.** RMS-z is a wash both directions (the gate already *selects*
candidates to minimise RMS-z regardless of anchor source, so better-voiced anchors
can't lower a distance that's already being optimised). More telling, the creative
**fool-rate dropped 50% → 25%** and its discrimination-set RMS-z *rose* 0.925 →
1.008. The mechanism is the flip side of the anchor-divergence finding: style
anchors are stylistically apt but **topically adrift** (same voice, unrelated
subject), so the generation loses the on-topic grounding the topic embedder
supplied and the judge more easily spots the content mismatch. Binoculars moved in
opposite directions by voice (creative better 14%→0%, research worse 28%→57%),
consistent with a weak, noisy detector at n=7.

The literature claim a style embedder is worth having still holds — but as a
*critic/scorer of drift*, not as a *retriever*: using it to pick generation anchors
trades away topical grounding. (Small-n caveat: fool-rate is 8 trials per cell,
±; the RMS-z direction, on 7-8 samples, is the more stable signal and agrees.)

---

## Upgrade 2 — Binoculars detector-in-loop

**Implemented.** Clean-room reimplementation (`detect.py`) of the Binoculars score
(Hans et al. 2024, BSD-3 upstream, no code copied): performer perplexity /
observer→performer cross-perplexity, over a shared-tokenizer base+instruct pair.
Wired into `scrub.detector_signal` and added to the gate as a final
reporting-then-gating step: a draft that reads clearly machine triggers one bounded
de-machine rewrite, accepted only if the detector improves without the fingerprint
regressing.

### Does it actually detect? — calibrated AUROC (real corpus vs generic AI)

Default pair **Qwen2.5-0.5B / -0.5B-Instruct** (~1.9 GB, tokenizer verified
identical):

| voice | AUROC (effective) | separating direction | human mean | AI mean | n (human/AI) |
|---|--:|---|--:|--:|--:|
| research | 0.675 | high = machine | 1.025 | 1.071 | 11 / 7 |
| creative | 0.650 | high = machine | 1.004 | 1.032 | 97 / 7 |

**Key finding — the canonical direction does not transfer.** With a small
same-size base+instruct pair, generic AI text scores *higher* than Patrick's real
writing, the *opposite* of the Falcon-7B "low = machine" convention (canonical-low
AUROC 0.325 / 0.350). The calibration is direction-aware and learns "high =
machine" empirically. Even then the pair is a **weak** detector here (AUROC
~0.65–0.68): a small-model Binoculars barely separates dense literary / academic
prose from generic AI. The larger Qwen2.5-1.5B pair was tried and **abandoned**:
its 152k-vocab cross-perplexity buffers exhaust the 7 GB VRAM budget at these
sequence lengths, confirming the small-pair constraint. **0.5B is the operating
pair.**

Implication for the "earn the strong claim (drafts that survive Binoculars)" goal:
with the weak small-pair detector, "surviving Binoculars" is a low bar and the
in-loop gate has thin signal to act on. Reported honestly rather than overstated.

### Fraction reading machine, baseline vs loop

The loop holds the base draft fixed and runs only the final detector step, so this
measures the loop's marginal effect, not generation noise.

| voice | baseline machine% | detect-loop machine% | Δ |
|---|--:|--:|--:|
| research | 29% (2/7) | 14% (1/7) | −1 draft |
| creative | 14% (1/7) | 14% (1/7) | score improved, label unchanged |

Research: of the 2 flagged drafts, one (`res_methods`) was rewritten machine→human
(and its RMS-z *improved*, 1.18→0.97); the other (`res_gap`) could not be improved
without regressing, so the accept-guard correctly left it alone. Creative: the one
flagged draft (`cre_myth`) was rewritten and its detector score moved toward human,
but not far enough to cross the threshold, so its label stayed machine. The loop
does the right thing on the drafts it flags — but with a weak detector, it flags
few, so the headline effect is small. Honest read: the detector-in-loop is a
**correct but low-yield** addition at this detector strength.

---

## Upgrade 3 — recalibrate_from_accepted

**Implemented.** Inference-time learning, no weight training (`accepted.py`,
`fingerprint.calibrate_weighted`, `mimesis accept` / `recalibrate`):

- **accept** → per-voice accepted set, added as high-priority exemplars in the
  compose kit AND folded into the fingerprint, recency-weighted (newest samples
  weighted most; base_weight 4.0, half-life 3.0).
- **edit** → a fresh `difflib` tracked-changes diff turns each edit into
  contrastive `ai_text`(avoid)/`human_text`(target) pairs — the same anchor type
  the research voice already ships — at document and per-changed-segment
  granularity.

### Mechanism check (offline): does the fold-in move the fingerprint toward what's kept?

Creative accepted set = a distinctive style cluster of 8 real pieces (selected by
clustering the corpus in z-space; centroid RMS-z 1.41 from the corpus mean — clear
headroom). After recalibration, **every feature moved toward the accepted
centroid**, e.g.:

| feature | base mean | recalibrated | accepted target | moved toward? |
|---|--:|--:|--:|:--:|
| sent_len_mean | 22.79 | 22.23 | 19.86 | yes |
| semicolons_per_100w | 0.44 | 0.56 | 1.15 | yes |
| para_len_mean | 9.46 | 9.34 | 8.68 | yes |
| ttr | 0.553 | 0.552 | 0.545 | yes |

### The test: do new outputs move toward the accepted set? _(pending recal generation)_

Distance-to-accepted (fixed ruler: accepted centroid, base-std scaled). Accepted
slice's own within-cluster spread ≈ 0.82 (floor).

| voice | baseline dist→accepted | recalibrate | Δ | RMS-z base→recal (corpus fidelity) |
|---|--:|--:|--:|--:|
| creative | 1.341 | **1.194** | **−0.147** (−11%) | 0.869 → 0.926 (+0.057, within gate) |

**The loop works.** New outputs move measurably toward the accepted set (−0.147,
about 11% of the way from baseline toward the ~0.82 within-slice floor), at a small
and expected cost: RMS-z-vs-corpus-mean rises 0.057 (still 0.99× self-baseline,
inside the 1.1 gate). The tradeoff is real and worth stating plainly — the accepted
slice sits 1.41 from the corpus mean, so converging on *what was kept* necessarily
moves *away from the corpus average*. Since the accepted set is genuine author
writing, that is the intended behavior: the voice bends toward the specific
sub-style the author keeps, which is exactly the "client kept rejecting outputs"
loop. Both the retrieval anchors and the recency-weighted fingerprint fold-in
contribute; 5 of 7 briefs moved closer, none catastrophically farther.

---

## Phase 4 — Combined & verdicts

### Full three-axis matrix (curated brief set)

| voice | config | RMS-z (×self) | Bino machine% | dist→accepted |
|---|---|--:|--:|--:|
| research | baseline | 0.794 (0.78×) | 29% | — |
| research | style | 0.801 (0.79×) | 57% | — |
| research | detect | 0.793 (0.78×) | 14% | — |
| creative | baseline | 0.869 (0.93×) | 14% | 1.341 |
| creative | style | 0.864 (0.92×) | 0% | 1.345 |
| creative | detect | 0.831 (0.89×) | 14% | 1.314 |
| creative | recalibrate | 0.926 (0.99×) | 14% | 1.194 |
| creative | **combined** | **0.837 (0.90×)** | **0%** | **1.089** |

Discrimination fool-rate: research baseline 67% / style 67%; creative baseline 50%
/ style 25%.

### The combined arm

Combined (style backend + accepted-recalibrate + detector-in-loop) is the best cell
on two of three axes: **dist→accepted 1.089** — the lowest anywhere, below
recalibrate-alone (1.194), because style's stylistic anchoring stacks with the
recalibrate fold-in and pulls hardest toward the kept sub-style — while *also*
holding RMS-z at 0.837 (better than recalibrate-alone's 0.926) and Binoculars at 0%
machine. The recalibrate signal is doing the work; style and the detector don't add
convergence but don't wreck it here either.

### Verdicts (helped / neutral / hurt)

1. **StyleDistance backend — NEUTRAL-TO-HURT. Do not adopt as the retriever.**
   Changed 91% of creative anchors yet moved mean RMS-z by ≤0.006 either way (the
   gate already optimises that distance). On creative it *hurt* the judge fool-rate
   (50%→25%) and discrimination RMS-z (0.925→1.008); on the topically-homogeneous
   research corpus it was a wash (67%→67%). Root cause: style-matched anchors are
   topically adrift, costing the on-topic grounding the topic embedder gave. The
   style embedder is worth keeping as a *drift critic/scorer*, not as the anchor
   retriever — which is what the literature actually claims.

2. **Binoculars detector-in-loop — CORRECT BUT LOW-YIELD.** The canonical
   "low=machine" direction does not transfer to a small pair (learned empirically:
   high=machine), the 0.5B pair is a weak detector here (AUROC 0.66), and the 1.5B
   pair exceeds the 7 GB VRAM budget. The loop does the right thing when it fires
   (research 2/7→1/7 machine; correct accept-guard rejections) but a weak detector
   flags few drafts, so the headline effect is small. Ship it as QA telemetry, not
   as a load-bearing gate at this detector strength.

3. **recalibrate_from_accepted — HELPED. The highest-value upgrade.** New outputs
   converge on the accepted set (dist→accepted 1.341→1.194 alone, →1.089 combined)
   at a small, in-gate fidelity cost (RMS-z +0.06). Every folded feature moved
   toward what was kept; 5/7 briefs moved closer. This is the loop that answers "the
   client kept rejecting outputs": the voice measurably bends toward what the author
   actually keeps, from 5–10 samples, with no weight training.

4. **Combined — the recalibrate win, undamaged by the others.** Best dist→accepted
   and Binoculars, strong RMS-z. If shipping one thing, ship Upgrade 3; keep 1 and 2
   as scorers/telemetry.

### Caveats (honest limits)

- Fool-rate cells are 6–8 trials (±); the RMS-z and dist→accepted deltas (7–8
  samples) are the more stable signals and are what the verdicts lean on.
- The detector is weak (AUROC ~0.66) and its thresholds are small-sample (7 AI
  decoys); machine% should be read as directional, not absolute.
- **recalibrate was demonstrated on creative only.** Research was not run because
  its most-distinctive corpus cluster sits just 0.19 RMS-z from the corpus mean —
  no headroom for convergence to be visible. The loop's measurable effect scales
  with how distinct the accepted set is from the base corpus.
- Combined discrimination (fool-rate) was omitted: the accepted slice overlaps the
  held-out real pieces (e.g. `105`), which would leak accepted exemplars into their
  own trial. Combined is reported on the leak-free curated axes only.

_Generator/judge: `sonnet`, held constant. Raw per-draft artifacts under
`evals/<config>/<voice>/`; regenerate this matrix with `python evals/aggregate.py`._
