# The cadence blind spot

**Claim.** The 13-feature stylometric fingerprint cannot see rhythm, the compose
gate optimises it anyway, and the resulting generations are separable from real
authorial prose on rhythm while being nearly indistinguishable on surface style.
This reproduces across three corpora and two authors.

All numbers below are reproducible from this repo. Client corpora are referenced
by role only; no external text is stored here, and the external audit writes its
output outside this repository by construction
(`evals/cadence_audit_external.py` refuses an `--out` inside the repo).

---

## 1. The fingerprint is blind to sentence order, by construction

Every feature in `fingerprint.FEATURES` is an order-invariant aggregate: a mean,
a standard deviation, a percentage, a per-100-word rate, a type-token ratio.
Shuffling a text's sentences leaves all thirteen unchanged. This is not a bug in
the implementation; it is a property of the feature set.

Measured (`evals/cadence_blindness.py`, creative corpus, n=93): sorting each
piece's sentences shortest-to-longest — a monotone ramp no writer produces —

| condition | mean RMS-z | delta |
|---|--:|--:|
| original order (control) | 0.7768 | — |
| sentences shuffled | 0.7766 | −0.0002 |
| sentences sorted into a ramp | 0.7791 | **+0.0023** |

and **0 of 93** rhythm-destroyed pieces were flagged off-voice.

## 2. Adding order-aware features fixes the metric

`cadence.py` adds 13 features that are functions of sentence order or of
within-sentence clause structure: lag-1/lag-2 length autocorrelation, normalized
successive deltas, direction-change rate, same-class run length, 3x3 transition
entropy, P(short | previous long), paragraph open/close length ratios, and
clause length/variability/count. Sentence classes are cut at the author's own
terciles, not fixed word counts.

Held-out A/B (`evals/cadence_ab.py`, calibrate on 70%, score on 30%):

| corpus | v1 AUROC (real vs ramp) | v2 AUROC | v2 cadence block alone |
|---|--:|--:|--:|
| creative (n=93) | 0.512 | **1.000** | 0.999 |
| personal (n=25) | 0.492 | **1.000** | 1.000 |
| external author, long-form (n=18) | 0.559 | **0.877** | — |
| external author, short-form (n=182) | 0.468 | **0.778** | — |

v1 sits at chance on all four. False-flagging of real prose stays at the p95
design point (creative: 1/28).

Honest limit: a random *shuffle* is only weakly detected (AUROC 0.533), because
a reshuffle of a human's sentences still has human-like rhythm statistics in
aggregate. v2 detects pathological rhythm, not any reordering.

## 3. The shipped engine's real failure is rhythm, not style

Scoring generations that already existed on disk, against a corpus-calibrated
fingerprint, with the human reference computed leave-one-out
(`evals/cadence_audit.py`, `evals/cadence_audit_external.py`):

| corpus / config | n | cadence z | surface z | AUROC cadence | AUROC surface |
|---|--:|--:|--:|--:|--:|
| creative — HUMAN | 111 | 0.851 | 0.773 | 0.500 | 0.500 |
| creative — baseline | 7 | 1.643 | 0.869 | **0.884** | 0.691 |
| creative — style | 7 | 1.627 | 0.864 | 0.867 | 0.689 |
| creative — detect | 7 | 1.586 | 0.831 | 0.871 | 0.677 |
| creative — recalibrate | 7 | 1.457 | 0.926 | 0.843 | 0.718 |
| creative — combined | 7 | 1.262 | 0.837 | 0.802 | 0.671 |
| external author — HUMAN | 18 | 1.021 | 1.020 | 0.500 | 0.500 |
| external author — shipped | 45 | 1.491 | 0.731 | **0.794** | **0.316** |

Two independent authors, two independent engines, same signature: surface style
lands close to the author while rhythm is plainly separable.

**The external surface AUROC of 0.316 is the sharper result.** Below 0.5 means
the generations sit *closer to the author's mean than his own writing does* —
more consistent than the human. That is mean-collapse: the gate optimises a
distance-to-mean and gets prose that is more average-author than the author. A
reader meets correct vocabulary and density carrying the wrong music, and
reports that it does not sound like them, which is exactly the feedback that
prompted this investigation.

## 4. The scrubber was causing part of it

Pooled worst cadence features across generations:

| feature | creative (35 gens) | external author (45 gens) |
|---|--:|--:|
| `sl_delta_abs` (jump violence) | +2.32 SD | +1.73 SD |
| `sl_delta_sd` | +2.07 SD | +2.31 SD |
| `sl_autocorr1` (mechanical alternation) | −2.05 SD | −1.13 SD |
| `sl_short_after_long` (the punch move) | +1.68 SD | +1.12 SD |
| `para_open_ratio` | — | −2.94 SD |

The burstiness advisory used to read *"Vary pacing; mix a short punchy line with
a long one."* Burstiness is sentence-length **stdev**, an order-invariant
statistic, and the cheapest way for a model to raise it is rigid long/short
alternation — which is precisely the `sl_autocorr1` / `sl_short_after_long`
signature above. The proxy was satisfied and the thing it proxied for got worse.
The advisory now tells the model to let passages run long or short rather than
alternating line by line.

## 5. A pre-existing v1 bug, found on the way

`emdash_per_100w` calibrated on 34 creative pieces gave mean 0.0022 / std
0.0117. The author's own held-out prose averaged **|z| = 12.05** on that one
feature, contributing ~145 to a sum of squares where every other feature
contributed 1–2. The RMS over all features had quietly become a single-feature
em-dash detector.

Fixed by winsorizing per-feature z at ±4 (`fingerprint.Z_CLIP`) before the RMS.
This improves v1 as much as v2, and is why the corrected v2 numbers in §2 are
stronger than the pre-fix run.

## 6. Structural change to the compose loop

The loop was: slate → fingerprint-select → quality-select → **scrub**. Selection
happened before half the criteria were applied, so the winner was chosen to
minimise RMS-z and then edited away from the point it was chosen for, with a
repair rewrite to patch the damage. That is the mechanism behind "faithful voice"
and "clean of AI tells" fighting each other.

Now: every candidate is scalpelled first (deterministic, meaning-preserving),
then scored, then the winner is picked from the Pareto frontier over
(fingerprint distance, scrub hard-flag count). Every candidate is measured in the
state it would actually ship in, and the LLM judge chooses among genuine
trade-offs rather than being handed a pre-damaged draft.

## 7. End-to-end: gating on v2 is NOT supported by the evidence

> **Read this section header carefully — an earlier version of this document
> claimed the opposite.** The A/B results below were real, but they were measured
> on a ruler that a later, better calibration superseded, and they did not
> survive the blind discrimination check. Section 7d is the current conclusion.
> The rest is kept so the retraction is auditable rather than quietly edited away.

`evals/ab_cadence_e2e.py`. Both arms calibrate on a 70% document split and
generate under their own gate; every output is scored on a v2 ruler calibrated
from the unseen 30%. Same briefs, same generator model, same seed, same
retrieval store. The only difference is which feature set the gate optimises.

Creative voice, 7 briefs per arm:

| arm | n | ruler RMS-z | cadence | cadence vs human |
|---|--:|--:|--:|--:|
| HUMAN (split-A pieces on the split-B ruler) | 77 | 1.005 | 0.999 | — |
| v1 gate | 7 | 1.601 | 1.915 | +0.916 |
| **v2 gate** | 7 | **1.099** | **0.984** | **−0.015** |

Paired by brief, since both arms ran the same set:

| brief | v1 cadence | v2 cadence | delta |
|---|--:|--:|--:|
| cre_solitude | 2.098 | 0.641 | −1.458 |
| cre_myth | 2.313 | 0.903 | −1.410 |
| cre_sea_figure | 1.969 | 0.675 | −1.294 |
| cre_doubt | 2.218 | 1.129 | −1.089 |
| cre_entropy | 2.064 | 1.469 | −0.595 |
| cre_time_memory | 1.501 | 0.948 | −0.553 |
| cre_desire | 1.245 | 1.123 | −0.121 |

**7/7 briefs improved on both cadence and overall RMS-z**, mean cadence delta
−0.931, exact two-sided sign test **p = 0.0156** — the smallest attainable p at
n = 7. The v2 arm's rhythm is statistically indistinguishable from the author's
own writing on a ruler neither arm saw.

Note this is not merely "the gate optimises what it is scored on": the gate ran
on a split-A calibration and the score came from a split-B calibration, and the
v1 arm's *overall* RMS-z also improved under v2 (1.601 → 1.099) despite v1 being
the arm that optimises the v1 features exclusively.

### Controlled replication: the effect survives the prompt fix

The run above used the defective burstiness instruction (§8) in *both* arms, so
v2's advantage might merely have been compensating for a bad prompt. Rerun with
the instruction fixed and everything else byte-identical (same profiles, same
70/30 split, same seed, same ruler, same briefs) — the instruction is the only
variable:

| run | v1 cadence | v2 cadence | improved | mean delta | sign-test p |
|---|--:|--:|--:|--:|--:|
| creative, defective kit | 1.915 | 0.984 | 7/7 | −0.931 | 0.0156 |
| creative, corrected kit | 1.621 | **1.008** | **7/7** | −0.612 | **0.0156** |

The prompt fix absorbed about a third of the apparent effect (−0.931 → −0.612),
which is the honest size of the confound. The effect itself survives: 7/7 again,
and v2 still reaches parity with the author's own writing (1.008 vs human 0.999).

### The same experiment on a second author shows no effect

External long-form author, 6 briefs per arm, same design:

| arm | n | ruler RMS-z | cadence | cadence vs human |
|---|--:|--:|--:|--:|
| HUMAN | 48 | 1.034 | 1.002 | — |
| v1 gate | 6 | 1.674 | 1.813 | +0.811 |
| v2 gate | 6 | 1.451 | 1.410 | +0.407 |

Paired: **5/6 improved on both axes, mean cadence delta −0.403, sign-test
p = 0.2188.** One brief regressed (+0.182).

So the direction agrees across authors and the gap roughly halves, but this arm
is neither significant at n=6 nor does it reach human parity the way the first
author's did (1.410 vs 1.002 human, against 0.984 vs 0.999). **Do not report the
two runs as a single replicated result.**

Two confounds specific to this run, both since fixed, make it a weak test rather
than a negative one:

1. The profile carried the hardcoded `rmsz_max: 1.1`, unreachable for this corpus
   (p95 1.701). **The gate never passed in either arm on any brief**, both rewrite
   iterations burned every time, and both arms fell back to "closest by RMS-z".
   So this measured the selection metric with the gate machinery disabled.
2. Calibration ran on unsegmented pieces (max/median = 22), giving a noisy ruler
   built from 15 usable pieces for 26 features.

Rerun with `rmsz_max: "auto"`, segmented calibration, and the corrected kit
instruction — the gate now fires, rewrite iterations drop from 12/12 briefs to
7/12, and mean time per brief falls from 305s to 111s:

| run | v1 cadence | v2 cadence | improved | mean delta | p |
|---|--:|--:|--:|--:|--:|
| external-longform, defective kit + unreachable gate | 1.813 | 1.410 | 5/6 | −0.403 | 0.2188 |
| external-longform, corrected kit + auto gate | **1.504** | 1.413 | 3/6 | −0.091 | 1.0000 |

**The v2 arm did not move (1.410 → 1.413). The v1 arm improved by 0.31 and closed
most of the gap on its own.** On this author, essentially all of the apparent
cadence benefit came from deleting one sentence from the prompt, not from gating.

### What actually governs the effect: the size of the pre-existing gap

A third arm settles it. The same author's SHORT-FORM voice (197 natural pieces,
179 usable, no THIN warning, 137 gate / 60 ruler) was predicted in advance to
behave like the first author's, on the theory that calibration density was the
mediator. **That prediction was wrong**, and the way it failed is the finding:

| run | human | v1 | v2 | v1 gap | v2 gap | % of gap closed | improved | p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| creative, defective kit | 0.999 | 1.915 | 0.984 | +0.916 | −0.015 | 102% | 7/7 | 0.0156 |
| creative, corrected kit | 0.999 | 1.621 | 1.008 | +0.622 | +0.009 | 99% | 7/7 | 0.0156 |
| external-longform essay, defective kit | 1.002 | 1.813 | 1.410 | +0.811 | +0.408 | 50% | 5/6 | 0.2188 |
| external-longform essay, corrected kit | 1.002 | 1.504 | 1.413 | +0.502 | +0.411 | 18% | 3/6 | 1.0000 |
| external-longform social | 1.123 | 1.155 | 1.093 | +0.032 | −0.030 | — | 4/6 | 0.6875 |

On the short-form voice **v1 was already at parity** (+0.032). There was no defect
to correct, and v2 correctly did nothing — it did not manufacture an improvement,
and it did not make things worse.

Correlation between the pre-existing v1 cadence gap and how much of it v2 closes:
**r = 0.789 (n = 5 runs)**.

Two factors, not one:

1. **Benefit scales with the gap.** The cadence gate is a *corrective*, not an
   enhancement. Applied where rhythm is already right, it is a no-op.
2. **Calibration quality sets the achievable floor.** v2 converges to a
   voice-specific floor — 1.00 on the first author (parity), 1.41 on the second
   author's long-form voice, stable across two structurally different runs. That
   second floor sits 0.41 above human, and its ruler was 12 segments estimating
   26 features. Corpus size does not govern the improvement; it governs how far
   down the improvement can reach.

The practical consequence is a workflow, not a switch: **measure the cadence gap
for a voice before deciding whether to gate on it.** `evals/cadence_audit.py`
answers that in seconds from existing generations, with no new compute.

### 7d. The retraction: better measurement kills the effect

Two independent checks, both run unattended overnight, both against the claim.

**(i) The effect collapses on a correctly calibrated ruler.** Every A/B above used
a ruler built from raw pieces. Section 8 shows raw-piece calibration is wrong for
this corpus (one 128,524-word document against a median of 822) and that
segmenting improves held-out discrimination from AUROC 0.962 to 1.000 while
correcting the false-flag rate from 0% to the designed 3.5%. Re-running the same
A/B on the better ruler:

| creative A/B | v1 | v2 | delta | improved | p |
|---|--:|--:|--:|--:|--:|
| raw ruler, defective kit | 1.915 | 0.984 | −0.931 | 7/7 | 0.0156 |
| raw ruler, corrected kit | 1.621 | 1.008 | −0.613 | 7/7 | 0.0156 |
| **segmented ruler, corrected kit** | 1.487 | 1.313 | **−0.174** | **5/7** | **0.4531** |

The headline result was measured on the instrument this same document had already
shown to be inferior. On the better instrument the effect is a third the size and
not significant.

**(ii) Blind discrimination shows no benefit, and possibly harm.** A judge is
shown a held-out real piece and a generation under leak controls and picks which
is real; fool-rate is how often it picks the generation. This is the only metric
here not computed from features chosen by the person who built the generator.

| voice | v1 fool-rate | v2 fool-rate | Fisher exact p |
|---|--:|--:|--:|
| creative | 2/8 (25.0%) | 2/8 (25.0%) | 1.0000 |
| external author, short-form | **7/8 (87.5%)** | **3/8 (37.5%)** | 0.1189 |

Identical on one voice. On the other, v1 fools the judge more than twice as often
as v2. n=8 per arm cannot resolve this (a difference below ~69 points is not
significant at this n), so it is not evidence that v2 *harms* output — but it is
squarely inconsistent with v2 helping, and it is the direction that matters.

**Current position.** The order-aware feature set is a genuine improvement as a
*measurement*: it is the only thing in the system that can see rhythm at all
(§1–§3), and the audit it enables located a real defect in shipped generations
across two authors. As a *generation gate* it is not justified: `create_profile`
therefore writes `feature_set: "v1"`, and a voice should be switched to v2 only
after an A/B on that voice earns it.

A plausible mechanism for why gating fails while measuring succeeds: selecting on
26 features instead of 13 over-constrains the slate. A candidate that satisfies
more statistical targets is not the same as a candidate that reads better, and
the blind judge scores the second thing. This is the same over-optimization that
produced surface AUROC 0.316 on the external author's shipped generations (§3) —
outputs more consistent than the human they imitate.

**What survives the retraction:** everything in §1–§6 and §8. The blindness proof
is a property of the feature set, measured offline with no generation involved.
The audit results are measurements of existing artifacts. The kit-instruction
Goodhart fix (§8) improved the v1 arm by 0.31 cadence on its own and costs
nothing. The calibration, threshold and winsorization fixes are independent of
any A/B.

### A corpus-size note (superseded as the primary explanation)

|  | first author | second author |
|---|--:|--:|
| calibration units (segmented) | 207 | 66 |
| gate calibration | 77 | 48 |
| **eval ruler** | **34** | **12** |
| v2 effect | 7/7, p = 0.0156 | 3/6, p = 1.0 |

The second author's ruler estimates 26 features from 12 segments, and his gate
from 48 — both inside the range `_report_calibration_confidence` flags as THIN.
A gate cannot steer toward a target whose per-feature standard deviations are not
determined, and the v2 arm's refusal to move across two structurally different
runs (1.410, 1.413) is what a poorly-estimated target looks like: consistent, and
stuck ~0.4 above the author's own baseline.

Read together, these are two points on the corpus-size-vs-fidelity curve, and the
operative claim is narrower than "cadence gating works": **cadence gating works
where the corpus supports estimating the cadence features, and the threshold sits
somewhere between 66 and 207 calibration units.** Locating it properly needs a
deliberate sweep, not two accidental samples.

## 8. Universality: what broke on the second author

Every failure on the external corpus traced to a constant typed by hand rather
than measured. The engine was calibrated to one author and treated that
calibration as physics. Fixes, all measured:

| defect | symptom on the external corpus | fix |
|---|---|---|
| `rmsz_max: 1.1` hardcoded | self-baseline 0.966, p95 1.701 — the ceiling sat *below* the author's own prose, so all 6 briefs failed the gate, burned both rewrite passes, and fell back to "closest by RMS-z" at ~5x cost | `rmsz_max: "auto"` uses the profile's own p95; default for new profiles, legacy default pinned |
| calibration over raw pieces | pieces ranged 353–17,717 words (max/median = 22), so feature stds partly measured document length; only 15 usable pieces for 26 features | segment at paragraph boundaries to ~1500 words when max/median >= 6; 18 pieces became 66 units, 15 usable became 63 |
| silent calibration at n=5 | no signal that a p95 threshold from 11 pieces is weaker than one from 111 | `calibrate` reports THIN / VERY THIN against a 3x-features rule of thumb |
| kit instruction | see below | rewritten |

The same lopsidedness was present in the *first* author's corpus and had gone
unnoticed: one 128,524-word document against a median of 822 (ratio 156). Held-out
effect of segmenting there:

| calibration | n | p95 | AUROC real vs ramp | real false-flagged |
|---|--:|--:|--:|--:|
| raw pieces | 77 | 2.109 | 0.962 | 0/26 (0.0%) |
| segmented | 144 | 1.401 | **1.000** | 2/57 (3.5%) |

The 0% false-flag rate at p95 is the tell: a threshold that never fires on real
writing is not calibrated, it is inert. 3.5% is the designed behaviour.

### The instruction that caused the damage

`_rules_block` — the style rules injected into **every** generation, not only
repairs — read:

> keep burstiness (sentence-length variety, stdev) above X: mix short punchy
> lines with long stacked-clause ones.

Burstiness is sentence-length stdev, an order-invariant statistic. The cheapest
way for a model to raise it is rigid long/short alternation, and across 80
generations from two independent authors that is precisely what came back
(`sl_short_after_long` +1.68 / +1.12 SD, `sl_autocorr1` −2.05 / −1.13 SD). The
rule now asks for variation at passage scale and explicitly forbids line-by-line
alternation. This is the clearest Goodhart case in the system: the proxy was
satisfied and the thing it proxied for got worse, in every generation, for the
life of the engine.

## 9. What is not yet established

- The external long-form author's v1-vs-v2 arm is still running.
- The cadence failure is **voice-dependent**. On the research voice it is absent
  (AUROC 0.494), but that corpus is 11 pieces and academic prose has constrained
  rhythm. Do not generalise the creative/long-form result to it.
- Everything here measures distance-to-author, not reader judgement. Whether a
  cadence-corrected draft actually reads more like the author to a human is
  untested, and is the claim that ultimately matters.
