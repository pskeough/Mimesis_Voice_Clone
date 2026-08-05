# Mimesis

**One voice engine, N voices, a discriminator in the loop.** Mimesis clones a
writer's voice from their own corpus and holds every draft to a measured
stylometric standard before it ships. It is the voice half of a two-system pair,
where a companion memory system holds what you know and Mimesis holds how you
sound.

The repo ships clean with zero personal data by construction. The only
checked-in profile is a synthetic public-domain-style example that runs
immediately, and `profiles/*` is gitignored for everything else.

---

## Thesis

Prompt-only voice imitation fails stylometric verification. Ask a model to
"write like X" and a classifier still separates the imitation from the real
thing (arXiv:2603.29454, 2509.14543). Generation with a discriminator in the
loop closes most of that gap (StealthRL 2602.08934, AuthorMist 2503.08716), but
evasion tuned against perplexity-family detectors leaves a residual style
fingerprint that style-embedding detectors still catch (2505.14608). So Mimesis
does not trust the model's taste about its own output. It generates a slate,
scores every candidate on a 13-feature stylometric fingerprint, and selects on
that distance, demoting the LLM judge to a secondary quality signal and adding
an AI-footprint scrubber and a fidelity audit as hard gates.

The design bet, that judge-free stylometric selection beats LLM-judge selection,
was confirmed empirically in prior work before it was hardened here:
fingerprint-chosen candidates scored better than the judge-preferred ones.

One caveat constrains what the lead citation licenses, so it belongs here and
not in a footnote. 2603.29454 tests GPT-4o only, prompting attacks only
with no DIPPER or STRAP-class paraphrase machinery, and short informal registers
only (Enron email, BOLT SMS, Twitter). The defensible reading is that naive
prompting-based impersonation fails forensic authorship verification in
short-form registers on one model. It does not license a general claim about
long-form prose, and nothing here leans on it harder than that.

---

## Architecture

```
 corpus ──ingest──> SQLite store ──calibrate──> fingerprint.json + scrub_calibration.json
 (docx/pdf/txt/md)  (chunks + FTS5 + vectors)         │                    │
                                                       │                    │
 task ─────────────────────────────┐                  ▼                    ▼
                                    │          ┌───────────────┐   ┌────────────────┐
   retrieve (RRF hybrid recall,     └────────► │  compose loop │   │    scrubber    │
   MMR λ=0.65 diversification) ───► anchors ──►│  (gate.py)    │──►│ dashes/banlist │──► voiced draft
   + transform demos                           │  slate → gate │   │ + fidelity     │
                                               │  → select     │   └────────────────┘
                                               │  → rewrite    │
                                               └───────────────┘
```

**ingest** turns `.docx/.pdf/.txt/.md` into roughly 180-word chunks and writes
them to SQLite as a chunk table, an FTS5 keyword index, and embedding blobs. An
unchanged corpus is hash-skipped.

**embed** runs `fast` by default, meaning bge-small under ONNX with no torch
dependency, with `style` available as an optional extra. Corpus matrices are
cached and keyed on store mtime.

**retrieve** fuses vector and FTS5 recall through Reciprocal Rank Fusion at
k=60, then diversifies with Maximal Marginal Relevance scored as
`0.65·sim − 0.35·max_overlap`, so the anchors handed to the generator are
genuinely different pieces and not five paraphrases of one. Profiles that
ship a `pairs.jsonl` additionally get contrastive AI-to-author transform
demonstrations, which are the strongest anchor available for rewriting prose
that already sounds like a model.

**fingerprint** measures 13 surface features: burstiness, short and long
sentence rates, colon, semicolon and em-dash density, question rate, commas per
sentence, paragraph length, first-person openers, type-token ratio, and mean
word length. Each is z-scored against a per-author calibration and collapsed to
one RMS-z distance, reported alongside an honest leave-one-out corpus
self-baseline. Per-feature z is winsorized at ±4 before the RMS, which matters
more than it sounds: on one corpus the em-dash feature reached mean |z| = 12.05
on the author's own held-out prose and contributed roughly 145 to a sum of
squares where every other feature contributed 1 to 2, so the composite distance
had quietly become a single-feature em-dash detector.

**gate** builds a slate of 3 to 5 genuinely different candidates through
verbalized sampling, scrubs and scores every one of them, applies the
fingerprint gate, takes the Pareto frontier over voice fidelity and scrub hard
flags, and quality-selects from the survivors. Candidates are scrubbed before
scoring rather than after, because the old order picked a draft and then edited
it away from the thing it was picked for. On zero survivors the loop issues one
targeted rewrite per candidate citing that candidate's worst features, capped at
two iterations.

**scrub** works in three tiers. Hard rules apply a prose-dash ceiling read from
the author's own corpus, a corpus-calibrated banlist minus a per-author
whitelist, a burstiness floor and a hedge ceiling. A fidelity audit compares
numbers, citations and acronyms in the source against the output and reports
what was dropped, altered or invented. An optional perplexity-ratio detector
signal is available for reporting and is never a gate. Scalpel mode strips the
always-safe things and flags the rest, and it never rewrites prose to force the
mean, because mean-collapse kills the voice it was supposed to protect. What
counts as always-safe is itself per-author: the em-dash strip runs only for
writers whose own corpus does not use prose dashes, and above that calibrated
band a draft is flagged against the author's own rate rather than rewritten.

Generation shells out to the local `claude` CLI in headless `-p` mode with a
default model of `sonnet`, so no API keys and no cloud services are required
beyond a working Claude Code install. Everything else runs fully offline.

---

## What the fingerprint cannot see

Every feature in the default set is an order-invariant aggregate, meaning a
mean, a standard deviation, a percentage, a per-100-word rate or a ratio.
Shuffling a text's sentences leaves all thirteen unchanged. Sorting them
shortest-to-longest into a monotone ramp that no writer produces leaves them
effectively unchanged as well, measured at +0.0023 mean RMS-z across 93 pieces,
with 0 of 93 rhythm-destroyed pieces flagged off-voice.

That is not a bug in the implementation. It is a property of the feature set,
and no amount of tuning reaches it. The fingerprint is blind to cadence by
construction, and a gate selecting on it constrains vocabulary and density while
leaving rhythm entirely free.

`cadence.py` supplies 13 order-aware features as `feature_set: "v2"`, including
sentence-length autocorrelation, successive-delta magnitude, direction-change
rate, same-class run length, class-transition entropy, and the probability of a
short sentence following a long one, with classes cut at the author's own
terciles and not at fixed word counts. On held-out data v2 separates real
prose from rhythm-destroyed prose at AUROC 1.000 where v1 sits at chance.

It ships as a diagnostic and not as the default gate, because as a gate it did
not earn its place. On a correctly segmented ruler the A/B effect fell from
−0.613 (7/7, p=0.016) to −0.174 (5/7, p=0.45), and blind discrimination showed
no difference on one voice (25% against 25% fool-rate) while favouring v1 on
another (87.5% against 37.5%, Fisher p=0.12). Selecting on 26 features instead
of 13 plausibly over-constrains the slate, since a candidate that hits more
statistical targets is not thereby a candidate that reads better. Run
`mimesis audit VOICE` to see whether a given voice has a rhythm gap at all, and
switch it to v2 only after an A/B on that voice earns it. The full evidence and
the retraction are in `evals/CADENCE_FINDINGS.md`.

One finding from that work already changed the shipped engine. The burstiness
advisory used to read "vary pacing, mix a short punchy line with a long one."
Burstiness is sentence-length standard deviation, an order-invariant statistic,
and the cheapest way for a model to raise it is rigid long-short alternation,
which is exactly the mechanical-alternation signature the cadence features
flagged at −2.05 SD. The proxy was satisfied while the thing it proxied for got
worse. I wrote that advisory. Finding that my own instruction was manufacturing
the defect is the reason cadence stayed a diagnostic instead of quietly becoming
a second gate, and I take it as the general warning here: an advisory phrased as
a target teaches a model to hit the statistic by the cheapest available route,
which is rarely the route a writer takes. The advisory now tells the model to
let whole passages run long or stay short instead of alternating line by line.

---

## Quickstart

One command, on macOS or Linux:

```bash
git clone https://github.com/pskeough/Mimesis_Voice_Clone.git
cd Mimesis_Voice_Clone
bash install.sh
```

On Windows:

```bash
powershell -ExecutionPolicy Bypass -File install.ps1
```

That creates the venv, installs the package, builds and calibrates every voice
that ships with a corpus, registers the MCP server with Claude Code, and runs a
health check on each profile. Both installers are safe to re-run and skip
finished work.

Then point it at your writing:

```bash
mimesis new myvoice --from ~/Documents/my-writing
```

That is the whole setup. `new` walks the folder recursively, copies every
supported document into the profile, ingests, and calibrates. Nested paths are
folded into filenames rather than collapsed, so `essays/2024/rain.txt` and
`notes/rain.txt` both survive, and re-running it is idempotent by content hash.
Add more writing later with `mimesis add myvoice --from ~/more`, or drop files
straight into `profiles/myvoice/source_documents/` and run `mimesis add myvoice`
to rebuild.

Always rebuild after adding documents. The fingerprint is a z-score against
corpus means, so new writing moves the baseline, and a stale `fingerprint.json`
does not raise an error. It silently scores every candidate against the wrong
writer.

Working with the shipped example voice, or driving the steps by hand:

```bash
mimesis profile list                 # the synthetic 'example' voice ships ready
mimesis ingest example               # build the store (downloads bge-small once)
mimesis calibrate example            # write fingerprint.json + scrub_calibration.json
mimesis doctor                       # self-test; expect all green

mimesis compose example "write a short note about morning routines"
mimesis compose example "..." --dry-run          # whole pipeline, no generation
mimesis scrub example "some draft text"          # vet a draft
mimesis eval example --fingerprint-only          # fingerprint distribution, no generation
mimesis audit example                            # cadence diagnostic
```

### What makes a good corpus

Use writing the person actually wrote, in the register they want back. Email,
essays, posts, chapters, letters. Keep out anything they edited, co-wrote, or
prompted a model into, because a corpus with borrowed prose in it produces a
voice that is partly someone else's and nothing downstream detects that. The
system has no way to know. It calibrates against whatever it is handed, and a
fingerprint built half from an assistant will pass its own gate happily while
sounding like nobody in particular. Aim for roughly 40 pieces at the length they normally
write; `calibrate` prints a `THIN CALIBRATION` warning below about three pieces
per feature, and standard deviations estimated from eight pieces are noise that
the gate then inherits.

Split by form rather than by topic. A single fingerprint computed across short
posts and book chapters measures format far more than it measures voice, which
shows up as a self-baseline well above 1.0 and a gate that nothing satisfies.
Give each form its own voice. The same applies to one very long document sitting
in a mixed corpus, where a book beside short pieces does not merely weight the
fingerprint, it becomes the fingerprint.

### As an MCP server

`scripts/register-mcp.ps1` and `scripts/register-mcp.sh` register a stdio server
named `mimesis-v2` with Claude Code at user scope. They are write-only helpers,
run them yourself when ready, and they never touch an existing `mimesis`
registration. Tools: `get_voice_guide`, `retrieve_style_examples`,
`retrieve_transform_demos`, `compose_in_voice`, `scrub_ai_footprint`,
`eval_voice`.

`CLAUDE.md` in this folder is the operating guide for Claude Code itself,
covering setup, verification, and the rules that must not be worked around.

---

## Install extras

| extra | pulls | enables |
|---|---|---|
| _(base)_ | fastmcp, fastembed, numpy, python-docx, pypdf | full engine, `fast` embeddings, PDF ingest |
| `[style]` | sentence-transformers, torch | StyleDistance `style` backend |
| `[detect]` | transformers, torch | Binoculars-style perplexity-ratio detector signal (reporting only, never a gate) |

Both extras pull torch, which is a multi-GB download. Stay on the default `fast`
backend unless you have a measured reason not to.

---

## Eval results

The discrimination eval mixes held-out real pieces with generations under leak
controls, where the brief is derived neutrally from the real piece and that
piece plus its three nearest neighbours are excluded from the anchors. A
`claude -p` judge is asked which text is the real author, and the harness
reports fool-rate alongside the generated RMS-z distribution against the corpus
self-baseline.

Acceptance bar: generated RMS-z within **2×** the corpus self-baseline.

Runs below are `mimesis eval <voice> --held-out 5` over a real personal corpus,
`fast` backend, `sonnet` as both generator and judge. Voice names are generic
and no corpus content is shown.

| voice | corpus self-baseline RMS-z | held-out real mean RMS-z | generated mean RMS-z | ratio | fool-rate | n |
|---|---|---|---|---|---|---|
| creative | 0.934 | 0.559 | 0.911 (sd 0.205) | 0.97× | 80% | 5 |
| research | 1.017 | 0.973 | 0.978 (sd 0.142) | 0.96× | 80% | 5 |

Both voices land under the acceptance bar, with generations sitting at roughly
0.97× the self-baseline, and both fool the judge on 4 of 5 held-out trials. I
take the ratio more seriously than the fool-rate. A fool-rate at n=5 is four
coin flips wearing a percentage sign, whereas the ratio is a distance against a
baseline measured from the author's own prose, and it is the number that moves
when something in the loop actually changes. Largest residual per-feature
deviations by mean |z| were type-token ratio 1.70,
short-sentence rate 1.53 and first-person openers 0.85 on creative, against
paragraph length 1.91, mean word length 1.43 and type-token ratio 1.40 on
research. One creative judge trial errored on a transient CLI exit and is
excluded from the fool-rate denominator.

Reproduce for any corpus with `mimesis eval <voice> --held-out 5`.

---

## What was tried and discarded

Three upgrades were specified, built, measured, and then rejected on their own
numbers. They are documented here because the negative results constrain the
design more usefully than the positive ones do, and because a system whose
whole premise is measurement should say what its measurements killed. Full
campaign in `evals/REPORT.md`.

**StyleDistance as the retriever.** Swapping the topic embedder for a
content-independent style embedder changed 91% of retrieved creative anchors,
yet moved fingerprint RMS-z by at most 0.006, hurt the creative judge fool-rate
from 50% down to 25%, and left research a wash. That result surprised me. I had
expected the style embedder to be strictly better anchors and had budgeted the
torch dependency for it. We read the outcome as a grounding loss rather than a
style failure: style-matched anchors are stylistically apt and topically adrift,
so the generation gives up the footing the topic embedder was supplying and pays
for prose it cannot use. If a style embedding is added later
it belongs as a drift scorer and never as the retriever, and the literature
favours LUAR over StyleDistance for that role, with STEB overall scores of 50.82
for LUAR-CRUD against 49.33 for StyleDistance and 36.47 for raw function-word
frequency.

**Cadence features as the generation gate.** Covered above. Better measurement
killed the effect, and the retraction is written up in full.

**A perplexity-ratio detector in the loop.** It detects. It is kept, and it
reports. It never gates, because optimising against a detector optimises against
that detector and nothing else, and the resulting prose is tuned to a
measurement instrument rather than to a reader.

The one upgrade that survived was recalibrating from the accepted set, which is
why `mimesis accept` and `mimesis recalibrate` exist: an author's edits to a
draft are the only supervised signal the system ever receives, and they are
worth more than any additional critic.

---

## Non-goals

No RL finetuning. The loop is an inference-time rewrite, not a weight update. No detector evasion as a product claim, since the scrubber is
framed as authenticity and QA. No cloud anything.

---

## License

Apache-2.0, see `LICENSE`. Design rationale in `docs/DESIGN.md`, cadence
evidence in `evals/CADENCE_FINDINGS.md`, upgrade campaign in `evals/REPORT.md`,
and the operating guide for Claude Code in `CLAUDE.md`.
