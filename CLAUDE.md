# Working in this repo

Mimesis clones a writer's voice from their own corpus and scores every draft
against a measured stylometric fingerprint before it ships. This file is for
Claude Code operating inside this folder. `README.md` explains what the system
is; this explains how to run it for someone.

Use the venv interpreter for everything: `.venv/bin/python` on macOS and Linux,
`.venv\Scripts\python.exe` on Windows. The examples below write `mimesis`, which
is that venv's console script; `python -m mimesis_voice.cli` is identical and
works before the venv is on PATH.

---

## Hard rules

**Never write in someone's voice from a guess.** If the tools are not connected
or the voice is not built, say so. An inferred voice profile is worse than
plain prose, because it is confidently wrong and the person cannot tell you why
it feels off.

**Never commit a corpus.** `profiles/` is gitignored except the synthetic
`example`, and that is a design property, not housekeeping. Someone's writing is
theirs. Before any `git add`, check `git status` and confirm nothing under
`profiles/` other than `example/` is staged. This applies to
`profiles/*/data/*.sqlite` too: the store holds the corpus **in plain text**, so
committing one publishes the writing as surely as committing the documents.

**Never hand-tune a threshold to make a draft pass.** Every number that decides
anything is measured from the author's corpus by `calibrate`. If drafts are
being rejected, the fingerprint is stale or the corpus is wrong; typing a looser
ceiling into `config.json` hides that and breaks the next author.
`scripts/portability_audit.py <voice>` reports which of a voice's thresholds are
genuinely calibrated and which are somebody else's constant.

**The example voice is a demo.** `profiles/example` is eight synthetic pieces. It
exists so the install can prove itself. Never answer a request to write in the
person's voice from it. Until they have a built voice of their own, every MCP
writing tool returns the intake protocol instead of a kit; follow it.

---

## First run: build the person's voice

Run this when the person says "set up my voice", right after a fresh install,
or whenever `list_voices` or a writing tool replies `NO VOICE IS SET UP`. Ask
each question and wait for the answer. Do not guess paths or fill in answers.

1. **Forms.** Ask what they want written in their voice: emails, social posts,
   essays, fiction, academic papers, speeches. Each form becomes its own voice,
   named `<name>-<form>` (`jane-posts`, `jane-essays`). One fingerprint across
   forms measures format, not voice (see "Split by form" below).
2. **Where the writing is.** For each form, ask for folders or files. Supported:
   `.txt .md .docx .pdf`. Common sources and what to do with them:

   | Source | Do this |
   |---|---|
   | Google Docs | File > Download > .docx |
   | Sent email | paste each body into its own .txt (no quoted replies, no signatures) |
   | LinkedIn / X / Substack / Medium export | one .txt or .md per post; a CSV with every post in one file will not split |
   | Scanned PDF | needs OCR first; `inspect` lists PDFs with no text |
   | Pages, .odt, .rtf, .doc | re-save as .docx |

   Target about 40 pieces or 20,000 words per form. 5,000 words is the floor.
3. **Purity.** Ask directly: did anyone else write or heavily edit any of these,
   and were any drafted or polished with ChatGPT, Claude or similar? Leave those
   out. The system calibrates against whatever it is given, so a corpus with
   machine prose in it teaches the voice the machine's habits, and the gate then
   passes them.
4. **Inspect.** `mimesis inspect PATH...` (MCP: `inspect_corpus`). Go through every
   `[STOP]` and `[WARN]` with them: split mixed forms into separate folders, give
   a dominating book its own voice, OCR or drop empty PDFs.
5. **Build.** `mimesis new NAME --from PATH... --author "Their Name"` (MCP:
   `create_voice`). It refuses a corpus under the floor unless `--force`. Read
   them the self-baseline and any `THIN CALIBRATION` line. The new voice becomes
   active when no real voice was.
6. **Purity check.** `mimesis purity NAME` ranks pieces by detector hits. Show
   them every piece with a nonzero score and ask whether it is really theirs.
   Move out the ones that are not, then `mimesis add NAME` to rebuild.
7. **VOICE.md.** Ask these, and write the answers in their own words into
   `profiles/NAME/VOICE.md` under the skeleton's headings:
   - How do you usually open a piece, and how do you end one?
   - Words or phrases you use a lot. Words you would never use.
   - What gives AI-written text away as not you?
   - Do you use em-dashes? If the answer is never, also run
     `python -m mimesis_voice.rhetoric_ext calibrate NAME --dash-zero`.
   - Anything a ghostwriter would need to be told on day one.
8. **Test piece.** Write something short and real with `compose_slate` +
   `gate_candidates` (two candidates from opposed briefs, `voice=NAME`), scrub
   it, show them the winner and the scrub report, and ask whether it sounds like
   them. Record the answer with `record_preference`.
9. **Routing, only with a yes.** Offer to add the block in `docs/GLOBAL_CLAUDE.md`
   to their `~/.claude/CLAUDE.md`, so "write this in my voice" in any folder uses
   these tools. Show them the exact text first. Never edit that file without an
   explicit yes.

---

## Setting someone up from scratch

```bash
git clone https://github.com/pskeough/Mimesis_Voice_Clone.git
cd Mimesis_Voice_Clone
bash install.sh                                    # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
mimesis inspect ~/path/to/their/writing
mimesis new theirvoice --from ~/path/to/their/writing --author "Their Name"
```

`install.sh` / `install.ps1` create the venv, install the package (plus the
optional spaCy tagger; `--no-structure` / `-NoStructure` skips it), build every
voice that ships with a corpus, register the MCP server as `mimesis`, and run a
health check. `--skip-mcp` / `-SkipMcp` leaves Claude Code's config alone. Both
are safe to re-run and skip finished work. If a server named `mimesis` is
already registered from somewhere else, registration stops and says so rather
than adding a second voice server; `MIMESIS_MCP_NAME` picks another name.

`mimesis new` walks the folder recursively, copies every `.txt/.md/.docx/.pdf`
into the profile, ingests, and calibrates. Nested paths are folded into
filenames so `essays/2024/rain.txt` and `notes/rain.txt` both survive. Re-running
it is idempotent by content hash. To add more writing later:

```bash
mimesis add theirvoice --from ~/more/writing        # imports and rebuilds
mimesis add theirvoice                              # rebuild after dropping files in by hand
```

Always rebuild after adding documents. The fingerprint is a z-score against
corpus means, so new writing moves the baseline, and a stale `fingerprint.json`
does not raise — it silently scores every candidate against the wrong writer.

### What to tell them to hand over

Anything they actually wrote, in the register they want back. Email, published
essays, LinkedIn posts, chapters, letters. Not things they edited, co-wrote, or
prompted a model into. A corpus with borrowed prose in it produces a voice that
is partly someone else's, and nothing downstream can detect that.

Aim for **~40 pieces** at the length they normally write. `calibrate` prints a
`THIN CALIBRATION` warning below roughly three pieces per feature; take it
seriously, since standard deviations from eight pieces are noise and the gate
inherits that noise.

### Hand-authored notes

`profiles/<voice>/VOICE.md` is the one hand-written slot per voice, and the
compose loop puts it in front of the measured rules. Everything else is an
average over the corpus, so it can only express habits that survive averaging.
Put the things no statistic reaches there: how they open a piece, moves they
reach for, anti-patterns to avoid even when the fingerprint would pass them.
A profile still carrying the generated skeleton is treated as having no notes,
so partial filling is fine and there is no cost to leaving it alone.

### Split by form, not by topic

One fingerprint computed across short posts and book chapters measures **format**
far more than voice. It shows up as a self-baseline well above ~1.0 and a gate
nothing can satisfy. Give each form its own voice — `jane-posts`,
`jane-essays` — rather than one blended profile. Same for one very long document
in a mixed corpus: a book beside short pieces does not weight the fingerprint, it
becomes it.

---

## Checking it worked

```bash
mimesis doctor theirvoice                     # self-test
mimesis compose theirvoice "a short note about how their week went"
scripts/portability_audit.py theirvoice       # are the thresholds theirs?
mimesis eval theirvoice --held-out 5          # fool-rate + RMS-z vs self-baseline
mimesis audit theirvoice                      # cadence gap: is the rhythm unlike them?
```

Read the self-baseline `calibrate` prints. It is a leave-one-out distance of the
corpus against itself, so it is the floor for what "sounds like them" can mean.
Generated drafts within **2×** that number is the acceptance bar.

Then read a draft. The fingerprint cannot see rhythm — every one of its thirteen
features is an order-invariant aggregate, so shuffled sentences score identically
to real prose. A draft can match on all thirteen and still not read like them.
That blind spot is documented in `evals/CADENCE_FINDINGS.md`; the practical lever
is more corpus, split by form.

### The structure block

`gate_candidates` and `scrub` print six further lines — coherence, openers,
paragraphs, cadence, vocabulary, and four constructions — from
`measures/`. They are the order- and architecture-sensitive half the
fingerprint cannot see, adopted from a nine-author, 4,392-cell study;
`docs/STRUCTURE_GATES.md` is the adoption record and every threshold's evidence.

They **do not gate**. Nothing there enters the ranking or the Pareto front, and
the hard flags are unchanged. What they do feed is the rewrite that was going to
happen anyway: `gate.compose` hands the findings to the fingerprint rewrite
prompt and runs one bounded structural repair afterwards, keeping it only if it
removed findings without pushing the fingerprint out of the gate, adding a hard
flag, or breaking source fidelity.

Four rules to keep in mind:

- **Never average them.** Coherence and voice move in opposite directions under a
  fine-tune, and the study lost weeks to three coherence features hidden inside
  an averaged index.
- **The bands come from the author's source documents, not the store.**
  `read_pieces` rejoins ~180-word retrieval chunks with blank lines, so every
  "paragraph" in a store text is one chunk. Calibrated that way this repo's own
  creative voice reported a paragraph CV of 0.06 against a true 0.58. The same
  trap applies to any text you paste in that came out of the store or the
  accepted set: the paragraph line will be reading the chunker.
- **A line that fires on the author's own prose is measuring the author.** Each
  line prints its measured false-positive rate on that corpus when it fires.
- **Only the machine direction reaches a rewrite prompt.** The report is
  two-sided; `repair_notes` is not. The study's reader preferred the draft that
  over-varied its paragraphs, so a prompt telling a model to even them out would
  optimise toward the failure. Do not "fix" that asymmetry.

Rebuild the bands on an existing voice without touching anything else:

```bash
mimesis calibrate theirvoice --structure-only
```

---

## Writing in a voice

The MCP server exposes:

| Tool | Use |
|---|---|
| `list_voices` | which voices exist, which is active, whether setup is needed |
| `inspect_corpus`, `create_voice` | the first-run intake above |
| `compose_slate` + `gate_candidates` | anything that matters: the host writes candidates, the gate scores and picks |
| `compose_in_voice` | quick drafts; returns a kit and a protocol with no discriminator behind it |
| `get_voice_guide`, `retrieve_style_examples`, `retrieve_transform_demos` | anchors on their own |
| `scrub_ai_footprint` | the final pass on every draft |
| `eval_voice` | fingerprint distribution for a voice |
| `record_preference` | the person kept, rejected or corrected a draft |

`mimesis compose` runs the whole loop headless by calling `claude -p`.

Four rules, each from a measured failure:

- **Pass `voice=` on every call**, including `scrub_ai_footprint` and
  `gate_candidates`. Without it the call scores against the active voice, and a
  piece fitted to the wrong voice passes every check. When reporting a fit
  number, say which voice it was measured against.
- **Two candidates from opposed briefs.** Before `compose_slate`, pick two
  structural approaches that pull apart (claim first vs case first; scene first
  vs argument first) and write one candidate for each. Four candidates off one
  brief collapsed to 19-55% of the author's own spread across about twenty gated
  sections; two off opposed briefs reached 86-148%. If the gate reports
  `COLLAPSED`, the briefs were not different.
- **Follow the protocol `compose_in_voice` returns**: six one-line observations
  of the passages (sentence length, clause structure, punctuation, stance,
  concreteness, endings), then write from the observations without looking back
  at the passages. Present the observations, the draft and the scrub report.
- **The hard flags are not negotiable.** `self-explanation`, `unheeded-reversal`,
  `way-simile` and `declarative-rating` were rejected by name in blind testing
  and sit at or near zero in human prose. `docs/DETECTORS.md` has the evidence.

Always run `scrub_ai_footprint` on a draft before presenting it, and fix every
flag. The scrubber is a scalpel by design: it strips what is always safe for
*this* author and flags the rest rather than rewriting prose toward the mean,
because mean-collapse kills a voice. "Always safe" is itself per-author — the
em-dash strip only runs for writers whose own corpus does not use prose dashes.

### Genre guidance packs

A profile with `"guidance": "research"` in `config.json` gets a third block in
the kit, `## WHAT THIS PASSAGE MUST DO`, between the style rules and the corpus
anchors. It says what the passage has to accomplish and in what order, one card per
rhetorical move (`src/mimesis_voice/guidance/research/<move>.md`), selected by
`format=` when it names a move and by `retrieve.infer_move(task)` otherwise. The
author's own rulings for that move live in `profiles/<voice>/guidance/<move>.md`
and are appended under the shipped card. `docs/RESEARCH_WRITING.md` is the
consolidated source the cards are sliced from; edit there first.

The block exists because the gate cannot see genre: a Discussion at fit 0.201 with
no hard flags was rejected as "a bunch of nothing". It is built to lose to the
voice block on phrasing. Cards are capped at 1,400 characters and never name words
to use; `tests/test_guidance.py` enforces both. Without the config key the kit is
unchanged.

When the person edits a draft, record it. That is the only supervised signal the
system gets:

```bash
mimesis accept theirvoice edited.txt --from draft.txt --task "the brief"
mimesis recalibrate theirvoice          # folds accepted work into the fingerprint
```

---

## Upgrading an existing install

`UPGRADE.md` is the procedure, and it is a **rebuild, not a migration**: keep
`source_documents/`, throw away everything derived from it, re-ingest and
recalibrate. There is no schema version and no migration code, deliberately.

Two cases need care:

- **Coming from the v6 standalone Voice MCP** (a `src/` with `stylometrics.py`
  and a `data/voice_db.sqlite`, no `profiles/` directory): the corpus lives
  inside the SQLite store. Recover it with
  `scripts/import_v6_voicedb.py` before doing anything else. Read that script's
  docstring first; it explains why it reads `chunks` and not `voice_cookbook`.
- **A second voice server still registered.** An old `mimesis-v2`, or a v6
  `<author>_voice`, keeps answering voice requests and nothing about that
  failure looks like a failure. The installer removes a `mimesis-v2` that points
  at this folder; check `claude mcp list` for anything else and remove it.
- **A store built before October 2026.** Re-ingest with `--force`: two chunker
  bugs (a truncated long document, YAML front matter read as prose) are fixed,
  and a plain `ingest` hash-skips an unchanged corpus.

## Optional: the memory system

Mimesis is the voice half of a pair; Claude Mind is the memory half. If both are
installed, set `mimesisProfilesRoot` in the vault's `.claude/lkhs.config.json` to
this repo's absolute `profiles/` path. Its nightly pass then mines stylistic edit
pairs from transcripts and feeds them into recalibration. Only stylistic
corrections are used; substantive ones never touch voice artifacts. Without the
key the loop stays dormant and nothing breaks.
