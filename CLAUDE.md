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

---

## Setting someone up from scratch

```bash
git clone https://github.com/pskeough/Mimesis_Voice_Clone.git
cd Mimesis_Voice_Clone
bash install.sh                                    # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
mimesis new theirvoice --from ~/path/to/their/writing
```

`install.sh` / `install.ps1` create the venv, install the package, build every
voice that ships with a corpus, register the MCP server, and run a health check.
Both are safe to re-run and skip finished work.

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

---

## Writing in a voice

Prefer the MCP tools (`compose_in_voice`, `get_voice_guide`,
`retrieve_style_examples`, `retrieve_transform_demos`, `scrub_ai_footprint`)
over the CLI when composing inside a conversation — they return anchors and the
gate protocol without shelling out. `mimesis compose` runs the same loop
autonomously by calling `claude -p`.

Always run `scrub_ai_footprint` on a draft before presenting it, and fix every
flag. The scrubber is a scalpel by design: it strips what is always safe for
*this* author and flags the rest rather than rewriting prose toward the mean,
because mean-collapse kills a voice. "Always safe" is itself per-author — the
em-dash strip only runs for writers whose own corpus does not use prose dashes.

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
- **An older `mimesis` MCP server still registered.** It will keep answering
  voice requests and nothing about that failure looks like a failure. Check
  `claude mcp list` and remove the stale one.

## Optional: the memory system

Mimesis is the voice half of a pair; Claude Mind is the memory half. If both are
installed, set `mimesisProfilesRoot` in the vault's `.claude/lkhs.config.json` to
this repo's absolute `profiles/` path. Its nightly pass then mines stylistic edit
pairs from transcripts and feeds them into recalibration. Only stylistic
corrections are used; substantive ones never touch voice artifacts. Without the
key the loop stays dormant and nothing breaks.
