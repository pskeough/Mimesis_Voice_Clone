# Upgrading an older Mimesis install

For a machine that already has an earlier Mimesis and a working voice profile
built from its own corpus. First-time installs should follow the Quickstart in
`README.md` instead.

Hand this file to Claude Code in the Mimesis folder:

> Read UPGRADE.md and upgrade this install, then rebuild my voice profile and
> verify it against my own corpus.

---

## Read this first: it is a rebuild, not a migration

There is no schema version and no migration code in this project, by design.
Profiles are cheap to regenerate and expensive to migrate correctly, so the
supported upgrade is: **keep your source documents, throw away everything
derived from them, and rebuild.**

What that means for you:

| Thing | What happens |
|---|---|
| `profiles/<you>/source_documents/` | **Kept.** This is your corpus and the only irreplaceable part. |
| `profiles/<you>/config.json` | Kept, then extended with any new keys. |
| `profiles/<you>/data/*.sqlite` | Rebuilt by `ingest`. |
| `profiles/<you>/data/fingerprint.json` | Rebuilt by `calibrate`. Required — the feature set changed. |
| `profiles/<you>/data/scrub_calibration.json` | Rebuilt by `calibrate`. |
| `profiles/<you>/pairs.jsonl`, `accepted/` | Kept if present. |

The fingerprint **must** be recalibrated. It is a z-scored distance against
per-author feature means, the feature set has changed since older builds, and a
stale `fingerprint.json` does not raise an error — it silently scores every
candidate against the wrong baseline and rejects good drafts.

---

## Coming from the v6 Voice MCP?

If your existing install is the standalone Voice MCP (a `src/` folder with
`stylometrics.py`, `compose.py`, `scrub.py` and a `data/voice_db.sqlite`), you
do not have a `profiles/` directory at all, and Steps 1 and 4 below need one
extra move first: your corpus lives *inside* that SQLite store rather than as
files on disk.

Recover it:

```bash
python scripts/import_v6_voicedb.py /path/to/data/voice_db.sqlite \
    --slug <you> --out profiles --dry-run
```

Drop `--dry-run` once the report looks right. It reassembles each document from
its chunks in order and writes `profiles/<you>-<format>/source_documents/`.

Three things it does that are worth understanding:

- **It reads the `chunks` table, not `voice_cookbook`.** The cookbook looks
  like it holds the originals, but its `passage` column is an excerpt: on a
  real store one document was 57 words there against 53,197 across its chunks.
  Anything preferring the cookbook silently keeps a fraction of the corpus.
- **It splits by format by default.** One fingerprint computed across LinkedIn
  posts and long-form documents together measures format far more than voice,
  which shows up as a self-baseline well above 1.0 and a gate nothing can
  satisfy. v6 calibrated its gate per format for the same reason. `--no-split`
  overrides this if you want one blended profile.
- **It warns when a single document dominates a profile.** A book sitting in
  the same bucket as short pieces does not merely weight the fingerprint, it
  becomes it. Give it its own profile.

Check the reported total against the store itself
(`select sum(word_count) from chunks`) before moving on. They should match
exactly. Then continue from Step 2, treating the recovered `source_documents/`
as your corpus, and back it up as in Step 1.

---

## Step 1 — back up the corpus

Do this before anything else, outside the repo folder.

```bash
cp -R profiles ~/mimesis-profiles-backup-$(date +%Y%m%d)
```

`profiles/` is gitignored (everything except the synthetic `example`), so git
will not protect it for you.

---

## Step 2 — get the new code

If this folder is already a git clone:

```bash
git pull
```

If it is not a clone (an older copy delivered as a folder), clone fresh
alongside it and move your corpus across:

```bash
cd ..
git clone <repo-url> Mimesis-new
cp -R Mimesis/profiles/<you> Mimesis-new/profiles/<you>
cd Mimesis-new
```

Keep the old folder until the new one is verified.

---

## Step 3 — reinstall

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Base install only. The optional extras each pull `torch`, which is a multi-GB
download:

- `pip install -e ".[style]"` — StyleDistance content-independent style
  embeddings. Better anchors, much heavier. Only worth it if the default
  backend is visibly underperforming.
- `pip install -e ".[detect]"` — perplexity-ratio detector signal, reporting
  only.

Stay on the default `fast` backend (bge-small, ONNX, no torch) unless you have
a reason not to.

---

## Step 4 — rebuild the profile

```bash
mimesis ingest <you>
mimesis calibrate <you>
```

`ingest` re-chunks `source_documents/` into the store and hash-skips anything
unchanged. `calibrate` recomputes `fingerprint.json` and
`scrub_calibration.json` from that store.

Check the self-baseline that `calibrate` prints. It is a leave-one-out distance
of the corpus against itself — the floor for what "sounds like you" can mean.
A number well above ~1.0 usually means the corpus is mixing formats (book
chapters together with short posts), and the fix is separate profiles per form
rather than one blended fingerprint that measures format instead of voice.

---

## Step 5 — verify

```bash
mimesis compose <you> --task "a short paragraph about how your week went"
python scripts/portability_audit.py <you>
```

The first should return a draft that passes the gate and the scrubber. Read it:
does it sound like you?

The second is the more important check and is worth running even though it is
not obvious why. Every threshold in a voice engine is either measured from your
corpus or typed by hand by whoever built it, and the hand-typed ones are
invisible until a second author arrives — at which point they fail *silently*
rather than loudly. A hardcoded ceiling tuned to a different writer does not
error; it rejects every candidate on every brief and quietly falls back at
several times the compute cost. The audit reports which of your thresholds are
genuinely calibrated and which are somebody else's constant. If it flags one,
that is a code fix, not something to work around in your config.

---

## Step 6 — re-register the MCP server

So Claude Code picks up the new build:

```bash
mimesis-mcp --help          # confirm the entry point resolves
claude mcp list             # confirm 'mimesis' is registered and pointing here
```

If the path moved in Step 2, re-register it so it points at the new folder.

---

## Optional: reconnect the memory system

If you also run Claude Mind, its nightly job can feed your accepted drafts and
your edits back into this profile's recalibration set. Set in the vault's
`.claude/lkhs.config.json`:

```json
{ "mimesisProfilesRoot": "/absolute/path/to/Mimesis/profiles" }
```

Without that key the voice loop stays dormant and nothing breaks. With it, the
vault's nightly pass mines stylistic edit pairs from your transcripts and
weekly recalibration runs once enough new signal has accumulated. Only
stylistic corrections are used; substantive ones never touch voice artifacts.

---

## If drafts come back wrong

Two failure modes dominate, and they look different:

**Everything is rejected / composition is very slow.** The gate cannot find a
candidate under the distance ceiling. Almost always a stale or mis-fitted
`fingerprint.json` — re-run `calibrate`, then `portability_audit.py` to check
the ceiling is yours and not inherited.

**Drafts pass the gate but do not sound like you.** The fingerprint is
order-invariant: every feature is a mean, a rate or a ratio, so it cannot see
rhythm, and prose can match you on all thirteen surface features while reading
nothing like your actual sentence flow. `evals/CADENCE_FINDINGS.md` documents
this blind spot and what is being done about it. The practical lever is the
corpus: more of it, and split by form.
