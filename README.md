# Mimesis

**One voice engine, N voices, a discriminator in the loop.** Mimesis clones a
writer's voice from their own corpus and holds every draft to a measured
stylometric standard before it ships. It is the voice half of a two-system pair:
a companion memory system is the *what you know*, Mimesis is the *how you sound*.

The repo ships clean with **zero personal data by construction** — the only
checked-in profile is a synthetic public-domain-style example you can run
immediately.

---

## Thesis

Prompt-only voice imitation fails stylometric verification: ask a model to "write
like X" and a classifier still tells the imitation from the real thing. Generation
with a *discriminator in the loop* closes most of that gap, but a footprint
survives that style-embedding detectors still catch. So Mimesis does not trust the
model's taste. It **generates a slate, scores every candidate on a 13-feature
stylometric fingerprint, and selects on that distance**, demoting the LLM judge to
a secondary quality signal and adding an AI-footprint scrubber and a fidelity audit
as hard gates.

The design bet — judge-free stylometric selection beats LLM-judge selection — was
confirmed empirically in prior work before it was hardened here: fingerprint-chosen
candidates scored better than the judge-preferred ones.

---

## Architecture

```
 corpus ──ingest──> SQLite store ──calibrate──> fingerprint.json + scrub_calibration.json
 (docx/txt/md)      (chunks + FTS5 + vectors)         │                    │
                                                       │                    │
 task ─────────────────────────────┐                  ▼                    ▼
                                    │          ┌───────────────┐   ┌────────────────┐
   retrieve (RRF hybrid recall,     └────────► │  compose loop │   │    scrubber    │
   MMR λ=0.65 diversification) ───► anchors ──►│  (gate.py)    │──►│ em-dash/banlist│──► voiced draft
   + transform demos                           │  slate → gate │   │ + fidelity     │
                                               │  → select     │   └────────────────┘
                                               │  → rewrite    │
                                               └───────────────┘
```

- **ingest** — `.docx/.txt/.md` → ~180-word chunks → SQLite (chunk table, FTS5
  keyword index, embedding blobs). Hash-skips an unchanged corpus.
- **embed** — `fast` (bge-small ONNX, default, no torch) or `style` (StyleDistance,
  optional `[style]` extra). Corpus matrices cached and mtime-keyed.
- **retrieve** — Reciprocal Rank Fusion of vector + FTS5 recall (k=60), then
  Maximal Marginal Relevance (`score = 0.65·sim − 0.35·max_overlap`) so anchors are
  diverse, not five paraphrases. Profiles with `pairs.jsonl` also get contrastive
  AI→author transform demonstrations.
- **fingerprint** — 13 surface features (burstiness, short/long-sentence rates,
  colon/semicolon/em-dash density, question rate, commas per sentence, paragraph
  length, first-person openers, TTR, mean word length), z-scored against a
  per-author calibration and collapsed to one RMS-z distance, with an honest
  leave-one-out corpus self-baseline.
- **gate** — slate of 3–5 genuinely different candidates (verbalized sampling) →
  fingerprint gate (RMS-z ≤ threshold) → quality-select among survivors → on zero
  survivors, one targeted rewrite per candidate citing the worst features (≤2
  iterations) → scalpel scrub last.
- **scrub** — three tiers: hard rules (em-dash zero, corpus-calibrated banlist
  minus a per-author whitelist, burstiness floor, hedge ceiling); a fidelity audit
  (numbers/citations/acronyms in source vs output → dropped/altered/invented); and
  an optional perplexity-ratio detector signal (reporting only). Scalpel mode
  strips the always-safe things and *flags* the rest; it never rewrites prose to
  force the mean, because mean-collapse kills the voice.

Generation shells out to the local `claude` CLI (`-p` headless, default model
`sonnet`), so no API keys or cloud services are required beyond a working Claude
Code install. Everything else runs fully offline.

---

## Quickstart

```bash
git clone https://github.com/pskeough/Mimesis_Voice_Clone.git
cd Mimesis_Voice_Clone

python -m venv .venv
# Windows: .venv\Scripts\activate    POSIX: source .venv/bin/activate
pip install -e .                     # base install (fast backend, no torch)

mimesis profile list                 # the synthetic 'example' voice ships ready
mimesis ingest example               # build the store (downloads bge-small once)
mimesis calibrate example            # write fingerprint.json + scrub_calibration.json
mimesis doctor                       # self-test; expect all green

# Autonomous generate-score-rewrite pass (calls claude -p):
mimesis compose example "write a short note about morning routines"
# Exercise the whole pipeline without generation:
mimesis compose example "..." --dry-run

mimesis scrub example "some draft text"          # vet a draft
mimesis eval example --fingerprint-only          # fingerprint distribution, no generation
```

Add your own voice:

```bash
mimesis profile new myvoice --backend fast       # or --backend style (needs [style] extra)
# drop .docx/.txt/.md into profiles/myvoice/source_documents/
mimesis ingest myvoice && mimesis calibrate myvoice
mimesis compose myvoice "a two-paragraph bio"
```

### As an MCP server

`scripts/register-mcp.ps1` / `.sh` register a stdio server named **`mimesis-v2`**
with Claude Code (user scope). They are write-only helpers — run them yourself
when ready; they never touch any existing `mimesis` registration. Tools:
`get_voice_guide`, `retrieve_style_examples`, `retrieve_transform_demos`,
`compose_in_voice`, `scrub_ai_footprint`, `eval_voice`.

---

## Install extras

| extra | pulls | enables |
|---|---|---|
| _(base)_ | fastmcp, fastembed, numpy, python-docx | full engine, `fast` embeddings |
| `[style]` | sentence-transformers, torch | StyleDistance `style` backend |
| `[detect]` | transformers, torch | perplexity-ratio detector signal (stub: reports "not installed" until `detect.py` ships) |

---

## Eval results

The discrimination eval mixes held-out real pieces with generations under leak
controls (a neutral brief derived from the real piece; the piece and its three
nearest neighbours excluded from the anchors), asks a `claude -p` judge which text
is the real author, and reports fool-rate plus the generated RMS-z distribution
against the corpus self-baseline.

Acceptance bar: generated RMS-z within **2×** the corpus self-baseline.

Runs below are `mimesis eval <voice> --held-out 5` over a real personal corpus
(`fast` backend, `sonnet` generator/judge). Voice names are generic; no corpus
content is shown.

| voice | corpus self-baseline RMS-z | held-out real mean RMS-z | generated mean RMS-z | ratio | fool-rate | n |
|---|---|---|---|---|---|---|
| creative | 0.934 | 0.559 | 0.911 (sd 0.205) | 0.97× | 80% | 5 |
| research | 1.017 | 0.973 | 0.978 (sd 0.142) | 0.96× | 80% | 5 |

Both voices land under the 2× acceptance bar (generations sit at ~0.97× the
self-baseline) and fool the `claude -p` judge on 4 of 5 held-out trials. Largest
residual per-feature deviations (mean |z|): creative — TTR 1.70, short-sentence
rate 1.53, first-person openers 0.85; research — paragraph length 1.91, mean word
length 1.43, TTR 1.40. One creative judge trial errored on a transient CLI exit
and is excluded from the fool-rate denominator.

Reproduce for any corpus: `mimesis eval <voice> --held-out 5`.

---

## License

Apache-2.0. See `LICENSE`. Design rationale in `docs/DESIGN.md`.
