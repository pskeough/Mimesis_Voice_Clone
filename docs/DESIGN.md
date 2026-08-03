# Mimesis v2 — Design

One voice engine, N voices, discriminator in the loop. Companion system to ClaudeMind (LKHS): ClaudeMind is memory, Mimesis is voice. Personal instance runs Patrick's voices; the repo ships clean with zero personal data by construction.

## Thesis (grounded)

Prompt-only voice imitation fails stylometric verification (arXiv:2603.29454, 2509.14543). Generation with a discriminator in the loop succeeds (StealthRL 2602.08934, AuthorMist 2503.08716). But evasion tuned against perplexity-family detectors leaves a residual style fingerprint that style-embedding detectors still catch (2505.14608). Therefore compose is a generate-score-rewrite loop whose critic is a stylometric fingerprint, with the AI-footprint scrubber and a fidelity audit as hard gates. This was independently confirmed empirically in an earlier external-corpus eval: judge-free fingerprint selection beat LLM-judge selection (0.939 vs judge-preferred candidates scoring worse than blueprint-only).

**Caveat on 2603.29454, added after reading it at source.** It tests GPT-4o only, prompting attacks only (no DIPPER/STRAP-class paraphrase machinery), and short informal registers only (Enron email, BOLT SMS, Twitter). The defensible statement is "naive prompting-based impersonation fails forensic AV in short-form registers, on one model." It does not license a general claim about long-form prose, and this design should not lean on it harder than that.

**A second critic (style-embedding distance) was specified here and is deliberately not implemented.** Measured on this repo's own corpora, swapping the topic embedder for StyleDistance changed 91% of retrieved creative anchors yet moved fingerprint RMS-z by <=0.006, *hurt* the creative judge fool-rate (50%->25%), and left research a wash. Style-matched anchors are stylistically apt but topically adrift, and the generation loses the grounding the topic embedder supplied. If a style embedding is added later it belongs as a drift *scorer*, never as the retriever, and the literature favours LUAR over StyleDistance for that role (STEB overall: LUAR-CRUD 50.82, StyleDistance 49.33, raw function-word frequency 36.47).

## What the fingerprint can and cannot see

Every feature in the original 13 is an **order-invariant aggregate** -- a mean, a stdev, a percentage, a per-100-word rate. Shuffle a text's sentences and all thirteen are unchanged; sort them shortest-to-longest into a monotone ramp no writer produces and they are *still* unchanged (measured: +0.0023 mean RMS-z, 0/93 pieces flagged). So the fingerprint is blind to cadence by construction, and a gate selecting on it constrains vocabulary and density while leaving rhythm free.

`cadence.py` supplies 13 order-aware features (length autocorrelation, successive-delta magnitude, direction changes, run lengths, class-transition entropy, P(short|previous long), paragraph open/close ratios, clause rhythm) as `feature_set: "v2"`. On held-out data v2 separates real prose from rhythm-destroyed prose at AUROC 1.000 where v1 sits at chance.

**It is a diagnostic, not the default gate.** As a generation gate it did not earn its place: on a correctly segmented ruler the A/B effect fell from -0.613 (7/7, p=0.016) to -0.174 (5/7, p=0.45), and blind discrimination showed no difference on one voice (25% vs 25% fool-rate) and favoured v1 on another (87.5% vs 37.5%, Fisher p=0.12). Selecting on 26 features plausibly over-constrains the slate: hitting more statistical targets is not the same as reading better. Run `mimesis audit VOICE` to see whether a voice has a rhythm gap at all; switch it to v2 only if an A/B on that voice earns it. Full evidence and the retraction: `evals/CADENCE_FINDINGS.md`.

## Lineage / what ports from where

- Shell, profile registry, MCP tools, RRF hybrid retrieval and ONNX-light ingest carry over from the author's own earlier style-clone MCP.
- StyleDistance embeddings, contrastive transform-demo anchoring and the fabrication/fidelity audit carry over from the author's own research voice-clone RAG work.
- The following were REIMPLEMENTED CLEAN for this project, from technique rather than from source, with no file copies and no third-party data of any kind: the 13-feature stylometric fingerprint with RMS-z distance, MMR-diverse anchor retrieval, slate generation ("verbalized sampling"), the two-stage gate (fingerprint gate then quality-select), the scalpel scrub, and calibration bands with positive exemplars.

## Repo

License Apache-2.0. Distribution name `mimesis-voice` (PyPI `mimesis` is taken by the fake-data library). Python >= 3.11, venv-local, fully offline by default.

```
Mimesis/
  pyproject.toml            # deps: fastmcp, fastembed, numpy; extras [style]: sentence-transformers+torch (StyleDistance); extras [detect]: detector deps
  README.md                 # portfolio-grade: thesis, architecture diagram, eval results, quickstart
  LICENSE                   # Apache-2.0
  .gitignore                # profiles/* (except profiles/example), *.sqlite, local_cache/
  src/mimesis_voice/
    server.py               # FastMCP stdio server; tools: get_voice_guide, retrieve_style_examples,
                            #   retrieve_transform_demos, compose_in_voice, scrub_ai_footprint, eval_voice
    config.py               # profile registry + resolution; voice x format grid; state.json (active voice)
    ingest.py               # docx/txt/md -> ~180-200w chunks -> sqlite (chunks + FTS5 + vectors); hash-skip
    embed.py                # backend abstraction: 'fast' = bge-small ONNX (default); 'style' = StyleDistance
                            #   (optional extra); per-profile choice in config.json; matrix cache keyed on db mtime
    retrieve.py             # RRF (vector + FTS5, k=60) -> MMR diversification (lambda=0.65); transform-demo
                            #   retrieval for profiles with pairs
    cadence.py              # 13 ORDER-AWARE features (feature_set "v2"): sentence-length lag-1/lag-2
                            #   autocorrelation, normalized successive deltas, direction-change rate,
                            #   same-class run length, 3x3 class-transition entropy, P(short|prev long),
                            #   paragraph open/close ratios, clause length/CV/count. Classes cut at the
                            #   AUTHOR'S OWN terciles, not fixed word counts. Ships plain-language rewrite
                            #   hints per feature; "sl_autocorr1 z=+2.3" is not actionable by a model.
                            #   Diagnostic by default -- see "What the fingerprint can and cannot see".
    fingerprint.py          # 13-feature stylometric fingerprint: sentence-mean/stdev(burstiness), pct short(<8w),
                            #   pct long(>30w), colon/semicolon/em-dash per 100w, question rate, comma per sentence,
                            #   paragraph length, first-person openers, TTR, mean word length.
                            #   calibrate(corpus) -> means/stds; distance(text) -> RMS-z vs corpus
    gate.py                 # compose loop: slate -> scalpel+score EVERY candidate -> fingerprint gate ->
                            #   Pareto frontier over (voice fidelity, scrub hard-flags) -> quality-select.
                            #   Candidates are scrubbed BEFORE scoring: the old order scored a draft, picked
                            #   it, then edited it away from the point it was picked for and patched the
                            #   damage with a repair rewrite. Rewrites run concurrently (4 workers).
                            #   rmsz_max accepts "auto" = the profile's own calibrated p95; a hardcoded
                            #   ceiling is a claim about one corpus and broke immediately on the second.
                            #   (legacy note) slate of 3-5 candidates -> fingerprint gate (RMS-z <= threshold,
                            #   default 1.1) -> optional style-embedding distance check -> quality-select among
                            #   survivors -> if zero survivors, targeted rewrite pass citing the failing features,
                            #   max 2 iterations -> scalpel scrub last
    scrub.py                # merged scrubber, three tiers:
                            #   1) hard rules: em-dash zero, corpus-calibrated banlist (~68 words/~35 phrases)
                            #      minus per-author whitelist, burstiness floor, hedge ceiling
                            #   2) fidelity audit: numbers/citations/acronyms extracted from source vs output;
                            #      flag dropped/altered/invented (critical for research voice)
                            #   3) optional [detect]: Binoculars-style perplexity ratio as reporting signal only
                            #   scalpel mode: strip/flag, never mean-enforce (hard-won: mean-enforcement collapses voice)
    evalcli.py              # discrimination eval: mix N held-out real pieces with generations (leak controls:
                            #   neutral brief derived from real piece; exclude piece + 3 nearest neighbors from
                            #   anchors); judge via `claude -p` picks the real one; report fool-rate, fingerprint
                            #   RMS-z distribution vs real-corpus self-baseline, per-feature diffs
    cli.py                  # mimesis profile new|list|use / ingest / calibrate / compose / scrub / eval / doctor
  profiles/
    example/                # checked-in template: config.json, VOICE.md guide skeleton, empty source_documents/
    (patrick voices live here locally, gitignored)
  docs/DESIGN.md            # this file
  scripts/register-mcp.ps1 / .sh   # adds server to claude mcp config as 'mimesis-v2' (does NOT touch existing 'mimesis')
```

## Profile model

A profile = voice x optional format cell. `profiles/<voice>/config.json`:

```json
{
  "author_name": "...",
  "embed_backend": "fast|style",
  "gate": {"rmsz_max": 1.1, "slate_size": 4, "max_rewrites": 2},
  "formats": {"essay": {...overrides...}, "post": {...}},
  "anchors": {"exemplars": true, "transform_pairs": "pairs/pairs.jsonl|null"},
  "whitelist": ["banned words this author actually uses"]
}
```

Fingerprint and scrubber calibration are per-profile artifacts (`data/fingerprint.json`, `data/scrub_calibration.json`) produced by `mimesis calibrate`.

## Shaping a voice: three worked configurations

Profiles are per-user data and are never committed (`profiles/*` is gitignored
except the synthetic `example`). These are the shapes worth copying, not
anyone's actual corpus.

| voice | corpus source | anchors | embed |
|---|---|---|---|
| research | a set of AI-to-author rewrite pairs, plus the author's own preprint prose | transform pairs primary, exemplars secondary | style |
| creative | the author's canonical creative work, excluding anything AI-assisted | exemplars | style if available, else fast |
| personal | informal writing: notes, letters, messages | exemplars | fast |

Two rules generalize from these. Keep AI-assisted text out of a corpus meant to
teach a human voice, or the fingerprint learns the assistant. And keep forms
apart: a profile blending long-form work with short posts measures format
rather than voice.

## MCP integration

Server name `mimesis-v2` during transition, so an existing `mimesis` registration keeps working until cutover. The global CLAUDE.md voice protocol continues to work unchanged after cutover because tool names are identical (plus new `eval_voice`).

## Eval acceptance bar (before claiming done)

Per voice: (1) `mimesis doctor` green; (2) compose round-trip produces a draft passing gate + scrub; (3) discrimination eval over >= 5 held-out pieces reports fool-rate and fingerprint distribution, generated RMS-z within 2x of the corpus self-baseline. Results table goes in the README as a worked demo, reproducible against any user's corpus.

## Non-goals (v2)

RL finetuning (the loop is inference-time rewrite, not weight updates), detector evasion as a product claim (scrubber is framed as authenticity/QA), cloud anything.
