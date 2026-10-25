# AI detectors, and editing your own writing

Mimesis is a voice engine, not a detector-evasion tool. This document covers
two narrower things that follow from one finding about how a current detector
reads research prose:

1. **Editing text you wrote** without turning it into model text
   (`edit_own_text` + `apply_own_edits`, CLI `mimesis own-edit`).
2. **A content-risk report** that marks the sentences of a research draft
   carrying the kinds of content the detector labels AI, so the author can
   write those sentences themselves (`contentrisk.py`).

Neither one gates a draft, and neither treats a detector score as the goal.
The point is narrower: when a person wrote a text, routine help with it should
not make it stop being theirs.

---

## What the detector reads

The evidence is a detection study run in October 2026 on research prose:
about 800 Pangram 4.1 scans, mixing one research author's unpublished paper
sections, model drafts and model edits of them, and pre-2022 open-access
papers as certain-human controls. Pangram returns a label per sentence span
and scores the text in 512-token windows at a stride of 256.

**The label follows the content of a sentence, not its wording.** Pangram
labels a sentence by whether its content came from a model. Rewording does
not move that label; keeping the author's sentences does.

| what was done to the text | result |
|---|---|
| model polish of human text, by achieved token similarity to the original | >= 0.90: 12/12 Human. 0.85-0.90: 14/16. 0.80-0.85: 7/9. 0.75-0.80: 3/5. < 0.75: 1/7 |
| the same polish at section length (about 1,000 words) | 12/12 Human at 0.90 |
| attribution-only fixes (similarity 0.96-0.996) | 7/8 Human |
| the author's own sections plus word fixes inside the author's sentences | 100% human |
| the same sections with sentences restructured and one new sentence added | 100% AI |
| model sentences that passed the voice gate, spliced into the author's text | 66% AI |

Two consequences shape the tools below.

- **One caught sentence costs a window.** Because labels are window-level, a
  single model-built sentence flipped about 280 unchanged surrounding words.
  Under about 460 words, one caught sentence can take the whole text.
- **Some content types read AI in any wording.** Interpretation of the paper's
  own results, ranking the work against prior work, contribution and gap
  claims, roadmaps, glosses that restate the previous sentence, consequences
  drawn about the study's own design, and specification content (prompt
  templates, JSON formats, formal definitions) were labelled AI across every
  wording tried. Reports of what cited studies did and found, concrete run
  facts, design differences tied to a named study, and aims or hypothesis
  statements read Human. Word bans, pronoun and stance changes, sentence length
  and recasting the paper's own work as a citation did not help.

The polish rows in the table use public papers, not the study author's own
prose. Polish of unpublished author prose is untested beyond the section-level
rows above.

---

## Editing your own text

Every compose path in Mimesis writes new sentences. That is right for a new
piece and wrong for a draft the author already wrote: a voicing pass over the
author's own abstract returned 95-99.5% word-identical output, and the parts it
did change are the parts that read as model text.

`edit_own_text` returns the draft split into numbered sentences, which ones
carry a hard flag, an advisory flag or a content-risk tag, and the rules for
proposing edits. The host model proposes edits as
`[{"n": 3, "text": "...", "fixes": ["hard:self-explanation"]}]`, and
`apply_own_edits` applies them under limits enforced in code:

| limit | value |
|---|---|
| shape of an edit | one sentence replaced by one sentence; no merges, splits, inserts, deleted sentences or line changes |
| words kept per sentence | at least 60% of the original words, in order |
| words added per sentence | at most 6 |
| which sentences may change | `hard` (default): hard-flagged only. `flagged`: any flagged sentence. `goal`: any sentence, for a stated goal |
| facts | numbers, citations and citation keys cannot change; no dashes added where the voice has none |
| new problems | an edit that adds a hard flag, a banned word or a content-risk move is refused |
| whole text | token similarity to the original at least 0.93 (configurable, never below 0.90) |
| every detector-sized window | token similarity at least 0.85 |

When the whole text or a window falls under its floor, edits are dropped,
least necessary first, until both hold. Similarity is `difflib` over
whitespace tokens with autojunk off, the measure behind the evidence table.
The result is the edited text, a word diff, the list of applied and refused
edits with reasons, and a provenance line: every sentence is still the
author's.

The default scope is `hard` because most advisory flags, on a real author's
corpus, mark that author's own habits. Fixing them would move the text away
from the author.

```bash
mimesis own-edit myvoice draft.md                         # the kit: numbered, flagged sentences
mimesis own-edit myvoice draft.md --ops edits.json --out fixed.md
mimesis own-edit myvoice draft.md --generate --out fixed.md   # claude -p proposes, the code decides
```

What it will not do: write a missing sentence. When a draft needs a new claim,
the author writes it, or a gated `compose_slate` candidate is used knowing it
is model text and may be read as such.

---

## The content-risk report

The edit kit and the edit result always carry a `DETECTION RISK` block. For
voices with `"guidance": "research"` in `config.json`, the scrub report carries
it too, and the research compose kit adds a short card describing the same
content types. The block lists sentences
that carry an AI-read move, runs of three or more of them, windows over half
risky, and citation density beside reference values from the study. It is
reporting only. Nothing in it enters the gate, the ranking or a rewrite prompt.

The tags are surface patterns over the moves named above, so they need
checking against real labels. `evals/pangram_content_check.py` does that
offline: within each scanned document that has both Human and AI sentences, are
tagged sentences labelled AI more often than untagged ones? Within-document is
the only fair comparison, because across documents a tag that merely separates
sources would look like a detector. The statistic is a Mantel-Haenszel odds
ratio stratified by document.

On the study's 773 scans (255 documents with both labels, 8,685 sentences):

| tag | n | MH odds ratio [95% CI] |
|---|---:|---|
| any risk class | 710 | 1.63 [1.36, 1.95] |
| specification | 70 | 2.21 [1.17, 4.16] |
| interpretation | 123 | 1.73 [1.15, 2.61] |
| roadmap | 115 | 1.63 [1.09, 2.44] |
| concept exposition | 196 | 1.63 [1.16, 2.31] |
| obligation | 11 | 16.0 [2.2, 118] |
| gloss, ranking, gap, contribution | 17-127 | 1.17-1.37, intervals cross 1 |
| *safe:* cited-report | 490 | 0.50 [0.39, 0.63] |
| *safe:* run fact | 3,151 | 0.81 [0.73, 0.90] |
| *safe:* aims | 38 | 0.12 [0.05, 0.29] |

Labels are window-level, so neighbouring sentences share them and these ratios
are diluted by construction; read them as lower bounds. On the older half of
the archive alone the pooled ratio was 1.29 [0.95, 1.74]. The report is a
reading aid. No offline detector tried in the study (a fine-tuned tagger, an
LLM judge, Binoculars, desklib) tracked Pangram forward in time, and this one
has not been tested forward either.

### Checking it on your own scans

The script reads one of two JSONL formats and writes aggregates only, never
sentence text:

```jsonc
// one row per sentence
{"item": "doc-1", "text": "The sentence.", "y": 1}          // y = 1 when labelled AI

// or one row per scan, labels as character spans of `text`
{"scan_id": "s-1", "text": "Full scanned text...",
 "sentence_labels": [{"start": 0, "end": 42, "label": "human"}], "round": 3}
```

```bash
python evals/pangram_content_check.py my_scans.jsonl --out evals/pangram_content/RESULTS.md
```

`evals/pangram_content/` is gitignored, since scan archives hold unpublished
text.

---

## Voice fit is not detection

The fingerprint gate measures distance from the author's corpus. It does not
predict detector labels: model sentences that passed the gate read 66% AI when
spliced into the author's own text. A small-model Binoculars score separated
real from generated prose at AUROC 0.65-0.68 here (`evals/REPORT.md`). A clean
scrub report means the draft avoids constructions this author rejected; it is
a statement about taste, not about detection.

Only a real scan answers a detection question. Scans cost credits, so decide
what a scan is testing before running it, and keep detectors out of selection:
choosing drafts by detector score tunes prose to the detector.

---

## Limits

- One research author, one genre and one detector version (Pangram 4.1).
  Other detectors and later versions may read text differently.
- The risk classes are English regular expressions over sentence surfaces.
  They miss paraphrased moves and tag some innocent sentences.
- Window sizes assume about 0.75 words per token.
- The edit limits come from the polish evidence above. They keep an edit small;
  they do not certify that the edited text will scan as human.
