# Research writing: the master guidance behind the `research` genre pack

This is the human-readable consolidation. The compose kit never sees this file; it
sees one card at a time from `src/mimesis_voice/guidance/research/`, selected by the
rhetorical move the brief asks for, capped at 1,400 characters, placed after the
voice rules and before the corpus anchors. The cards are slices of this document.
Edit here first, then re-slice, so the two do not drift.

Sources are listed at the end and in `guidance/research/SOURCES.md`. Function and
ordering were taken from them. Phrasing was not, and must not be: a card that hands
the model words is the failure `scrub_ai_footprint` exists to catch.

## Why structure is a separate layer from voice

The fingerprint, the scrubber, the presence floors and the rhetoric detectors all
answer one question: does this text read like the author. On 2026-09-04 a Discussion
scored fit 0.201 with no hard flags and was rejected in one line ("a bunch of
nothing. What are we discussing here?"). A Related Work at fit 0.105, the closest
of the day, was rejected as "an AI report and not a research paper". The gate had
no instrument pointed at what a section is for.

PeerCheck (arXiv 2606.20897) ran the ablation directly on academic review
generation: structure-only prompts beat style-only prompts on the quality
dimensions, with gains coming from structured guidance rather than stylistic
imitation. The plan-and-write literature records the cost: outline conditioning
buys coherence and pays in local range. This system already logs slate collapse to
16-49% of the author's spread when content is fixed to the sentence. So the layer
is scoped to one move, capped, told in its own header that the voice block wins on
phrasing, and forbidden from adding a sentence to fill a slot.

## Principles that hold for every passage

From Gopen and Swan (1990), stated as mechanisms:

1. A grammatical subject is followed as soon as possible by its verb. Anything long
   between them is read as an interruption and discounted.
2. The new information the reader should weigh sits at the end of the sentence, at
   the point of syntactic closure (the stress position). A semicolon or colon that
   closes a clause creates a second stress position.
3. The sentence opens (the topic position) with whose story it is and with the old
   information that links back to the previous sentence.
4. The action of a clause is in its verb. A verb list of is, has, are presumed to be
   means the actions are hidden in nouns and the reader has to guess them.
5. Context comes before anything new.
6. A sentence is too long when it has more items worth stressing than it has
   stress positions. Word count is not the test.
7. Every unit of discourse, at every size, makes one point.

Gopen and Swan call these principles rather than rules and say the best stylists
violate them to effect by fulfilling them most of the time. That is the reason the
voice block wins on phrasing.

From Mensh and Kording (2017):

8. One central contribution. A reader should be able to state it a year later.
9. Context, content, conclusion at every scale: paper, section, paragraph.
10. Results are ordered by the logic of the argument, not by chronology.

From Lipton and Steinhardt (2018):

11. Explanation and speculation are kept apart and speculation is labelled.
12. Where a gain has several possible sources, isolate the source or say you have
    not. Ablation, error analysis and robustness checks are what the strongest
    empirical papers have in common.
13. Mathematics appears where it removes ambiguity, not to impress.
14. Terms of art are used in their technical sense. No suggestive names, no
    overloaded terms, no anthropomorphic verbs doing argumentative work.
15. Say which problems are open.

From Docherty and Smith (1999): the Discussion does not sell the paper. "Don't
corrupt your evidence with speculation."

## Section by section

### Abstract
Nature's summary paragraph spec, sentence by sentence: one or two of field
context any scientist can follow; two or three of closer background; one stating
the problem; one stating what this paper did ("here we"); two or three saying what
the result reveals against prior knowledge; one or two of general consequence.
Mensh and Kording: the abstract is a complete context-content-conclusion story on
its own, and the most common mistake is reporting results before the reader knows
the gap. Every number in it appears identically in the body.

### Introduction
Swales' CARS. Move 1, establish the territory: claim centrality, make topic
generalisations, review prior items for what they found. Move 2, establish the
niche: counter-claim, indicate a gap, raise a question, or continue a tradition.
The niche is signalled by contrast (what studies examined against what they did
not) rather than by adjectives about their quality. Move 3, occupy the niche:
outline purpose or announce the present research, announce principal findings,
indicate structure. Mensh and Kording's funnel: each paragraph before the last
runs context-content-conclusion and narrows; the last paragraph states the study
and finding without restating context. Significance is part of the problem
statement, not a separate paragraph: knowing this gives a benefit or not knowing
it incurs a cost, for a named reader.

### Related work
Organised by question or approach. Each cited work is a subject doing a verb with
its result. Placement relative to this paper is a reason: what the work
established, what its design could not see. The section ends by locating this
paper on that map. Length is set by the number of distinct positions in the field.
The catalogue (one sentence per paper, approach as subject, cite cluster) is the
form the author has rejected by name.

### Contribution statement
Each contribution is what the reader can now know or do, with a pointer to its
evidence. No adjectives. Artefacts are not findings. The count matches the body
and the Conclusion answers each.

### Methods
The bar is reproduction by an expert and checkability by a reviewer (NeurIPS
criteria). Order: population or corpus, materials and data with provenance,
procedure as executed, measures, analysis. Every choice that could move a result
carries its value: N, exclusions with counts, thresholds, seeds, model versions,
prompts, dates. EQUATOR checklists (CONSORT, PRISMA, STROBE) are the model for
completeness. General design first, this study's instantiation second, detail to
the appendix third: the author's own ordering rule. Procedures list steps in the
order they ran, each with input, operation, output. Notation defines each symbol
once, one symbol per concept, and appears only where prose is ambiguous.

### Results
Ordered by the argument. Each paragraph names what was tested, gives the number
with uncertainty and N, and closes on what the number supports. The number comes
first. Null results at equal weight. Figures and tables referenced by what they
show; the figure title states the conclusion, the legend how it was computed.
Reporting: units, CI or SD, N per number, test, effect size, both quantities in a
comparison, unambiguous direction, consistent precision, text numbers identical to
table numbers.

### Interpretation
What the data shows, then what is inferred, in that order. One interpretation per
paragraph, tied to a named result. Mechanisms labelled as candidates.
Alternatives named with whether the design rules each out, weakens it, or leaves
it open.

### Discussion
Docherty and Smith's order: principal finding in one sentence; strengths and
weaknesses with equal weight, weaknesses first because editors and readers look
for them and lose trust when they find one the authors did not mention; relation
to prior studies including differences and why, without assuming this study is
the one that is right; meaning, with restraint, leaving the reader to judge;
unanswered questions. Mensh and Kording: each paragraph names one strength or
weakness and evaluates it against the literature. Fails as a re-sale of the
results, as repeated numbers, as speculation dressed as explanation, as a
literature inventory. The author's ruling: when the brief supplies his stated view,
the Discussion is built from it in his order.

### Limitations
Each with its direction of effect on the finding, whether the design mitigates
it, and which conclusions survive it. The reviewer's first objection comes first.
A limitation that voids a claim voids it. Not disguised strengths, not deferred to
future work, not generic to the study type. NeurIPS: authors are rewarded for
stating them up front.

### Conclusion
Answers the Introduction's question in the Introduction's terms. The single
contribution and what it changes. What is open. Nothing new. Not a summary of
every section, not the abstract again, no closing note that widens scope.

### Figure captions and table descriptions
Caption: first sentence is the claim the figure supports; then axes with units,
N, error bars, computation. Table: what rows and columns are, the cell that
carries the finding with its value, the pattern and its exception. Neither
re-lists numbers.

### Reviewer response
Quote the concern. What changed, where, with the new text or number. Where
declined, the reason and its evidence, stated as disagreement. One concern per
item in the reviewer's order. No thanks before the answer, no defence.

## How the model receives this

The kit order is: task, author's craft notes (`profiles/<voice>/VOICE.md`),
measured style rules, the structure block, contrastive demos, accepted exemplars,
corpus examples. The structure block is `## WHAT THIS PASSAGE MUST DO`, and its
header states the division of authority: voice rules win on phrasing, the block
wins on what is included and its order, and no sentence is added to fill a slot.
Under the shipped card, the author's own rulings for that move
(`profiles/<voice>/guidance/<move>.md`) are appended and marked as outranking
the textbook. The compose protocol tells the model to satisfy the block as content
and sequence while building every sentence from the six observations.

The move comes from `format=` when it names one (`discussion`, `related_work`,
`abstract`), otherwise from `retrieve.infer_move(task)`, which is keyword based and
deliberately cheap. A brief that names no move gets the common principles and no
card. A profile without `"guidance"` in `config.json` gets no block at all.

## What to measure before trusting it

Three numbers from the existing harness, with and without the block, on the same
briefs: fingerprint fit (RMS-z against the self-baseline), hard flags, and slate
spread as a fraction of the corpus spread. Spread is where structure charges its
fee. The author's blind read is the only quality verdict; `blind.packet()` builds
the unlabelled pair.

## Sources

- Gopen, G. D. & Swan, J. A. (1990). The Science of Scientific Writing. American Scientist 78(6).
- Mensh, B. & Kording, K. (2017). Ten simple rules for structuring papers. PLoS Comput Biol 13(9): e1005619.
- Swales, J. M. (1990). Genre Analysis: English in Academic and Research Settings. CUP. (CARS model.)
- Docherty, M. & Smith, R. (1999). The case for structuring the discussion of scientific papers. BMJ 318:1224-5.
- Lipton, Z. C. & Steinhardt, J. (2018). Troubling Trends in Machine Learning Scholarship. arXiv:1807.03341.
- Nature (2019). How to construct a Nature summary paragraph. nature.com/documents/nature-summary-paragraph.pdf.
- NeurIPS 2025 Reviewer Guidelines. neurips.cc/Conferences/2025/ReviewerGuidelines.
- EQUATOR Network. Reporting guidelines. equator-network.org/reporting-guidelines/.
- PeerCheck: Enhancing LLM-Generated Academic Reviews Towards Human-Level Quality. arXiv:2606.20897.
- Yang & Carpuat (2025). arXiv:2505.00679. Framework-constrained register analysis; the basis for the six fixed observation slots already in the protocol.
- Wang et al. (2025). arXiv:2509.14543. Content-similarity exemplar selection underperforms random same-author selection; the basis for move-matched demo re-ranking.
