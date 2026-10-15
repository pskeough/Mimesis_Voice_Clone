# Sources for the research guidance cards

Function taxonomy and ordering, not phrasing, was taken from each. Nothing in the
cards quotes a source; a card that names words to use is a bug.

- Gopen, G. D. & Swan, J. A. (1990). The Science of Scientific Writing. American
  Scientist 78(6). Seven reader-expectation principles: subject followed soon by
  verb; new information in the stress position; the story's subject in the topic
  position; old information in the topic position for linkage; action in the verb;
  context before new material; structural emphasis matching substantive emphasis.
  Also: a sentence is too long when it has more stress-worthy items than stress
  positions. The authors state these are principles to be violated to effect, not
  rules. `_common.md` carries them as mechanisms.
- Mensh, B. & Kording, K. (2017). Ten simple rules for structuring papers. PLoS
  Comput Biol 13(9). Single central contribution; context-content-conclusion at
  every scale; results ordered by logic; abstract as a complete story; discussion
  paragraphs each evaluating one strength or weakness. `abstract`, `introduction`,
  `results`, `discussion`, `contribution_statement`.
- Swales, J. M. (1990). Genre Analysis. The CARS model: establish a territory,
  establish a niche (counter-claim, gap, question, continuation), occupy the niche
  (purpose, present research, principal findings, structure). `introduction`,
  `gap_statement`.
- Docherty, M. & Smith, R. (1999). The case for structuring the discussion of
  scientific papers. BMJ 318:1224. Principal finding; strengths and weaknesses;
  relation to other studies including differences; meaning with restraint;
  unanswered questions. "Don't corrupt your evidence with speculation." `discussion`,
  `limitations`, `discussion_implications`.
- Lipton, Z. C. & Steinhardt, J. (2018). Troubling Trends in Machine Learning
  Scholarship. arXiv:1807.03341. Explanation vs speculation; identify the source of
  empirical gains; mathiness; suggestive or overloaded terms; say which problems are
  open. `_common.md`, `interpretation`, `methods_notation`.
- Nature (2019). How to construct a Nature summary paragraph. Sentence-by-sentence
  abstract structure with counts. `abstract`.
- NeurIPS 2025 Reviewer Guidelines. Claims supported by evidence; reproducibility
  as the bar for Methods; limitations rewarded when stated up front. `methods`,
  `limitations`, `results_reporting`.
- EQUATOR Network reporting guidelines (CONSORT, PRISMA, STROBE). Completeness
  checklists per study type; the model for "every choice that could move a result
  is stated with its value". `methods`, `methods_procedure`.
- PeerCheck (arXiv 2606.20897). Ablation showing structure-only prompts beat
  style-only prompts on quality dimensions for academic review generation. The
  reason this layer is separated from the voice block rather than folded into it.

Deliberately excluded: the Manchester Academic Phrasebank. It is a repository of
formulaic phrases, and importing phrases into a voice cloner is the failure
`scrub_ai_footprint` exists to catch. Its function taxonomy overlaps Swales and
was not needed separately.
