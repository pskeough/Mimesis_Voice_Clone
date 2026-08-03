"""Compose loop: slate -> fingerprint gate -> quality-select -> scalpel scrub.

The discriminator-in-the-loop, implemented fresh from docs/DESIGN.md:

1. **Slate.** One ``claude -p`` call is prompted to return 3-5 *genuinely
   different* candidates (verbalized sampling), so the gate has real variety to
   choose from instead of one draft and four paraphrases.
2. **Fingerprint gate.** Every candidate is scored by the 13-feature stylometric
   fingerprint; survivors are those with RMS-z <= the profile threshold. The
   fingerprint is the authority on "sounds like the author"; the LLM judge is
   demoted to a secondary quality signal.
3. **Quality-select.** Among survivors, one ``claude -p`` call picks the best on
   quality/faithfulness. With zero survivors, each candidate gets one targeted
   rewrite citing ``Fingerprint.worst_features()`` (max 2 iterations).
4. **Scalpel scrub.** The winner is passed through the scrubber last: em-dashes
   stripped deterministically, everything else flagged, never mean-enforced.

Generation shells out to the local ``claude`` CLI (``-p`` headless, model default
sonnet), ported from the v1/RVCR subprocess pattern, with ``shell=True`` on
Windows so the ``claude.cmd`` shim resolves.
"""
from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
import subprocess
from dataclasses import dataclass, field

from . import accepted as accepted_mod
from . import cadence as cadence_mod
from . import composite as composite_mod
from . import config, fidelity, ingest, retrieve, textnorm
from .fingerprint import Fingerprint
from .scrub import ScrubCalibration, ScrubReport, analyze, render, scalpel

# Host env vars that would redirect the subprocess to managed-provider auth that
# never arrives; strip them so the CLI uses its own OAuth/keychain (ported RVCR).
_STRIP_ENV = {
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_ENTRYPOINT",
    "AI_AGENT",
}

_STATIC_SYSTEM = (
    "You are a precise writing engine. Follow the instructions in the message "
    "exactly. Output only what is asked, with no preamble, no commentary, and no "
    "markdown code fences."
)

_SLATE_MARK = "===CANDIDATE==="


def _claude_bin() -> str:
    return shutil.which("claude") or "claude"


def claude_generate(prompt: str, model: str = "sonnet", timeout: int = 900) -> str:
    """Run one headless ``claude -p`` generation and return its text.

    Raises ``RuntimeError`` on a non-zero exit; callers decide whether to fall
    back (compose surfaces this so ``--dry-run`` stays useful when the CLI is
    unavailable).
    """
    exe = _claude_bin()
    cmd = [
        exe, "-p",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--output-format", "text",
        "--system-prompt", _STATIC_SYSTEM,
    ]
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
    is_win = os.name == "nt"
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            shell=is_win,  # resolve the claude.cmd shim on Windows
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"claude CLI not found on PATH: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"claude exited {result.returncode}. "
            f"stderr: {(result.stderr or '')[:600]}"
        )
    return (result.stdout or "").strip()


# --- kit assembly -------------------------------------------------------------


def _rules_block(profile: config.Profile, cal: ScrubCalibration,
                 markup: str = "plain") -> str:
    whitelist = ", ".join(cal.whitelist) if cal.whitelist else "(none)"
    banned = ", ".join(cal.banned_words[:35]) + ("..." if len(cal.banned_words) > 35 else "")
    # The dash rule has to be stated in the target markup's own terms. Telling a
    # model rewriting LaTeX to "never use --" is telling it to avoid EN-DASH
    # notation, which is how every numeric range in the document is written. The
    # generic wording was in the kit for the pass that turned "0.17--0.21" into
    # "0.17, 0.21" in three table rows.
    if markup == "latex":
        dash_rule = (
            "- Never use em-dashes. In LaTeX that means `---` and the literal `—`.\n"
            "  `--` is EN-DASH notation for numeric ranges (0.17--0.21) and page\n"
            "  spans: leave every one of them exactly as written. Converting a range\n"
            "  to `0.17, 0.21` turns one interval into two point estimates."
        )
    else:
        dash_rule = ("- Never use em-dashes (— or --). Use commas, colons, or split "
                     "the sentence.")
    # The burstiness line used to end "mix short punchy lines with long
    # stacked-clause ones", and it was the single most damaging instruction in the
    # kit -- present in EVERY generation, not just repairs. Burstiness is
    # sentence-length stdev, an order-invariant statistic, and the cheapest way to
    # satisfy it is rigid long/short alternation. Audited across 80 generations
    # from two independent authors, that is exactly what came back:
    # sl_short_after_long +1.68 SD (creative) and +1.12 SD (external author),
    # sl_autocorr1 -2.05 / -1.13 SD. The metric was met and the rhythm got worse.
    # The rule now asks for variation at passage scale, which raises the same
    # statistic without inducing a metronome.
    return f"""## STYLE RULES FOR {profile.name.upper()}'S VOICE
{dash_rule}
- Target mean sentence length ~{cal.mean_sentence_len:.1f} words, and vary sentence
  length across the piece (stdev above {cal.burstiness_floor:.2f}) by letting whole
  passages run long or stay short. Do NOT alternate long and short line by line:
  that hits the number while making the rhythm mechanical, which is the opposite
  of the goal.
- Keep hedging under {cal.hedge_ceiling:.2f} per 200 words; prefer definite verbs.
- Do not use these AI-tell words: {banned}
- {profile.name} naturally uses these, so they are allowed: {whitelist}
- Preserve every fact, number, and citation from the task exactly. Invent nothing."""


def _anchor_block(
    task: str,
    profile: config.Profile,
    n_examples: int,
    exclude_files: set[str] | None = None,
) -> str:
    parts: list[str] = []
    # Contrastive transform demos: the profile's own pairs plus any mined from the
    # author's edits (Upgrade 3). Edits are the same anchor type, so they render
    # identically and stack with the shipped pairs.
    demos = retrieve.transform_demos(task, 4, profile) if profile.pairs_path else []
    if accepted_mod.has_accepted(profile):
        demos = demos + accepted_mod.demo_pairs(profile, task, 3)
    if demos:
        parts.append(
            "## CONTRASTIVE TRANSFORM DEMONSTRATIONS (apply the pattern, never the wording)"
        )
        for i, d in enumerate(demos, 1):
            parts.append(
                f"--- Demo {i} (move: {d.get('move', '?')}) ---\n"
                f"AI DRAFT:\n{d['ai_text']}\n\n{profile.name.upper()} REWRITE:\n{d['human_text']}"
            )
    # Accepted exemplars (Upgrade 3): the drafts the author actually kept, given
    # extra weight as the strongest positive signal, ahead of general corpus recall.
    accepted_hits = accepted_mod.retrieve_accepted(profile, task, 3)
    if accepted_hits:
        parts.append(
            f"## ACCEPTED EXEMPLARS ({profile.name} kept these; match their voice most closely)"
        )
        for i, r in enumerate(accepted_hits, 1):
            parts.append(f"--- Accepted {i} ---\n{r['text']}")
    # A composite voice draws anchors from every declared source, allocated by
    # influence share, so a thin primary corpus is not the only thing the model has
    # to imitate. Each hit is labelled with the register it came from: the model
    # should know an example is essayistic rather than academic, or it flattens
    # the registers together and the composite reads as an average of them.
    try:
        if composite_mod.is_composite(profile):
            hits = composite_mod.retrieve_blended(
                task, n_examples, profile, exclude_files=exclude_files
            )
        else:
            hits = retrieve.retrieve(task, n_examples, profile, exclude_files=exclude_files)
    except FileNotFoundError:
        hits = []
    if hits:
        parts.append(f"## RELEVANT EXAMPLES FROM {profile.name.upper()}'S WRITING")
        for i, h in enumerate(hits, 1):
            src = h.get("_source")
            tag = f" | register: {src}" if src else ""
            parts.append(
                f"--- Example {i} from \"{h['filename']}\"{tag} ---\n{h['text']}"
            )
    if not parts:
        parts.append("(No indexed corpus anchors available; rely on the style rules.)")
    return "\n\n".join(parts)


def build_kit(
    task: str,
    profile: config.Profile,
    cal: ScrubCalibration,
    n_examples: int = 5,
    exclude_files: set[str] | None = None,
    markup: str = "plain",
    source: str | None = None,
) -> str:
    """Assemble the full composition kit (rules + anchors + task).

    ``source`` is the document being rewritten, when there is one. Its rhythm is
    stated as a constraint rather than left implicit: a model asked only to
    "improve" prose reliably shortens it, and the measured result of that is a
    manuscript with 9.8% less sentence-length variance than the one it replaced.
    """
    extra = ""
    if source:
        prose = textnorm.to_prose(source, markup)
        lens = [len(s.split()) for s in re.split(r"(?<=[.!?])\s+", prose)
                if len(s.split()) >= 2]
        if len(lens) >= 8:
            import statistics as _st
            longs = sum(1 for L in lens if L >= 35)
            shorts = sum(1 for L in lens if L < 12)
            extra = (
                f"\n\n## RHYTHM OF THE SOURCE, WHICH YOU MUST NOT FLATTEN\n"
                f"The passage you are rewriting has {len(lens)} sentences: mean "
                f"{_st.mean(lens):.1f} words, standard deviation {_st.pstdev(lens):.1f}, "
                f"with {longs} at 35+ words and {shorts} under 12.\n"
                f"Keep that spread. Do not split the long sentences into medium ones "
                f"and do not pad the short ones. A rewrite that lowers the standard "
                f"deviation is rejected even when every sentence in it is an "
                f"improvement, because uniform sentence length is the single "
                f"strongest signal that prose was machine-written."
            )
    return (
        f"# VOICE COMPOSITION KIT — write in {profile.name}'s voice\n\n"
        f"TASK: {task}\n\n"
        f"{_rules_block(profile, cal, markup=markup)}{extra}\n\n"
        f"{_anchor_block(task, profile, n_examples, exclude_files=exclude_files)}"
    )


# --- prompts ------------------------------------------------------------------


def _slate_prompt(kit: str, n: int) -> str:
    return (
        f"{kit}\n\n## OUTPUT\n"
        f"Produce {n} GENUINELY DIFFERENT candidate responses to the TASK, each in "
        f"the author's voice. Make them differ in structure, opening, and rhythm, not "
        f"just word choice. Do not rank them. Separate each candidate with a line "
        f"containing exactly:\n{_SLATE_MARK}\n"
        f"Output only the candidates and the separators."
    )


def _parse_slate(raw: str, n: int) -> list[str]:
    parts = [p.strip() for p in raw.split(_SLATE_MARK)]
    cands = [p for p in parts if len(p.split()) >= 20]
    return cands[:n] if cands else ([raw.strip()] if raw.strip() else [])


def _quality_prompt(task: str, candidates: list[str]) -> str:
    listing = "\n\n".join(
        f"[{i}]\n{c}" for i, c in enumerate(candidates)
    )
    return (
        f"TASK: {task}\n\n"
        f"Below are numbered candidate responses. Choose the ONE that best fulfils the "
        f"task with the most natural, faithful prose. Reply with ONLY the number of the "
        f"best candidate, nothing else.\n\n{listing}"
    )


def _rewrite_prompt(kit: str, draft: str, worst: list[tuple[str, float]]) -> str:
    # Cadence features get plain-language instructions; a model cannot act on
    # "sl_autocorr1 z=+2.3", but it can act on "long sentences bunch together".
    parts = []
    for name, z in worst:
        h = cadence_mod.hint(name, z)
        parts.append(h if h else f"{name} is {'too high' if z > 0 else 'too low'} (z={z:+.2f})")
    feats = "; ".join(parts)
    return (
        f"{kit}\n\n## REWRITE\n"
        f"This draft is close but its rhythm and stylometric fingerprint deviate from the "
        f"author: {feats}.\nRewrite it to correct those specific problems while preserving "
        f"meaning and every fact. Changing the pacing means changing where sentences start "
        f"and end, not swapping words. Output only the rewritten text.\n\nDRAFT:\n{draft}"
    )


def _detector_rewrite_prompt(kit: str, draft: str, author: str) -> str:
    return (
        f"{kit}\n\n## DE-MACHINE PASS\n"
        f"A machine-text detector (a perplexity-ratio classifier) reads the draft below "
        f"as machine-generated rather than written by {author}. Rewrite it so it reads as "
        f"genuinely human-authored in {author}'s voice: break the too-even rhythm, let "
        f"real specificity and idiosyncrasy through, avoid the smooth generic register a "
        f"model defaults to. Preserve meaning, every fact and number, and the author's "
        f"voice. Never add em-dashes. Output only the rewritten text.\n\nDRAFT:\n{draft}"
    )


def _scrub_rewrite_prompt(kit: str, draft: str, report: ScrubReport, author: str) -> str:
    return (
        f"{kit}\n\n## REPAIR\n"
        f"This draft is in the right voice but the scrubber flagged hard issues:\n\n"
        f"{render(report, author=author)}\n\n"
        f"Fix every flagged issue while preserving meaning, voice, and every fact. "
        f"Fidelity issues (dropped or invented numbers/citations) take priority. "
        f"Never add em-dashes. Output only the repaired text.\n\nDRAFT:\n{draft}"
    )


# --- result types -------------------------------------------------------------


@dataclass
class Candidate:
    text: str
    rmsz: float
    zs: dict = field(default_factory=dict)
    scrub: ScrubReport | None = None
    emdash_fixed: int = 0
    detector: dict = field(default_factory=dict)
    # Populated only when compose() was given the document being rewritten.
    fid: "fidelity.FidelityReport | None" = None


@dataclass
class ComposeResult:
    output: str | None
    chosen: Candidate | None
    candidates: list[Candidate]
    survivors: list[Candidate]
    iterations: int
    kit: str
    dry_run: bool
    notes: list[str] = field(default_factory=list)
    fidelity: "fidelity.FidelityReport | None" = None


# --- gate machinery -----------------------------------------------------------


def _score(text: str, fp: Fingerprint) -> tuple[float, dict]:
    return fp.distance_detail(text)



def _target_distance(rmsz: float, fp: Fingerprint, mode: str) -> float:
    """How far a candidate is from where it SHOULD sit.

    "minimize" ranks by raw RMS-z, which aims at the corpus centroid. Nothing the
    author wrote lives there: his own pieces score at the self-baseline (0.77-0.95
    across these profiles), so minimising drives every generation toward a point
    none of his prose occupies. Measured consequence: six generations from three
    different voice profiles had mean pairwise distance 0.657 while seven of his
    own comparable pieces had 1.065 -- the outputs were 38% tighter than the author.
    That is what optimising toward a mean produces.

    "band" ranks by |rmsz - self_baseline|, so the target is where his writing
    actually sits. Default stays "minimize" until the A/B in
    evals/ab_selection.py earns the change on a given voice.
    """
    if mode == "band" and fp.self_baseline > 0:
        return abs(rmsz - fp.self_baseline)
    return rmsz


def slate_spread(cands: list["Candidate"], fp: Fingerprint) -> float:
    """Mean pairwise fingerprint distance within a slate.

    The slate is generated and then thrown away except for the winner, which
    discards a free signal: if four candidates prompted for genuine difference all
    land in the same place, that sameness is the base model's prior rather than
    the author's voice. The same collapse showed up as three arms closing on one
    rhetorical move. Compare this against the author's own corpus spread.
    """
    if len(cands) < 2:
        return 0.0
    vecs = []
    for c in cands:
        _, zs = fp.distance_detail(c.text)
        vecs.append([zs[k] for k in fp.features])
    n, tot, pairs = len(vecs[0]), 0.0, 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            tot += (sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j])) / n) ** 0.5
            pairs += 1
    return tot / pairs if pairs else 0.0


def _pareto_front(cands: list[Candidate]) -> list[Candidate]:
    """Candidates not beaten on BOTH voice fidelity and scrub cleanliness.

    The old loop selected on RMS-z alone, then scrubbed the winner -- so it
    optimised a distance and then edited the text away from the point it had
    optimised, and patched the damage with a repair rewrite. Scrub state is
    knowable before selection (the scalpel is deterministic and the analysis is
    free), so it belongs in the selection, not after it.

    A candidate is dominated when another is at least as good on both axes and
    strictly better on one. Everything on the resulting frontier is a genuine
    trade-off, and the LLM judge picks among those on quality alone.
    """
    def cost(c: Candidate) -> tuple[float, int]:
        return (c.rmsz, len(c.scrub.hard_flags) if c.scrub else 0)

    front = []
    for c in cands:
        cc = cost(c)
        if not any(
            (o is not c)
            and (oc := cost(o))
            and oc[0] <= cc[0]
            and oc[1] <= cc[1]
            and (oc[0] < cc[0] or oc[1] < cc[1])
            for o in cands
        ):
            front.append(c)
    return front or cands


def _quality_select(task: str, survivors: list[Candidate], model: str) -> int:
    if len(survivors) == 1:
        return 0
    raw = claude_generate(_quality_prompt(task, [c.text for c in survivors]), model=model)
    m = re.search(r"\d+", raw)
    idx = int(m.group()) if m else 0
    return idx if 0 <= idx < len(survivors) else 0


def compose(
    task: str,
    profile: config.Profile,
    fmt: str | None = None,
    source: str | None = None,
    dry_run: bool = False,
    n_examples: int = 5,
    model: str = "sonnet",
    exclude_files: set[str] | None = None,
    use_detector: bool = False,
    detector_threshold: float | None = None,
    detector_calibrated: bool = False,
    detector_direction: str = "low_is_machine",
) -> ComposeResult:
    """Run the full compose loop for ``task`` in ``profile``'s voice.

    Pass ``source`` when this is a REWRITE of an existing document rather than a
    fresh generation. It turns on fidelity.verify, which compares the output
    against the document instead of against the corpus, and which is the only
    layer that can see a lost section, a flattened rhythm, or a register the
    source never used. Without it those checks cannot run at all: the corpus does
    not know what this document was before.

    ``dry_run`` exercises everything except generation: it builds the kit and
    self-tests the fingerprint + scrub scoring path on a real corpus anchor, then
    reports without calling the model. Raises ``FileNotFoundError`` if the profile
    was never calibrated.
    """
    if not profile.fingerprint_path.exists():
        raise FileNotFoundError(
            f"'{profile.slug}' is not calibrated. Run: mimesis calibrate {profile.slug}"
        )
    fp = Fingerprint.load(profile.fingerprint_path)
    cal = ScrubCalibration.load(profile.scrub_path)
    fmt_cfg = profile.format_config(fmt)
    gate_cfg = fmt_cfg["gate"]
    notes: list[str] = []
    # The MARKUP language, which is not the same thing as the profile's named
    # format. "methods-section" is a format; LaTeX is a markup. Sniffed from the
    # source when there is one, because that is the ground truth for what the
    # output will be, and from the task otherwise.
    markup = textnorm.guess_format(source or task)
    # "auto" uses the profile's own calibrated p95 fit threshold instead of a
    # fixed number. A hardcoded 1.1 is a claim about one corpus: on an external
    # long-form corpus whose self-baseline is 0.966 and fit_threshold 1.701, a
    # 1.1 ceiling sits BELOW what the author's own writing reliably scores, so
    # every candidate failed, both rewrite iterations burned, and the loop fell
    # back to "closest by RMS-z" on every single brief -- at ~5x the generation
    # cost. Left opt-in rather than defaulted so existing profiles keep the exact
    # thresholds their reported numbers were measured against.
    # "minimize" (legacy) or "band" (aim at the author's self-baseline).
    select_mode = str(gate_cfg.get("select", "minimize")).lower()
    _rz = gate_cfg.get("rmsz_max", 1.1)
    if isinstance(_rz, str) and _rz.lower() == "auto":
        rmsz_max = fp.fit_threshold if fp.fit_threshold > 0 else fp.self_baseline * 1.6
        notes.append(f"gate: rmsz_max=auto -> {rmsz_max:.3f} (profile p95)")
    else:
        rmsz_max = float(_rz)
    slate_size = int(gate_cfg.get("slate_size", 4))
    max_rewrites = int(gate_cfg.get("max_rewrites", 2))

    kit = build_kit(task, profile, cal, n_examples=n_examples,
                    exclude_files=exclude_files, markup=markup, source=source)

    if dry_run:
        # Prove the scoring path works on real corpus text, without generation.
        #
        # Score a whole corpus PIECE, not a retrieval anchor. Anchors are ~180-200
        # word chunks while the fingerprint is calibrated on full pieces (or on
        # ~1500-word segments), and short texts have unstable feature values that
        # inflate the distance: on a segmented profile the anchor self-test read
        # RMS-z 2.073 against a 1.471 threshold and looked like a failure when
        # nothing was wrong. A self-test must compare like with like.
        pieces = [t for t in ingest.read_pieces(profile.db_path).values() if t.strip()]
        if pieces:
            sample = max(pieces, key=len)
            rmsz, _ = _score(sample, fp)
            rep = analyze(sample, cal, fp=fp)
            notes.append(
                f"dry-run self-test on a real corpus piece ({len(sample.split())} words): "
                f"RMS-z={rmsz:.3f} (threshold {rmsz_max:.3f}, corpus self-baseline "
                f"{fp.self_baseline:.3f}), "
                f"scrub={'clean' if rep.is_clean else 'flags: ' + ','.join(rep.hard_flags)}"
            )
        else:
            notes.append("dry-run: no corpus anchors available to self-test.")
        notes.append("dry-run: generation skipped (no claude -p call made).")
        return ComposeResult(
            output=None, chosen=None, candidates=[], survivors=[], iterations=0,
            kit=kit, dry_run=True, notes=notes,
        )

    def _build(text: str) -> Candidate:
        """Scalpel first, then score. The scalpel is deterministic and
        meaning-preserving, so running it before scoring means every candidate is
        measured in the state it would actually ship in -- rather than scoring a
        draft, picking it, and then editing it into a different one.

        Scoring happens on PROSE, not on markup. Every fingerprint feature is
        defined over sentences and words, so handing it raw LaTeX measures
        backslash commands and table cells: a rewrite pass reported healthy
        per-subsection RMS-z figures that were computed over ``\\subsection{...}``
        and ``$d_{\\mathrm{pop}}$`` and meant nothing. The scalpel still operates
        on the original text, because that is what ships.
        """
        fixed, n_em = scalpel(text, fmt=markup)
        r, zs = _score(textnorm.to_prose(fixed, markup), fp)
        c = Candidate(text=fixed, rmsz=r, zs=zs, emdash_fixed=n_em)
        c.scrub = analyze(fixed, cal, source=task, fp=fp)
        if source:
            c.fid = fidelity.verify(source, fixed, fmt=markup)
        return c

    # 1. Slate.
    raw = claude_generate(_slate_prompt(kit, slate_size), model=model)
    texts = _parse_slate(raw, slate_size)
    candidates = [_build(t) for t in texts]
    if not candidates:
        raise RuntimeError("model returned no usable candidates")
    spread = slate_spread(candidates, fp)
    notes.append(f"slate spread {spread:.3f} across {len(candidates)} candidates"
                 + (f" (author corpus spread {fp.meta.get('corpus_spread'):.3f})"
                    if fp.meta.get("corpus_spread") else ""))
    cs = fp.meta.get("corpus_spread") or 0.0
    if cs and spread < 0.5 * cs:
        notes.append(
            f"SLATE COLLAPSE: candidates are {spread / cs:.0%} as varied as the "
            f"author's own writing. They were prompted to differ and did not, which "
            f"points at the base model's prior rather than this voice."
        )

    # 2. Fingerprint gate + targeted rewrites on failure.
    iterations = 0

    def _passes(c: Candidate) -> bool:
        """Voice distance AND source fidelity. Both, or the gate is decorative.

        The failure this guards against is specific: a rewrite pass accepted on
        asset checks alone, printed a fingerprint distance to a report, and
        shipped a manuscript with a lost section and a flattened cadence. A
        number that nothing filters on is a number nobody is using.
        """
        return c.rmsz <= rmsz_max and (c.fid is None or c.fid.ok)

    survivors = [c for c in candidates if _passes(c)]
    while not survivors and iterations < max_rewrites:
        iterations += 1
        notes.append(f"iteration {iterations}: zero survivors, targeted rewrite pass")
        # The candidates are independent, so rewrite them concurrently. Each call
        # is a separate `claude -p` subprocess dominated by startup and network
        # latency, and a slate of 4 run serially was the single largest cost in
        # the loop: measured at 220-390s per brief on a long-form profile, against
        # 30-60s when the gate passed on the first pass and no rewrite ran.
        def _rewrite_one(c: Candidate) -> Candidate:
            worst = fp.worst_features(c.text, k=3)
            prompt = _rewrite_prompt(kit, c.text, worst)
            # Naming the actual fidelity breakages is what the earlier retry loop
            # failed to do: it resampled with an identical prompt and got the same
            # class of output back. A rewrite told "you flattened the cadence and
            # lost a section" can fix those; one told nothing cannot.
            if c.fid and c.fid.hard:
                prompt += ("\n\nTHIS DRAFT ALSO BROKE FIDELITY AGAINST THE SOURCE. "
                           "Fix every item, and change nothing else:\n"
                           + "\n".join(f"  - {p}" for p in c.fid.problems()))
            try:
                return _build(claude_generate(prompt, model=model))
            except RuntimeError:
                return c

        with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as pool:
            candidates = list(pool.map(_rewrite_one, candidates))
        survivors = [c for c in candidates if _passes(c)]

    if survivors:
        # 3. Pareto frontier over (voice fidelity, scrub flags), then quality-select
        # among the genuine trade-offs. Selecting on RMS-z alone and scrubbing
        # afterwards is what made "faithful voice" and "clean of AI tells" fight
        # each other: the winner was chosen before half the criteria were applied.
        pool = _pareto_front(survivors)
        if len(pool) < len(survivors):
            notes.append(
                f"pareto: {len(pool)}/{len(survivors)} survivors on the "
                f"fidelity/scrub frontier"
            )
    else:
        # Fidelity breakages outrank voice distance in the fallback. A draft that
        # sounds slightly less like the author is a stylistic cost; a draft that
        # dropped a section is a broken document, and there is no RMS-z good
        # enough to pay for that.
        pool = sorted(
            candidates,
            key=lambda c: (len(c.fid.hard) if c.fid else 0,
                           _target_distance(c.rmsz, fp, select_mode)),
        )[:1]
        worst_fid = pool[0].fid
        notes.append(
            "no candidate passed the gate after rewrites; selecting the least-bad"
            + (f" ({len(worst_fid.hard)} fidelity breakage(s) REMAIN: "
               f"{'; '.join(worst_fid.problems())})" if worst_fid and worst_fid.hard
               else " by RMS-z.")
        )

    try:
        idx = _quality_select(task, pool, model)
    except RuntimeError:
        idx = min(range(len(pool)),
                  key=lambda i: _target_distance(pool[i].rmsz, fp, select_mode))
        notes.append(f"quality-select call failed; fell back to {select_mode} selection.")
    chosen = pool[idx]

    # 4. The chosen candidate is already scalpelled and scored. If hard flags remain
    # (a banned phrase, or a fidelity number/citation drop), do one bounded repair
    # rewrite citing the scrub report. A single pass keeps the loop bounded; residual
    # flags are surfaced in notes rather than looped on, since forcing further
    # rewrites risks the mean-collapse the scrubber exists to avoid.
    if chosen.emdash_fixed:
        notes.append(f"scalpel: stripped {chosen.emdash_fixed} em-dash(es).")

    if chosen.scrub.hard_flags:
        notes.append(f"scrub gate: hard flags {chosen.scrub.hard_flags}; one repair pass")
        try:
            repaired = claude_generate(
                _scrub_rewrite_prompt(kit, chosen.text, chosen.scrub, profile.name),
                model=model,
            )
            r_fixed, r_em = scalpel(repaired, fmt=markup)
            r_rep = analyze(r_fixed, cal, source=task, fp=fp)
            r_fid = fidelity.verify(source, r_fixed, fmt=markup) if source else None
            r_rmsz = _score(textnorm.to_prose(r_fixed, markup), fp)[0]
            # Accept the repair only if it reduced hard flags AND either cleared them
            # entirely or kept the fingerprint within the gate threshold -- and
            # never if it broke source fidelity the draft had intact. A repair pass
            # rewrites whole sentences, which is exactly when a section heading or
            # a numeric range goes missing.
            fewer_flags = len(r_rep.hard_flags) < len(chosen.scrub.hard_flags)
            fp_ok = r_rep.is_clean or r_rmsz <= rmsz_max
            fid_ok = r_fid is None or len(r_fid.hard) <= len(chosen.fid.hard if chosen.fid else [])
            if fewer_flags and fp_ok and fid_ok:
                chosen.text = r_fixed
                chosen.emdash_fixed += r_em
                chosen.rmsz, chosen.zs = _score(textnorm.to_prose(r_fixed, markup), fp)
                chosen.scrub = r_rep
                chosen.fid = r_fid
                notes.append(f"scrub gate: repaired, remaining flags {r_rep.hard_flags or 'none'}")
            else:
                notes.append(
                    "scrub gate: repair rejected ("
                    + ("broke source fidelity" if not fid_ok
                       else "no improvement or fingerprint regressed") + ")"
                )
        except RuntimeError:
            notes.append("scrub gate: repair call failed; surfacing flagged draft")

    # 5. Detector gate (Upgrade 2), last. Report the Binoculars signal always; if
    # it reads clearly machine-generated (score below the calibrated threshold),
    # do ONE bounded de-machine rewrite, accepting it only if the detector score
    # improves AND the fingerprint stays within the gate (never trade voice
    # fidelity for a detector win, the same accept-guard the scrub repair uses).
    if use_detector:
        from .scrub import detector_signal
        det = detector_signal(
            chosen.text, threshold=detector_threshold, calibrated=detector_calibrated,
            direction=detector_direction,
        )
        chosen.detector = det
        if det.get("available") and det.get("label") == "machine":
            notes.append(
                f"detector: reads machine (score {det.get('score')} < threshold "
                f"{detector_threshold}); one de-machine rewrite"
            )
            try:
                demachined = claude_generate(
                    _detector_rewrite_prompt(kit, chosen.text, profile.name), model=model
                )
                d_fixed, d_em = scalpel(demachined, fmt=markup)
                d_rmsz = _score(textnorm.to_prose(d_fixed, markup), fp)[0]
                d_fid = fidelity.verify(source, d_fixed, fmt=markup) if source else None
                d_det = detector_signal(
                    d_fixed, threshold=detector_threshold, calibrated=detector_calibrated,
                    direction=detector_direction,
                )
                # "more human" means moving away from the machine side, which
                # depends on the calibrated direction (high vs low = machine).
                _new, _old = d_det.get("score"), det.get("score")
                if d_det.get("available") and _new is not None and _old is not None:
                    improved = (_new < _old) if detector_direction == "high_is_machine" else (_new > _old)
                else:
                    improved = False
                fp_ok = d_rmsz <= rmsz_max or d_rmsz <= chosen.rmsz + 1e-6
                # Same guard as the scrub repair: a detector win is never worth a
                # broken document.
                fid_ok = d_fid is None or len(d_fid.hard) <= len(chosen.fid.hard if chosen.fid else [])
                if improved and fp_ok and fid_ok:
                    chosen.text = d_fixed
                    chosen.emdash_fixed += d_em
                    chosen.rmsz, chosen.zs = _score(textnorm.to_prose(d_fixed, markup), fp)
                    chosen.scrub = analyze(d_fixed, cal, source=task)
                    chosen.fid = d_fid
                    chosen.detector = d_det
                    notes.append(
                        f"detector: de-machine accepted (score {det.get('score')} -> "
                        f"{d_det.get('score')}, label {d_det.get('label')})"
                    )
                else:
                    notes.append(
                        "detector: de-machine rejected (no score gain or fingerprint regressed)"
                    )
            except RuntimeError:
                notes.append("detector: de-machine call failed; surfacing flagged draft")
        elif det.get("available"):
            notes.append(f"detector: score {det.get('score')} label {det.get('label')} (no rewrite)")

    # Recompute against the FINAL text. The repair and de-machine passes rewrite
    # sentences after selection, so a report taken at selection time describes a
    # draft that is no longer the one being returned.
    final_fid = fidelity.verify(source, chosen.text, fmt=markup) if source else None
    if final_fid and final_fid.hard:
        notes.append("FIDELITY: " + "; ".join(final_fid.problems()))
    return ComposeResult(
        output=chosen.text,
        chosen=chosen,
        candidates=candidates,
        survivors=survivors,
        iterations=iterations,
        kit=kit,
        dry_run=False,
        notes=notes,
        fidelity=final_fid,
    )
