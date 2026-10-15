"""FastMCP stdio server exposing the Mimesis voice tools.

Setup: ``list_voices``, ``inspect_corpus``, ``create_voice``.
Writing: ``compose_slate`` + ``gate_candidates`` (gated), ``compose_in_voice``
(kit only), ``get_voice_guide``, ``retrieve_style_examples``,
``retrieve_transform_demos``.
Checking and learning: ``scrub_ai_footprint``, ``eval_voice``, ``record_preference``.

Registered with Claude Code as ``mimesis`` by ``scripts/register-mcp.*``.

Inside an MCP host the *host model* is the generator, so the compose tools
return a corpus-anchored kit and a protocol, and ``gate_candidates`` runs the
same discriminator ``gate.compose`` uses on the CLI. The active voice and on/off
state resolve on every call, so ``/voice use`` and ``/voice-off`` take effect
with no restart.

Until the person has a built voice of their own, every writing tool returns the
intake protocol instead of a kit. The shipped ``example`` voice is a demo, and
answering "write this in my voice" from it is the failure this guards against.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastmcp import FastMCP

from . import accepted as accepted_mod
from . import config, evalcli, retrieve
from . import intake as intake_mod
from . import harvest as harvest_mod
from . import presence as presence_mod
from . import rhetoric_ext as rhetoric_ext_mod
from . import scrub as scrub_mod
from .measures import calibration as structure_mod
from .measures import report as structure_report
from .fingerprint import Fingerprint
from .scrub import ScrubCalibration

mcp = FastMCP("mimesis")


def _log_event(profile, tool: str, **fields) -> None:
    """Per-call event trail -> profiles/<slug>/preferences.jsonl.

    Marks WHERE composition happened so the LKHS preference miner knows which
    session transcripts to walk (the drafts themselves live in the transcript;
    the host model is the generator on the MCP path). Append-only; must never
    break the tool call it annotates.
    """
    try:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "tool": tool, "voice": profile.slug, **{k: v for k, v in fields.items() if v is not None}}
        accepted_mod._append_jsonl(profile.root / "preferences.jsonl", rec)
    except Exception:
        pass

_DISABLED = (
    "Mimesis voice cloning is OFF (the user ran /voice-off). Do not write in any "
    "author's voice; write normally. Re-enable with /voice-on."
)


SETUP_PROTOCOL = """NO VOICE IS SET UP FOR THIS PERSON YET.
Mimesis has only its synthetic demo voice, so nothing here can write in their voice.
Do not imitate their style from memory or from this conversation. Run the intake with them:

1. Ask what they want written in their voice and in which forms (emails, social posts,
   essays, fiction, academic papers). Each form becomes its own voice, named like
   jane-posts and jane-essays, because one fingerprint across forms measures format.
2. Ask where their writing is: folders or files of .txt, .md, .docx or .pdf. Target about
   40 pieces or 20,000 words per form; 5,000 words is the floor. Only writing they produced
   alone: nothing co-written, nothing heavily edited by someone else, nothing drafted with
   AI. Ask about AI-assisted pieces directly, since recent work is often mixed.
   Exports: Google Docs -> .docx; sent email -> paste bodies into .txt files;
   LinkedIn/Substack/Medium exports -> one .txt or .md per post.
3. Call inspect_corpus(paths=[...]) and go through every warning with them.
4. Call create_voice(name=..., paths=[...], author="Their Name"). For a very large corpus,
   run `mimesis new NAME --from PATH --author "Their Name"` in the Mimesis folder instead.
5. Ask the VOICE.md questions in the repo's CLAUDE.md ("First run") and write their answers
   into profiles/NAME/VOICE.md.
Then retry the original request with voice=NAME."""


def _profile(voice: str | None):
    """Resolve a named voice for one call, or the active one.

    Returns (profile, None) or (None, message). The message is the intake
    protocol whenever the person has no built voice and did not ask for the
    demo by name.
    """
    if voice:
        p = config.resolve_named(voice)
        if p is None:
            msg = (f"No voice profile '{voice}'. Available: "
                   f"{', '.join(config.list_profiles()) or '(none)'}.")
            return None, (msg + "\n\n" + SETUP_PROTOCOL) if config.needs_setup() else msg
        if p.slug != config.EXAMPLE_SLUG and not config.is_built(p.slug):
            return None, (f"Voice '{p.slug}' exists but is not built. "
                          f"Run create_voice again or: mimesis add {p.slug}")
        return p, None
    if config.needs_setup():
        return None, SETUP_PROTOCOL
    p = config.resolve_active()
    if p.slug == config.EXAMPLE_SLUG:
        built = [s for s in config.user_voices() if config.is_built(s)]
        return None, (
            "The active voice is the synthetic demo. This person's voices: "
            f"{', '.join(built)}. Pass voice=NAME, or run `mimesis profile use NAME`."
        )
    return p, None


def _rules(profile) -> str | None:
    if not profile.scrub_path.exists():
        return None
    cal = ScrubCalibration.load(profile.scrub_path)
    from .gate import _rules_block  # internal reuse; keeps one rules source

    return _rules_block(profile, cal)


@mcp.tool(
    description=(
        "Load the active author's calibrated voice blueprint: pacing/burstiness/hedging "
        "rules, AI-tell bans, whitelisted vocabulary, and a few real samples. Call this "
        "(or compose_in_voice) whenever asked to write in the author's voice instead of "
        "guessing. Pass voice=NAME to target a specific profile."
    )
)
def get_voice_guide(voice: str | None = None) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    rules = _rules(prof)
    if rules is None:
        return f"'{prof.slug}' is not calibrated yet. Run: mimesis calibrate {prof.slug}"
    header = f"# {prof.name.upper()}'S VOICE BLUEPRINT\n"
    try:
        hits = retrieve.retrieve("a representative passage", 3, prof)
        samples = "\n\n".join(f'--- from "{h["filename"]}" ---\n{h["text"]}' for h in hits)
    except Exception:
        samples = ""
    parts = [header, rules]
    if samples:
        parts.append("## SAMPLES\n" + samples)
    return "\n\n".join(parts)


@mcp.tool(
    description=(
        "Hybrid (vector + keyword) search over the active author's real writing, MMR-"
        "diversified. Returns their most relevant paragraphs as few-shot style anchors. "
        "Pass the draft excerpt or topic as query_text; voice=NAME to target a profile."
    )
)
def retrieve_style_examples(query_text: str, limit: int = 5, voice: str | None = None) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    try:
        hits = retrieve.retrieve(query_text, max(1, min(limit, 12)), prof)
    except FileNotFoundError:
        return f"No store for '{prof.slug}'. Run: mimesis ingest {prof.slug}"
    if not hits:
        return f"No style examples indexed for '{prof.slug}'."
    return "RETRIEVED STYLE EXAMPLES (study tone and rhythm; do not copy wording):\n\n" + "\n\n".join(
        f'--- Example {i} from "{h["filename"]}" ---\n{h["text"]}' for i, h in enumerate(hits, 1)
    )


@mcp.tool(
    description=(
        "Retrieve contrastive AI->author rewrite demonstrations closest in style to the "
        "query, for profiles that ship transform pairs (e.g. a research voice). The "
        "strongest anchor for rewriting AI-sounding text. Empty if the profile has no pairs."
    )
)
def retrieve_transform_demos(query_text: str, k: int = 4, voice: str | None = None) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    demos = retrieve.transform_demos(query_text, max(1, k), prof)
    if not demos:
        return f"No transform pairs configured for '{prof.slug}'."
    out = ["CONTRASTIVE TRANSFORM DEMONSTRATIONS (apply the pattern, never the wording):"]
    for i, d in enumerate(demos, 1):
        out.append(
            f"\n--- Demo {i} (move: {d.get('move', '?')}) ---\n"
            f"AI DRAFT:\n{d['ai_text']}\n\n{prof.name.upper()} REWRITE:\n{d['human_text']}"
        )
    return "\n".join(out)


@mcp.tool(
    description=(
        "THE primary tool for writing anything in the active author's voice. Returns the "
        "style rules, the most relevant real example passages (MMR-diversified), any "
        "transform demos, a structure block saying what this kind of passage must accomplish "
        "(when the profile has a guidance pack), and a fixed-slot observe-then-write protocol. "
        "Follow the protocol, run scrub_ai_footprint, then present. Pass voice=NAME to target a "
        "profile, format=MOVE (e.g. discussion, related_work, abstract) to name the rhetorical "
        "slot when the task text does not. No discriminator: when the output matters use "
        "compose_slate + gate_candidates. If no voice is set up, returns the intake protocol."
    )
)
def compose_in_voice(
    task: str, examples: int = 5, voice: str | None = None, format: str | None = None
) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    if not prof.scrub_path.exists():
        return f"'{prof.slug}' is not calibrated yet. Run: mimesis calibrate {prof.slug}"
    cal = ScrubCalibration.load(prof.scrub_path)
    from .gate import build_kit

    kit = build_kit(task, prof, cal, n_examples=max(1, min(examples, 12)), fmt=format)
    _log_event(prof, "compose", task=task[:300], format=format)
    # Replaced the Draft A / Draft B protocol 2026-08-28 on blind evidence.
    #
    # Four framings were tested on the same brief in three voices, each arm written
    # by an ISOLATED agent that did not know the other arms existed (an earlier
    # round where one model wrote every arm produced a result that reversed within
    # the same evening, which is what the isolation is for). Rank points across the
    # three voices, lower better:
    #
    #   analyse (fixed-slot observation, then write)   5   never below 2nd
    #   echo    ("echo the rhythm, syntax and stance")  6   won 2 of 3, last in the 3rd
    #   rules-only (no exemplars at all)                9
    #   observe ("read them, then set them aside")     10   worst overall
    #
    # The six slots are fixed rather than open-ended on the strength of Yang &
    # Carpuat 2025 (arXiv:2505.00679), where framework-constrained register analysis
    # beat both plain imitation and open-ended style description. Open-ended
    # self-description collapses into adjective soup and scores well on trait-based
    # judges while moving no authorship metric.
    protocol = (
        "## HOW TO PRODUCE THE OUTPUT\n"
        "Work in two steps.\n\n"
        "STEP 1. Observe the passages above and fill in these six slots, one line "
        "each. Report only what you can actually see in them.\n"
        "  1. SENTENCE LENGTH: typical length, and how much it varies within a paragraph.\n"
        "  2. CLAUSE STRUCTURE: how clauses join. Subordination, coordination, "
        "apposition, fragments.\n"
        "  3. PUNCTUATION INVENTORY: which marks appear and at what frequency, "
        "including question marks, parentheses, ellipses, semicolons.\n"
        "  4. STANCE: how present the writer is, how certain, how much is hedged, "
        "whether they judge what they describe.\n"
        "  5. CONCRETENESS: what gets named specifically and what stays abstract.\n"
        "  6. ENDINGS: what the final sentence of a passage does.\n\n"
        "STEP 2. Write the piece from those six observations. Do not look back at the "
        "passages while writing, and do not reuse their imagery, subject matter, or "
        "phrasing. Never copy four or more consecutive words from them.\n"
        "If the kit carries a WHAT THIS PASSAGE MUST DO block, it sets what the piece "
        "includes and in what order; the six observations set how every sentence is "
        "built. Satisfy the block as content and sequence, never as added sentences, "
        "and leave out any slot the task gives you nothing for.\n\n"
        "Then call scrub_ai_footprint on the result and fix every flag. The hard "
        "flags are not style advice: self-explanation, unheeded-reversal, way-simile "
        "and declarative-rating are machine constructions a reader rejected by name in "
        "blind testing, each measured at or near zero in the human corpora they were "
        "checked against (docs/DETECTORS.md).\n"
        "Present the six observation lines, the draft, and the scrub report.\n"
        "For a fully autonomous generate-score-rewrite pass, use the CLI: "
        f"mimesis compose {prof.slug} \"{task}\"."
    )
    return kit + "\n\n" + protocol


@mcp.tool(
    description=(
        "Vet any draft meant to match the active author's voice against their measured "
        "style (em-dashes, burstiness, hedging, AI-cliche vocabulary/phrases), check "
        "two-sided fingerprint fit (catches drafts that are unlike the author by being "
        "too plain, not only by showing AI tells), and, when source is given, audit "
        "number/citation fidelity. Returns repair instructions. Run as the final pass "
        "before delivering voiced text."
    )
)
def scrub_ai_footprint(text: str, source: str | None = None, voice: str | None = None) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    if not prof.scrub_path.exists():
        return f"'{prof.slug}' is not calibrated yet. Run: mimesis calibrate {prof.slug}"
    cal = ScrubCalibration.load(prof.scrub_path)
    # The fingerprint is what makes this a resemblance check rather than only an
    # AI-tell check. The CLI compose gate has always used it; this path did not,
    # so a draft that was simply unlike the author scored CLEAN here.
    fp = Fingerprint.load(prof.fingerprint_path) if prof.fingerprint_path.exists() else None
    # Presence is what neither the banlist nor the fingerprint can see: whether anyone is visibly
    # thinking in the draft. Falls back to the published-field floors when a voice has not been
    # recalibrated since this check existed, so it works on every profile immediately.
    pres = (
        presence_mod.PresenceCalibration.load(prof.presence_path)
        if getattr(prof, "presence_path", None) and prof.presence_path.exists()
        else presence_mod.PresenceCalibration.default()
    )
    rep = scrub_mod.analyze(text, cal, source=source, fp=fp, pres=pres,
                            ext=rhetoric_ext_mod.load_for(prof))
    _log_event(prof, "scrub", text_sha=hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], chars=len(text))
    # Keep the draft itself, not only its hash. Every draft produced before this
    # existed had to be recovered from Claude Code transcripts by `mimesis harvest`.
    harvest_mod.record_draft(
        prof, text, "scrub",
        verdict=",".join(rep.hard_flags) if getattr(rep, "hard_flags", None) else "clean",
    )
    return "\n".join([scrub_mod.render(rep, prof.name), _structure_block(prof, text)])


@mcp.tool(
    description=(
        "Report the fingerprint distribution for the active voice: the corpus self-"
        "baseline RMS-z and the held-out real pieces' RMS-z (fingerprint-only, no "
        "generation). Use the CLI `mimesis eval` for the full discrimination eval."
    )
)
def eval_voice(voice: str | None = None, held_out: int = 5) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    try:
        res = evalcli.run_eval(prof, held_out=held_out, fingerprint_only=True)
    except FileNotFoundError as e:
        return str(e)
    return evalcli.render(res, prof)


@mcp.tool(
    description=(
        "Record the user's preference on a voiced draft the moment it is expressed in "
        "chat: kind='accept' when they keep a draft (pass the full accepted text so it "
        "joins the recalibration set), 'reject' when they discard one, 'feedback' for a "
        "stylistic correction ('too formal', 'shorter'). Only for STYLE signal on drafts "
        "in the author's voice — never for factual corrections. Optional but valuable; "
        "the nightly miner reconstructs chains from transcripts either way."
    )
)
def record_preference(
    kind: str, voice: str | None = None, text: str | None = None,
    task: str | None = None, note: str | None = None,
) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    kind = (kind or "").strip().lower()
    if kind not in ("accept", "reject", "feedback"):
        return "kind must be one of: accept, reject, feedback."
    _log_event(prof, "preference", kind=kind, task=(task or "")[:300] or None, note=(note or "")[:500] or None,
               text_sha=hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else None)
    if kind == "accept" and text and text.strip():
        rec = accepted_mod.record_accept(
            prof, text, task=task, source="mcp-accept",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        return (
            f"Recorded accept {rec['id']} for voice '{prof.slug}' ({len(text.split())} words). "
            f"It now anchors future compose kits; run `mimesis recalibrate {prof.slug}` after ~5 new accepts."
        )
    return f"Recorded {kind} for voice '{prof.slug}'."


def _structure_block(prof, text: str) -> str:
    """The structural report for one draft, or a line saying why there is none.

    Reporting only. Every line is a band fitted to the author's own corpus, and
    a two-sigma band flags a twentieth of his real writing by arithmetic; see
    docs/STRUCTURE_GATES.md for the measured per-line rate on each voice.
    """
    return structure_report.render(text, structure_mod.load_for(prof), prof.name)


def _structure_flags(cal, text: str) -> str:
    """The flagged line names for one candidate, for the ranking table."""
    if cal is None:
        return "-"
    try:
        hits = [ln.key for ln in structure_report.analyse(text, cal) if ln.flagged]
    except Exception:
        return "-"
    return ",".join(hits) if hits else "ok"


# --- gated composition (discriminator on the MCP path) ------------------------
#
# compose_in_voice returns a kit and TRUSTS the host model to follow a protocol.
# That is the whole architecture on this path: no slate, no fingerprint gate, no
# selection, no repair loop. Every number in README/evals describes gate.compose,
# which only ever ran on the CLI, and the CLI generator is `claude -p`, so the
# gate is unavailable whenever that CLI cannot authenticate.
#
# These two tools put the real gate on the MCP path with the host model as the
# generator: compose_slate asks for N genuinely different candidates, and
# gate_candidates runs the identical scalpel -> analyze -> score -> band-target
# -> Pareto machinery gate.compose uses, then reports. Same discriminator, no
# subprocess, no second auth.


@mcp.tool(
    description=(
        "GATED composition, step 1 of 2. Returns the voice kit plus instructions to write N "
        "candidate drafts (default 2) from deliberately OPPOSED structural briefs. Write all N, "
        "then pass them to gate_candidates, which scores and picks. Use this instead of "
        "compose_in_voice whenever the output matters: compose_in_voice has no discriminator. "
        "Always pass voice=NAME."
    )
)
def compose_slate(
    task: str, n: int = 2, examples: int = 5, voice: str | None = None,
    format: str | None = None,
) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    if not prof.scrub_path.exists():
        return f"'{prof.slug}' is not calibrated yet. Run: mimesis calibrate {prof.slug}"
    cal = ScrubCalibration.load(prof.scrub_path)
    from .gate import build_kit

    n = max(2, min(int(n), 8))
    kit = build_kit(task, prof, cal, n_examples=max(1, min(examples, 12)), fmt=format)
    _log_event(prof, "compose_slate", task=task[:300], format=format, slate=n)
    protocol = (
        "## HOW TO PRODUCE THE OUTPUT\n"
        f"Write {n} SEPARATE candidate drafts of the task above, labeled CANDIDATE 1..{n}.\n"
        f"Before drafting, choose {n} structural approaches that are opposed rather than "
        "adjacent (lead with the claim vs lead with the case; scene first vs argument first; "
        "chronological vs problem first) and commit one candidate to each. Measured over about "
        "twenty gated sections: four candidates written off one brief covered 19-55% of the "
        "author's own variety and the gate reported COLLAPSED every time; two written off "
        "opposed briefs covered 86-148% and passed. A slate of near-identical drafts measures "
        "the base model's prior, not the author's range.\n"
        "Do not self-edit toward the rules while drafting; the gate scores what you wrote.\n"
        "Never copy four or more consecutive words from the examples.\n"
        "If the kit carries a WHAT THIS PASSAGE MUST DO block, every candidate satisfies it "
        "as content and order; the candidates differ in how they get there, not in whether.\n"
        f"Then call gate_candidates(candidates=[...all {n} texts...], task=..., "
        f"voice='{prof.slug}') and report its ranking. Do NOT pick a winner yourself "
        "before the gate runs."
    )
    return kit + "\n\n" + protocol


@mcp.tool(
    description=(
        "GATED composition, step 2 of 2. Scores a slate of candidate drafts through the real "
        "gate: deterministic em-dash scalpel, full scrub analysis, 13-feature fingerprint "
        "distance, band-mode target (aims at the author's self-baseline, NOT the corpus "
        "centroid), Pareto front over voice-fit and hard flags, and slate-spread collapse "
        "detection against the author's own corpus spread. Returns a ranked table, the "
        "surviving front, and the winner's repair instructions."
    )
)
def gate_candidates(
    candidates: list[str], task: str | None = None, voice: str | None = None,
    select: str = "band", source: str | None = None,
) -> str:
    if not config.is_enabled():
        return _DISABLED
    prof, err = _profile(voice)
    if err:
        return err
    if not prof.scrub_path.exists():
        return f"'{prof.slug}' is not calibrated yet. Run: mimesis calibrate {prof.slug}"
    texts = [t for t in (candidates or []) if t and t.strip()]
    if not texts:
        return "No candidates given."
    from . import gate as gate_mod
    from .gate import Candidate
    from .textnorm import guess_format

    cal = ScrubCalibration.load(prof.scrub_path)
    if not prof.fingerprint_path.exists():
        return f"'{prof.slug}' has no fingerprint. Run: mimesis calibrate {prof.slug}"
    fp = Fingerprint.load(prof.fingerprint_path)
    pres = (
        presence_mod.PresenceCalibration.load(prof.presence_path)
        if getattr(prof, "presence_path", None) and prof.presence_path.exists()
        else presence_mod.PresenceCalibration.default()
    )

    markup = guess_format(task or texts[0])
    allow = gate_mod._allow_dashes(cal)
    mode = (select or "band").lower()

    cands: list[Candidate] = []
    for t in texts:
        # Scalpel first, then score. The scalpel is deterministic, so scoring the
        # pre-scalpel text would rank a candidate on characters the gate is about
        # to remove anyway.
        fixed, n_em = scrub_mod.scalpel(t, fmt=markup, allow_dashes=allow)
        rmsz, zs = fp.distance_detail(fixed)
        c = Candidate(text=fixed, rmsz=rmsz, zs=zs, emdash_fixed=n_em)
        # source is the DOCUMENT being rewritten, if this is a rewrite. Never the
        # brief: numbers in a brief ("about 350 words") are not facts the draft
        # owes back, and a fidelity flag is HARD, so a brief-induced one fires on
        # every candidate and flattens the Pareto front's second axis.
        c.scrub = scrub_mod.analyze(fixed, cal, source=source, fp=fp, pres=pres,
                                    ext=rhetoric_ext_mod.load_for(prof))
        cands.append(c)

    threshold = fp.fit_threshold if fp.fit_threshold > 0 else fp.self_baseline * 1.6
    passing = [c for c in cands if c.rmsz <= threshold]
    front = gate_mod._pareto_front(passing or cands)
    front = sorted(front, key=lambda c: gate_mod._target_distance(c.rmsz, fp, mode))
    winner = front[0]

    spread = gate_mod.slate_spread(cands, fp)
    corpus_spread = float(fp.meta.get("corpus_spread") or 0.0)

    out: list[str] = []
    out.append(f"# GATE REPORT - {prof.name} ({prof.slug}), select={mode}")
    out.append(
        f"self_baseline={fp.self_baseline:.3f}  fit_threshold(p95)={threshold:.3f}  "
        f"corpus_spread={corpus_spread:.3f}"
    )
    out.append("")
    # Structure lines are REPORTED per candidate and never scored into the
    # ranking. They measure construction, and the study that produced them found
    # construction and voice moving in opposite directions under a fine-tune, so
    # a candidate must not lose the front for being architecturally unlike the
    # author when the reader might prefer exactly that.
    struct_cal = structure_mod.load_for(prof)
    struct = [_structure_flags(struct_cal, c.text) for c in cands]
    out.append("| # | words | RMS-z | target-dist | hard flags | banned | em-dash fixed | presence | structure |")
    out.append("|---|-------|-------|-------------|------------|--------|---------------|----------|-----------|")
    for i, c in enumerate(cands, 1):
        r = c.scrub
        hf = ",".join(r.hard_flags) if r and r.hard_flags else "-"
        banned = len((r.banned_words or []) + (r.banned_phrases or [])) if r else 0
        pres_ok = "MISSING" if (r and getattr(r, "presence_missing", False)) else "ok"
        mark = "  <-- winner" if c is winner else ""
        out.append(
            f"| {i}{mark} | {len(c.text.split())} | {c.rmsz:.3f} | "
            f"{gate_mod._target_distance(c.rmsz, fp, mode):.3f} | {hf} | {banned} | "
            f"{c.emdash_fixed} | {pres_ok} | {struct[i - 1]} |"
        )
    out.append("")
    out.append(
        f"passing fit gate: {len(passing)}/{len(cands)}   Pareto front: {len(front)}   "
        f"winner: candidate {cands.index(winner) + 1}"
    )
    if corpus_spread > 0:
        pct = 100.0 * spread / corpus_spread
        verdict = "COLLAPSED" if pct < 70 else ("tight" if pct < 90 else "ok")
        out.append(
            f"slate spread: {spread:.3f} vs author corpus {corpus_spread:.3f} "
            f"({pct:.0f}% of the author's own variety), {verdict}"
        )
        if pct < 70:
            out.append(
                "  Candidates prompted to differ landed in the same place. That sameness is "
                "the base model's prior, and no gate can add variety that was never "
                "generated. Re-slate with a genuinely different brief per candidate."
            )
    out.append("")
    out.append("## WINNER SCRUB REPORT")
    out.append(scrub_mod.render(winner.scrub, prof.name) if winner.scrub else "(none)")
    out.append(_structure_block(prof, winner.text))
    _log_event(
        prof, "gate", task=(task or "")[:300] or None, slate=len(cands),
        winner_rmsz=round(winner.rmsz, 4), spread=round(spread, 4), select=mode,
    )
    # A slate is the cheapest preference signal this system gets: the gate has
    # already labelled winner and losers, so persisting all of them costs nothing
    # and yields a ranked pair per loser.
    for c in cands:
        harvest_mod.record_draft(
            prof, c.text, "gate", task=task,
            verdict="winner" if c is winner else "loser",
        )
    return "\n".join(out)


# --- setup ---------------------------------------------------------------------


@mcp.tool(
    description=(
        "List this person's voices: name, author, document count, whether it is built, which "
        "one is active, and whether voice mode is on. Call it first in a new session, or when "
        "unsure which voice= to pass. If they have no voice yet, it returns the intake protocol "
        "for setting one up from their writing."
    )
)
def list_voices() -> str:
    config.ensure_profiles()
    st = config.read_state()
    lines = [f"Voice mode: {'ON' if st['enabled'] else 'OFF'}   active: {st.get('active') or '(none)'}"]
    for slug in config.list_profiles():
        cfg = config.load_profile_config(slug)
        prof = config.resolve(slug)
        try:
            n_docs = sum(1 for f in prof.source_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTS)
        except OSError:
            n_docs = 0
        tag = " (synthetic demo)" if slug == config.EXAMPLE_SLUG else ""
        state = "built" if config.is_built(slug) else "NOT BUILT"
        lines.append(f"  {slug:<22} {cfg['author_name']:<24} {n_docs:>4} docs  {state}{tag}")
    if config.needs_setup():
        lines += ["", SETUP_PROTOCOL]
    return "\n".join(lines)


@mcp.tool(
    description=(
        "Inspect folders or files of a person's writing BEFORE building a voice from them. "
        "Reads every .txt/.md/.docx/.pdf the way ingest will and reports piece count, words, "
        "length bands, and problems: too little writing, short posts mixed with long chapters, "
        "one document that would dominate, PDFs with no extractable text, unsupported exports. "
        "Writes nothing. Paths are absolute, or relative to the Mimesis folder."
    )
)
def inspect_corpus(paths: list[str]) -> str:
    if not paths:
        return "Pass paths=[...]: folders or files of the person's own writing."
    return intake_mod.render(intake_mod.inspect([str(x) for x in paths]))


@mcp.tool(
    description=(
        "Build a new voice from a person's own writing: copies every supported document under "
        "the given folders/files into profiles/NAME, ingests, calibrates, and makes it the "
        "active voice if no real voice was active. Run inspect_corpus first and resolve its "
        "warnings with the person. Refuses a corpus under the word floor unless force=True. "
        "Takes seconds to a few minutes; the first run downloads a small embedding model."
    )
)
def create_voice(name: str, paths: list[str], author: str | None = None, force: bool = False) -> str:
    import argparse
    import contextlib
    import io

    from . import cli

    if not name or not paths:
        return "Pass name='jane-essays' and paths=[...] (folders or files of their writing)."
    slug = config.slugify(name)
    if slug == config.EXAMPLE_SLUG:
        return "'example' is reserved for the demo voice. Pick another name."
    args = argparse.Namespace(
        name=name, sources=[str(x) for x in paths], author=author,
        backend="fast", no_build=False, force=bool(force),
    )
    buf = io.StringIO()
    # stdout is the MCP transport. Anything the CLI prints must go to the buffer,
    # or the JSON-RPC stream is corrupted mid-call.
    with contextlib.redirect_stdout(buf):
        try:
            rc = cli.cmd_new(args)
        except Exception as e:  # report, never kill the server
            print(f"create_voice failed: {type(e).__name__}: {e}")
            rc = 1
    # Per-document ingest progress is noise to the agent reading this; keep the
    # import, store, calibration and warning lines.
    out = "\n".join(
        ln for ln in buf.getvalue().splitlines()
        if not ln.startswith("Processing: ") and " paragraphs (" not in ln
    ).strip()
    if rc == 0 and config.is_built(slug):
        out += (
            f"\n\nVoice '{slug}' is built. Next: ask the VOICE.md questions (CLAUDE.md, "
            f"'First run') and write the answers into profiles/{slug}/VOICE.md, then write a "
            f"short test piece with compose_slate(voice='{slug}') + gate_candidates and ask "
            "them whether it sounds like them."
        )
    return out or f"create_voice returned {rc} with no output."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
