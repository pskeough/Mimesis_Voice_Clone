"""FastMCP stdio server exposing the Mimesis voice tools.

Tools: ``get_voice_guide``, ``retrieve_style_examples``, ``retrieve_transform_demos``,
``compose_in_voice``, ``scrub_ai_footprint``, ``eval_voice``. Registered as
``mimesis-v2`` (the v1 ``mimesis`` server is left untouched during transition).

Inside an MCP host the *host model* is the generator, so ``compose_in_voice``
returns a corpus-anchored composition kit and directs the Draft A / Draft B +
scrub workflow, exactly like v1. The autonomous generate-score-rewrite loop
(``gate.compose``) is the headless path, driven by ``mimesis compose`` on the CLI.
The active voice and on/off state resolve on every call, so ``/voice use`` and
``/voice-off`` take effect with no restart.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastmcp import FastMCP

from . import accepted as accepted_mod
from . import config, evalcli, retrieve
from . import presence as presence_mod
from . import scrub as scrub_mod
from .fingerprint import Fingerprint
from .scrub import ScrubCalibration

mcp = FastMCP("mimesis-v2")


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


def _profile(voice: str | None):
    """Resolve a named voice for one call, or the active one."""
    if voice:
        p = config.resolve_named(voice)
        if p is None:
            return None, (
                f"No voice profile '{voice}'. Available: "
                f"{', '.join(config.list_profiles()) or '(none)'}."
            )
        return p, None
    return config.resolve_active(), None


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
        "transform demos, and a Draft A (rules-only) / Draft B (corpus-anchored) protocol. "
        "Produce both drafts, run scrub_ai_footprint on each, then present them. Pass "
        "voice=NAME to target a profile, format=NAME for a format cell."
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

    kit = build_kit(task, prof, cal, n_examples=max(1, min(examples, 12)))
    _log_event(prof, "compose", task=task[:300], format=format)
    protocol = (
        "## HOW TO PRODUCE THE OUTPUT\n"
        "Produce TWO labeled drafts:\n"
        "- DRAFT A (rules only): apply the style rules above; do not lean on the examples.\n"
        "- DRAFT B (corpus-anchored): also echo the rhythm, syntax, and stance of the "
        "examples and demos. Never copy four or more consecutive words from them.\n"
        "Then call scrub_ai_footprint on each draft and fix every flag (no em-dashes). "
        "Present both with their scrub reports and recommend the stronger one.\n"
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
    rep = scrub_mod.analyze(text, cal, source=source, fp=fp, pres=pres)
    _log_event(prof, "scrub", text_sha=hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], chars=len(text))
    return scrub_mod.render(rep, prof.name)


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
