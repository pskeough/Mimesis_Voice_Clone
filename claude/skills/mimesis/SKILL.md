---
name: mimesis
description: Write, draft, restyle, or edit anything in the user's OWN voice/style, including fixing a draft they wrote ("edit my draft", "fix this paragraph I wrote"). Use when the user says "in my voice", "sound like me", "write as me", "write it as if I wrote it", "match my writing", "make it sound like me", "in my style", "research voice", or "my academic voice" - in ANY folder, and even when the request leads with a topic ("write a report on X in my voice"). Clones the user's voice from their real corpus with a stylometric fingerprint gate and an AI-footprint scrubber. Delegates to the Mimesis MCP tools (compose_in_voice, compose_slate, gate_candidates, edit_own_text, apply_own_edits, scrub_ai_footprint).
---

# mimesis: write in the user's real voice

Mimesis is an **MCP server**, not a prompt trick. It composes from the user's real
corpus, scores every draft against a stylometric fingerprint, and runs an
AI-footprint scrubber. This skill is a router entry: it says WHEN to reach for the
engine and HOW to drive it. The engine is the `mimesis` MCP tools.

Install: copy this folder to `~/.claude/skills/mimesis/`. It ships with the Mimesis
repo so it updates with `git pull`; copy it again after an upgrade.

## When to use

Whenever the user asks to write, draft, restyle or edit prose that should read as
**they** wrote it. Trigger even when the request leads with a topic ("draft a post
about X in my voice") or names the register ("my research voice"). Never imitate
their style from memory or from an inferred profile. Drive the tools or say the
tools are unavailable.

## When NOT to use

- Ordinary writing with no request to sound like them.
- Switching or creating voices, or checking status. That is `/voice`, or
  `mimesis profile` on the command line.
- Writing as somebody else. Do not point these tools at a third party's voice
  unless the user asks for that explicitly and the profile exists.

## Which voice

The voices on this install are whatever `list_voices` (or `/voice`) reports, and
the active one is easy to be wrong about. Pass `voice=...` explicitly whenever the
register matters, for example a research profile for papers and reports and a
personal one for posts, letters and notes.

## How to drive it

If the `mimesis` MCP tools are not loaded, load them first (they may be deferred:
search for `compose_in_voice`), then:

1. Call **`compose_in_voice(task=..., voice=...)`** to get the composition kit. It
   returns the style rules, retrieved passages from the corpus, and a protocol.
   **Follow the protocol it returns**, do not substitute your own.
2. That protocol is: fill fixed observation slots from the retrieved passages
   (sentence length, clause structure, punctuation inventory, stance,
   concreteness, endings), then write the piece **from those observations**, not
   from the passages. Do not reuse their imagery or subject matter, and never copy
   four or more consecutive words.
3. Run **`scrub_ai_footprint`** and fix every flag it raises.
4. Present the observations, the draft, and the scrub report. The measurement is
   part of the deliverable, not an internal detail.

**When the output matters**, use the gated path instead: **`compose_slate`** for a
kit plus instructions for N genuinely different candidates, then
**`gate_candidates`** to score them through the real fingerprint, Pareto and
scrub machinery and select. `compose_in_voice` has no discriminator behind it: it
returns a prompt and trusts you to follow it.

Two candidates from opposed briefs beat four similar ones. Slates that vary only
in wording collapse to one candidate under the gate.

## Editing the user's own text

When the input is something the user wrote and the job is to fix it, do not
compose. Call `edit_own_text(text=..., goal=..., voice=...)`, return edits only for
the sentences its scope allows, then `apply_own_edits(...)`. The limits are enforced
in code: one sentence in and one out, at least 60% of each sentence kept, at most
six words added, numbers and citations unchanged, whole-text similarity at least
0.93. Show the diff, any refused edits, and the detection-risk list. Sentences on
that list are the user's to rewrite; do not rephrase them yourself.

## Feedback is signal

When the user says a draft is wrong, that judgment is worth more than the draft.
Ask what was wrong in words rather than silently producing a new draft. If the
Claude Mind memory system is installed, it mines in-session voice feedback nightly
and moves the voice toward what the user actually keeps.

## If the engine is not connected

If the Mimesis MCP tools are unavailable, say so plainly. Do not fall back to
writing "in an inferred voice".
