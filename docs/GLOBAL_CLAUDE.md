# Routing "in my voice" requests to Mimesis from any folder

Claude Code reads `~/.claude/CLAUDE.md` in every session. Adding the block below
makes a request like "write this email in my voice" use the Mimesis tools in any
folder, instead of a model imitating the person from memory.

Step 9 of "First run" in `CLAUDE.md` offers this. Show the person the filled-in
block and add it only after an explicit yes. Replace the angle-bracket parts
with their voices before showing it.

```markdown
## Writing in my voice (Mimesis)

The `mimesis` MCP server writes in my voice from my own writing. Whenever I ask
to write, draft, rewrite or edit anything in my voice ("in my voice", "sound
like me", "write as me", "match my writing"), in any folder, use its tools.
Never imitate my voice from memory. If the tools are not connected, say so.

1. Pick the voice by the form of the piece and pass voice= on every call:
   <jane-posts for LinkedIn posts; jane-essays for long-form and newsletters;
   jane-email for email>. Call list_voices if unsure.
2. For anything I will send or publish: compose_slate(voice=...) with two
   candidates written from opposed structural briefs, then gate_candidates.
   compose_in_voice is fine for a quick draft.
3. Run scrub_ai_footprint(voice=...) on the final text and fix every flag.
4. Show me the draft and the scrub report. When I keep, reject or correct a
   draft, call record_preference.
```

To remove it later, delete the block. Nothing else in Mimesis depends on it.
