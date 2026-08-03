#!/usr/bin/env bash
# Register the Mimesis v2 MCP server with Claude Code as 'mimesis-v2'.
#
# Adds a stdio MCP server named 'mimesis-v2' at user scope, pointing at this
# repo's venv Python and the server module. It NEVER touches an existing
# 'mimesis' registration, so the v1 server keeps working during transition.
#
# Run this manually when you are ready to wire the server into Claude Code;
# it is not executed as part of the build.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venv/bin/python"
[ -x "$py" ] || py="python3"

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH. Install Claude Code, then re-run."
  exit 1
fi

# Remove first so re-running never duplicates or errors. Only touches 'mimesis-v2'.
claude mcp remove mimesis-v2 --scope user >/dev/null 2>&1 || true
claude mcp add --scope user --transport stdio mimesis-v2 -- "$py" "-m" "mimesis_voice.server"
echo "Registered 'mimesis-v2' (user scope). The existing 'mimesis' server was left untouched."
