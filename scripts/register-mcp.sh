#!/usr/bin/env bash
# Register the Mimesis MCP server with Claude Code at user scope.
#
# Name: 'mimesis', or $MIMESIS_MCP_NAME if set. Safe to re-run.
#
#   * A registration under that name that already points at THIS repo is
#     replaced (remove, then add), so re-running never duplicates.
#   * A registration under that name that points somewhere else is left alone
#     and reported. Two voice servers answering the same requests is a silent
#     failure, so this script will not create the second one.
#   * The old 'mimesis-v2' name, used by earlier versions of this script, is
#     removed when it points at this repo.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${MIMESIS_MCP_NAME:-mimesis}"
py="$repo/.venv/bin/python"
[ -x "$py" ] || py="$repo/.venv/Scripts/python.exe"
if [ ! -x "$py" ]; then
  echo "No venv interpreter under $repo/.venv. Run install.sh first."
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH. Install Claude Code, then re-run."
  exit 1
fi

# Lower-case, forward slashes: the same repo can be printed either way on Windows.
norm() { printf '%s' "$1" | tr '\\' '/' | tr '[:upper:]' '[:lower:]'; }
repo_n="$(norm "$repo")"
points_here() { case "$(norm "$1")" in *"$repo_n"*) return 0;; *) return 1;; esac; }

if [ "$name" != "mimesis-v2" ]; then
  legacy="$(claude mcp get mimesis-v2 2>/dev/null || true)"
  if [ -n "$legacy" ] && points_here "$legacy"; then
    claude mcp remove mimesis-v2 --scope user >/dev/null 2>&1 || true
    echo "Removed the old 'mimesis-v2' registration (same server, old name)."
  fi
fi

existing="$(claude mcp get "$name" 2>/dev/null || true)"
if [ -n "$existing" ] && ! points_here "$existing"; then
  echo "An MCP server named '$name' is already registered and points elsewhere:"
  printf '%s\n' "$existing" | sed 's/^/    /'
  echo "Not replacing it. Either remove it (claude mcp remove $name --scope user) and re-run,"
  echo "or register this install under another name: MIMESIS_MCP_NAME=mimesis-2 bash scripts/register-mcp.sh"
  exit 2
fi

claude mcp remove "$name" --scope user >/dev/null 2>&1 || true
claude mcp add --scope user --transport stdio "$name" -- "$py" "-m" "mimesis_voice.server"
echo "Registered '$name' (user scope) -> $py -m mimesis_voice.server"
echo "Restart Claude Code (or start a new session) to load the tools."
