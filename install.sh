#!/usr/bin/env bash
# One-command install for macOS and Linux.
#
#   bash install.sh
#
# Creates the venv, installs the package, builds and calibrates every profile
# that ships with a corpus but no store, registers the MCP server with Claude
# Code, and finishes by running the doctor on each voice.
#
# Safe to re-run. Ingest hash-skips an unchanged corpus, calibration is
# deterministic, and MCP registration is remove-then-add so it never duplicates.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
fail() { printf "\n! %s\n  FIX: %s\n" "$1" "$2"; exit 1; }

bold "== 1/5  Python =="
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || fail "no Python 3.11 or newer found" \
  "install Python 3.11+ (macOS: brew install python@3.12), then re-run bash install.sh"
echo "   $("$PY" --version) at $(command -v "$PY")"

bold "== 2/5  Virtual environment =="
if [ ! -d .venv ]; then
  "$PY" -m venv .venv || fail "could not create .venv" "check disk space and permissions, then re-run"
  echo "   created .venv"
else
  echo "   .venv already present (reusing)"
fi
# POSIX venvs put the interpreter in bin/; Windows (Git Bash, WSL-adjacent) puts
# it in Scripts/. Checking only one misreports a perfectly healthy venv as broken
# and sends the user off to delete it.
VPY=".venv/bin/python"
[ -x "$VPY" ] || VPY=".venv/Scripts/python.exe"
[ -x "$VPY" ] || fail ".venv exists but has no interpreter" "rm -rf .venv && bash install.sh"

bold "== 3/5  Package =="
"$VPY" -m pip install --quiet --upgrade pip
# Editable install: the profiles live in this directory, so the package is meant
# to be used from here rather than copied into site-packages.
"$VPY" -m pip install --quiet -e . || fail "pip install failed" "re-run and read the error above"
echo "   mimesis-voice installed (fast backend; no torch)"

bold "== 4/5  Voices =="
built=0
for dir in profiles/*/; do
  slug="$(basename "$dir")"
  [ -d "$dir/source_documents" ] || continue
  # Any real document at all? An empty corpus is a profile stub, not a voice.
  if ! find "$dir/source_documents" -type f \( -name '*.txt' -o -name '*.md' -o -name '*.docx' -o -name '*.pdf' \) 2>/dev/null | grep -q .; then
    continue
  fi
  if [ -f "$dir/data/store.sqlite" ] && [ -f "$dir/data/fingerprint.json" ]; then
    echo "   $slug: already built (skipping; delete data/ to force a rebuild)"
    continue
  fi
  echo "   $slug: ingesting..."
  "$VPY" -m mimesis_voice.cli ingest "$slug" >/dev/null 2>&1 \
    || { echo "     ! ingest failed for $slug; run: $VPY -m mimesis_voice.cli ingest $slug"; continue; }
  echo "   $slug: calibrating..."
  "$VPY" -m mimesis_voice.cli calibrate "$slug" 2>&1 | grep -E "self-baseline|THIN" | sed 's/^/     /'
  built=$((built+1))
done
[ "$built" -gt 0 ] || echo "   (nothing new to build)"

bold "== 5/5  Claude Code =="
if command -v claude >/dev/null 2>&1; then
  bash scripts/register-mcp.sh || echo "   ! MCP registration failed; run scripts/register-mcp.sh by hand"
  # An older voice server left registered will keep answering voice requests,
  # and nothing about that failure looks like a failure.
  if claude mcp list 2>/dev/null | grep -qE '^\s*mimesis\s'; then
    printf "\n   NOTE: an older MCP server named 'mimesis' is still registered.\n"
    printf "   If that is a previous voice install, remove it so it cannot answer:\n"
    printf "       claude mcp remove mimesis --scope user\n"
  fi
else
  echo "   claude CLI not on PATH; skipped MCP registration."
  echo "   Install Claude Code, then run: bash scripts/register-mcp.sh"
fi

bold "== Health =="
status=0
for dir in profiles/*/; do
  slug="$(basename "$dir")"
  [ -f "$dir/data/fingerprint.json" ] || continue
  line="$("$VPY" -m mimesis_voice.cli doctor "$slug" 2>&1 | grep -E "RESULT" | head -1)"
  printf "   %-22s %s\n" "$slug" "${line:-no result}"
  case "$line" in *fail*|*FAIL*) status=1;; esac
done

cat <<'EOF'

===========================================================
 Done. Try a voice:

   .venv/bin/python -m mimesis_voice.cli profile list
   .venv/bin/python -m mimesis_voice.cli compose <voice> "a short note about your week"
   (Windows: use .venv\Scripts\python.exe)

 Or just ask Claude Code, in this folder, to write something in your voice.
 Re-run `bash install.sh` any time; it is safe and skips finished work.
===========================================================
EOF
exit $status
