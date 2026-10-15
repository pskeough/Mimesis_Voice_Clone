#!/usr/bin/env bash
# One-command install for macOS and Linux.
#
#   bash install.sh                 # everything
#   bash install.sh --skip-mcp      # do not touch Claude Code's MCP config
#   bash install.sh --no-structure  # skip the optional spaCy tagger (~50 MB)
#
# Creates the venv, installs the package, builds and calibrates every profile
# that ships with a corpus but no store, registers the MCP server with Claude
# Code as 'mimesis', and finishes by running the doctor on each voice.
#
# Safe to re-run. Ingest hash-skips an unchanged corpus, calibration is
# deterministic, and MCP registration is remove-then-add so it never duplicates.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_MCP="${MIMESIS_SKIP_MCP:-0}"
STRUCTURE=1
for arg in "$@"; do
  case "$arg" in
    --skip-mcp) SKIP_MCP=1 ;;
    --no-structure) STRUCTURE=0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

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
# Optional: the spaCy tagger behind the opener-class line of the structure
# report. Everything else works without it, so a failure here is a note.
if [ "$STRUCTURE" = 1 ]; then
  if "$VPY" -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
    echo "   spaCy tagger already present"
  elif "$VPY" -m pip install --quiet -e ".[structure]" && "$VPY" -m spacy download en_core_web_sm >/dev/null 2>&1; then
    echo "   spaCy tagger installed (opener check on)"
  else
    echo "   ! spaCy tagger not installed; the opener line will report unavailable."
    echo "     Retry later: $VPY -m pip install -e '.[structure]' && $VPY -m spacy download en_core_web_sm"
  fi
fi

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
if [ "$SKIP_MCP" = 1 ]; then
  echo "   skipped (--skip-mcp). Register later with: bash scripts/register-mcp.sh"
elif command -v claude >/dev/null 2>&1; then
  bash scripts/register-mcp.sh | sed 's/^/   /' || echo "   ! MCP registration did not complete; read the lines above, then run scripts/register-mcp.sh"
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
 Done. Next, build a voice from your own writing.

 Easiest: open Claude Code in this folder and say "set up my voice".
 It will ask what you write, where the files are, and walk you through it.

 By hand:
   .venv/bin/python -m mimesis_voice.cli inspect ~/path/to/your/writing
   .venv/bin/python -m mimesis_voice.cli new myvoice --from ~/path/to/your/writing --author "Your Name"
   (Windows: use .venv\Scripts\python.exe)

 Re-run `bash install.sh` any time; it is safe and skips finished work.
===========================================================
EOF
exit $status
