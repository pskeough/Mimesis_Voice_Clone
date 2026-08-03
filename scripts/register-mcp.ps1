<#
.SYNOPSIS
  Register the Mimesis v2 MCP server with Claude Code as 'mimesis-v2'.

.DESCRIPTION
  Adds a stdio MCP server named 'mimesis-v2' at user scope, pointing at this
  repo's venv Python and the server module. It NEVER touches an existing
  'mimesis' registration, so the v1 server keeps working during transition.

  This script does not run itself as part of the build; run it manually when you
  are ready to wire the server into Claude Code.
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$claude = (Get-Command claude -ErrorAction SilentlyContinue)
if (-not $claude) {
    Write-Host "claude CLI not found on PATH. Install Claude Code, then re-run."
    exit 1
}

# Remove first so re-running never duplicates. Only touches 'mimesis-v2'. On a
# first run the server does not exist yet and `remove` exits non-zero; that is
# expected, so this one call must not abort the script under -ErrorAction Stop.
try { & claude mcp remove mimesis-v2 --scope user 2>$null | Out-Null } catch { }
& claude mcp add --scope user --transport stdio mimesis-v2 -- "$py" "-m" "mimesis_voice.server"
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to register 'mimesis-v2' (claude mcp add exited $LASTEXITCODE)."; exit 1 }
Write-Host "Registered 'mimesis-v2' (user scope). The existing 'mimesis' server was left untouched."
