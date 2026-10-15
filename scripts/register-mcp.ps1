<#
.SYNOPSIS
  Register the Mimesis MCP server with Claude Code at user scope.

.DESCRIPTION
  Name: 'mimesis', or $env:MIMESIS_MCP_NAME if set. Safe to re-run.

  * A registration under that name that already points at THIS repo is
    replaced (remove, then add), so re-running never duplicates.
  * A registration under that name that points somewhere else is left alone
    and reported. Two voice servers answering the same requests is a silent
    failure, so this script will not create the second one.
  * The old 'mimesis-v2' name, used by earlier versions of this script, is
    removed when it points at this repo.
#>
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$name = if ($env:MIMESIS_MCP_NAME) { $env:MIMESIS_MCP_NAME } else { 'mimesis' }
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $repo ".venv\bin\python" }
if (-not (Test-Path $py)) { Write-Host "No venv interpreter under $repo\.venv. Run install.ps1 first."; exit 1 }

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host "claude CLI not found on PATH. Install Claude Code, then re-run."
    exit 1
}

# Lower-case, forward slashes: the same repo can be printed either way.
function Norm($s) { ("$s").ToLower().Replace('\', '/') }
$repoN = Norm $repo
function PointsHere($text) { (Norm $text).Contains($repoN) }
# `claude mcp get` exits non-zero when the name is not registered. No 2>&1 on a
# native command here: in Windows PowerShell 5.1 that wraps stderr lines in
# NativeCommandError records.
function McpGet($n) { $o = (& claude mcp get $n 2>$null) -join "`n"; if ($LASTEXITCODE -eq 0) { $o } else { '' } }

if ($name -ne 'mimesis-v2') {
    $legacy = McpGet 'mimesis-v2'
    if ($legacy -and (PointsHere $legacy)) {
        & claude mcp remove mimesis-v2 --scope user 2>$null | Out-Null
        Write-Host "Removed the old 'mimesis-v2' registration (same server, old name)."
    }
}

$existing = McpGet $name
if ($existing -and -not (PointsHere $existing)) {
    Write-Host "An MCP server named '$name' is already registered and points elsewhere:"
    $existing -split "`n" | ForEach-Object { Write-Host "    $_" }
    Write-Host "Not replacing it. Either remove it (claude mcp remove $name --scope user) and re-run,"
    Write-Host "or register this install under another name: `$env:MIMESIS_MCP_NAME='mimesis-2'; .\scripts\register-mcp.ps1"
    exit 2
}

& claude mcp remove $name --scope user 2>$null | Out-Null
& claude mcp add --scope user --transport stdio $name -- "$py" "-m" "mimesis_voice.server"
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to register '$name' (claude mcp add exited $LASTEXITCODE)."; exit 1 }
Write-Host "Registered '$name' (user scope) -> $py -m mimesis_voice.server"
Write-Host "Restart Claude Code (or start a new session) to load the tools."
