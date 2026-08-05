# One-command install for Windows.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Same five steps as install.sh: venv, package, build and calibrate every
# profile that ships with a corpus but no store, register the MCP server with
# Claude Code, run the doctor on each voice.
#
# Safe to re-run. Ingest hash-skips an unchanged corpus, calibration is
# deterministic, and MCP registration is remove-then-add so it never duplicates.

$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot

function Write-Bold($m) { Write-Host $m -ForegroundColor White }
function Fail($what, $fix) {
    Write-Host "`n! $what"  -ForegroundColor Red
    Write-Host "  FIX: $fix"
    exit 1
}

Write-Bold "== 1/5  Python =="
$py = $null
foreach ($c in @('python3.13', 'python3.12', 'python3.11', 'python3', 'python', 'py')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    # The Microsoft Store stub on PATH is named python.exe and exits 9009 rather
    # than running anything, so version-probe every candidate instead of trusting
    # the name.
    & $cmd.Source -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $cmd.Source; break }
}
if (-not $py) {
    Fail "no Python 3.11 or newer found" "install Python 3.11+ from python.org (tick 'Add to PATH'), then re-run install.ps1"
}
Write-Host "   $(& $py --version) at $py"

Write-Bold "== 2/5  Virtual environment =="
if (-not (Test-Path .venv)) {
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv" "check disk space and permissions, then re-run" }
    Write-Host "   created .venv"
} else {
    Write-Host "   .venv already present (reusing)"
}
$vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $vpy)) { $vpy = Join-Path $PSScriptRoot '.venv\bin\python' }
if (-not (Test-Path $vpy)) { Fail ".venv exists but has no interpreter" "Remove-Item -Recurse -Force .venv, then re-run install.ps1" }

Write-Bold "== 3/5  Package =="
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet -e .
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" "re-run and read the error above" }
Write-Host "   mimesis-voice installed (fast backend; no torch)"

Write-Bold "== 4/5  Voices =="
$built = 0
$exts = @('.txt', '.md', '.docx', '.pdf')
foreach ($dir in (Get-ChildItem -Path profiles -Directory -ErrorAction SilentlyContinue)) {
    $slug = $dir.Name
    $src  = Join-Path $dir.FullName 'source_documents'
    if (-not (Test-Path $src)) { continue }
    # Any real document at all? An empty corpus is a profile stub, not a voice.
    $docs = Get-ChildItem -Path $src -File -ErrorAction SilentlyContinue |
            Where-Object { $exts -contains $_.Extension.ToLower() }
    if (-not $docs) { continue }
    if ((Test-Path (Join-Path $dir.FullName 'data\store.sqlite')) -and
        (Test-Path (Join-Path $dir.FullName 'data\fingerprint.json'))) {
        Write-Host "   ${slug}: already built (skipping; delete data\ to force a rebuild)"
        continue
    }
    Write-Host "   ${slug}: ingesting..."
    & $vpy -m mimesis_voice.cli ingest $slug *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     ! ingest failed for $slug; run: .venv\Scripts\python.exe -m mimesis_voice.cli ingest $slug"
        continue
    }
    Write-Host "   ${slug}: calibrating..."
    # No 2>&1 here: in Windows PowerShell 5.1 redirecting a native command's
    # stderr wraps each line in a NativeCommandError and sets $? false even on a
    # clean exit. The CLI reports on stdout, so there is nothing to merge.
    & $vpy -m mimesis_voice.cli calibrate $slug |
        Select-String -Pattern 'self-baseline|THIN' |
        ForEach-Object { "     $_" }
    $built++
}
if ($built -eq 0) { Write-Host "   (nothing new to build)" }

Write-Bold "== 5/5  Claude Code =="
if (Get-Command claude -ErrorAction SilentlyContinue) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'scripts\register-mcp.ps1')
    if ($LASTEXITCODE -ne 0) { Write-Host "   ! MCP registration failed; run scripts\register-mcp.ps1 by hand" }
    # An older voice server left registered will keep answering voice requests,
    # and nothing about that failure looks like a failure.
    $listed = (& claude mcp list 2>$null) -join "`n"
    if ($listed -match '(?m)^\s*mimesis\s') {
        Write-Host "`n   NOTE: an older MCP server named 'mimesis' is still registered."
        Write-Host "   If that is a previous voice install, remove it so it cannot answer:"
        Write-Host "       claude mcp remove mimesis --scope user"
    }
} else {
    Write-Host "   claude CLI not on PATH; skipped MCP registration."
    Write-Host "   Install Claude Code, then run: powershell -File scripts\register-mcp.ps1"
}

Write-Bold "== Health =="
$status = 0
foreach ($dir in (Get-ChildItem -Path profiles -Directory -ErrorAction SilentlyContinue)) {
    $slug = $dir.Name
    if (-not (Test-Path (Join-Path $dir.FullName 'data\fingerprint.json'))) { continue }
    $line = (& $vpy -m mimesis_voice.cli doctor $slug |
             Select-String -Pattern 'RESULT' | Select-Object -First 1)
    if (-not $line) { $line = 'no result' }
    Write-Host ("   {0,-22} {1}" -f $slug, $line)
    if ("$line" -match 'fail') { $status = 1 }
}

Write-Host @'

===========================================================
 Done. Try a voice:

   .venv\Scripts\python.exe -m mimesis_voice.cli profile list
   .venv\Scripts\python.exe -m mimesis_voice.cli compose <voice> "a short note about your week"

 Or build a voice from a folder of your own writing:

   .venv\Scripts\python.exe -m mimesis_voice.cli new myvoice --from "C:\path\to\my\writing"

 Or just ask Claude Code, in this folder, to write something in your voice.
 Re-run install.ps1 any time; it is safe and skips finished work.
===========================================================
'@
exit $status
