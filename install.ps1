# One-command install for Windows.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1               # everything
#   powershell -ExecutionPolicy Bypass -File install.ps1 -SkipMcp      # leave Claude Code's MCP config alone
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoStructure  # skip the optional spaCy tagger (~50 MB)
#
# Same five steps as install.sh: venv, package, build and calibrate every
# profile that ships with a corpus but no store, register the MCP server with
# Claude Code as 'mimesis', run the doctor on each voice.
#
# Safe to re-run. Ingest hash-skips an unchanged corpus, calibration is
# deterministic, and MCP registration is remove-then-add so it never duplicates.

param([switch]$SkipMcp, [switch]$NoStructure)
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
if ($env:MIMESIS_SKIP_MCP -eq '1') { $SkipMcp = $true }

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
# Optional: the spaCy tagger behind the opener-class line of the structure
# report. Everything else works without it, so a failure here is a note.
if (-not $NoStructure) {
    & $vpy -c "import spacy; spacy.load('en_core_web_sm')" *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   spaCy tagger already present"
    } else {
        & $vpy -m pip install --quiet -e ".[structure]"
        $ok = ($LASTEXITCODE -eq 0)
        if ($ok) { & $vpy -m spacy download en_core_web_sm *> $null; $ok = ($LASTEXITCODE -eq 0) }
        if ($ok) { Write-Host "   spaCy tagger installed (opener check on)" }
        else {
            Write-Host "   ! spaCy tagger not installed; the opener line will report unavailable."
            Write-Host "     Retry later: .venv\Scripts\python.exe -m pip install -e .[structure]; .venv\Scripts\python.exe -m spacy download en_core_web_sm"
        }
    }
}

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
if ($SkipMcp) {
    Write-Host "   skipped (-SkipMcp). Register later with: powershell -File scripts\register-mcp.ps1"
} elseif (Get-Command claude -ErrorAction SilentlyContinue) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'scripts\register-mcp.ps1')
    if ($LASTEXITCODE -ne 0) { Write-Host "   ! MCP registration did not complete; read the lines above, then run scripts\register-mcp.ps1" }
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
 Done. Next, build a voice from your own writing.

 Easiest: open Claude Code in this folder and say "set up my voice".
 It will ask what you write, where the files are, and walk you through it.

 By hand:
   .venv\Scripts\python.exe -m mimesis_voice.cli inspect "C:\path\to\my\writing"
   .venv\Scripts\python.exe -m mimesis_voice.cli new myvoice --from "C:\path\to\my\writing" --author "Your Name"

 Re-run install.ps1 any time; it is safe and skips finished work.
===========================================================
'@
exit $status
