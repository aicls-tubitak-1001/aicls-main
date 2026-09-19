<#
.SYNOPSIS
  Skeleton: reproduce a progress-report snapshot from a clean clone (docs/raporlama/ip-eslestirme.md §2).

.DESCRIPTION
  Given a `report-YYYY-MM` tag, this script (when the eval/report CLI lands):
    1. clones the repository at the tag into a scratch directory outside OneDrive
    2. bootstraps a venv (scripts/bootstrap.ps1 -SkipInstall + hash-locked requirements)
    3. runs `poe ci`
    4. re-runs every gate-free evaluation (`aicls eval parser|cas|nlu|book|latency`) on the frozen
       datasets/ and compares the produced reports/metrics/*.json with the tagged latest.json
       (fingerprint + value tolerance from ADR-0015)
    5. runs `poe report-check` and `python tools/check_docs_map.py --strict`
    6. builds the docs site and prints the summary table (docs/raporlama/sonuclar.md)
  Gated evaluations (real OCR/ASR test sets) are only re-run when the ledger rows exist and the
  data root is present; otherwise they are reported as "not reproduced here".

.PARAMETER Tag
  Report tag, e.g. report-2027-05
.PARAMETER Workdir
  Scratch directory (default: C:\src\aicls-report)
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Workdir = "C:\src\aicls-report"
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ($Workdir -match '(?i)OneDrive') { throw "Workdir must be outside OneDrive: $Workdir" }
if ($Tag -notmatch '^report-\d{4}-\d{2}$') { throw "Tag must look like report-YYYY-MM: $Tag" }

Write-Host "== reproduce_report: $Tag -> $Workdir"
New-Item -ItemType Directory -Force -Path $Workdir | Out-Null
$Clone = Join-Path $Workdir $Tag
if (-not (Test-Path $Clone)) {
    git clone --branch $Tag --depth 1 $RepoRoot $Clone
}
Set-Location $Clone

Write-Host "== step 2: environment (skeleton — bootstrap with hash-locked requirements)"
& (Join-Path $Clone "scripts\bootstrap.ps1") -VenvPath (Join-Path $Workdir "venv-$Tag")
$VenvPython = Join-Path $Workdir "venv-$Tag\Scripts\python.exe"

Write-Host "== step 3: poe ci"
& $VenvPython -m poethepoet ci

Write-Host "== step 4: evaluations (added when `aicls eval` lands; frozen datasets under datasets/)"
Write-Host "   TODO: aicls eval parser|cas|nlu|book|latency --compare reports/metrics/latest.json"

Write-Host "== step 5: report-check + strict docs map"
& $VenvPython tools/check_docs_map.py --strict
& $VenvPython -m poethepoet report-check

Write-Host "== step 6: docs"
& $VenvPython -m mkdocs build --strict
Write-Host "Done: $Clone\site"
