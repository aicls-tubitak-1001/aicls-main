<#
.SYNOPSIS
  Bootstrap the AI-CLS development environment on Windows (ADR-0011).

.DESCRIPTION
  - Creates the virtual environment OUTSIDE OneDrive (C:\Users\<user>\.venvs\ai-cls, --system-site-packages
    so the system sympy / torch-cpu / transformers are reused)
  - Installs the project with the pure-Python "dev" extras (pip install -e .[dev])
  - Installs pre-commit hooks (pre-commit, commit-msg, pre-push)
  - Creates the data root outside OneDrive (D:\aicls-data, else C:\aicls-data)
  - Sets PYTHONPYCACHEPREFIX for the user
  Nothing gated happens here: no CUDA torch, no datasets, no weights (ADR-0012).

.PARAMETER VenvPath
  Where to create the venv. Default: $env:USERPROFILE\.venvs\ai-cls

.PARAMETER SkipInstall
  Only create the venv and data root; do not run pip.

.EXAMPLE
  pwsh scripts/bootstrap.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvPath = (Join-Path $env:USERPROFILE ".venvs\ai-cls"),
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-UnderOneDrive([string]$Path) {
    return ($Path -match '(?i)[\\/]OneDrive')
}

if (Test-UnderOneDrive $VenvPath) {
    throw "Refusing to create the venv under OneDrive: $VenvPath (ADR-0011)"
}
if (Test-UnderOneDrive $RepoRoot) {
    Write-Warning "Repository is under OneDrive ($RepoRoot). Expect sync noise; move to C:\src\ai-cls once the remote exists."
}

# --- Python 3.11 -----------------------------------------------------------------------------
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) { $PythonLauncher = @("py", "-3.11") } else { $PythonLauncher = @("python") }
$ver = & $PythonLauncher[0] $PythonLauncher[1..($PythonLauncher.Length)] -c "import sys;print('%d.%d'%sys.version_info[:2])"
if ($ver -ne "3.11") { throw "Python 3.11 required, found $ver" }

# --- venv ------------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "Creating venv at $VenvPath (--system-site-packages)"
    & $PythonLauncher[0] $PythonLauncher[1..($PythonLauncher.Length)] -m venv --system-site-packages $VenvPath
} else {
    Write-Host "venv already present: $VenvPath"
}
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

# --- data root (outside OneDrive) ------------------------------------------------------------
$DataRoot = $env:AICLS_DATA_ROOT
if (-not $DataRoot) {
    $DataRoot = if (Test-Path "D:\") { "D:\aicls-data" } else { "C:\aicls-data" }
}
if (Test-UnderOneDrive $DataRoot) { throw "AICLS_DATA_ROOT must be outside OneDrive: $DataRoot" }
foreach ($sub in @("", "models", "runs", "mlruns", "audio\raw", "datasets")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DataRoot $sub) | Out-Null
}
Write-Host "Data root: $DataRoot"

# --- user environment variables --------------------------------------------------------------
$CachePrefix = Join-Path $env:USERPROFILE ".cache\ai-cls\pycache"
New-Item -ItemType Directory -Force -Path $CachePrefix | Out-Null
[Environment]::SetEnvironmentVariable("PYTHONPYCACHEPREFIX", $CachePrefix, "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
if (-not [Environment]::GetEnvironmentVariable("AICLS_DATA_ROOT", "User")) {
    [Environment]::SetEnvironmentVariable("AICLS_DATA_ROOT", $DataRoot, "User")
}
$env:PYTHONPYCACHEPREFIX = $CachePrefix
$env:AICLS_DATA_ROOT = $DataRoot

# --- install ---------------------------------------------------------------------------------
if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip
    if (Test-Path "requirements\base.txt" -PathType Leaf) {
        # hash-locked path (poe lock)
        & $VenvPython -m pip install --require-hashes -r requirements\base.txt -r requirements\dev.txt
        & $VenvPython -m pip install --no-deps -e .
    } else {
        & $VenvPython -m pip install -e ".[dev]"
    }
    & $VenvPython -m pre_commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
}

# --- .env ------------------------------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    $secret = & $VenvPython -c "import secrets;print(secrets.token_hex(32))"
    (Get-Content ".env") -replace '^AICLS_JWT_SECRET=$', "AICLS_JWT_SECRET=$secret" | Set-Content ".env" -Encoding utf8
    (Get-Content ".env") -replace '^AICLS_DATA_ROOT=.*$', "AICLS_DATA_ROOT=$DataRoot" | Set-Content ".env" -Encoding utf8
    Write-Host "Created .env with a fresh AICLS_JWT_SECRET"
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  $VenvPath\Scripts\Activate.ps1"
Write-Host "  poe env-check ; poe test-fast"
Write-Host "  (GPU stack is GATED: scripts/install_ml.ps1 -IHaveApproval after the ledger row — ADR-0012)"
