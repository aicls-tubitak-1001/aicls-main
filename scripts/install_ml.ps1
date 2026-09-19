<#
.SYNOPSIS
  GATED (approval gate 3, CUDA_INSTALL — ADR-0012): install the CUDA PyTorch stack for the RTX 5090.

.DESCRIPTION
  Refuses to run unless ALL THREE hold:
    1. -IHaveApproval switch (CLI flag)
    2. $env:AICLS_GATE_CUDA_INSTALL -eq "approved"
    3. an `| CUDA_INSTALL | approved | ... |` row in docs/gelistirme/onay-defteri.md (latest row wins)
  Then:
    - pins the PyTorch CUDA 12.8 wheel index (https://download.pytorch.org/whl/cu128) — Blackwell sm_120
    - installs torch / torchvision / torchaudio from that index, then peft + accelerate
    - verifies torch.cuda.get_device_capability() == (12, 0) and a bf16 matmul smoke test
    - appends an entry to $env:AICLS_DATA_ROOT\gate_invocations.log
  Exit code 3 on any refused gate condition; 1 on a failed post-install check.

.PARAMETER IHaveApproval
  Explicit acknowledgement that the CUDA_INSTALL gate was approved and recorded in the ledger.

.PARAMETER TorchVersion
  Torch version pin (default 2.11.0). Must exist on the cu128 index.

.PARAMETER DryRun
  Print the pip command and checks without installing.

.EXAMPLE
  $env:AICLS_GATE_CUDA_INSTALL = "approved"; pwsh scripts/install_ml.ps1 -IHaveApproval
#>
[CmdletBinding()]
param(
    [switch]$IHaveApproval,
    [Alias("Approve")][switch]$ApproveAlias,
    [string]$TorchVersion = "2.11.0",
    [string]$TorchvisionVersion = "0.26.0",
    [string]$TorchaudioVersion = "2.11.0",
    [string]$IndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$GateName = "CUDA_INSTALL"
$Ledger = Join-Path $RepoRoot "docs\gelistirme\onay-defteri.md"
$VenvPython = Join-Path $env:USERPROFILE ".venvs\ai-cls\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = "python" }

if ($ApproveAlias) { $IHaveApproval = $true }

# ---- triple lock ----------------------------------------------------------------------------
$missing = @()
if (-not $IHaveApproval) { $missing += "CLI flag -IHaveApproval (or -Approve) not given" }
if ($env:AICLS_GATE_CUDA_INSTALL -ne "approved") { $missing += "AICLS_GATE_CUDA_INSTALL is not 'approved'" }

$ledgerOk = $false
if (Test-Path $Ledger) {
    $rows = Get-Content $Ledger -Encoding utf8 | Where-Object { $_ -match "^\|\s*$GateName\s*\|" }
    if ($rows) {
        $last = $rows[-1]
        $cells = ($last.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
        $ledgerOk = ($cells.Count -ge 2 -and $cells[1] -eq "approved")
    }
}
if (-not $ledgerOk) { $missing += "no 'approved' row for $GateName in docs/gelistirme/onay-defteri.md" }
if ($env:CI -or $env:GITHUB_ACTIONS) { $missing += "refusing to install the CUDA stack in CI" }

if ($missing.Count -gt 0) {
    Write-Host "GATE $GateName REFUSED (ADR-0012). Missing conditions:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 3
}

# ---- install --------------------------------------------------------------------------------
$pipArgs = @(
    "-m", "pip", "install", "--upgrade",
    "--index-url", $IndexUrl,
    "torch==$TorchVersion", "torchvision==$TorchvisionVersion", "torchaudio==$TorchaudioVersion"
)
$extraArgs = @("-m", "pip", "install", "--upgrade", "peft>=0.13,<1", "accelerate>=1.0,<2")

Write-Host "Installing CUDA stack from $IndexUrl (torch $TorchVersion, sm_120 / Blackwell)"
if ($DryRun) {
    Write-Host "DRY RUN: $VenvPython $($pipArgs -join ' ')"
    Write-Host "DRY RUN: $VenvPython $($extraArgs -join ' ')"
} else {
    & $VenvPython @pipArgs
    & $VenvPython @extraArgs
}

# ---- post-install verification (sm_120 + bf16 smoke) -----------------------------------------
$check = @'
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() is False"); sys.exit(1)
cap = torch.cuda.get_device_capability()
print("device", torch.cuda.get_device_name(0), "capability", cap)
if cap != (12, 0):
    print("FAIL: expected compute capability (12, 0) for RTX 5090 (sm_120), got", cap); sys.exit(1)
a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
c = (a @ b).float()
if not torch.isfinite(c).all():
    print("FAIL: bf16 matmul produced non-finite values"); sys.exit(1)
print("OK: sm_120 + bf16 matmul smoke test passed")
'@

if ($DryRun) {
    Write-Host "DRY RUN: would verify torch.cuda.get_device_capability() == (12, 0) and bf16 matmul"
} else {
    $check | & $VenvPython -
    if ($LASTEXITCODE -ne 0) { Write-Host "Post-install verification failed" -ForegroundColor Red; exit 1 }
}

# ---- audit log ------------------------------------------------------------------------------
$dataRoot = $env:AICLS_DATA_ROOT
if ($dataRoot) {
    New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
    $sha = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
    $driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null)
    $line = "{0}`t{1}`t{2}`t{3}`tgit={4}`tindex={5}`ttorch={6}`tdriver={7}`tdry_run={8}" -f `
        (Get-Date -Format "o"), $GateName, $env:USERNAME, $env:COMPUTERNAME, $sha, $IndexUrl, $TorchVersion, $driver, $DryRun.IsPresent
    Add-Content -Path (Join-Path $dataRoot "gate_invocations.log") -Value $line -Encoding utf8
}

Write-Host "Done. Record driver/toolkit versions in the ledger row scope (docs/gelistirme/onay-defteri.md)."
