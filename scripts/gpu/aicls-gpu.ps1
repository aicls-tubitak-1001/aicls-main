<#
.SYNOPSIS
    Windows-side wrapper for scripts/gpu/aicls-gpu.sh (ADR-0020).

.DESCRIPTION
    The RTX 5090 is only reachable through WSL2 + nvidia-docker, because the Windows venv carries
    a CPU-only torch. This wrapper lets an operator stay in PowerShell:

        .\scripts\gpu\aicls-gpu.ps1 --print train --config training\configs\ocr\smoke_synthetic.yaml
        .\scripts\gpu\aicls-gpu.ps1 train --config training/configs/ocr/smoke_synthetic.yaml --synthetic

    Every argument is forwarded verbatim to the shell runner, so `aicls train --dry-run` keeps its
    own flag. The wrapper itself does three things and nothing else:

      1. translates the repository path to its /mnt/<drive>/... form;
      2. shares the approval-gate variables with WSL through WSLENV, so that a gated command
         behaves inside the container exactly as it does outside it (ADR-0012). It never sets one;
      3. translates a Windows-style AICLS_DATA_ROOT / AICLS_MODELS_ROOT into a POSIX path. A
         Windows path handed to the container becomes a RELATIVE PosixPath and silently resolves
         weights next to the working directory - the runner refuses one, and this is the
         conversion that keeps `D:\aicls-data` in a PowerShell profile from causing that refusal.

.NOTES
    Documentation: scripts/gpu/README.md,
    docs/adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md,
    docs/gelistirme/kurulum.md (7. bölüm), docs/ml/egitim-el-kitabi.md (2. bölüm).
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Gate variables (ADR-0012) plus the CI switches that make every gate refuse unconditionally.
# Forwarding them is a safety property: without it, running through Docker would silently drop
# the CI refusal and the environment half of the triple lock.
$GateVariables = @(
    "AICLS_GATE_DATASET_DOWNLOAD",
    "AICLS_GATE_WEIGHTS_DOWNLOAD",
    "AICLS_GATE_CUDA_INSTALL",
    "AICLS_GATE_TRAINING",
    "CI",
    "GITHUB_ACTIONS"
)

# Ordinary settings that may travel with the operator's shell (never a secret, never a path).
$PassthroughVariables = @(
    "AICLS_ENV",
    "AICLS_PROFILE",
    "AICLS_LOG_LEVEL",
    "AICLS_TEST_GPU",
    "AICLS_TEST_DATA"
)

function ConvertTo-WslPath {
    <#
    .SYNOPSIS
        Translate C:\dir\file to /mnt/c/dir/file without calling wslpath.
    #>
    param([Parameter(Mandatory = $true)][string] $Path)

    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    if ($full.StartsWith("/")) {
        return $full
    }
    throw "'$Path' is not a local drive path; the GPU runner needs one it can bind-mount into WSL."
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "wsl.exe not found. The GPU path needs WSL2 + nvidia-docker (see scripts/gpu/README.md)."
    exit 2
}

$repoWindows = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $repoWindows ".git"))) {
    Write-Error ("'{0}' has no .git. The runner mounts the repository itself so that RunManifest can attribute the run to a commit; a copy without .git is not citable." -f $repoWindows)
    exit 2
}

$repoWsl = ConvertTo-WslPath $repoWindows
$runnerWsl = "$repoWsl/scripts/gpu/aicls-gpu.sh"

# ------------------------------------------------------------------------------------------
# data roots: convert, never forward, a Windows path
# ------------------------------------------------------------------------------------------
$extraArgs = @()
foreach ($pair in @(@("AICLS_DATA_ROOT", "--data-root"), @("AICLS_MODELS_ROOT", "--models-root"))) {
    $name = $pair[0]
    $flag = $pair[1]
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        if ($value -match '^[A-Za-z]:[\\/]' -or $value.Contains("\")) {
            $value = ConvertTo-WslPath $value
        }
        $extraArgs += @($flag, $value)
    }
}

# ------------------------------------------------------------------------------------------
# WSLENV: share the gate variables with the Linux side (values are copied, never invented)
# ------------------------------------------------------------------------------------------
$share = @()
foreach ($name in ($GateVariables + $PassthroughVariables)) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $share += $name
    }
}
if ($share.Count -gt 0) {
    $existing = $env:WSLENV
    $entries = @()
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        $entries += $existing.Split(":") | Where-Object { $_ -ne "" }
    }
    foreach ($name in $share) {
        if ($entries -notcontains $name) { $entries += $name }
    }
    $env:WSLENV = ($entries -join ":")
}

$distro = $env:AICLS_GPU_WSL_DISTRO
$wslArgs = @()
if (-not [string]::IsNullOrWhiteSpace($distro)) {
    $wslArgs += @("-d", $distro)
}
$wslArgs += @("-e", "bash", $runnerWsl)
$wslArgs += $extraArgs
if ($args.Count -gt 0) {
    $wslArgs += $args
}

# Windows PowerShell 5.1 wraps a native command's stderr in ErrorRecords whenever the caller
# redirects it (`... 2>&1 | ...`), and under `Stop` the first such record kills this script
# before the exit code is forwarded. `aicls` logs to stderr on every run, so this is the normal
# case, not an edge case: drop back to Continue for the child and let the exit code decide.
$ErrorActionPreference = "Continue"
& wsl.exe @wslArgs
exit $LASTEXITCODE
