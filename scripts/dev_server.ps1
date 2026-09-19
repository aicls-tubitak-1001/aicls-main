<#
.SYNOPSIS
  Run the FastAPI development server with reload (same as `poe dev`).
.DESCRIPTION
  Loads .env if present (pydantic-settings does this too), then starts
  uvicorn aicls.api.app:create_app --factory --reload on 127.0.0.1:8000.
  OpenAPI UI: http://127.0.0.1:8000/docs
.PARAMETER Port
  TCP port (default 8000).
.PARAMETER Profile
  AICLS_PROFILE override: full | collect
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [ValidateSet("full", "collect")][string]$Profile = "full"
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$VenvPython = Join-Path $env:USERPROFILE ".venvs\ai-cls\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = "python" }
$env:AICLS_PROFILE = $Profile
$env:PYTHONPATH = "src"
& $VenvPython -m uvicorn aicls.api.app:create_app --factory --reload --host 127.0.0.1 --port $Port
