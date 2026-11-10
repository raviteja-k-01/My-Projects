<#
.SYNOPSIS
Starts the packaged VisionTrack Streamlit dashboard.

.DESCRIPTION
This launcher selects either the local or cloud database profile and passes it
to Sarfaraz's dashboard through its existing DB_* environment contract. The
dashboard UI source is not modified or imported into the inference process.
#>
[CmdletBinding()]
param(
    [ValidateSet("local", "cloud")]
    [string]$Database = "cloud",

    [string]$BindAddress = "127.0.0.1",

    [ValidateRange(1, 65535)]
    [int]$Port = 8501,

    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $VirtualEnvironmentPython) {
    $Python = $VirtualEnvironmentPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python was not found. Create .venv and install requirements.txt."
    }
    $Python = $PythonCommand.Source
}

Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$DashboardArguments = @(
    "-m", "visiontrack",
    "dashboard",
    "--database", $Database,
    "--bind-address", $BindAddress,
    "--port", $Port
)

if ($Headless) {
    $DashboardArguments += "--headless"
}

Write-Host "Starting VisionTrack dashboard ($Database database)..." -ForegroundColor Cyan
& $Python @DashboardArguments
exit $LASTEXITCODE
