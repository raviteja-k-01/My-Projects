<#
.SYNOPSIS
Starts the supported VisionTrack video processor.

.DESCRIPTION
This script is a convenience wrapper. It selects the project Python interpreter,
adds src to Python's import path, and forwards the chosen options to
the VisionTrack application entry point. Detection, tracking, event generation,
and database persistence are implemented in the Python package rather than in this script.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Video = "1",

    [ValidateSet("off", "local", "cloud")]
    [string]$Database = "off",

    [string]$DatabaseName,

    [string]$Model = "six_class_final",
    [string]$Device = "auto",
    [switch]$NoDisplay,
    [string]$OutputVideo,

    [ValidateRange(1, 2147483647)]
    [int]$MaxFrames,

    [switch]$EstimateMissingBoxes,
    [switch]$HideConveyorBelt
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $VirtualEnvironmentPython) {
    $Python = $VirtualEnvironmentPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python was not found. Create .venv and install requirements as described in README.md."
    }
    $Python = $PythonCommand.Source
}

Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if ($DatabaseName) {
    if ($Database -eq "off") {
        throw "-DatabaseName requires -Database local or -Database cloud."
    }
    $ProfileName = $Database.ToUpperInvariant() + "_DB_NAME"
    [System.Environment]::SetEnvironmentVariable(
        $ProfileName,
        $DatabaseName,
        "Process"
    )
}

$CliArguments = @(
    "-m", "visiontrack",
    "video",
    "--video", $Video,
    "--db-mode", $Database,
    "--model-key", $Model,
    "--device", $Device
)

if ($NoDisplay) {
    $CliArguments += "--no-display"
}

if ($OutputVideo) {
    $CliArguments += @("--output-video", $OutputVideo)
}

if ($MaxFrames -gt 0) {
    $CliArguments += @("--max-frames", $MaxFrames)
}

if ($EstimateMissingBoxes) {
    $CliArguments += "--estimate-missing-boxes"
}

if ($HideConveyorBelt) {
    $CliArguments += "--hide-conveyor-belt"
}

Write-Host "Starting VisionTrack ($Database database mode)..." -ForegroundColor Cyan
& $Python @CliArguments

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
