<#
.SYNOPSIS
Creates or resets an isolated local VisionTrack database and loads seed data.

.DESCRIPTION
This command refuses non-local hosts. It redirects hard-coded USE statements in
the supplied schema and seed files to the requested local database. Existing
tables are never replaced unless -Reset is explicitly supplied.
#>
[CmdletBinding()]
param(
    [string]$Database = "visiontrack_local",
    [string]$SnapshotPath = "database\seeds\visiontrack_demo.sql",
    [string[]]$MigrationPaths = @(
        "database\migrations\002_add_event_processor_fields.sql",
        "database\migrations\003_lifecycle_processor_indexes.sql"
    ),
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Database -notmatch "^[A-Za-z0-9_]+$") {
    throw "Database must contain only letters, numbers, and underscores."
}

foreach ($Path in @($SnapshotPath) + $MigrationPaths) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required SQL file was not found: $Path"
    }
}

$Environment = @{}
Get-Content -LiteralPath ".env" | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
        $Environment[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$HostName = $Environment["LOCAL_DB_HOST"]
if ([string]::IsNullOrWhiteSpace($HostName)) {
    $HostName = $Environment["DB_HOST"]
}

if ($HostName -notin @("localhost", "127.0.0.1", "::1")) {
    throw "Refusing to seed a non-local database host: $HostName"
}

$Port = $Environment["LOCAL_DB_PORT"]
if ([string]::IsNullOrWhiteSpace($Port)) {
    $Port = $Environment["DB_PORT"]
}

$User = $Environment["LOCAL_DB_USER"]
if ([string]::IsNullOrWhiteSpace($User)) {
    $User = $Environment["DB_USER"]
}

$Password = $Environment["LOCAL_DB_PASSWORD"]
if ([string]::IsNullOrWhiteSpace($Password)) {
    $Password = $Environment["DB_PASSWORD"]
}

if ([string]::IsNullOrWhiteSpace($User)) {
    throw "Local database user is missing from .env."
}

$MySql = Get-Command mysql -ErrorAction SilentlyContinue
if ($null -eq $MySql) {
    throw "The MySQL command-line client was not found on PATH."
}

$env:MYSQL_PWD = $Password
$ConnectionArguments = @(
    "--protocol=TCP",
    "--connect-timeout=5",
    "-h", $HostName,
    "-P", $Port,
    "-u", $User
)

$DatabaseExists = & $MySql.Source @ConnectionArguments -N -e (
    "SELECT COUNT(*) FROM information_schema.SCHEMATA " +
    "WHERE SCHEMA_NAME='$Database';"
)
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect local database $Database."
}

$ExistingObjectCount = 0
if ([int]$DatabaseExists -gt 0) {
    $ExistingObjectCount = & $MySql.Source @ConnectionArguments -N -e (
    "SELECT COUNT(*) FROM information_schema.TABLES " +
    "WHERE TABLE_SCHEMA='$Database';"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect objects in local database $Database."
    }
}

if ([int]$ExistingObjectCount -gt 0 -and -not $Reset) {
    throw (
        "Database $Database already contains objects. " +
        "Use -Reset only when you intend to replace that isolated local data."
    )
}

if ([int]$DatabaseExists -eq 0) {
    & $MySql.Source @ConnectionArguments -e (
        "CREATE DATABASE ``$Database`` " +
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create local database $Database."
    }
}

$UseStatement = 'USE `' + $Database + '`;'
$UsePattern = '(?im)^\s*USE\s+`[^`]+`;\s*$'
$CreateStatement = (
    'CREATE DATABASE IF NOT EXISTS `' + $Database + '` ' +
    'CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;'
)
$CreatePattern = '(?im)^\s*CREATE DATABASE\b.*;\s*$'

$Snapshot = Get-Content -LiteralPath $SnapshotPath -Raw
$Snapshot = [regex]::Replace($Snapshot, $CreatePattern, $CreateStatement, 1)
$Snapshot = [regex]::Replace($Snapshot, $UsePattern, $UseStatement, 1)
$Snapshot | & $MySql.Source @ConnectionArguments
if ($LASTEXITCODE -ne 0) {
    throw "Demo database import failed for $Database."
}

foreach ($MigrationPath in $MigrationPaths) {
    $Migration = Get-Content -LiteralPath $MigrationPath -Raw
    ($UseStatement + [Environment]::NewLine + $Migration) |
        & $MySql.Source @ConnectionArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Migration failed: $MigrationPath"
    }
}

Write-Host "Realtime Warehouse Vision demo database ready: $Database" -ForegroundColor Green
& $MySql.Source @ConnectionArguments --table -e (
    "SELECT 'boxes' AS item, COUNT(*) AS rows_loaded FROM ``$Database``.boxes " +
    "UNION ALL SELECT 'pallets', COUNT(*) FROM ``$Database``.pallets " +
    "UNION ALL SELECT 'orders', COUNT(*) FROM ``$Database``.orders " +
    "UNION ALL SELECT 'video_processing_runs', COUNT(*) " +
    "FROM ``$Database``.video_processing_runs;"
)
