@echo off
setlocal

if /I "%~1"=="help" goto :help
if /I "%~1"=="--help" goto :help
if /I "%~1"=="-h" goto :help
if /I "%~1"=="dashboard" goto :dashboard

set "VIDEO=%~1"
set "DATABASE=%~2"
set "DEVICE=%~3"
set "DISPLAY_MODE=%~4"

if not defined VIDEO set "VIDEO=1"
if not defined DATABASE set "DATABASE=local"
if not defined DEVICE set "DEVICE=gpu"

if /I "%DEVICE%"=="gpu" set "DEVICE=cuda:0"

if /I "%DATABASE%"=="local" goto :local_database
if /I "%DATABASE%"=="cloud" goto :cloud_database
if /I "%DATABASE%"=="off" goto :database_off

echo Invalid database mode: %DATABASE%
echo Choose local, cloud, or off.
exit /b 2

:local_database
set "DATABASE_ARGUMENTS=-Database local -DatabaseName visiontrack_local"
goto :run

:cloud_database
set "DATABASE_ARGUMENTS=-Database cloud"
goto :run

:database_off
set "DATABASE_ARGUMENTS=-Database off"

:run
set "DISPLAY_ARGUMENT="
if /I "%DISPLAY_MODE%"=="headless" set "DISPLAY_ARGUMENT=-NoDisplay"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_video.ps1" -Video "%VIDEO%" %DATABASE_ARGUMENTS% -Device "%DEVICE%" %DISPLAY_ARGUMENT%
exit /b %ERRORLEVEL%

:dashboard
set "DASHBOARD_DATABASE=%~2"
set "DASHBOARD_DISPLAY=%~3"
if not defined DASHBOARD_DATABASE set "DASHBOARD_DATABASE=cloud"

if /I "%DASHBOARD_DATABASE%"=="local" goto :dashboard_database_ok
if /I "%DASHBOARD_DATABASE%"=="cloud" goto :dashboard_database_ok

echo Invalid dashboard database: %DASHBOARD_DATABASE%
echo Choose local or cloud.
exit /b 2

:dashboard_database_ok
set "DASHBOARD_ARGUMENTS="
if /I "%DASHBOARD_DISPLAY%"=="headless" set "DASHBOARD_ARGUMENTS=-Headless"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_dashboard.ps1" -Database "%DASHBOARD_DATABASE%" %DASHBOARD_ARGUMENTS%
exit /b %ERRORLEVEL%

:help
echo VisionTrack short launcher
echo.
echo Video:     run [video] [database] [device] [headless]
echo Dashboard: run dashboard [local^|cloud] [headless]
echo.
echo Examples:
echo   run 43 local gpu
echo   run 43 local gpu headless
echo   run dashboard
echo   run dashboard local
echo   run dashboard cloud headless
echo   run 1 off cpu
echo.
echo Video defaults: video=1, database=local, device=gpu
echo Dashboard default: database=cloud
exit /b 0
