@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
set "VLA_EXIT=%errorlevel%"
if not "%VLA_EXIT%"=="0" pause
exit /b %VLA_EXIT%
