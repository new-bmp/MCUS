@echo off
setlocal
title alice blue Launcher
cd /d "%~dp0"

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo PowerShell was not found on this computer.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
set "VLA_EXIT=%errorlevel%"
if not "%VLA_EXIT%"=="0" (
  echo.
  echo alice blue failed to start. Review the error above.
  pause
)
exit /b %VLA_EXIT%
