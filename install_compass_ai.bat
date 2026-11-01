@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul 2>nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install_and_run.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :fail
exit /b 0
:fail
echo.
echo [COMPASS AI] Setup or launch failed (Exit Code: %EXIT_CODE%). Please check the messages above.
if /I not "%COMPASS_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
