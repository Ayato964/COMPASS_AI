@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul 2>nul
set "ELECTRON_RUN_AS_NODE="
cd /d "%~dp0"

if exist "%~dp0frontend\node_modules" goto :check_dist
echo [COMPASS AI] Installing frontend dependencies...
pushd "%~dp0frontend"
call npm install
set "NPM_INSTALL_ERR=%ERRORLEVEL%"
popd
if not "%NPM_INSTALL_ERR%"=="0" goto :err_install

:check_dist
if exist "%~dp0frontend\dist\index.html" goto :launch
echo [COMPASS AI] Building Chromium frontend production bundle...
pushd "%~dp0frontend"
call npm run build
set "NPM_BUILD_ERR=%ERRORLEVEL%"
popd
if not "%NPM_BUILD_ERR%"=="0" goto :err_build

:launch
pushd "%~dp0frontend"
if exist ".\node_modules\electron\dist\electron.exe" (
    start "" /wait ".\node_modules\electron\dist\electron.exe" .
) else (
    call npx electron .
)
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:err_install
echo [COMPASS AI] npm install failed. Please verify your Node.js installation.
if /I not "%COMPASS_NO_PAUSE%"=="1" pause
exit /b 1

:err_build
echo [COMPASS AI] Frontend build failed.
if /I not "%COMPASS_NO_PAUSE%"=="1" pause
exit /b 1
