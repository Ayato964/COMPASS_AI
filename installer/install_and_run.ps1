# COMPASS AI - Windows One-Click Installer & Launcher (PowerShell)
param(
    [switch]$SetupOnly = $false,
    [switch]$SkipPip = $false,
    [switch]$OpenModelsDir = $false,
    [switch]$ServerOnly = $false,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs = @()
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location -LiteralPath $RootDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " COMPASS AI - ローカル音楽生成エンジン 自動構築・起動システム" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

function Test-RealPython {
    try {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $cmd) { return $false }
        $out = & python -c "import sys; print(sys.version_info[0])" 2>$null
        return ($LASTEXITCODE -eq 0 -and $out -match "^3")
    } catch {
        return $false
    }
}

$IsSetupOnly = $SetupOnly -or ($RemainingArgs -contains "--setup-only") -or ($RemainingArgs -contains "--SetupOnly")
$IsSkipPip = $SkipPip -or ($RemainingArgs -contains "--skip-pip") -or ($RemainingArgs -contains "--SkipPip")
$IsOpenModels = $OpenModelsDir -or ($RemainingArgs -contains "--open-models-dir") -or ($RemainingArgs -contains "--OpenModelsDir")
$IsServerOnly = $ServerOnly -or ($RemainingArgs -contains "--server-only") -or ($RemainingArgs -contains "--ServerOnly")

$FilteredRemaining = @()
foreach ($arg in $RemainingArgs) {
    if ($arg -notin @("--setup-only", "--SetupOnly", "--skip-pip", "--SkipPip", "--open-models-dir", "--OpenModelsDir", "--server-only", "--ServerOnly")) {
        $FilteredRemaining += $arg
    }
}

$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$CreatedNewVenv = $false

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (-not (Test-RealPython)) {
        Write-Host "[COMPASS AI] Python 3.11 を winget 経由で自動インストールしています..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    Write-Host "[COMPASS AI] 専用仮想環境 (.venv) を作成しています..." -ForegroundColor Green
    if ($IsSkipPip) {
        python -m venv --system-site-packages .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $CreatedNewVenv = $true
}

# Step 1: Provision & verify environment ONLY (`--setup-only`).
# Rollback of `$CreatedNewVenv` happens strictly if provisioning itself fails — NEVER on runtime server exit!
$SetupArgs = @("installer\bootstrap.py", "--setup-only")
if ($IsSkipPip) { $SetupArgs += "--skip-pip" }
if ($IsOpenModels) { $SetupArgs += "--open-models-dir" }
if ($FilteredRemaining.Count -gt 0) { $SetupArgs += $FilteredRemaining }

& $VenvPython @SetupArgs
$SetupExit = $LASTEXITCODE
if ($SetupExit -ne 0) {
    if ($CreatedNewVenv -and (Test-Path -LiteralPath $VenvDir)) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    exit $SetupExit
}

# Provisioning succeeded: permanently commit `.venv`.
$CreatedNewVenv = $false

if ($IsSetupOnly) {
    exit 0
}

# Step 2: Launch full COMPASS AI Desktop Studio when Node/npm is available (unless `--server-only`),
# otherwise launch the local FastAPI engine on a verified conflict-free port without risking `.venv`.
$StartBat = Join-Path $RootDir "start_compass_ai.bat"
$HasNpm = $null -ne (Get-Command npm -ErrorAction SilentlyContinue)
if ((-not $IsServerOnly) -and (Test-Path -LiteralPath $StartBat) -and $HasNpm) {
    Write-Host "[COMPASS AI] デスクトップワークステーションを起動しています..." -ForegroundColor Green
    & cmd.exe /d /c .\start_compass_ai.bat
    exit $LASTEXITCODE
}

$RuntimeArgs = @("installer\bootstrap.py", "--skip-pip")
if ($FilteredRemaining.Count -gt 0) { $RuntimeArgs += $FilteredRemaining }
& $VenvPython @RuntimeArgs
exit $LASTEXITCODE
