$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

Write-Host "=== [1/4] Python MORTM 4.5 Model Layer Runtime Suite ==="
$PythonExe = "python"
$VenvCandidate = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvCandidate) {
    & $VenvCandidate -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('fastapi') else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $PythonExe = $VenvCandidate }
}
$TestScript = Join-Path $RepoRoot "tests\test_model_engine.py"
& $PythonExe $TestScript
if ($LASTEXITCODE -ne 0) { throw "Python model engine tests failed with exit code $LASTEXITCODE" }

Write-Host "`n=== [2/4] Frontend TypeScript + Vite Production Build ==="
$FrontendDir = Join-Path $RepoRoot "frontend"
Set-Location -LiteralPath $FrontendDir
npm run build
if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE" }
npx tsc --noEmit --skipLibCheck --allowJs --checkJs --noImplicitAny false --strict false --target es2022 --module commonjs --moduleResolution node --types node electron/ipc_security.cjs electron/main.cjs electron/preload.cjs electron/verify_e2e_audit.cjs
if ($LASTEXITCODE -ne 0) { throw "Electron CommonJS semantic check failed with exit code $LASTEXITCODE" }

Write-Host "`n=== [3/4] Frontend Vitest Unit & Component Suite ==="
npm run test
if ($LASTEXITCODE -ne 0) { throw "Frontend Vitest suite failed with exit code $LASTEXITCODE" }

Write-Host "`n=== [4/4] Physical Electron Chromium + Live Python FastAPI E2E & Anti-AI Design Audit ==="
$ElectronExe = Join-Path $FrontendDir "node_modules\electron\dist\electron.exe"
$AuditScript = Join-Path $FrontendDir "electron\verify_e2e_audit.cjs"
$PrevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    if (Test-Path -LiteralPath $ElectronExe) {
        & $ElectronExe $AuditScript 2>&1 | ForEach-Object { "$_" }
    } else {
        npx electron "electron\verify_e2e_audit.cjs" 2>&1 | ForEach-Object { "$_" }
    }
    $AuditExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PrevEap
}
if ($AuditExit -ne 0) { throw "Electron Chromium E2E & UI Design audit failed with exit code $AuditExit" }

Set-Location -LiteralPath $RepoRoot
Write-Host "`n[ALL PHYSICAL INTEGRITY & UI DESIGN CHECKS PASSED]"
