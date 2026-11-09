# Starts the Solv3R demo on http://127.0.0.1:7860 (builds the web client first if it is missing).
#   .\demo\run.ps1                    # local only
#   .\demo\run.ps1 -BindHost 0.0.0.0  # reachable from the network (the webcam needs localhost or HTTPS)
#   .\demo\run.ps1 -Rebuild           # rebuild the web client
param(
    [int]$Port = 7860,
    [string]$BindHost = "127.0.0.1",
    [switch]$Rebuild
)
$ErrorActionPreference = "Stop"
$demo = $PSScriptRoot
$root = Split-Path $demo -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No Python environment at $root\.venv. Run .\demo\setup.ps1 first." }

$dist = Join-Path $demo "web\dist\index.html"
if ($Rebuild -or -not (Test-Path $dist)) {
    Push-Location (Join-Path $demo "web")
    try {
        if (-not (Test-Path "node_modules")) { npm install }
        npm run build
    } finally { Pop-Location }
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location $demo
# uvicorn logs to stderr; Windows PowerShell would turn that into terminating errors under "Stop"
$ErrorActionPreference = "Continue"
Write-Host "Solv3R demo: http://${BindHost}:${Port}  (Ctrl+C to stop)"
& $py -m uvicorn server.app:app --host $BindHost --port $Port
