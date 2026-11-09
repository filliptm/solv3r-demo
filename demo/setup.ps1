# One-time setup of the Solv3R demo on Windows: Python 3.11 environment with CUDA PyTorch, the solv3r package,
# the server dependencies, and the web client. Run from anywhere: .\demo\setup.ps1
#   -Cuda cu128   PyTorch CUDA build (cu128 supports Blackwell / RTX 50xx and older GPUs)
param([string]$Cuda = "cu128")
$ErrorActionPreference = "Stop"
$demo = $PSScriptRoot
$root = Split-Path $demo -Parent
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is required: https://docs.astral.sh/uv/" }
if (-not (Test-Path "solv3r")) { git clone https://github.com/ubc-vision/solv3r.git }
if (-not (Test-Path "ml-sharp")) { git clone --depth 1 https://github.com/apple/ml-sharp.git }
if (-not (Test-Path ".venv")) { uv venv .venv --python 3.11 }

$env:VIRTUAL_ENV = Join-Path $root ".venv"
uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
uv pip install -e .\solv3r -r (Join-Path $demo "requirements.txt")
# SHARP pins torch 2.8; its code runs on the newer torch above, so it is installed without its pins
uv pip install --no-deps -e .\ml-sharp

Push-Location (Join-Path $demo "web")
try { npm install; npm run build } finally { Pop-Location }

& (Join-Path $root ".venv\Scripts\python.exe") -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
Write-Host "Done. Start the demo with .\demo\run.ps1"
