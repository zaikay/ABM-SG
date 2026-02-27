param(
    [string]$RepoRoot = ".",
    [string]$WheelDir = "offline_wheels"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Online Wheel Preparation ==="
Set-Location $RepoRoot

if (-not (Test-Path ".\requirements.txt")) {
    throw "requirements.txt not found in $((Get-Location).Path)"
}

if (-not (Test-Path $WheelDir)) {
    New-Item -ItemType Directory -Path $WheelDir | Out-Null
}

Write-Host "Python version:"
python --version

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Downloading requirements wheels/sdists..."
python -m pip download -r .\requirements.txt -d $WheelDir

Write-Host "Downloading explicit extras (SALib, tqdm)..."
python -m pip download SALib tqdm -d $WheelDir

Write-Host "Downloading offline build tools..."
python -m pip download setuptools wheel build -d $WheelDir

Write-Host "Done. Wheel folder: $((Resolve-Path $WheelDir).Path)"
Write-Host "Copy this folder to offline machines."

