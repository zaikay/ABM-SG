param(
    [string]$RepoRoot = ".",
    [string]$WheelDir = "offline_wheels",
    [string]$VenvDir = ".venv",
    [switch]$RunSmoke,
    [string]$SmokeOutDir = "results/sobol/smoke",
    [int]$SmokeNBase = 8,
    [int]$SmokeSeed = 11,
    [int]$SmokeJobs = 2
)

$ErrorActionPreference = "Stop"

Write-Host "=== Offline Environment Setup ==="
Set-Location $RepoRoot

if (-not (Test-Path ".\requirements.txt")) {
    throw "requirements.txt not found in $((Get-Location).Path)"
}

if (-not (Test-Path $WheelDir)) {
    throw "Wheel directory not found: $WheelDir"
}

Write-Host "Python version:"
python --version

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment: $VenvDir"
    python -m venv $VenvDir
}

$Py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Python executable not found in venv: $Py"
}

Write-Host "Upgrading pip in venv..."
& $Py -m pip install --upgrade pip

Write-Host "Installing offline build tools..."
& $Py -m pip install --no-index --find-links ".\$WheelDir" setuptools wheel build

Write-Host "Installing requirements from offline wheels..."
& $Py -m pip install --no-index --find-links ".\$WheelDir" --no-build-isolation -r .\requirements.txt

Write-Host "Ensuring SALib + tqdm..."
& $Py -m pip install --no-index --find-links ".\$WheelDir" SALib tqdm

Write-Host "Verifying imports..."
& $Py -c "import SALib, tqdm, mesa, numpy, pandas; print('Dependency check OK')"

Write-Host "Compiling Sobol scripts..."
& $Py -m py_compile scripts/sobol/run_sobol_joint.py scripts/sobol/merge_outputs_and_analyze.py

if ($RunSmoke) {
    Write-Host "Running smoke test..."
    & $Py scripts/sobol/run_sobol_joint.py --outdir $SmokeOutDir --n_base $SmokeNBase --seeds $SmokeSeed --jobs $SmokeJobs --smoke
}

Write-Host "Offline setup complete."

