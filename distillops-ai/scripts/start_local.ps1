# Start the local API and the React app. Uses the existing virtual environment
# and the frozen checkpoint. Does not train or score the locked test.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing $Python. Create the project virtual environment before starting."
}

$Tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if ((Test-Path $Tesseract) -and -not $env:TESSERACT_CMD) {
    $env:TESSERACT_CMD = $Tesseract
}

Start-Process -FilePath $Python -ArgumentList @(
    "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $Root

Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev") -WorkingDirectory (Join-Path $Root "frontend")

Write-Output "API:   http://127.0.0.1:8000"
Write-Output "App:   http://127.0.0.1:5173"
Write-Output "The first extraction loads the frozen student checkpoint."
