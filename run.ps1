# Sloth Study - start everything with one command on Windows.
# From the sloth-study folder run:
#     powershell -ExecutionPolicy Bypass -File .\run.ps1

Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found. Install it from https://www.python.org/downloads/windows/ (tick 'Add Python to PATH')." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating the virtual environment (.venv)..."
    python -m venv .venv
}

Write-Host "Installing packages..."
& .\.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Make sure OLLAMA_MODEL matches a tag from 'ollama list'." -ForegroundColor Yellow
}

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Installed Ollama models:"
    ollama list
} else {
    Write-Host "Ollama was not found. Install it from https://ollama.com/download and open it, then run this script again." -ForegroundColor Yellow
}

& .\.venv\Scripts\python.exe app.py
