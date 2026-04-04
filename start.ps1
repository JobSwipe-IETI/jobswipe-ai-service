#!/usr/bin/env pwsh
# Script to start the JobSwipe AI service with environment variables from .env

param(
    [string]$Port = "8000"
)

# Load environment variables from .env file
$EnvFile = Join-Path (Get-Location) ".env"

if (Test-Path $EnvFile) {
    Write-Host "Loading environment variables from .env..."
    $EnvContent = Get-Content $EnvFile
    foreach ($line in $EnvContent) {
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line -split "=", 2
            if ($parts.Length -eq 2) {
                [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
                Write-Host "  Set: $($parts[0].Trim())"
            }
        }
    }
} else {
    Write-Warning ".env file not found. Expecting environment variables to be set externally."
}

$env:PYTHONPATH = (Get-Location).Path

# Find Python executable
$PythonExe = "C:\Users\Esteban Aguilera\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe"
    exit 1
}

Write-Host ""
Write-Host "Starting JobSwipe AI Service on port $Port..."
Write-Host "Using Python: $PythonExe"
Write-Host "PYTHONPATH: $($env:PYTHONPATH)"
Write-Host ""

# Start uvicorn
& $PythonExe -m uvicorn app.main:app --reload --port $Port
