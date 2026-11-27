param(
    [ValidateSet("auto", "vulkan", "cuda")]
    [string]$Backend = "auto",
    [string]$Profile = "",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PicoCode v0.7.0 — One-Click Local Coding AI Setup" -ForegroundColor Cyan
Write-Host "  (4-6 GB VRAM + 32 GB RAM | NO Visual Studio / C++ needed!)" -ForegroundColor DarkCyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

function Find-Python {
    foreach ($cmd in @("py", "python", "python3")) {
        $p = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($p) { return $cmd }
    }
    return $null
}

# 1. Python check / auto-install via winget
$PyCmd = Find-Python
if (-not $PyCmd) {
    Write-Host "[1/3] Python not found — installing Python 3.12 via winget..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
        Refresh-Path
        $PyCmd = Find-Python
    }
    if (-not $PyCmd) {
        Write-Host "ERROR: Python not found. Install Python 3.10+ from https://www.python.org/downloads/." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[1/3] Python ready ($PyCmd) — installing Terminal TUI packages..." -ForegroundColor Green
& $PyCmd -m pip install -q -r "$RepoRoot\requirements.txt" 2>$null

# 2. Download official prebuilt llama.cpp (ZERO compile, ZERO Visual Studio!)
Write-Host ""
Write-Host "[2/3] Downloading official prebuilt llama.cpp engine (zero compile)..." -ForegroundColor Yellow
& $PyCmd "$RepoRoot\scripts\download_models.py" --install-engine $Backend
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to download llama.cpp engine!" -ForegroundColor Red
    exit 1
}

# 3. Download coding models
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "[3/3] Downloading coding models..." -ForegroundColor Yellow
    if ($Profile -ne "") {
        & $PyCmd "$RepoRoot\scripts\download_models.py" --profile $Profile
    } else {
        & $PyCmd "$RepoRoot\scripts\download_models.py"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Model download interrupted — re-run .\setup.cmd to resume from where it stopped." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[3/3] Skipped model download (-SkipModels)." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "  Terminal Coding Agent :  .\agent.cmd" -ForegroundColor White
Write-Host "  Web Studio (Browser)  :  .\web-studio.cmd" -ForegroundColor White
Write-Host "  VS Code OpenAI API    :  .\picocode.cmd" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
