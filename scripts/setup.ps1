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
Write-Host "  PicoCode v0.1.0 — Tek Tikla Yerel Kodlama AI Kurulumu" -ForegroundColor Cyan
Write-Host "  (4-6 GB VRAM + 32 GB RAM | Visual Studio / C++ GEREKMEZ!)" -ForegroundColor DarkCyan
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
    Write-Host "[1/3] Python bulunamadi — winget ile Python 3.12 kuruluyor..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
        Refresh-Path
        $PyCmd = Find-Python
    }
    if (-not $PyCmd) {
        Write-Host "HATA: Python bulunamadi. Lutfen https://www.python.org/downloads/ adresinden Python 3.10+ kurun." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[1/3] Python hazir ($PyCmd)" -ForegroundColor Green

# 2. Download official prebuilt llama.cpp (ZERO compile, ZERO Visual Studio!)
Write-Host ""
Write-Host "[2/3] Resmi hazir llama.cpp motoru indiriliyor (derleme yok!)..." -ForegroundColor Yellow
& $PyCmd "$RepoRoot\scripts\download_models.py" --install-engine $Backend
if ($LASTEXITCODE -ne 0) {
    Write-Host "HATA: llama.cpp motoru indirilemedi!" -ForegroundColor Red
    exit 1
}

# 3. Download coding models
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "[3/3] Kodlama modeli indiriliyor..." -ForegroundColor Yellow
    if ($Profile -ne "") {
        & $PyCmd "$RepoRoot\scripts\download_models.py" --profile $Profile
    } else {
        & $PyCmd "$RepoRoot\scripts\download_models.py"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "UYARI: Model indirme yarim kaldi — .\setup.cmd tekrar calistirildiginda kaldigi yerden devam eder." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[3/3] Model indirme atlandi (-SkipModels)." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  KURULUM TAMAMLANDI!" -ForegroundColor Green
Write-Host "  Web Studio (Tarayici):  .\web-studio.cmd" -ForegroundColor White
Write-Host "  VS Code OpenAI API   :  .\picocode.cmd" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
