param(
  [string]$AudioPath = "",
  [switch]$NoImages,
  [string]$Name = "Claire",
  [string]$Title = "Story",
  [switch]$Print,
  [bool]$Open = $true
)

# Force UTF-8 output (must be AFTER param in Windows PowerShell)
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ErrorActionPreference = "Stop"

function Slugify([string]$s, [int]$maxLen = 40) {
  if (-not $s) { return "Story" }
  $t = $s -replace "[^A-Za-z0-9 _-]", ""
  $t = ($t -replace "\s+", "_").Trim("_")
  if ($t.Length -gt $maxLen) { $t = $t.Substring(0, $maxLen).Trim("_") }
  if (-not $t) { $t = "Story" }
  return $t
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
  . $venvActivate
} else {
  Write-Warning "Virtual environment not found at .\.venv\Scripts\Activate.ps1. Continuing without activation."
}

$env:STORY_ENHANCE_MODE="openai"
$env:STORY_TARGET_PAGES="2"
$env:STORY_WORDS_PER_PAGE="260"
$env:STORY_VOICE_MODE="kid"
$env:STORY_FIDELITY_MODE="fun"
if ($NoImages) { $env:STORY_IMAGE_MODE="none" } else { $env:STORY_IMAGE_MODE="openai" }

if (-not $AudioPath -or $AudioPath.Trim() -eq "") {
  $latestAudio = Get-ChildItem -Path ".\out\audio\*.wav" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $latestAudio) { throw "No .wav files found in .\out\audio" }
  $AudioPath = $latestAudio.FullName
}

$out = python -c "from src.pipeline.orchestrator import run_once_from_audio; print(run_once_from_audio(r'$AudioPath'))" 2>&1
$pdf = ($out | Select-String -Pattern "out\\books\\.*\.pdf" | Select-Object -Last 1).ToString()
if (-not $pdf) { Write-Host $out; throw "Could not find generated PDF path in output." }

$nameSlug  = Slugify $Name 40
$titleSlug = Slugify $Title 40
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$friendly = Join-Path "out\books" ("{0}_{1}_{2}.pdf" -f $nameSlug, $titleSlug, $stamp)
$latest = Join-Path "out\books" "LATEST.pdf"

Copy-Item -Force $pdf $friendly
Copy-Item -Force $friendly $latest

Write-Host ("OK: Generated: {0}" -f $pdf)
Write-Host ("OK: Friendly:  {0}" -f $friendly)
Write-Host ("OK: Latest:    {0}" -f $latest)

if ($Open) { Start-Process $friendly }
if ($Print) { Start-Process -FilePath $friendly -Verb Print }
