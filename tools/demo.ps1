# Force UTF-8 output
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

param(
  [string]$AudioPath = "";
  [switch]$NoImages;
  [string]$Name = "Claire";
  [string]$Title = "Story";
  [switch]$Print;
  [bool]$Open = $true
)

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

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "Using Python: $py"

$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
  . $venvActivate
} else {
  Write-Warning "Virtual environment not found at .\.venv\Scripts\Activate.ps1. Continuing without activation."
}

if (-not $AudioPath -or $AudioPath.Trim() -eq "") {
  $latestAudio = Get-ChildItem -Path ".\out\audio\*.wav" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $latestAudio) { throw "No .wav files found in .\out\audio" }
  $AudioPath = $latestAudio.FullName
}

$env:STORY_ENHANCE_MODE = "openai"
$env:STORY_IMAGE_MODE = if ($NoImages) { "none" } else { "openai" }
$env:STORY_TARGET_PAGES = "2"
$env:STORY_WORDS_PER_PAGE = "260"
$env:STORY_VOICE_MODE = "kid"
$env:STORY_FIDELITY_MODE = "fun"

$env:STORY_AUDIO_PATH = $AudioPath
$escapedAudioPath = $AudioPath.Replace("'", "''")
$command = "from src.pipeline.orchestrator import run_once_from_audio; print(run_once_from_audio(r'$escapedAudioPath'))"
$out = & $py -c $command 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output $out
    throw "Demo run failed with exit code $LASTEXITCODE."
}

$pdfPath = $null
foreach ($line in ($out -split "`r?`n")) {
    if ($line -match "out[\\/]books[\\/].+\.pdf") {
        $pdfPath = $line
        break
    }
}

if (-not $pdfPath) {
    throw "Could not find generated PDF path in output."
}

$nameSlug  = Slugify $Name 40
$titleSlug = Slugify $Title 40
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$friendly = Join-Path "out\books" ("{0}_{1}_{2}.pdf" -f $nameSlug, $titleSlug, $stamp)
$latest = Join-Path "out\books" "LATEST.pdf"

Copy-Item -Force $pdfPath $friendly
Copy-Item -Force $friendly $latest

Write-Host ("OK: Generated: {0}" -f $pdfPath)
Write-Host ("OK: Friendly:  {0}" -f $friendly)
Write-Host ("OK: Latest:    {0}" -f $latest)

if ($Open) { Start-Process $friendly }
if ($Print) { Start-Process -FilePath $friendly -Verb Print }
