# Force UTF-8 so emojis render correctly
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

Param(
    [string]$AudioPath,
    [switch]$Open = $true,
    [switch]$NoImages
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Warning "Virtual environment not found at .\\.venv\\Scripts\\Activate.ps1. Continuing without activation."
}

if (-not $AudioPath) {
    $latestAudio = Get-ChildItem -Path ".\out\audio\*.wav" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latestAudio) {
        throw "No .wav files found in .\\out\\audio. Provide -AudioPath."
    }
    $AudioPath = $latestAudio.FullName
}

$env:STORY_ENHANCE_MODE = "openai"
$env:STORY_IMAGE_MODE = if ($NoImages) { "none" } else { "openai" }
$env:STORY_TARGET_PAGES = "2"
$env:STORY_WORDS_PER_PAGE = "260"
$env:STORY_VOICE_MODE = "kid"
$env:STORY_FIDELITY_MODE = "fun"

$escapedAudioPath = $AudioPath.Replace("'", "''")
$command = "from src.pipeline.orchestrator import run_once_from_audio; print(run_once_from_audio(r'$escapedAudioPath'))"

$output = & python -c $command
if ($LASTEXITCODE -ne 0) {
    throw "Demo run failed with exit code $LASTEXITCODE."
}

$pdfPath = $null
foreach ($line in ($output -split "`r?`n")) {
    if ($line -match "out[\\/]books[\\/].+\.pdf") {
        $pdfPath = $line
        break
    }
}

if (-not $pdfPath) {
    throw "Could not find generated PDF path in output."
}

Write-Output "✅ Book generated: $pdfPath"

if ($Open) {
    Start-Process $pdfPath
}
