# Force UTF-8 so emojis render correctly
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

param(
    [string]$AudioPath = ""
    [switch]$Open = $true
    [switch]$NoImages
    [string]$Name = "Claire"
    [string]$Title = "Story"
    [switch]$Print
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

function New-SafeSlug {
    param(
        [string]$Text
    )
    $clean = $Text -replace "[^A-Za-z0-9 _-]", ""
    $clean = $clean -replace "\s+", "_"
    $clean = $clean.Trim("_")
    if ($clean.Length -gt 40) {
        $clean = $clean.Substring(0, 40)
    }
    return $clean
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

$nameSlug = New-SafeSlug -Text $Name
$titleSlug = New-SafeSlug -Text $Title
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$friendly = "out\books\${nameSlug}_${titleSlug}_${stamp}.pdf"
$latest = "out\books\LATEST.pdf"

Copy-Item -Path $pdfPath -Destination $friendly -Force
Copy-Item -Path $friendly -Destination $latest -Force

Write-Output "OK: Generated: $pdfPath"
Write-Output "OK: Friendly: $friendly"
Write-Output "OK: Latest: $latest"

if ($Open) {
    Start-Process $friendly
}

if ($Print) {
    Start-Process -FilePath $friendly -Verb Print
}
