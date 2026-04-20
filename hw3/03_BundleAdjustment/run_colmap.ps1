param(
    [string]$DatasetPath = "data",
    [string]$ColmapExe = "colmap",
    [switch]$UseCpu,
    [switch]$SkipDense
)

$ErrorActionPreference = "Stop"

$imagePath = Join-Path $DatasetPath "images"
$colmapPath = Join-Path $DatasetPath "colmap"
$sparsePath = Join-Path $colmapPath "sparse"
$densePath = Join-Path $colmapPath "dense"
$databasePath = Join-Path $colmapPath "database.db"

$colmapExeResolved = (Resolve-Path $ColmapExe).Path
$colmapBinDir = Split-Path -Parent $colmapExeResolved
$colmapRootDir = Split-Path -Parent $colmapBinDir
$pluginDir = Join-Path $colmapRootDir "plugins"
$platformPluginDir = Join-Path $pluginDir "platforms"

if (Test-Path $pluginDir) {
    $env:QT_PLUGIN_PATH = $pluginDir
}
if (Test-Path $platformPluginDir) {
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = $platformPluginDir
}
if (Test-Path $colmapBinDir) {
    $pathParts = $env:PATH -split ';' | Where-Object { $_ -and ($_ -notmatch 'PyQt5\\Qt5\\plugins\\platforms') }
    $env:PATH = ($colmapBinDir + ';' + ($pathParts -join ';'))
}

New-Item -ItemType Directory -Force -Path $sparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $densePath | Out-Null

$featureArgs = @(
    "feature_extractor",
    "--database_path", $databasePath,
    "--image_path", $imagePath,
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.single_camera", "1"
)

$matchArgs = @(
    "exhaustive_matcher",
    "--database_path", $databasePath
)

if ($UseCpu) {
    $featureArgs += @("--FeatureExtraction.use_gpu", "0")
    $matchArgs += @("--FeatureMatching.use_gpu", "0", "--SiftMatching.cpu_brute_force_matcher", "1")
}

function Invoke-Colmap {
    param([string[]]$CmdArgs)
    & $colmapExeResolved @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP command failed: $($CmdArgs -join ' ')"
    }
}

Write-Host "=== Step 1: Feature Extraction ==="
Invoke-Colmap -CmdArgs $featureArgs

Write-Host "=== Step 2: Feature Matching ==="
Invoke-Colmap -CmdArgs $matchArgs

Write-Host "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
Invoke-Colmap -CmdArgs @(
    "mapper",
    "--database_path", $databasePath,
    "--image_path", $imagePath,
    "--output_path", $sparsePath
)

if ($SkipDense) {
    Write-Host "=== Dense reconstruction skipped ==="
    Write-Host "Sparse result:" (Join-Path $sparsePath "0")
    exit 0
}

Write-Host "=== Step 4: Image Undistortion ==="
Invoke-Colmap -CmdArgs @(
    "image_undistorter",
    "--image_path", $imagePath,
    "--input_path", (Join-Path $sparsePath "0"),
    "--output_path", $densePath
)

Write-Host "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
Invoke-Colmap -CmdArgs @(
    "patch_match_stereo",
    "--workspace_path", $densePath
)

Write-Host "=== Step 6: Stereo Fusion ==="
Invoke-Colmap -CmdArgs @(
    "stereo_fusion",
    "--workspace_path", $densePath,
    "--output_path", (Join-Path $densePath "fused.ply")
)

Write-Host "=== Done ==="
Write-Host "Sparse result:" (Join-Path $sparsePath "0")
Write-Host "Dense result :" (Join-Path $densePath "fused.ply")
