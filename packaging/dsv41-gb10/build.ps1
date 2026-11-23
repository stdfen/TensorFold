param(
    [string]$Image = 'tensorfold-dsv41:v1.2.1-sm121',
    [ValidateSet('toolchain', 'runtime')]
    [string]$Target = 'runtime',
    [int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$commit = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the TensorFold commit.' }

$arguments = @(
    'buildx', 'build',
    '--platform', 'linux/arm64',
    '--file', (Join-Path $PSScriptRoot 'Dockerfile'),
    '--target', $Target,
    '--build-arg', "BUILD_JOBS=$Jobs",
    '--build-arg', "TENSORFOLD_COMMIT=$commit",
    '--tag', $Image,
    '--load',
    $repo
)

& docker @arguments
if ($LASTEXITCODE -ne 0) { throw "Docker build failed with exit code $LASTEXITCODE." }

