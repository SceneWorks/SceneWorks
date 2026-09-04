param(
  [string]$BootstrapPython = $env:STARVECTOR_METRICS_BOOTSTRAP_PYTHON,
  [string]$MetricsLock = (Join-Path $PSScriptRoot '..\release\starvector-terminal-metrics-lock-v1.json')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'select-starvector-windows-python.ps1')

$bootstrap = Select-StarVectorBootstrapPython -CandidatePaths @($BootstrapPython)
$bootstrapPython = $bootstrap.Executable
$packages = @((Get-Content -LiteralPath $MetricsLock -Raw | ConvertFrom-Json -ErrorAction Stop).required_packages)
if ($packages.Count -ne 7) { throw 'the StarVector metric lock must contain exactly seven direct packages' }

$specs = @()
$lockedPackages = @()
$seen = @{}
foreach ($package in $packages) {
  $name = [string]$package.name
  $version = [string]$package.version
  $canonicalName = ConvertTo-StarVectorPythonDistributionName $name
  if ([string]::IsNullOrWhiteSpace($version) -or $seen.ContainsKey($canonicalName)) {
    throw 'the StarVector metric lock contains an invalid or duplicate package identity'
  }
  $seen[$canonicalName] = $true
  $specs += "$name==$version"
  $lockedPackages += [pscustomobject]@{ name = $name; canonical_name = $canonicalName; version = $version }
}

$root = Join-Path $env:RUNNER_TEMP ("starvector-metric-wheel-contract-{0}" -f [Guid]::NewGuid().ToString('N'))
$wheelhouse = Join-Path $root 'wheelhouse'
$venv = Join-Path $root 'venv'
New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null

$downloadArgs = @('-m', 'pip', 'download', '--disable-pip-version-check', '--only-binary=:all:', '--retries', '5', '--timeout', '60', '--dest', $wheelhouse) + $specs
& $bootstrapPython @downloadArgs
if ($LASTEXITCODE -ne 0) { throw 'the pinned StarVector metric closure did not resolve entirely to wheels' }

$downloads = @(Get-ChildItem -LiteralPath $wheelhouse -File)
if ($downloads.Count -eq 0 -or @($downloads | Where-Object { $_.Extension -cne '.whl' }).Count -ne 0) {
  throw 'the pinned StarVector metric closure produced a non-wheel distribution'
}

& $bootstrapPython -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'failed to create the hosted StarVector metric verification venv' }
$venvPython = Join-Path $venv 'Scripts\python.exe'
$installArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', '--find-links', $wheelhouse, '--only-binary=:all:') + $specs
& $venvPython @installArgs
if ($LASTEXITCODE -ne 0) { throw 'the wheel-only StarVector metric closure did not install offline' }

foreach ($package in $lockedPackages) {
  $observedJson = & $venvPython -c 'import importlib.metadata,json,sys;d=importlib.metadata.distribution(sys.argv[1]);print(json.dumps({"name":d.metadata["Name"],"version":d.version}))' ([string]$package.name)
  if ($LASTEXITCODE -ne 0) { throw "the installed StarVector metric package identity was unreadable: $($package.name)" }
  try { $observed = ([string]$observedJson).Trim() | ConvertFrom-Json -ErrorAction Stop } catch { throw "the installed StarVector metric package identity was invalid: $($package.name)" }
  $observedCanonicalName = ConvertTo-StarVectorPythonDistributionName ([string]$observed.name)
  if ($observedCanonicalName -cne [string]$package.canonical_name -or [string]$observed.version -cne [string]$package.version) {
    throw "the installed StarVector metric package identity drifted: $($package.name)"
  }
}

& $venvPython -c 'import PIL,lpips,numpy,open_clip,skimage,torch,torchvision'
if ($LASTEXITCODE -ne 0) { throw 'the compiled StarVector metric packages did not import after wheel-only installation' }

Write-Host ("verified {0} wheel files for CPython {1} {2}" -f $downloads.Count, ($bootstrap.Identity.version -join '.'), $bootstrap.Identity.architecture)
