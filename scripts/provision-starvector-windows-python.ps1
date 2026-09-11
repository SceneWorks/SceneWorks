$script:StarVectorWindowsPythonPackageUri = 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg'
$script:StarVectorWindowsPythonPackageSha256 = '0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8'
$script:StarVectorWindowsPythonPackageSha512 = 'bbda4dcf688a94211b62d50968a91b38f305d0b8d1ecd90269f74a86f8a0a4fcebb7ca162a0753a47691eb3df0c964009bd3d8194c6fd19afae8d5fd01e1cc0f'
$script:StarVectorWindowsPythonPackageBytes = 14515433
$script:StarVectorWindowsPythonExeSha256 = '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'
$script:StarVectorWindowsPythonDllSha256 = '9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120'
$script:StarVectorWindowsPythonVersion = @(3, 12, 10)
$script:StarVectorWindowsPythonMaximumPackageBytes = 20MB
$script:StarVectorWindowsPythonMaximumExpandedBytes = 80MB
$script:StarVectorWindowsPythonMaximumArchiveEntries = 1400

. (Join-Path $PSScriptRoot 'select-starvector-windows-python.ps1')

function Invoke-StarVectorWindowsPythonLock {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LockPath,
    [Parameter(Mandatory = $true)]
    [scriptblock]$ScriptBlock,
    [ValidateRange(1, 1800)]
    [int]$TimeoutSeconds = 1800
  )

  $canonicalLock = Assert-StarVectorWindowsPathComponents -Path $LockPath -LeafType File -AllowMissingLeaf
  $lockParent = Split-Path -Parent $canonicalLock
  Assert-StarVectorWindowsPathComponents -Path $lockParent -LeafType Directory -AllowMissingLeaf | Out-Null
  [IO.Directory]::CreateDirectory($lockParent) | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $lockParent -LeafType Directory | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $canonicalLock -LeafType File -AllowMissingLeaf | Out-Null

  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  $stream = $null
  while (-not $stream) {
    try {
      $stream = [IO.File]::Open($canonicalLock, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch [IO.IOException] {
      if ([DateTime]::UtcNow -ge $deadline) {
        throw "timed out waiting for the exclusive portable Python lock: $canonicalLock"
      }
      Start-Sleep -Milliseconds 250
    }
  }
  try {
    return & $ScriptBlock
  } finally {
    $stream.Dispose()
  }
}

function Get-StarVectorSha512 {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  Assert-StarVectorWindowsPathComponents -Path $Path -LeafType File | Out-Null
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
  if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "refusing to hash a non-regular package file: $Path"
  }
  return (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA512 -ErrorAction Stop).Hash.ToLowerInvariant()
}

function Get-StarVectorSha256 {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  Assert-StarVectorWindowsPathComponents -Path $Path -LeafType File | Out-Null
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
  if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "refusing to hash a non-regular package file: $Path"
  }
  return (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
}

function Save-StarVectorWindowsPythonPackage {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Uri,
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha256,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha512,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 104857600)]
    [long]$ExpectedBytes,
    [ValidateRange(1, 300)]
    [int]$TimeoutSeconds = 120,
    [ValidateRange(1, 104857600)]
    [long]$MaximumBytes = $script:StarVectorWindowsPythonMaximumPackageBytes
  )

  $parsedUri = $null
  if (-not [Uri]::TryCreate($Uri, [UriKind]::Absolute, [ref]$parsedUri) -or $parsedUri.Scheme -cne 'https') {
    throw 'portable Python package download requires an absolute HTTPS URL'
  }
  if ($ExpectedSha256 -notmatch '^[a-f0-9]{64}$' -or $ExpectedSha512 -notmatch '^[a-f0-9]{128}$') {
    throw 'portable Python package download requires exact lowercase SHA-256 and SHA-512 identities'
  }
  if ($ExpectedBytes -gt $MaximumBytes) {
    throw 'portable Python package identity exceeds the configured download limit'
  }
  $destinationPath = [IO.Path]::GetFullPath($Destination)
  if ($destinationPath -notmatch '^[A-Za-z]:\\' -or (Test-Path -LiteralPath $destinationPath)) {
    throw 'portable Python package destination must be a new absolute Windows file path'
  }
  $destinationParent = Split-Path -Parent $destinationPath
  Assert-StarVectorWindowsPathComponents -Path $destinationParent -LeafType Directory | Out-Null

  Add-Type -AssemblyName System.Net.Http
  $handler = [Net.Http.HttpClientHandler]::new()
  $handler.AllowAutoRedirect = $false
  $client = [Net.Http.HttpClient]::new($handler)
  $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
  $client.MaxResponseContentBufferSize = $MaximumBytes
  try {
    $response = $client.GetAsync($parsedUri).GetAwaiter().GetResult()
    try {
      if (-not $response.IsSuccessStatusCode) {
        throw "portable Python package download returned HTTP $([int]$response.StatusCode)"
      }
      if ($response.Content.Headers.ContentLength -and $response.Content.Headers.ContentLength -ne $ExpectedBytes) {
        throw "portable Python package content length is not the expected $ExpectedBytes bytes"
      }
      $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
      if ($bytes.LongLength -ne $ExpectedBytes) {
        throw "portable Python package is not the expected $ExpectedBytes bytes"
      }
      [IO.File]::WriteAllBytes($destinationPath, $bytes)
    } finally {
      $response.Dispose()
    }
  } finally {
    $client.Dispose()
    $handler.Dispose()
  }

  if ((Get-StarVectorSha256 $destinationPath) -cne $ExpectedSha256 -or (Get-StarVectorSha512 $destinationPath) -cne $ExpectedSha512) {
    Remove-Item -LiteralPath $destinationPath -Force -ErrorAction SilentlyContinue
    throw 'portable Python package checksums do not match the pinned PSF NuGet identity'
  }
  return $destinationPath
}

function Expand-StarVectorWindowsPythonPackage {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,
    [ValidateRange(1, 268435456)]
    [long]$MaximumExpandedBytes = $script:StarVectorWindowsPythonMaximumExpandedBytes,
    [ValidateRange(1, 10000)]
    [int]$MaximumEntries = $script:StarVectorWindowsPythonMaximumArchiveEntries
  )

  Assert-StarVectorWindowsPathComponents -Path $ArchivePath -LeafType File | Out-Null
  $canonicalRoot = Assert-StarVectorWindowsPathComponents -Path $DestinationRoot -LeafType Directory -AllowMissingLeaf
  if (Test-Path -LiteralPath $canonicalRoot) {
    throw 'portable Python staging root must not already exist'
  }
  $stagingParent = Split-Path -Parent $canonicalRoot
  Assert-StarVectorWindowsPathComponents -Path $stagingParent -LeafType Directory | Out-Null
  [IO.Directory]::CreateDirectory($canonicalRoot) | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $canonicalRoot -LeafType Directory | Out-Null
  $canonicalRoot = $canonicalRoot.TrimEnd('\')
  $rootBoundary = "$canonicalRoot\"
  $expandedBytes = [long]0
  $entryCount = 0

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
  try {
    foreach ($entry in $archive.Entries) {
      $entryCount += 1
      if ($entryCount -gt $MaximumEntries) {
        throw "portable Python package exceeds the $MaximumEntries-entry archive limit"
      }
      if ([string]::IsNullOrWhiteSpace($entry.FullName) -or $entry.FullName -match '[\\:\0]') {
        throw "portable Python package contains an unsafe archive path: $($entry.FullName)"
      }
      $relativePath = $entry.FullName.Replace('/', [IO.Path]::DirectorySeparatorChar)
      $targetPath = [IO.Path]::GetFullPath((Join-Path $canonicalRoot $relativePath))
      if (-not $targetPath.StartsWith($rootBoundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "portable Python package path escapes its staging root: $($entry.FullName)"
      }
      $expandedBytes += [long]$entry.Length
      if ($expandedBytes -gt $MaximumExpandedBytes) {
        throw "portable Python package exceeds the $MaximumExpandedBytes-byte expansion limit"
      }
      if ([string]::IsNullOrEmpty($entry.Name)) {
        [IO.Directory]::CreateDirectory($targetPath) | Out-Null
        continue
      }
      [IO.Directory]::CreateDirectory((Split-Path -Parent $targetPath)) | Out-Null
      $source = $entry.Open()
      $destination = $null
      try {
        $destination = [IO.File]::Open($targetPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $source.CopyTo($destination)
      } finally {
        if ($destination) { $destination.Dispose() }
        $source.Dispose()
      }
    }
  } finally {
    $archive.Dispose()
  }
}

function Assert-StarVectorWindowsPythonPackageMetadata {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot
  )

  try {
    [xml]$nuspec = Get-Content -LiteralPath (Join-Path $PackageRoot 'python.nuspec') -Raw -ErrorAction Stop
    $metadata = $nuspec.SelectSingleNode("/*[local-name()='package']/*[local-name()='metadata']")
    $id = $metadata.SelectSingleNode("*[local-name()='id']").InnerText
    $version = $metadata.SelectSingleNode("*[local-name()='version']").InnerText
    $authors = $metadata.SelectSingleNode("*[local-name()='authors']").InnerText
    $repository = $metadata.SelectSingleNode("*[local-name()='repository']")
  } catch {
    throw 'portable Python package metadata is unreadable'
  }
  if ($id -cne 'python' -or $version -cne '3.12.10' -or $authors -cne 'Python Software Foundation' -or
      $repository.GetAttribute('type') -cne 'git' -or $repository.GetAttribute('url') -cne 'https://github.com/Python/CPython.git' -or
      $repository.GetAttribute('commit') -cne '0cc8128') {
    throw 'portable Python package metadata does not match the pinned PSF CPython 3.12.10 release'
  }
  if ($metadata.SelectNodes("*[local-name()='dependencies']/*").Count -ne 0) {
    throw 'portable Python package unexpectedly declares dependencies'
  }
}

function Get-StarVectorInstalledWindowsPython {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha256,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha512,
    [Parameter(Mandatory = $true)]
    [int[]]$ExpectedVersion,
    [long]$ExpectedPackageBytes = 0,
    [string]$ExpectedPythonSha256 = '',
    [string]$ExpectedDllSha256 = ''
  )

  $package = Join-Path $DestinationRoot '_provenance\python.3.12.10.nupkg'
  $python = Join-Path $DestinationRoot 'tools\python.exe'
  $pythonDll = Join-Path $DestinationRoot 'tools\python312.dll'
  try {
    Assert-StarVectorWindowsPathComponents -Path $DestinationRoot -LeafType Directory | Out-Null
    if ((Get-StarVectorSha256 $package) -cne $ExpectedSha256 -or (Get-StarVectorSha512 $package) -cne $ExpectedSha512) { return $null }
    if ($ExpectedPackageBytes -gt 0 -and (Get-Item -LiteralPath $package -Force -ErrorAction Stop).Length -ne $ExpectedPackageBytes) { return $null }
    if (-not [string]::IsNullOrEmpty($ExpectedPythonSha256) -and (Get-StarVectorSha256 $python) -cne $ExpectedPythonSha256) { return $null }
    if (-not [string]::IsNullOrEmpty($ExpectedDllSha256) -and (Get-StarVectorSha256 $pythonDll) -cne $ExpectedDllSha256) { return $null }
    return Select-StarVectorBootstrapPython -CandidatePaths @($python) -ExpectedVersion $ExpectedVersion
  } catch {
    return $null
  }
}

function Install-StarVectorWindowsPythonArchive {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha256,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedSha512,
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,
    [Parameter(Mandatory = $true)]
    [string]$AllowedRoot,
    [int[]]$ExpectedVersion = $script:StarVectorWindowsPythonVersion,
    [long]$ExpectedPackageBytes = 0,
    [string]$ExpectedPythonSha256 = '',
    [string]$ExpectedDllSha256 = ''
  )

  Assert-StarVectorWindowsPathComponents -Path $ArchivePath -LeafType File | Out-Null
  $canonicalDestination = Assert-StarVectorWindowsPathComponents -Path $DestinationRoot -LeafType Directory -AllowMissingLeaf
  $canonicalAllowed = Assert-StarVectorWindowsPathComponents -Path $AllowedRoot -LeafType Directory -AllowMissingLeaf
  if (-not [StringComparer]::OrdinalIgnoreCase.Equals($canonicalDestination, $canonicalAllowed)) {
    throw 'portable Python destination must equal its exact allowed root'
  }
  if ($ExpectedPackageBytes -gt 0 -and (Get-Item -LiteralPath $ArchivePath -Force -ErrorAction Stop).Length -ne $ExpectedPackageBytes) {
    throw 'portable Python package size does not match the pinned PSF NuGet identity'
  }
  if ((Get-StarVectorSha256 $ArchivePath) -cne $ExpectedSha256 -or (Get-StarVectorSha512 $ArchivePath) -cne $ExpectedSha512) {
    throw 'portable Python package checksums do not match the pinned PSF NuGet identity'
  }
  $existing = Get-StarVectorInstalledWindowsPython -DestinationRoot $DestinationRoot -ExpectedSha256 $ExpectedSha256 -ExpectedSha512 $ExpectedSha512 -ExpectedVersion $ExpectedVersion -ExpectedPackageBytes $ExpectedPackageBytes -ExpectedPythonSha256 $ExpectedPythonSha256 -ExpectedDllSha256 $ExpectedDllSha256
  if ($existing) { return $existing.Executable }

  $destinationParent = Split-Path -Parent $canonicalDestination
  Assert-StarVectorWindowsPathComponents -Path $destinationParent -LeafType Directory -AllowMissingLeaf | Out-Null
  [IO.Directory]::CreateDirectory($destinationParent) | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $destinationParent -LeafType Directory | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $canonicalDestination -LeafType Directory -AllowMissingLeaf | Out-Null
  Remove-StarVectorWindowsDirectoryTree -TargetRoot $DestinationRoot -AllowedRoot $AllowedRoot
  $stagingRoot = "$DestinationRoot.staging-$([Guid]::NewGuid().ToString('N'))"
  $published = $false
  try {
    Expand-StarVectorWindowsPythonPackage -ArchivePath $ArchivePath -DestinationRoot $stagingRoot
    foreach ($required in @('python.nuspec', '.signature.p7s', 'tools\python.exe', 'tools\python312.dll', 'tools\Lib\venv\__init__.py', 'tools\Lib\ensurepip\__init__.py')) {
      if (-not (Test-Path -LiteralPath (Join-Path $stagingRoot $required) -PathType Leaf)) {
        throw "portable Python package lacks required file: $required"
      }
    }
    Assert-StarVectorWindowsPythonPackageMetadata -PackageRoot $stagingRoot
    $provenanceRoot = Join-Path $stagingRoot '_provenance'
    [IO.Directory]::CreateDirectory($provenanceRoot) | Out-Null
    [IO.File]::Copy($ArchivePath, (Join-Path $provenanceRoot 'python.3.12.10.nupkg'), $false)
    if (-not (Get-StarVectorInstalledWindowsPython -DestinationRoot $stagingRoot -ExpectedSha256 $ExpectedSha256 -ExpectedSha512 $ExpectedSha512 -ExpectedVersion $ExpectedVersion -ExpectedPackageBytes $ExpectedPackageBytes -ExpectedPythonSha256 $ExpectedPythonSha256 -ExpectedDllSha256 $ExpectedDllSha256)) {
      throw 'portable Python package did not publish exact CPython 3.12.10 x64'
    }
    [IO.Directory]::Move($stagingRoot, $DestinationRoot)
    $published = $true
    $installed = Get-StarVectorInstalledWindowsPython -DestinationRoot $DestinationRoot -ExpectedSha256 $ExpectedSha256 -ExpectedSha512 $ExpectedSha512 -ExpectedVersion $ExpectedVersion -ExpectedPackageBytes $ExpectedPackageBytes -ExpectedPythonSha256 $ExpectedPythonSha256 -ExpectedDllSha256 $ExpectedDllSha256
    if (-not $installed) {
      throw 'published portable Python root failed exact identity validation'
    }
    return $installed.Executable
  } catch {
    if ($published -and (Test-Path -LiteralPath $DestinationRoot)) {
      Remove-StarVectorWindowsDirectoryTree -TargetRoot $DestinationRoot -AllowedRoot $AllowedRoot
    }
    throw
  } finally {
    if (Test-Path -LiteralPath $stagingRoot) {
      Remove-StarVectorWindowsDirectoryTree -TargetRoot $stagingRoot -AllowedRoot $stagingRoot
    }
  }
}

function Install-StarVectorWindowsPythonPackage {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,
    [Parameter(Mandatory = $true)]
    [string]$AllowedRoot,
    [Parameter(Mandatory = $true)]
    [string]$RunnerTemp
  )

  $canonicalDestination = Assert-StarVectorWindowsPathComponents -Path $DestinationRoot -LeafType Directory -AllowMissingLeaf
  $canonicalAllowed = Assert-StarVectorWindowsPathComponents -Path $AllowedRoot -LeafType Directory -AllowMissingLeaf
  if (-not [StringComparer]::OrdinalIgnoreCase.Equals($canonicalDestination, $canonicalAllowed)) {
    throw 'portable Python destination must equal its exact allowed root'
  }
  $canonicalRunnerTemp = Assert-StarVectorWindowsPathComponents -Path $RunnerTemp -LeafType Directory
  $existing = Get-StarVectorInstalledWindowsPython -DestinationRoot $DestinationRoot -ExpectedSha256 $script:StarVectorWindowsPythonPackageSha256 -ExpectedSha512 $script:StarVectorWindowsPythonPackageSha512 -ExpectedVersion $script:StarVectorWindowsPythonVersion -ExpectedPackageBytes $script:StarVectorWindowsPythonPackageBytes -ExpectedPythonSha256 $script:StarVectorWindowsPythonExeSha256 -ExpectedDllSha256 $script:StarVectorWindowsPythonDllSha256
  if ($existing) { return $existing.Executable }

  $downloadRoot = Join-Path $canonicalRunnerTemp ("starvector-python-{0}" -f [Guid]::NewGuid().ToString('N'))
  Assert-StarVectorWindowsPathComponents -Path $downloadRoot -LeafType Directory -AllowMissingLeaf | Out-Null
  [IO.Directory]::CreateDirectory($downloadRoot) | Out-Null
  Assert-StarVectorWindowsPathComponents -Path $downloadRoot -LeafType Directory | Out-Null
  try {
    $archive = Join-Path $downloadRoot 'python.3.12.10.nupkg'
    Save-StarVectorWindowsPythonPackage -Uri $script:StarVectorWindowsPythonPackageUri -Destination $archive -ExpectedSha256 $script:StarVectorWindowsPythonPackageSha256 -ExpectedSha512 $script:StarVectorWindowsPythonPackageSha512 -ExpectedBytes $script:StarVectorWindowsPythonPackageBytes | Out-Null
    return Install-StarVectorWindowsPythonArchive -ArchivePath $archive -ExpectedSha256 $script:StarVectorWindowsPythonPackageSha256 -ExpectedSha512 $script:StarVectorWindowsPythonPackageSha512 -DestinationRoot $DestinationRoot -AllowedRoot $AllowedRoot -ExpectedVersion $script:StarVectorWindowsPythonVersion -ExpectedPackageBytes $script:StarVectorWindowsPythonPackageBytes -ExpectedPythonSha256 $script:StarVectorWindowsPythonExeSha256 -ExpectedDllSha256 $script:StarVectorWindowsPythonDllSha256
  } finally {
    if (Test-Path -LiteralPath $downloadRoot) {
      Remove-StarVectorWindowsDirectoryTree -TargetRoot $downloadRoot -AllowedRoot $downloadRoot
    }
  }
}
