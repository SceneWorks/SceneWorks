$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'provision-starvector-windows-python.ps1')

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Assert-True {
  param([bool]$Value, [string]$Message)
  if (-not $Value) { throw $Message }
}

function Assert-Equal {
  param([object]$Expected, [object]$Actual, [string]$Message)
  if ($Expected -ne $Actual) { throw "$Message (expected '$Expected', got '$Actual')" }
}

function New-CanonicalZipFromDirectory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$DestinationPath
  )

  $canonicalSourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
  $stream = [IO.File]::Open($DestinationPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
  try {
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
      foreach ($file in Get-ChildItem -LiteralPath $canonicalSourceRoot -File -Recurse | Sort-Object FullName) {
        $relativePath = $file.FullName.Substring($canonicalSourceRoot.Length).TrimStart([IO.Path]::DirectorySeparatorChar)
        $entryName = $relativePath.Replace([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
        $entry = $archive.CreateEntry($entryName, [IO.Compression.CompressionLevel]::Optimal)
        $sourceStream = [IO.File]::OpenRead($file.FullName)
        try {
          $entryStream = $entry.Open()
          try {
            $sourceStream.CopyTo($entryStream)
          } finally {
            $entryStream.Dispose()
          }
        } finally {
          $sourceStream.Dispose()
        }
      }
    } finally {
      $archive.Dispose()
    }
  } finally {
    $stream.Dispose()
  }
}

function New-SingleEntryZip {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationPath,
    [Parameter(Mandatory = $true)]
    [string]$EntryName,
    [Parameter(Mandatory = $true)]
    [string]$Content
  )

  $stream = [IO.File]::Open($DestinationPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
  try {
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
      $entry = $archive.CreateEntry($EntryName)
      $writer = [IO.StreamWriter]::new($entry.Open())
      try { $writer.Write($Content) } finally { $writer.Dispose() }
    } finally {
      $archive.Dispose()
    }
  } finally {
    $stream.Dispose()
  }
}

$root = Join-Path $env:RUNNER_TEMP ("starvector portable python {0}" -f [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($root) | Out-Null
try {
  $lockParent = Join-Path $root 'lock-parent'
  [IO.Directory]::CreateDirectory($lockParent) | Out-Null
  $lockPath = Join-Path $lockParent 'portable-python.lock'
  $heldLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
  try {
    $lockTimedOut = $false
    try {
      Invoke-StarVectorWindowsPythonLock -LockPath $lockPath -TimeoutSeconds 1 -ScriptBlock { throw 'contended lock ran its protected body' }
    } catch {
      $lockTimedOut = $_.Exception.Message -like 'timed out waiting for the exclusive portable Python lock:*'
    }
    Assert-True $lockTimedOut 'exclusive lock did not reject a concurrent shared-root provisioner'
  } finally {
    $heldLock.Dispose()
  }
  Assert-Equal 'lock-acquired' (Invoke-StarVectorWindowsPythonLock -LockPath $lockPath -ScriptBlock { 'lock-acquired' }) 'exclusive lock did not release cleanly'

  $source = @'
using System;
using System.Diagnostics;
using System.IO;

public static class FakePortablePython {
    public static int Main() {
        string executable = Process.GetCurrentProcess().MainModule.FileName;
        string escaped = executable.Replace("\\", "\\\\").Replace("\"", "\\\"");
        Console.Out.WriteLine("{\"executable\":\"" + escaped + "\",\"version\":[3,12,10],\"implementation\":\"CPython\",\"architecture\":\"AMD64\",\"pointer_bits\":64}");
        return 0;
    }
}
'@
  $fixtureRoot = Join-Path $root 'fixture'
  $fixtureTools = Join-Path $fixtureRoot 'tools'
  [IO.Directory]::CreateDirectory((Join-Path $fixtureTools 'Lib\venv')) | Out-Null
  [IO.Directory]::CreateDirectory((Join-Path $fixtureTools 'Lib\ensurepip')) | Out-Null
  Add-Type -TypeDefinition $source -Language CSharp -OutputAssembly (Join-Path $fixtureTools 'python.exe') -OutputType ConsoleApplication
  foreach ($relative in @('.signature.p7s', 'tools\python312.dll', 'tools\Lib\venv\__init__.py', 'tools\Lib\ensurepip\__init__.py')) {
    [IO.File]::WriteAllText((Join-Path $fixtureRoot $relative), $relative)
  }
  [IO.File]::WriteAllText((Join-Path $fixtureRoot 'python.nuspec'), @'
<?xml version="1.0" encoding="utf-8"?>
<package><metadata><id>python</id><version>3.12.10</version><authors>Python Software Foundation</authors><repository type="git" url="https://github.com/Python/CPython.git" commit="0cc8128" /></metadata></package>
'@)
  $fixtureExeSha256 = Get-StarVectorSha256 (Join-Path $fixtureTools 'python.exe')
  $fixtureDllSha256 = Get-StarVectorSha256 (Join-Path $fixtureTools 'python312.dll')
  $fixtureArchive = Join-Path $root 'python.3.12.10.fixture.nupkg'
  New-CanonicalZipFromDirectory -SourceRoot $fixtureRoot -DestinationPath $fixtureArchive
  $fixtureSha256 = Get-StarVectorSha256 $fixtureArchive
  $fixtureSha512 = Get-StarVectorSha512 $fixtureArchive
  $fixturePackage = @{
    ArchivePath = $fixtureArchive
    ExpectedSha256 = $fixtureSha256
    ExpectedSha512 = $fixtureSha512
    ExpectedPackageBytes = (Get-Item -LiteralPath $fixtureArchive).Length
    ExpectedPythonSha256 = $fixtureExeSha256
    ExpectedDllSha256 = $fixtureDllSha256
  }

  $partialRoot = Join-Path $root 'durable-python'
  [IO.Directory]::CreateDirectory($partialRoot) | Out-Null
  [IO.File]::WriteAllText((Join-Path $partialRoot 'python-3.12.10-amd64.exe'), 'partial setup-python installer root')
  $oldToolCache = $env:RUNNER_TOOL_CACHE
  $oldAgentTools = $env:AGENT_TOOLSDIRECTORY
  $env:RUNNER_TOOL_CACHE = 'D:\actions-runner\_work\_tool'
  $env:AGENT_TOOLSDIRECTORY = 'E:\different-runner\_work\_tool'
  try {
    $installed = Install-StarVectorWindowsPythonArchive @fixturePackage -DestinationRoot $partialRoot -AllowedRoot $partialRoot
  } finally {
    $env:RUNNER_TOOL_CACHE = $oldToolCache
    $env:AGENT_TOOLSDIRECTORY = $oldAgentTools
  }
  Assert-Equal ([IO.Path]::GetFullPath((Join-Path $partialRoot 'tools\python.exe'))) $installed 'partial root was not replaced by the exact portable interpreter'
  Assert-True (-not (Test-Path -LiteralPath (Join-Path $partialRoot 'python-3.12.10-amd64.exe'))) 'partial setup-python installer survived repair'
  Assert-True ($installed -notlike 'D:\actions-runner\*' -and $installed -notlike 'E:\different-runner\*') 'portable interpreter depended on a runner-specific toolcache'
  Assert-Equal $installed (Install-StarVectorWindowsPythonArchive @fixturePackage -DestinationRoot $partialRoot -AllowedRoot $partialRoot) 'valid portable root was not reused idempotently'

  [IO.File]::WriteAllText((Join-Path $partialRoot 'tools\python312.dll'), 'corrupt extracted runtime')
  $repaired = Install-StarVectorWindowsPythonArchive @fixturePackage -DestinationRoot $partialRoot -AllowedRoot $partialRoot
  Assert-Equal $installed $repaired 'corrupt portable root was not deterministically rebuilt'
  Assert-Equal $fixtureDllSha256 (Get-StarVectorSha256 (Join-Path $partialRoot 'tools\python312.dll')) 'corrupt extracted runtime survived the rebuild'

  $preservedRoot = Join-Path $root 'preserved-partial-root'
  [IO.Directory]::CreateDirectory($preservedRoot) | Out-Null
  [IO.File]::WriteAllText((Join-Path $preservedRoot 'sentinel.txt'), 'preserve before checksum validation')
  $badHashRejected = $false
  try {
    Install-StarVectorWindowsPythonArchive -ArchivePath $fixtureArchive -ExpectedSha256 ('0' * 64) -ExpectedSha512 $fixtureSha512 -DestinationRoot $preservedRoot -AllowedRoot $preservedRoot | Out-Null
  } catch {
    $badHashRejected = $_.Exception.Message -eq 'portable Python package checksums do not match the pinned PSF NuGet identity'
  }
  Assert-True $badHashRejected 'checksum mismatch was not rejected before partial-root cleanup'
  Assert-True (Test-Path -LiteralPath (Join-Path $preservedRoot 'sentinel.txt') -PathType Leaf) 'checksum mismatch deleted the existing partial root'

  $wrongRootRejected = $false
  try {
    Install-StarVectorWindowsPythonArchive -ArchivePath $fixtureArchive -ExpectedSha256 $fixtureSha256 -ExpectedSha512 $fixtureSha512 -DestinationRoot $preservedRoot -AllowedRoot (Join-Path $root 'wrong-root') | Out-Null
  } catch {
    $wrongRootRejected = $_.Exception.Message -eq 'portable Python destination must equal its exact allowed root'
  }
  Assert-True $wrongRootRejected 'partial-root replacement did not enforce the exact allowed root'
  Assert-True (Test-Path -LiteralPath (Join-Path $preservedRoot 'sentinel.txt') -PathType Leaf) 'wrong-root refusal deleted the existing partial root'

  $externalRoot = Join-Path $root 'external-target'
  [IO.Directory]::CreateDirectory($externalRoot) | Out-Null
  [IO.File]::WriteAllText((Join-Path $externalRoot 'sentinel.txt'), 'must survive junction rejection')
  foreach ($junctionKind in @('destination', 'parent')) {
    $junction = Join-Path $root "$junctionKind-junction"
    cmd /c mklink /J "`"$junction`"" "`"$externalRoot`"" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "failed to create the $junctionKind junction fixture" }
    try {
      $junctionDestination = if ($junctionKind -ceq 'destination') { $junction } else { Join-Path $junction 'nested-python' }
      $junctionRejected = $false
      try {
        Install-StarVectorWindowsPythonArchive -ArchivePath $fixtureArchive -ExpectedSha256 $fixtureSha256 -ExpectedSha512 $fixtureSha512 -DestinationRoot $junctionDestination -AllowedRoot $junctionDestination | Out-Null
      } catch {
        $junctionRejected = $_.Exception.Message -like 'path must not traverse a reparse point:*'
      }
      Assert-True $junctionRejected "portable Python did not reject a $junctionKind-root junction"
      Assert-True (Test-Path -LiteralPath (Join-Path $externalRoot 'sentinel.txt') -PathType Leaf) "$junctionKind-root rejection changed its external target"
    } finally {
      [IO.Directory]::Delete($junction, $false)
    }
  }

  $traversalArchive = Join-Path $root 'traversal.nupkg'
  New-SingleEntryZip -DestinationPath $traversalArchive -EntryName '../escape.txt' -Content 'escape'
  $traversalRoot = Join-Path $root 'traversal-root'
  $traversalRejected = $false
  try {
    Expand-StarVectorWindowsPythonPackage -ArchivePath $traversalArchive -DestinationRoot $traversalRoot
  } catch {
    $traversalRejected = $_.Exception.Message -like 'portable Python package path escapes its staging root:*'
  }
  Assert-True $traversalRejected 'archive path traversal was not rejected'
  Assert-True (-not (Test-Path -LiteralPath (Join-Path $root 'escape.txt'))) 'archive path traversal wrote outside the staging root'

  $backslashArchive = Join-Path $root 'backslash-name.nupkg'
  New-SingleEntryZip -DestinationPath $backslashArchive -EntryName 'tools\python.exe' -Content 'unsafe noncanonical entry'
  $backslashRoot = Join-Path $root 'backslash-root'
  $backslashRejected = $false
  try {
    Expand-StarVectorWindowsPythonPackage -ArchivePath $backslashArchive -DestinationRoot $backslashRoot
  } catch {
    $backslashRejected = $_.Exception.Message -like 'portable Python package contains an unsafe archive path:*'
  }
  Assert-True $backslashRejected 'archive entry with a backslash name was not rejected'

  $entryLimitRoot = Join-Path $root 'entry-limit-root'
  $entryLimitRejected = $false
  try {
    Expand-StarVectorWindowsPythonPackage -ArchivePath $fixtureArchive -DestinationRoot $entryLimitRoot -MaximumEntries 1
  } catch {
    $entryLimitRejected = $_.Exception.Message -eq 'portable Python package exceeds the 1-entry archive limit'
  }
  Assert-True $entryLimitRejected 'archive entry-count limit was not enforced'

  $expansionLimitRoot = Join-Path $root 'expansion-limit-root'
  $expansionLimitRejected = $false
  try {
    Expand-StarVectorWindowsPythonPackage -ArchivePath $fixtureArchive -DestinationRoot $expansionLimitRoot -MaximumExpandedBytes 1
  } catch {
    $expansionLimitRejected = $_.Exception.Message -eq 'portable Python package exceeds the 1-byte expansion limit'
  }
  Assert-True $expansionLimitRejected 'archive expansion-size limit was not enforced'

  $officialRoot = Join-Path $root 'official-python'
  $official = Install-StarVectorWindowsPythonPackage -DestinationRoot $officialRoot -AllowedRoot $officialRoot -RunnerTemp $root
  $officialIdentity = Select-StarVectorBootstrapPython -CandidatePaths @($official) -ExpectedVersion @(3, 12, 10)
  Assert-Equal ([IO.Path]::GetFullPath((Join-Path $officialRoot 'tools\python.exe'))) $officialIdentity.Executable 'official portable interpreter path changed'
  Assert-Equal 10 $officialIdentity.Identity.version[2] 'official portable interpreter micro version changed'
  Assert-Equal $script:StarVectorWindowsPythonExeSha256 (Get-StarVectorSha256 (Join-Path $officialRoot 'tools\python.exe')) 'official portable python.exe hash changed'
  Assert-Equal $script:StarVectorWindowsPythonDllSha256 (Get-StarVectorSha256 (Join-Path $officialRoot 'tools\python312.dll')) 'official portable python312.dll hash changed'

  $venvRoot = Join-Path $root 'official-venv'
  & $official -m venv --copies $venvRoot
  if ($LASTEXITCODE -ne 0) { throw 'official portable CPython failed to create a venv' }
  $venvPython = Join-Path $venvRoot 'Scripts\python.exe'
  $venvProbe = Invoke-StarVectorPythonIdentityProbe -Executable $venvPython -IncludeBaseExecutable
  Assert-Equal 0 $venvProbe.ExitCode 'portable CPython venv identity probe failed'
  $venvIdentity = $venvProbe.StdOut | ConvertFrom-Json -ErrorAction Stop
  Assert-Equal 10 $venvIdentity.version[2] 'portable CPython venv micro version changed'
  Assert-Equal ([IO.Path]::GetFullPath($official)) (Resolve-StarVectorWindowsExecutable $venvIdentity.base_executable) 'portable CPython venv base identity changed'
} finally {
  if (Test-Path -LiteralPath $root) {
    Remove-StarVectorWindowsDirectoryTree -TargetRoot $root -AllowedRoot $root
  }
}
