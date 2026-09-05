function Assert-StarVectorWindowsPathComponents {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [ValidateSet('Any', 'File', 'Directory')]
    [string]$LeafType = 'Any',
    [switch]$AllowMissingLeaf
  )

  if ([string]::IsNullOrWhiteSpace($Path) -or $Path -match '[\r\n"]' -or $Path -notmatch '^[A-Za-z]:\\') {
    throw 'expected an absolute Windows path without control characters'
  }
  try {
    $canonical = [IO.Path]::GetFullPath($Path)
    $pathRoot = [IO.Path]::GetPathRoot($canonical)
  } catch {
    throw 'expected a canonical absolute Windows path'
  }
  if ($canonical -notmatch '^[A-Za-z]:\\' -or [string]::IsNullOrWhiteSpace($pathRoot)) {
    throw 'expected a canonical absolute Windows path'
  }

  $rootItem = Get-Item -LiteralPath $pathRoot -Force -ErrorAction Stop
  if (-not $rootItem.PSIsContainer -or ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "path root must be a regular directory: $pathRoot"
  }
  $parts = @($canonical.Substring($pathRoot.Length) -split '\\' | Where-Object { -not [string]::IsNullOrEmpty($_) })
  $current = $pathRoot
  $missing = $false
  for ($index = 0; $index -lt $parts.Count; $index++) {
    $current = Join-Path $current $parts[$index]
    $isLeaf = $index -eq ($parts.Count - 1)
    if (-not (Test-Path -LiteralPath $current)) {
      $missing = $true
      continue
    }
    if ($missing) {
      throw "path changed while its components were validated: $canonical"
    }
    $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw "path must not traverse a reparse point: $($item.FullName)"
    }
    if (-not $isLeaf -and -not $item.PSIsContainer) {
      throw "path parent must be a directory: $($item.FullName)"
    }
    if ($isLeaf -and $LeafType -ceq 'File' -and $item.PSIsContainer) {
      throw "path leaf must be a regular file: $($item.FullName)"
    }
    if ($isLeaf -and $LeafType -ceq 'Directory' -and -not $item.PSIsContainer) {
      throw "path leaf must be a regular directory: $($item.FullName)"
    }
  }
  if ($missing -and -not $AllowMissingLeaf) {
    throw "path does not exist: $canonical"
  }
  return $canonical
}

function Resolve-StarVectorWindowsExecutable {
  param([object]$Value)

  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text) -or $text -match '[\r\n"]' -or $text -notmatch '^[A-Za-z]:\\') {
    return $null
  }
  try {
    $full = [IO.Path]::GetFullPath($text)
  } catch {
    return $null
  }
  if ($full -notmatch '^[A-Za-z]:\\' -or -not (Test-Path -LiteralPath $full -PathType Leaf)) {
    return $null
  }
  try {
    Assert-StarVectorWindowsPathComponents -Path $full -LeafType File | Out-Null
  } catch {
    return $null
  }
  return $full
}

function ConvertTo-StarVectorPythonDistributionName {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  $canonical = [regex]::Replace($Name.Trim().ToLowerInvariant(), '[-_.]+', '-')
  if ($canonical -notmatch '^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$') {
    throw "invalid Python distribution name: $Name"
  }
  return $canonical
}

function Remove-StarVectorWindowsDirectoryTree {
  param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,
    [Parameter(Mandatory = $true)]
    [string]$AllowedRoot
  )

  foreach ($value in @($TargetRoot, $AllowedRoot)) {
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match '[\r\n"]' -or $value -notmatch '^[A-Za-z]:\\') {
      throw 'refusing to remove a metrics venv outside the exact workflow-owned terminal root'
    }
  }
  try {
    $canonicalTarget = [IO.Path]::GetFullPath($TargetRoot)
    $canonicalAllowed = [IO.Path]::GetFullPath($AllowedRoot)
  } catch {
    throw 'refusing to remove a metrics venv outside the exact workflow-owned terminal root'
  }
  if (-not [StringComparer]::OrdinalIgnoreCase.Equals($canonicalTarget, $canonicalAllowed)) {
    throw 'refusing to remove a metrics venv outside the exact workflow-owned terminal root'
  }
  if (-not (Test-Path -LiteralPath $canonicalTarget)) { return }

  $pending = New-Object System.Collections.Stack
  $pending.Push([pscustomobject]@{ Path = $canonicalTarget; Expanded = $false })
  while ($pending.Count -gt 0) {
    $frame = $pending.Pop()
    $item = Get-Item -LiteralPath $frame.Path -Force -ErrorAction Stop
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw "refusing to remove a metrics venv containing a reparse point: $($item.FullName)"
    }
    if (-not $item.PSIsContainer) {
      Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
      continue
    }
    if (-not $frame.Expanded) {
      $children = @(Get-ChildItem -LiteralPath $item.FullName -Force -ErrorAction Stop)
      foreach ($child in $children) {
        if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
          throw "refusing to remove a metrics venv containing a reparse point: $($child.FullName)"
        }
      }
      $pending.Push([pscustomobject]@{ Path = $item.FullName; Expanded = $true })
      for ($index = $children.Count - 1; $index -ge 0; $index--) {
        $pending.Push([pscustomobject]@{ Path = $children[$index].FullName; Expanded = $false })
      }
      continue
    }
    if (@(Get-ChildItem -LiteralPath $item.FullName -Force -ErrorAction Stop).Count -ne 0) {
      throw "refusing to remove a metrics venv that changed during validation: $($item.FullName)"
    }
    Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
  }
}

function Invoke-StarVectorPythonIdentityProbe {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [switch]$IncludeBaseExecutable,
    [ValidateRange(1, 120000)]
    [int]$TimeoutMilliseconds = 15000
  )

  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $Executable
  if ($IncludeBaseExecutable) {
    $startInfo.Arguments = '-c "import ensurepip,json,pip,platform,struct,sys,venv;print(json.dumps({''executable'':sys.executable,''base_executable'':getattr(sys,''_base_executable'',None),''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],''implementation'':platform.python_implementation(),''architecture'':platform.machine(),''pointer_bits'':struct.calcsize(''P'')*8}))"'
  } else {
    $startInfo.Arguments = '-c "import ensurepip,json,pip,platform,struct,sys,venv;print(json.dumps({''executable'':sys.executable,''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],''implementation'':platform.python_implementation(),''architecture'':platform.machine(),''pointer_bits'':struct.calcsize(''P'')*8}))"'
  }
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) {
      return [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = 'process did not start' }
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutMilliseconds)) {
      try { $process.Kill() } catch { }
      try { $process.WaitForExit(5000) | Out-Null } catch { }
      return [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = "Python identity probe exceeded $TimeoutMilliseconds ms" }
    }
    $process.WaitForExit()
    return [pscustomobject]@{
      ExitCode = $process.ExitCode
      StdOut = $stdoutTask.Result
      StdErr = $stderrTask.Result
    }
  } catch {
    return [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = $_.Exception.Message }
  } finally {
    $process.Dispose()
  }
}

function Select-StarVectorBootstrapPython {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$CandidatePaths,
    [ValidateCount(2, 3)]
    [int[]]$ExpectedVersion = @(3, 12)
  )

  foreach ($candidate in $CandidatePaths) {
    $canonicalCandidate = Resolve-StarVectorWindowsExecutable $candidate
    if (-not $canonicalCandidate) { continue }
    $probe = Invoke-StarVectorPythonIdentityProbe -Executable $canonicalCandidate
    if ($probe.ExitCode -ne 0) { continue }
    try {
      $identity = $probe.StdOut | ConvertFrom-Json -ErrorAction Stop
    } catch {
      continue
    }
    $canonicalExecutable = Resolve-StarVectorWindowsExecutable $identity.executable
    $versionMatches = $identity.version.Count -eq 3 -and
      [int]$identity.version[0] -eq $ExpectedVersion[0] -and
      [int]$identity.version[1] -eq $ExpectedVersion[1] -and
      ($ExpectedVersion.Count -eq 2 -or [int]$identity.version[2] -eq $ExpectedVersion[2])
    if ($versionMatches -and
        [string]$identity.implementation -ceq 'CPython' -and
        [string]$identity.architecture -ceq 'AMD64' -and
        [int]$identity.pointer_bits -eq 64 -and
        $canonicalExecutable -and
        [StringComparer]::OrdinalIgnoreCase.Equals($canonicalCandidate, $canonicalExecutable)) {
      return [pscustomobject]@{ Executable = $canonicalExecutable; Identity = $identity }
    }
  }

  throw 'StarVector terminal metrics require the explicitly provisioned CPython 3.12 x64 interpreter'
}
