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
  return $full
}

function Invoke-StarVectorPythonIdentityProbe {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [switch]$IncludeBaseExecutable
  )

  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $Executable
  if ($IncludeBaseExecutable) {
    $startInfo.Arguments = '-c "import json,sys;print(json.dumps({''executable'':sys.executable,''base_executable'':getattr(sys,''_base_executable'',None),''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro]}))"'
  } else {
    $startInfo.Arguments = '-c "import json,sys;print(json.dumps({''executable'':sys.executable,''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro]}))"'
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
  param([string[]]$CandidatePaths)

  if ($null -eq $CandidatePaths) {
    $CandidatePaths = @(Get-Command python.exe -All -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object Source | Where-Object { $_ } | Select-Object -Unique)
  }

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
    if ($identity.version.Count -eq 3 -and [int]$identity.version[0] -eq 3 -and [int]$identity.version[1] -ge 12 -and $canonicalExecutable) {
      return [pscustomobject]@{ Executable = $canonicalExecutable; Identity = $identity }
    }
  }

  throw 'the self-hosted CUDA runner requires a directly executable Python 3.12 or newer on PATH'
}
