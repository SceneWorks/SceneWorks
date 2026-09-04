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
    $startInfo.Arguments = '-c "import json,platform,struct,sys;print(json.dumps({''executable'':sys.executable,''base_executable'':getattr(sys,''_base_executable'',None),''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],''implementation'':platform.python_implementation(),''architecture'':platform.machine(),''pointer_bits'':struct.calcsize(''P'')*8}))"'
  } else {
    $startInfo.Arguments = '-c "import json,platform,struct,sys;print(json.dumps({''executable'':sys.executable,''version'':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],''implementation'':platform.python_implementation(),''architecture'':platform.machine(),''pointer_bits'':struct.calcsize(''P'')*8}))"'
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
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$CandidatePaths
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
    if ($identity.version.Count -eq 3 -and
        [int]$identity.version[0] -eq 3 -and
        [int]$identity.version[1] -eq 12 -and
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
