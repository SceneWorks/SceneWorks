$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'select-starvector-windows-python.ps1')

function Assert-True {
  param([bool]$Value, [string]$Message)
  if (-not $Value) { throw $Message }
}

function Assert-Equal {
  param([object]$Expected, [object]$Actual, [string]$Message)
  if ($Expected -ne $Actual) { throw "$Message (expected '$Expected', got '$Actual')" }
}

$root = Join-Path $env:RUNNER_TEMP ("starvector python probe {0}" -f [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
try {
  $source = @'
using System;
using System.Diagnostics;
using System.IO;

public static class FakePython {
    public static int Main() {
        string executable = Process.GetCurrentProcess().MainModule.FileName;
        string name = Path.GetFileName(executable);
        if (name.StartsWith("01-")) {
            Console.Error.WriteLine("Traceback from the first fake Python candidate");
            return 7;
        }
        if (name.StartsWith("02-")) {
            Console.Out.WriteLine("{not-json");
            return 0;
        }
        string escaped = executable.Replace("\\", "\\\\").Replace("\"", "\\\"");
        Console.Out.WriteLine("{\"executable\":\"" + escaped + "\",\"version\":[3,12,7]}");
        return 0;
    }
}
'@
  $template = Join-Path $root 'fake-python-template.exe'
  Add-Type -TypeDefinition $source -Language CSharp -OutputAssembly $template -OutputType ConsoleApplication
  $bad = Join-Path $root '01-bad python.exe'
  $malformed = Join-Path $root '02-malformed python.exe'
  $valid = Join-Path $root '03-valid python.exe'
  Copy-Item -LiteralPath $template -Destination $bad
  Copy-Item -LiteralPath $template -Destination $malformed
  Copy-Item -LiteralPath $template -Destination $valid

  $badProbe = Invoke-StarVectorPythonIdentityProbe -Executable $bad
  Assert-Equal 7 $badProbe.ExitCode 'stderr-writing candidate exit code was not captured'
  Assert-True ($badProbe.StdErr -match 'Traceback') 'stderr-writing candidate stderr was not captured'

  $selection = Select-StarVectorBootstrapPython -CandidatePaths @($bad, $malformed, "$valid`"", $valid)
  Assert-Equal ([IO.Path]::GetFullPath($valid)) $selection.Executable 'selection did not continue to the valid Python candidate'
  Assert-Equal 3 $selection.Identity.version[0] 'selected Python major version changed'
  Assert-Equal 12 $selection.Identity.version[1] 'selected Python minor version changed'
  Assert-Equal 7 $selection.Identity.version[2] 'selected Python micro version changed'
  Assert-True ($null -eq (Resolve-StarVectorWindowsExecutable "$valid`"")) 'quoted executable path must fail closed'

  $allFailed = $false
  try {
    Select-StarVectorBootstrapPython -CandidatePaths @($bad, $malformed, "$valid`"") | Out-Null
  } catch {
    $allFailed = $_.Exception.Message -eq 'the self-hosted CUDA runner requires a directly executable Python 3.12 or newer on PATH'
  }
  Assert-True $allFailed 'all invalid candidates must fail closed with the expected error'
} finally {
  Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
