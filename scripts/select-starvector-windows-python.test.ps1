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
        int minor = name.StartsWith("03-") ? 11 : (name.StartsWith("04-") ? 14 : 12);
        string architecture = name.StartsWith("05-") ? "ARM64" : "AMD64";
        string escaped = executable.Replace("\\", "\\\\").Replace("\"", "\\\"");
        Console.Out.WriteLine("{\"executable\":\"" + escaped + "\",\"version\":[3," + minor + ",7],\"implementation\":\"CPython\",\"architecture\":\"" + architecture + "\",\"pointer_bits\":64}");
        return 0;
    }
}
'@
  $template = Join-Path $root 'fake-python-template.exe'
  Add-Type -TypeDefinition $source -Language CSharp -OutputAssembly $template -OutputType ConsoleApplication
  $bad = Join-Path $root '01-bad python.exe'
  $malformed = Join-Path $root '02-malformed python.exe'
  $belowMinimum = Join-Path $root '03-python311.exe'
  $newerUnsupported = Join-Path $root '04-python314.exe'
  $wrongArchitecture = Join-Path $root '05-python312-arm64.exe'
  $valid = Join-Path $root '06-python312-amd64.exe'
  Copy-Item -LiteralPath $template -Destination $bad
  Copy-Item -LiteralPath $template -Destination $malformed
  Copy-Item -LiteralPath $template -Destination $belowMinimum
  Copy-Item -LiteralPath $template -Destination $newerUnsupported
  Copy-Item -LiteralPath $template -Destination $wrongArchitecture
  Copy-Item -LiteralPath $template -Destination $valid

  $badProbe = Invoke-StarVectorPythonIdentityProbe -Executable $bad
  Assert-Equal 7 $badProbe.ExitCode 'stderr-writing candidate exit code was not captured'
  Assert-True ($badProbe.StdErr -match 'Traceback') 'stderr-writing candidate stderr was not captured'

  $selection = Select-StarVectorBootstrapPython -CandidatePaths @($bad, $malformed, $belowMinimum, $newerUnsupported, $wrongArchitecture, "$valid`"", $valid)
  Assert-Equal ([IO.Path]::GetFullPath($valid)) $selection.Executable 'selection did not continue to the valid Python candidate'
  Assert-Equal 3 $selection.Identity.version[0] 'selected Python major version changed'
  Assert-Equal 12 $selection.Identity.version[1] 'selected Python minor version changed'
  Assert-Equal 7 $selection.Identity.version[2] 'selected Python micro version changed'
  Assert-Equal 'CPython' $selection.Identity.implementation 'selected Python implementation changed'
  Assert-Equal 'AMD64' $selection.Identity.architecture 'selected Python architecture changed'
  Assert-Equal 64 $selection.Identity.pointer_bits 'selected Python pointer width changed'
  Assert-True ($null -eq (Resolve-StarVectorWindowsExecutable "$valid`"")) 'quoted executable path must fail closed'

  $allFailed = $false
  try {
    Select-StarVectorBootstrapPython -CandidatePaths @($bad, $malformed, $belowMinimum, $newerUnsupported, $wrongArchitecture, "$valid`"") | Out-Null
  } catch {
    $allFailed = $_.Exception.Message -eq 'StarVector terminal metrics require the explicitly provisioned CPython 3.12 x64 interpreter'
  }
  Assert-True $allFailed 'all invalid candidates must fail closed with the expected error'
} finally {
  Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
