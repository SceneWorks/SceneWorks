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

  Assert-Equal 'open-clip-torch' (ConvertTo-StarVectorPythonDistributionName 'Open.CLIP_Torch') 'distribution-name normalization must fold case, dots, underscores, and hyphens'
  Assert-Equal 'scikit-image' (ConvertTo-StarVectorPythonDistributionName 'scikit---image') 'distribution-name normalization must collapse repeated separators'

  $outside = Join-Path $root 'outside-sentinel'
  $safeTree = Join-Path $root 'metrics-tree'
  $wrongAllowedRoot = Join-Path $root 'wrong-metrics-tree'
  $junction = Join-Path $safeTree 'escape-junction'
  New-Item -ItemType Directory -Path $outside | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $safeTree 'nested') | Out-Null
  [IO.File]::WriteAllText((Join-Path $outside 'sentinel.txt'), 'outside must survive')
  [IO.File]::WriteAllText((Join-Path $safeTree 'nested\inside.txt'), 'safe tree file')

  $wrongRootBlocked = $false
  try {
    Remove-StarVectorWindowsDirectoryTree -TargetRoot $safeTree -AllowedRoot $wrongAllowedRoot
  } catch {
    $wrongRootBlocked = $_.Exception.Message -eq 'refusing to remove a metrics venv outside the exact workflow-owned terminal root'
  }
  Assert-True $wrongRootBlocked 'directory deletion must require the exact allowed root'
  Assert-True (Test-Path -LiteralPath (Join-Path $safeTree 'nested\inside.txt') -PathType Leaf) 'exact-root refusal must preserve the target tree'

  New-Item -ItemType Junction -Path $junction -Target $outside | Out-Null
  $reparseBlocked = $false
  try {
    Remove-StarVectorWindowsDirectoryTree -TargetRoot $safeTree -AllowedRoot $safeTree
  } catch {
    $reparseBlocked = $_.Exception.Message -like 'refusing to remove a metrics venv containing a reparse point:*'
  }
  Assert-True $reparseBlocked 'directory deletion must refuse a descendant junction'
  Assert-True (Test-Path -LiteralPath (Join-Path $outside 'sentinel.txt') -PathType Leaf) 'junction refusal must preserve the outside sentinel'

  [IO.Directory]::Delete($junction)
  Remove-StarVectorWindowsDirectoryTree -TargetRoot $safeTree -AllowedRoot $safeTree
  Assert-True (-not (Test-Path -LiteralPath $safeTree)) 'a normal metrics tree must be removed after validation'
  Assert-True (Test-Path -LiteralPath (Join-Path $outside 'sentinel.txt') -PathType Leaf) 'normal tree deletion must not affect the outside sentinel'

  $rootJunction = Join-Path $root 'metrics-root-junction'
  New-Item -ItemType Junction -Path $rootJunction -Target $outside | Out-Null
  $rootReparseBlocked = $false
  try {
    Remove-StarVectorWindowsDirectoryTree -TargetRoot $rootJunction -AllowedRoot $rootJunction
  } catch {
    $rootReparseBlocked = $_.Exception.Message -like 'refusing to remove a metrics venv containing a reparse point:*'
  }
  Assert-True $rootReparseBlocked 'directory deletion must refuse a root junction'
  Assert-True (Test-Path -LiteralPath (Join-Path $outside 'sentinel.txt') -PathType Leaf) 'root-junction refusal must preserve the outside sentinel'
  [IO.Directory]::Delete($rootJunction)
} finally {
  Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
