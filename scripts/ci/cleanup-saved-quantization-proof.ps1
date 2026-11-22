param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$resolved=[IO.Path]::GetFullPath($Root).TrimEnd('\')
$expected="D:\Codex\work\saved-quantization-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
if ($resolved -cne $expected) { throw "Unexpected proof root: $resolved" }
$owner=(Get-Content -LiteralPath "$resolved\.saved-quantization-proof" -Raw).Trim()
if ($owner -ne "$($env:GITHUB_RUN_ID):$env:GITHUB_RUN_ATTEMPT") { throw 'Proof ownership marker mismatch' }
$active=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine.IndexOf($resolved,[StringComparison]::OrdinalIgnoreCase) -ge 0 })
if ($active.Count) { throw "Proof outputs retained: active process IDs $($active.ProcessId -join ',')" }
$removed=@()
foreach ($name in @('harness','data','source','target')) {
    $path=Join-Path $resolved $name
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $ancestor=Get-Item -LiteralPath $path -Force
    while ($null -ne $ancestor) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point retained: $($ancestor.FullName)" }
        $ancestor=$ancestor.Parent
    }
    $stack=[Collections.Generic.Stack[IO.DirectoryInfo]]::new();$stack.Push([IO.DirectoryInfo]::new($path));$bytes=[long]0
    while ($stack.Count) {
        foreach ($entry in $stack.Pop().EnumerateFileSystemInfos()) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point retained: $($entry.FullName)" }
            if ($entry -is [IO.DirectoryInfo]) { $stack.Push($entry) } else {
                $bytes += $entry.Length
                if ($entry.Name -eq '.cargo-lock' -or $entry.Name -eq '.lock') {
                    $lock=[IO.File]::Open($entry.FullName,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
                    $lock.Dispose()
                }
            }
        }
    }
    Remove-Item -LiteralPath $path -Recurse -Force
    if (Test-Path -LiteralPath $path) { throw "Cleanup incomplete: $path" }
    $removed += [pscustomobject]@{path=$path;bytes=$bytes;verifiedAbsent=$true}
}
$removed | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$resolved\cleanup.json"
Write-Output "Removed $(($removed | Measure-Object -Property bytes -Sum).Sum) bytes; JSON acceptance receipts and shared Cargo caches retained."
