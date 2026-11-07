param([string]$Payload, [switch]$Library)
$ErrorActionPreference = 'Stop'

function Assert-OrdinaryAncestors([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($item) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($item.FullName)" }
        $item = if ($item -is [IO.FileInfo]) { $item.Directory } else { $item.Parent }
    }
}

function Get-StorageTree([string]$Path) {
    Assert-OrdinaryAncestors $Path
    $stack = New-Object 'System.Collections.Generic.Stack[string]'
    $stack.Push($Path)
    $locks = New-Object 'System.Collections.Generic.List[string]'
    $bytes = 0L
    $latest = [datetime]::MinValue
    while ($stack.Count) {
        $directory = $stack.Pop()
        foreach ($item in (Get-ChildItem -LiteralPath $directory -Force)) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($item.FullName)" }
            if ($item.PSIsContainer) { $stack.Push($item.FullName) }
            else {
                $bytes += $item.Length
                if ($item.LastWriteTimeUtc -gt $latest) { $latest = $item.LastWriteTimeUtc }
                if ($item.CreationTimeUtc -gt $latest) { $latest = $item.CreationTimeUtc }
                if ($item.Name -eq '.cargo-lock') { $locks.Add($item.FullName) }
            }
        }
    }
    return @{ bytes = $bytes; locks = @($locks.ToArray()); latest = $latest }
}

function Get-OwnWorker($Processes) {
    $cursor = $PID
    $visited = @{}
    while ($cursor -and -not $visited.ContainsKey($cursor)) {
        $visited[$cursor] = $true
        $process = $Processes | Where-Object ProcessId -eq $cursor | Select-Object -First 1
        if (-not $process) { return $null }
        if ($process.Name -eq 'Runner.Worker.exe') { return $process.ProcessId }
        $cursor = $process.ParentProcessId
    }
    return $null
}

function Get-MossInputs([string]$RunnerWorkspace) {
    $installation = Split-Path -Parent (Split-Path -Parent $RunnerWorkspace)
    # Discover sibling installations by their runner configuration, including
    # offline listeners. Do not assume four listeners or use runner names as labels.
    $installations = @(Get-ChildItem -LiteralPath (Split-Path -Parent $installation) -Directory -Force |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName '.runner') -PathType Leaf } |
        ForEach-Object { $_.FullName })
    foreach ($process in (Get-CimInstance Win32_Process | Where-Object Name -eq 'Runner.Listener.exe')) {
        if (-not $process.ExecutablePath) { throw 'An installed listener is unreadable' }
        $installations += Split-Path -Parent (Split-Path -Parent $process.ExecutablePath)
    }
    $inputs = @()
    foreach ($root in ($installations | Sort-Object -Unique)) {
        Assert-OrdinaryAncestors $root
        $configPath = Join-Path $root '.runner'
        Assert-OrdinaryAncestors $configPath
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if (-not $config.workFolder -or [IO.Path]::IsPathRooted($config.workFolder)) { throw 'Unrecognized listener work folder' }
        $work = [IO.Path]::GetFullPath((Join-Path $root $config.workFolder))
        if (-not $work.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Listener work folder escaped its installation' }
        $parent = Join-Path $work 'SceneWorks\moss-sc-26752'
        if (-not (Test-Path -LiteralPath $parent)) { continue }
        Assert-OrdinaryAncestors $parent
        foreach ($bundle in (Get-ChildItem -LiteralPath $parent -Directory -Force | Where-Object Name -match '^moss-bundle-[0-9]+-[0-9]+$')) {
            $path = Join-Path $bundle.FullName 'campaign.json'
            Assert-OrdinaryAncestors $path
            if ((Get-Item -LiteralPath $path).Length -gt 16MB) { throw 'Unbounded bundle manifest' }
            $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
            $hf = $manifest.backends.cuda.hfHome
            if (-not ($hf -is [string]) -or -not [IO.Path]::IsPathRooted($hf)) { throw 'Bundle cache input is unresolved' }
            $inputs += [IO.Path]::GetFullPath($hf)
        }
    }
    return @($inputs | Sort-Object -Unique)
}

function Get-StorageBlocker([string]$Path, [string]$Mode, $Processes, $OwnWorker) {
    $normalized = $Path.Replace('/', '\').ToLowerInvariant().TrimEnd('\')
    foreach ($process in $Processes) {
        if ($process.ProcessId -eq $PID) { continue }
        $command = ([string]$process.CommandLine).Replace('/', '\').ToLowerInvariant()
        $exe = ([string]$process.ExecutablePath).Replace('/', '\').ToLowerInvariant()
        if ($exe.StartsWith($normalized + '\') -or $command.Contains($normalized + '\') -or $command.Contains($normalized + '"')) { return "Process $($process.ProcessId) references this directory" }
        if ($Mode -eq 'moss') {
            # Legacy jobs have no leases. Fail closed across the entire host, including
            # other listeners and local model consumers whose HF_HOME is not in CIM.
            if ($process.Name -eq 'Runner.Worker.exe' -and $process.ProcessId -ne $OwnWorker) { return 'Another Actions job is active' }
            if ($process.Name -match '^(cargo|rustc|rust-analyzer|clippy-driver|sceneworks.*|candle.*|memory[-_].*|runtime[-_].*|moss.*|llama.*|vllm.*|ollama.*|python(?:[0-9]+(?:\.[0-9]+)*)?w?)\.exe$') { return 'A build or model process is active' }
        }
    }
    return $null
}

function Remove-StorageLeaf([string]$Path, [string]$Parent, [string]$Mode, [bool]$DryRun, $Processes, $OwnWorker) {
    $pathFull = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    if (-not [IO.Path]::GetDirectoryName($pathFull).Equals($parentFull, [StringComparison]::OrdinalIgnoreCase)) { throw 'Target must be a direct child of the allowed parent' }
    if ([IO.Path]::GetFileName($pathFull) -ne $(if ($Mode -eq 'moss') { 'hf' } else { 'target' })) { throw 'Unexpected storage leaf name' }
    Assert-OrdinaryAncestors $parentFull
    $tree = Get-StorageTree $pathFull
    if ($Mode -eq 'cargo' -and -not ((Test-Path -LiteralPath "$pathFull\.rustc_info.json" -PathType Leaf) -or (Test-Path -LiteralPath "$pathFull\CACHEDIR.TAG" -PathType Leaf) -or $tree.locks.Count)) { throw 'No Cargo target marker' }
    if ($Mode -eq 'moss' -and $tree.latest -gt [datetime]::UtcNow.AddDays(-1)) { throw 'Model cache has recent writes or copies' }
    $blocker = Get-StorageBlocker $pathFull $Mode $Processes $OwnWorker
    if ($blocker) { return @{ status = 'skipped'; path = $pathFull; reason = $blocker } }
    $handles = @()
    try {
        foreach ($lock in $tree.locks) {
            # Deny new read/write opens for the duration, but permit deleting the locked
            # directory. An existing Cargo build makes this open fail without touching it.
            $handles += [IO.File]::Open($lock, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::Delete)
        }
        # Resolve again just before removal, with LiteralPath and no shell handoff.
        $resolved = (Resolve-Path -LiteralPath $pathFull).ProviderPath.TrimEnd('\')
        if (-not $resolved.Equals($pathFull, [StringComparison]::OrdinalIgnoreCase)) { throw 'Resolved target changed' }
        Assert-OrdinaryAncestors $pathFull
        if ($DryRun) { return @{ status = 'would-remove'; path = $pathFull; bytes = $tree.bytes } }
        Remove-Item -LiteralPath $pathFull -Recurse -Force
        return @{ status = 'removed'; path = $pathFull; bytes = $tree.bytes }
    } finally { foreach ($handle in $handles) { $handle.Dispose() } }
}

if ($Library) { return }
$mutex = $null
$acquired = $false
try {
    if ($env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'self-hosted') { throw 'Only self-hosted Actions jobs may invoke automatic cleanup' }
    $request = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Payload)) | ConvertFrom-Json
    $workspace = [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE).TrimEnd('\')
    $runnerWorkspace = [IO.Path]::GetFullPath($env:RUNNER_WORKSPACE).TrimEnd('\')
    if (-not $workspace.StartsWith($runnerWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Checkout is outside this listener workspace' }
    Assert-OrdinaryAncestors $workspace
    $mutex = New-Object Threading.Mutex($false, 'Global\SceneWorksRunnerStorageCleanup')
    try { $acquired = $mutex.WaitOne(30000) } catch [Threading.AbandonedMutexException] { $acquired = $true }
    if (-not $acquired) { throw 'Another storage operation holds the lock' }
    if ($request.mode -eq 'barrier') { @{ status = 'registered' } | ConvertTo-Json -Compress; return }
    if ($request.mode -eq 'moss-inputs') {
        @{ status = 'verified'; inputs = @(Get-MossInputs $runnerWorkspace) } | ConvertTo-Json -Compress -Depth 4
        return
    }
    $mossParent = Join-Path (Split-Path -Parent $workspace) 'moss-sc-26752'
    $pathFull = [IO.Path]::GetFullPath($request.path).TrimEnd('\')
    $parentFull = [IO.Path]::GetFullPath($request.parent).TrimEnd('\')
    $cargoParents = @($workspace, (Join-Path $workspace 'source'), (Join-Path $workspace 'inference'))
    $mossRoot = [IO.Path]::GetDirectoryName($pathFull)
    $mossLeaf = [IO.Path]::GetFileName($mossRoot)
    $ownedMoss = [IO.Path]::GetDirectoryName($mossRoot).Equals($mossParent, [StringComparison]::OrdinalIgnoreCase) -and $mossLeaf -match '^moss-(build|execute)-[0-9]+-[0-9]+-cuda$'
    if ($request.mode -eq 'inspect-parent') {
        if ($pathFull -ne $mossParent -and -not ([IO.Path]::GetDirectoryName($pathFull) -eq $mossParent -and [IO.Path]::GetFileName($pathFull) -match '^moss-(build|execute|bundle)-[0-9]+-[0-9]+(?:-cuda)?$')) { throw 'Unrecognized MOSS parent' }
        Assert-OrdinaryAncestors $pathFull
        @{ status = 'verified'; path = $pathFull } | ConvertTo-Json -Compress
        return
    }
    if ($request.mode -eq 'cargo') {
        if ($parentFull -notin $cargoParents -and -not $ownedMoss) { throw 'Cargo target is outside the job-owned storage namespace' }
        # Ordinary workspaces may never contain tracked files under a target directory.
        if ($parentFull -in $cargoParents) {
            $tracked = & git --no-optional-locks -C $parentFull ls-files -- target 2>&1
            if ($LASTEXITCODE -ne 0 -or $tracked) { throw 'Target has tracked files or repository ownership cannot be verified' }
        }
    } elseif ($request.mode -eq 'moss') {
        if ($env:GITHUB_REPOSITORY -ne 'SceneWorks/SceneWorks' -or -not $ownedMoss) { throw 'Not a private MOSS cache on this listener' }
    } else { throw 'Unrecognized cleanup mode' }
    $processes = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine)
    $ownWorker = Get-OwnWorker $processes
    if (-not $ownWorker) { throw 'Could not identify this Actions worker' }
    if (@($processes | Where-Object { $_.Name -match '^Runner\.(Worker|Listener)\.exe$' -and (-not $_.ExecutablePath -or -not $_.CommandLine) }).Count) { throw 'An Actions process is unreadable' }
    Remove-StorageLeaf $pathFull $parentFull $request.mode ([bool]$request.dryRun) $processes $ownWorker | ConvertTo-Json -Compress
} catch {
    @{ status = 'skipped'; reason = $_.Exception.Message } | ConvertTo-Json -Compress
} finally {
    if ($acquired) { $mutex.ReleaseMutex() }
    if ($mutex) { $mutex.Dispose() }
}
