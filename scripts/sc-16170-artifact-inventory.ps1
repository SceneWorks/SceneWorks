param(
    [Parameter(Mandatory = $true)][ValidateSet('base', 'control', 'adapter')][string]$Role,
    [Parameter(Mandatory = $true)][string]$ArtifactPath,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$ResolvedRevision,
    [Parameter(Mandatory = $true)][string]$Variant
)

$resolved = (Resolve-Path -LiteralPath $ArtifactPath -ErrorAction Stop).Path
$isDirectory = Test-Path -LiteralPath $resolved -PathType Container
$rootPrefix = $resolved.TrimEnd('\') + '\'
function Relative-ArtifactPath([string]$fullName) {
    if (-not $isDirectory) { return [IO.Path]::GetFileName($fullName) }
    return $fullName.Substring($rootPrefix.Length).Replace('\', '/')
}
$items = if ($isDirectory) {
    Get-ChildItem -LiteralPath $resolved -Recurse -File |
        Sort-Object { Relative-ArtifactPath $_.FullName }
} else {
    Get-Item -LiteralPath $resolved
}
if (-not $items) {
    throw "Artifact has no files: $resolved"
}

$rows = @(foreach ($item in $items) {
    $relative = Relative-ArtifactPath $item.FullName
    $digest = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{ path = $relative; bytes = [uint64]$item.Length; sha256 = $digest }
})

$canonical = ($rows | ForEach-Object { "$($_.path)`t$($_.bytes)`t$($_.sha256)" }) -join "`n"
$canonicalBytes = [Text.UTF8Encoding]::new($false).GetBytes($canonical)
$hasher = [Security.Cryptography.SHA256]::Create()
try {
    $treeDigest = if ($isDirectory) {
        ([BitConverter]::ToString($hasher.ComputeHash($canonicalBytes))).Replace('-', '').ToLowerInvariant()
    } else {
        $rows[0].sha256
    }
} finally {
    $hasher.Dispose()
}
$totalBytes = [uint64](($rows | Measure-Object -Property bytes -Sum).Sum)
$artifact = [ordered]@{
    role = $Role
    path = $resolved
    bytes = $totalBytes
    sha256 = $treeDigest
    repository = $Repository
    resolvedRevision = $ResolvedRevision
    variant = $Variant
    fileCount = @($rows).Count
}

"ARTIFACT_JSON $($artifact | ConvertTo-Json -Compress)"
foreach ($row in $rows) {
    "FILE_JSON $($row | ConvertTo-Json -Compress)"
}
"CAPTURED_AT $([DateTime]::UtcNow.ToString('o'))"
"SCENEWORKS_BASELINE dda526776bffe856164a4991f67229159a06443b"
"INFERENCE_REVISION 7410dea8364f184dbbb07a37c77c1c00e6039a23"
