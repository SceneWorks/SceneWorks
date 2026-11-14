# A bounded heartbeat for the cold hosted Windows release build. Cargo prints
# crate starts, but not the active child or its resource use during codegen.
while ($true) {
    Start-Sleep -Seconds 120
    $processes = @(Get-Process -Name rustc, cl, nvcc, cicc, ptxas, link -ErrorAction SilentlyContinue)
    $cpuSeconds = ($processes | Measure-Object -Property CPU -Sum).Sum
    $workingBytes = ($processes | Measure-Object -Property WorkingSet64 -Sum).Sum
    $os = Get-CimInstance Win32_OperatingSystem
    $disk = Get-PSDrive D
    Write-Host ("release-compile telemetry utc={0:u} processes={1} cpu_s={2:N0} working_gib={3:N1} free_ram_gib={4:N1} free_disk_gib={5:N1}" -f `
        [DateTime]::UtcNow, $processes.Count, $cpuSeconds, ($workingBytes / 1GB), `
        ($os.FreePhysicalMemory / 1MB), ($disk.Free / 1GB))
    $processes | Sort-Object WorkingSet64 -Descending | Select-Object -First 8 Name, Id, CPU, `
        @{Name='WorkingGiB';Expression={[Math]::Round($_.WorkingSet64 / 1GB, 2)}} | Format-Table | Out-String | Write-Host
}
