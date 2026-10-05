import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

const LUID = /^luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i;
const COUNTER = String.raw`\GPU Process Memory(*)\Dedicated Usage`;

export function sharedDeviceProof(bytes, gpuId) {
  const digest = createHash("sha256").update(bytes).digest("hex");
  const wrapper = JSON.parse(bytes.toString("utf8"));
  const census = JSON.parse(wrapper.census);
  const device = census.validatedDevice;
  if (wrapper.backend !== "cuda" || !/^before-[a-z0-9-]+$/.test(wrapper.label ?? "") || wrapper.admitted !== true ||
      census.physicalMode !== "shared-gpu1" || census.admission !== true || census.commandExit !== 0 ||
      device?.physicalMode !== "shared-gpu1" || device.physicalIndex !== gpuId || device.cudaOrdinal !== 0 ||
      !LUID.test(device.luid ?? "") || !/^GPU-[0-9a-f-]+$/i.test(device.uuid ?? "") ||
      !/^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$/i.test(device.pci ?? "") ||
      Object.keys(census.diagnosticFiles ?? {}).length !== 29 ||
      Object.keys(census.diagnosticFileBytesB64 ?? {}).length !== 29 ||
      Object.keys(census.diagnosticFiles).some((name) => !(name in census.diagnosticFileBytesB64))) {
    throw new Error("shared CUDA device proof does not bind one admitted physical GPU and LUID");
  }
  return { sha256: digest, physicalIndex: device.physicalIndex, cudaOrdinal: 0,
    uuid: device.uuid, pci: device.pci, luid: device.luid.toLowerCase() };
}

export function parseOwnedCounter(body, { pid, cargoPid, cargoStartedAt, luid, expectedIdentity = null, now = Date.now() }) {
  const value = typeof body === "string" ? JSON.parse(body) : body;
  const createdAt = Date.parse(value.createdUtc);
  const executablePath = value.executablePath;
  if (value.pid !== pid || value.parentPid !== cargoPid || !Number.isFinite(createdAt) ||
      createdAt < cargoStartedAt - 5000 || createdAt > now + 5000 ||
      typeof executablePath !== "string" || !/[\\/]sceneworks_worker-[0-9a-f]+\.exe$/i.test(executablePath) ||
      typeof value.commandLine !== "string" || !value.commandLine.includes("yue2_memory_profile::capture_case") ||
      value.counter !== COUNTER || !Array.isArray(value.rows)) {
    throw new Error("owned CUDA counter process identity or counter source is invalid");
  }
  const identity = { pid, parentPid: cargoPid, createdUtc: new Date(createdAt).toISOString(), executablePath };
  if (expectedIdentity && (expectedIdentity.createdUtc !== identity.createdUtc ||
      expectedIdentity.executablePath !== identity.executablePath)) {
    throw new Error("owned CUDA process identity changed during capture");
  }
  const prefix = `pid_${pid}_${luid}_phys_0`.toLowerCase();
  let bytes = 0;
  for (const row of value.rows) {
    if (typeof row.instance !== "string" || row.instance.toLowerCase() !== prefix ||
        row.status !== "0" || !Number.isSafeInteger(row.cookedValue) || row.cookedValue < 0) {
      throw new Error("owned CUDA counter has wrong PID/LUID, invalid status, or invalid bytes");
    }
    bytes += row.cookedValue;
    if (!Number.isSafeInteger(bytes)) throw new Error("owned CUDA counter total overflows safe integer");
  }
  if (!value.rows.length) throw new Error("owned CUDA counter has no selected-PID/LUID rows");
  return { identity, bytes, rows: value.rows.length };
}

/** One read-only, selected-PID/LUID Windows counter sample; all inputs are validated before use. */
export function queryOwnedCounter(pid, luid) {
  if (!Number.isSafeInteger(pid) || pid < 1 || !LUID.test(luid)) throw new Error("invalid owned CUDA PID/LUID");
  const script = `
$ErrorActionPreference = 'Stop'
$item = Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}' -ErrorAction Stop
if ($null -eq $item) { throw 'owned test process disappeared' }
$counter = Get-Counter -Counter '${COUNTER}' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
$prefix = 'pid_${pid}_${luid.toLowerCase()}_phys_0'
$rows = @($counter.CounterSamples | Where-Object { $_.InstanceName.ToLowerInvariant() -eq $prefix } |
  ForEach-Object { @{ instance = $_.InstanceName; status = [string]$_.Status; cookedValue = $_.CookedValue } })
@{ pid = [int]$item.ProcessId; parentPid = [int]$item.ParentProcessId;
  createdUtc = $item.CreationDate.ToUniversalTime().ToString('o');
  executablePath = $item.ExecutablePath; commandLine = $item.CommandLine;
  counter = '${COUNTER}'; rows = $rows } | ConvertTo-Json -Compress -Depth 5
`;
  return execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script],
    { encoding: "utf8", timeout: 15000, maxBuffer: 4 * 1024 * 1024 }).trim();
}

/** Bind a live native test's process file, stage requests and Windows reading before attribution. */
export function readOwnedReading(shared, { cargoPid, cargoStartedAt, expectedIdentity = null }, query = queryOwnedCounter) {
  const marks = readFileSync(shared.marksFile, "utf8").trim();
  if (marks && JSON.parse(marks.split("\n").at(-1)).stage === "done") return null;
  const pid = JSON.parse(readFileSync(shared.processFile, "utf8")).processId;
  if (!Number.isSafeInteger(pid) || pid < 1) throw new Error("profile process file has no owned test PID");
  try {
    const request = JSON.parse(readFileSync(path.join(shared.boundaryDir, "request.json"), "utf8"));
    if (request.processId !== pid) throw new Error("stage request changed owned test PID");
  } catch (cause) { if (cause?.code !== "ENOENT") throw cause; }
  const startedAt = Date.now() / 1000;
  const raw = query(pid, shared.proof.luid);
  const reading = parseOwnedCounter(raw, { pid, cargoPid, cargoStartedAt,
    luid: shared.proof.luid, expectedIdentity });
  return { pid, startedAt, raw: JSON.parse(raw), ...reading };
}

/** Only a not-yet-published process or counter may be absent without invalidating coverage. */
export function ownedSamplerFault(cause, hasVerifiedProcess) {
  if (!hasVerifiedProcess && (cause?.code === "ENOENT" ||
      cause?.message === "owned CUDA counter has no selected-PID/LUID rows")) return null;
  return String(cause?.message ?? cause);
}
