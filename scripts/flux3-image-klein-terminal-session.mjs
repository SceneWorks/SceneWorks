// Concrete owned API/worker launcher. Offline by default; no build or dispatch.
// Only explicitly approved provisioning uses the shared legacy credential boundary.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCb } from 'node:child_process';
import { createServer } from 'node:http';
import { promisify } from 'node:util';
import { createHash, randomBytes } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { cp, mkdir, open, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { userInfo } from 'node:os';
import { ROOT, shaPattern, digestPattern } from './flux3-image-klein-terminal-plan.mjs';
import { attemptIdentity, bindOwnedEnv, boundApiUrl, ownedApiController, runnerHfCredential, authorizeFlux3Repository, redactSensitiveText } from './flux3-image-klein-app-lifecycle.mjs';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { validateApprovedPlan, planHash, auditLedger, reserveInvocation, boundInputSpecSha256, admittedHost, candleReplayPrefix, validateCandleDeviceTransitionBinding } from './flux3-image-klein-terminal-budget.mjs';
const execFile = promisify(execFileCb);
// Physical Macs: runner name -> its GitHub runner target and unique label.
export const MAC_RUNNERS = Object.freeze({ 'nax-macos': Object.freeze({ runnerTarget: 'primary', runnerLabel: 'rw-starvector' }), 'nax-macos-2': Object.freeze({ runnerTarget: 'secondary', runnerLabel: 'rw-mage' }) });
export async function fileHash(file) { const h = createHash('sha256'); for await (const chunk of createReadStream(file)) h.update(chunk); return h.digest('hex'); }
export async function contained(file, root) {
  const resolved = await realpath(file), relative = path.relative(await realpath(root), resolved);
  assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'file escaped admitted root');
  assert((await stat(resolved)).isFile(), 'regular file required'); return resolved;
}
export function admittedCandleDevice(a) {
  const transition = a.approvedPlan?.candleDeviceTransition === undefined ? undefined : validateCandleDeviceTransitionBinding(a.approvedPlan);
  const expected = transition?.nextDevice ?? { physicalGpuId: '1' };
  const physicalGpuId = a.physicalGpuId ?? '1', cudaVisibleDevices = a.cudaVisibleDevices ?? '1', logicalCudaOrdinal = a.logicalCudaOrdinal ?? 0;
  assert.equal(physicalGpuId, expected.physicalGpuId, 'physical GPU destination must match approved device transition (legacy GPU1 only)');
  assert.equal(cudaVisibleDevices, expected.physicalGpuId, 'CUDA visibility must mask exactly the approved physical GPU');
  assert.equal(logicalCudaOrdinal, 0, 'masked CUDA logical ordinal must be zero');
  if (transition) {
    assert.equal(a.sourceSha, transition.nextSourceSha, 'device transition source differs from admission');
    assert.equal(a.gpuUuid?.toLowerCase(), expected.gpuUuid.toLowerCase(), 'GPU UUID differs from approved device transition');
    assert.equal(a.pciBusId?.toLowerCase(), expected.pciBusId.toLowerCase(), 'GPU PCI bus differs from approved device transition');
    assert.equal(a.approvedPlanSha256, planHash(a.approvedPlan), 'device transition requires exact approved plan binding');
  }
  return { physicalGpuId, cudaVisibleDevices, logicalCudaOrdinal, gpuUuid: a.gpuUuid, pciBusId: a.pciBusId };
}
export async function validateAdmission(a, backend, { git = async (cwd, ...args) => (await execFile('git', args, { cwd, timeout: 30_000 })).stdout.trim(), root = ROOT, account = () => userInfo().username } = {}) {
  assert.equal(a.actualRun, true, 'execution requires explicit actualRun:true; planning never launches');
  assert.equal(a.backend, backend); assert(a.admissionId && a.approvedBy && a.phases?.length, 'parent resource admission identity/phases required');
  assert.equal(a.runnerAccount, account(), 'admission must name the actual selected runner account (no credentials in receipt)');
  assert(shaPattern.test(a.sourceSha ?? '') && shaPattern.test(a.inferenceMainSha ?? ''), 'exact source and inference MAIN SHAs required');
  assert.equal(await git(root, 'rev-parse', 'HEAD'), a.sourceSha);
  assert.equal(await git(root, 'status', '--porcelain'), '', 'clean committed source required');
  assert(a.inferenceRepository && path.isAbsolute(a.inferenceRepository), 'local inference checkout required for MAIN ancestry proof');
  assert.equal(await git(a.inferenceRepository, 'rev-parse', 'refs/remotes/origin/main'), a.inferenceMainTipSha, 'record the locally fetched inference MAIN tip');
  await git(a.inferenceRepository, 'merge-base', '--is-ancestor', a.inferenceMainSha, 'refs/remotes/origin/main');
  assert(shaPattern.test(a.inferenceMainTipSha ?? ''));
  assert(path.isAbsolute(a.binary ?? '') && digestPattern.test(a.binarySha256 ?? '') && a.binarySourceSha === a.sourceSha && a.binaryBuildReceipt?.path && a.binaryBuildReceipt.root && digestPattern.test(a.binaryBuildReceipt.sha256), 'prebuilt sealed output and parent-approved actual producer receipt required; caller source labels are not proof');
  assert(path.isAbsolute(a.outputRoot ?? '') && path.isAbsolute(a.hfHome ?? ''), 'stable absolute output/cache roots required');
  const relative = path.relative(root, a.outputRoot); assert(relative.startsWith('..') || path.isAbsolute(relative), 'retained output must be outside checkout');
  assert(Number.isInteger(a.timeoutSeconds) && a.timeoutSeconds >= 60 && a.timeoutSeconds <= 10800, 'bounded run timeout required');
  // Which Mac a plan admits is its host lanes' decision (admittedHost); an unsplit plan keeps nax-macos-2.
  if (backend === 'mlx') assert(Object.hasOwn(MAC_RUNNERS, a.runner ?? ''), 'only a known physical Mac runner admitted');
  else { assert(a.runner && typeof a.physicalGpuId === 'string' && typeof a.cudaVisibleDevices === 'string' && a.logicalCudaOrdinal === 0 && /^GPU-[a-f0-9-]+$/i.test(a.gpuUuid ?? ''), 'explicit physical GPU/masked CUDA0 admission required'); admittedCandleDevice(a); }
}
export function safeEnvironment(env) {
  const base = {};
  // Filter keys BEFORE reading values: installed-only planning/tests never inspect ambient secrets.
  for (const k of Object.keys(env)) if (!k.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(k) && !['HF_ENDPOINT', 'HF_TOKEN_PATH', 'HF_XET_CACHE', 'TRANSFORMERS_CACHE', 'CARGO_TARGET_DIR'].includes(k)) base[k] = env[k];
  return base;
}
export function offlineChildEnv(env, admission, identity, url) {
  const base = safeEnvironment(env), device = admission.backend === 'candle' ? admittedCandleDevice(admission) : undefined;
  Object.assign(base, { HF_HOME: admission.hfHome, HF_HUB_CACHE: path.join(admission.hfHome, 'hub'), HUGGINGFACE_HUB_CACHE: path.join(admission.hfHome, 'hub'), HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', SCENEWORKS_HUGGINGFACE_BASE_URL: 'http://127.0.0.1:9', SCENEWORKS_RESOLVED_CACHE_ENABLED: 'false', SCENEWORKS_POLL_SECONDS: '1', SCENEWORKS_HEARTBEAT_SECONDS: '2', SCENEWORKS_BACKEND_CANDLE_ENABLED: String(admission.backend === 'candle'), SCENEWORKS_BACKEND_MLX_ENABLED: String(admission.backend === 'mlx'), SCENEWORKS_CANDLE_REQUIRED: admission.backend === 'candle' ? '1' : '0', CUDA_VISIBLE_DEVICES: device ? device.cudaVisibleDevices : '-1', CUDA_DEVICE_ORDER: 'PCI_BUS_ID', SCENEWORKS_GPU_ID: device ? device.physicalGpuId : 'mlx' });
  return { ...bindOwnedEnv(base, identity, url), ...(url ? { SCENEWORKS_WORKER_ONLY: '1', SCENEWORKS_WORKER_CHILD: '1' } : {}) };
}
// Driver API UUID query runs in a separate short process with the SAME visibility mask.
const cudaUuidProbe = `Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class TerminalCudaIdentity { [DllImport("nvcuda.dll")] public static extern int cuInit(uint flags); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetCount(out int count); [DllImport("nvcuda.dll")] public static extern int cuDeviceGet(out int device, int ordinal); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetUuid_v2([Out] byte[] uuid, int device); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetLuid([Out] byte[] luid, out uint mask, int device); }'; if([TerminalCudaIdentity]::cuInit(0) -ne 0){throw "CUDA init"}; $n=0; if([TerminalCudaIdentity]::cuDeviceGetCount([ref]$n) -ne 0 -or $n -ne 1){throw "CUDA mask"}; $d=0; if([TerminalCudaIdentity]::cuDeviceGet([ref]$d,0) -ne 0){throw "CUDA ordinal"}; $u=New-Object byte[] 16; if([TerminalCudaIdentity]::cuDeviceGetUuid_v2($u,$d) -ne 0){throw "CUDA UUID"}; $h=([BitConverter]::ToString($u)).Replace("-","").ToLowerInvariant(); $l=New-Object byte[] 8; $m=[uint32]0; $luid=""; if([TerminalCudaIdentity]::cuDeviceGetLuid($l,[ref]$m,$d) -eq 0){$luid=(" luid=0x{0:x8}_0x{1:x8}" -f [BitConverter]::ToUInt32($l,4),[BitConverter]::ToUInt32($l,0))}; Write-Output ("GPU-"+$h.Substring(0,8)+"-"+$h.Substring(8,4)+"-"+$h.Substring(12,4)+"-"+$h.Substring(16,4)+"-"+$h.Substring(20,12)+$luid)`;
// Windows WDDM C and C+G rows can have N/A SMI memory, regardless of compute activity.
// Both use the SAME bounded dedicated-allocation accounting on CUDA's actual adapter LUID.
// Dedicated counters never establish CUDA exclusivity or competing-compute inactivity.
export const FOREIGN_GRAPHICS_PROCESS_LIMIT_BYTES = 1024 * 1024 ** 2;
export const FOREIGN_GRAPHICS_TOTAL_LIMIT_BYTES = 3 * 1024 ** 3;
export function gpuProcessMemoryScript(luid, requiredProcesses) {
  assert(/^0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i.test(luid));
  assert(requiredProcesses.length && requiredProcesses.every(p => Number.isSafeInteger(p.pid) && p.pid > 0));
  // Capture (do not discard) nonterminating provider errors. The JS validator below permits
  // partial results ONLY when known invalid UNREQUIRED instances explain those errors and
  // every required PID/start-time/physical-node sample is independently valid.
  return `$ErrorActionPreference='Stop'; $wanted=@(${requiredProcesses.map(p => p.pid).join(',')}); $counterErrors=@();
function Get-RequiredCounterMetadata {
  $set=Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop;
  if ($null -eq $set.PathsWithInstances) { throw 'Independent GPU counter path metadata unavailable' };
  $prefix='\\GPU Process Memory('; $suffix=')\\Dedicated Usage'; $found=@(); $seen=@{};
  foreach ($counterPath in $set.PathsWithInstances) {
    $offset=$counterPath.LastIndexOf($prefix,[StringComparison]::OrdinalIgnoreCase);
    if ($offset -lt 0 -or -not $counterPath.EndsWith($suffix,[StringComparison]::OrdinalIgnoreCase)) { continue };
    $instance=$counterPath.Substring($offset+$prefix.Length,$counterPath.Length-$offset-$prefix.Length-$suffix.Length);
    if ($instance -notmatch '^pid_([0-9]+)_luid_(0x[0-9a-f]{8}_0x[0-9a-f]{8})_') { continue };
    $pidValue=[int]$Matches[1]; $luidValue=$Matches[2].ToLowerInvariant();
    if ($wanted -notcontains $pidValue -or $luidValue -ne '${luid.toLowerCase()}') { continue };
    if ($instance -notmatch '^pid_[0-9]+_luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}_phys_([0-9]+)$') { throw 'Required GPU counter instance metadata malformed' };
    $physicalNode=[int]$Matches[1]; $key=([string]$pidValue)+':'+$physicalNode;
    if ($seen.ContainsKey($key)) { throw 'Duplicate required GPU counter instance metadata' }; $seen[$key]=$true;
    $found += [pscustomobject]@{pid=$pidValue;luid=$luidValue;physicalNode=$physicalNode}
  };
  foreach ($requiredPid in $wanted) { if (-not @($found | Where-Object { $_.pid -eq $requiredPid }).Count) { throw 'Required GPU counter metadata missing' } };
  $found
};
$expectedBefore=@(Get-RequiredCounterMetadata);
$queryPaths=@($expectedBefore | ForEach-Object { '\\GPU Process Memory(pid_'+$_.pid+'_luid_'+$_.luid+'_phys_'+$_.physicalNode+')\\Dedicated Usage' });
$sets=@(Get-Counter -Counter $queryPaths -MaxSamples 1 -ErrorAction Continue -ErrorVariable counterErrors 2>$null);
$samples=@($sets | ForEach-Object { $_.CounterSamples } | ForEach-Object {
  $pidValue=-1; $luidValue=$null; $physicalNode=-1;
  if ($_.InstanceName -match '^pid_([0-9]+)_luid_(0x[0-9a-f]{8}_0x[0-9a-f]{8})_phys_([0-9]+)$') { $pidValue=[int]$Matches[1]; $luidValue=$Matches[2].ToLowerInvariant(); $physicalNode=[int]$Matches[3] };
  $row=[ordered]@{pid=$pidValue;luid=$luidValue;physicalNode=$physicalNode;status=[long]$_.Status}; $value=[double]$_.CookedValue;
  if (($_.Status -eq 0 -or $_.Status -eq 1) -and -not [double]::IsNaN($value) -and -not [double]::IsInfinity($value) -and $value -ge 0) { $row.cookedBytes=$value }; [pscustomobject]$row
});
$expectedAfter=@(Get-RequiredCounterMetadata);
$identities=@(Get-CimInstance Win32_Process | Where-Object { $wanted -contains $_.ProcessId } | ForEach-Object { ${requiredProcesses.some(p => Object.hasOwn(p, 'sessionId')) ? "if ($null -eq $_.SessionId) { throw 'Required GPU process session identity unavailable' }; " : ''}[pscustomobject]@{pid=[int]$_.ProcessId;startedAt=$_.CreationDate.ToUniversalTime().ToString('o')${requiredProcesses.some(p => Object.hasOwn(p, 'sessionId')) ? ';sessionId=[int]$_.SessionId' : ''}} });
$errors=@($counterErrors | ForEach-Object { $code='UNKNOWN_COUNTER_PROVIDER_ERROR'; if (([string]$_.FullyQualifiedErrorId).StartsWith('CounterApiError,')) { $code='CounterApiError' }; [pscustomobject]@{category=[string]$_.CategoryInfo.Category;code=$code} });
[pscustomobject]@{expectedBefore=$expectedBefore;expectedAfter=$expectedAfter;samples=$samples;identities=$identities;errors=$errors} | ConvertTo-Json -Depth 5 -Compress`;
}
export function validateGpuProcessMemory(observation, luid, requiredProcesses) {
  assert(/^0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i.test(luid));
  assert(requiredProcesses.length && new Set(requiredProcesses.map(p => p.pid)).size === requiredProcesses.length);
  assert(Array.isArray(observation?.samples) && Array.isArray(observation.identities) && Array.isArray(observation.errors) && Array.isArray(observation.expectedBefore) && Array.isArray(observation.expectedAfter), 'structured GPU counter observation and independent path metadata required');
  const required = new Map(requiredProcesses.map(p => { assert(Number.isSafeInteger(p.pid) && p.pid > 0 && typeof p.startedAt === 'string' && Number.isFinite(Date.parse(p.startedAt)), 'required GPU process start-time proof missing'); return [p.pid, p.startedAt]; }));
  const expected = metadata => {
    const keys = new Set(), pids = new Set();
    for (const row of metadata) {
      assert(row && Object.keys(row).sort().join(',') === 'luid,physicalNode,pid' && required.has(row.pid) && row.luid === luid.toLowerCase() && Number.isSafeInteger(row.physicalNode) && row.physicalNode >= 0, 'required GPU counter path metadata malformed');
      const key = `${row.pid}:${row.physicalNode}`; assert(!keys.has(key), 'duplicate required GPU counter path metadata'); keys.add(key); pids.add(row.pid);
    }
    for (const pid of required.keys()) assert(pids.has(pid), 'required GPU counter path metadata missing');
    return keys;
  };
  // Independently enumerated before/after via ListSet.PathsWithInstances, NEVER inferred
  // from a partial value response. A single sample per PID is not completeness evidence.
  const expectedBefore = expected(observation.expectedBefore), expectedAfter = expected(observation.expectedAfter);
  assert.deepEqual([...expectedBefore].sort(), [...expectedAfter].sort(), 'required GPU counter physical-node set changed during observation');
  const identities = new Map();
  for (const p of observation.identities) { assert(required.has(p.pid) && !identities.has(p.pid), 'unexpected/duplicate GPU counter process identity'); identities.set(p.pid, p.startedAt); }
  for (const [pid, startedAt] of required) assert.equal(identities.get(pid), startedAt, 'required GPU process missing/reused during counter observation');
  for (const process of requiredProcesses) if (Object.hasOwn(process, 'sessionId')) { assert(Number.isSafeInteger(process.sessionId) && process.sessionId >= 0, 'required GPU session identity malformed'); assert.equal(observation.identities.find(p => p.pid === process.pid)?.sessionId, process.sessionId, 'required GPU session identity missing/changed'); }
  const memory = new Map(), seen = new Set(), observedRequired = new Set(); let invalidUnrequired = 0;
  for (const sample of observation.samples) {
    assert(Number.isSafeInteger(sample.pid) && sample.pid > 0 && sample.luid === luid.toLowerCase() && Number.isSafeInteger(sample.physicalNode) && sample.physicalNode >= 0 && Number.isSafeInteger(sample.status) && sample.status >= 0 && sample.status <= 0xffffffff, 'malformed GPU counter sample identity/status');
    const key = `${sample.pid}:${sample.physicalNode}`; assert(!seen.has(key), 'duplicate GPU counter physical-node sample'); seen.add(key);
    const valid = [0, 1].includes(sample.status) && Number.isFinite(sample.cookedBytes) && sample.cookedBytes >= 0;
    if (required.has(sample.pid)) {
      assert(expectedBefore.has(key), 'required GPU counter value has no independently enumerated physical-node path');
      assert(valid, 'required foreign GPU process has invalid counter data'); observedRequired.add(key);
      memory.set(sample.pid, (memory.get(sample.pid) ?? 0) + sample.cookedBytes);
    } else if (!valid) {
      // SDK PdhMsg.h: per-instance NO_INSTANCE / INVALID_DATA. Unknown provider statuses
      // never become an exception merely because the affected PID is not required.
      assert([0x800007d1, 0xc0000bba].includes(sample.status), 'unknown unrequired GPU counter failure'); invalidUnrequired++;
    }
  }
  assert.deepEqual([...observedRequired].sort(), [...expectedBefore].sort(), 'required GPU counter physical-node coverage incomplete');
  for (const pid of required.keys()) assert(memory.has(pid) && Number.isSafeInteger(memory.get(pid)) && memory.get(pid) >= 0, 'required foreign GPU counter sample missing/nonintegral');
  assert(observation.errors.every(e => e?.category === 'InvalidResult' && e.code === 'CounterApiError'), 'GPU counter provider/permission failure');
  assert((observation.errors.length > 0) === (invalidUnrequired > 0), 'unexplained GPU counter provider error/invalid sample');
  return memory;
}
/** Historical API name: every foreign C AND C+G is measured against the same limits.
 * Dedicated allocation is not activity, nor is device-total usage attributable to our process. */
export function assessForeignGraphics(foreignPids, memoryByPid, { processLimit = FOREIGN_GRAPHICS_PROCESS_LIMIT_BYTES, totalLimit = FOREIGN_GRAPHICS_TOTAL_LIMIT_BYTES } = {}) {
  let totalBytes = 0, maxProcessBytes = 0;
  for (const pid of foreignPids) {
    const bytes = memoryByPid.get(pid);
    assert(Number.isSafeInteger(bytes) && bytes >= 0, `foreign C/C+G process ${pid} on approved GPU is unmeasurable; refusing`);
    assert(bytes <= processLimit, `foreign C/C+G process ${pid} uses ${Math.ceil(bytes / 1024 ** 2)} MiB of approved GPU (limit ${processLimit / 1024 ** 2} MiB)`);
    totalBytes += bytes; maxProcessBytes = Math.max(maxProcessBytes, bytes);
  }
  assert(totalBytes <= totalLimit, `foreign C/C+G processes use ${Math.ceil(totalBytes / 1024 ** 2)} MiB of approved GPU in total (limit ${totalLimit / 1024 ** 2} MiB)`);
  return { processes: foreignPids.length, totalBytes, maxProcessBytes, processLimitBytes: processLimit, totalLimitBytes: totalLimit };
}
export async function observeProcessTable(checked, platform) {
  if (platform === 'win32') {
    const script = 'Get-CimInstance Win32_Process | Select-Object @{n="pid";e={$_.ProcessId}},@{n="ppid";e={$_.ParentProcessId}},@{n="command";e={$_.Name}},@{n="executable";e={$_.ExecutablePath}},@{n="startedAt";e={$_.CreationDate.ToUniversalTime().ToString("o")}},@{n="sessionId";e={$_.SessionId}} | ConvertTo-Json -Compress';
    const value = JSON.parse((await checked('powershell.exe', ['-NoProfile','-Command',script], { timeout: 5000, maxBuffer: 4 * 1024 * 1024 })).stdout); return Array.isArray(value) ? value : [value];
  }
  const text = (await checked('ps', ['-axo','pid=,ppid=,lstart=,comm='], { timeout: 5000, maxBuffer: 4 * 1024 * 1024 })).stdout;
  return text.split(/\r?\n/).filter(line => line.trim()).map(line => { const m = /^\s*(\d+)\s+(\d+)\s+(\S+\s+\S+\s+\d+\s+\S+\s+\d+)\s+(.+)$/.exec(line); assert(m, 'process identity observation unavailable'); return { pid: Number(m[1]), ppid: Number(m[2]), startedAt: m[3].replace(/\s+/g,' '), command: m[4].trim() }; });
}
export function processAncestors(table, pid) { const ids = new Set(); let current = pid; while (current > 0 && !ids.has(current)) { ids.add(current); current = table.find(p => p.pid === current)?.ppid ?? 0; } return ids; }
export async function readMacWorkloadLease(a) {
  assert(a.macWorkloadLease?.path && a.macWorkloadLease.root && digestPattern.test(a.macWorkloadLease.sha256), 'secondary Mac requires coordinator exclusive-workload attestation lease');
  const file = await contained(a.macWorkloadLease.path, a.macWorkloadLease.root), bytes = await readFile(file); assert(bytes.length <= 1024 * 1024 && createHash('sha256').update(bytes).digest('hex') === a.macWorkloadLease.sha256, 'coordinator lease bytes changed'); return JSON.parse(bytes);
}
const ownedGpuIdentities = new WeakMap();
/** Fixed safe observability for the opaque GPU-headroom family: an assertion keeps its exact
 * text and predicate, and additionally carries one allowlisted reason code for the safe report. */
async function gpuStage(code, predicate) {
  try { return await predicate(); }
  catch (error) {
    if (['AssertionError', 'SyntaxError'].includes(error?.name) && !error.safeCode) { const tagged = new Error(error.message); tagged.name = 'AssertionError'; tagged.safeCode = code; throw tagged; }
    throw error;
  }
}
export async function guardHardware(a, ownPids = [], { exec = execFile, platform = process.platform, arch = process.arch, env = process.env, processTable, leaseReader = readMacWorkloadLease, controllerPid = process.pid, now = Date.now, onPhase = () => {} } = {}) {
  onPhase('queue_ownership');
  const checked = (command, args, options = {}) => exec(command, args, { ...options, env: { ...safeEnvironment(env), ...options.env } });
  const table = await (processTable ? processTable() : observeProcessTable(checked, platform)); assert(Array.isArray(table) && table.some(p => p.pid === controllerPid), 'cannot prove controller process ancestry');
  const ancestors = processAncestors(table, controllerPid), runners = table.filter(p => /^Runner\.Worker(?:\.exe)?$/i.test(path.basename(p.command)));
  // A queued campaign may admit exactly three companions, but only after the signed four-slot
  // proof revalidates API job identity, live heartbeats and actual PID/start-time/executable ancestry.
  // Never use ownPids as a Runner.Worker allowlist. Local and Mac guards are unchanged.
  let companions = [];
  if (a.queueReservation) {
    assert(platform === 'win32' && a.backend === 'candle', 'queue reservations are Windows Candle only');
    const { verifyQueueReservation } = await import('./flux3-terminal-queue.mjs');
    companions = await verifyQueueReservation(a, table, controllerPid, now);
  }
  assert(runners.every(p => ancestors.has(p.pid) || companions.includes(p.pid)), 'foreign Runner.Worker active; only the ancestor hosting THIS job or verified queue companions are allowed');
  if (a.backend === 'mlx') {
    assert(platform === 'darwin' && arch === 'arm64', 'physical Apple Silicon Mac required');
    const mac = MAC_RUNNERS[a.runner]; assert(mac, 'only a known physical Mac runner admitted');
    assert.equal(a.runnerTarget, mac.runnerTarget); assert.equal(a.runnerLabel, mac.runnerLabel); assert.equal(env.RUNNER_NAME, a.runner, 'admitted Mac runner identity required (the label is not its runner name)');
    const host = (await checked('hostname', [], { timeout: 5000 })).stdout.trim(); assert.equal(host.split('.')[0], a.hostName, 'admitted Mac physical hostname mismatch');
    assert(a.hostName, 'explicit admitted secondary hostname required');
    const lease = await leaseReader(a); assert.equal(lease.kind, 'flux3-secondary-mac-exclusive-workload-lease'); assert(lease.coordinatorApprovalId && lease.approvedBy && lease.exclusiveGpuWorkload === true && lease.noForeignGpuWorkload === true, 'exclusive Metal workload cannot be inferred from hostname; coordinator must attest it');
    for (const key of ['admissionId','sourceSha','inferenceMainSha','runnerAccount','hostName']) assert.equal(lease[key], a[key], `Mac lease ${key} mismatch`);
    assert(Date.parse(lease.observedAt) <= now() && Date.parse(lease.expiresAt) >= (a.runDeadline ?? now() + a.timeoutSeconds * 1000), 'exclusive-workload lease expired/does not cover the admitted run');
    assert.deepEqual([...runners.map(p => p.pid)].sort(), [...(lease.ancestorRunnerPids ?? [])].sort(), 'runner lease must identify the actual hosting ancestors, not foreign runners');
    assert(Array.isArray(lease.baselineProcesses));
    for (const p of table.filter(p => p.pid > 0)) { if (processAncestors(table, p.pid).has(controllerPid) || ownPids.includes(p.pid)) continue; assert(lease.baselineProcesses.some(old => old.pid === p.pid && old.ppid === p.ppid && old.command === p.command && old.startedAt === p.startedAt && p.startedAt), 'new/reused foreign process invalidates exclusive Metal lease; return admission to coordinator'); }
    return { runner: a.runner, runnerTarget: a.runnerTarget, runnerLabel: a.runnerLabel, host, runtime: { platform, arch }, backend: 'mlx', coordinatorLeaseId: lease.coordinatorApprovalId, exclusivity: 'coordinator-attested-and-process-monitored' };
  }
  assert.equal(platform, 'win32');
  const device = admittedCandleDevice(a), smi = async args => (await checked('nvidia-smi', args, { timeout: 5000 })).stdout.trim();
  const owned = new Set();
  for (const pid of ownPids) {
    const matches = table.filter(p => p.pid === pid); assert.equal(matches.length, 1, 'owned GPU process identity missing/ambiguous');
    const p = matches[0], controller = table.find(p => p.pid === controllerPid);
    assert(p.startedAt && Number.isFinite(Date.parse(p.startedAt)) && Number.isSafeInteger(p.sessionId) && p.sessionId >= 0, 'owned GPU process start/session identity missing');
    const descendant = pid !== controllerPid && processAncestors(table, pid).has(controllerPid);
    // The adapter-native-boundary CLI has its sealed adapter as its immediate parent.
    const adapterParent = controller.ppid === pid && path.isAbsolute(a.binary ?? '') && path.isAbsolute(p.executable ?? '') && path.resolve(p.executable) === path.resolve(a.binary);
    assert(descendant || adapterParent, 'unrelated PID cannot be classified as owned GPU process');
    const identities = ownedGpuIdentities.get(a) ?? new Map(); ownedGpuIdentities.set(a, identities);
    if (identities.has(pid)) assert.deepEqual(p, identities.get(pid), 'owned GPU PID identity changed/reused across monitoring');
    else identities.set(pid, structuredClone(p));
    owned.add(pid);
  }
  onPhase('physical_gpu_uuid');
  const [index, uuid, pciBusId] = (await smi(['-i', device.physicalGpuId, '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'])).split(',').map(s => s.trim());
  assert.equal(index, device.physicalGpuId, 'SMI index differs from approved physical GPU'); assert.equal(uuid?.toLowerCase(), a.gpuUuid?.toLowerCase(), 'SMI UUID differs from approved physical GPU');
  assert(/^GPU-[a-f0-9-]+$/i.test(uuid ?? '') && /^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$/i.test(pciBusId ?? ''), 'SMI UUID/PCI identity unavailable');
  if (device.pciBusId !== undefined) assert.equal(pciBusId.toLowerCase(), device.pciBusId.toLowerCase(), 'SMI PCI differs from approved physical GPU');
  onPhase('masked_cuda_uuid');
  const [masked = '', luidField] = (await checked('powershell.exe', ['-NoProfile', '-Command', cudaUuidProbe], { timeout: 30_000, env: { CUDA_VISIBLE_DEVICES: device.cudaVisibleDevices, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' } })).stdout.trim().split(/\s+/);
  assert.equal(masked.toLowerCase(), uuid.toLowerCase(), 'masked CUDA0 must be approved physical GPU');
  const luid = /^luid=(0x[0-9a-f]{8}_0x[0-9a-f]{8})$/i.exec(luidField ?? '')?.[1]?.toLowerCase();
  assert(luid, 'approved GPU is unmeasurable: CUDA reported no adapter LUID; refusing');
  onPhase('gpu_process_identity');
  const xml = await smi(['-i', device.physicalGpuId, '-q', '-x']); assert(xml.includes('<processes>') && xml.includes(uuid), 'approved GPU process identity unavailable');
  const types = new Map();
  for (const [, block] of xml.matchAll(/<process_info>([\s\S]*?)<\/process_info>/g)) {
    const pid = Number(/<pid>([^<]+)<\/pid>/.exec(block)?.[1]), type = /<type>([^<]+)<\/type>/.exec(block)?.[1]?.trim();
    assert(Number.isSafeInteger(pid) && pid > 0 && ['C', 'G', 'C+G'].includes(type) && !types.has(pid), 'unknown/duplicate GPU process identity/type'); types.set(pid, type);
  }
  const apps = (await smi(['--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])).split(/\r?\n/).filter(Boolean).map(line => line.split(',').map(x => x.trim()));
  const foreign = [...types].filter(([pid, type]) => ['C','C+G'].includes(type) && !owned.has(pid)).map(([pid]) => pid), seenApps = new Set();
  for (const row of apps) {
    const [pidText, physicalUuid] = row, pid = Number(pidText); assert(row.length === 2 && Number.isSafeInteger(pid) && pid > 0 && /^GPU-[a-f0-9-]+$/i.test(physicalUuid), 'compute-app identity unavailable');
    const key = `${pid}:${physicalUuid.toLowerCase()}`; assert(!seenApps.has(key), 'duplicate compute-app identity'); seenApps.add(key);
    if (owned.has(pid)) assert.equal(physicalUuid.toLowerCase(), uuid.toLowerCase(), 'owned process touched wrong physical GPU UUID');
    if (physicalUuid.toLowerCase() === uuid.toLowerCase()) assert(['C','C+G'].includes(types.get(pid)), 'compute-app has unidentified type on approved GPU');
  }
  for (const [pid, type] of types) if (['C','C+G'].includes(type)) assert(seenApps.has(`${pid}:${uuid.toLowerCase()}`), 'GPU process compute-app identity missing');
  onPhase('gpu_headroom');
  const [totalBytes, freeBytes] = await gpuStage('GPU_HEADROOM_COUNTERS_UNAVAILABLE', async () => {
    const headroom = (await smi(['-i', device.physicalGpuId, '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'])).split(',');
    assert(headroom.length === 2 && headroom.every(x => /^\d+(?:\.\d+)?$/.test(x.trim())), 'GPU headroom counters unavailable');
    return headroom.map(x => Number(x.trim()) * 1024 ** 2);
  });
  await gpuStage('GPU_HEADROOM_INVALID', async () => assert([totalBytes,freeBytes].every(Number.isSafeInteger) && totalBytes > 0 && freeBytes > 0 && freeBytes <= totalBytes, 'valid positive GPU headroom required'));
  let memory = new Map(), counterProof;
  if (foreign.length) {
    const requiredProcesses = await gpuStage('GPU_FOREIGN_IDENTITY_UNAVAILABLE', async () => foreign.map(pid => { const matches = table.filter(p => p.pid === pid); assert.equal(matches.length, 1, 'foreign GPU process identity missing/ambiguous'); const p = matches[0]; assert(Number.isSafeInteger(p.sessionId) && p.sessionId >= 0, 'foreign GPU process session identity missing'); return { pid, startedAt: p.startedAt, sessionId: p.sessionId }; }));
    await gpuStage('GPU_FOREIGN_COUNTER_INVALID', async () => {
      const observation = JSON.parse((await checked('powershell.exe', ['-NoProfile', '-Command', gpuProcessMemoryScript(luid, requiredProcesses)], { timeout: 30_000 })).stdout);
      memory = validateGpuProcessMemory(observation, luid, requiredProcesses); counterProof = { requiredProcesses, ...observation };
    });
  }
  const after = await (processTable ? processTable() : observeProcessTable(checked, platform));
  await gpuStage('GPU_FOREIGN_IDENTITY_CHANGED', async () => {
    for (const pid of new Set([...owned, ...foreign])) { const before = table.find(p => p.pid === pid), matches = after.filter(p => p.pid === pid); assert.equal(matches.length, 1, 'GPU process exited/reused during observation'); assert.deepEqual(matches[0], before, 'GPU process identity changed/reused during observation'); }
  });
  // Historical wire name retained: this is the combined C AND C+G allocation, not graphics-only.
  const foreignGraphics = { luid, ...await gpuStage('GPU_FOREIGN_CAP_EXCEEDED', async () => assessForeignGraphics(foreign, memory)) };
  if (counterProof) Object.defineProperty(foreignGraphics, 'counterProof', { value: counterProof, enumerable: false });
  return { physicalGpuId: device.physicalGpuId, gpuUuid: uuid, logicalCudaOrdinal: 0, ownComputePids: [...owned].filter(pid => seenApps.has(`${pid}:${uuid.toLowerCase()}`)), foreignGraphics, pciBusId, gpuHeadroom: {totalBytes,freeBytes}, sharedPolicy: { sharingMode: 'bounded-shared', foreignProcessTypes: ['C','C+G'], exclusiveThroughputClaim: false, deviceTotalAttributionClaim: false, computeActivityAssessment: 'unsupported-dedicated-memory-only' } };
}
// Called ONLY by the adapter's synchronous native-boundary child CLI, before reservation.
// No resource template can supply its own process/build/hardware observation.
export async function verifyResourceProducerHardware(a, context, dependencies = {}) {
  const io = { validate: validateAdmission, verifyBuild: async admission => (await import('./flux3-image-klein-terminal-build.mjs')).verifyTerminalBuild(admission, { role: 'resource-adapter' }), guard: guardHardware, exec: execFile, hash: fileHash, realpath, parentPid: process.ppid, controllerPid: process.pid, platform: process.platform, env: process.env, ...dependencies };
  assert(Number.isInteger(context.adapterPid) && context.adapterPid > 0 && context.adapterPid === io.parentPid, 'native-boundary CLI must be the ACTUAL adapter child, not a caller-labelled PID');
  await io.validate(a, a.backend); await io.verifyBuild(a);
  const approved = validateApprovedPlan(a.approvedPlan); admittedHost(a.approvedPlan, a); assert.equal(context.approvedPlanSha256, approved.approvedPlanSha256); assert.equal(a.approvedPlanSha256, approved.approvedPlanSha256);
  assert.equal(a.approvedPlan.sourceSha, a.sourceSha); assert.equal(a.approvedPlan.inferenceMainSha, a.inferenceMainSha);
  for (const key of ['sourceSha','binarySha256']) assert.equal(context[key], a[key], 'adapter context does not match sealed source/build');
  const invocation = a.approvedPlan.plan.generationInvocations.find(i => i.invocationId === context.invocationId);
  assert(invocation && invocation.backend === a.backend && invocation.caseId === context.caseId && ['resource_anchor','resource_supplemental'].includes(invocation.kind), 'actual resource invocation is absent from immutable backend plan');
  assert(digestPattern.test(context.requestSha256) && invocation.requestSha256 === context.requestSha256, 'actual resource request differs from immutable normalized request');
  const checked = (command, args, options = {}) => io.exec(command, args, { ...options, env: { ...safeEnvironment(io.env), ...options.env } });
  const observe = io.processTable ?? (() => observeProcessTable(checked, io.platform));
  const before = await observe(), producer = before.find(p => p.pid === context.adapterPid);
  assert(producer?.startedAt && before.some(p => p.pid === io.controllerPid && p.ppid === context.adapterPid), 'actual adapter/inner CLI process identity unavailable');
  const executable = producer.executable ?? producer.command; assert(path.isAbsolute(executable ?? ''), 'actual adapter executable path unavailable; process name is not compiled identity');
  const resolved = await io.realpath(executable); assert.equal(resolved, await io.realpath(a.binary), 'actual adapter process is not the sealed admitted executable'); assert.equal(await io.hash(resolved), a.binarySha256, 'actual running adapter executable bytes differ from owned build');
  const proof = await io.guard(a, [context.adapterPid]);
  if (a.backend === 'candle') { const target = admittedCandleDevice(a); assert.equal(proof.physicalGpuId, target.physicalGpuId); assert.equal(proof.gpuUuid?.toLowerCase(), a.gpuUuid.toLowerCase()); if (target.pciBusId !== undefined) assert.equal(proof.pciBusId?.toLowerCase(), target.pciBusId.toLowerCase(), 'resource producer PCI proof differs from admitted target'); assert.equal(proof.logicalCudaOrdinal, target.logicalCudaOrdinal); assert(proof.ownComputePids?.includes(context.adapterPid), 'actual adapter PID/approved GPU UUID compute observation missing'); }
  else { assert(Object.hasOwn(MAC_RUNNERS, a.runner ?? '') && proof.runner === a.runner, 'observed Mac runner differs from the admitted lane host'); assert.equal(proof.host?.split('.')[0], a.hostName); assert.deepEqual(proof.runtime, { platform: 'darwin', arch: 'arm64' }); assert(proof.coordinatorLeaseId && proof.exclusivity === 'coordinator-attested-and-process-monitored', 'actual secondary Mac exclusive runtime observation missing'); }
  const after = (await observe()).find(p => p.pid === context.adapterPid); assert.deepEqual(after, producer, 'adapter exited/changed identity during hardware observation');
  return { kind: 'flux3-resource-producer-hardware-observation', backend: a.backend, adapterPid: context.adapterPid, caseId: context.caseId, invocationId: context.invocationId, requestSha256: context.requestSha256, sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, binarySha256: a.binarySha256, buildReceiptSha256: a.binaryBuildReceipt.sha256, approvedPlanSha256: approved.approvedPlanSha256, observedAt: new Date().toISOString(), adapterProcess: producer, proof };
}
export class ProvisioningBlocked extends Error {}
export function selectProductInvocation(approval, cell, index, body, replayPrefix) {
  assert(Number.isSafeInteger(index) && index >= 0);
  const invocations = approval.plan.generationInvocations.filter(i => i.backend === cell.backend && i.caseId === cell.caseId && i.kind === 'product_generation');
  if (replayPrefix && cell.caseId === replayPrefix.caseId) {
    assert(cell.caseId === 'candle.distilled.representative-repro.bf16' && cell.kind === 'replay' && cell.backend === 'candle');
    assert.equal(index, 0, 'the fixed replay continuation permits exactly one new render');
    assert.equal(replayPrefix.authorizationSha256, approval.candleSourceTransitionSha256);
    assert.equal(invocations.length, 2); assert.equal(invocations[0].invocationId, replayPrefix.consumedInvocationId); assert.equal(invocations[1].invocationId, replayPrefix.nextInvocationId);
    assert(typeof body.projectId === 'string' && /^[a-zA-Z0-9_-]+$/.test(body.projectId));
    assert.deepEqual(body, { ...replayPrefix.recordedReplayRequest, projectId: body.projectId }, 'only project lineage may differ from the retained replay request');
    return invocations[1];
  }
  return invocations[index];
}
export async function launchTerminalSession(a, dependencies = {}) {
  // Dependency overrides are for weights-free tests; production CLI never accepts a launcher module.
  const io = { mkdir, cp, open, readFile, writeFile, createServer, readCredential: runnerHfCredential, authorize: authorizeFlux3Repository, reserve: reserveInvocation, execWitness: execFile, hash: fileHash, spawn, guard: guardHardware, controller: ownedApiController, sleep: ms => new Promise(r => setTimeout(r, ms)), now: Date.now, env: process.env, ...dependencies };
  a = { ...a, runDeadline: io.now() + a.timeoutSeconds * 1000 };
  const identity = attemptIdentity(a.outputRoot), children = [], handles = [], abort = new AbortController();
  let artifactMapping, replayPrefix, initialLedgerReceipt;
  const execution = { sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, runner: a.runner, runnerAccount: a.runnerAccount, binarySha256: a.binarySha256, buildReceiptSha256: a.binaryBuildReceipt?.sha256 };
  let endpoint, worker, currentCase, workerPaused = false, generation = 0, monitoring = false, restarting = false, closing = false, monitor, monitorTask, faultServer, faultUrl;
  const invocationIndexes = new Map(), permits = [];
  const faultRequests = [], authorizations = [], secrets = [], cleanup = []; let credential;
  const childEnv = url => { const env = offlineChildEnv(io.env, a, identity, url); if (faultUrl) { env.SCENEWORKS_HUGGINGFACE_BASE_URL = faultUrl; env.HF_ENDPOINT = faultUrl; if (!url) { delete env.HF_HUB_OFFLINE; delete env.TRANSFORMERS_OFFLINE; } } if (!url && a.approvedProvisioning?.length) { delete env.HF_HUB_OFFLINE; delete env.TRANSFORMERS_OFFLINE; env.HF_ENDPOINT = 'https://huggingface.co'; env.SCENEWORKS_HUGGINGFACE_BASE_URL = 'https://huggingface.co'; if (credential) env.HF_TOKEN = credential; } return env; };
  const deadline = setTimeout(() => abort.abort(new Error('terminal admitted run deadline exceeded')), a.timeoutSeconds * 1000);
  const alive = () => { if (abort.signal.aborted) throw abort.signal.reason; };
  const wait = async (fn, seconds = 180) => { const end = io.now() + seconds * 1000; while (io.now() < end) { alive(); const value = await fn(); if (value) return value; await io.sleep(500); } throw new Error('owned session readiness timeout'); };
  const launch = async (name, env) => {
    alive(); const logPath = path.join(a.outputRoot, `${identity.nonce}-${name}.log`), log = await io.open(logPath, 'wx'); handles.push(log);
    const child = io.spawn(a.binary, [], { cwd: ROOT, env, windowsHide: true, stdio: [env.SCENEWORKS_WORKER_CHILD ? 'pipe' : 'ignore', log.fd, log.fd] });
    child.logPath = logPath; child.done = new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => resolve({ code, signal })); }); child.done.catch(() => abort.abort(new Error('owned child launch failed'))); children.push(child); return child;
  };
  const stop = async child => {
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    child.stdin?.end(); let timer;
    const exited = await Promise.race([child.done.then(() => true), new Promise(resolve => { timer = setTimeout(() => resolve(false), 20_000); })]); clearTimeout(timer);
    if (!exited) { child.kill('SIGKILL'); await child.done; }
  };
  const guard = () => io.guard(a, children.filter(c => c.exitCode === null && c.signalCode === null).map(c => c.pid));
  const close = async () => {
    closing = true; clearTimeout(deadline); clearInterval(monitor); await monitorTask; endpoint?.close(); let failed = false;
    for (const c of [...children].reverse()) { try { await stop(c); } catch { failed = true; } cleanup.push({ pid: c.pid, exited: c.exitCode !== null || c.signalCode !== null, exitCode: c.exitCode, signal: c.signalCode }); }
    for (const h of handles) try { await h.close(); } catch { failed = true; }
    if (secrets.length) for (const c of children) try { await io.writeFile(c.logPath, redactSensitiveText(await io.readFile(c.logPath, 'utf8'), secrets)); } catch { failed = true; }
    credential = undefined; if (faultServer) await new Promise(resolve => faultServer.close(resolve));
    if (failed || cleanup.some(c => !c.exited)) throw new Error('owned child/log cleanup failed; do not release admission until reconciled');
  };
  try {
    if (!dependencies.fixtureOnly) {
      await validateAdmission(a, a.backend); const budget = validateApprovedPlan(a.approvedPlan); assert.equal(budget.approvedPlanSha256, a.approvedPlanSha256); assert.equal(a.ledgerId, a.approvedPlan.ledgerId); admittedHost(a.approvedPlan, a);
      initialLedgerReceipt = await auditLedger({ approval: a.approvedPlan, backend: a.backend, ledgerRoot: a.ledgerRoot, inputSpecAddendum: a.inputSpecAddendum, host: a.runner });
      const { verifyTerminalBuild } = await import('./flux3-image-klein-terminal-build.mjs'); await verifyTerminalBuild(a);
    }
    if (a.replayPrefixAuthorizationSha256 !== undefined) {
      assert(a.backend === 'candle' && digestPattern.test(a.replayPrefixAuthorizationSha256) && a.replayPrefixAuthorizationSha256 === a.approvedPlan?.candleSourceTransitionSha256, 'explicit current approval must bind the retained replay prefix');
      const receipt = initialLedgerReceipt ?? await auditLedger({ approval: a.approvedPlan, backend: 'candle', ledgerRoot: a.ledgerRoot });
      const carried = candleReplayPrefix(receipt, a.approvedPlan);
      assert.equal(carried.authorizationSha256, a.replayPrefixAuthorizationSha256);
      assert.equal(receipt.state.sequence, 2); assert.equal(receipt.state.generationCount, 1);
      assert.deepEqual(receipt.state.consumedIds, [carried.consumedInvocationId]);
      assert(!receipt.state.consumedIds.includes(carried.nextInvocationId), 'the retained replay continuation cannot repeat its second generation');
      const { validateReplayPrefixBaseline } = await import('./flux3-image-klein-terminal-driver.mjs');
      replayPrefix = await validateReplayPrefixBaseline({ previousApproval: carried.previousApproval, previousReceipt: carried.previousReceipt, authorization: carried.authorization, ownedRoot: path.resolve(a.ledgerRoot, '../..') });
      replayPrefix.authorizationSha256 = carried.authorizationSha256;
    }
    assert.equal(await io.hash(a.binary), a.binarySha256, 'prebuilt binary mismatch');
    if (!dependencies.skipPinReadForFixture) assertPins(await Promise.all(['Cargo.toml', 'crates/sceneworks-worker/Cargo.toml', 'crates/sceneworks-memory-adapter/Cargo.toml'].map(f => io.readFile(path.join(ROOT, f), 'utf8'))), await io.readFile(path.join(ROOT, 'Cargo.lock'), 'utf8'), a.inferenceMainSha);
    const deviceAtLaunch = await guard();
    if (a.approvedProvisioning?.length) {
      assert(a.actualRun === true && !a.faultProfile, 'provisioning and injected faults are mutually exclusive');
      try {
        if (a.approvedProvisioning.some(p => p.repository !== 'black-forest-labs/flux-3-image-klein-pup')) { credential = await io.readCredential({ HF_HOME: a.hfHome, HOME: io.env.HOME }); secrets.push(credential); }
        for (const p of a.approvedProvisioning) authorizations.push({ ...await io.authorize(p.repository, p.resolvedRevision, credential, { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30_000)]) }), ...execution });
      } catch (error) { throw new ProvisioningBlocked(redactSensitiveText(error.message, secrets)); }
    }
    await io.mkdir(identity.state); await io.mkdir(path.join(identity.state, 'data/projects'), { recursive: true }); await io.mkdir(path.join(identity.state, 'credentials')); await io.cp(path.join(ROOT, 'config'), path.join(identity.state, 'config'), { recursive: true });
    if (!dependencies.fixtureOnly || a.artifactCells) {
      assert(a.artifactCells?.length && a.artifactInputs && a.artifactTargets, 'concrete launcher requires exact approved artifact mapping inputs before any service launch');
      assert.equal(planHash(a.artifactInputs), boundInputSpecSha256(a.approvedPlan, a.backend, a.inputSpecAddendum, a.runner));
      const { stageTerminalArtifacts } = await import('./flux3-image-klein-terminal-artifacts.mjs'); artifactMapping = await stageTerminalArtifacts({ admission: a, identity, inputSpec: a.artifactInputs, cells: a.artifactCells, registry: a.artifactTargets }, { diskFree: io.diskFree });
    }
    if (a.faultProfile) {
      assert(a.installationFaultAdmission === true && ['missing-install', 'hf403'].includes(a.faultProfile), 'explicit installation fault admission required');
      // Deliberately empty task-owned hub; never corrupt or remove an existing cache.
      const emptyHome = path.join(identity.state, 'fault-hf'); await io.mkdir(path.join(emptyHome, 'hub'), { recursive: true }); a = { ...a, hfHome: emptyHome };
      {
        const status = a.faultProfile === 'hf403' ? 403 : 404;
        faultServer = io.createServer((request, response) => { faultRequests.push({ method: request.method, path: request.url }); response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify({ error: status === 403 ? 'S10 injected gated-repository denial: authorize the selected runner account' : 'S10 injected missing pinned package: repository files not found (404)' })); });
        await new Promise((resolve, reject) => { faultServer.once('error', reject); faultServer.listen(0, '127.0.0.1', resolve); }); faultUrl = `http://127.0.0.1:${faultServer.address().port}`;
      }
    }
    const api = await launch('api', childEnv());
    const url = await wait(async () => boundApiUrl(await io.readFile(api.logPath, 'utf8')));
    endpoint = io.controller({ url, state: identity.state, api, controller: abort }); await wait(() => endpoint.verify());
    const consume = async (invocationId, kind, body) => {
      const receipt = await io.reserve({ approval: a.approvedPlan, ledgerRoot: a.ledgerRoot, ledgerId: a.ledgerId, approvedPlanSha256: a.approvedPlanSha256, invocationId, backend: a.backend, caseId: currentCase.caseId, kind, requestSha256: planHash(body), inputSpecAddendum: a.inputSpecAddendum, host: a.runner }); permits.push(receipt); return receipt;
    };
    const beforeMutation = async (method, route, body) => {
      if (method !== 'POST') return;
      const image = route === '/api/v1/image/jobs', retry = /^\/api\/v1\/jobs\/[^/]+\/retry$/.test(route), mutation = retry && body?.payloadChanges;
      if (image || retry || /\/training\/jobs$/.test(route)) assert(currentCase, 'begin an explicitly approved case before any job start');
      if (mutation || (image && (currentCase.kind === 'reject' || currentCase.status === 'BLOCKED'))) { assert(workerPaused && (worker.exitCode !== null || worker.signalCode !== null), 'negative probes require only OUR owned GPU worker stopped; never risk an uncounted generation'); return; }
      if (image) {
        await guard();
        assert(!workerPaused && worker.exitCode === null); assert.equal(body.count, 1, 'batch generation requires individually named permits'); assert(!body.hiresFix?.enabled && !body.advanced?.hiresFix?.enabled, 'hidden extra generation passes are forbidden');
        const index = invocationIndexes.get(currentCase.caseId) ?? 0, invocation = selectProductInvocation(a.approvedPlan, currentCase, index, body, replayPrefix);
        assert(invocation, 'no approved single-use generation permit; no auto retry/repeat/spare allocation'); await consume(invocation.invocationId, invocation.kind, body); invocationIndexes.set(currentCase.caseId, index + 1);
      } else if (/\/training\/jobs$/.test(route) || retry) {
        await guard();
        const training = a.approvedPlan.plan.trainingLifecycles.find(t => t.backend === a.backend && t.caseId === currentCase.caseId); assert(training, 'only the TWO explicitly listed bounded training lifecycles are allowed');
        if (retry) { assert.equal(body.resume, true); await consume(training.resumeId, 'training_resume', body); }
        else { assert.equal(body.dryRun, false); assert.equal(body.targetId, currentCase.options.trainingMode === 'edit' ? 'flux3_klein_edit_lora' : 'flux3_klein_lora'); assert.equal(body.config.batchSize, 1); assert.equal(body.config.gradientAccumulation, 1); for (const key of ['steps', 'rank', 'resolution', 'saveEvery']) assert.equal(body.config[key], training[key]); assert.equal(body.config.advanced.sampleEvery, 0, 'training previews are unallocated generations'); await consume(training.startId, 'training_start', body); }
      } else if (route === '/api/v1/jobs') assert(!['image_generate', 'training'].includes(body?.type), 'raw job queues cannot bypass terminal permits');
    };
    const ordinaryCall = endpoint.call.bind(endpoint); endpoint.call = async (method, route, body) => { await beforeMutation(method, route, body); return ordinaryCall(method, route, body); };
    if (endpoint.exchange) { const ordinaryExchange = endpoint.exchange.bind(endpoint); endpoint.exchange = async (method, route, body, options) => { await beforeMutation(method, route, body); return ordinaryExchange(method, route, body, options); }; }
    const startWorker = async () => {
      await guard(); worker = await endpoint.startWorker(() => launch(`worker-${++generation}`, childEnv(url)));
      const record = await wait(async () => { const w = (await endpoint.call('GET', '/api/v1/workers')).find(w => w.id === identity.workerId); if (!w || w.status !== 'idle') return false; assert(w.capabilities?.includes('gpu')); assert.equal(w.gpuId, a.backend === 'candle' ? admittedCandleDevice(a).physicalGpuId : 'mlx'); if (a.backend === 'candle') assert(w.capabilities.includes('candle')); else assert(!w.capabilities.includes('candle')); return w; });
      return { pid: worker.pid, record };
    };
    const started = await startWorker();
    monitor = setInterval(() => { if (monitoring || restarting || closing) return; monitoring = true; monitorTask = (async () => { try { alive(); assert(api.exitCode === null && api.signalCode === null && worker.exitCode === null && worker.signalCode === null, 'owned service exited'); await guard(); } catch (e) { abort.abort(e); } finally { monitoring = false; } })(); }, 2000);
    return { endpoint, identity, execution, artifactMapping, deviceAtLaunch, replayPrefix, worker: started.record, workerPid: started.pid, wait, alive, faultProfile: a.faultProfile, faultRequests, authorizations, cleanup, permits, redact: text => redactSensitiveText(text, secrets),
      async trainingWitness(plan) {
        assert(workerPaused, 'CPU witness must run with only our GPU worker paused');
        if (!dependencies.fixtureOnly) { const { verifyTerminalBuild } = await import('./flux3-image-klein-terminal-build.mjs'); await verifyTerminalBuild(a); }
        const bytes = Buffer.from(JSON.stringify(plan)), file = path.join(identity.state, `native-plan-${randomBytes(16).toString('hex')}.json`); await io.writeFile(file, bytes, { flag: 'wx' });
        const env = offlineChildEnv(io.env, a, identity); Object.assign(env, { CUDA_VISIBLE_DEVICES: '-1', SCENEWORKS_GPU_ID: 'cpu', SCENEWORKS_BACKEND_CANDLE_ENABLED: 'false', SCENEWORKS_BACKEND_MLX_ENABLED: 'false', SCENEWORKS_RUN_UTILITY_INPROCESS: '0', SCENEWORKS_FLUX3_WITNESS_ROOT: identity.state }); delete env.SCENEWORKS_WORKER_ONLY; delete env.SCENEWORKS_GPU_CHECK;
        const result = await io.execWitness(a.binary, ['--flux3-training-config-witness', file], { cwd: ROOT, env, timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
        const value = JSON.parse(result.stdout); assert(value.status === 'ok' && value.witness?.formatVersion === 1 && value.witness.inputSha256 === createHash('sha256').update(bytes).digest('hex'), 'native CPU witness does not bind exact persisted plan bytes');
        return { ...value.witness, sourceSha: execution.sourceSha, inferenceMainSha: execution.inferenceMainSha, binarySha256: execution.binarySha256, buildReceiptSha256: execution.buildReceiptSha256 };
      },
      beginCase(cell) { alive(); const approved = a.approvedPlan?.plan.cases.find(c => c.caseId === cell.caseId && c.backend === a.backend); assert(dependencies.fixtureOnly || (approved && planHash(approved) === planHash(cell)), 'case/options not in immutable user-approved plan'); currentCase = cell; },
      failClosed(error) { abort.abort(error); },
      async pauseWorker() { restarting = true; await monitorTask; alive(); await stop(worker); workerPaused = true; return { pid: worker.pid }; },
      async restartWorker() { restarting = true; try { await monitorTask; alive(); const previous = worker.pid; await stop(worker); identity.workerId = `flux3-app-${identity.nonce}-resume-${generation}`; const next = await startWorker(); assert.notEqual(next.pid, previous, 'resume must use a fresh process'); this.workerPid = next.pid; this.worker = next.record; workerPaused = false; return next; } finally { restarting = false; } },
      async assertDeviceProof() { const proof = await guard(); if (a.backend === 'candle') assert(proof.ownComputePids.includes(worker.pid), 'actual worker PID/approved GPU UUID proof unavailable'); return proof; }, close };
  } catch (error) { await close(); throw error; }
}
