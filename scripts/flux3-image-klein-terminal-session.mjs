// Concrete owned API/worker launcher. Offline by default; no build or dispatch.
// Only explicitly approved provisioning uses the shared legacy credential boundary.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCb } from 'node:child_process';
import { createServer } from 'node:http';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { cp, mkdir, open, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { userInfo } from 'node:os';
import { ROOT, shaPattern, digestPattern } from './flux3-image-klein-terminal-plan.mjs';
import { attemptIdentity, bindOwnedEnv, boundApiUrl, ownedApiController, runnerHfCredential, authorizeFlux3Repository, redactSensitiveText } from './flux3-image-klein-app-lifecycle.mjs';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { validateApprovedPlan, planHash, auditLedger, reserveInvocation } from './flux3-image-klein-terminal-budget.mjs';
const execFile = promisify(execFileCb);
export async function fileHash(file) { const h = createHash('sha256'); for await (const chunk of createReadStream(file)) h.update(chunk); return h.digest('hex'); }
export async function contained(file, root) {
  const resolved = await realpath(file), relative = path.relative(await realpath(root), resolved);
  assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'file escaped admitted root');
  assert((await stat(resolved)).isFile(), 'regular file required'); return resolved;
}
export async function validateAdmission(a, backend, { git = async (cwd, ...args) => (await execFile('git', args, { cwd, timeout: 30_000 })).stdout.trim(), root = ROOT, account = () => userInfo().username } = {}) {
  assert.equal(a.actualRun, true, 'execution requires explicit actualRun:true; planning never launches');
  assert.equal(a.backend, backend); assert(a.admissionId && a.approvedBy && a.phases?.length, 'parent resource admission identity/phases required');
  assert.equal(a.runnerAccount, account(), 'admission must name the actual selected runner account (no credentials in receipt)');
  assert(shaPattern.test(a.sourceSha ?? '') && shaPattern.test(a.inferenceMainSha ?? ''), 'exact source and inference MAIN SHAs required');
  assert.equal(await git(root, 'rev-parse', 'HEAD'), a.sourceSha);
  assert.equal(await git(root, 'status', '--porcelain'), '', 'clean committed source required');
  await git(root, 'merge-base', '--is-ancestor', a.sourceSha, 'refs/remotes/origin/main');
  assert(a.inferenceRepository && path.isAbsolute(a.inferenceRepository), 'local inference checkout required for MAIN ancestry proof');
  assert.equal(await git(a.inferenceRepository, 'rev-parse', 'refs/remotes/origin/main'), a.inferenceMainTipSha, 'record the locally fetched inference MAIN tip');
  await git(a.inferenceRepository, 'merge-base', '--is-ancestor', a.inferenceMainSha, 'refs/remotes/origin/main');
  assert(shaPattern.test(a.inferenceMainTipSha ?? ''));
  assert(path.isAbsolute(a.binary ?? '') && digestPattern.test(a.binarySha256 ?? '') && a.binarySourceSha === a.sourceSha, 'prebuilt exact-source binary/hash required; no automatic build');
  assert(path.isAbsolute(a.outputRoot ?? '') && path.isAbsolute(a.hfHome ?? ''), 'stable absolute output/cache roots required');
  const relative = path.relative(root, a.outputRoot); assert(relative.startsWith('..') || path.isAbsolute(relative), 'retained output must be outside checkout');
  assert(Number.isInteger(a.timeoutSeconds) && a.timeoutSeconds >= 60 && a.timeoutSeconds <= 10800, 'bounded run timeout required');
  if (backend === 'mlx') assert.equal(a.runner, 'nax-macos-2', 'only secondary nax-macos-2 admitted');
  else assert(a.runner && a.physicalGpuId === '1' && a.cudaVisibleDevices === '1' && a.logicalCudaOrdinal === 0 && /^GPU-[a-f0-9-]+$/i.test(a.gpuUuid ?? ''), 'physical GPU1/masked CUDA0 admission required');
}
export function safeEnvironment(env) {
  const base = {};
  // Filter keys BEFORE reading values: installed-only planning/tests never inspect ambient secrets.
  for (const k of Object.keys(env)) if (!k.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(k) && !['HF_ENDPOINT', 'HF_TOKEN_PATH', 'HF_XET_CACHE', 'TRANSFORMERS_CACHE', 'CARGO_TARGET_DIR'].includes(k)) base[k] = env[k];
  return base;
}
export function offlineChildEnv(env, admission, identity, url) {
  const base = safeEnvironment(env);
  Object.assign(base, { HF_HOME: admission.hfHome, HF_HUB_CACHE: path.join(admission.hfHome, 'hub'), HUGGINGFACE_HUB_CACHE: path.join(admission.hfHome, 'hub'), HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', SCENEWORKS_HUGGINGFACE_BASE_URL: 'http://127.0.0.1:9', SCENEWORKS_RESOLVED_CACHE_ENABLED: 'false', SCENEWORKS_POLL_SECONDS: '1', SCENEWORKS_HEARTBEAT_SECONDS: '2', SCENEWORKS_BACKEND_CANDLE_ENABLED: String(admission.backend === 'candle'), SCENEWORKS_BACKEND_MLX_ENABLED: String(admission.backend === 'mlx'), SCENEWORKS_CANDLE_REQUIRED: admission.backend === 'candle' ? '1' : '0', CUDA_VISIBLE_DEVICES: admission.backend === 'candle' ? '1' : '-1', CUDA_DEVICE_ORDER: 'PCI_BUS_ID', SCENEWORKS_GPU_ID: admission.backend === 'candle' ? '1' : 'mlx' });
  return { ...bindOwnedEnv(base, identity, url), ...(url ? { SCENEWORKS_WORKER_ONLY: '1', SCENEWORKS_WORKER_CHILD: '1' } : {}) };
}
// Driver API UUID query runs in a separate short process with the SAME visibility mask.
const cudaUuidProbe = `Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class TerminalCudaIdentity { [DllImport("nvcuda.dll")] public static extern int cuInit(uint flags); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetCount(out int count); [DllImport("nvcuda.dll")] public static extern int cuDeviceGet(out int device, int ordinal); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetUuid_v2([Out] byte[] uuid, int device); }'; if([TerminalCudaIdentity]::cuInit(0) -ne 0){throw "CUDA init"}; $n=0; if([TerminalCudaIdentity]::cuDeviceGetCount([ref]$n) -ne 0 -or $n -ne 1){throw "CUDA mask"}; $d=0; if([TerminalCudaIdentity]::cuDeviceGet([ref]$d,0) -ne 0){throw "CUDA ordinal"}; $u=New-Object byte[] 16; if([TerminalCudaIdentity]::cuDeviceGetUuid_v2($u,$d) -ne 0){throw "CUDA UUID"}; $h=([BitConverter]::ToString($u)).Replace("-","").ToLowerInvariant(); Write-Output ("GPU-"+$h.Substring(0,8)+"-"+$h.Substring(8,4)+"-"+$h.Substring(12,4)+"-"+$h.Substring(16,4)+"-"+$h.Substring(20,12))`;
export async function guardHardware(a, ownPids = [], { exec = execFile, platform = process.platform, arch = process.arch, env = process.env } = {}) {
  const checked = (command, args, options = {}) => exec(command, args, { ...options, env: { ...safeEnvironment(env), ...options.env } });
  if (a.backend === 'mlx') {
    assert(platform === 'darwin' && arch === 'arm64', 'physical Apple Silicon Mac required');
    assert.equal(a.runner, 'nax-macos-2'); assert.equal(a.runnerTarget, 'secondary'); assert.equal(a.runnerLabel, 'rw-mage'); assert.equal(env.RUNNER_NAME, 'nax-macos-2', 'secondary runner identity required (rw-mage is its label, not its runner name)');
    const host = (await checked('hostname', [], { timeout: 5000 })).stdout.trim(); assert.equal(host.split('.')[0], a.hostName, 'admitted secondary physical hostname mismatch');
    assert(a.hostName, 'explicit admitted secondary hostname required'); return { runner: a.runner, runnerTarget: a.runnerTarget, runnerLabel: a.runnerLabel, host, backend: 'mlx' };
  }
  assert.equal(platform, 'win32');
  const runners = (await checked('tasklist.exe', ['/FI', 'IMAGENAME eq Runner.Worker.exe', '/FO', 'CSV', '/NH'], { timeout: 5000 })).stdout;
  assert(!/^"Runner\.Worker\.exe"/im.test(runners), 'Runner.Worker active; stop and return admission to coordinator');
  const smi = async args => (await checked('nvidia-smi', args, { timeout: 5000 })).stdout.trim();
  const [index, uuid] = (await smi(['-i', '1', '--query-gpu=index,uuid', '--format=csv,noheader'])).split(',').map(s => s.trim());
  assert.equal(index, '1'); assert.equal(uuid.toLowerCase(), a.gpuUuid.toLowerCase());
  const masked = (await checked('powershell.exe', ['-NoProfile', '-Command', cudaUuidProbe], { timeout: 30_000, env: { CUDA_VISIBLE_DEVICES: '1', CUDA_DEVICE_ORDER: 'PCI_BUS_ID' } })).stdout.trim();
  assert.equal(masked.toLowerCase(), uuid.toLowerCase(), 'masked CUDA0 must be physical GPU1');
  const xml = await smi(['-i', '1', '-q', '-x']); assert(xml.includes('<processes>') && xml.includes(uuid), 'GPU1 process identity unavailable');
  for (const [, block] of xml.matchAll(/<process_info>([\s\S]*?)<\/process_info>/g)) {
    const pid = Number(/<pid>([^<]+)<\/pid>/.exec(block)?.[1]), type = /<type>([^<]+)<\/type>/.exec(block)?.[1]?.trim();
    assert(Number.isInteger(pid) && ['C', 'G', 'C+G'].includes(type));
    assert(type !== 'C' || ownPids.includes(pid), 'foreign pure-compute process on GPU1');
  }
  const apps = (await smi(['--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])).split(/\r?\n/).filter(Boolean).map(line => line.split(',').map(x => x.trim()));
  for (const [pid, device] of apps) if (ownPids.includes(Number(pid))) assert.equal(device.toLowerCase(), uuid.toLowerCase(), 'owned process touched reserved GPU0');
  return { physicalGpuId: '1', gpuUuid: uuid, logicalCudaOrdinal: 0, ownComputePids: apps.filter(([pid]) => ownPids.includes(Number(pid))).map(([pid]) => Number(pid)) };
}
export class ProvisioningBlocked extends Error {}
export async function launchTerminalSession(a, dependencies = {}) {
  // Dependency overrides are for weights-free tests; production CLI never accepts a launcher module.
  const io = { mkdir, cp, open, readFile, writeFile, createServer, readCredential: runnerHfCredential, authorize: authorizeFlux3Repository, reserve: reserveInvocation, hash: fileHash, spawn, guard: guardHardware, controller: ownedApiController, sleep: ms => new Promise(r => setTimeout(r, ms)), now: Date.now, env: process.env, ...dependencies };
  const identity = attemptIdentity(a.outputRoot), children = [], handles = [], abort = new AbortController();
  const execution = { sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, runner: a.runner, runnerAccount: a.runnerAccount };
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
      await validateAdmission(a, a.backend); const budget = validateApprovedPlan(a.approvedPlan); assert.equal(budget.approvedPlanSha256, a.approvedPlanSha256); assert.equal(a.ledgerId, a.approvedPlan.ledgerId);
      await auditLedger({ approval: a.approvedPlan, backend: a.backend, ledgerRoot: a.ledgerRoot });
    }
    assert.equal(await io.hash(a.binary), a.binarySha256, 'prebuilt binary mismatch');
    if (!dependencies.skipPinReadForFixture) assertPins(await Promise.all(['Cargo.toml', 'crates/sceneworks-worker/Cargo.toml', 'crates/sceneworks-memory-adapter/Cargo.toml'].map(f => io.readFile(path.join(ROOT, f), 'utf8'))), await io.readFile(path.join(ROOT, 'Cargo.lock'), 'utf8'), a.inferenceMainSha);
    await guard();
    if (a.approvedProvisioning?.length) {
      assert(a.actualRun === true && !a.faultProfile, 'provisioning and injected faults are mutually exclusive');
      try {
        if (a.approvedProvisioning.some(p => p.repository !== 'black-forest-labs/flux-3-image-klein-pup')) { credential = await io.readCredential({ HF_HOME: a.hfHome, HOME: io.env.HOME }); secrets.push(credential); }
        for (const p of a.approvedProvisioning) authorizations.push({ ...await io.authorize(p.repository, p.resolvedRevision, credential, { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30_000)]) }), ...execution });
      } catch (error) { throw new ProvisioningBlocked(redactSensitiveText(error.message, secrets)); }
    }
    await io.mkdir(identity.state); await io.mkdir(path.join(identity.state, 'data/projects'), { recursive: true }); await io.mkdir(path.join(identity.state, 'credentials')); await io.cp(path.join(ROOT, 'config'), path.join(identity.state, 'config'), { recursive: true });
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
      const receipt = await io.reserve({ approval: a.approvedPlan, ledgerRoot: a.ledgerRoot, ledgerId: a.ledgerId, approvedPlanSha256: a.approvedPlanSha256, invocationId, backend: a.backend, caseId: currentCase.caseId, kind, requestSha256: planHash(body) }); permits.push(receipt); return receipt;
    };
    const beforeMutation = async (method, route, body) => {
      if (method !== 'POST') return;
      const image = route === '/api/v1/image/jobs', retry = /^\/api\/v1\/jobs\/[^/]+\/retry$/.test(route), mutation = retry && body?.payloadChanges;
      if (image || retry || /\/training\/jobs$/.test(route)) assert(currentCase, 'begin an explicitly approved case before any job start');
      if (mutation || (image && (currentCase.kind === 'reject' || currentCase.status === 'BLOCKED'))) { assert(workerPaused && (worker.exitCode !== null || worker.signalCode !== null), 'negative probes require only OUR owned GPU worker stopped; never risk an uncounted generation'); return; }
      if (image) {
        assert(!workerPaused && worker.exitCode === null); assert.equal(body.count, 1, 'batch generation requires individually named permits'); assert(!body.hiresFix?.enabled && !body.advanced?.hiresFix?.enabled, 'hidden extra generation passes are forbidden');
        const index = invocationIndexes.get(currentCase.caseId) ?? 0, invocation = a.approvedPlan.plan.generationInvocations.filter(i => i.backend === a.backend && i.caseId === currentCase.caseId && i.kind === 'product_generation')[index];
        assert(invocation, 'no approved single-use generation permit; no auto retry/repeat/spare allocation'); await consume(invocation.invocationId, invocation.kind, body); invocationIndexes.set(currentCase.caseId, index + 1);
      } else if (/\/training\/jobs$/.test(route) || retry) {
        const training = a.approvedPlan.plan.trainingLifecycles.find(t => t.backend === a.backend && t.caseId === currentCase.caseId); assert(training, 'only the TWO explicitly listed bounded training lifecycles are allowed');
        if (retry) { assert.equal(body.resume, true); await consume(training.resumeId, 'training_resume', body); }
        else { assert.equal(body.dryRun, false); assert.equal(body.targetId, currentCase.options.trainingMode === 'edit' ? 'flux3_klein_edit_lora' : 'flux3_klein_lora'); assert.equal(body.config.batchSize, 1); assert.equal(body.config.gradientAccumulation, 1); for (const key of ['steps', 'rank', 'resolution', 'saveEvery']) assert.equal(body.config[key], training[key]); assert.equal(body.config.advanced.sampleEvery, 0, 'training previews are unallocated generations'); await consume(training.startId, 'training_start', body); }
      } else if (route === '/api/v1/jobs') assert(!['image_generate', 'training'].includes(body?.type), 'raw job queues cannot bypass terminal permits');
    };
    const ordinaryCall = endpoint.call.bind(endpoint); endpoint.call = async (method, route, body) => { await beforeMutation(method, route, body); return ordinaryCall(method, route, body); };
    if (endpoint.exchange) { const ordinaryExchange = endpoint.exchange.bind(endpoint); endpoint.exchange = async (method, route, body, options) => { await beforeMutation(method, route, body); return ordinaryExchange(method, route, body, options); }; }
    const startWorker = async () => {
      await guard(); worker = await endpoint.startWorker(() => launch(`worker-${++generation}`, childEnv(url)));
      const record = await wait(async () => { const w = (await endpoint.call('GET', '/api/v1/workers')).find(w => w.id === identity.workerId); if (!w || w.status !== 'idle') return false; assert(w.capabilities?.includes('gpu')); assert.equal(w.gpuId, a.backend === 'candle' ? '1' : 'mlx'); if (a.backend === 'candle') assert(w.capabilities.includes('candle')); else assert(!w.capabilities.includes('candle')); return w; });
      return { pid: worker.pid, record };
    };
    const started = await startWorker();
    monitor = setInterval(() => { if (monitoring || restarting || closing) return; monitoring = true; monitorTask = (async () => { try { alive(); assert(api.exitCode === null && api.signalCode === null && worker.exitCode === null && worker.signalCode === null, 'owned service exited'); await guard(); } catch (e) { abort.abort(e); } finally { monitoring = false; } })(); }, 2000);
    return { endpoint, identity, execution, worker: started.record, workerPid: started.pid, wait, alive, faultProfile: a.faultProfile, faultRequests, authorizations, cleanup, permits, redact: text => redactSensitiveText(text, secrets),
      beginCase(cell) { alive(); const approved = a.approvedPlan?.plan.cases.find(c => c.caseId === cell.caseId && c.backend === a.backend); assert(dependencies.fixtureOnly || (approved && planHash(approved) === planHash(cell)), 'case/options not in immutable user-approved plan'); currentCase = cell; },
      failClosed(error) { abort.abort(error); },
      async pauseWorker() { restarting = true; await monitorTask; alive(); await stop(worker); workerPaused = true; return { pid: worker.pid }; },
      async restartWorker() { restarting = true; try { await monitorTask; alive(); const previous = worker.pid; await stop(worker); identity.workerId = `flux3-app-${identity.nonce}-resume-${generation}`; const next = await startWorker(); assert.notEqual(next.pid, previous, 'resume must use a fresh process'); this.workerPid = next.pid; this.worker = next.record; workerPaused = false; return next; } finally { restarting = false; } },
      async assertDeviceProof() { const proof = await guard(); if (a.backend === 'candle') assert(proof.ownComputePids.includes(worker.pid), 'actual worker PID/GPU1 UUID proof unavailable'); return proof; }, close };
  } catch (error) { await close(); throw error; }
}
