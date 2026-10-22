// sc-26243: four queued slots, ONE executor, THREE CPU-only holders on one Windows box.
// Not a global GPU lock. No runner administration, uploads, retries, or ledger reset.
// The API/heartbeat supervisors stay independent of the credential-free blocking model child.
import assert from 'node:assert/strict';
import { createHash, generateKeyPairSync, randomBytes, sign, verify } from 'node:crypto';
import { spawn, execFile as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, writeFile, mkdir, rename, open, link, unlink } from 'node:fs/promises';
import path from 'node:path';
import { hostname } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
const exec = promisify(execCb);
export const QUEUE_ROOT = 'D:/tmp/sc26243-run/queue';
export const CONTROL_FILE = 'D:/tmp/sc26243-run/queue-control.json';
export const SOURCE_ROOT = 'D:/repos/SceneWorks/.claude/worktrees/agent-aafe2575fa619862f';
export const EXECUTOR = 'cuda-windows';
// Repo-scoped runners 3/4 are absent from the organization endpoint. Workflow CVD=1 overrides
// runner3's default CVD=0, so ALL FOUR slots are required. Holders never open either GPU.
export const SLOTS = Object.freeze({
  'cuda-windows': Object.freeze({ id: 2313, scope: 'organization', worker: 'D:/actions-runner/bin/Runner.Worker.exe' }),
  'cuda-windows-2': Object.freeze({ id: 2619, scope: 'organization', worker: 'D:/actions-runner-2/bin/Runner.Worker.exe' }),
  'cuda-windows-3': Object.freeze({ id: 25, scope: 'repository', worker: 'D:/actions-runner-3/bin/Runner.Worker.exe' }),
  'cuda-windows-4': Object.freeze({ id: 26, scope: 'repository', worker: 'D:/actions-runner-4/bin/Runner.Worker.exe' }),
});
export const HEARTBEAT_MS = 2000, STALE_MS = 20000, API_FRESH_MS = 90000;
export const WAIT_MS = 60 * 60 * 1000, CAMPAIGN_MS = 8 * 60 * 60 * 1000;
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/, number = /^[1-9][0-9]*$/;
export const digest = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value)).digest('hex');
const normalized = p => String(p).replaceAll('\\', '/').toLowerCase();
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const json = async file => JSON.parse(await readFile(file, 'utf8'));
const log = value => console.log(JSON.stringify(value));
export function queueInputs(env) {
  assert(revision.test(env.FLUX3_QUEUE_SOURCE ?? '') && revision.test(env.FLUX3_QUEUE_INFERENCE ?? ''), 'queue requires exact source/inference SHAs');
  assert(sha.test(env.FLUX3_QUEUE_CONTROL_SHA256 ?? ''), 'queue control sentinel none/empty refused: supply exact sha256');
  assert.equal(env.GITHUB_REPOSITORY, 'SceneWorks/SceneWorks');
  assert.equal(env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual queue only');
  assert.equal(env.GITHUB_SHA, env.FLUX3_QUEUE_SOURCE, 'event head must be the exact approved source');
  assert(number.test(env.GITHUB_RUN_ID ?? '') && number.test(env.GITHUB_RUN_ATTEMPT ?? ''), 'exact GitHub run/attempt required');
  return { repository: env.GITHUB_REPOSITORY, runId: Number(env.GITHUB_RUN_ID), attempt: Number(env.GITHUB_RUN_ATTEMPT), sourceSha: env.FLUX3_QUEUE_SOURCE, inferenceMainSha: env.FLUX3_QUEUE_INFERENCE, controlSha256: env.FLUX3_QUEUE_CONTROL_SHA256 };
}
export function electedRole(runner) { assert(Object.hasOwn(SLOTS, runner ?? ''), 'unrecognized runner: no queue reservation or device probe'); return runner === EXECUTOR ? 'executor' : 'holder'; }
export function signed(value, privateKey) { return { value, signature: sign(null, Buffer.from(JSON.stringify(value)), privateKey).toString('base64') }; }
export function authenticated(envelope, publicKey) {
  assert(envelope?.value && typeof envelope.signature === 'string' && verify(null, Buffer.from(JSON.stringify(envelope.value)), publicKey, Buffer.from(envelope.signature, 'base64')), 'queue heartbeat signature mismatch');
  return envelope.value;
}
export function validateApi(run, jobs, identity, { complete = true } = {}) {
  assert.equal(run.repository?.full_name, identity.repository); assert.equal(run.id, identity.runId);
  assert.equal(run.run_attempt, identity.attempt); assert.equal(run.head_sha, identity.sourceSha);
  assert.equal(run.event, 'workflow_dispatch'); assert.equal(run.status, 'in_progress');
  assert.equal(run.path?.split('@')[0], '.github/workflows/memory-catalog-campaign.yml');
  const rows = jobs.filter(j => /^flux3-terminal-queue \/ reserve \([1-4]\)$/.test(j.name));
  assert(rows.length <= 4, 'unexpected extra queue jobs');
  assert(rows.every(j => ['queued','waiting','pending','requested','in_progress'].includes(j.status)), 'queue peer ended before release; all reservations must stop');
  assert.equal(new Set(rows.map(j => j.id)).size, rows.length, 'duplicate job ID');
  assert.equal(new Set(rows.map(j => j.name)).size, rows.length, 'duplicate matrix slot');
  const found = {};
  for (const job of rows.filter(j => j.status === 'in_progress')) {
    assert.equal(job.run_id, identity.runId); assert.equal(job.head_sha, identity.sourceSha);
    assert(Object.hasOwn(SLOTS, job.runner_name ?? ''), 'unrecognized queue runner');
    assert.equal(job.runner_id, SLOTS[job.runner_name].id, 'queue runner ID mismatch');
    assert(!Object.hasOwn(found, job.runner_name), 'duplicate queue runner placement');
    assert(Number.isSafeInteger(job.id) && job.id > 0); found[job.runner_name] = job.id;
  }
  if (complete) assert.deepEqual(Object.keys(found).sort(), Object.keys(SLOTS).sort(), 'all four distinct runner jobs must be running before the barrier');
  return found;
}
export function ancestors(table, pid) { const seen = new Set(); while (pid > 0 && !seen.has(pid)) { seen.add(pid); pid = table.find(p => p.pid === pid)?.ppid ?? 0; } return seen; }
export function processIdentity(table, pid) {
  const p = table.find(p => p.pid === pid); assert(p?.startedAt && p.executable && path.win32.isAbsolute(p.executable), 'queue process identity unavailable');
  return { pid: p.pid, ppid: p.ppid, startedAt: p.startedAt, executable: p.executable };
}
export function workerIdentity(table, controllerPid, runner) {
  electedRole(runner);
  const chain = ancestors(table, controllerPid), workers = table.filter(p => chain.has(p.pid) && /^Runner\.Worker\.exe$/i.test(path.win32.basename(p.executable ?? p.command ?? '')));
  assert.equal(workers.length, 1, 'queue must descend from exactly one actual hosting Worker');
  const p = processIdentity(table, workers[0].pid); assert.equal(normalized(p.executable), normalized(SLOTS[runner].worker), 'queue Worker executable is not the admitted runner slot');
  return p;
}
function sameProcess(table, expected) { assert.deepEqual(processIdentity(table, expected.pid), expected, 'queue PID/start time/executable identity changed'); }
function sameIdentity(actual, expected) { for (const key of ['repository','runId','attempt','sourceSha','inferenceMainSha','controlSha256','nonce']) assert.equal(actual[key], expected[key], `queue ${key} mismatch`); }
export function verifyRoster(binding, pulses, table, controllerPid, now = Date.now()) {
  assert(binding?.kind === 'flux3-four-slot-reservation' && sha.test(binding.identity?.nonce ?? ''), 'explicit four-slot reservation required');
  assert.equal(binding.identity.repository, 'SceneWorks/SceneWorks');
  assert(now <= binding.deadline && binding.deadline - now <= CAMPAIGN_MS + WAIT_MS, 'queue deadline expired/invalid');
  assert.deepEqual(Object.keys(binding.members).sort(), Object.keys(SLOTS).sort(), 'exact four-member reservation required');
  const jobs = validateApi(binding.api.run, binding.api.jobs, binding.identity);
  assert(now >= binding.api.observedAt && now - binding.api.observedAt <= API_FRESH_MS, 'queue API proof stale');
  const workers = [], supervisors = [];
  for (const [runner, slot] of Object.entries(SLOTS)) {
    const member = binding.members[runner]; sameIdentity(member.identity, binding.identity);
    assert.equal(member.runner, runner); assert.equal(member.role, electedRole(runner)); assert.equal(member.jobId, jobs[runner]);
    assert.equal(member.hostName, binding.members[EXECUTOR].hostName, 'queue slots are not on the same physical host');
    assert.equal(normalized(member.worker.executable), normalized(slot.worker));
    sameProcess(table, member.worker); sameProcess(table, member.supervisor);
    assert(ancestors(table, member.supervisor.pid).has(member.worker.pid), 'queue supervisor is not a descendant of its hosting Worker');
    const pulse = authenticated(pulses[runner], member.publicKey);
    sameIdentity(pulse.identity, binding.identity); assert.equal(pulse.runner, runner); assert.equal(pulse.jobId, member.jobId);
    assert.equal(pulse.status, 'holding', 'queue reservation released');
    assert(Number.isInteger(pulse.sequence) && pulse.sequence > 0);
    assert(now >= pulse.at && now - pulse.at <= STALE_MS && pulse.at <= pulse.apiValidUntil && now <= pulse.apiValidUntil, 'queue heartbeat/API proof expired');
    workers.push(member.worker.pid); supervisors.push(member.supervisor.pid);
  }
  assert.equal(new Set(workers).size, 4); assert.equal(new Set(supervisors).size, 4);
  const chain = ancestors(table, controllerPid);
  assert(chain.has(binding.members[EXECUTOR].supervisor.pid), 'queue caller is not owned by the admitted executor supervisor');
  assert(chain.has(binding.members[EXECUTOR].worker.pid), 'queue caller lacks executor Worker ancestry');
  const actualWorkers = table.filter(p => /^Runner\.Worker(?:\.exe)?$/i.test(path.win32.basename(p.executable ?? p.command ?? '')));
  assert(actualWorkers.every(p => workers.includes(p.pid)), 'foreign fifth Runner.Worker is active');
  return Object.keys(SLOTS).filter(r => r !== EXECUTOR).map(r => binding.members[r].worker.pid);
}
async function atomic(file, value) { const tmp = `${file}.${randomBytes(8).toString('hex')}.tmp`; await writeFile(tmp, JSON.stringify(value), { flag: 'wx' }); await rename(tmp, file); }
// Publish a COMPLETE immutable document atomically, refusing any prior target. Readers never
// see an empty/part-written nonce/member/binding while siblings rendezvous on the same NTFS volume.
export async function publishNew(file, value) {
  const tmp = `${file}.${randomBytes(8).toString('hex')}.tmp`;
  await writeFile(tmp, JSON.stringify(value), { flag: 'wx' });
  try { await link(tmp, file); } finally { await unlink(tmp); }
}
async function optional(file) { try { return await json(file); } catch (error) { if (error.code === 'ENOENT') return undefined; throw error; } }
async function apiSnapshot(identity, env) {
  const call = async suffix => JSON.parse((await exec('gh', ['api', `repos/${identity.repository}/actions/runs/${identity.runId}/attempts/${identity.attempt}${suffix}`], { env, timeout: 15000, maxBuffer: 4 * 1024 * 1024 })).stdout);
  const run = await call(''), jobs = (await call('/jobs?per_page=100')).jobs;
  return { run: { id: run.id, repository: { full_name: run.repository?.full_name }, run_attempt: run.run_attempt, head_sha: run.head_sha, event: run.event, status: run.status, path: run.path },
    jobs: jobs.map(j => ({ id: j.id, name: j.name, run_id: j.run_id, head_sha: j.head_sha, status: j.status, runner_name: j.runner_name, runner_id: j.runner_id })), observedAt: Date.now() };
}
async function census() { const { observeProcessTable } = await import('./flux3-image-klein-terminal-session.mjs'); return observeProcessTable(exec, 'win32'); }
export async function verifyQueueReservation(admission, table, controllerPid, now = Date.now()) {
  const ref = admission.queueReservation; assert(ref && sha.test(ref.sha256 ?? ''), 'pinned queue reservation required');
  assert.equal(normalized(path.dirname(ref.path)), normalized(QUEUE_ROOT + `/${ref.runId}-${ref.attempt}`)); assert.equal(path.basename(ref.path), 'binding.json');
  const bytes = await readFile(ref.path); assert.equal(digest(bytes), ref.sha256, 'queue binding bytes changed'); const binding = JSON.parse(bytes);
  assert.equal(binding.identity.sourceSha, admission.sourceSha); assert.equal(binding.identity.inferenceMainSha, admission.inferenceMainSha);
  assert.equal(binding.identity.runId, ref.runId); assert.equal(binding.identity.attempt, ref.attempt);
  const dir = path.dirname(ref.path), pulses = {};
  for (const runner of Object.keys(SLOTS)) pulses[runner] = await json(path.join(dir, `${runner}.pulse.json`));
  // The immutable admission pins the executor signing public key. No model child has its private
  // key or API credential; refreshed API observations must be signed by that live supervisor.
  const update = authenticated(await json(path.join(dir, 'api.json')), binding.members[EXECUTOR].publicKey);
  sameIdentity(update.identity, binding.identity); binding.api = update.api;
  return verifyRoster(binding, pulses, table, controllerPid, now);
}
export function driverEnvironment(env, reservation) {
  const keep = {};
  for (const k of Object.keys(env)) if (!/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ACTIONS_|GITHUB_|GH_|SCENEWORKS_|FLUX3_QUEUE_/i.test(k)) keep[k] = env[k];
  return { ...keep, CUDA_VISIBLE_DEVICES: '-1', FLUX3_QUEUE_BINDING: JSON.stringify(reservation) };
}
/** Model work is unreachable until tick verifies the full barrier; ticks live outside its process. */
export async function superviseChild({ start, tick, release, wait = sleep, interval = HEARTBEAT_MS }) {
  let child, exit, failure, stopped = false;
  try {
    await tick(); child = await start(); child.done.then(value => { exit = value; stopped = true; }, error => { failure = error; stopped = true; });
    while (!stopped) { await wait(interval); if (!stopped) await tick(); }
    if (failure) throw failure; assert.equal(exit, 0, 'queue campaign child failed; no retry');
  } finally { try { if (child && !stopped) await child.stop(); } finally { await release(); } }
}
/** Terminate only the live descendants of OUR child, with a fresh start-time/executable check.
 * Used on cancellation/heartbeat loss; never a process-name kill or a foreign job cancellation. */
async function stopOwnedChild(child, identity) {
  const table = await census(); sameProcess(table, identity);
  const owned = table.filter(p => ancestors(table, p.pid).has(child.pid));
  owned.sort((a, b) => ancestors(table, b.pid).size - ancestors(table, a.pid).size);
  for (const p of owned) {
    const expected = processIdentity(table, p.pid), live = await census();
    if (!live.some(row => row.pid === p.pid)) continue;
    sameProcess(live, expected); process.kill(p.pid, 'SIGTERM');
  }
}
async function supervisor(env) {
  assert.equal(process.platform, 'win32'); const identity = queueInputs(env), runner = env.RUNNER_NAME, role = electedRole(runner);
  assert.equal(env.CUDA_VISIBLE_DEVICES, '-1', 'all reservation supervisors are CPU-only');
  assert.equal((await exec('git', ['-C', SOURCE_ROOT, 'rev-parse', 'HEAD'])).stdout.trim(), identity.sourceSha);
  assert.equal((await exec('git', ['-C', SOURCE_ROOT, 'status', '--porcelain'])).stdout.trim(), '');
  const controlBytes = await readFile(CONTROL_FILE); assert.equal(digest(controlBytes), identity.controlSha256, 'local queue control hash mismatch');
  const control = JSON.parse(controlBytes); assert.equal(control.sourceSha, identity.sourceSha); assert.equal(control.inferenceMainSha, identity.inferenceMainSha);
  const dir = `${QUEUE_ROOT}/${identity.runId}-${identity.attempt}`; await mkdir(dir, { recursive: true });
  // Durable per-run role claim: stale supervisor state is never replaced/recovered automatically.
  const claim = await open(path.join(dir, `${runner}.claim`), 'wx'); await claim.close();
  const pairFile = path.join(dir, 'nonce.json');
  if (role === 'executor') await publishNew(pairFile, { ...identity, nonce: randomBytes(32).toString('hex') });
  const waitDeadline = Date.now() + WAIT_MS;
  let aborted = false; const abort = () => { aborted = true; }; process.on('SIGTERM', abort); process.on('SIGINT', abort);
  let sequence = 0, released = false, campaignDeadline, api, member, privateKey, heartbeatTimer, heartbeatTask, heartbeatError;
  const pulse = status => atomic(path.join(dir, `${runner}.pulse.json`), signed({ identity, runner, jobId: member.jobId, at: Date.now(), apiValidUntil: api.observedAt + API_FRESH_MS, sequence: ++sequence, status }, privateKey));
  const release = async () => {
    if (released) return; released = true; clearInterval(heartbeatTimer);
    await heartbeatTask?.catch(() => {});
    if (member) await pulse('released'); log({ runner, released: true });
  };
  const check = async () => {
    assert(!aborted, 'queue supervisor interrupted'); if (heartbeatError) throw heartbeatError;
    assert(Date.now() < (campaignDeadline ?? waitDeadline), 'queue bounded deadline');
    if (!api || Date.now() - api.observedAt > 15000) api = await apiSnapshot(identity, env);
    validateApi(api.run, api.jobs, identity, { complete: false });
    assert(Date.now() - api.observedAt <= API_FRESH_MS, 'queue API observation expired');
    if (member) await pulse('holding');
  };
  try {
    let pair; while (!(pair = await optional(pairFile))) { await check(); await sleep(HEARTBEAT_MS); }
    for (const key of Object.keys(identity)) assert.equal(pair[key], identity[key]); Object.assign(identity, { nonce: pair.nonce });
    const keys = generateKeyPairSync('ed25519'); privateKey = keys.privateKey;
    const table = await census(), worker = workerIdentity(table, process.pid, runner), supervisorProcess = processIdentity(table, process.pid);
    await check(); const ownJobs = validateApi(api.run, api.jobs, identity, { complete: false }); assert(ownJobs[runner]);
    member = { role, identity, runner, hostName: hostname(), jobId: ownJobs[runner], worker, supervisor: supervisorProcess, publicKey: keys.publicKey.export({ type: 'spki', format: 'pem' }) };
    await publishNew(path.join(dir, `${runner}.member.json`), member);
    await pulse('holding');
    // Independent of asynchronous API/census requests AND the separate blocking model process.
    // API failure still expires at its last observation: publishing a pulse never renews authority.
    heartbeatTimer = setInterval(() => {
      if (heartbeatTask || released) return;
      heartbeatTask = pulse('holding').catch(e => { heartbeatError = e; }).finally(() => { heartbeatTask = undefined; });
    }, HEARTBEAT_MS);
    const members = {};
    let complete = false;
    while (!complete) {
      await check();
      for (const name of Object.keys(SLOTS)) members[name] = await optional(path.join(dir, `${name}.member.json`));
      if (Object.values(members).every(Boolean)) {
        // Only a legitimate scheduler wait is retried. Identity/API mismatches are fatal, not hidden.
        const running = api.jobs.filter(j => /^flux3-terminal-queue \/ reserve \([1-4]\)$/.test(j.name) && j.status === 'in_progress');
        if (running.length === 4) { validateApi(api.run, api.jobs, identity); complete = true; }
      }
      if (!complete) await sleep(HEARTBEAT_MS);
    }
    campaignDeadline = Date.now() + CAMPAIGN_MS;
    const binding = { kind: 'flux3-four-slot-reservation', identity, deadline: campaignDeadline, members, api };
    // Each holder verifies the same full roster, but guard callers must descend from executor.
    const readPulses = async () => { const pulses = {}; for (const name of Object.keys(SLOTS)) pulses[name] = await optional(path.join(dir, `${name}.pulse.json`)); return pulses; };
    let pulses;
    while (!(pulses = await readPulses()) || !Object.values(pulses).every(Boolean)) { await check(); await sleep(HEARTBEAT_MS); }
    verifyRoster(binding, pulses, await census(), members[EXECUTOR].supervisor.pid);
    if (role === 'executor') {
      const bindingFile = path.join(dir, 'binding.json'); await publishNew(bindingFile, binding);
      const ref = { path: bindingFile, sha256: digest(JSON.stringify(binding)), runId: identity.runId, attempt: identity.attempt };
      const tick = async () => {
        await check(); validateApi(api.run, api.jobs, identity);
        await atomic(path.join(dir, 'api.json'), signed({ identity, api }, privateKey));
        await verifyQueueReservation({ sourceSha: identity.sourceSha, inferenceMainSha: identity.inferenceMainSha, queueReservation: ref }, await census(), process.pid);
      };
      await tick(); log({ barrier: 'verified', runId: identity.runId, attempt: identity.attempt, jobs: Object.fromEntries(Object.entries(members).map(([r, m]) => [r, m.jobId])) });
      let logHandle;
      await superviseChild({ tick, release: async () => { try { await release(); } finally { await logHandle?.close(); } }, start: async () => {
        logHandle = await open(path.join(dir, 'campaign.local.log'), 'wx');
        const child = spawn(process.execPath, [fileURLToPath(import.meta.url), 'campaign'], { cwd: SOURCE_ROOT, env: driverEnvironment(env, ref), stdio: ['ignore', logHandle.fd, logHandle.fd] });
        const done = new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', code => resolve(code)); }); done.catch(() => {});
        let observed;
        try {
          for (let i = 0; i < 20 && !observed; i++) { const rows = await census(); if (rows.some(p => p.pid === child.pid)) observed = processIdentity(rows, child.pid); else await sleep(100); }
          assert(observed, 'owned campaign process identity missing');
          return { done, stop: async () => { if (child.exitCode === null && child.signalCode === null) await stopOwnedChild(child, observed); await done; } };
        } catch (error) {
          if (child.exitCode === null && child.signalCode === null) { if (observed) await stopOwnedChild(child, observed); else child.kill('SIGTERM'); }
          await done.catch(() => {}); throw error;
        }
      } });
    } else {
      while (true) {
        // Observe the executor's terminal pulse BEFORE requiring its API job to remain in_progress.
        const p = authenticated(await json(path.join(dir, `${EXECUTOR}.pulse.json`)), members[EXECUTOR].publicKey);
        sameIdentity(p.identity, identity); if (p.status === 'released') break;
        await check(); validateApi(api.run, api.jobs, identity); binding.api = api;
        verifyRoster(binding, await readPulses(), await census(), members[EXECUTOR].supervisor.pid);
        await sleep(HEARTBEAT_MS);
      }
    }
  } catch (error) {
    // Local only, never echoed to the public job log or uploaded.
    await writeFile(path.join(dir, `${runner}.failure.json`), JSON.stringify({ at: Date.now(), error: error.message }), { flag: 'wx' });
    throw error;
  } finally { try { await release(); } finally { process.off('SIGTERM', abort); process.off('SIGINT', abort); } }
}
export async function main(args = process.argv.slice(2), env = process.env) {
  if (args[0] === 'gate') { queueInputs(env); log({ gate: 'passed' }); return; }
  if (args[0] === 'campaign') { const { runQueuedCandleCampaign } = await import('./flux3-terminal-queue-campaign.mjs'); await runQueuedCandleCampaign(env); return; }
  assert(args[0] === 'reserve' && args.length === 1, 'one homogeneous queue reservation role required');
  await supervisor(env);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch(() => { console.error('BLOCKED: queued terminal reservation refused/failed; preserve local diagnostics and spent permits, no automatic retry'); process.exitCode = 1; });
