// sc-26243: four queued slots, ONE executor, THREE CPU-only holders on one Windows box.
// Not a global GPU lock. No runner administration, uploads, retries, or ledger reset.
// The API/heartbeat supervisors stay independent of the credential-free blocking model child.
import assert from 'node:assert/strict';
import { createHash, generateKeyPairSync, randomBytes, sign, verify } from 'node:crypto';
import { execFile as execCb } from 'node:child_process';
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
  return { pid: p.pid, ppid: p.ppid, startedAt: p.startedAt, executable: p.executable, ...(Number.isInteger(p.sessionId) ? { sessionId: p.sessionId } : {}) };
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
    assert(Number.isInteger(member.context?.sessionId) && /^S-[0-9-]+$/.test(member.context?.ownerSid ?? ''), 'queue actual Windows context required');
    assert.deepEqual(member.context, binding.members[EXECUTOR].context, 'queue Local job namespace/account differs across slots');
    assert.equal(member.worker.sessionId, member.context.sessionId); assert.equal(member.supervisor.sessionId, member.context.sessionId);
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
  if (binding.container) {
    const attested = authenticated(binding.containerAttestation, binding.members[EXECUTOR].publicKey);
    sameIdentity(attested.identity, binding.identity); assert.deepEqual(attested.container, binding.container);
    sameProcess(table, binding.container.helper);
    assert.deepEqual(binding.container.context, binding.members[EXECUTOR].context, 'container namespace/account differs from reserved slots');
    assert.equal(binding.container.helper.sessionId, binding.container.context.sessionId);
    assert(ancestors(table, binding.container.helper.pid).has(binding.members[EXECUTOR].supervisor.pid), 'container helper is not owned by executor');
    if (controllerPid !== binding.members[EXECUTOR].supervisor.pid) assert(chain.has(binding.container.helper.pid), 'model caller is outside owned containment');
  }
  return Object.keys(SLOTS).filter(r => r !== EXECUTOR).map(r => binding.members[r].worker.pid);
}
export function verifyAccessProofs(binding, proofs) {
  for (const runner of Object.keys(SLOTS).filter(r => r !== EXECUTOR)) {
    const p = authenticated(proofs[runner], binding.members[runner].publicKey);
    sameIdentity(p.identity, binding.identity); assert.equal(p.runner, runner);
    assert.equal(p.containerSha256, digest(binding.container)); assert.equal(p.jobName, binding.container.jobName);
    assert.deepEqual(p.context, binding.container.context); assert.equal(p.queryAccess, true); assert.equal(p.zeroBeforeStart, true);
  }
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
export async function verifyObservedRoster(observe, controllerPid, clock = Date.now) {
  assert.equal(typeof clock, 'function');
  const { binding, pulses, table } = await observe();
  return verifyRoster(binding, pulses, table, controllerPid, clock());
}
export async function verifyQueueReservation(admission, table, controllerPid, clock = Date.now) {
  return verifyObservedRoster(async () => {
    const ref = admission.queueReservation; assert(ref && sha.test(ref.sha256 ?? ''), 'pinned queue reservation required');
    assert.equal(normalized(path.dirname(ref.path)), normalized(QUEUE_ROOT + `/${ref.runId}-${ref.attempt}`)); assert.equal(path.basename(ref.path), 'binding.json');
    const bytes = await readFile(ref.path); assert.equal(digest(bytes), ref.sha256, 'queue binding bytes changed'); const binding = JSON.parse(bytes);
    assert(binding.container && binding.containerAttestation, 'queued model work requires owned kill-on-close containment');
    assert.equal(binding.identity.sourceSha, admission.sourceSha); assert.equal(binding.identity.inferenceMainSha, admission.inferenceMainSha);
    assert.equal(binding.identity.runId, ref.runId); assert.equal(binding.identity.attempt, ref.attempt);
    const dir = path.dirname(ref.path), pulses = {};
    for (const runner of Object.keys(SLOTS)) pulses[runner] = await json(path.join(dir, `${runner}.pulse.json`));
    // The immutable admission pins the executor signing public key. No model child has its private
    // key or API credential; refreshed API observations must be signed by that live supervisor.
    const update = authenticated(await json(path.join(dir, 'api.json')), binding.members[EXECUTOR].publicKey);
    sameIdentity(update.identity, binding.identity); binding.api = update.api;
    const access = {}; for (const runner of Object.keys(SLOTS).filter(r => r !== EXECUTOR)) access[runner] = await json(path.join(dir, `${runner}.container-access.json`));
    verifyAccessProofs(binding, access);
    return { binding, pulses, table };
  }, controllerPid, clock);
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
  } finally {
    // Root exit is not tree cleanup. Always clean/verify the containment, even after done rejects
    // or the direct child has exited. Cleanup failure MUST NOT acknowledge/release reservations.
    if (child) await child.stop();
    await release();
  }
}
// Failure/cancel/deadline requests cleanup; it never turns an unverified ACK into slot release.
export async function holdUntilSafe({ probe, heartbeat, wait = sleep }) {
  while (true) {
    let safe = false; try { safe = await probe(); } catch { /* retain on missing/corrupt cleanup proof */ }
    if (safe === true) return;
    try { await heartbeat(); } catch { /* stale heartbeat requests executor cleanup; not release */ }
    await wait(HEARTBEAT_MS);
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
  const claim = await open(path.join(dir, `${runner}.claim`), 'wx'); await claim.close();
  const pairFile = path.join(dir, 'nonce.json');
  if (role === 'executor') await publishNew(pairFile, { ...identity, nonce: randomBytes(32).toString('hex') });
  const waitDeadline = Date.now() + WAIT_MS;
  let aborted = false, state = 'holding', sequence = 0, released = false, campaignDeadline, api, member, privateKey, container, members = {};
  let heartbeatTimer, heartbeatTask, heartbeatError, cleanupProofSha256;
  const recordCleanup = async proof => {
    const value = signed({ identity, runner, proof }, privateKey);
    await atomic(path.join(dir, `${runner}.cleanup.json`), value); cleanupProofSha256 = digest(value);
  };
  const abort = () => { aborted = true; }; process.on('SIGTERM', abort); process.on('SIGINT', abort);
  const pulse = () => atomic(path.join(dir, `${runner}.pulse.json`), signed({ identity, runner, jobId: member.jobId, at: Date.now(), apiValidUntil: api.observedAt + API_FRESH_MS, sequence: ++sequence, status: state, containerCreated: Boolean(container), cleanupProofSha256: cleanupProofSha256 ?? null }, privateKey));
  const requestFailure = async () => { state = 'failure_requested'; if (member) await pulse(); };
  const release = async () => {
    if (released) return;
    // No API exception, deadline, or SIGINT authorizes releasing a holder ahead of cleanup.
    if (role === 'executor' && container && !cleanupProofSha256) await recordCleanup(await container.stop());
    released = true; clearInterval(heartbeatTimer); await heartbeatTask?.catch(() => {});
    state = 'safe_to_release'; if (member) await pulse(); log({ runner, cleanupAcknowledged: true, released: true });
  };
  const check = async () => {
    assert(!aborted, 'queue supervisor interrupted'); if (heartbeatError) throw heartbeatError;
    assert(Date.now() < (campaignDeadline ?? waitDeadline), 'queue bounded deadline requests cleanup');
    if (!api || Date.now() - api.observedAt > 15000) api = await apiSnapshot(identity, env);
    validateApi(api.run, api.jobs, identity, { complete: false });
    assert(Date.now() - api.observedAt <= API_FRESH_MS, 'queue API observation expired');
    if (member) await pulse();
  };
  // Only container readiness/cleanup keys attested by our actual executor supervisor count.
  const containerFrom = async binding => {
    const execMember = members[EXECUTOR] ?? await optional(path.join(dir, `${EXECUTOR}.member.json`)); assert(execMember, 'executor member unavailable');
    members[EXECUTOR] = execMember; sameIdentity(execMember.identity, identity);
    const attested = authenticated(binding.containerAttestation, execMember.publicKey); sameIdentity(attested.identity, identity);
    assert.deepEqual(attested.container, binding.container); return binding.container;
  };
  const safeHolderRelease = async () => {
    const { cleanupStatus } = await import('./flux3-queue-container.mjs');
    await holdUntilSafe({ probe: async () => {
      // Do not run the ordinary aborted/API/deadline checks here: they requested cleanup, NOT release.
      const binding = await optional(path.join(dir, 'binding.json'));
      if (binding) { const owned = await containerFrom(binding), proof = await cleanupStatus(owned, identity); if (proof) { await recordCleanup(proof); return true; } }
      else {
        const executor = members[EXECUTOR] ?? await optional(path.join(dir, `${EXECUTOR}.member.json`));
        if (executor) {
          members[EXECUTOR] = executor; sameIdentity(executor.identity, identity);
          const envelope = await optional(path.join(dir, `${EXECUTOR}.pulse.json`));
          if (envelope) { const p = authenticated(envelope, executor.publicKey); sameIdentity(p.identity, identity); if (p.status === 'safe_to_release' && p.containerCreated === false) return true; }
          const live = await census();
          // The preserved source publishes binding BEFORE any child start. Once executor is gone,
          // an absent immutable binding proves this attempt never reached model execution.
          if (!live.some(p => p.pid === executor.supervisor.pid && p.startedAt === executor.supervisor.startedAt)) {
            if (!await optional(path.join(dir, 'binding.json'))) return true;
          }
        } else {
          // No executor announcement means the four-member barrier could never have passed.
          const live = await census();
          if (!live.some(p => normalized(p.executable) === normalized(SLOTS[EXECUTOR].worker))) return true;
        }
      }
      return false;
    }, heartbeat: async () => { if (member) await pulse(); } });
  };
  try {
    let pair; while (!(pair = await optional(pairFile))) { await check(); await sleep(HEARTBEAT_MS); }
    for (const key of Object.keys(identity)) assert.equal(pair[key], identity[key]); Object.assign(identity, { nonce: pair.nonce });
    const keys = generateKeyPairSync('ed25519'); privateKey = keys.privateKey;
    const table = await census(), worker = workerIdentity(table, process.pid, runner), supervisorProcess = processIdentity(table, process.pid);
    const context = await (await import('./flux3-queue-container.mjs')).currentSecurityContext();
    assert.equal(worker.sessionId, context.sessionId); assert.equal(supervisorProcess.sessionId, context.sessionId);
    await check(); const ownJobs = validateApi(api.run, api.jobs, identity, { complete: false }); assert(ownJobs[runner]);
    member = { role, identity, runner, context, hostName: hostname(), jobId: ownJobs[runner], worker, supervisor: supervisorProcess, publicKey: keys.publicKey.export({ type: 'spki', format: 'pem' }) };
    await publishNew(path.join(dir, `${runner}.member.json`), member); await pulse();
    heartbeatTimer = setInterval(() => {
      if (heartbeatTask || released) return;
      heartbeatTask = pulse().catch(e => { heartbeatError = e; }).finally(() => { heartbeatTask = undefined; });
    }, HEARTBEAT_MS);
    let complete = false;
    while (!complete) {
      await check();
      for (const name of Object.keys(SLOTS)) members[name] = await optional(path.join(dir, `${name}.member.json`));
      if (Object.values(members).every(Boolean)) {
        const running = api.jobs.filter(j => /^flux3-terminal-queue \/ reserve \([1-4]\)$/.test(j.name) && j.status === 'in_progress');
        if (running.length === 4) { validateApi(api.run, api.jobs, identity); complete = true; }
      }
      if (!complete) await sleep(HEARTBEAT_MS);
    }
    campaignDeadline = Date.now() + CAMPAIGN_MS;
    const binding = { kind: 'flux3-four-slot-reservation', identity, deadline: campaignDeadline, members, api };
    const readPulses = async () => { const p = {}; for (const n of Object.keys(SLOTS)) p[n] = await optional(path.join(dir, `${n}.pulse.json`)); return p; };
    let pulses; while (!(pulses = await readPulses()) || !Object.values(pulses).every(Boolean)) { await check(); await sleep(HEARTBEAT_MS); }
    verifyRoster(binding, pulses, await census(), members[EXECUTOR].supervisor.pid);
    if (role === 'executor') {
      const { prepareContainer } = await import('./flux3-queue-container.mjs');
      container = await prepareContainer({ dir, identity, deadline: campaignDeadline, env: driverEnvironment(env, null) });
      binding.container = container.definition;
      assert.deepEqual(binding.container.context, member.context, 'container helper Windows context mismatch');
      binding.containerAttestation = signed({ identity, container: binding.container }, privateKey);
      const bindingFile = path.join(dir, 'binding.json'); await publishNew(bindingFile, binding);
      const ref = { path: bindingFile, sha256: digest(JSON.stringify(binding)), runId: identity.runId, attempt: identity.attempt };
      // All holders must actually open/query this EMPTY Local Job under their own real context.
      // No child CreateProcess/start file exists until every signed access acknowledgement arrives.
      while (true) {
        await check(); binding.api = api; verifyRoster(binding, await readPulses(), await census(), process.pid);
        const access = {}; for (const name of Object.keys(SLOTS).filter(r => r !== EXECUTOR)) access[name] = await optional(path.join(dir, `${name}.container-access.json`));
        if (Object.values(access).every(Boolean)) { verifyAccessProofs(binding, access); break; }
        await sleep(HEARTBEAT_MS);
      }
      const tick = async () => {
        await check(); validateApi(api.run, api.jobs, identity);
        await atomic(path.join(dir, 'api.json'), signed({ identity, api }, privateKey));
        await verifyQueueReservation({ sourceSha: identity.sourceSha, inferenceMainSha: identity.inferenceMainSha, queueReservation: ref }, await census(), process.pid);
      };
      await tick(); log({ barrier: 'verified', runId: identity.runId, attempt: identity.attempt, jobs: Object.fromEntries(Object.entries(members).map(([r, m]) => [r, m.jobId])) });
      await superviseChild({ tick, release, start: async () => {
        await container.start({ executable: process.execPath, args: [fileURLToPath(import.meta.url), 'campaign'], childEnv: driverEnvironment(env, ref), cwd: SOURCE_ROOT, localLog: path.join(dir, 'campaign.local.log') });
        return container;
      } });
    } else {
      let accessProved = false;
      while (true) {
        if (!accessProved) {
          const prepared = await optional(path.join(dir, 'binding.json'));
          if (prepared) {
            const owned = await containerFrom(prepared), context = await (await import('./flux3-queue-container.mjs')).probeContainerAccess(owned);
            assert.deepEqual(context, member.context);
            await publishNew(path.join(dir, `${runner}.container-access.json`), signed({ identity, runner, jobName: owned.jobName, containerSha256: digest(owned), context, queryAccess: true, zeroBeforeStart: true }, privateKey));
            accessProved = true;
          }
        }
        const p = authenticated(await json(path.join(dir, `${EXECUTOR}.pulse.json`)), members[EXECUTOR].publicKey); sameIdentity(p.identity, identity);
        if (p.status === 'safe_to_release') { await safeHolderRelease(); break; }
        await check(); validateApi(api.run, api.jobs, identity); binding.api = api;
        verifyRoster(binding, await readPulses(), await census(), members[EXECUTOR].supervisor.pid);
        await sleep(HEARTBEAT_MS);
      }
    }
  } catch (error) {
    await writeFile(path.join(dir, `${runner}.failure.json`), JSON.stringify({ at: Date.now(), error: error.message }), { flag: 'wx' });
    await requestFailure();
    throw error;
  } finally {
    try {
      if (!released) {
        await requestFailure();
        if (role === 'executor') { if (container) await container.stop(); }
        else await safeHolderRelease();
        await release();
      }
    } finally { clearInterval(heartbeatTimer); process.off('SIGTERM', abort); process.off('SIGINT', abort); }
  }
}
export async function main(args = process.argv.slice(2), env = process.env) {
  if (args[0] === 'gate') { queueInputs(env); log({ gate: 'passed' }); return; }
  if (args[0] === 'campaign') { const { runQueuedCandleCampaign } = await import('./flux3-terminal-queue-campaign.mjs'); await runQueuedCandleCampaign(env); return; }
  assert(args[0] === 'reserve' && args.length === 1, 'one homogeneous queue reservation role required');
  await supervisor(env);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch(() => { console.error('BLOCKED: queued terminal reservation refused/failed; preserve local diagnostics and spent permits, no automatic retry'); process.exitCode = 1; });
