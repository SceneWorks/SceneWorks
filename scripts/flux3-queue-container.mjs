// Windows zero-privilege nested Job Object wrapper. This helper, not the model root, owns the
// kernel containment handle and cleanup-signing key. It watches supervisor death independently.
import assert from 'node:assert/strict';
import { spawn, execFile as execCb } from 'node:child_process';
import { createPublicKey, randomBytes, verify } from 'node:crypto';
import { readFile, open } from 'node:fs/promises';
import { promisify } from 'node:util';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { publishNew, processIdentity } from './flux3-terminal-queue.mjs';
const exec = promisify(execCb), sleep = ms => new Promise(r => setTimeout(r, ms));
async function optional(file) { try { return JSON.parse(await readFile(file, 'utf8')); } catch (e) { if (e.code === 'ENOENT') return undefined; throw e; } }
export function verifyCleanupAck(envelope, container, identity) {
  assert(typeof envelope?.payload === 'string' && verify('sha256', Buffer.from(envelope.payload), createPublicKey({ key: container.publicKey, format: 'jwk' }), Buffer.from(envelope.signature ?? '', 'base64')), 'container cleanup ACK signature invalid');
  const ack = JSON.parse(envelope.payload);
  assert.equal(ack.kind, 'flux3-container-cleanup-ack'); assert.equal(ack.jobName, container.jobName);
  for (const key of ['repository','runId','attempt','sourceSha','inferenceMainSha','controlSha256','nonce']) assert.equal(ack.identity[key], identity[key], `cleanup ACK ${key} mismatch`);
  assert(ack.cleanupVerified === true && ack.activeProcesses === 0, 'container has not proven zero owned processes');
  return ack;
}
export async function readCleanupAck(container, identity) {
  const ack = await optional(container.ackFile); return ack ? verifyCleanupAck(ack, container, identity) : undefined;
}
// libuv/GitHub may kill the helper together with an abruptly dying supervisor. KILL_ON_CLOSE
// still kills its tree. Holders must independently verify the kernel job is zero/destroyed, never
// infer cleanup merely from parent/helper disappearance. A named Job is destroyed only after its
// handles are closed AND associated processes terminate; query errors are NOT an absent object.
export async function currentSecurityContext() {
  const value = await exec('powershell.exe', ['-NoProfile','-Command', '@{sessionId=(Get-Process -Id $PID).SessionId;ownerSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}|ConvertTo-Json -Compress'], { timeout: 5000 });
  return JSON.parse(value.stdout);
}
export function assertContainerContext(actual, expected) {
  assert(Number.isInteger(actual?.sessionId) && /^S-[0-9-]+$/.test(actual?.ownerSid ?? ''), 'actual Windows security context required');
  assert.deepEqual(actual, expected, 'owned job namespace/account context mismatch: absence is NOT cleanup proof');
}
async function queryContainer(container) {
  assert(/^Local\\SceneWorksFlux3-[0-9]+-[0-9]+-[a-f0-9]{40}$/.test(container.jobName), 'invalid owned kernel job name');
  const root = process.env.SystemRoot ?? process.env.SYSTEMROOT;
  const query = await exec(path.join(root, 'System32/WindowsPowerShell/v1.0/powershell.exe'), ['-NoProfile', '-File', fileURLToPath(new URL('./flux3-queue-container.ps1', import.meta.url)), '-QueryJob', container.jobName], { env: (await import('./flux3-terminal-queue.mjs')).driverEnvironment(process.env, null), timeout: 15000 });
  const result = JSON.parse(query.stdout);
  assertContainerContext(result.context, container.context);
  return result;
}
export async function probeContainerAccess(container) {
  const result = await queryContainer(container);
  assert(result.exists === true && result.activeProcesses === 0, 'holders must prove query access while the container is empty before model execution');
  return result.context;
}
export async function cleanupStatus(container, identity) {
  const ack = await readCleanupAck(container, identity); if (ack) return ack;
  const { observeProcessTable } = await import('./flux3-image-klein-terminal-session.mjs');
  const rows = await observeProcessTable(exec, 'win32'), expected = container.helper;
  if (rows.some(p => p.pid === expected.pid && p.startedAt === expected.startedAt && p.executable === expected.executable)) return undefined;
  const result = await queryContainer(container);
  if (!(result.exists === true && result.activeProcesses === 0) && !(result.exists === false && result.kernelObjectAbsent === true && result.activeProcesses === -1)) return undefined;
  return { kind: 'flux3-container-kernel-zero-observation', identity, jobName: container.jobName, cleanupVerified: true, activeProcesses: 0, helperGone: true, kernelObjectAbsent: result.kernelObjectAbsent, exitCode: 1 };
}
export async function prepareContainer({ dir, identity, deadline, env, supervisorPid = process.pid }) {
  assert.equal(process.platform, 'win32');
  const nonce = randomBytes(20).toString('hex'), prefix = path.join(dir, `container-${nonce}`);
  const jobName = `Local\\SceneWorksFlux3-${identity.runId}-${identity.attempt}-${nonce}`;
  const paths = { readyFile: `${prefix}.ready.json`, ackFile: `${prefix}.ack.json`, startFile: `${prefix}.start.json`, stopFile: `${prefix}.stop.json` };
  const boot = `${prefix}.boot.json`; await publishNew(boot, { ...paths, identity, jobName, supervisorPid, deadline });
  const systemRoot = env.SystemRoot ?? env.SYSTEMROOT; assert(path.win32.isAbsolute(systemRoot ?? ''), 'Windows system directory required');
  const exe = path.join(systemRoot, 'System32/WindowsPowerShell/v1.0/powershell.exe'), log = await open(`${prefix}.helper.log`, 'wx');
  const helper = spawn(exe, ['-NoProfile', '-File', fileURLToPath(new URL('./flux3-queue-container.ps1', import.meta.url)), '-BootFile', boot], { env, windowsHide: true, stdio: ['ignore', log.fd, log.fd] });
  let exited = false;
  const exit = new Promise((resolve, reject) => { helper.once('error', reject); helper.once('exit', code => { exited = true; resolve(code); }); }); exit.catch(() => { exited = true; });
  let definition, started = false;
  const stop = async () => {
    if (!exited) { try { await publishNew(paths.stopFile, { stop: true }); } catch (e) { if (e.code !== 'EEXIST') throw e; } }
    await exit;
    const ack = definition && await cleanupStatus(definition, identity);
    assert(ack, 'container cleanup not acknowledged: do not release reserved slots');
    return ack;
  };
  try {
    const limit = Date.now() + 60000; let ready;
    while (!(ready = await optional(paths.readyFile))) { assert(!exited && Date.now() < limit, 'container failed before ready/exec; no model child started'); await sleep(50); }
    assert.equal(ready.helperPid, helper.pid); assert.equal(ready.jobName, jobName); assert.equal(ready.nonce, identity.nonce); assert.equal(ready.killOnClose, true);
    const { observeProcessTable } = await import('./flux3-image-klein-terminal-session.mjs');
    const rows = await observeProcessTable(exec, 'win32'), helperIdentity = processIdentity(rows, helper.pid);
    assert.equal(helperIdentity.ppid, process.pid); assert.equal(helperIdentity.executable.toLowerCase(), exe.toLowerCase());
    definition = { jobName, publicKey: ready.publicKey, helper: helperIdentity, ackFile: paths.ackFile, context: ready.context };
    const done = exit.then(async () => { try { const ack = await cleanupStatus(definition, identity); assert(ack, 'owned container exited without verified zero-tree cleanup'); return ack.exitCode; } finally { await log.close(); } }); done.catch(() => {});
    return { definition,
      async start({ executable, args, childEnv, cwd, localLog }) {
        assert(!started, 'container starts once'); started = true;
        await publishNew(paths.startFile, { exe: executable, args, env: childEnv, cwd, log: localLog });
      },
      // Root exit is NOT done. The helper kills lingering descendants then signs the kernel-zero ACK.
      done,
      stop,
    };
  } catch (error) {
    // Even before start, request helper cleanup; no detached helper/container is abandoned.
    try { if (!exited) await publishNew(paths.stopFile, { stop: true }); await exit; } finally { await log.close(); }
    throw error;
  }
}
