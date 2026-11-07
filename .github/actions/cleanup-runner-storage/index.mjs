import { appendFile, readFile, readdir, lstat } from 'node:fs/promises';
import { dirname, resolve, basename, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execute = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));
const day = 24 * 60 * 60 * 1000;
const warn = message => console.log(`::warning title=Runner storage cleanup::${String(message).replace(/[\r\n]/g, ' ')}`);
const inside = (path, root) => { const r = relative(root, path); return r === '' || (!r.startsWith('..') && !isAbsolute(r)); };

export function mossOwner(root, runtime) {
  const match = /^moss-(build|execute)-(\d+)-(\d+)-cuda$/.exec(basename(root));
  if (!match || !runtime || runtime.backend !== 'cuda' ||
      resolve(runtime.root ?? '') !== root || resolve(runtime.hfHome ?? '') !== resolve(root, 'hf') ||
      !/^[a-f0-9]{40}$/.test(runtime.sourceSha ?? '') || !/^[a-f0-9]{40}$/.test(runtime.inferenceSha ?? '')) return null;
  return { run: match[2], attempt: match[3], root };
}

export function retiredOwners(owners, now, currentRun) {
  // Retain the two newest completed cache owners, including failed attempts, for recovery.
  const completed = owners.filter(o => o.run !== currentRun && o.status === 'completed' &&
    Number.isFinite(Date.parse(o.updated_at))).sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
  return completed.slice(2).filter(o => now - Date.parse(o.updated_at) >= day);
}

export function disposableTargets(workspace, env) {
  const targets = [resolve(workspace, 'target'), resolve(workspace, 'source/target'), resolve(workspace, 'inference/target')];
  // Never follow arbitrary CARGO_TARGET_DIR / HF_HOME values outside this job's checkout.
  // A MOSS target has an exact current run/attempt namespace, outside checkout cleanup.
  if (env.MOSS_ROOT) {
    const root = resolve(env.MOSS_ROOT);
    const parent = resolve(workspace, '../moss-sc-26752');
    if (dirname(root) === parent && basename(root) === `moss-execute-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-cuda`) targets.push(resolve(root, 'target'));
  }
  return targets;
}

async function json(path) { return JSON.parse(await readFile(path, 'utf8')); }
async function exists(path) { try { await lstat(path); return true; } catch (e) { if (e.code === 'ENOENT') return false; throw e; } }

async function native(mode, path, parent, { dryRun = false } = {}) {
  const payload = Buffer.from(JSON.stringify({ mode, path, parent, dryRun })).toString('base64');
  const { stdout } = await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
    resolve(here, 'storage.ps1'), '-Payload', payload], { windowsHide: true, maxBuffer: 1024 * 1024 });
  const result = JSON.parse(stdout.trim());
  console.log(JSON.stringify(result));
  return result;
}

export async function runAttempt(owner, env) {
  if (!env.INPUT_TOKEN) throw Error('No Actions token; retaining historical MOSS caches');
  const response = await fetch(`${env.GITHUB_API_URL ?? 'https://api.github.com'}/repos/${env.GITHUB_REPOSITORY}/actions/runs/${owner.run}/attempts/${owner.attempt}`, {
    headers: { Authorization: `Bearer ${env.INPUT_TOKEN}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' },
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) throw Error(`Cannot verify run ${owner.run}/${owner.attempt}: HTTP ${response.status}`);
  const value = await response.json();
  if (String(value.id) !== owner.run || String(value.run_attempt) !== owner.attempt || value.repository?.full_name !== env.GITHUB_REPOSITORY) throw Error('Actions run ownership mismatch');
  // A retry can consume an earlier attempt's retained files. Verify the latest
  // run state too, rather than treating a completed old attempt as idle.
  const latestResponse = await fetch(`${env.GITHUB_API_URL ?? 'https://api.github.com'}/repos/${env.GITHUB_REPOSITORY}/actions/runs/${owner.run}`, {
    headers: { Authorization: `Bearer ${env.INPUT_TOKEN}`, Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(15000),
  });
  if (!latestResponse.ok) throw Error(`Cannot verify latest state of run ${owner.run}`);
  const latest = await latestResponse.json();
  if (String(latest.id) !== owner.run || latest.repository?.full_name !== env.GITHUB_REPOSITORY) throw Error('Latest run ownership mismatch');
  return { ...value, status: latest.status === 'completed' ? value.status : latest.status, updated_at: latest.updated_at };
}

export async function sweepMoss(workspace, env, { attempt = runAttempt, storage = native, now = Date.now() } = {}) {
  const parent = resolve(workspace, '../moss-sc-26752');
  if (!await exists(parent)) return;
  // Refuse links before readdir or reading any ownership records.
  if ((await storage('inspect-parent', parent, dirname(parent))).status !== 'verified') return;
  const owners = [], protectedPaths = [];
  let uncertainOwner = false;
  // A bundle on listener 2 can read a private cache left on listener 1. Read
  // configured listener manifests across this host before collecting any copy.
  const inputs = await storage('moss-inputs', workspace, dirname(workspace));
  if (inputs.status !== 'verified' || !Array.isArray(inputs.inputs)) { warn('Cannot verify cross-listener cache inputs; MOSS sweep skipped'); return; }
  protectedPaths.push(...inputs.inputs.map(path => resolve(path)));
  if (env.MOSS_HF_HOME) protectedPaths.push(resolve(env.MOSS_HF_HOME));
  for (const name of await readdir(parent)) {
    const root = resolve(parent, name);
    if (!/^moss-(build|execute)-\d+-\d+-cuda$/.test(name) || (await lstat(root)).isSymbolicLink()) continue;
    try {
      if ((await storage('inspect-parent', root, parent)).status !== 'verified') { uncertainOwner = true; continue; }
      // Only copies which actually exist count towards the two retained owners.
      const hasHf = await exists(resolve(root, 'hf'));
      if (!hasHf && !await exists(resolve(root, 'target'))) continue;
      // The current read-only smoke has no private HF runtime. Target-only roots
      // still require API ownership plus the native Cargo marker/lock guards.
      const runtime = hasHf ? await json(resolve(root, 'runtime.json')) : {};
      const match = /^moss-(build|execute)-(\d+)-(\d+)-cuda$/.exec(name);
      const owner = hasHf ? mossOwner(root, runtime) : { run: match[2], attempt: match[3], root };
      if (!owner) throw Error('Cannot verify private cache ownership');
      if (owner.run === env.GITHUB_RUN_ID) {
        for (const key of ['hfHome', 'packageSource']) if (typeof runtime[key] === 'string') protectedPaths.push(resolve(runtime[key]));
        continue;
      }
      const remote = await attempt(owner, env);
      owners.push({ ...owner, status: remote.status, updated_at: remote.updated_at });
      // Inputs of active/unknown owners must never be collected, even if their source owner completed.
      if (remote.status !== 'completed') for (const key of ['hfHome', 'packageSource']) if (typeof runtime[key] === 'string') protectedPaths.push(resolve(runtime[key]));
    } catch (e) { uncertainOwner = true; warn(`${name}: ${e.message}; retained`); }
  }
  if (uncertainOwner) { warn('Unresolved MOSS ownership; historical sweep skipped'); return; }
  // Bundle caches can be inputs to future smoke/retry jobs. Preserve their referenced source.
  for (const name of await readdir(parent)) if (/^moss-bundle-\d+-\d+$/.test(name)) {
    const root = resolve(parent, name);
    if ((await storage('inspect-parent', root, parent)).status !== 'verified') { warn('Unresolved bundle path; MOSS sweep skipped'); return; }
    try {
      const bundle = await json(resolve(root, 'campaign.json'));
      if (typeof bundle.backends?.cuda?.hfHome === 'string') protectedPaths.push(resolve(bundle.backends.cuda.hfHome));
    } catch (e) {
      // An unreadable bundle may name any historical input. Do not guess which one.
      warn(`${name}: cannot resolve model inputs (${e.message}); MOSS sweep skipped`); return;
    }
  }
  for (const owner of retiredOwners(owners, now, env.GITHUB_RUN_ID)) {
    if (protectedPaths.some(p => inside(p, owner.root) || inside(owner.root, p))) continue;
    // Recheck remote completion immediately before removal; reruns are not treated as completed.
    const fresh = await attempt(owner, env);
    if (fresh.status !== 'completed' || !Number.isFinite(Date.parse(fresh.updated_at)) || now - Date.parse(fresh.updated_at) < day) continue;
    for (const leaf of ['hf', 'target']) if (await exists(resolve(owner.root, leaf))) await storage(leaf === 'hf' ? 'moss' : 'cargo', resolve(owner.root, leaf), owner.root);
  }
}

export async function main(env = process.env) {
  if (process.platform !== 'win32' || env.RUNNER_ENVIRONMENT !== 'self-hosted') return;
  const workspace = resolve(env.GITHUB_WORKSPACE);
  if (!env.STATE_registered) {
    // Registered before rust-cache: post actions run in reverse order, so uploads finish first.
    // The barrier coordinates a newly starting model reader with a running deletion.
    if ((await native('barrier', workspace, dirname(workspace))).status !== 'registered') throw Error('Could not register storage safety barrier');
    await appendFile(env.GITHUB_STATE, 'registered=true\n');
    return;
  }
  for (const target of disposableTargets(workspace, env)) {
    if (await exists(target)) await native('cargo', target, dirname(target));
  }
  if (env.INPUT_MOSS === 'true' && env.GITHUB_REPOSITORY === 'SceneWorks/SceneWorks') await sweepMoss(workspace, env);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { await main(); } catch (e) {
    warn(`${e.message}; remaining storage retained`);
    // A new reader must not proceed if it failed to coordinate with a deletion.
    if (!process.env.STATE_registered) process.exitCode = 1;
  }
}
