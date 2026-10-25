// One stopped sc-26752 recovery only. No configurable deletion paths or model execution.
import { createHash } from 'node:crypto';
import { readFile, rm, cp, mkdir, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { win32 as path } from 'node:path';
const execute = promisify(execFile);
const check = (value, message) => { if (!value) throw Error(message); };
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const normal = value => path.normalize(value).toLowerCase();
const native = '857f7a1996e3678a0773c3509ac0087522c19ce6';
export const OWNED_RUNS = Object.freeze([
  { run: '40145054879', candidate: 'sc-26752-final-r4', source: '37d5ad93f858025d3202a2ddcab465c67174e0c7', inference: '878f26aba50bfb6c371437da9ca7ead81052c25a', workspace: 'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks', runtimeSha: '1b8948de0a9f7ede92019896359683605fe2c4e7577971520da881dda6aca710' },
  { run: '40152001538', candidate: 'sc-26752-final-r5', source: '9ac78275e2f5f536541c1790fc840232fd5bdfdd', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: '7ac52e00e6e712012b907a740e96e5c9d39c72483499205f637916e286e70b43' },
  { run: '40155160639', candidate: 'sc-26752-final-r6', source: '2490b2eed9bf986967c389ac69171d03b04835d7', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: 'cfb0491496d4e29aafc466404526592fe2d2950a12f1566c74fcab9e728f007e' },
].map(item => Object.freeze({ ...item, attempt: 1, root: path.join(item.workspace, `moss-build-${item.run}-1-cuda`) })));
export function destination(env) {
  check(env.GITHUB_REPOSITORY === 'SceneWorks/SceneWorks', 'wrong repository');
  check(/^\d+$/.test(env.GITHUB_RUN_ID ?? '') && /^\d+$/.test(env.GITHUB_RUN_ATTEMPT ?? ''), 'missing current Actions identity');
  check(!OWNED_RUNS.some(item => item.run === env.GITHUB_RUN_ID), 'cannot reclaim the executing run');
  check(/^[a-f0-9]{40}$/.test(env.MOSS_SOURCE_SHA ?? '') && /^[a-f0-9]{40}$/.test(env.MOSS_INFERENCE_SHA ?? ''), 'missing current source identity');
  check(/^D:\\actions-runner(?:-[234])?\\_work\\SceneWorks\\SceneWorks$/i.test(path.normalize(env.GITHUB_WORKSPACE ?? '')), 'unexpected Windows workspace');
  const root = path.join(env.GITHUB_WORKSPACE, `moss-build-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-cuda`);
  check(normal(env.MOSS_ROOT ?? '') === normal(root), 'current run root mismatch');
  return root;
}
export function validateStopped(binding, latest, attempt, jobs) {
  for (const run of [latest, attempt]) check(String(run.id) === binding.run && run.run_attempt === 1 && run.head_sha === binding.source && run.status === 'completed' && run.conclusion, 'old run identity/attempt is not stopped');
  check(jobs.length > 0 && jobs.every(job => job.status === 'completed' && job.conclusion && Number(job.run_id) === Number(binding.run)), 'old run still has nonterminal jobs');
}
export function validateRuntime(binding, bytes) {
  check(sha(bytes) === binding.runtimeSha, 'old runtime bytes changed');
  const runtime = JSON.parse(bytes);
  check(runtime.candidateId === binding.candidate && runtime.sourceSha === binding.source && runtime.inferenceSha === binding.inference && runtime.backend === 'cuda', 'old runtime identity mismatch');
  for (const [field, expected] of Object.entries({ root: binding.root, hfHome: path.join(binding.root, 'hf'), sourceRoot: path.join(binding.workspace, 'source'), packageSource: 'E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9' })) check(normal(runtime[field] ?? '') === normal(expected), `old runtime ${field} mismatch`);
  return runtime;
}
export function validateTree(snapshot, expected) {
  check(normal(snapshot.root ?? '') === normal(expected) && normal(path.parse(expected).root) === 'd:\\', 'tree escaped exact D: path');
  check(typeof snapshot.exists === 'boolean', 'missing existence observation');
  check(snapshot.safe === true && snapshot.reparsePoints === 0, 'reparse/ancestry confinement failed');
  check(Number.isSafeInteger(snapshot.bytes) && snapshot.bytes >= 0 && Number.isSafeInteger(snapshot.fileCount) && snapshot.fileCount >= 0, 'invalid byte inventory');
  return snapshot;
}
// PowerShell is used only for Windows file attributes and volume telemetry. Traversal refuses
// every reparse tag BEFORE descending, including junctions and non-symlink reparse points.
export async function powershell(script) {
  const encoded = Buffer.from(script, 'utf16le').toString('base64');
  const { stdout } = await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], { maxBuffer: 16 * 1024 * 1024, windowsHide: true });
  return JSON.parse(stdout.replace(/^\uFEFF/, '').trim());
}
export async function inspectTree(root) {
  const encoded = Buffer.from(root).toString('base64');
  return powershell(`$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encoded}'))
$cursor=[IO.Path]::GetFullPath($p)
while($cursor){if([IO.File]::Exists($cursor) -or [IO.Directory]::Exists($cursor)){if(([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse ancestor'}};$next=[IO.Path]::GetDirectoryName($cursor);if($next -eq $cursor){break};$cursor=$next}
$exists=[IO.Directory]::Exists($p);$bytes=[long]0;$count=0;$visited=0;$files=@();$stack=New-Object 'Collections.Generic.Stack[string]';if($exists){$stack.Push($p)}
while($stack.Count){$dir=$stack.Pop();foreach($entry in [IO.Directory]::EnumerateFileSystemEntries($dir)){$visited++;if($visited -gt 250000){throw 'Inventory bound exceeded'};$a=[IO.File]::GetAttributes($entry);if(($a -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse descendant'};if(($a -band [IO.FileAttributes]::Directory) -ne 0){$stack.Push($entry)}else{$size=([IO.FileInfo]::new($entry)).Length;$bytes+=$size;$count++;if($p.EndsWith('\\hf')){$files+=@{path=$entry.Substring($p.Length+1);bytes=$size};if($files.Count -gt 10000){throw 'HF inventory bound exceeded'}}}}}
@{root=$p;exists=$exists;safe=$true;reparsePoints=0;bytes=$bytes;fileCount=$count;files=@($files)}|ConvertTo-Json -Depth 5 -Compress`);
}
export async function volume() {
  return powershell("$ErrorActionPreference='Stop';$v=Get-Volume -DriveLetter D;@{drive='D';size=[long]$v.Size;free=[long]$v.SizeRemaining}|ConvertTo-Json -Compress");
}
export async function actionsApi(token, route) {
  check(token, 'Actions read token unavailable');
  const response = await fetch(`https://api.github.com/repos/SceneWorks/SceneWorks/${route}`, { headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(30000) });
  check(response.ok, `Actions identity query failed HTTP${response.status}`);
  return response.json();
}
export async function stopped(binding, api) {
  const latest = await api(`actions/runs/${binding.run}`), attempt = await api(`actions/runs/${binding.run}/attempts/1`);
  const jobs = [];
  for (let page = 1; page <= 10; page++) {
    const result = await api(`actions/runs/${binding.run}/attempts/1/jobs?per_page=100&page=${page}`);
    check(Array.isArray(result.jobs) && Number.isInteger(result.total_count), 'incomplete Actions jobs response');
    jobs.push(...result.jobs);
    if (jobs.length === result.total_count) { validateStopped(binding, latest, attempt, jobs); return { run: binding.run, attempt: 1, head: latest.head_sha, status: latest.status, jobs: jobs.map(({ id, status, conclusion }) => ({ id, status, conclusion })) }; }
    check(result.jobs.length === 100, 'truncated Actions jobs response');
  }
  throw Error('Actions jobs page bound exceeded');
}
export async function maintain(env, io) {
  const root = destination(env), target = path.join(root, 'target');
  const receipt = { version: 1, kind: 'sc-26752-stopped-run-disk-recovery', currentRun: env.GITHUB_RUN_ID, currentSource: env.MOSS_SOURCE_SHA, before: await io.volume(), candidates: [], operations: [] };
  validateTree(await io.inspect(root), root);
  for (const binding of OWNED_RUNS) {
    const stop = await io.stopped(binding), hf = path.join(binding.root, 'hf');
    const inventory = validateTree(await io.inspect(hf), hf);
    let cache = null, cacheProbeError = null;
    if (binding.run === '40155160639') {
      try { cache = validateTree(await io.inspect(path.join(binding.root, 'target')), path.join(binding.root, 'target')); }
      catch (error) { cacheProbeError = String(error.message).slice(0, 200); } // Optional seed stays untouched.
    }
    if (inventory.exists || cache?.exists) validateRuntime(binding, await io.read(path.join(binding.root, 'runtime.json')));
    receipt.candidates.push({ ...binding, stop, inventory, cache, cacheProbeError });
  }
  // The full disk may prevent any local pre-write. Emit the bounded before inventory into the
  // retained Actions log first; the same receipt is saved outside HF once space is reclaimed.
  io.report({ ...receipt, stage: 'validated-before-mutation' });
  for (const candidate of receipt.candidates) {
    if (!candidate.inventory.exists) continue;
    await io.stopped(candidate);
    validateRuntime(candidate, await io.read(path.join(candidate.root, 'runtime.json')));
    const leaf = path.join(candidate.root, 'hf');
    validateTree(await io.inspect(leaf), leaf);
    await io.remove(leaf); // Exact fixed HF leaf only; no root/cache/model-source wildcard.
    receipt.operations.push({ kind: 'remove-owned-hf', path: leaf, after: await io.volume() });
    io.report({ stage: 'hf-reclaimed', operation: receipt.operations.at(-1) });
  }
  const original = receipt.candidates.find(item => item.run === '40155160639'), from = path.join(original.root, 'target');
  if (original.cache?.exists) {
    await io.stopped(original);
    validateRuntime(original, await io.read(path.join(original.root, 'runtime.json')));
    const cacheInventory = validateTree(await io.inspect(from), from);
    const dest = validateTree(await io.inspect(target), target);
    const freeBeforeCopy = await io.volume();
    const completeProjections = receipt.candidates.filter(item => item.run !== original.run && item.inventory.exists);
    const projectionReserveBytes = Math.max(0, ...completeProjections.map(item => item.inventory.bytes));
    const rebuildReserveBytes = 4 * 1024 ** 3;
    const requiredFreeBytes = cacheInventory.bytes + projectionReserveBytes + rebuildReserveBytes;
    const budget = { cacheBytes: cacheInventory.bytes, freeBeforeCopy, projectionReserveBytes, rebuildReserveBytes, requiredFreeBytes };
    if (dest.exists) receipt.cache = { state: 'destination-exists-original-cache-retained', from, to: target, ...budget };
    else if (!completeProjections.length || !Number.isSafeInteger(freeBeforeCopy.free) || freeBeforeCopy.free < requiredFreeBytes) {
      receipt.cache = { state: 'insufficient-proved-copy-headroom-original-cache-retained', from, to: target, ...budget };
    } else {
      const originalBinarySha256 = sha(await io.read(path.join(from, 'release', 'sceneworks-rust-api.exe')));
      await io.ensure(root);
      validateTree(await io.inspect(root), root);
      check(!(await io.inspect(target)).exists, 'destination appeared before cache copy');
      await io.copy(from, target);
      const copied = validateTree(await io.inspect(target), target);
      check(copied.bytes === cacheInventory.bytes && copied.fileCount === cacheInventory.fileCount, 'copied cache inventory differs');
      check(sha(await io.read(path.join(target, 'release', 'sceneworks-rust-api.exe'))) === originalBinarySha256, 'copied cache binary differs');
      receipt.cache = { kind: 'copied-build-cache-not-new-source-certification', from, to: target, ...budget, afterCopy: await io.volume(), originalRun: original.run, originalSource: original.source, originalInference: original.inference, originalRuntimeSha256: original.runtimeSha, originalBinarySha256, originalCacheRetained: true, requiredNextStep: 'normal cargo build at the current ref and actual binary inventory; never relabel this cache receipt' };
    }
  }
  if (original.cacheProbeError) receipt.cache = { state: 'unproved-cache-retained-without-copy', error: original.cacheProbeError };
  receipt.after = await io.volume();
  io.report({ ...receipt, stage: 'completed' });
  await io.ensure(root);
  await io.write(path.join(root, 'windows-maintenance.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  return receipt;
}
if (process.argv[2] === '--moss-disk-maintenance') {
  check(process.platform === 'win32', 'disk maintenance is Windows-only');
  const token = process.env.MOSS_MAINTENANCE_TOKEN;
  try {
    await maintain(process.env, { volume, inspect: inspectTree, stopped: binding => stopped(binding, route => actionsApi(token, route)), read: readFile,
      remove: leaf => rm(leaf, { recursive: true, force: false }), copy: async (from, to) => { await mkdir(to); await cp(from, to, { recursive: true, dereference: false, force: false, errorOnExist: true }); }, ensure: root => mkdir(root, { recursive: true }),
      write: (file, bytes) => writeFile(file, bytes, { flag: 'wx' }), report: value => console.log(JSON.stringify(value)) });
  } catch (error) { console.error(JSON.stringify({ status: 'STOP', stage: 'owned-disk-maintenance', message: error.message })); process.exitCode = 1; }
}
