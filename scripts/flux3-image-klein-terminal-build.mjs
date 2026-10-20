// Explicit FUTURE build producer; importing/planning never builds. No credentials, GPU or network.
import assert from 'node:assert/strict';
import { execFile as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import { copyFile, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomBytes, createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { contained, fileHash, safeEnvironment } from './flux3-image-klein-terminal-session.mjs';
const execFile = promisify(execCb), rootDefault = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifests = ['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'];
export function buildRecipe(backend, platform, role = 'api-worker') {
  assert((backend === 'candle' && platform === 'win32') || (backend === 'mlx' && platform === 'darwin'));
  assert(['api-worker', 'resource-adapter'].includes(role), 'only fixed known owned binary roles may be built');
  const target = backend === 'candle' ? 'x86_64-pc-windows-msvc' : 'aarch64-apple-darwin';
  return { role, package: role === 'api-worker' ? 'sceneworks-rust-api' : 'sceneworks-memory-adapter', bin: role === 'api-worker' ? 'sceneworks-rust-api' : `memory-${backend}-adapter`, profile: 'release', target, features: role === 'api-worker' ? backend === 'candle' ? ['backend-candle'] : [] : [backend], jobs: 2, offline: true };
}
async function sourceObservation(root, expectedSource, inference, io) {
  assert.equal((await io.exec('git', ['rev-parse','HEAD'], { cwd: root })).stdout.trim(), expectedSource, 'observed build HEAD differs from approved source');
  assert.equal((await io.exec('git', ['status','--porcelain'], { cwd: root })).stdout.trim(), '', 'build source must be clean before/after');
  const texts = await Promise.all(manifests.map(f => io.readFile(path.join(root, f), 'utf8'))), lock = await io.readFile(path.join(root, 'Cargo.lock'), 'utf8'); assertPins(texts, lock, inference);
  return { sourceSha: expectedSource, cargoLockSha256: hash(lock), manifestSha256: Object.fromEntries(manifests.map((f,i) => [f, hash(texts[i])])) };
}
export async function produceTerminalBuild(a, dependencies = {}) {
  assert(a.actualBuild === true && a.userBuildApprovalId && a.resourceAdmissionId, 'separate explicit user build/resource admission required; runtime never auto-builds');
  const io = { exec: execFile, readFile, copyFile, mkdir, writeFile, hash: fileHash, platform: process.platform, env: process.env, root: rootDefault, ...dependencies }, recipe = buildRecipe(a.backend, io.platform, a.binaryRole ?? 'api-worker');
  assert(/^[a-f0-9]{40}$/.test(a.sourceSha) && /^[a-f0-9]{40}$/.test(a.inferenceMainSha));
  for (const field of ['cargoExecutable','rustcExecutable','targetDir','buildOutputRoot','inferenceRepository']) assert(path.isAbsolute(a[field] ?? ''), `absolute admitted ${field} required`);
  for (const directory of [a.buildOutputRoot, a.targetDir]) { const relative = path.relative(io.root, directory); assert(relative.startsWith('..') || path.isAbsolute(relative), 'sealed output/admitted warm target must remain outside checkout'); }
  if (a.backend === 'candle') { assert.equal(a.msvcVersion, '14.44'); assert(path.isAbsolute(a.msvcCompilerExecutable ?? '')); assert.equal(await io.hash(a.msvcCompilerExecutable), a.msvcCompilerSha256); }
  const before = await sourceObservation(io.root, a.sourceSha, a.inferenceMainSha, io); assert.equal(before.cargoLockSha256, a.cargoLockSha256);
  await io.exec('git', ['merge-base','--is-ancestor',a.inferenceMainSha,'refs/remotes/origin/main'], { cwd: a.inferenceRepository });
  assert.equal(await io.hash(a.cargoExecutable), a.cargoExecutableSha256); assert.equal(await io.hash(a.rustcExecutable), a.rustcExecutableSha256);
  const directory = path.join(a.buildOutputRoot, `producer-${randomBytes(24).toString('hex')}`); await io.mkdir(directory);
  const temp = path.join(directory, 'temp'); await io.mkdir(temp);
  const env = { ...safeEnvironment(io.env), CUDA_VISIBLE_DEVICES: '-1', CARGO_NET_OFFLINE: 'true', RUSTFLAGS: a.rustflags ?? '', RUSTC: a.rustcExecutable, TEMP: temp, TMP: temp, TMPDIR: temp };
  delete env.RUST_TEST_THREADS;
  assert(!env.CARGO_ENCODED_RUSTFLAGS && !env.RUSTC_WRAPPER && !env.RUSTC_WORKSPACE_WRAPPER, 'unadmitted compiler wrapper/encoded flags refused; request a clean explicitly admitted build environment');
  let msvcObserved = null, cudaObserved = null;
  if (a.backend === 'candle') {
    assert(path.isAbsolute(a.cudaCompilerExecutable ?? '') && /^[a-f0-9]{64}$/.test(a.cudaCompilerSha256 ?? ''), 'actual admitted CUDA 12.9 compiler identity required');
    assert.equal(await io.hash(a.cudaCompilerExecutable), a.cudaCompilerSha256);
    const cudaVersion = (await io.exec(a.cudaCompilerExecutable, ['--version'], { env, timeout: 30000 })).stdout;
    assert(/release 12\.9(?:\D|$)/.test(cudaVersion), 'actual CUDA compiler must be 12.9');
    cudaObserved = { compilerSha256: a.cudaCompilerSha256, versionOutputSha256: hash(cudaVersion) };
    env.CUDA_PATH = path.dirname(path.dirname(a.cudaCompilerExecutable)); env.CXX = a.msvcCompilerExecutable; env.NVCC_CCBIN = path.dirname(a.msvcCompilerExecutable);
    env.PATH = [path.dirname(a.msvcCompilerExecutable), path.dirname(a.cudaCompilerExecutable), env.PATH ?? ''].join(';');
  }
  if (a.backend === 'candle') { let result; try { result = await io.exec(a.msvcCompilerExecutable, ['/Bv'], { env, timeout: 30000 }); } catch (error) { result = error; } const text = `${result.stdout ?? ''}\n${result.stderr ?? ''}`; assert(/Version\s+19\.44\./i.test(text), 'actual MSVC compiler must be 14.44/19.44, not a caller version label'); msvcObserved = { compilerSha256: a.msvcCompilerSha256, versionOutputSha256: hash(text) }; }
  const rustc = (await io.exec(a.rustcExecutable, ['--version','--verbose'], { env, timeout: 30000 })).stdout; assert(rustc.includes(`host: ${recipe.target}`), 'actual compiler host differs from admitted platform');
  const metadataCommand = ['metadata','--locked','--offline','--format-version=1'];
  const metadataText = (await io.exec(a.cargoExecutable, metadataCommand, { cwd: io.root, env, timeout: 120000, maxBuffer: 64 * 1024 * 1024 })).stdout, metadata = JSON.parse(metadataText);
  const edges = metadata.packages.filter(p => /github\.com\/SceneWorks\/inference(?:\.git)?[?#]/i.test(p.source ?? '')).map(p => ({ name: p.name, source: p.source }));
  assert(edges.length && edges.every(p => p.source.endsWith(`#${a.inferenceMainSha}`)), 'mixed/unresolved inference crate edges refused');
  const command = ['build','--locked','--offline','--release','--jobs','2','--package',recipe.package,'--bin',recipe.bin,'--target',recipe.target,'--target-dir',a.targetDir,'--message-format=json-render-diagnostics', ...(recipe.features.length ? ['--features',recipe.features.join(',')] : [])];
  const result = await io.exec(a.cargoExecutable, command, { cwd: io.root, env, timeout: a.timeoutMs ?? 10800000, maxBuffer: 64 * 1024 * 1024 });
  const artifactPath = path.join(a.targetDir, recipe.target, 'release', `${recipe.bin}${a.backend === 'candle' ? '.exe' : ''}`);
  const messages = result.stdout.split(/\r?\n/).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
  const artifact = messages.find(m => m.reason === 'compiler-artifact' && m.target?.name === recipe.bin && m.target?.kind?.includes('bin') && m.executable && path.resolve(m.executable) === path.resolve(artifactPath)); assert(artifact, 'successful cargo process must report the ACTUAL executable artifact, not an older arbitrary file');
  assert(messages.some(m => m.reason === 'build-finished' && m.success === true));
  const after = await sourceObservation(io.root, a.sourceSha, a.inferenceMainSha, io); assert.deepEqual(after, before, 'source/manifests/lock changed during build');
  const sealedBinary = path.join(directory, path.basename(artifactPath)); await io.copyFile(artifactPath, sealedBinary); const binarySha256 = await io.hash(sealedBinary); assert.equal(binarySha256, await io.hash(artifactPath));
  const transcript = { schemaVersion: 1, kind: 'flux3-owned-build-producer-transcript', sourceBefore: before, sourceAfter: after, inferenceMainSha: a.inferenceMainSha, inferenceEdges: edges, metadataSha256: hash(metadataText), recipe, command, cargoExecutableSha256: a.cargoExecutableSha256, rustcExecutableSha256: a.rustcExecutableSha256, rustcSha256: hash(rustc), rustflags: a.rustflags ?? '', msvcVersion: a.msvcVersion ?? null, msvcObserved, cudaObserved, temp, targetDir: a.targetDir, cudaVisibleDevices: '-1', artifact, buildFinished: true, buildStdoutSha256: hash(result.stdout), buildStderrSha256: hash(result.stderr ?? '') };
  await io.writeFile(path.join(directory, 'build-stdout.log'), result.stdout, { flag: 'wx' });
  await io.writeFile(path.join(directory, 'build-stderr.log'), result.stderr ?? '', { flag: 'wx' });
  await io.writeFile(path.join(directory, 'producer-transcript.json'), JSON.stringify(transcript) + '\n', { flag: 'wx' });
  const receipt = { schemaVersion: 1, kind: 'flux3-owned-build-producer-receipt', producerScriptSha256: await io.hash(fileURLToPath(import.meta.url)), userBuildApprovalId: a.userBuildApprovalId, resourceAdmissionId: a.resourceAdmissionId, sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, backend: a.backend, cargoLockSha256: before.cargoLockSha256, recipe, binary: { path: sealedBinary, sha256: binarySha256 }, transcriptSha256: hash(JSON.stringify(transcript) + '\n') };
  const receiptPath = path.join(directory, 'producer-receipt.json'); await io.writeFile(receiptPath, JSON.stringify(receipt) + '\n', { flag: 'wx' }); return { receiptPath, receiptSha256: hash(JSON.stringify(receipt) + '\n'), receipt };
}
export async function verifyTerminalBuild(a, { sourceRoot = rootDefault, role = 'api-worker' } = {}) {
  const recipe = buildRecipe(a.backend, a.backend === 'candle' ? 'win32' : 'darwin', role);
  const binding = (role === 'api-worker' ? a.approvedPlan?.buildReceipts : a.approvedPlan?.resourceBuildReceipts)?.[a.backend]; assert(binding, 'parent-approved concrete producer receipt required; binarySourceSha labels are not build proof');
  const file = await contained(a.binaryBuildReceipt.path, a.binaryBuildReceipt.root), bytes = await readFile(file); assert.equal(hash(bytes), binding.receiptSha256); assert.equal(hash(bytes), a.binaryBuildReceipt.sha256);
  const receipt = JSON.parse(bytes); assert.equal(receipt.kind, 'flux3-owned-build-producer-receipt'); assert.equal(receipt.schemaVersion, 1); assert.equal(receipt.sourceSha, a.sourceSha); assert.equal(receipt.inferenceMainSha, a.inferenceMainSha); assert.equal(receipt.backend, a.backend); assert(receipt.userBuildApprovalId && receipt.resourceAdmissionId);
  assert.equal(receipt.producerScriptSha256, await fileHash(fileURLToPath(import.meta.url)), 'producer source changed since this observed build'); assert.equal(receipt.cargoLockSha256, binding.cargoLockSha256); assert.equal(hash(await readFile(path.join(sourceRoot, 'Cargo.lock'))), receipt.cargoLockSha256, 'current lock differs from actually built lock');
  const directory = path.dirname(file), transcriptFile = await contained(path.join(directory, 'producer-transcript.json'), directory), transcriptBytes = await readFile(transcriptFile); assert.equal(hash(transcriptBytes), receipt.transcriptSha256); assert.equal(hash(transcriptBytes), binding.transcriptSha256);
  const t = JSON.parse(transcriptBytes); assert.equal(t.kind, 'flux3-owned-build-producer-transcript'); assert.equal(t.sourceBefore.sourceSha, a.sourceSha); assert.equal(t.sourceBefore.cargoLockSha256, receipt.cargoLockSha256); assert.deepEqual(t.sourceBefore, t.sourceAfter); assert.equal(t.inferenceMainSha, a.inferenceMainSha); assert(t.inferenceEdges.length && t.inferenceEdges.every(e => e.source.endsWith(`#${a.inferenceMainSha}`))); assert.equal(t.buildFinished, true); assert.equal(t.cudaVisibleDevices, '-1');
  for (const [file, expected] of [['build-stdout.log', t.buildStdoutSha256], ['build-stderr.log', t.buildStderrSha256]]) assert.equal(await fileHash(await contained(path.join(directory, file), directory)), expected, 'actual retained build transcript bytes changed');
  assert.deepEqual(t.recipe, recipe, 'sealed build role/package/features differ from actual executable role'); assert.deepEqual(receipt.recipe, t.recipe);
  assert.deepEqual(t.command, ['build','--locked','--offline','--release','--jobs','2','--package',t.recipe.package,'--bin',t.recipe.bin,'--target',t.recipe.target,'--target-dir',t.targetDir,'--message-format=json-render-diagnostics', ...(t.recipe.features.length ? ['--features',t.recipe.features.join(',')] : [])]);
  assert.equal(hash(t.rustflags), binding.rustflagsSha256); assert.equal(t.cargoExecutableSha256, binding.cargoExecutableSha256); assert.equal(t.rustcExecutableSha256, binding.rustcExecutableSha256);
  if (a.backend === 'candle') { assert.equal(t.msvcVersion, '14.44'); assert.equal(t.msvcObserved?.compilerSha256, binding.msvcCompilerSha256); assert(/^[a-f0-9]{64}$/.test(t.msvcObserved.versionOutputSha256)); assert.equal(t.cudaObserved?.compilerSha256, binding.cudaCompilerSha256); assert(/^[a-f0-9]{64}$/.test(t.cudaObserved.versionOutputSha256)); } assert.equal(t.artifact.reason, 'compiler-artifact'); assert(t.artifact.target.kind.includes('bin') && t.artifact.target.name === t.recipe.bin);
  assert.equal(receipt.binary.sha256, binding.binarySha256); assert.equal(receipt.binary.sha256, a.binarySha256); assert.equal(await realpath(a.binary), await contained(receipt.binary.path, directory), 'launcher must use the owned sealed build output'); assert.equal(await fileHash(a.binary), receipt.binary.sha256);
  return receipt;
}
