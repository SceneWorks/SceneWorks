import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { win32 as path } from 'node:path';
import { OWNED_RUNS, R9_DIAGNOSTIC, diagnoseR9, destination, validateRuntime, validateTree, validateStopped, stopped, maintain } from './ci/moss-owned-cache-maintenance.mjs';
const runtimes = await Promise.all([4, 5, 6].map(async round => gunzipSync(await readFile(new URL(`./fixtures/moss-stopped-runtime/r${round}.json.gz`, import.meta.url)))));
const env = { GITHUB_REPOSITORY: 'SceneWorks/SceneWorks', GITHUB_RUN_ID: '99999999999', GITHUB_RUN_ATTEMPT: '1', GITHUB_WORKSPACE: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', MOSS_ROOT: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-99999999999-1-cuda', MOSS_SOURCE_SHA: 'a'.repeat(40), MOSS_INFERENCE_SHA: 'b'.repeat(40) };
const state = binding => ({ id: Number(binding.run), run_attempt: 1, head_sha: binding.source, status: 'completed', conclusion: 'failure' });
const jobs = binding => [{ id: 1, run_id: Number(binding.run), status: 'completed', conclusion: 'failure' }];
const tree = (root, exists = true) => ({ root, exists, safe: true, reparsePoints: 0, bytes: exists ? 100 : 0, fileCount: exists ? 2 : 0, files: [] });
function fixture({ mutateRuntime, mutateInspect, activeAtMutation = false, destinationExists = false, freeBytes = 8 * 1024 ** 3 } = {}) {
  const roots = new Set(OWNED_RUNS.map(item => path.join(item.root, 'hf')));
  const from = path.join(OWNED_RUNS[2].root, 'target'), to = path.join(env.MOSS_ROOT, 'target');
  roots.add(from); if (destinationExists) roots.add(to);
  const mutations = [], stops = new Map(), reports = [], writes = [];
  const io = {
    volume: async () => ({ drive: 'D', free: freeBytes + mutations.length * 100, size: 16 * 1024 ** 3 }),
    inspect: async root => mutateInspect?.(root) ?? tree(root, roots.has(root)),
    stopped: async binding => {
      const count = (stops.get(binding.run) ?? 0) + 1; stops.set(binding.run, count);
      const latest = state(binding); if (activeAtMutation && count > 1) latest.status = 'in_progress';
      validateStopped(binding, latest, state(binding), jobs(binding)); return { run: binding.run, status: 'completed' };
    },
    read: async file => {
      if(file===path.join(R9_DIAGNOSTIC.root,'runtime.json'))throw Object.assign(Error('absent owned fixture'),{code:'ENOENT'});
      const index = OWNED_RUNS.findIndex(binding => path.join(binding.root, 'runtime.json') === file);
      if (index >= 0) return mutateRuntime?.(runtimes[index], index) ?? runtimes[index];
      assert.ok([from, to].some(root => file === path.join(root, 'release', 'sceneworks-rust-api.exe'))); return Buffer.from('original R6 binary fixture');
    },
    remove: async leaf => { assert.ok(OWNED_RUNS.some(item => leaf === path.join(item.root, 'hf'))); roots.delete(leaf); mutations.push(['remove', leaf]); },
    copy: async (source, target) => { assert.equal(source, from); assert.equal(target, to); assert.ok(!roots.has(target)); roots.add(target); mutations.push(['copy', source, target]); },
    ensure: async root => { assert.equal(root, env.MOSS_ROOT); roots.add(root); },
    write: async (file, bytes) => { assert.equal(file, path.join(env.MOSS_ROOT, 'windows-maintenance.json')); writes.push(JSON.parse(bytes)); },
    report: value => reports.push(structuredClone(value)),
  };
  return { io, mutations, reports, writes, roots, from, to };
}
test('actual immutable R4/R5/R6 runtime bytes bind only the stopped HF leaves', () => {
  assert.deepEqual(OWNED_RUNS.map(item => item.run), ['40145054879', '40152001538', '40155160639']);
  for (const [index, binding] of OWNED_RUNS.entries()) {
    assert.equal(validateRuntime(binding, runtimes[index]).candidateId, binding.candidate);
    assert.throws(() => validateRuntime(binding, Buffer.concat([runtimes[index], Buffer.from(' ')])), /runtime bytes/);
    assert.throws(() => validateRuntime({ ...binding, source: '0'.repeat(40) }, runtimes[index]), /identity/);
    assert.throws(() => validateRuntime({ ...binding, root: 'E:\\original-models' }, runtimes[index]), /root/);
  }
});
test('current destination rejects E:, unrelated workspace, stale run and changed run root', () => {
  assert.equal(destination(env), env.MOSS_ROOT);
  for (const changes of [{ GITHUB_WORKSPACE: 'E:\\models' }, { MOSS_ROOT: OWNED_RUNS[0].root }, { GITHUB_RUN_ID: OWNED_RUNS[2].run }, { GITHUB_REPOSITORY: 'other/repo' }]) assert.throws(() => destination({ ...env, ...changes }));
});
test('terminal run, exact head/attempt and every paginated job are mandatory', async () => {
  const binding = OWNED_RUNS[0];
  for (const changes of [{ status: 'in_progress' }, { head_sha: 'f'.repeat(40) }, { run_attempt: 2 }, { id: 9 }, { conclusion: null }]) assert.throws(() => validateStopped(binding, { ...state(binding), ...changes }, state(binding), jobs(binding)));
  for (const changed of [[], [{ ...jobs(binding)[0], status: 'queued' }], [{ ...jobs(binding)[0], run_id: 9 }]]) assert.throws(() => validateStopped(binding, state(binding), state(binding), changed));
  const hundred = Array.from({ length: 100 }, (_, id) => ({ ...jobs(binding)[0], id }));
  let pages = 0;
  const receipt = await stopped(binding, async route => route.includes('/jobs?') ? (++pages === 1 ? { jobs: hundred, total_count: 101 } : { jobs: [{ ...jobs(binding)[0], id: 100 }], total_count: 101 }) : state(binding));
  assert.equal(receipt.jobs.length, 101); assert.equal(pages, 2);
  await assert.rejects(() => stopped(binding, async route => route.includes('/jobs?') ? { jobs: [], total_count: 1 } : state(binding)), /truncated/);
});
test('E: redirects, reparse ancestry/descendants and unproved inventories fail closed', () => {
  const root = path.join(OWNED_RUNS[0].root, 'hf');
  for (const changes of [{ root: 'E:\\originals' }, { safe: false }, { reparsePoints: 1 }, { bytes: -1 }, { bytes: Infinity }, { exists: undefined }]) assert.throws(() => validateTree({ ...tree(root), ...changes }, root));
});
test('recovery removes exactly three HF leaves, copies cache only with headroom and retains original paths, retains original identity', async () => {
  const f = fixture(), receipt = await maintain(env, f.io);
  assert.deepEqual(f.mutations.map(item => item[0]), ['remove', 'remove', 'remove', 'copy']);
  assert.equal(f.reports[0].stage, 'validated-before-mutation'); assert.equal(f.reports[0].operations.length, 0);
  assert.equal(receipt.cache.originalSource, OWNED_RUNS[2].source); assert.equal(receipt.cache.originalInference, OWNED_RUNS[2].inference);
  assert.equal(receipt.cache.originalRuntimeSha256, OWNED_RUNS[2].runtimeSha); assert.match(receipt.cache.originalBinarySha256, /^[a-f0-9]{64}$/);
  assert.match(receipt.cache.kind, /not-new-source-certification/); assert.equal(f.writes.length, 1);
  assert.equal(f.roots.has(f.from), true); assert.equal(f.roots.has(f.to), true);
});
test('all guards precede deletion; a new active attempt between planning and removal is refused', async () => {
  for (const options of [
    { mutateRuntime: (bytes, index) => index === 2 ? Buffer.from('{}') : bytes },
    { mutateInspect: root => root === path.join(OWNED_RUNS[1].root, 'hf') ? { ...tree(root), reparsePoints: 1 } : undefined },
    { activeAtMutation: true },
  ]) {
    const f = fixture(options); await assert.rejects(() => maintain(env, f.io)); assert.deepEqual(f.mutations, []);
  }
});
test('an existing destination never overwrites either cache', async () => {
  const f = fixture({ destinationExists: true }), receipt = await maintain(env, f.io);
  assert.equal(f.mutations.some(item => item[0] === 'copy'), false); assert.equal(f.roots.has(f.from), true);
  assert.equal(receipt.cache.state, 'destination-exists-original-cache-retained');
});
test('an unsafe optional cache is left untouched without blocking exact HF reclamation', async () => {
  const f = fixture({ mutateInspect: root => root === path.join(OWNED_RUNS[2].root, 'target') ? { ...tree(root), reparsePoints: 1 } : undefined });
  const receipt = await maintain(env, f.io);
  assert.equal(f.mutations.filter(item => item[0] === 'remove').length, 3);
  assert.equal(f.mutations.some(item => item[0] === 'copy'), false); assert.equal(f.roots.has(f.from), true);
  assert.equal(receipt.cache.state, 'unproved-cache-retained-without-copy');
});
test('insufficient free capacity skips cache copying and preserves original cache', async () => {
  const f = fixture({ freeBytes: 0 }), receipt = await maintain(env, f.io);
  assert.equal(f.mutations.some(item => item[0] === 'copy'), false);
  assert.equal(f.roots.has(f.from), true); assert.equal(f.roots.has(f.to), false);
  assert.equal(receipt.cache.state, 'insufficient-proved-copy-headroom-original-cache-retained');
  assert.ok(receipt.cache.requiredFreeBytes > receipt.cache.freeBeforeCopy.free);
});
test('action-free recovery gates native job preparation and keeps the normal fresh build', async () => {
  const workflow = await readFile(new URL('../.github/workflows/moss-edit-terminal.yml', import.meta.url), 'utf8');
  const recovery = workflow.slice(workflow.indexOf('  windows-maintenance:'), workflow.indexOf('  build-native:'));
  const build = workflow.slice(workflow.indexOf('  build-native:')), step = recovery.slice(recovery.indexOf('    steps:'));
  assert.match(recovery, /^    if: inputs.phase == 'execute'$/m);
  assert.match(recovery, /^    runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]$/m);
  assert.doesNotMatch(recovery, /\buses:/); assert.doesNotMatch(recovery, /\bneeds:/);
  assert.match(build, /^    needs: windows-maintenance$/m);
  assert.doesNotMatch(build, /--moss-disk-maintenance/);
  assert.match(step, /contents\/scripts\/ci\/moss-owned-cache-maintenance.mjs\?ref=\$env:MOSS_SOURCE_SHA/);
  assert.match(step, /externals\\node24\\bin\\node.exe/); assert.match(step, /\$program \| & \$node --input-type=module - --moss-disk-maintenance/);
  assert.doesNotMatch(step, /WriteAllText|Set-Content|npm |cargo |nvidia-smi|Remove-Item/);
  assert.match(build, /node source\\scripts\\moss-edit-terminal-actions.mjs build/);
  const helper = await readFile(new URL('./ci/moss-owned-cache-maintenance.mjs', import.meta.url), 'utf8');
  assert.match(helper, /await mkdir\(to\); await cp\(from, to, \{ recursive: true, dereference: false, force: false, errorOnExist: true \}\)/);
  assert.doesNotMatch(helper, /\brename\b/);
  assert.ok(helper.indexOf('GetAttributes($entry)') < helper.indexOf('$stack.Push($entry)'));
  assert.match(helper, /Reparse ancestor/); assert.match(helper, /Reparse descendant/);

});

test('fixed stopped R9 observation captures actual runtime bytes and HF capacity without authorizing mutation',async()=>{
  const runtime={candidateId:R9_DIAGNOSTIC.candidate,sourceSha:R9_DIAGNOSTIC.source,inferenceSha:R9_DIAGNOSTIC.inference,backend:'cuda',root:R9_DIAGNOSTIC.root,hfHome:path.join(R9_DIAGNOSTIC.root,'hf'),sourceRoot:path.join(R9_DIAGNOSTIC.workspace,'source'),packageSource:'E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9'},bytes=Buffer.from(JSON.stringify(runtime));
  const calls=[],io={stopped:async binding=>{assert.equal(binding,R9_DIAGNOSTIC);return {status:'completed'};},inspect:async file=>{calls.push(file);return tree(file);},read:async file=>{assert.equal(file,path.join(R9_DIAGNOSTIC.root,'runtime.json'));return bytes;}};
  const proof=await diagnoseR9(io);assert.equal(proof.readOnly,true);assert.equal(proof.runtime.utf8,bytes.toString('utf8'));assert.equal(proof.runtime.sha256.length,64);assert.equal(proof.inventory.bytes,100);assert.equal(OWNED_RUNS.some(item=>item.run===R9_DIAGNOSTIC.run),false);assert.equal(calls.length,2);
  await assert.rejects(()=>diagnoseR9({...io,read:async()=>Buffer.from(JSON.stringify({...runtime,sourceSha:'0'.repeat(40)}))}),/identity/);
  await assert.rejects(()=>diagnoseR9({...io,inspect:async file=>({...tree(file),reparsePoints:1})}),/reparse/);
  await assert.rejects(()=>diagnoseR9({...io,stopped:async()=>{throw Error('active');}}),/active/);
});
