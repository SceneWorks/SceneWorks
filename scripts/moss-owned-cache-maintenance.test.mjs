import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, link, symlink, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { win32 as path } from 'node:path';
import { SELECTED_HF, selectedHfEnvelope, selectedArtifact, selectedRuntime, partialBuildRuntime, selectedEntries, selectedProcesses, maintainSelectedHf, OWNED_RUNS, validateHfOnlyEnvelope, copyCacheExclusive, observeHf, hfEntries, validateHfObservation, R9_DIAGNOSTIC, diagnoseR9, destination, validateRuntime, validateTree, validateStopped, stopped, maintain } from './ci/moss-owned-cache-maintenance.mjs';
const runtimes = await Promise.all([4, 5, 6, 10, 12, 13, 14, 15].map(async round => gunzipSync(await readFile(new URL(`./fixtures/moss-stopped-runtime/r${round}.json.gz`, import.meta.url)))));
const env = { GITHUB_REPOSITORY: 'SceneWorks/SceneWorks', GITHUB_RUN_ID: '99999999999', GITHUB_RUN_ATTEMPT: '1', GITHUB_WORKSPACE: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', MOSS_ROOT: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-99999999999-1-cuda', MOSS_SOURCE_SHA: 'a'.repeat(40), MOSS_INFERENCE_SHA: 'b'.repeat(40) };
const state = binding => ({ id: Number(binding.run), run_attempt: 1, head_sha: binding.source, status: 'completed', conclusion: 'failure' });
const jobs = binding => [{ id: 1, run_id: Number(binding.run), status: 'completed', conclusion: 'failure' }];
const tree = (root, exists = true) => ({ root, exists, safe: true, reparsePoints: 0, bytes: exists ? 100 : 0, fileCount: exists ? 2 : 0, files: [] });
function fixture({ mutateRuntime, mutateInspect, activeAtMutation = false, destinationExists = false, freeBytes = 8 * 1024 ** 3, r10Cache = false, r12Cache = false, r12Hf = false, r13Cache = false, r13Hf = false, r14Cache = false, r14Hf = false, r15Hf = false } = {}) {
  const roots = new Set(OWNED_RUNS.filter((item,index)=>(index!==4||r12Cache||r12Hf)&&(index!==5||r13Cache||r13Hf)&&(index!==6||r14Cache||r14Hf)&&(index!==7||r15Hf)).map(item => path.join(item.root, 'hf')));
  const from = path.join(OWNED_RUNS[r14Cache?6:r13Cache?5:r12Cache?4:r10Cache?3:2].root, 'target'), to = path.join(env.MOSS_ROOT, 'target');
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
    metadata: async runtime => ({root:runtime.hfHome,entries:runtime.packageFiles.map((item,index)=>({path:item.file,type:'file',dev:'1',ino:String(9007199254740993n+BigInt(index)),nlink:index?'1':'2',byteLength:'1',sha256:item.sha256}))}),
    hash: async file => {assert.ok([from,to].some(root=>file===path.join(root,'release','sceneworks-rust-api.exe')));return r14Cache?OWNED_RUNS[6].cacheBinarySha256:r13Cache?OWNED_RUNS[5].cacheBinarySha256:r12Cache?OWNED_RUNS[4].cacheBinarySha256:r10Cache?OWNED_RUNS[3].cacheBinarySha256:createHash('sha256').update('original R6 binary fixture').digest('hex');},
    remove: async leaf => { assert.ok(OWNED_RUNS.some(item => leaf === path.join(item.root, 'hf'))); roots.delete(leaf); mutations.push(['remove', leaf]); },
    copy: async (source, target) => { assert.equal(source, from); assert.equal(target, to); assert.ok(!roots.has(target)); roots.add(target); mutations.push(['copy', source, target]); },
    ensure: async root => { assert.equal(root, env.MOSS_ROOT); roots.add(root); },
    write: async (file, bytes) => { assert.equal(file, path.join(env.MOSS_ROOT, 'windows-maintenance.json')); writes.push(JSON.parse(bytes)); },
    report: value => reports.push(structuredClone(value)),
  };
  return { io, mutations, reports, writes, roots, from, to };
}
test('actual immutable R4/R5/R6/R10/R12 runtime bytes bind only the stopped HF leaves', () => {
  assert.deepEqual(OWNED_RUNS.map(item => item.run), ['40145054879', '40152001538', '40155160639', '40163530825', '40167292776', '40169818205', '40174802721', '40181253824']);
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
test('recovery removes exactly four HF leaves, copies cache only with headroom and retains original paths, retains original identity', async () => {
  const f = fixture(), receipt = await maintain(env, f.io);
  assert.deepEqual(f.mutations.map(item => item[0]), ['remove', 'remove', 'remove', 'remove', 'copy']);
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
  assert.equal(f.mutations.filter(item => item[0] === 'remove').length, 4);
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
  assert.match(recovery, /^    if: \(inputs.phase == 'execute' && inputs.native_run_id == ''\) \|\| inputs.phase == 'cuda-rebase'$/m);
  assert.match(build, /inputs.phase == 'execute' && inputs.native_run_id == ''/);
  assert.match(build, /^    if: \(inputs.phase == 'execute' && inputs.native_run_id == ''\) \|\| inputs.phase == 'cuda-rebase'$/m);
  const reused=workflow.slice(workflow.indexOf('  execute-cuda:'),workflow.indexOf('  continue-mlx:'));
  assert.match(reused, /inputs.native_run_id != '' && inputs.mlx_artifact_id != ''/);
  assert.match(reused, /execute-reused-cuda/);
  assert.doesNotMatch(reused, /--moss-disk-maintenance/);
  assert.match(recovery, /^    runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]$/m);
  assert.doesNotMatch(recovery, /\buses:/); assert.doesNotMatch(recovery, /\bneeds:/);
  assert.match(build, /^    needs: windows-maintenance$/m);
  assert.doesNotMatch(build, /--moss-disk-maintenance/);
  assert.match(step, /contents\/scripts\/ci\/moss-owned-cache-maintenance.mjs\?ref=\$env:MOSS_SOURCE_SHA/);
  assert.match(step, /externals\\node24\\bin\\node.exe/); assert.match(step, /\$program \| & \$node --input-type=module - --moss-disk-maintenance/);
  assert.doesNotMatch(step, /WriteAllText|Set-Content|npm |cargo |nvidia-smi|Remove-Item/);
  assert.match(build, /node source\\scripts\\moss-edit-terminal-actions.mjs build/);
  const helper = await readFile(new URL('./ci/moss-owned-cache-maintenance.mjs', import.meta.url), 'utf8');
  assert.match(helper, /copy: copyCacheExclusive/);
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

test('fixed actual R10 runtime binds durable HF only and retains precleanup metadata before removal',async()=>{
  const binding=OWNED_RUNS[3],runtime=validateRuntime(binding,runtimes[3]);assert.equal(runtimes[3].length,71286);assert.equal(binding.runtimeSha,'546808654e2a0d44d26a5d594b9e7c20961b2b9d947e900effdefdf9f8c2d625');
  assert.equal(runtime.hfHome,path.join(binding.root,'hf'));assert.equal(binding.metadataRequired,true);
  const f=fixture(),remove=f.io.remove;f.io.remove=async leaf=>{if(leaf===runtime.hfHome){const proof=f.reports.find(item=>item.stage==='fixed-r10-readonly-metadata-before-hf-cleanup');assert.ok(proof);assert.equal(proof.runtimeSha256,binding.runtimeSha);assert.equal(proof.metadata.root,leaf);assert.equal(proof.metadata.entries[0].ino,'9007199254740993');}return remove(leaf);};
  await maintain(env,f.io);assert.equal(f.mutations.filter(item=>item[0]==='remove'&&item[1]===runtime.hfHome).length,1);assert.equal(f.roots.has(f.from),true);
  assert.throws(()=>validateRuntime({...binding,root:'E:\\originals'},runtimes[3]),/root/);
  for(const bad of [{root:'E:\\originals',entries:[]},{root:runtime.hfHome,entries:[{path:'../outside',type:'file',dev:'1',ino:'1',nlink:'2',byteLength:'1'}]}]){const invalid=fixture();invalid.io.metadata=async()=>bad;await assert.rejects(()=>maintain(env,invalid.io));assert.deepEqual(invalid.mutations,[]);}
  const changed=fixture();let calls=0;changed.io.metadata=async()=>({root:runtime.hfHome,entries:[{path:'owned',type:'file',dev:'1',ino:String(++calls),nlink:'2',byteLength:'1',sha256:'a'.repeat(64)}]});await assert.rejects(()=>maintain(env,changed.io),/changed before cleanup/);assert.equal(changed.roots.has(runtime.hfHome),true);
  const badRuntime=fixture({mutateRuntime:(bytes,index)=>index===3?Buffer.from('{}'):bytes});await assert.rejects(()=>maintain(env,badRuntime.io));assert.deepEqual(badRuntime.mutations,[]);
});
test('read-only HF observation retains bigint link identities, hashes aliases and never follows a link',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-hf-stat-'));t.after(()=>rm(root,{recursive:true,force:true}));const repo='hub/models--chinoll--Moss-Image-Edit',snapshot=`${repo}/snapshots/${'a'.repeat(40)}/adapter_config.json`,blob=`${repo}/blobs/${'b'.repeat(40)}`,bytes=Buffer.from('metadata-only private blob');
  await mkdir(resolve(root,repo,'snapshots','a'.repeat(40)),{recursive:true});await mkdir(resolve(root,repo,'blobs'));await writeFile(resolve(root,blob),bytes);await link(resolve(root,blob),resolve(root,snapshot));await symlink('/does-not-exist',resolve(root,'unexpected-link'));
  const observation=await observeHf({hfHome:root,editAdapterFiles:[snapshot]}),a=observation.entries.find(item=>item.path===snapshot),b=observation.entries.find(item=>item.path===blob);
  assert.equal(a.nlink,'2');assert.equal(a.ino,b.ino);assert.equal(a.dev,b.dev);assert.equal(a.sha256,createHash('sha256').update(bytes).digest('hex'));assert.equal(a.sha256,b.sha256);assert.equal(observation.entries.find(item=>item.path==='unexpected-link').type,'symlink');assert.deepEqual(await readFile(resolve(root,blob)),bytes);
  await symlink(root,resolve(root,'alias'));await assert.rejects(()=>hfEntries(resolve(root,'alias')),/real directory/);
});

test('fixed R10 cache copy requires original artifact binary SHA and proved complete projection capacity, retains original target',async()=>{
  const f=fixture({r10Cache:true}),receipt=await maintain(env,f.io);assert.equal(receipt.cache.originalRun,OWNED_RUNS[3].run);assert.equal(receipt.cache.originalBinarySha256,OWNED_RUNS[3].cacheBinarySha256);assert.equal(receipt.cache.originalCacheRetained,true);assert.equal(f.roots.has(f.from),true);assert.equal(f.roots.has(f.to),true);assert.ok(receipt.candidates[3].metadata.completeProjection);assert.equal(receipt.cache.requiredFreeBytes,receipt.cache.cacheBytes+receipt.cache.projectionReserveBytes+receipt.cache.rebuildReserveBytes);
  const only=fixture({r10Cache:true});for(const binding of OWNED_RUNS.slice(0,3))only.roots.delete(path.join(binding.root,'hf'));const seeded=await maintain(env,only.io);assert.equal(seeded.cache.originalRun,OWNED_RUNS[3].run);assert.equal(seeded.cache.projectionReserveBytes,100);assert.ok(only.mutations.some(item=>item[0]==='copy'));
  const incomplete=fixture({r10Cache:true});for(const binding of OWNED_RUNS.slice(0,3))incomplete.roots.delete(path.join(binding.root,'hf'));const observe=incomplete.io.metadata;incomplete.io.metadata=async runtime=>{const proof=await observe(runtime);proof.entries.pop();return proof;};const skipped=await maintain(env,incomplete.io);assert.match(skipped.cache.state,/insufficient-proved/);assert.equal(incomplete.mutations.some(item=>item[0]==='copy'),false);
  const wrong=fixture({r10Cache:true});wrong.io.hash=async()=> '0'.repeat(64);await assert.rejects(()=>maintain(env,wrong.io),/source cache binary differs/);assert.equal(wrong.mutations.some(item=>item[0]==='copy'),false);assert.equal(wrong.roots.has(wrong.from),true);
  const low=fixture({r10Cache:true,freeBytes:1});await maintain(env,low.io);assert.equal(low.mutations.some(item=>item[0]==='copy'),false);assert.equal(low.roots.has(low.from),true);
  const copied=fixture({r10Cache:true});copied.io.hash=async file=>file.startsWith(copied.to)?'0'.repeat(64):OWNED_RUNS[3].cacheBinarySha256;await assert.rejects(()=>maintain(env,copied.io),/copied cache binary differs/);
});

test('mixed cache selection never promotes an incomplete R10 or unproved R6 projection',async()=>{
  for(const scenario of ['r6-fallback-r10-incomplete','r6-fallback-r10-target-unsafe','r10-selected-r6-unproved','r10-selected-r6-only']) {
    const r10Selected=scenario.startsWith('r10-selected'),f=fixture({r10Cache:r10Selected});
    for(const binding of OWNED_RUNS.slice(0,2))f.roots.delete(path.join(binding.root,'hf'));
    if(!r10Selected)f.roots.delete(path.join(OWNED_RUNS[2].root,'hf'));
    if(scenario==='r10-selected-r6-only')f.roots.delete(path.join(OWNED_RUNS[3].root,'hf'));
    else {const observe=f.io.metadata;f.io.metadata=async runtime=>{const proof=await observe(runtime);proof.entries.pop();return proof;};}
    if(scenario==='r6-fallback-r10-target-unsafe'){const inspect=f.io.inspect;f.io.inspect=async root=>root===path.join(OWNED_RUNS[3].root,'target')?{...tree(root),reparsePoints:1}:inspect(root);}
    const receipt=await maintain(env,f.io);assert.match(receipt.cache.state,/insufficient-proved/,scenario);assert.equal(receipt.cache.projectionReserveBytes,0,scenario);assert.equal(f.mutations.some(item=>item[0]==='copy'),false,scenario);assert.equal(f.roots.has(f.from),true,scenario);
  }
  // The explicit historical R4 policy can still provide the reserve for either selected target.
  for(const r10Cache of [false,true]) {
    const f=fixture({r10Cache});f.roots.delete(path.join(OWNED_RUNS[1].root,'hf'));f.roots.delete(path.join(OWNED_RUNS[2].root,'hf'));f.roots.delete(path.join(OWNED_RUNS[3].root,'hf'));
    const receipt=await maintain(env,f.io);assert.equal(receipt.cache.projectionReserveBytes,100);assert.ok(f.mutations.some(item=>item[0]==='copy'));
  }
});

test('production native filesystem cache copy creates absent nested destination and retains original',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-copy-native-'));t.after(()=>rm(root,{recursive:true,force:true}));const source=resolve(root,'original'),target=resolve(root,'fresh');
  await mkdir(resolve(source,'release/build/native'),{recursive:true});const files=[['release/sceneworks-rust-api.exe','original binary bytes'],['release/build/native/object.o','original object bytes'],['CACHEDIR.TAG','cache tag']];
  for(const [file,bytes] of files)await writeFile(resolve(source,file),bytes);
  await copyCacheExclusive(source,target);for(const [file,bytes] of files){assert.equal(await readFile(resolve(source,file),'utf8'),bytes);assert.equal(await readFile(resolve(target,file),'utf8'),bytes);}
  assert.deepEqual((await readdir(source)).sort(),(await readdir(target)).sort());
});
test('production native filesystem cache copy refuses existing empty and populated destinations without mutation',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-copy-existing-'));t.after(()=>rm(root,{recursive:true,force:true}));const source=resolve(root,'original');await mkdir(resolve(source,'release'),{recursive:true});await writeFile(resolve(source,'release/api.exe'),'source binary');
  for(const populated of [false,true]) {
    const target=resolve(root,populated?'populated':'empty');await mkdir(target);
    if(populated){await mkdir(resolve(target,'release'));await writeFile(resolve(target,'release/api.exe'),'existing user bytes');await writeFile(resolve(target,'sentinel'),'keep');}
    await assert.rejects(()=>copyCacheExclusive(source,target),/already exists/);
    if(populated){assert.equal(await readFile(resolve(target,'release/api.exe'),'utf8'),'existing user bytes');assert.equal(await readFile(resolve(target,'sentinel'),'utf8'),'keep');}
    else assert.deepEqual(await readdir(target),[]);
    assert.equal(await readFile(resolve(source,'release/api.exe'),'utf8'),'source binary');
  }
  await assert.rejects(()=>copyCacheExclusive(source,source),/already exists/);assert.equal(await readFile(resolve(source,'release/api.exe'),'utf8'),'source binary');
});

test('fixed actual R12 runtime permits only its HF leaf and verified original-preserving cache copy',async()=>{
  const binding=OWNED_RUNS[4],runtime=validateRuntime(binding,runtimes[4]);assert.equal(runtimes[4].length,71286);assert.equal(binding.runtimeSha,'47a25951ea0e1d6fd076802c2ca68421e36d90a296fa1813fc344e6d0a461355');assert.equal(binding.source,'e6475176bf1cf07a46530adbe659aa0d6492c71a');assert.equal(binding.cacheBinarySha256,'07cb4ef31caba2f875d6750c461715f880c79288201af88af11997e31809d69d');assert.equal(runtime.hfHome,path.join(binding.root,'hf'));
  const f=fixture({r12Cache:true}),remove=f.io.remove;f.io.remove=async leaf=>{if(leaf===runtime.hfHome){const proof=f.reports.find(item=>item.stage==='fixed-r12-readonly-metadata-before-hf-cleanup');assert.ok(proof);assert.equal(proof.runtimeSha256,binding.runtimeSha);assert.equal(proof.metadata.completeProjection,true);}return remove(leaf);};
  const receipt=await maintain(env,f.io);assert.equal(receipt.cache.originalRun,binding.run);assert.equal(receipt.cache.originalBinarySha256,binding.cacheBinarySha256);assert.equal(receipt.cache.originalCacheRetained,true);assert.equal(f.roots.has(f.from),true);assert.equal(f.roots.has(f.to),true);assert.equal(f.roots.has(runtime.hfHome),false);
  assert.throws(()=>validateRuntime({...binding,root:'E:\\models'},runtimes[4]),/root/);assert.throws(()=>validateRuntime(binding,Buffer.concat([runtimes[4],Buffer.from(' ')])),/runtime bytes/);
  const bad=fixture({r12Cache:true,mutateRuntime:(bytes,index)=>index===4?Buffer.from('{}'):bytes});await assert.rejects(()=>maintain(env,bad.io));assert.deepEqual(bad.mutations,[]);
  const reparse=fixture({r12Cache:true,mutateInspect:root=>root===runtime.hfHome?{...tree(root),reparsePoints:1}:undefined});await assert.rejects(()=>maintain(env,reparse.io));assert.deepEqual(reparse.mutations,[]);
  const wrong=fixture({r12Cache:true});wrong.io.hash=async()=> '0'.repeat(64);await assert.rejects(()=>maintain(env,wrong.io),/source cache binary differs/);assert.equal(wrong.mutations.some(item=>item[0]==='copy'),false);
  const copied=fixture({r12Cache:true});copied.io.hash=async file=>file.startsWith(copied.to)?'0'.repeat(64):binding.cacheBinarySha256;await assert.rejects(()=>maintain(env,copied.io),/copied cache binary differs/);
  const existing=fixture({r12Cache:true,destinationExists:true});await maintain(env,existing.io);assert.equal(existing.mutations.some(item=>item[0]==='copy'),false);assert.equal(existing.roots.has(existing.from),true);
});
test('R12 completeness is independent of R12/R10/R6 cache selection and needs actual all43 hashes',async()=>{
  for(const selected of ['r12','r10','r6']) {
    const f=fixture({r12Cache:selected==='r12',r10Cache:selected==='r10',r12Hf:true});for(const binding of OWNED_RUNS.slice(0,4))f.roots.delete(path.join(binding.root,'hf'));
    const observe=f.io.metadata;f.io.metadata=async runtime=>{const proof=await observe(runtime);proof.entries.pop();return proof;};const receipt=await maintain(env,f.io);assert.match(receipt.cache.state,/insufficient-proved/);assert.equal(receipt.cache.projectionReserveBytes,0);assert.equal(f.mutations.some(item=>item[0]==='copy'),false);
  }
  const complete=fixture({r12Cache:true});for(const binding of OWNED_RUNS.slice(0,4))complete.roots.delete(path.join(binding.root,'hf'));const seeded=await maintain(env,complete.io);assert.equal(seeded.cache.originalRun,OWNED_RUNS[4].run);assert.equal(seeded.cache.projectionReserveBytes,100);assert.ok(complete.mutations.some(item=>item[0]==='copy'));
  const low=fixture({r12Cache:true,freeBytes:1});await maintain(env,low.io);assert.equal(low.mutations.some(item=>item[0]==='copy'),false);
});

test('fixed actual R13 runtime permits only its HF leaf and verified original-preserving cache copy',async()=>{
  const binding=OWNED_RUNS[5],runtime=validateRuntime(binding,runtimes[5]);assert.equal(runtimes[5].length,71262);assert.equal(binding.runtimeSha,'76554eb462c580cd6afa7b5102aaeaa434de85049fe1b6701fe7ba2347f6d61f');assert.equal(binding.source,'ba6bdd824823f01f8c4dc435f9fad50f45f97893');assert.equal(binding.cacheBinarySha256,'09088a12b26968177adf5a6ead76b088be2f791e5756224497db4fb7d20cf8fd');assert.equal(runtime.hfHome,path.join(binding.root,'hf'));
  assert.equal(path.resolve(runtime.sourceRoot,'../..','moss-sc-26752',path.basename(runtime.root),'target'),path.join(runtime.root,'target'));
  const f=fixture({r13Cache:true}),remove=f.io.remove;f.io.remove=async leaf=>{if(leaf===runtime.hfHome){const proof=f.reports.find(item=>item.stage==='fixed-r13-readonly-metadata-before-hf-cleanup');assert.ok(proof);assert.equal(proof.runtimeSha256,binding.runtimeSha);assert.equal(proof.metadata.completeProjection,true);}return remove(leaf);};
  const receipt=await maintain(env,f.io);assert.equal(receipt.cache.originalRun,binding.run);assert.equal(receipt.cache.originalBinarySha256,binding.cacheBinarySha256);assert.equal(receipt.cache.originalCacheRetained,true);assert.equal(f.roots.has(f.from),true);assert.equal(f.roots.has(f.to),true);assert.equal(f.roots.has(runtime.hfHome),false);
  assert.throws(()=>validateRuntime({...binding,root:'E:\\models'},runtimes[5]),/root/);assert.throws(()=>validateRuntime(binding,Buffer.concat([runtimes[5],Buffer.from(' ')])),/runtime bytes/);
  const bad=fixture({r13Cache:true,mutateRuntime:(bytes,index)=>index===5?Buffer.from('{}'):bytes});await assert.rejects(()=>maintain(env,bad.io));assert.deepEqual(bad.mutations,[]);
  const reparse=fixture({r13Cache:true,mutateInspect:root=>root===runtime.hfHome?{...tree(root),reparsePoints:1}:undefined});await assert.rejects(()=>maintain(env,reparse.io));assert.deepEqual(reparse.mutations,[]);
  const wrong=fixture({r13Cache:true});wrong.io.hash=async()=> '0'.repeat(64);await assert.rejects(()=>maintain(env,wrong.io),/source cache binary differs/);assert.equal(wrong.mutations.some(item=>item[0]==='copy'),false);
  const copied=fixture({r13Cache:true});copied.io.hash=async file=>file.startsWith(copied.to)?'0'.repeat(64):binding.cacheBinarySha256;await assert.rejects(()=>maintain(env,copied.io),/copied cache binary differs/);
  const existing=fixture({r13Cache:true,destinationExists:true});await maintain(env,existing.io);assert.equal(existing.mutations.some(item=>item[0]==='copy'),false);assert.equal(existing.roots.has(existing.from),true);
});
test('R13 completeness is independent of R13/R10/R6 cache selection and needs actual all43 hashes',async()=>{
  for(const selected of ['r13','r10','r6']) {
    const f=fixture({r13Cache:selected==='r13',r10Cache:selected==='r10',r13Hf:true});for(const binding of OWNED_RUNS.slice(0,5))f.roots.delete(path.join(binding.root,'hf'));
    const observe=f.io.metadata;f.io.metadata=async runtime=>{const proof=await observe(runtime);proof.entries.pop();return proof;};const receipt=await maintain(env,f.io);assert.match(receipt.cache.state,/insufficient-proved/);assert.equal(receipt.cache.projectionReserveBytes,0);assert.equal(f.mutations.some(item=>item[0]==='copy'),false);
  }
  const complete=fixture({r13Cache:true});for(const binding of OWNED_RUNS.slice(0,5))complete.roots.delete(path.join(binding.root,'hf'));const seeded=await maintain(env,complete.io);assert.equal(seeded.cache.originalRun,OWNED_RUNS[5].run);assert.equal(seeded.cache.projectionReserveBytes,100);assert.ok(complete.mutations.some(item=>item[0]==='copy'));
  const low=fixture({r13Cache:true,freeBytes:1});await maintain(env,low.io);assert.equal(low.mutations.some(item=>item[0]==='copy'),false);
});

test('fixed actual R14 runtime permits only its HF leaf and verified original-preserving cache copy',async()=>{
  const binding=OWNED_RUNS[6],runtime=validateRuntime(binding,runtimes[6]);assert.equal(runtimes[6].length,71286);assert.equal(binding.runtimeSha,'4082f9c28ae31aee7dda8e850d04b197d7741a01edbc3856aa53ddac6ba8a532');assert.equal(binding.source,'12e1b364360b43b28e864879be4c68bb28a696bc');assert.equal(binding.cacheBinarySha256,'ab36062090356912f9131654fa5c3ec87ed808f5e0fc0cbd32e388773edb75c2');assert.equal(runtime.hfHome,path.join(binding.root,'hf'));
  assert.equal(path.resolve(runtime.sourceRoot,'../..','moss-sc-26752',path.basename(runtime.root),'target'),path.join(runtime.root,'target'));
  const f=fixture({r14Cache:true}),remove=f.io.remove;f.io.remove=async leaf=>{if(leaf===runtime.hfHome){const proof=f.reports.find(item=>item.stage==='fixed-r14-readonly-metadata-before-hf-cleanup');assert.ok(proof);assert.equal(proof.runtimeSha256,binding.runtimeSha);assert.equal(proof.metadata.completeProjection,true);}return remove(leaf);};
  const receipt=await maintain(env,f.io);assert.equal(receipt.cache.originalRun,binding.run);assert.equal(receipt.cache.originalBinarySha256,binding.cacheBinarySha256);assert.equal(receipt.cache.originalCacheRetained,true);assert.equal(f.roots.has(f.from),true);assert.equal(f.roots.has(f.to),true);assert.equal(f.roots.has(runtime.hfHome),false);
  assert.throws(()=>validateRuntime({...binding,root:'E:\\models'},runtimes[6]),/root/);assert.throws(()=>validateRuntime(binding,Buffer.concat([runtimes[6],Buffer.from(' ')])),/runtime bytes/);
  const bad=fixture({r14Cache:true,mutateRuntime:(bytes,index)=>index===6?Buffer.from('{}'):bytes});await assert.rejects(()=>maintain(env,bad.io));assert.deepEqual(bad.mutations,[]);
  const reparse=fixture({r14Cache:true,mutateInspect:root=>root===runtime.hfHome?{...tree(root),reparsePoints:1}:undefined});await assert.rejects(()=>maintain(env,reparse.io));assert.deepEqual(reparse.mutations,[]);
  const wrong=fixture({r14Cache:true});wrong.io.hash=async()=> '0'.repeat(64);await assert.rejects(()=>maintain(env,wrong.io),/source cache binary differs/);assert.equal(wrong.mutations.some(item=>item[0]==='copy'),false);
  const copied=fixture({r14Cache:true});copied.io.hash=async file=>file.startsWith(copied.to)?'0'.repeat(64):binding.cacheBinarySha256;await assert.rejects(()=>maintain(env,copied.io),/copied cache binary differs/);
  const existing=fixture({r14Cache:true,destinationExists:true});await maintain(env,existing.io);assert.equal(existing.mutations.some(item=>item[0]==='copy'),false);assert.equal(existing.roots.has(existing.from),true);
});
test('R14 completeness is independent of R14/R10/R6 cache selection and needs actual all43 hashes',async()=>{
  for(const selected of ['r14','r13','r12','r10','r6']) {
    const f=fixture({r14Cache:selected==='r14',r13Cache:selected==='r13',r12Cache:selected==='r12',r10Cache:selected==='r10',r14Hf:true});for(const binding of OWNED_RUNS.slice(0,6))f.roots.delete(path.join(binding.root,'hf'));
    const observe=f.io.metadata;f.io.metadata=async runtime=>{const proof=await observe(runtime);proof.entries.pop();return proof;};const receipt=await maintain(env,f.io);assert.match(receipt.cache.state,/insufficient-proved/);assert.equal(receipt.cache.projectionReserveBytes,0);assert.equal(f.mutations.some(item=>item[0]==='copy'),false);
  }
  const complete=fixture({r14Cache:true});for(const binding of OWNED_RUNS.slice(0,6))complete.roots.delete(path.join(binding.root,'hf'));const seeded=await maintain(env,complete.io);assert.equal(seeded.cache.originalRun,OWNED_RUNS[6].run);assert.equal(seeded.cache.projectionReserveBytes,100);assert.ok(complete.mutations.some(item=>item[0]==='copy'));
  const low=fixture({r14Cache:true,freeBytes:1});await maintain(env,low.io);assert.equal(low.mutations.some(item=>item[0]==='copy'),false);
});

test('fixed R15 HF-only recovery binds actual runtime and repeats full metadata before only that leaf removal',async()=>{
 const binding=OWNED_RUNS[7],runtime=validateRuntime(binding,runtimes[7]);assert.equal(runtimes[7].length,71286);assert.equal(binding.runtimeSha,'a4c17012e7143f8775809decbd72216a7c3114209bf91c3f7f0d9608f0585cc4');assert.equal(binding.source,'a0692191b09813fab859659bf382bd7c91a60140');assert.equal(runtime.hfHome,path.join(binding.root,'hf'));
 const f=fixture({r15Hf:true}),remove=f.io.remove;f.io.remove=async leaf=>{if(leaf===runtime.hfHome){const proof=f.reports.find(item=>item.stage==='fixed-r15-readonly-metadata-before-hf-cleanup');assert.equal(proof.metadata.completeProjection,true);assert.equal(proof.runtimeSha256,binding.runtimeSha);}return remove(leaf);};const receipt=await maintain(env,f.io);assert.equal(f.roots.has(runtime.hfHome),false);assert.equal(receipt.candidates[7].cache,null);
 for(const mutate of [bytes=>Buffer.concat([bytes,Buffer.from(' ')]),()=>Buffer.from('{}')]){const bad=fixture({r15Hf:true,mutateRuntime:(bytes,index)=>index===7?mutate(bytes):bytes});await assert.rejects(()=>maintain(env,bad.io));assert.deepEqual(bad.mutations,[]);}
 const active=fixture({r15Hf:true,activeAtMutation:true});await assert.rejects(()=>maintain(env,active.io));assert.deepEqual(active.mutations,[]);
});

test('fixed reference HF-only recovery never copies or removes any target even with abundant headroom',async()=>{
 const f=fixture({r15Hf:true,freeBytes:1024**4});const result=await maintain(env,f.io,{copyCache:false});assert.equal(result.cache.state,'hf-only-recovery-original-targets-retained-no-copy');assert.equal(f.mutations.some(x=>x[0]==='copy'),false);assert.ok(f.mutations.some(x=>x[1]===path.join(OWNED_RUNS[7].root,'hf')));assert.equal(f.roots.has(f.from),true);assert.equal(f.roots.has(f.to),false);
});

test('reference cleanup CLI requires exact Actions/source and the same closed reference artifact envelope',()=>{
 const base={GITHUB_ACTIONS:'true',MOSS_SOURCE_SHA:'a'.repeat(40),GITHUB_SHA:'a'.repeat(40),MOSS_REFERENCE_BACKEND:'mlx',MOSS_REFERENCE:JSON.stringify({phase:'upload',artifact_id:null,artifact_run_id:null})};validateHfOnlyEnvelope(base);
 for(const changes of [{GITHUB_ACTIONS:'false'},{GITHUB_SHA:'b'.repeat(40)},{MOSS_REFERENCE_BACKEND:'candle'},{MOSS_REFERENCE:'{}'},{MOSS_REFERENCE:JSON.stringify({phase:'upload',artifact_id:null,artifact_run_id:null,command:'id'})}])assert.throws(()=>validateHfOnlyEnvelope({...base,...changes}));
});

// CI_IMPACT: metadata-only fixed-owner recovery; inert IO, no Windows process/model invocation.
// Fixture contains exact archived JSON only: R19/H39/H42 execution runtime and R19/H39 transfer.
const selectedRecords=JSON.parse(gunzipSync(await readFile(new URL('./fixtures/moss-stopped-runtime/h39-h42-selected.json.gz',import.meta.url))));
const selectedEnv={...env,GITHUB_ACTIONS:'true',GITHUB_JOB:'moss-owned-hf-maintenance',GITHUB_SHA:env.MOSS_SOURCE_SHA,MOSS_REFERENCE_BACKEND:'candle',MOSS_REFERENCE:JSON.stringify({phase:'inventory',candidate:null})};
function selectedFixture({candidate=null,mutateRead,mutateEntries,mutateInspect,activeAtMutation=false,processMatch=false,artifactChange}={}) {
  const runtimes=SELECTED_HF.map((_,i)=>Buffer.from(selectedRecords[i])),packages=JSON.parse(selectedRecords[3]).after;
  const files=new Map(SELECTED_HF.map((b,i)=>[path.join(b.root,'runtime.json'),runtimes[i]]));
  files.set(path.join(SELECTED_HF[0].root,'hf-transfer.json'),Buffer.from(selectedRecords[3]));files.set(path.join(SELECTED_HF[1].root,'hf-transfer.json'),Buffer.from(selectedRecords[4]));files.set(path.join(path.dirname(SELECTED_HF[2].leaf),'runtime.json'),partialBuildRuntime(JSON.parse(selectedRecords[2])));
  const present=new Set(SELECTED_HF.map(b=>b.leaf)),calls=[],reports=[],removed=[],stopCounts=new Map();
  const entries=b=>(b.key==='h42-partial'?packages.slice(0,3):packages).map((item,i)=>({path:item.file,type:'file',dev:'1',ino:String(9007199254740993n+BigInt(i)),nlink:'1',byteLength:String(b.key==='h42-partial'?Math.min(item.byteLength,13):item.byteLength)}));
  const io={
    volume:async()=>{calls.push('volume');return {drive:'D',size:1024**4,free:1000+removed.length*987};},
    read:async file=>{calls.push(['read',file]);assert.ok(files.has(file),'only fixed ownership JSON may be read');return mutateRead?.(file,files.get(file))??files.get(file);},
    inspect:async file=>{calls.push(['inspect',file]);const b=SELECTED_HF.find(b=>b.leaf===file);const list=b&&present.has(file)?entries(b):[];return mutateInspect?.(file)??{...tree(file,b?present.has(file):true),bytes:list.reduce((n,x)=>n+Number(x.byteLength),0),fileCount:list.length};},
    entries:async leaf=>{calls.push(['entries',leaf]);const b=SELECTED_HF.find(b=>b.leaf===leaf);assert.ok(b);return mutateEntries?.(b,entries(b))??entries(b);},
    stopped:async b=>{calls.push(['stopped',b.key]);const count=(stopCounts.get(b.key)??0)+1;stopCounts.set(b.key,count);const latest={...state(b),...(activeAtMutation&&count>1?{status:'in_progress'}:{})};validateStopped(b,latest,state(b),jobs(b));return {run:b.run,head:b.source,status:'completed',jobs:jobs(b)};},
    artifact:async b=>({id:Number(b.artifact),name:`moss-execution-cuda-${b.run}-1`,size_in_bytes:b.artifactBytes,digest:`sha256:${b.artifactSha}`,expired:false,workflow_run:{id:Number(b.run),head_sha:b.source},...artifactChange}),
    processes:async b=>{calls.push(['processes',b.key]);return {complete:true,matches:processMatch?[{pid:8692,name:'ambiguous recycled PID'}]:[]};},
    remove:async leaf=>{assert.ok(SELECTED_HF.slice(1).some(b=>b.leaf===leaf));calls.push(['remove',leaf]);assert.ok(reports.some(x=>x.stage==='validated-before-single-leaf-cleanup'));removed.push(leaf);present.delete(leaf);},
    report:value=>reports.push(structuredClone(value)),
  };
  return {io,calls,reports,removed,present,files,packages,env:{...selectedEnv,MOSS_REFERENCE:JSON.stringify({phase:candidate?'cleanup':'inventory',candidate})}};
}
test('selected HF fixture authenticates exact archived producer bytes and separate H42 derived build config',()=>{
  for(const [i,b]of SELECTED_HF.entries())assert.equal(selectedRuntime(b,Buffer.from(selectedRecords[i])).sourceSha,b.product);
  const bytes=partialBuildRuntime(JSON.parse(selectedRecords[2]));assert.equal(bytes.length,71584);assert.equal(createHash('sha256').update(bytes).digest('hex'),'449d09a6d6f2dddb5c3ef9293fe3385f3b6d659da82f48b531710b50d9b95c21');
  assert.equal(JSON.parse(bytes).packageSource,SELECTED_HF[0].leaf);assert.equal(JSON.parse(bytes).root,path.dirname(SELECTED_HF[2].leaf));
  assert.equal(SELECTED_HF[1].source,'63874fe81e45750a8b98a5bf1b821383af40ae72');assert.notEqual(SELECTED_HF[1].source,SELECTED_HF[1].product);
  for(const i of [3,4]){const b=SELECTED_HF[i===3?0:1];assert.equal(createHash('sha256').update(selectedRecords[i]).digest('hex'),b.transferSha);assert.equal(JSON.parse(selectedRecords[i]).after.reduce((n,x)=>n+x.byteLength,0),59241637664);}
});
test('manual inventory authenticates all three owners and metadata without any deletion/process/weight reads',async()=>{
  const f=selectedFixture(),receipt=await maintainSelectedHf(f.env,f.io);assert.equal(receipt.candidates.length,3);assert.deepEqual(f.removed,[]);assert.equal(f.calls.some(x=>Array.isArray(x)&&x[0]==='processes'),false);assert.equal(receipt.candidates[0].protected,true);assert.equal(receipt.candidates[0].inventory.bytes,59241637664);assert.equal(receipt.candidates[2].derivedRuntime.sha256,SELECTED_HF[2].derivedRuntimeSha);
  for(const report of f.reports)assert.equal(report.receiptSha256,createHash('sha256').update(JSON.stringify(report.receipt)).digest('hex'));assert.equal(receipt.before.free,receipt.after.free);
});
test('explicit cleanup removes exactly selected HF leaf after repeated stopped/runtime/metadata/process checks',async()=>{
  for(const candidate of ['h39','h42-partial']){const f=selectedFixture({candidate}),receipt=await maintainSelectedHf(f.env,f.io),binding=SELECTED_HF.find(b=>b.key===candidate);assert.deepEqual(f.removed,[binding.leaf]);assert.equal(f.present.has(SELECTED_HF[0].leaf),true);assert.equal(receipt.operations.length,1);assert.equal(receipt.after.free-receipt.before.free,987);assert.equal(f.calls.filter(x=>Array.isArray(x)&&x[0]==='processes').length,2);assert.equal(f.calls.filter(x=>Array.isArray(x)&&x[0]==='stopped'&&x[1]===candidate).length,2);assert.notEqual(receipt.operations[0].observedBytesBefore,987);}
});
test('closed maintenance ref/backend/job/run/phase/candidate failures precede all IO',async()=>{
  for(const changes of [{GITHUB_ACTIONS:'false'},{GITHUB_REPOSITORY:'other/repo'},{GITHUB_JOB:'moss-upstream-reference'},{GITHUB_SHA:'b'.repeat(40)},{MOSS_REFERENCE_BACKEND:'mlx'},{GITHUB_RUN_ID:SELECTED_HF[0].run},{GITHUB_RUN_ATTEMPT:'0'},{GITHUB_WORKSPACE:'E:\\user-cache'},...[{}, {phase:'inventory',candidate:'h39'},{phase:'cleanup',candidate:null},{phase:'cleanup',candidate:'r19-protected'},{phase:'cleanup',candidate:'H39'},{phase:'cleanup',candidate:['h39','h42-partial']},{phase:'cleanup',candidate:'h39',path:SELECTED_HF[0].leaf},{phase:'execute',candidate:'h39'}].map(x=>({MOSS_REFERENCE:JSON.stringify(x)}))]){const f=selectedFixture();await assert.rejects(()=>maintainSelectedHf({...f.env,...changes},f.io));assert.deepEqual(f.calls,[]);assert.deepEqual(f.removed,[]);}
  assert.deepEqual(selectedHfEnvelope(selectedEnv),{phase:'inventory',candidate:null});
});
test('artifact/runtime/reparse/foreign/nonexclusive/extent/live ownership failures prevent every cleanup',async()=>{
  const h42Runtime=path.join(path.dirname(SELECTED_HF[2].leaf),'runtime.json');
  for(const options of [
    {artifactChange:{digest:'sha256:'+'0'.repeat(64)}},{artifactChange:{expired:true}},{artifactChange:{workflow_run:{id:1,head_sha:SELECTED_HF[0].product}}},
    {mutateRead:(file,bytes)=>file===h42Runtime?Buffer.concat([bytes,Buffer.from(' ')]):bytes},
    {mutateRead:(file,bytes)=>file===path.join(SELECTED_HF[1].root,'runtime.json')?Buffer.from('{}'):bytes},
    {mutateInspect:file=>file===SELECTED_HF[2].leaf?{...tree(file),reparsePoints:1}:undefined},
    ...[x=>({...x,nlink:'2'}),x=>({...x,type:'symlink'}),x=>({...x,path:'../../user-cache'}),x=>({...x,byteLength:String(2**48)}),x=>({...x,ino:'0'})].map(change=>({mutateEntries:(b,list)=>b.key==='h42-partial'?[change(list[0]),...list.slice(1)]:list})),
    {activeAtMutation:true},{processMatch:true},
  ]){const f=selectedFixture({candidate:'h42-partial',...options});await assert.rejects(()=>maintainSelectedHf(f.env,f.io));assert.deepEqual(f.removed,[]);}
  assert.throws(()=>selectedProcesses({complete:false,matches:[]}));assert.throws(()=>selectedProcesses({complete:true,matches:[{pid:1}]}));
  const f=selectedFixture(),entries=[{path:f.packages[0].file,type:'file',dev:'1',ino:'1',nlink:'1',byteLength:'1'}];assert.throws(()=>selectedEntries(SELECTED_HF[2],[...entries,{...entries[0],path:f.packages[1].file}],f.packages),/ownership/);
  assert.throws(()=>selectedArtifact(SELECTED_HF[1],{}));assert.throws(()=>selectedRuntime(SELECTED_HF[1],Buffer.from('{}')));
});
test('assembled manual workflow streams exact helper with no checkout, cache copy, model job or old-route selection changes',async()=>{
  const source=await readFile(new URL('../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8'),job=source.slice(source.indexOf('  moss-owned-hf-maintenance:'),source.indexOf('  moss-reference-maintenance:'));
  assert.match(job,/if: inputs.profile == 'moss-owned-hf-maintenance'/);assert.match(job,/runs-on: \[self-hosted, Windows, X64, cuda\]/);assert.doesNotMatch(job,/real-weights|uses:|checkout|cargo|nvidia|npm /);assert.match(job,/contents: read/);assert.match(job,/actions: read/);assert.match(job,/--moss-selected-hf/);assert.match(job,/externals\\node24\\bin\\node.exe/);assert.match(job,/moss-owned-cache-maintenance.mjs\?ref=\$env:MOSS_SOURCE_SHA/);
  const helper=await readFile(new URL('./ci/moss-owned-cache-maintenance.mjs',import.meta.url));assert.ok(job.includes(createHash('sha256').update(helper).digest('hex')),'workflow must bind exact helper SHA');
  const old=source.slice(source.indexOf('  moss-reference-maintenance:'),source.indexOf('  moss-upstream-reference:'));assert.match(old,/inputs.profile == 'moss-upstream-reference'/);assert.doesNotMatch(old,/moss-selected-hf|h42-partial/);
});
