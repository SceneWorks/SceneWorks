import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, link, symlink, rm, readdir, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { win32 as path } from 'node:path';
import { H47_OWNER, H47_PROCESS_FILTER, validateH47Processes, validateH47Projection, h47SourceEntries, OWNED_RUNS, validateHfOnlyEnvelope, copyCacheExclusive, hashMetadataFile, observeHf, hfEntries, validateHfObservation, R9_DIAGNOSTIC, diagnoseR9, destination, validateRuntime, validateTree, validateStopped, stopped, maintain } from './ci/moss-owned-cache-maintenance.mjs';
const runtimes = await Promise.all([4, 5, 6, 10, 12, 13, 14, 15].map(async round => gunzipSync(await readFile(new URL(`./fixtures/moss-stopped-runtime/r${round}.json.gz`, import.meta.url)))));
const env = { GITHUB_REPOSITORY: 'SceneWorks/SceneWorks', GITHUB_RUN_ID: '99999999999', GITHUB_RUN_ATTEMPT: '1', GITHUB_WORKSPACE: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', MOSS_ROOT: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-99999999999-1-cuda', MOSS_SOURCE_SHA: 'a'.repeat(40), MOSS_INFERENCE_SHA: 'b'.repeat(40) };
const state = binding => ({ id: Number(binding.run), run_attempt: 1, head_sha: binding.actionsSource ?? binding.source, status: 'completed', conclusion: 'failure' });
const jobs = binding => [{ id: 1, run_id: Number(binding.run), status: 'completed', conclusion: 'failure' }];
const tree = (root, exists = true) => ({ root, exists, safe: true, reparsePoints: 0, bytes: exists ? 100 : 0, fileCount: exists ? 2 : 0, files: [] });
function fixture({ mutateRuntime, mutateInspect, activeAtMutation = false, destinationExists = false, freeBytes = 8 * 1024 ** 3, r10Cache = false, r12Cache = false, r12Hf = false, r13Cache = false, r13Hf = false, r14Cache = false, r14Hf = false, r15Hf = false } = {}) {
  const roots = new Set(OWNED_RUNS.slice(0,8).filter((item,index)=>(index!==4||r12Cache||r12Hf)&&(index!==5||r13Cache||r13Hf)&&(index!==6||r14Cache||r14Hf)&&(index!==7||r15Hf)).map(item => path.join(item.root, 'hf')));
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
  assert.deepEqual(OWNED_RUNS.map(item => item.run), ['40145054879', '40152001538', '40155160639', '40163530825', '40167292776', '40169818205', '40174802721', '40181253824', '40415337753']);
  for (const [index, binding] of OWNED_RUNS.slice(0,8).entries()) {
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

// CI_IMPACT: metadata-only fixed owner fixtures; no models/network/Windows process execution.
// Exact archived H47 JSON bytes, gzip inline to keep the existing two-file change boundary.
const h47Records=JSON.parse(gunzipSync(Buffer.from("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",'base64')));

const h47Runtime=JSON.parse(h47Records['runtime.json']),h47Transfer=JSON.parse(h47Records['hf-transfer.json']);
const h47Entry=(item,index)=>({path:item.file,type:'file',dev:'1',ino:String(index+10),nlink:'1',byteLength:String(item.byteLength),sha256:item.sha256,mtimeNs:'100',ctimeNs:'101'});
function h47Fixture(){
  const f=fixture(),hf=h47Runtime.hfHome;f.roots.clear();f.roots.add(hf);
  const oldRead=f.io.read,oldHash=f.io.hash,oldInspect=f.io.inspect,eventPath='event.json';
  f.io.inspect=async file=>file===hf?{...tree(file,f.roots.has(hf)),bytes:h47Transfer.after.reduce((n,item)=>n+item.byteLength,0),fileCount:43}:oldInspect(file);
  f.env={...env,GITHUB_ACTIONS:'true',GITHUB_JOB:'windows-maintenance',GITHUB_SHA:env.MOSS_SOURCE_SHA,MOSS_NATIVE_RUN_ID:'',GITHUB_EVENT_PATH:eventPath};
  f.io.read=async file=>file===eventPath?Buffer.from(JSON.stringify({inputs:{moss_terminal:JSON.stringify({phase:'execute',native_run_id:''})}})):path.dirname(file)===H47_OWNER.root&&Object.hasOwn(h47Records,path.basename(file))?Buffer.from(h47Records[path.basename(file)]):oldRead(file);
  f.io.hash=async file=>file===path.join(path.dirname(H47_OWNER.root),'moss-build-40415337753-1-cuda','hf-owner.json')?h47Transfer.originalOwner.sha256:oldHash(file);
  f.io.metadata=async()=>({root:hf,entries:h47Transfer.after.map(h47Entry)});
  f.io.entries=async()=>h47Transfer.after.map((item,i)=>({...h47Entry(item,i),mtimeNs:'100',ctimeNs:'101'}));
  f.io.processes=async()=>[];
  return f;
}
test('H47 archived runtime and transfer prove exact separate product/Actions/private-source identities',async()=>{
 assert.equal(createHash('sha256').update(h47Records['runtime.json']).digest('hex'),H47_OWNER.runtimeSha);
 assert.equal(validateRuntime(H47_OWNER,Buffer.from(h47Records['runtime.json'])).root,H47_OWNER.root);
 validateStopped(H47_OWNER,state(H47_OWNER),state(H47_OWNER),jobs(H47_OWNER));
 assert.throws(()=>validateStopped(H47_OWNER,{...state(H47_OWNER),head_sha:H47_OWNER.source},state(H47_OWNER),jobs(H47_OWNER)));
 for(const root of ['E:\\models',path.join(path.dirname(H47_OWNER.root),'moss-execute-40349954885-1-cuda'),path.join(path.dirname(H47_OWNER.root),'moss-execute-40331712229-1-cuda')])assert.throws(()=>validateRuntime({...H47_OWNER,root},Buffer.from(h47Records['runtime.json'])));
 const f=h47Fixture(),receipt=await maintain(f.env,f.io);assert.deepEqual(f.mutations,[['remove',h47Runtime.hfHome]]);assert.equal(receipt.candidates.at(-1).owner.protectedSource,JSON.parse(h47Records['cuda-package-preparation.json']).readOnlySource);
 assert.equal(receipt.candidates.at(-1).cache,null);assert.equal(receipt.operations.length,1);
});
test('H47 accepts exclusive adapter blob extras, refuses hardlinks, unknown entries and changed projection',()=>{
 const entries=h47Transfer.after.map(h47Entry),observation={root:h47Runtime.hfHome,entries};
 for(const [i,file]of h47Runtime.editAdapterFiles.entries()){const item=h47Transfer.after.find(x=>x.file===file);entries.push({...h47Entry(item,50+i),path:`hub/models--chinoll--Moss-Image-Edit/blobs/${String(i+1).repeat(40)}`});}
 assert.equal(validateH47Projection(observation,h47Runtime,h47Transfer).entries.length,45);
 for(const mutate of [x=>x.entries.pop()&&x.entries.shift(),x=>x.entries[0].nlink='2',x=>x.entries[0].type='symlink',x=>x.entries[0].sha256='0'.repeat(64),x=>x.entries[0].byteLength='1',x=>x.entries.push({...x.entries.at(-1),path:'user-photo.png'}),x=>x.entries[0].path='../outside']){const bad=structuredClone(observation);mutate(bad);assert.throws(()=>validateH47Projection(bad,h47Runtime,h47Transfer));}
});
test('H47 live process guard refuses owned/unknown identities and proves PID reuse without killing',()=>{
 const unrelated={pid:99,startedAt:'2026-10-27T00:00:00Z',executable:'C:\\Windows\\system32\\notepad.exe',commandLine:'notepad.exe'};
 assert.equal(validateH47Processes([]).ownedProcessesAbsent,true);
 assert.ok(H47_PROCESS_FILTER.includes("Name='sceneworks-rust-api.exe'")&&H47_PROCESS_FILTER.includes('ProcessId=45652'));
 for(const root of [H47_OWNER.root,h47Runtime.buildRoot,path.join(path.dirname(H47_OWNER.root),'moss-build-40415337753-1-cuda')])for(const value of [root.replaceAll('\\','\\\\'),root.replaceAll('\\','/')])for(const field of ['CommandLine','ExecutablePath'])assert.ok(H47_PROCESS_FILTER.includes(`${field} LIKE '%${value}%'`));
 assert.equal(validateH47Processes([{...unrelated,pid:45652}]).reusedPids.length,1);
 for(const patch of [{pid:45652,startedAt:'2026-10-26T23:50:37Z'},{executable:null},{commandLine:null},{startedAt:'unknown'},{executable:path.join(h47Runtime.buildRoot,'cuda','sceneworks-rust-api.exe')},{commandLine:`runner --state "${H47_OWNER.root}\\state"`},{executable:path.join(H47_OWNER.root,'worker.exe')},{commandLine:`node task --root ${H47_OWNER.root.replaceAll('\\','/').toUpperCase()}`}])assert.throws(()=>validateH47Processes([{...unrelated,...patch}]));
});
test('H47 all ownership and live guards precede removal and repeat at the mutation boundary',async()=>{
 const scenarios=[
  f=>{const read=f.io.read;f.io.read=async file=>file===path.join(H47_OWNER.root,'hf-transfer.json')?Buffer.from('{}'):read(file);},
  f=>{const read=f.io.read;f.io.read=async file=>file===path.join(H47_OWNER.root,'cuda-package-preparation.json')?Buffer.from('{}'):read(file);},
  f=>{f.io.hash=async()=> '0'.repeat(64);},
  f=>{const inspect=f.io.inspect;f.io.inspect=async file=>file===h47Runtime.hfHome?{...tree(file),reparsePoints:1}:inspect(file);},
  f=>{f.io.processes=async()=>[{pid:45652,startedAt:'2026-10-26T23:50:37Z',executable:path.join(h47Runtime.buildRoot,'cuda','sceneworks-rust-api.exe'),commandLine:'sceneworks-rust-api.exe'}];},
  f=>{let n=0;f.io.processes=async()=>++n===3?[{pid:45652,startedAt:'2026-10-26T23:50:37Z',executable:path.join(h47Runtime.buildRoot,'cuda','sceneworks-rust-api.exe'),commandLine:'sceneworks-rust-api.exe'}]:[];},
  f=>{const metadata=f.io.metadata;let n=0;f.io.metadata=async()=>{const x=await metadata();if(++n===2)x.entries[0].ino='999';return x;};},
  f=>{const entries=f.io.entries;let n=0;f.io.entries=async()=>{const x=await entries();if(++n===2)x[0].mtimeNs='999';return x;};},
  f=>{const stopped=f.io.stopped;let n=0;f.io.stopped=async binding=>{if(binding.run===H47_OWNER.run&&++n===3)throw Error('new active run');return stopped(binding);};},
  f=>{f.env.GITHUB_JOB='other-job';},f=>{f.env.MOSS_NATIVE_RUN_ID='40349954885';},
  f=>{const read=f.io.read;f.io.read=async file=>file==='event.json'?Buffer.from(JSON.stringify({inputs:{phase:'cuda-rebase'}})):read(file);},
  f=>{const read=f.io.read;let n=0;f.io.read=async file=>file===path.join(H47_OWNER.root,'runtime.json')&&++n===4?Buffer.from('{}'):read(file);},
  f=>{const entries=f.io.entries;f.io.entries=async file=>{const x=await entries(file);if(file===h47Runtime.hfHome)x[0].nlink='2';return x;};},
  f=>{const entries=f.io.entries;f.io.entries=async file=>{const x=await entries(file);x[0].nlink='2';return x;};},
 ];
 const reasons=[/archived ownership record changed/,/archived ownership record changed/,/original private build owner changed/,/reparse/,/owned H47 process/,/owned H47 process/,/metadata changed before cleanup/,/protected original HF changed/,/new active run/,/normal new-pair/,/normal new-pair/,/ordinary fresh execute/,/runtime bytes changed/,/HF identity changed at removal boundary/,/protected original source ownership changed/];
 assert.equal(scenarios.length,reasons.length);
 for(const [index,setup] of scenarios.entries()){const f=h47Fixture();setup(f);await assert.rejects(()=>maintain(f.env,f.io),reasons[index]);assert.deepEqual(f.mutations,[]);}
 const reference=h47Fixture();await maintain(reference.env,reference.io,{copyCache:false});assert.deepEqual(reference.mutations,[]);assert.ok(reference.roots.has(h47Runtime.hfHome));
});
test('H47 protected-source metadata uses exact bigint timestamps and detects real small-file link changes',async t=>{
 const root=await mkdtemp(resolve(tmpdir(),'moss-h47-source-'));t.after(()=>rm(root,{recursive:true,force:true}));await writeFile(resolve(root,'model.metadata'),'unlearned bytes');
 const before=await h47SourceEntries(root);assert.match(before[0].mtimeNs,/^[0-9]+$/);assert.match(before[0].ctimeNs,/^[0-9]+$/);assert.equal(before[0].nlink,'1');
 await link(resolve(root,'model.metadata'),resolve(root,'alias'));const linked=await h47SourceEntries(root);assert.ok(linked.every(x=>x.nlink==='2'));assert.equal(await readFile(resolve(root,'model.metadata'),'utf8'),'unlearned bytes');
});

test('H47 final target timestamp drift refuses before final process query and removal',async()=>{
 for(const key of ['mtimeNs','ctimeNs']){
  const f=h47Fixture(),entries=f.io.entries;let processChecks=0;
  f.io.processes=async()=>{processChecks++;return [];};
  f.io.entries=async file=>{const observation=await entries(file);if(file===h47Runtime.hfHome)observation[0][key]='999';return observation;};
  await assert.rejects(()=>maintain(f.env,f.io),/H47 HF identity changed at removal boundary/);
  assert.equal(processChecks,2);assert.deepEqual(f.mutations,[]);
 }
 const unchanged=h47Fixture();let processChecks=0;unchanged.io.processes=async()=>{processChecks++;return [];};
 await maintain(unchanged.env,unchanged.io);assert.equal(processChecks,3);assert.deepEqual(unchanged.mutations,[['remove',h47Runtime.hfHome]]);
});
test('authenticated target timestamps come from bigint stat and remain bound across content hashing',async t=>{
 const root=await mkdtemp(resolve(tmpdir(),'moss-h47-target-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const file=resolve(root,'metadata.json');await writeFile(file,'metadata');
 const [entry]=await hfEntries(root,{timestamps:true});
 assert.match(entry.mtimeNs,/^[0-9]+$/);assert.match(entry.ctimeNs,/^[0-9]+$/);
 assert.equal(await hashMetadataFile(file,entry),createHash('sha256').update('metadata').digest('hex'));
 for(const key of ['mtimeNs','ctimeNs'])await assert.rejects(()=>hashMetadataFile(file,{...entry,[key]:String(BigInt(entry[key])+1n)}),/changed before hash/);
 await writeFile(file,'changed!');
 await utimes(file,new Date('2020-01-01T00:00:00Z'),new Date('2020-01-01T00:00:00Z'));
 const [changed]=await hfEntries(root,{timestamps:true});assert.equal(changed.ino,entry.ino);assert.equal(changed.byteLength,entry.byteLength);assert.notEqual(changed.mtimeNs,entry.mtimeNs);
 await assert.rejects(()=>hashMetadataFile(file,entry),/changed before hash/);
});
