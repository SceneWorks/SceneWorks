// DECLARED SYNTHETIC local ledger fixtures only. ZERO physical GPU/model invocations.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, writeFile, rm, mkdir, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { loadProductTargets, terminalPlan, terminalCatalog } from './flux3-image-klein-terminal-plan.mjs';
import { validateApprovedPlan, initializeLedger, reserveInvocation, auditLedger, aggregateLedgerReceipts, planHash, verifyLedgerReceipt, candleNextApprovalBaseSha256, validateCandleSourceCompatibility, validateCandleDeviceTransitionBinding, validateCandleMeasurementTransitionBinding, validateCandleDependencySecurityUpdate, validateCandleDependencySecurityLocks, verifyCandleDependencySecurityArchive, candleReplayPrefix, appendCandleSourceTransition } from './flux3-image-klein-terminal-budget.mjs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { produceTerminalBuild } from './flux3-image-klein-terminal-build.mjs';
import { prepareReplayRequestFixture, makeReplayBaselineFixture } from './lib/flux3-replay-prefix-fixtures.mjs';
import { declaredSyntheticReadyBackend } from './lib/flux3-native-boundary-fixtures.mjs';
const sources = await loadProductTargets(), profile = terminalPlan(sources), digest = 'd'.repeat(64);
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sc26243-synthetic-ledger-only-')); t.after(() => rm(root, { recursive: true, force: true }));
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'declared-fixture-approval', userPlanApprovalId: 'DECLARED_SYNTHETIC_NOT_USER_AUTHORIZATION', approvedBy: 'synthetic CPU test', sourceSha: 'a'.repeat(40), inferenceMainSha: 'b'.repeat(40), ledgerId: 'declared-fixture-ledger', ledgerRoots: { mlx: path.join(root, 'mlx-ledger'), candle: path.join(root, 'candle-ledger') }, inputSpecSha256: { mlx: digest, candle: digest }, plan: structuredClone(profile) };
  return { root, approval };
}
// Declared fixture coordinator observations, separate from candidate receipts under mutation.
const heads = (approval, receipts) => ({kind:'flux3-terminal-finalized-ledger-heads',coordinatorClosureId:'DECLARED_SYNTHETIC_TRUSTED_CLOSURE',producersStopped:true,collectionManifestSha256:digest,ledgerId:approval.ledgerId,approvedPlanSha256:planHash(approval),backends:Object.fromEntries(receipts.map(r => [r.manifest.backend, { sequence:r.state.sequence,headSha256:r.state.headSha256 }]))});
const reservation = (approval, backend, invocation = approval.plan.generationInvocations.find(i => i.backend === backend && i.kind === 'product_generation')) => ({ approval, backend, ledgerRoot: approval.ledgerRoots[backend], ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), invocationId: invocation.invocationId, caseId: invocation.caseId, kind: invocation.kind, requestSha256: digest });
test('representative default expands EXACTLY 26 product +12 resource =38 globally, no automatic repeats/spares', () => {
  assert.equal(profile.generationInvocations.length, 38); assert.equal(profile.generationInvocations.filter(i => i.kind === 'product_generation').length, 26); assert.equal(profile.resourceCases.length, 12);
  assert.deepEqual(profile.budget.byBackend, { mlx: 17, candle: 21 }); assert.equal(profile.budget.globalHardCeiling, 48); assert.equal(profile.budget.unallocatedNotAuthorized, 10);
  assert(profile.generationInvocations.every(i => i.nativeGenerationCalls === 1 && i.warmups === 0 && i.repeats === 0 && i.requestTemplate.count === 1));
  assert.equal(profile.trainingLifecycles.length, 2); assert(profile.trainingLifecycles.every(t => t.steps === 4 && t.saveEvery === 1 && t.generationReloads === 1));
  assert(profile.resourceCases.every(c => c.status === 'BLOCKED')); assert(profile.uncovered.some(c => /Base q4\/q8/.test(c.requirement) && c.status === 'NOT_RUN')); assert(!profile.uncovered.some(c => /HF403/.test(c.reason))); assert(profile.uncovered.every(c => c.status !== 'PASS'));
  for (const backend of ['mlx', 'candle']) {
    assert.deepEqual(new Set(profile.cases.filter(c => c.backend === backend && c.options.enhancePrompt).map(c => c.options.pupMode)), new Set(['t2i', 'i2i', 't2i_bbox', 'i2i_bbox']));
    for (const modelId of ['flux3_klein', 'flux3_klein_base']) for (const count of [1, 10]) assert(profile.cases.some(c => c.backend === backend && c.modelId === modelId && c.options.references === count));
  }
  assert.equal(terminalCatalog(sources).diagnosticOnly, true); assert(profile.cases.length < terminalCatalog(sources).cases.length);
});
test('approval rejects diagnostic catalog, >48 inner calls, implicit multiplication, duplicate permits and missing user approval', async t => {
  const { approval } = await fixture(t); assert.equal(validateApprovedPlan(approval).generationCount, 38);
  for (const mutate of [a => delete a.userPlanApprovalId, a => a.plan = terminalCatalog(sources), a => a.plan.generationInvocations[0].nativeGenerationCalls = 2, a => a.plan.generationInvocations[0].warmups = 1, a => a.plan.generationInvocations[0].repeats = 2, a => a.plan.generationInvocations[1].invocationId = a.plan.generationInvocations[0].invocationId, a => a.plan.trainingLifecycles[0].steps = 20, a => a.plan.trainingLifecycles.push(a.plan.trainingLifecycles[0]), a => { while (a.plan.generationInvocations.length < 49) a.plan.generationInvocations.push({ ...a.plan.generationInvocations[0], invocationId: `unapproved-extra-${a.plan.generationInvocations.length}` }); }]) { const bad = structuredClone(approval); mutate(bad); assert.throws(() => validateApprovedPlan(bad)); }
});
test('disjoint backend ledgers enforce one GLOBAL immutable quota; coordinator aggregates unique receipts', async t => {
  const { approval } = await fixture(t);
  for (const backend of ['mlx', 'candle']) { await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] }); await reserveInvocation(reservation(approval, backend)); }
  const receipts = await Promise.all(['mlx', 'candle'].map(backend => auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })));
  assert.equal(aggregateLedgerReceipts(approval, receipts, heads(approval, receipts)).generationCount, 2); assert.equal(aggregateLedgerReceipts(approval, receipts, heads(approval, receipts)).approvedTotal, 38);
  await assert.rejects(reserveInvocation({ ...reservation(approval, 'mlx'), ...{ invocationId: approval.plan.generationInvocations.find(i => i.backend === 'candle').invocationId } }), /absent|consumed/);
  assert.throws(() => aggregateLedgerReceipts(approval, [receipts[0], receipts[0]], heads(approval, receipts)), /duplicate/);
  const corrupt = structuredClone(receipts[0]); corrupt.events[0].requestSha256 = 'e'.repeat(64); assert.throws(() => aggregateLedgerReceipts(approval, [corrupt], heads(approval, receipts)));
});
test('failed/canceled reservations stay spent across resumed sessions; no automatic retry or spare-ten allocation', async t => {
  const { approval } = await fixture(t), backend = 'candle'; await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const first = reservation(approval, backend); await reserveInvocation(first);
  await assert.rejects(reserveInvocation(first), /already consumed/);
  await assert.rejects(reserveInvocation({ ...first, invocationId: 'unallocated-spare-ten', kind: 'retry' }), /absent/);
  const receipt = await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] }); assert.equal(receipt.state.generationCount, 1); assert.equal(receipt.events.length, 1);
});
test('atomic local reservation lock prevents concurrent double consumption', async t => {
  const { approval } = await fixture(t), backend = 'mlx'; await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const attempts = await Promise.allSettled([reserveInvocation(reservation(approval, backend)), reserveInvocation(reservation(approval, backend))]);
  assert.equal(attempts.filter(a => a.status === 'fulfilled').length, 1); assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 1);
});
test('journal/state crash mismatch blocks all further starts rather than refunding/recovering a permit', async t => {
  const { approval } = await fixture(t), backend = 'candle', ledgerRoot = approval.ledgerRoots[backend]; await initializeLedger({ approval, backend, ledgerRoot }); await reserveInvocation(reservation(approval, backend));
  const file = path.join(ledgerRoot, 'state.json'), state = JSON.parse(await readFile(file, 'utf8')); state.generationCount = 0; await writeFile(file, JSON.stringify(state));
  await assert.rejects(auditLedger({ approval, backend, ledgerRoot }), /crash\/tamper/); await assert.rejects(reserveInvocation(reservation(approval, backend)), /crash\/tamper/);
});
test('runtime never initializes missing ledger; durable sibling claim forbids reset after directory deletion', async t => {
  const { approval } = await fixture(t), backend = 'mlx', ledgerRoot = approval.ledgerRoots[backend];
  await assert.rejects(reserveInvocation(reservation(approval, backend)), /missing\/busy/);
  await initializeLedger({ approval, backend, ledgerRoot }); await rm(ledgerRoot, { recursive: true });
  await assert.rejects(initializeLedger({ approval, backend, ledgerRoot }), /EEXIST/); await assert.rejects(reserveInvocation(reservation(approval, backend)), /missing\/busy/);
  await assert.rejects(reserveInvocation({ ...reservation(approval, backend), ledgerRoot: path.join(path.dirname(ledgerRoot), 'alternate-reset') }), /alternate ledger/);
});
test('only two fixed training lifecycles, one start and one planned resume each; reload still consumes generation permit', async t => {
  const { approval } = await fixture(t), backend = 'candle', training = approval.plan.trainingLifecycles.find(t => t.backend === backend); await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const base = { ...reservation(approval, backend), caseId: training.caseId };
  await assert.rejects(reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' }), /initial training start/);
  await reserveInvocation({ ...base, invocationId: training.startId, kind: 'training_start' }); await reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' });
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 0);
  const reload = approval.plan.generationInvocations.find(i => i.caseId === training.caseId); await reserveInvocation(reservation(approval, backend, reload));
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 1);
  await assert.rejects(reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' }), /already consumed/);
});
test('resource permits remain BLOCKED without real owner-verified single-call boundary; exact normalized request required', async t => {
  const { approval } = await fixture(t), backend = 'mlx', resource = approval.plan.generationInvocations.find(i => i.backend === backend && i.kind === 'resource_anchor'); await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  await assert.rejects(reserveInvocation(reservation(approval, backend, resource)), /boundary is unverified/);
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 0);
  // Separate DECLARED SYNTHETIC approved fixture, never a claim that native enforcement exists.
  const second = await fixture(t), a = second.approval; declaredSyntheticReadyBackend(a, backend); // full receipt for every resource row
  const invocation = a.plan.generationInvocations.find(i => i.caseId === resource.caseId); await initializeLedger({ approval: a, backend, ledgerRoot: a.ledgerRoots[backend] });
  await assert.rejects(reserveInvocation({ ...reservation(a, backend, invocation), requestSha256: 'e'.repeat(64) }), /exact normalized/);
  await reserveInvocation({ ...reservation(a, backend, invocation), requestSha256: invocation.requestSha256 }); assert.equal((await auditLedger({ approval: a, backend, ledgerRoot: a.ledgerRoots[backend] })).state.generationCount, 1);
});
test('aggregation validates complete ledger manifests, contiguous chains and final state; forged/truncated receipts never become zero-spend proof', async t => {
  const { approval } = await fixture(t), backend = 'candle', ledgerRoot = approval.ledgerRoots[backend]; await initializeLedger({ approval, backend, ledgerRoot }); await reserveInvocation(reservation(approval, backend));
  const receipt = await auditLedger({ approval, backend, ledgerRoot });
  await initializeLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx});const other=await auditLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx}),closure=heads(approval,[receipt,other]);
  assert.equal(aggregateLedgerReceipts(approval,[receipt,other],closure).caseCompletionProven,false);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt],closure),/partial backend/);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,other]),/trusted coordinator closure/);
  const rehash = r => { const {eventSha256,...event}=r.events[0];r.events[0].eventSha256=planHash(event);r.state.headSha256=r.events[0].eventSha256; };
  for (const mutate of [r=>r.events=[],r=>r.state.generationCount=38,r=>r.manifest.ledgerId='different-ledger',r=>r.manifest.backend='mlx',r=>r.manifest.trainingIds=[],r=>r.manifest.permitIds=[],r=>{r.events[0].ledgerId='different';rehash(r);},r=>{r.events[0].sequence=7;rehash(r);},r=>{r.events[0].previousSha256=digest;rehash(r);},r=>{r.events[0].approvedPlanSha256=digest;rehash(r);},r=>r.state.consumedIds=[]]) { const bad=structuredClone(receipt);mutate(bad);assert.throws(()=>aggregateLedgerReceipts(approval,[bad,other],closure)); }
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,receipt],closure),/duplicate ledger\/backend/);
  assert.throws(()=>aggregateLedgerReceipts(approval,[{state:{generationCount:38},events:[]},other],closure),/manifest/);
  const next=approval.plan.generationInvocations.filter(i=>i.backend===backend&&i.kind==='product_generation')[1];await reserveInvocation(reservation(approval,backend,next));const latest=await auditLedger({approval,backend,ledgerRoot}),latestClosure=heads(approval,[latest,other]);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,other],latestClosure),/replayed\/stale/);assert.equal(aggregateLedgerReceipts(approval,[latest,other],latestClosure).generationCount,2);
  for(const mutate of [c=>c.producersStopped=false,c=>c.ledgerId='wrong',c=>c.approvedPlanSha256=digest,c=>delete c.backends.mlx]){const bad=structuredClone(latestClosure);mutate(bad);assert.throws(()=>aggregateLedgerReceipts(approval,[latest,other],bad));}
});
test('adapter cases bind their own backend training export, on the trained checkpoint, strictly after training', async t => {
  const { approval } = await fixture(t); validateApprovedPlan(approval);
  const invocations = approval.plan.generationInvocations;
  for (const backend of ['mlx', 'candle']) {
    const reload = invocations.findIndex(i => i.backend === backend && i.purpose === 'training-reload');
    const adapters = invocations.map((i, index) => ({ i, index })).filter(({ i }) => i.backend === backend && i.adapterBinding);
    assert.deepEqual(adapters.map(({ i }) => i.caseId), [`${backend}.base.lora-zero.bf16`, `${backend}.base.lora-on.bf16`]);
    assert(adapters.every(({ i, index }) => index > reload && i.modelId === 'flux3_klein_base' && i.adapterBinding.lifecycleId === invocations[reload].caseId));
  }
  const first = invocations.findIndex(i => i.adapterBinding), find = (a, pred) => a.plan.generationInvocations.find(pred);
  for (const [mutate, pattern] of [
    // ORDERING mutation: the adapter case moved ahead of its LoRA's training lifecycle.
    [a => { const list = a.plan.generationInvocations; const [moved] = list.splice(first, 1); list.unshift(moved); }, /strictly after/],
    // Wrong checkpoint: a base-trained LoRA bound to a distilled adapter case is refused.
    [a => { find(a, i => i.adapterBinding).modelId = 'flux3_klein'; }, /different checkpoint/],
    [a => { find(a, i => i.adapterBinding).adapterBinding.modelId = 'flux3_klein'; }, /different checkpoint/],
    [a => { delete find(a, i => i.adapterBinding).adapterBinding; }, /must bind the plan's own exported LoRA/],
  ]) {
    const changed = structuredClone(approval); mutate(changed);
    assert.throws(() => validateApprovedPlan(changed), pattern);
  }
  const crossBackend = structuredClone(approval);
  find(crossBackend, i => i.adapterBinding && i.backend === 'candle').adapterBinding.lifecycleId = 'mlx.base.train-interrupt-resume-export-captioned.bf16';
  assert.throws(() => validateApprovedPlan(crossBackend), /own backend/);
});
test('both lifecycles are captioned; edit-LoRA training is an explicit uncovered NOT_RUN, never PASS', () => {
  assert.deepEqual(profile.trainingLifecycles.map(t => t.caseId).sort(), ['candle.base.train-interrupt-resume-export-captioned.bf16', 'mlx.base.train-interrupt-resume-export-captioned.bf16']);
  assert(!profile.cases.some(c => c.options.trainingMode === 'edit'), 'no paired edit data: no edit lifecycle is planned');
  const edit = profile.uncovered.find(u => /Edit-LoRA training/.test(u.requirement));
  assert.equal(edit?.status, 'NOT_RUN'); assert.match(edit.reason, /No paired edit data/);
  assert(profile.uncovered.every(u => u.status !== 'PASS'));
  assert.equal(profile.budget.trainingLifecycles, 2); assert.equal(profile.budget.stepsPerLifecycle, 4);
});

// Mocked producer executes ZERO compilers; full real verifier still checks owned files and
// checked-source producer/Cargo.lock hashes. No sourceRoot override or validation escape hatch.
async function appendTransitionFixture(t, {crossDevice = false} = {}) {
  const {root,approval:previousApproval}=await fixture(t), sourceRoot=fileURLToPath(new URL('..',import.meta.url));
  const manifest=await readFile(path.join(sourceRoot,'Cargo.toml'),'utf8'), inference=manifest.match(/rev = "([a-f0-9]{40})"/)[1];
  previousApproval.sourceSha=originalSource;previousApproval.inferenceMainSha=inference;
  previousApproval.ledgerRoots={mlx:path.join(root,'ledgers','mlx'),candle:path.join(root,'ledgers','candle')};await mkdir(path.join(root,'ledgers'));
  const output=path.join(root,'build-output'),targetDir=path.join(root,'warm-target');await mkdir(output);
  const tools=Object.fromEntries(['cargo','rustc','cl','nvcc'].map(name=>[name,path.join(root,`${name}.exe`)]));
  for(const [name,file] of Object.entries(tools)) await writeFile(file,`SYNTHETIC NEVER EXECUTED ${name}`);
  const hashes=Object.fromEntries(await Promise.all(Object.entries(tools).map(async([name,file])=>[name,rawSha(await readFile(file))])));
  const build=async(sourceSha,role)=>{
    const a={actualBuild:true,userBuildApprovalId:'SYNTHETIC_CPU_NOT_BUILD_AUTHORITY',resourceAdmissionId:'SYNTHETIC',sourceSha,inferenceMainSha:inference,backend:'candle',binaryRole:role,cargoExecutable:tools.cargo,cargoExecutableSha256:hashes.cargo,rustcExecutable:tools.rustc,rustcExecutableSha256:hashes.rustc,targetDir,buildOutputRoot:output,inferenceRepository:sourceRoot,cargoLockSha256:rawSha(await readFile(path.join(sourceRoot,'Cargo.lock'))),msvcVersion:'14.44',msvcCompilerExecutable:tools.cl,msvcCompilerSha256:hashes.cl,cudaCompilerExecutable:tools.nvcc,cudaCompilerSha256:hashes.nvcc,rustflags:'-C debuginfo=0'};
    const exec=async(command,args)=>{
      if(command==='git')return{stdout:args[0]==='rev-parse'?sourceSha:''};
      if(command===tools.rustc)return{stdout:'SYNTHETIC rustc\nhost: x86_64-pc-windows-msvc\n'};
      if(command===tools.cl)return{stdout:'Microsoft C/C++ Compiler Version 19.44.0000'};
      if(command===tools.nvcc)return{stdout:'Cuda compilation tools, release 12.9, V12.9.86'};
      assert.equal(command,tools.cargo);
      if(args[0]==='metadata')return{stdout:JSON.stringify({packages:[{name:'gen-core',source:`git+https://github.com/SceneWorks/inference?rev=${inference}#${inference}`}]})};
      assert.equal(args[0],'build');const bin=args[args.indexOf('--bin')+1],binary=path.join(targetDir,'x86_64-pc-windows-msvc','release',`${bin}.exe`);
      await mkdir(path.dirname(binary),{recursive:true});await writeFile(binary,`SYNTHETIC NEVER EXECUTED ${bin}`);
      return{stdout:[JSON.stringify({reason:'compiler-artifact',target:{name:bin,kind:['bin']},executable:binary}),JSON.stringify({reason:'build-finished',success:true})].join('\n'),stderr:''};
    };
    const produced=await produceTerminalBuild(a,{root:sourceRoot,platform:'win32',exec,env:{PATH:'SYNTHETIC'}});
    return {artifact:{path:produced.receiptPath,sha256:produced.receiptSha256},binding:{receiptSha256:produced.receiptSha256,transcriptSha256:produced.receipt.transcriptSha256,binarySha256:produced.receipt.binary.sha256,cargoLockSha256:a.cargoLockSha256,rustflagsSha256:rawSha(a.rustflags),cargoExecutableSha256:hashes.cargo,rustcExecutableSha256:hashes.rustc,msvcCompilerSha256:hashes.cl,cudaCompilerSha256:hashes.nvcc},receipt:produced.receipt};
  };
  const oldApi=await build(originalSource,'api-worker'),oldResource=await build(originalSource,'resource-adapter');
  previousApproval.buildReceipts={candle:oldApi.binding};declaredSyntheticReadyBackend(previousApproval,'candle',{binarySha256:oldResource.binding.binarySha256,buildReceiptSha256:oldResource.binding.receiptSha256});previousApproval.resourceBuildReceipts.candle=oldResource.binding;
  await initializeLedger({approval:previousApproval,backend:'candle',ledgerRoot:previousApproval.ledgerRoots.candle});
  const prepared=prepareReplayRequestFixture(previousApproval),first=previousApproval.plan.generationInvocations.find(i=>i.invocationId===consumedId);
  await reserveInvocation({...reservation(previousApproval,'candle',first),requestSha256:prepared.firstRequestSha256});
  const previousReceipt=await auditLedger({approval:previousApproval,backend:'candle',ledgerRoot:previousApproval.ledgerRoots.candle});
  const fixtureBaseline=await makeReplayBaselineFixture({root,approval:previousApproval,previousReceipt,...(crossDevice ? {historicalDevice:{gpuUuid:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',pciBusId:'00000000:C1:00.0'}} : {})});
  const approval=structuredClone(previousApproval);approval.sourceSha='c'.repeat(40);approval.approvalId='SYNTHETIC_NEW_FIXED_SOURCE';
  const newApi=await build(approval.sourceSha,'api-worker'),newResource=await build(approval.sourceSha,'resource-adapter');
  approval.buildReceipts.candle=newApi.binding;declaredSyntheticReadyBackend(approval,'candle',{binarySha256:newResource.binding.binarySha256,buildReceiptSha256:newResource.binding.receiptSha256});approval.resourceBuildReceipts.candle=newResource.binding;
  const save=async(name,value)=>{const file=path.join(root,name),json=JSON.stringify(value,null,2)+'\n';await writeFile(file,json);return{path:file,sha256:rawSha(json)};};
  const previousApprovalBinding=await save('previous-approval.json',previousApproval),previousReceiptBinding=await save('previous-final-receipt.json',previousReceipt);
  const claim=path.join(root,'ledgers',`.flux3-${approval.ledgerId}-candle.initialized`);
  const authorization={schemaVersion:1,kind:'flux3-candle-source-transition-authorization',authorizationId:'SYNTHETIC_ONLY_FIXED_CONTINUATION',approvedBy:'CPU test',producersStopped:true,previousSourceSha:originalSource,previousApprovedPlanSha256:planHash(previousApproval),previousHeadSha256:previousReceipt.state.headSha256,previousSequence:1,previousGenerationCount:1,previousConsumedIds:[consumedId],previousClaimSha256:rawSha(await readFile(claim)),previousApproval:previousApprovalBinding,previousReceipt:previousReceiptBinding,previousBuilds:{api:oldApi.artifact,resource:oldResource.artifact},nextBuilds:{api:newApi.artifact,resource:newResource.artifact},nextSourceSha:approval.sourceSha,nextApprovalBaseSha256:candleNextApprovalBaseSha256(approval),baseline:fixtureBaseline.baseline};
  const reseal=async()=>{const continuation=await save('parent-authorization.json',authorization);approval.candleSourceTransitionSha256=continuation.sha256;return{approval,ledgerRoot:approval.ledgerRoots.candle,continuation};};
  return {root,approval,previousApproval,previousReceipt,authorization,reseal,claim,oldApi,newApi,fixtureBaseline,args:await reseal()};
}
test('actual append path fully verifies synthetic owned producer outputs and CPU baseline before one idempotent transition',async t=>{
  const f=await appendTransitionFixture(t),root=f.args.ledgerRoot;
  const files=[path.join(root,'manifest.json'),path.join(root,'events','000001.json'),f.claim],bytes=await Promise.all(files.map(file=>readFile(file)));
  const result=await appendCandleSourceTransition(f.args);assert.equal(result.alreadyApplied,false);assert.equal(result.receipt.state.sequence,2);assert.equal(result.receipt.state.generationCount,1);assert.deepEqual(result.receipt.state.consumedIds,[consumedId]);
  const again=await appendCandleSourceTransition(f.args);assert.equal(again.alreadyApplied,true);assert.equal(again.eventSha256,result.eventSha256);
  for(const [index,file] of files.entries())assert.deepEqual(await readFile(file),bytes[index]);
  await reserveInvocation(reservation(f.approval,'candle',f.approval.plan.generationInvocations.find(i=>i.invocationId===nextId)));
  assert.equal((await auditLedger({approval:f.approval,backend:'candle',ledgerRoot:root})).state.generationCount,2);
  await assert.rejects(appendCandleSourceTransition(f.args));
});
test('append refuses authorized wrong old head/count/ID, wrong base/new source/unknown auth before journal changes',async t=>{
  const f=await appendTransitionFixture(t),original=structuredClone(f.authorization),root=f.args.ledgerRoot,stateBytes=await readFile(path.join(root,'state.json'));
  for(const mutate of [a=>a.previousHeadSha256=digest,a=>a.previousSequence=2,a=>a.previousGenerationCount=0,a=>a.previousConsumedIds=[nextId],a=>a.nextApprovalBaseSha256=digest,a=>a.nextSourceSha='f'.repeat(40),a=>a.extra='unknown',a=>a.baseline.allowedLineageRemap=['projectId','seed'],a=>a.previousClaimSha256=digest,a=>a.previousReceipt.sha256=digest,a=>a.baseline.png.sha256=digest]) {
    for(const key of Object.keys(f.authorization))delete f.authorization[key];Object.assign(f.authorization,structuredClone(original));mutate(f.authorization);
    await assert.rejects(appendCandleSourceTransition(await f.reseal()));
    await assert.rejects(readFile(path.join(root,'events','000002.json')),error=>error.code==='ENOENT');assert.deepEqual(await readFile(path.join(root,'state.json')),stateBytes);
  }
});
test('append rejects lexical foreign artifact paths before realpath and does not create a journal event',async t=>{
  const f=await appendTransitionFixture(t),root=f.args.ledgerRoot;
  const outside=path.resolve(f.root,'..','SYNTHETIC_NONEXISTENT_NOT_ACCESSED','parent-authorization.json');
  await assert.rejects(appendCandleSourceTransition({...f.args,continuation:{path:outside,sha256:f.args.continuation.sha256}}),/lexical owned root/);
  const original=structuredClone(f.authorization);
  for(const mutate of [a=>a.previousApproval.path=outside,a=>a.previousReceipt.path=outside,a=>a.previousBuilds.api.path=outside,a=>a.nextBuilds.resource.path=outside,a=>a.baseline.readback.path=outside]) {
    for(const key of Object.keys(f.authorization))delete f.authorization[key];Object.assign(f.authorization,structuredClone(original));mutate(f.authorization);
    await assert.rejects(appendCandleSourceTransition(await f.reseal()),/lexical owned root/);
    await assert.rejects(readFile(path.join(root,'events','000002.json')),error=>error.code==='ENOENT');
  }
});
test('append refuses changed baseline/binary bytes, journal partials and changed old latest state without repair',async t=>{
  const f=await appendTransitionFixture(t),root=f.args.ledgerRoot;
  for(const file of [f.authorization.baseline.png.path,f.authorization.previousReceipt.path,f.claim,f.newApi.receipt.binary.path]) {
    const bytes=await readFile(file);await writeFile(file,'SYNTHETIC TAMPER');await assert.rejects(appendCandleSourceTransition(f.args));await writeFile(file,bytes);
    await assert.rejects(readFile(path.join(root,'events','000002.json')),error=>error.code==='ENOENT');
  }
  const partial=path.join(root,'state-SYNTHETIC-PARTIAL.tmp');await writeFile(partial,'{}\n');await assert.rejects(appendCandleSourceTransition(f.args),/partial state/);await rm(partial);
  const stateFile=path.join(root,'state.json'),stateBytes=await readFile(stateFile);await writeFile(stateFile,JSON.stringify({...f.previousReceipt.state,generationCount:0}));await assert.rejects(appendCandleSourceTransition(f.args),/crash\/tamper/);await writeFile(stateFile,stateBytes);
  await writeFile(path.join(root,'events','000002.json'),'{}\n');await assert.rejects(appendCandleSourceTransition(f.args));
  assert.equal((JSON.parse(await readFile(path.join(root,'state.json'),'utf8'))).sequence,1);
});
const rawSha = bytes => createHash('sha256').update(bytes).digest('hex');
const originalSource = '7090f742a7d2973dc3cc1c144aa8a272cdc293c9';
const replayCase = 'candle.distilled.representative-repro.bf16';
const consumedId = `${replayCase}.generation-1`, nextId = `${replayCase}.generation-2`;
const oldApiBinary = 'd264bf21ba5e5a19012388041da92bc54764d2b3144c3d10065505b1a383e52d';
const oldResourceBinary = '4795a3e8fd5d0ee2d60c6586a72153759133209403d6b15120af7d50fcc48df5';
const previousSecurityLock = '96feb71e685fa37faa01509108824463d09e88ed057ac7079ba0f11ea2b60e92';
const nextSecurityLock = 'c33fba4fef58de889073666728c920a67c04687f7b24f4d4aaf5657cccd9a0c7';
async function pureTransitionFixture(t, {measurement = false, security = false} = {}) {
  const {root,approval:previousApproval} = await fixture(t);
  previousApproval.sourceSha = originalSource;
  previousApproval.ledgerRoots = {mlx:path.join(root,'ledgers','mlx'),candle:path.join(root,'ledgers','candle')};
  await mkdir(path.join(root,'ledgers'));
  previousApproval.buildReceipts = {candle:{receiptSha256:digest,transcriptSha256:digest,binarySha256:measurement ? oldApiBinary : digest,cargoLockSha256:digest}};
  declaredSyntheticReadyBackend(previousApproval,'candle',measurement ? {binarySha256:oldResourceBinary} : {});
  if (security) for(const key of ['buildReceipts','resourceBuildReceipts']) previousApproval[key].candle.cargoLockSha256=previousSecurityLock;
  await initializeLedger({approval:previousApproval,backend:'candle',ledgerRoot:previousApproval.ledgerRoots.candle});
  const first = previousApproval.plan.generationInvocations.find(i=>i.invocationId===consumedId);
  await reserveInvocation(reservation(previousApproval,'candle',first));
  const previousReceipt = await auditLedger({approval:previousApproval,backend:'candle',ledgerRoot:previousApproval.ledgerRoots.candle});
  const approval = structuredClone(previousApproval); approval.sourceSha='c'.repeat(40); approval.approvalId='SYNTHETIC_ONE_FIXED_CONTINUATION';
  approval.buildReceipts.candle.receiptSha256='e'.repeat(64); approval.buildReceipts.candle.transcriptSha256='f'.repeat(64);
  if (measurement) approval.buildReceipts.candle.binarySha256 = '1'.repeat(64);
  declaredSyntheticReadyBackend(approval,'candle',{buildReceiptSha256:'e'.repeat(64),...(measurement ? {binarySha256:'2'.repeat(64)} : {})});
  if (security) for(const key of ['buildReceipts','resourceBuildReceipts']) approval[key].candle.cargoLockSha256=nextSecurityLock;
  const binding = name => ({path:path.join(root,name),sha256:digest});
  const previousApprovalJson=JSON.stringify(previousApproval,null,2)+'\n', previousReceiptJson=JSON.stringify(previousReceipt,null,2)+'\n';
  const authorization={schemaVersion:1,kind:'flux3-candle-source-transition-authorization',authorizationId:'SYNTHETIC_ONLY',approvedBy:'synthetic CPU test',producersStopped:true,previousSourceSha:originalSource,previousApprovedPlanSha256:planHash(previousApproval),previousHeadSha256:previousReceipt.state.headSha256,previousSequence:1,previousGenerationCount:1,previousConsumedIds:[consumedId],previousClaimSha256:digest,previousApproval:{...binding('old-approval.json'),sha256:rawSha(previousApprovalJson)},previousReceipt:{...binding('old-receipt.json'),sha256:rawSha(previousReceiptJson)},previousBuilds:{api:binding('old-api.json'),resource:binding('old-resource.json')},nextBuilds:{api:binding('new-api.json'),resource:binding('new-resource.json')},nextSourceSha:approval.sourceSha,nextApprovalBaseSha256:candleNextApprovalBaseSha256(approval),baseline:{caseId:replayCase,consumedInvocationId:consumedId,nextInvocationId:nextId,png:binding('old.png'),sidecar:binding('sidecar.json'),report:binding('report.json'),readback:binding('readback.json'),firstRequestSha256:digest,secondRecordedRequestSha256:digest,oldJobId:'synthetic-job',oldWorkerId:'synthetic-worker',oldWorkerPid:123,allowedLineageRemap:['projectId']}};
  const authorizationJson=JSON.stringify(authorization,null,2)+'\n'; approval.candleSourceTransitionSha256=rawSha(authorizationJson);
  const record={sequence:2,previousSha256:previousReceipt.state.headSha256,approvedPlanSha256:planHash(approval),ledgerId:approval.ledgerId,backend:'candle',kind:'candle_source_transition',previousApprovalJson,previousReceiptJson,authorizationJson,authorizationSha256:rawSha(authorizationJson)};
  const event={...record,eventSha256:planHash(record)};
  const receipt={...structuredClone(previousReceipt),events:[...previousReceipt.events,event],state:{...previousReceipt.state,sequence:2,headSha256:event.eventSha256}};
  return {root,approval,previousApproval,previousReceipt,receipt,event,authorization};
}
test('fixed source prefix authenticates old and new history, preserves one old spend, then cumulative generation two',async t=>{
  const f=await pureTransitionFixture(t), {approval,previousReceipt,receipt}=f, ledgerRoot=approval.ledgerRoots.candle;
  assert.equal(validateCandleSourceCompatibility(f.previousApproval,approval),true);
  assert.equal(verifyLedgerReceipt(approval,receipt).state.generationCount,1);
  assert.equal(candleReplayPrefix(receipt,approval).consumedInvocationId,consumedId);
  const manifestBytes=await readFile(path.join(ledgerRoot,'manifest.json')),eventBytes=await readFile(path.join(ledgerRoot,'events','000001.json'));
  const claim=path.join(path.dirname(ledgerRoot),`.flux3-${approval.ledgerId}-candle.initialized`), claimBytes=await readFile(claim);
  // Synthetic carried event only: production append separately requires full builds and baseline.
  await writeFile(path.join(ledgerRoot,'events','000002.json'),JSON.stringify(f.event)+'\n');
  await writeFile(path.join(ledgerRoot,'state.json'),JSON.stringify(receipt.state)+'\n');
  await assert.rejects(reserveInvocation(reservation(approval,'candle',approval.plan.generationInvocations.find(i=>i.invocationId===consumedId))),/already consumed/);
  await reserveInvocation(reservation(approval,'candle',approval.plan.generationInvocations.find(i=>i.invocationId===nextId)));
  const current=await auditLedger({approval,backend:'candle',ledgerRoot});
  assert.equal(current.state.sequence,3); assert.equal(current.state.generationCount,2);
  assert.deepEqual(current.state.consumedIds,[consumedId,nextId]);
  assert.deepEqual(candleReplayPrefix(current,approval).previousReceipt,previousReceipt);
  for(const [file,bytes] of [[path.join(ledgerRoot,'manifest.json'),manifestBytes],[path.join(ledgerRoot,'events','000001.json'),eventBytes],[claim,claimBytes]]) assert.deepEqual(await readFile(file),bytes);
  await initializeLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx});
  const mlx=await auditLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx});
  assert.equal(aggregateLedgerReceipts(approval,[current,mlx],heads(approval,[current,mlx])).generationCount,2);
});
test('root authorization pin refuses fresh Candle initialization and every missing, unknown or repeated transition',async t=>{
  const {approval,previousApproval,previousReceipt,receipt,event}=await pureTransitionFixture(t);
  await assert.rejects(initializeLedger({approval,backend:'candle',ledgerRoot:approval.ledgerRoots.candle}),/cannot initialize/);
  assert.throws(()=>verifyLedgerReceipt(approval,previousReceipt),/transition required/);
  const fresh={manifest:{...receipt.manifest,approvedPlanSha256:planHash(approval),approvalId:approval.approvalId},state:{sequence:0,generationCount:0,consumedIds:[],headSha256:digest},events:[]};
  assert.throws(()=>verifyLedgerReceipt(approval,fresh),/transition required/);
  assert.throws(()=>verifyLedgerReceipt(previousApproval,receipt),/unknown\/unapproved/);
  const doubled=structuredClone(receipt);doubled.events.push(event); assert.throws(()=>verifyLedgerReceipt(approval,doubled),/exactly once/);
  const random=structuredClone(approval); random.candleSourceTransitionSha256='0'.repeat(64);assert.throws(()=>verifyLedgerReceipt(random,receipt));
});
test('next approval base hashes all unknown/nested fields and ignores ONLY its root authorization pin',async t=>{
  const {approval,previousApproval}=await pureTransitionFixture(t), base=candleNextApprovalBaseSha256(approval);
  const differentPin=structuredClone(approval); differentPin.candleSourceTransitionSha256='0'.repeat(64);assert.equal(candleNextApprovalBaseSha256(differentPin),base);
  const reversed=Object.fromEntries(Object.entries(approval).reverse());assert.equal(candleNextApprovalBaseSha256(reversed),base);
  for(const mutate of [a=>a.extra='unknown',a=>a.plan.candleSourceTransitionSha256=digest,a=>a.inputSpecSha256.candle='0'.repeat(64),a=>a.ledgerRoots.candle+='-reset',a=>a.inferenceMainSha='0'.repeat(40),a=>a.plan.generationInvocations[0].options.extra=true,a=>a.plan.generationInvocations.reverse(),a=>a.buildReceipts.candle.binarySha256='0'.repeat(64),a=>a.plan.trainingLifecycles[0].extra=true,a=>a.plan.cases[0].options.extra=true]) {
    const bad=structuredClone(approval);mutate(bad);assert.notEqual(candleNextApprovalBaseSha256(bad),base);assert.throws(()=>validateCandleSourceCompatibility(previousApproval,bad));
  }
  const second=structuredClone(previousApproval);second.candleSourceTransitionSha256=digest;assert.throws(()=>validateCandleSourceCompatibility(second,approval),/second/);
});
test('carried raw authorization, old approval, receipt, state, and native proof mutations all fail closed',async t=>{
  const {approval,receipt}=await pureTransitionFixture(t);
  const rehash=r=>{const {eventSha256,...record}=r.events[1];r.events[1].eventSha256=planHash(record);r.state.headSha256=r.events[1].eventSha256;};
  for(const mutate of [r=>r.events[0].requestSha256='0'.repeat(64),r=>r.events[1].previousApprovalJson+=' ',r=>r.events[1].previousReceiptJson+=' ',r=>r.events[1].authorizationJson+=' ',r=>r.events[1].kind='product_generation',r=>r.events[1].invocationId=nextId,r=>r.state.generationCount=0,r=>r.state.generationCount=3,r=>r.state.sequence=1,r=>r.events[1].previousSha256=digest]) {
    const bad=structuredClone(receipt);mutate(bad);rehash(bad);assert.throws(()=>verifyLedgerReceipt(approval,bad));
  }
  for(const mutate of [a=>a.plan.nativeBoundaryReceipts.candle.sourceSha='0'.repeat(40),a=>a.plan.nativeBoundaryReceipts.candle.cases[0].requestSha256=digest,a=>a.plan.resourceCases.find(r=>r.backend==='candle').nativeBoundaryReceiptSha256=digest]) {const bad=structuredClone(approval);mutate(bad);assert.throws(()=>verifyLedgerReceipt(bad,receipt));}
});

function bindDeviceTransition(f) {
  const a = f.authorization;
  f.approval.candleDeviceTransition = {schemaVersion:1,
    ...Object.fromEntries(['previousSourceSha','nextSourceSha','previousApprovedPlanSha256','previousHeadSha256','previousSequence','previousGenerationCount','previousConsumedIds','previousClaimSha256'].map(k => [k, structuredClone(a[k])])),
    previousDevice:{physicalGpuId:'1',gpuUuid:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',pciBusId:'00000000:C1:00.0'},
    nextDevice:{physicalGpuId:'0',gpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',pciBusId:'00000000:21:00.0'}};
  a.deviceTransition = structuredClone(f.approval.candleDeviceTransition);
  a.nextApprovalBaseSha256 = candleNextApprovalBaseSha256(f.approval);
}
function bindMeasurementTransition(f) {
  bindDeviceTransition(f);
  const builds = approval => Object.fromEntries([['api','buildReceipts'],['resource','resourceBuildReceipts']].map(([role,key]) => [role, {binarySha256:approval[key].candle.binarySha256,receiptSha256:approval[key].candle.receiptSha256}]));
  f.approval.candleMeasurementTransition = {schemaVersion:1,previousSourceSha:originalSource,nextSourceSha:f.approval.sourceSha,
    measurementKind:'windows-owned-process-accounting-v1',reason:'measurement-only owned-process producer/consumer accounting; renderer and inference unchanged',
    previousBuilds:builds(f.previousApproval),nextBuilds:builds(f.approval)};
  f.authorization.measurementTransition = structuredClone(f.approval.candleMeasurementTransition);
  for (const side of ['previousBuilds','nextBuilds']) for (const role of ['api','resource']) f.authorization[side][role].sha256 = f.approval.candleMeasurementTransition[side][role].receiptSha256;
}
function bindSecurityTransition(f) {
  bindMeasurementTransition(f);
  f.approval.candleMeasurementTransition.dependencySecurityUpdate={schemaVersion:1,advisory:'RUSTSEC-2026-0386',package:'tokio',previousVersion:'1.53.1',nextVersion:'1.53.3',previousSourceSha:originalSource,nextSourceSha:f.approval.sourceSha,previousCargoLockSha256:previousSecurityLock,nextCargoLockSha256:nextSecurityLock};
  f.approval.candleMeasurementTransition.reason='measurement owned-process producer/consumer accounting and RUSTSEC-2026-0386 Tokio security update; renderer source and inference pin unchanged';
  f.authorization.measurementTransition=structuredClone(f.approval.candleMeasurementTransition);
  f.authorization.previousCargoLock={path:path.join(f.root,'historical-source-7090','Cargo.lock'),sha256:previousSecurityLock};
}
function resealCarriedTransition(f) {
  f.authorization.nextApprovalBaseSha256 = candleNextApprovalBaseSha256(f.approval);
  const authorizationJson = JSON.stringify(f.authorization, null, 2) + '\n';
  f.approval.candleSourceTransitionSha256 = rawSha(authorizationJson);
  const {eventSha256, ...record} = f.receipt.events[1];
  Object.assign(record, {authorizationJson, authorizationSha256:rawSha(authorizationJson), approvedPlanSha256:planHash(f.approval)});
  f.receipt.events[1] = {...record,eventSha256:planHash(record)};
  f.receipt.state.headSha256 = f.receipt.events[1].eventSha256;
}
test('strict device transition binds destination and fixed old prefix while legacy format stays unchanged', async t => {
  const f = await pureTransitionFixture(t);
  assert.equal(validateCandleDeviceTransitionBinding(f.approval), undefined);
  bindDeviceTransition(f); resealCarriedTransition(f);
  assert.deepEqual(validateCandleDeviceTransitionBinding(f.approval), f.authorization.deviceTransition);
  assert.equal(validateCandleSourceCompatibility(f.previousApproval,f.approval,f.authorization,f.previousReceipt),true);
  assert.throws(() => validateCandleSourceCompatibility(f.previousApproval,f.approval));
  assert.equal(verifyLedgerReceipt(f.approval,f.receipt).state.generationCount,1);
  assert.equal(candleReplayPrefix(f.receipt,f.approval).authorization.deviceTransition.nextDevice.physicalGpuId,'0');
  await assert.rejects(initializeLedger({approval:f.approval,backend:'candle',ledgerRoot:f.approval.ledgerRoots.candle}),/cannot initialize/);
  const second = structuredClone(f.approval); second.sourceSha = 'f'.repeat(40);
  assert.throws(() => validateCandleSourceCompatibility(f.approval,second,f.authorization,f.receipt));
  // Destination source is the actual approval, not a hardcoded future merge SHA.
  const another = await pureTransitionFixture(t); another.approval.sourceSha = 'e'.repeat(40);
  declaredSyntheticReadyBackend(another.approval,'candle',{buildReceiptSha256:'e'.repeat(64)});
  another.authorization.nextSourceSha = another.approval.sourceSha;
  bindDeviceTransition(another); resealCarriedTransition(another);
  assert.equal(verifyLedgerReceipt(another.approval,another.receipt).state.generationCount,1);
});
test('device transition rejects physical index UUID PCI swaps unknown fields and mismatched authorization even when rehashed', async t => {
  const original = await pureTransitionFixture(t); bindDeviceTransition(original); resealCarriedTransition(original);
  const mutations = [];
  for (const field of ['previousDevice','nextDevice']) for (const key of ['physicalGpuId','gpuUuid','pciBusId']) {
    mutations.push(f => {f.approval.candleDeviceTransition[field][key] = 'WRONG'; f.authorization.deviceTransition = structuredClone(f.approval.candleDeviceTransition);});
  }
  mutations.push(
    f => {const b=f.approval.candleDeviceTransition; [b.previousDevice,b.nextDevice]=[b.nextDevice,b.previousDevice]; f.authorization.deviceTransition=structuredClone(b);},
    f => f.authorization.deviceTransition.nextDevice.physicalGpuId='1',
    f => delete f.authorization.deviceTransition,
    f => delete f.approval.candleDeviceTransition,
    f => f.approval.candleDeviceTransition.extra=true,
    f => f.approval.candleDeviceTransition.nextDevice.extra=true,
    f => f.authorization.deviceTransition.extra=true,
    f => f.authorization.extra=true,
    f => f.approval.unrecognizedDeviceTransition={physicalGpuId:'0'},
    f => f.approval.candleDeviceTransition.nextSourceSha='f'.repeat(40),
    f => {f.authorization.nextSourceSha='f'.repeat(40); f.authorization.deviceTransition.nextSourceSha='f'.repeat(40);},
    f => {f.approval.candleDeviceTransition.previousSourceSha='f'.repeat(40); f.authorization.deviceTransition=structuredClone(f.approval.candleDeviceTransition);},
  );
  for (const mutate of mutations) {
    const f=structuredClone(original); mutate(f); resealCarriedTransition(f);
    assert.throws(() => verifyLedgerReceipt(f.approval,f.receipt));
  }
  for (const mutate of [a=>a.plan.cases[0].options.extra=true,a=>a.inputSpecSha256.candle='0'.repeat(64),a=>a.ledgerRoots.candle+='-reset',a=>a.buildReceipts.candle.binarySha256='0'.repeat(64)]) {
    const f=structuredClone(original);mutate(f.approval);resealCarriedTransition(f);
    assert.throws(() => verifyLedgerReceipt(f.approval,f.receipt));
  }
});
test('device binding cannot substitute old prefix plan head count source or claim provenance', async t => {
  const original=await pureTransitionFixture(t);bindDeviceTransition(original);resealCarriedTransition(original);
  for (const [key,value] of [['previousApprovedPlanSha256','0'.repeat(64)],['previousHeadSha256','0'.repeat(64)],['previousSequence',0],['previousGenerationCount',0],['previousConsumedIds',[nextId]],['previousSourceSha','f'.repeat(40)]]) {
    const f=structuredClone(original);f.approval.candleDeviceTransition[key]=value;f.authorization[key]=value;f.authorization.deviceTransition[key]=value;resealCarriedTransition(f);
    assert.throws(() => verifyLedgerReceipt(f.approval,f.receipt));
  }
  const claim=structuredClone(original);claim.approval.candleDeviceTransition.previousClaimSha256='0'.repeat(64);claim.authorization.deviceTransition.previousClaimSha256='0'.repeat(64);resealCarriedTransition(claim);
  assert.throws(() => verifyLedgerReceipt(claim.approval,claim.receipt),/previousClaimSha256/);
  const swapped=await pureTransitionFixture(t);bindDeviceTransition(swapped);resealCarriedTransition(swapped);
  const f=structuredClone(original);f.authorization.deviceTransition=swapped.authorization.deviceTransition;resealCarriedTransition(f);
  assert.throws(() => verifyLedgerReceipt(f.approval,f.receipt),/differs from approval/);
});
test('fixed measurement-only binary exception authenticates both exact role receipts under one device/source transition', async t => {
  const f=await pureTransitionFixture(t,{measurement:true});
  // Without the explicit exception even an otherwise approved binary change is refused.
  bindDeviceTransition(f);resealCarriedTransition(f);
  assert.throws(()=>verifyLedgerReceipt(f.approval,f.receipt),/immutable/);
  bindMeasurementTransition(f);resealCarriedTransition(f);
  assert.deepEqual(validateCandleMeasurementTransitionBinding(f.approval),f.authorization.measurementTransition);
  assert.equal(verifyLedgerReceipt(f.approval,f.receipt).state.generationCount,1);
  assert.equal(candleReplayPrefix(f.receipt,f.approval).authorization.measurementTransition.measurementKind,'windows-owned-process-accounting-v1');
  const second=structuredClone(f.approval);second.sourceSha='f'.repeat(40);
  assert.throws(()=>validateCandleSourceCompatibility(f.approval,second,f.authorization,f.receipt));
  await assert.rejects(initializeLedger({approval:f.approval,backend:'candle',ledgerRoot:f.approval.ledgerRoots.candle}),/cannot initialize/);
});
test('measurement transition rejects unknown fields generic remaps wrong role receipts source kind and renderer changes', async t => {
  const original=await pureTransitionFixture(t,{measurement:true});bindMeasurementTransition(original);resealCarriedTransition(original);
  const mutations=[
    f=>delete f.authorization.measurementTransition,
    f=>delete f.approval.candleMeasurementTransition,
    f=>delete f.approval.candleDeviceTransition,
    f=>f.approval.candleMeasurementTransition.extra=true,
    f=>f.approval.candleMeasurementTransition.nextBuilds.api.extra=true,
    f=>f.authorization.measurementTransition.extra=true,
    f=>f.approval.candleMeasurementTransition.measurementKind='device-delta-stable-background-with-owned-process-accounting-v1',
    f=>f.approval.candleMeasurementTransition.reason='arbitrary binary replacement',
    f=>f.approval.candleMeasurementTransition.previousSourceSha='f'.repeat(40),
    f=>f.approval.candleMeasurementTransition.nextSourceSha='f'.repeat(40),
    f=>f.authorization.previousBuilds.api.sha256='0'.repeat(64),
    f=>f.authorization.nextBuilds.resource.sha256='0'.repeat(64),
    f=>{const b=f.approval.candleMeasurementTransition.nextBuilds;[b.api,b.resource]=[b.resource,b.api];},
    f=>f.approval.plan.cases[0].options.extra=true,
    f=>f.approval.buildReceipts.candle.cargoLockSha256='0'.repeat(64),
    f=>f.approval.inferenceMainSha='f'.repeat(40),
  ];
  for(const side of ['previousBuilds','nextBuilds'])for(const role of ['api','resource'])for(const key of ['binarySha256','receiptSha256'])mutations.push(f=>{f.approval.candleMeasurementTransition[side][role][key]='0'.repeat(64);f.authorization.measurementTransition=structuredClone(f.approval.candleMeasurementTransition);});
  for(const mutate of mutations){const f=structuredClone(original);mutate(f);resealCarriedTransition(f);assert.throws(()=>verifyLedgerReceipt(f.approval,f.receipt));}
  const f=structuredClone(original);f.approval.candleMeasurementTransition.previousBuilds.api.receiptSha256='0'.repeat(64);f.authorization.measurementTransition=structuredClone(f.approval.candleMeasurementTransition);f.authorization.previousBuilds.api.sha256='0'.repeat(64);resealCarriedTransition(f);
  assert.throws(()=>verifyLedgerReceipt(f.approval,f.receipt),/old approval/);
});
test('fixed Tokio security update binds exact old/new sources locks receipts and immutable Git archive', async t => {
  const f=await pureTransitionFixture(t,{measurement:true,security:true});bindSecurityTransition(f);resealCarriedTransition(f);
  assert.equal(validateCandleDependencySecurityUpdate(f.approval).advisory,'RUSTSEC-2026-0386');
  assert.equal(verifyLedgerReceipt(f.approval,f.receipt).state.generationCount,1);
  const sourceRoot=fileURLToPath(new URL('..',import.meta.url));
  const oldBytes=execFileSync('git',['-C',sourceRoot,'show',`${originalSource}:Cargo.lock`],{maxBuffer:16*1024*1024});
  // The recorded security transition is the historical lock plus EXACTLY the fixed Tokio update.
  // The working tree's lock legitimately advanced its inference pin when main was merged into the
  // FLUX 3 feature branch, which is outside this transition's claim (and is covered by the bump's
  // own closure/skew gates). Reconstruct the transition target instead of freezing the live lock.
  const nextBytes=Buffer.from(oldBytes.toString()
    .replace('version = "1.53.1"','version = "1.53.3"')
    .replace('checksum = "202caea871b69668250d242070849eb495be178ed697a3e98aebce5bc81a0bed"','checksum = "ce3335fa71841cda333a58d7615b03901380ecf09d59b3296d21f8bbac0dde4e"'));
  const liveLock=await readFile(path.join(sourceRoot,'Cargo.lock'),'utf8');
  assert.match(liveLock,/name = "tokio"\nversion = "1\.53\.3"/,'the live lock retains the fixed Tokio release');
  assert.equal(validateCandleDependencySecurityLocks(oldBytes,nextBytes),true);
  await mkdir(path.dirname(f.authorization.previousCargoLock.path));await writeFile(f.authorization.previousCargoLock.path,oldBytes);
  assert.equal(await verifyCandleDependencySecurityArchive(f.approval,f.authorization,f.root),await realpath(path.dirname(f.authorization.previousCargoLock.path)));
  for(const bytes of [nextBytes,Buffer.from(oldBytes.toString().replace('1.53.1','1.53.2')),Buffer.concat([oldBytes,Buffer.from('\n')])]) {
    await writeFile(f.authorization.previousCargoLock.path,bytes);
    await assert.rejects(verifyCandleDependencySecurityArchive(f.approval,f.authorization,f.root));
  }
  await writeFile(f.authorization.previousCargoLock.path,oldBytes);
  for(const bytes of [oldBytes,Buffer.from(nextBytes.toString().replace('1.53.3','1.53.4')),Buffer.from(nextBytes.toString().replace('name = "tokio-macros"','name = "unexpected-macro"')),Buffer.concat([nextBytes,Buffer.from('\n')])])assert.throws(()=>validateCandleDependencySecurityLocks(oldBytes,bytes));
});
test('security transition rejects stripped exception unknown keys reason advisory versions source and arbitrary lock remaps', async t => {
  const original=await pureTransitionFixture(t,{measurement:true,security:true});bindSecurityTransition(original);resealCarriedTransition(original);
  const mutations=[
    f=>delete f.authorization.previousCargoLock,
    f=>f.authorization.previousCargoLock.extra=true,
    f=>f.authorization.previousCargoLock.sha256=nextSecurityLock,
    f=>f.authorization.previousCargoLock.path=path.join(f.root,'not-Cargo.lock'),
    f=>f.authorization.measurementTransition.dependencySecurityUpdate.extra=true,
    f=>{delete f.approval.candleMeasurementTransition.dependencySecurityUpdate;delete f.authorization.measurementTransition.dependencySecurityUpdate;f.approval.candleMeasurementTransition.reason=f.authorization.measurementTransition.reason='measurement-only owned-process producer/consumer accounting; renderer and inference unchanged';delete f.authorization.previousCargoLock;},
    f=>f.approval.candleMeasurementTransition.reason='measurement-only owned-process producer/consumer accounting; renderer and inference unchanged',
    f=>f.approval.buildReceipts.candle.cargoLockSha256=previousSecurityLock,
    f=>f.approval.resourceBuildReceipts.candle.cargoLockSha256='0'.repeat(64),
    f=>f.approval.resourceBuildReceipts.candle.manifestSha256='0'.repeat(64),
  ];
  for(const key of ['advisory','package','previousVersion','nextVersion','previousSourceSha','nextSourceSha','previousCargoLockSha256','nextCargoLockSha256'])mutations.push(f=>{f.approval.candleMeasurementTransition.dependencySecurityUpdate[key]='unknown';f.authorization.measurementTransition=structuredClone(f.approval.candleMeasurementTransition);});
  for(const mutate of mutations){const f=structuredClone(original);mutate(f);resealCarriedTransition(f);assert.throws(()=>verifyLedgerReceipt(f.approval,f.receipt));}
  await assert.rejects(initializeLedger({approval:original.approval,backend:'candle',ledgerRoot:original.approval.ledgerRoots.candle}),/cannot initialize/);
});
test('cross-device append preserves old claim and refuses forged claim and reset without ledger changes', async t => {
  const f=await appendTransitionFixture(t,{crossDevice:true});bindDeviceTransition(f);
  const originalClaim=f.authorization.previousClaimSha256;
  f.authorization.previousClaimSha256='0'.repeat(64);f.authorization.deviceTransition.previousClaimSha256='0'.repeat(64);f.approval.candleDeviceTransition.previousClaimSha256='0'.repeat(64);
  f.authorization.nextApprovalBaseSha256=candleNextApprovalBaseSha256(f.approval);
  await assert.rejects(appendCandleSourceTransition(await f.reseal()),/raw bytes mismatch/);
  await assert.rejects(readFile(path.join(f.args.ledgerRoot,'events','000002.json')),e=>e.code==='ENOENT');
  f.authorization.previousClaimSha256=originalClaim;f.authorization.deviceTransition.previousClaimSha256=originalClaim;f.approval.candleDeviceTransition.previousClaimSha256=originalClaim;
  f.authorization.nextApprovalBaseSha256=candleNextApprovalBaseSha256(f.approval);
  const args=await f.reseal(), claimBytes=await readFile(f.claim), result=await appendCandleSourceTransition(args);
  assert.equal(result.receipt.state.sequence,2);assert.equal(result.receipt.state.generationCount,1);
  assert.deepEqual(await readFile(f.claim),claimBytes);
  assert.equal((await appendCandleSourceTransition(args)).alreadyApplied,true);
  await assert.rejects(initializeLedger({approval:f.approval,backend:'candle',ledgerRoot:path.join(f.root,'reset')}),/alternate ledger/);
});

test('same-physical-device destination is accepted only as the exact historical GPU1 tuple', async t => {
  const f = await pureTransitionFixture(t);
  bindDeviceTransition(f);
  f.approval.candleDeviceTransition.nextDevice = structuredClone(f.approval.candleDeviceTransition.previousDevice);
  f.authorization.deviceTransition = structuredClone(f.approval.candleDeviceTransition);
  resealCarriedTransition(f);
  assert.equal(validateCandleSourceCompatibility(f.previousApproval,f.approval,f.authorization,f.previousReceipt),true);
  assert.deepEqual(validateCandleDeviceTransitionBinding(f.approval).nextDevice, {physicalGpuId:'1',gpuUuid:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',pciBusId:'00000000:C1:00.0'});
  assert.equal(candleReplayPrefix(f.receipt,f.approval).authorization.deviceTransition.nextDevice.physicalGpuId,'1');
  // Mixed UUID/PCI tuples, arbitrary devices and a changed historical device are still refused.
  for (const [label, mutate] of [
    ['GPU1 id with the GPU0 UUID', x => { x.approval.candleDeviceTransition.nextDevice.gpuUuid = 'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0'; }],
    ['GPU1 id with the GPU0 PCI', x => { x.approval.candleDeviceTransition.nextDevice.pciBusId = '00000000:21:00.0'; }],
    ['arbitrary device tuple', x => { x.approval.candleDeviceTransition.nextDevice = {physicalGpuId:'2',gpuUuid:'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff',pciBusId:'00000000:41:00.0'}; }],
    ['changed historical device', x => { x.approval.candleDeviceTransition.previousDevice = {physicalGpuId:'0',gpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',pciBusId:'00000000:21:00.0'}; }],
  ]) {
    const bad = structuredClone(f); mutate(bad);
    bad.authorization.deviceTransition = structuredClone(bad.approval.candleDeviceTransition);
    resealCarriedTransition(bad);
    assert.throws(() => verifyLedgerReceipt(bad.approval, bad.receipt), undefined, label);
  }
});
