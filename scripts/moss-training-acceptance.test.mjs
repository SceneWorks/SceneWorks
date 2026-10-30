import {isPreparationCommand, PRE_MODEL_CPU_TEST} from "./moss-native-api-validation.mjs";
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';
import { recordInspection, parseInspectionArgs } from './moss-preview-inspection.mjs';
import { digest } from './moss-consumer-assertions.mjs';
import { inventoryMossTrainingCache } from './moss-training-cache-inventory.mjs';
import { BOUNDED_PRODUCT, BOUNDED_GPU_UUID, BOUNDED_GPU_CAP_BYTES, BOUNDED_GPU_RESERVE_BYTES, checkBoundedDiscovery, validateBoundedClaim, validateBoundedSample, boundedReaderBindings, MANIFEST_SHA256, PREFIX, PREFLIGHT, CELLS, BOUNDS, NATIVE_TEST_STACK_BYTES, testsFor, parseArgs, artifactMembers, verifyBundle, checkDiscovery, validatePreflight, validateNativeRoot, executeSupervised, remainingCaseMs, fileDigestChecked, validatePreflightChecked, decodePreviewPng, run } from './moss-training-acceptance.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const fixture=t=>{const dir=mkdtempSync(path.join(os.tmpdir(),'moss-harness-unit-'));t.after(()=>{rmSync(dir,{recursive:true,force:true});rmSync(`${dir}-selected-artifacts`,{recursive:true,force:true});});return dir;};
const source={workflowSource:'a'.repeat(40),inferencePin:'b'.repeat(40),manifestSha256:MANIFEST_SHA256,cargoLockSha256:'c'.repeat(64),driverSha256:'d'.repeat(64),workflowSha256:'e'.repeat(64)};
const entries=(backend='mlx')=>[...testsFor('capture',backend),...testsFor('cross-load',backend),PREFLIGHT].map(name=>({name,kind:'test'}));
const goodOutput=exact=>`running 1 test\ntest ${exact} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n`;

test('sealed manifest hash and fixed bounds are source contracts',()=>{
 assert.equal(digest(readFileSync(path.join(root,'scripts/ci/moss-training-acceptance/campaign.json'))),MANIFEST_SHA256);
 assert.equal(BOUNDS.case,3600_000);assert.equal(BOUNDS.build,7200_000);
});
test('closed four cases in each phase, not fuzzy filters',()=>{
 assert.equal(CELLS.length,4);assert.equal(new Set([...testsFor('capture'),...testsFor('cross-load')]).size,8);
 assert.deepEqual(checkDiscovery(entries(),entries()).sort(),entries().map(e=>e.name).sort());
});
test('discovery rejects missing, extra, duplicate and not ignored exact FQNs',()=>{
 for(const list of [entries().slice(1),[...entries(),{name:PREFIX+'unexpected',kind:'test'}],[...entries(),entries()[0]]])assert.throws(()=>checkDiscovery(list,entries()),/discovery/);
 assert.throws(()=>checkDiscovery(entries(),entries().slice(1)),/ignored/);
});
test('ordinary metadata contracts remain outside dispatch-only discovery inventory',()=>{
 const name='tests::moss_training_acceptance_contracts::weight_preflight_uses_complete_production_component_map_without_model_execution';
 assert.equal(name.startsWith(PREFIX),false);
 assert.equal(checkDiscovery([...entries(),{name,kind:'test'}],entries()).length,9);
});
test('argument parser requires explicit closed phase and paths',()=>{
 const args=['--phase','capture','--backend','mlx','--root','/source','--output','/output','--manifest','/manifest'];
 assert.equal(parseArgs(args).phase,'capture');
 assert.throws(()=>parseArgs([...args,'--phase','finalize']),/unique/);
 assert.throws(()=>parseArgs(args.map(v=>v==='capture'?'cross-load':v)),/peer-artifacts/);
 assert.throws(()=>parseArgs(args.map(v=>v==='/output'?'/source/out':v)),/outside/);
});
test('absolute deadline charges elapsed and queue time, rejects future and expired',()=>{
 const now=Date.now();assert.equal(remainingCaseMs(new Date(now-1234).toISOString(),now),BOUNDS.case-1234);
 assert.throws(()=>remainingCaseMs(new Date(now-BOUNDS.case).toISOString(),now),/expired/);
 assert.throws(()=>remainingCaseMs(new Date(now+1).toISOString(),now),/future/);
});
test('preflight uses selected actual capacity and native provenance, not env asserted cap',async t=>{
 const dir=fixture(t),weight=path.join(dir,'cached');writeFileSync(weight,'cached header');
 const p={...source,phase:'capture',observationSchema:{gradientObservations:true,fixedEvaluationPair:true,numericFactorDelta:true,preparedTensorSnapshots:true},backend:'mlx',caseId:'mlx/base/lora',verdict:'admitted',refusalReason:null,modelRevisions:{dit:'f'.repeat(40)},resolvedCapProvenance:'native host capacity + product rounding',headerDemandProvenance:'native estimate all phases',offline:true,weightsLoaded:false,deviceIdentity:'claimed device',gpuMemoryLimitFraction:.60,totalCapacityBytes:1000,resolvedCapBytes:600,headerDerivedDemandBytes:500,weights:[{path:weight,present:true,sha256:digest(readFileSync(weight))}]};
 assert.equal(validatePreflight(p,'mlx',source,p.caseId),600);
 assert.equal(await validatePreflightChecked(p,'mlx',source,p.caseId,'capture',()=>{}),600);
 assert.equal(validatePreflight({...p,phase:'cross-load',observationSchema:null},'mlx',source,p.caseId,'cross-load'),600);
 assert.throws(()=>validatePreflight({...p,phase:'cross-load'},'mlx',source,p.caseId),/phase/);
 for(const patch of [{resolvedCapBytes:601},{headerDerivedDemandBytes:601},{offline:false},{weightsLoaded:true},{deviceIdentity:''},{caseId:'mlx/sft/lora'},{modelRevisions:{}},{verdict:'refused'},{resolvedCapProvenance:''},{headerDemandProvenance:''},{observationSchema:{}},{observationSchema:{gradientObservations:true,fixedEvaluationPair:true,numericFactorDelta:true,preparedTensorSnapshots:false}}])assert.throws(()=>validatePreflight({...p,...patch},'mlx',source,p.caseId));
 assert.throws(()=>validatePreflight({...p,weights:[{...p.weights[0],present:false}]},'mlx',source,p.caseId));
 writeFileSync(weight,'changed');assert.throws(()=>validatePreflight(p,'mlx',source,p.caseId),/changed/);
 await assert.rejects(validatePreflightChecked(p,'mlx',source,p.caseId,'capture',()=>{}),/changed/);
});
test('bundle binds every member, rejects changed/additional bytes and source mismatches',t=>{
 const dir=fixture(t);writeFileSync(path.join(dir,'raw.log'),'native bytes');
 const r={ok:true,phase:'capture',backend:'mlx',source,cases:CELLS.map(c=>({caseId:`mlx/${c}`,executionPassed:true})),members:artifactMembers(dir)};
 writeFileSync(path.join(dir,'driver.json'),JSON.stringify(r));assert.equal(verifyBundle(dir,source,'capture','mlx').ok,true);
 assert.throws(()=>verifyBundle(dir,{...source,inferencePin:'f'.repeat(40)},'capture','mlx'),/source mismatch/);
 writeFileSync(path.join(dir,'raw.log'),'changed');assert.throws(()=>verifyBundle(dir,source,'capture','mlx'),/members/);
});

function mockRun(t, {badCounts=false,missingEvidence=false,missingRss=false,unsafe=false,previews=false,backend='mlx'}={}) {
 const output=fixture(t);let executions=0;
 const execute=async(_root,command,log,_limit,state,{env})=>{
  executions++;if(isPreparationCommand(root,command))state.completion={scope:"root-and-stdio",completed:!unsafe};else {state.cleanup={scope:'native-api-worker-root',released:!unsafe,stdioClosed:!unsafe,metadataHelperAbsenceClaim:false};state.rootProcess={scope:'native-api-worker-root',pid:4242,executable:'/mock/api',binarySha256:'f'.repeat(64)};}
  let text='';
  if(command[0]==='rustc')text='rustc 1.97.1 (SCHEMA_TEST_ONLY)\nhost: aarch64-apple-darwin\n';
  else if(command[0]==='cargo' && command[1]==='-V')text='cargo 1.97.1 (SCHEMA_TEST_ONLY)\n';
  else if(command[0]==='cargo')text=JSON.stringify({reason:'compiler-artifact',executable:'/mock/api',target:{name:'sceneworks_rust_api',kind:['lib']}});
  else if(command.includes('--list'))text=[...entries(backend),...(!command.includes('--ignored')?[{name:PRE_MODEL_CPU_TEST}]:[])].map(e=>`${e.name}: test`).join('\n');
  else if(command[1]===PRE_MODEL_CPU_TEST)text=goodOutput(PRE_MODEL_CPU_TEST);
  else if(command[1]===PREFLIGHT){writeFileSync(env.MOSS_ACCEPTANCE_DEVICE_PREFLIGHT,'{}');text=goodOutput(PREFLIGHT);}
  else {text=badCounts?'test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out':goodOutput(command[1]);if(!missingEvidence) {
   const raw=previews?mockPreviewEvidence(env.MOSS_ACCEPTANCE_OUTPUT,env.MOSS_ACCEPTANCE_CASE_ID):{fixtureLabel:'NOT ACCEPTANCE EVIDENCE',raw:{}};
   Object.assign(raw.raw,{serverJoined:true,nativeBodyReturnedAtUnixMs:Date.now(),rootProcess:literalRootReceipt(missingRss?[]:[10,20])});
   writeFileSync(path.join(env.MOSS_ACCEPTANCE_OUTPUT,'evidence.json'),JSON.stringify(raw));
  }}
  writeFileSync(log,text);return {exit:0,output:text};
 };
 return {options:{root,output,backend,phase:'capture',manifest:path.join(root,'scripts/ci/moss-training-acceptance/campaign.json'),testPlatform:true,env:{GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_REF:'refs/heads/SCHEMA_TEST_ONLY',GITHUB_SHA:source.workflowSource,GITHUB_RUN_ID:'123',GITHUB_RUN_ATTEMPT:'1',GITHUB_JOB:'capture'},sourceProvider:()=>source,cacheInventory:()=>({ready:true,missing:[]}),execute,statExecutable:()=>true,executableDigest:()=> 'f'.repeat(64),preflightValidator:()=>100},count:()=>executions};
}
test('mock capture performs sequential exact native tests and stores raw evidence without asserting learning',async t=>{
 const m=mockRun(t);const r=await run(m.options);assert.equal(r.ok,true);assert.equal(r.cases.length,4);assert.ok(r.cases.every(c=>c.executionPassed));assert.equal(m.count(),14);assert.equal(r.toolchain.binarySha256,'f'.repeat(64));assert.match(r.toolchain.rustc,/SCHEMA_TEST_ONLY/);assert.equal(r.workflow.runId,'123');
 for(const command of r.commands.slice(0,6)){assert.equal(command.supervision.completion.scope,'root-and-stdio');assert.equal(command.supervision.cleanup,undefined);}
 assert.deepEqual(r.preModelCpu,{test:PRE_MODEL_CPU_TEST,log:'pre-model-cpu.log',counts:{passed:1,failed:0,ignored:0,exit_code:0}});
 assert.equal(r.commands.filter(c=>c.command[1]===PRE_MODEL_CPU_TEST).length,1);
 assert.equal(r.commands.filter(c=>c.command[1]?.startsWith(PREFIX)).length,8,'native command count remains unchanged');
 assert.equal(r.accepted,undefined);assert.ok(r.members.some(m=>m.path.endsWith('evidence.json')));
 for(const command of r.commands.filter(c=>c.command[1]?.startsWith(PREFIX)))assert.ok(command.command.includes('--exact')&&command.command.includes('--ignored'));
});
test('native case rejects a CPU-only completion claim and never starts the next case',async t=>{
 const m=mockRun(t),execute=m.options.execute;
 m.options.execute=async(...args)=>{const r=await execute(...args);if(args[1][1]===testsFor('capture','mlx')[0]){delete args[4].cleanup;args[4].completion={scope:'root-and-stdio',completed:true};}return r;};
 const r=await run(m.options);assert.equal(r.ok,false);assert.equal(m.count(),8);assert.match(r.failures.join(),/completion\/cleanup/);
});
test('nonzero CPU preparation exit aborts before discovery or native execution',async t=>{
 const m=mockRun(t),execute=m.options.execute;
 m.options.execute=async(...args)=>({...await execute(...args),exit:7});
 const r=await run(m.options);assert.equal(r.ok,false);assert.equal(m.count(),1);assert.equal(r.cases.length,0);assert.match(r.failures.join(),/rustc-version exited 7/);
});
// CI_IMPACT: mocked commands only. No native binary, model or GPU executes here.
test('pre-model CPU contract entry is exact, bounded and completed before any native case',async t=>{
 const m=mockRun(t),execute=m.options.execute;
 m.options.execute=async(...args)=>{
  if(args[1][1]===PRE_MODEL_CPU_TEST){assert.ok(args[3]>0&&args[3]<=BOUNDS.list);assert.equal(args[5].env.MOSS_ACCEPTANCE_CASE_ID,undefined);}
  return execute(...args);
 };
 const r=await run(m.options);assert.equal(r.ok,true,r.failures.join());
 assert.deepEqual(r.commands[5].command,['/mock/api',PRE_MODEL_CPU_TEST,'--exact','--nocapture','--test-threads=1']);
 assert.equal(r.commands[6].command[1],PREFLIGHT);
 assert.ok(Date.parse(r.preparationFinishedAt)>=Date.parse(r.commands[5].finishedAt));
});
test('missing, duplicated or ignored CPU discovery refuses before CPU or model execution',async t=>{
 for(const variant of ['missing','duplicate','ignored']){
  const m=mockRun(t),execute=m.options.execute;
  m.options.execute=async(...args)=>{
   const r=await execute(...args);
   if(args[1].includes('--list')){
    if(variant==='missing'&&!args[1].includes('--ignored'))r.output=r.output.replace(`${PRE_MODEL_CPU_TEST}: test`,'');
    if(variant==='duplicate'&&!args[1].includes('--ignored'))r.output+=`\n${PRE_MODEL_CPU_TEST}: test`;
    if(variant==='ignored'&&args[1].includes('--ignored'))r.output+=`\n${PRE_MODEL_CPU_TEST}: test`;
   }
   return r;
  };
  const r=await run(m.options);assert.equal(r.ok,false);assert.equal(r.cases.length,0);assert.equal(m.count(),5);assert.match(r.failures.join(),/unignored CPU preparation test/);
 }
});
test('CPU contract refuses failed, ignored, zero, wrong-test and duplicate greens before native preflight',async t=>{
 for(const variant of ['exit','ignored','zero','wrong','duplicate','cleanup-only']){
  const m=mockRun(t),execute=m.options.execute;
  m.options.execute=async(...args)=>{
   const r=await execute(...args);
   if(args[1][1]===PRE_MODEL_CPU_TEST){
    if(variant==='exit')r.exit=7;
    if(variant==='ignored')r.output=`test ${PRE_MODEL_CPU_TEST} ... ignored\ntest result: ok. 0 passed; 0 failed; 1 ignored;`;
    if(variant==='zero')r.output='test result: ok. 0 passed; 0 failed; 0 ignored;';
    if(variant==='wrong')r.output=goodOutput('different::test');
    if(variant==='duplicate')r.output+=goodOutput(PRE_MODEL_CPU_TEST);
    if(variant==='cleanup-only'){delete args[4].completion;args[4].cleanup={released:true};}
   }
   return r;
  };
  const r=await run(m.options);assert.equal(r.ok,false,variant);assert.equal(r.cases.length,0,variant);assert.equal(m.count(),6,variant);assert.equal(r.preModelCpu,undefined);
 }
});
test('CPU contract time remains charged to the original preparation deadline',async t=>{
 const m=mockRun(t),execute=m.options.execute,RealDate=Date;let clock=RealDate.now();
 globalThis.Date=class extends RealDate {constructor(...args){super(...(args.length?args:[clock]));}static now(){return clock;}};
 try {
  m.options.execute=async(...args)=>{const result=await execute(...args);if(args[1][1]===PRE_MODEL_CPU_TEST)clock+=BOUNDS.build+1;return result;};
  const r=await run(m.options);assert.equal(r.ok,false);assert.equal(r.cases.length,0);assert.equal(m.count(),6);assert.match(r.failures.join(),/build preparation deadline/);
 } finally {globalThis.Date=RealDate;}
});
test('continuation requires the original successful CPU preparation identity and result',async t=>{
 for(const variant of ['missing','test','counts']){
  const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const before=m.count();
  const file=path.join(m.options.output,'driver.json'),receipt=JSON.parse(readFileSync(file));
  if(variant==='missing')delete receipt.preModelCpu;
  if(variant==='test')receipt.preModelCpu.test='different::test';
  if(variant==='counts')receipt.preModelCpu.counts.passed=0;
  writeFileSync(file,JSON.stringify(receipt));
  const r=await run({...m.options,'capture-cell':CELLS[1]});assert.equal(r.ok,false);assert.equal(m.count(),before);assert.match(r.failures.join(),/CPU preparation/);
 }
});
test('continuation rejects a preparation receipt that claims only generic whole-tree release',async t=>{
 const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const before=m.count();
 const file=path.join(m.options.output,'driver.json'),receipt=JSON.parse(readFileSync(file));
 delete receipt.commands[0].supervision.completion;receipt.commands[0].supervision.cleanup={released:true};writeFileSync(file,JSON.stringify(receipt));
 await assert.rejects(run({...m.options,'capture-cell':CELLS[1]}),/prior capture commands not cleanly closed/);assert.equal(m.count(),before);
});
test('mock test-count success without raw evidence fails and retains logs',async t=>{
 const m=mockRun(t,{missingEvidence:true});const r=await run(m.options);assert.equal(r.ok,false);assert.match(r.failures.join(),/evidence missing/);assert.equal(r.cases.length,1);assert.ok(r.members.some(m=>m.path.endsWith('.log')));
});
test('mock ignored or zero-test green is never an execution pass',async t=>{
 const r=await run(mockRun(t,{badCounts:true}).options);assert.equal(r.ok,false);assert.equal(r.cases.length,1);
});
test('mock missing RSS fails closed before next cell',async t=>{
 const r=await run(mockRun(t,{missingRss:true}).options);assert.equal(r.ok,false);assert.match(r.failures.join(),/RSS/);assert.equal(r.cases.length,1);
});
test('mock unproven release blocks even discovery and next case',async t=>{
 const m=mockRun(t,{unsafe:true});const r=await run(m.options);assert.equal(r.ok,false);assert.equal(m.count(),1);assert.match(r.failures.join(),/cleanup/);
});
test('workflow is reusable only, read-only, guarded uploads, exclusive dispatcher jobs',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 assert.match(workflow,/workflow_call:/);assert.doesNotMatch(workflow,/^  (push|pull_request|schedule):/m);assert.doesNotMatch(workflow,/contents: write|pull-requests: write/);assert.match(workflow,/real-weights/);assert.match(workflow,/always\(\)/);
 const dispatcher=readFileSync(path.join(root,'.github/workflows/memory-catalog-campaign.yml'),'utf8');
 for(const job of ['flux3-app-lifecycle','flux3-terminal-host','flux3-terminal-queue','qwen-app-lifecycle','iris-mlx-real-weights','supra2-mlx-real-weights','mlx','candle'])assert.ok(dispatcher.includes(`  ${job}:\n    if: \${{ needs.moss-selector-preflight.outputs.enabled != 'true'`));
 assert.match(dispatcher,/moss_options_json:/);assert.match(dispatcher,/needs: moss-selector-preflight/);
});

// Generated mock PNGs only: no SDK, native command, model or GPU is used.
function mockPng(value=0) {
 const crc32=b=>{let c=0xffffffff;for(const v of b){c^=v;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (c^0xffffffff)>>>0;};
 const chunk=(kind,data)=>{const type=Buffer.from(kind),out=Buffer.alloc(data.length+12);out.writeUInt32BE(data.length);type.copy(out,4);data.copy(out,8);out.writeUInt32BE(crc32(Buffer.concat([type,data])),out.length-4);return out;};
 const header=Buffer.alloc(13);header.writeUInt32BE(256);header.writeUInt32BE(256,4);header[8]=8;header[9]=2;
 const rows=Buffer.alloc(256*(256*3+1),value);for(let y=0;y<256;y++)rows[y*(256*3+1)]=0;
 return Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',header),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
}
function mockPreviewEvidence(directory,caseId) {
 const images={};
 for(const [i,role] of ['before-untrained','after-trained','disable-control'].entries()) {
  const bytes=mockPng(i),name=`${role}.png`;writeFileSync(path.join(directory,name),bytes);
  images[role]={artifact:{path:name,sha256:digest(bytes)}};
 }
 return {caseId,raw:{images}};
}
function reviewArgs(m,cell='base-lora') {
 return {preview:path.join(m.options.output,'previews',cell,'index.json'),output:path.join(m.options.output,'..',path.basename(m.options.output)+'-inspections.json'),reviewer:'schema-test reviewer',findings:{'before-untrained':['observed mock baseline'],'after-trained':['observed mock trained view'],'disable-control':['observed mock disabled view']}};
}
test('Windows actual round-trip UTC format is canonical Z in both job starts, never +00:00',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 // Actual .NET o-format contracts for the same instant, including seven digits.
 const offset='2026-10-25T14:12:09.1234567+00:00',utc='2026-10-25T14:12:09.1234567Z';
 const rustUtc=/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/;
 assert.equal(rustUtc.test(offset),false);assert.equal(rustUtc.test(utc),true);assert.equal(Date.parse(offset),Date.parse(utc));
 for(const job of ['capture','cross-load']) {
  const section=workflow.split(`  ${job}:\n`)[1].split(/^  [a-z][a-z-]+:\n/m)[0];
  assert.match(section,/MOSS_CROSS_LOAD_JOB_STARTED_AT=' \+ \[DateTimeOffset\]::UtcNow\.UtcDateTime\.ToString\('o'\)/);
  assert.doesNotMatch(section,/UtcNow\.ToString\('o'\)/);
 }
});
test('both MOSS CUDA phases use authorized PCI device 0 and preserve serial preparation',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 const authoritative=readFileSync(path.join(root,'.github/workflows/windows-candle.yml'),'utf8');
 assert.match(authoritative,/group: windows-candle-gpu-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}-iris-worker-smoke\n      cancel-in-progress: false/);
 assert.match(authoritative,/CUDA_DEVICE_ORDER: PCI_BUS_ID\n      CUDA_VISIBLE_DEVICES: "1"/);
 for(const job of ['capture','cross-load']) {
  const section=workflow.split(`  ${job}:\n`)[1].split(/^  [a-z][a-z-]+:\n/m)[0];
  assert.match(section,/matrix\.backend == 'cuda' && 'windows-candle-gpu-real-weights' \|\| format\('moss-training-\{0\}-\{1\}-\{2\}-mlx', github\.run_id, github\.run_attempt, github\.job\)/);
  assert.match(section,/cancel-in-progress: false/);assert.match(section,/CUDA_DEVICE_ORDER: PCI_BUS_ID/);
  assert.match(section,/CUDA_VISIBLE_DEVICES: \$\{\{ matrix\.backend == 'cuda' && '0' \|\| '' \}\}/);
  assert.match(section,/NVIDIA_VISIBLE_DEVICES: \$\{\{ matrix\.backend == 'cuda' && '0' \|\| '' \}\}/);
  assert.match(section,/prepare-rust-runner\n        if: \$\{\{ matrix\.backend == 'cuda' \}\}\n        with:\n          workspace-directory: sceneworks-source/);
 }
});
test('live preview notices use native shells and byte-identical message content',()=>{
 const action=readFileSync(path.join(root,'.github/actions/moss-capture-cell/action.yml'),'utf8');
 const check=text=>{
  const mlx=text.split('    - name: Announce immediately accessible live preview inspection artifact\n')[1]?.split('    - name:')[0];
  const cuda=text.split('    - name: Announce immediately accessible live preview inspection artifact on Windows\n')[1];
  assert.ok(mlx&&cuda);assert.match(mlx,/inputs.backend == 'mlx'[\s\S]*shell: bash/);assert.match(cuda,/inputs.backend == 'cuda'[\s\S]*shell: powershell/);
  const instruction=section=>/^        PREVIEW_INSTRUCTION: (.+)$/m.exec(section)?.[1];
  assert.ok(instruction(mlx));assert.equal(instruction(mlx),instruction(cuda));
  const values=['base/lora','https://example.invalid/owned-artifact',instruction(mlx)];let index=0;
  const unix=/printf '([^']+)'/.exec(mlx)?.[1].replace(/\\n$/,'').replace(/%s/g,()=>values[index++]);
  const windows=/Write-Output \('([^']+)' -f/.exec(cuda)?.[1].replace(/\{([0-9]+)\}/g,(_,n)=>values[Number(n)]);
  assert.ok(unix&&windows);assert.deepEqual(Buffer.from(unix),Buffer.from(windows));
 };
 check(action);
 assert.throws(()=>check(action.replace('shell: powershell','shell: bash')));
 assert.throws(()=>check(action.replace('PREVIEW_INSTRUCTION: download now','PREVIEW_INSTRUCTION: wait until capture completes')));
});
test('sequential capture publishes each cell live, reuses preparation, and preserves complete backend receipt',async t=>{
 for(const backend of ['mlx','cuda']) {
 const m=mockRun(t,{previews:true,backend});let originalClock,firstMembers;
 for(const [i,cell] of CELLS.entries()) {
  const r=await run({...m.options,'capture-cell':cell});
  assert.equal(r.failures.length,0);assert.equal(r.cellComplete,true);assert.equal(r.ok,i===3);assert.equal(r.cases.length,i+1);assert.equal(m.count(),6+2*(i+1));assert.equal(r.commands.length,6+2*(i+1));
  if(i===0){originalClock=r.preparationStartedAt;firstMembers=r.members;}assert.equal(r.preparationStartedAt,originalClock);
  const index=JSON.parse(readFileSync(path.join(m.options.output,'previews',cell.replace('/','-'),'index.json')));
  assert.equal(index.caseId,`${backend}/${cell}`);assert.equal(index.workflow.runId,'123');assert.deepEqual(index.source,source);
  assert.equal(index.caseStartedAt,r.cases[i].caseStartedAt);assert.equal(index.caseDeadlineAt,r.cases[i].caseDeadlineAt);assert.equal(Date.parse(index.caseDeadlineAt)-Date.parse(index.caseStartedAt),BOUNDS.case);assert.equal(index.images.length,3);
  for(const image of index.images) {
   const bytes=readFileSync(path.join(m.options.output,'previews',cell.replace('/','-'),image.file));assert.equal(digest(bytes),image.reviewedArtifact.sha256);assert.deepEqual(decodePreviewPng(bytes),image.decoded);assert.ok(image.reviewedArtifact.path.startsWith(`capture-${backend}/${cell}/`));
  }
  if(i<3)assert.throws(()=>verifyBundle(m.options.output,source,'capture',backend),/incomplete/);
 }
 const final=verifyBundle(m.options.output,source,'capture',backend);for(const member of firstMembers)assert.deepEqual(final.members.find(x=>x.path===member.path),member);
 assert.equal(final.commands.filter(c=>c.log==='compile.log').length,1);assert.equal(final.commands.filter(c=>c.log==='pre-model-cpu.log').length,1);assert.ok(final.members.some(m=>m.path==='base/lora/evidence.json'));
 }
});
test('continuation rejects skipped, repeated, invalid and noncapture selectors without commands',async t=>{
 const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const before=m.count();
 await assert.rejects(run({...m.options,'capture-cell':CELLS[2]}),/exact closed case prefix/);await assert.rejects(run({...m.options,'capture-cell':CELLS[0]}),/new\/empty/);
 await assert.rejects(run({...m.options,phase:'cross-load','capture-cell':CELLS[1]}),/invalid capture-cell/);await assert.rejects(run({...m.options,'capture-cell':'extra'}),/invalid capture-cell/);assert.equal(m.count(),before);
 const args=['--phase','capture','--backend','mlx','--root','/source','--output','/output','--manifest','/manifest'];assert.equal(parseArgs([...args,'--capture-cell',CELLS[0]])['capture-cell'],CELLS[0]);assert.throws(()=>parseArgs([...args,'--capture-cell','base/extra']),/exact capture cell/);
});
test('continuation authenticates every old member against changed, missing or added bytes',async t=>{
 for(const mutation of ['changed','missing','extra']) {
  const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const before=m.count();
  if(mutation==='missing')rmSync(path.join(m.options.output,'rustc-version.log'));else writeFileSync(path.join(m.options.output,mutation==='extra'?'stale.log':'rustc-version.log'),'untrusted');
  await assert.rejects(run({...m.options,'capture-cell':CELLS[1]}),/members changed/);assert.equal(m.count(),before);
 }
});
test('continuation rejects changed source, run, attempt, job, binary and original preparation budget',async t=>{
 for(const variant of ['source','run','attempt','job','binary','prep']) {
  const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const before=m.count(),options={...m.options,'capture-cell':CELLS[1]};
  if(variant==='source')options.sourceProvider=()=>({...source,inferencePin:'f'.repeat(40)});
  if(variant==='run')options.env={...options.env,GITHUB_RUN_ID:'124'};if(variant==='attempt')options.env={...options.env,GITHUB_RUN_ATTEMPT:'2'};if(variant==='job')options.env={...options.env,GITHUB_JOB:'other'};if(variant==='binary')options.executableDigest=()=> '0'.repeat(64);
  if(variant==='prep'){const file=path.join(options.output,'driver.json'),prior=JSON.parse(readFileSync(file));prior.preparationStartedAt=new Date(Date.parse(prior.preparationFinishedAt)-BOUNDS.build-1).toISOString();writeFileSync(file,JSON.stringify(prior));}
  const r=await run(options);assert.equal(r.ok,false);assert.equal(r.cellComplete,false);assert.equal(r.cases.length,1);assert.equal(m.count(),before);assert.match(r.failures.join(),/source proof|run\/attempt\/job|binary changed|preparation deadline/);
 }
});
test('failed, cancelled and interrupted prior capture cells cannot continue or pass peer verification',async t=>{
 const m=mockRun(t,{previews:true,missingRss:true}),r=await run({...m.options,'capture-cell':CELLS[0]});assert.equal(r.ok,false);assert.equal(r.cellComplete,false);const before=m.count();
 await assert.rejects(run({...m.options,'capture-cell':CELLS[1]}),/not cleanly closed/);assert.equal(m.count(),before);
 const cancelled=mockRun(t,{previews:true}),execute=cancelled.options.execute;
 cancelled.options.execute=async(...args)=>{const result=await execute(...args);if(args[1][1]===testsFor('capture')[0])args[4].cancelled=true;return result;};
 const cancelledReceipt=await run({...cancelled.options,'capture-cell':CELLS[0]});assert.equal(cancelledReceipt.ok,false);assert.equal(cancelledReceipt.cases[0].executionPassed,undefined);await assert.rejects(run({...cancelled.options,'capture-cell':CELLS[1]}),/not cleanly closed/);
 const clean=mockRun(t,{previews:true});await run({...clean.options,'capture-cell':CELLS[0]});const file=path.join(clean.options.output,'driver.json'),prior=JSON.parse(readFileSync(file));prior.failures.push('signal interruption');writeFileSync(file,JSON.stringify(prior));await assert.rejects(run({...clean.options,'capture-cell':CELLS[1]}),/not cleanly closed/);
});
test('peer verification requires all four exact ordered completed cells even when receipt claims ok',t=>{
 const dir=fixture(t);writeFileSync(path.join(dir,'raw.log'),'mock bytes');const full=CELLS.map(cell=>({caseId:`mlx/${cell}`,executionPassed:true}));
 for(const cases of [full.slice(0,1),full.slice(0,3),[full[1],full[0],...full.slice(2)],full.map((c,i)=>i===3?{...c,executionPassed:false}:c)]) {
  writeFileSync(path.join(dir,'driver.json'),JSON.stringify({phase:'capture',backend:'mlx',ok:true,source,cases,members:artifactMembers(dir).filter(m=>m.path!=='driver.json')}));assert.throws(()=>verifyBundle(dir,source,'capture','mlx'),/all four/);
 }
});
test('preview publication fails closed on missing, changed or undecodable PNGs',async t=>{
 for(const variant of ['missing','hash','decode']) {
  const m=mockRun(t,{previews:true}),execute=m.options.execute;
  m.options.execute=async(...args)=>{const result=await execute(...args),env=args[5].env;if(args[1][1]===testsFor('capture')[0]){const rawFile=path.join(env.MOSS_ACCEPTANCE_OUTPUT,'evidence.json'),raw=JSON.parse(readFileSync(rawFile)),image=path.join(env.MOSS_ACCEPTANCE_OUTPUT,raw.raw.images['after-trained'].artifact.path);if(variant==='missing')rmSync(image);else{writeFileSync(image,'not PNG');if(variant==='decode')raw.raw.images['after-trained'].artifact.sha256=digest(Buffer.from('not PNG'));}writeFileSync(rawFile,JSON.stringify(raw));}return result;};
  const r=await run({...m.options,'capture-cell':CELLS[0]});assert.equal(r.ok,false);assert.equal(r.cellComplete,false);assert.equal(r.cases.length,1);await assert.rejects(run({...m.options,'capture-cell':CELLS[1]}),/not cleanly closed/);
 }
 const png=mockPng();assert.equal(decodePreviewPng(png).width,256);png[50]^=1;assert.throws(()=>decodePreviewPng(png),/CRC/);
});
test('inspection records actual Date.now and finalizer hashes, appending partial JSON during capture',async t=>{
 const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const args=reviewArgs(m),before=Date.now(),document=recordInspection(args),after=Date.now();t.after(()=>rmSync(args.output,{force:true}));
 assert.equal(document.cases.length,1);assert.equal(document.captureRunId,'123');assert.equal(document.cases[0].captureWorkflow.attempt,1);assert.equal(m.count(),8);
 for(const inspection of document.cases[0].inspections){assert.ok(Date.parse(inspection.completedAt)>=before && Date.parse(inspection.completedAt)<=after);assert.equal(inspection.beforeFinalAcceptance,true);assert.equal(digest(readFileSync(path.join(m.options.output,...inspection.reviewedArtifact.path.split('/').slice(1)))),inspection.reviewedArtifact.sha256);}
 await run({...m.options,'capture-cell':CELLS[1]});const appended=recordInspection({...args,preview:reviewArgs(m,'base-lokr').preview});assert.equal(appended.cases.length,2);assert.deepEqual(appended.cases[0],document.cases[0]);assert.throws(()=>recordInspection(args),/already recorded/);
});
test('inspection permits actual post-deadline review, rejects forged clocks, empty findings and cross-run carry',async t=>{
 const m=mockRun(t,{previews:true});await run({...m.options,'capture-cell':CELLS[0]});const args=reviewArgs(m),original=readFileSync(args.preview),index=JSON.parse(original);t.after(()=>rmSync(args.output,{force:true}));
 assert.throws(()=>parseInspectionArgs(['--timestamp','2026-01-01']),/no timestamp override/);
 const oldStart=Date.now()-BOUNDS.case-100;
 writeFileSync(args.preview,JSON.stringify({...index,caseStartedAt:new Date(oldStart).toISOString(),caseDeadlineAt:new Date(oldStart+BOUNDS.case).toISOString(),publishedAt:new Date(oldStart+100).toISOString()}));assert.equal(recordInspection(args).cases.length,1);rmSync(args.output);
 writeFileSync(args.preview,JSON.stringify({...index,publishedAt:new Date(Date.now()+10000).toISOString()}));assert.throws(()=>recordInspection(args),/publication/);
 writeFileSync(args.preview,JSON.stringify({...index,caseDeadlineAt:new Date(Date.parse(index.caseDeadlineAt)+1).toISOString()}));assert.throws(()=>recordInspection(args),/deadline/);
 writeFileSync(args.preview,original);assert.throws(()=>recordInspection({...args,findings:{...args.findings,'after-trained':[]}}),/findings/);recordInspection(args);
 await run({...m.options,'capture-cell':CELLS[1]});const second=reviewArgs(m,'base-lokr'),secondIndex=JSON.parse(readFileSync(second.preview));writeFileSync(second.preview,JSON.stringify({...secondIndex,workflow:{...secondIndex.workflow,runId:'999'}}));assert.throws(()=>recordInspection({...args,preview:second.preview}),/another source\/run/);
 writeFileSync(second.preview,JSON.stringify(secondIndex));writeFileSync(path.join(path.dirname(second.preview),'after-trained.png'),'changed');assert.throws(()=>recordInspection({...args,preview:second.preview}),/PNG hash/);
});
test('workflow publishes each cell before continuing, retains backend bundle layout, and fails on upload errors',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8'),capture=workflow.split('  capture:\n')[1].split('  cross-load:\n')[0];let previous=-1;
 for(const cell of CELLS){const start=capture.indexOf(`      - name: Capture and publish ${cell} before continuing`);assert.ok(start>previous);previous=start;const step=capture.slice(start).split('\n      - ')[0];assert.match(step,/uses: \.\/sceneworks-source\/\.github\/actions\/moss-capture-cell/);assert.ok(step.includes(`cell: ${cell}\n`));assert.ok(step.includes(`preview-name: ${cell.replace('/','-')}\n`));assert.match(step,/output: .*\/capture-\$\{\{ matrix.backend \}\}/);}
 assert.ok(capture.indexOf('name: Preserve raw logs')>previous);assert.match(capture,/name: capture-\$\{\{ matrix.backend \}\}/);
 const composite=readFileSync(path.join(root,'.github/actions/moss-capture-cell/action.yml'),'utf8');assert.ok(composite.indexOf('node scripts/moss-training-acceptance.mjs')<composite.indexOf('uses: actions/upload-artifact@'));assert.match(composite,/--capture-cell/);assert.match(composite,/steps.preview.outputs.artifact-url/);assert.match(composite,/::notice title=MOSS live preview inspection/);assert.match(composite,/if-no-files-found: error/);assert.doesNotMatch(composite,/continue-on-error|always\(\)|timestamp|deadline.*override/);assert.match(composite,/path: \$\{\{ inputs.output \}\}\/previews\/\$\{\{ inputs.preview-name \}\}\//);
});

test('cross-load separates pre-checkout preparation from each bounded import phase',async t=>{
 const peers=fixture(t);
 for(const backend of ['mlx','cuda']) {
  const m=mockRun(t,{backend});m.options.output=path.join(peers,`capture-${backend}-attempt-1`);await run(m.options);
 }
 const m=mockRun(t),execute=m.options.execute,RealDate=Date;let clock=RealDate.now();
 m.options.env.MOSS_CROSS_LOAD_JOB_STARTED_AT=new RealDate(clock).toISOString();
 globalThis.Date=class extends RealDate {constructor(...args){super(...(args.length?args:[clock]));}static now(){return clock;}};
 try {
  m.options.execute=async(...args)=>{
   const command=args[1],environment=args[5].env;
   if(command[0]==='rustc')clock+=3700_000; // Preparation may exceed the 3600s compute bound.
   if(command[1]?.endsWith('_cross_backend')) {
    assert.equal(environment.MOSS_ACCEPTANCE_PHASE,'cross-load');
    assert.equal(environment.MOSS_ACCEPTANCE_CASE_STARTED_AT,environment.MOSS_ACCEPTANCE_CROSS_LOAD_STARTED_AT);
    clock+=1000_000; // Four imports together exceed 3600s, each stays bounded.
   }
   return execute(...args);
  };
  const receipt=await run({...m.options,phase:'cross-load','peer-artifacts':peers});
  assert.equal(receipt.ok,true,receipt.failures.join());assert.equal(receipt.cases.length,4);
  assert.deepEqual(receipt.sourceCapture,{backend:'cuda',driverSha256:digest(readFileSync(path.join(peers,'capture-cuda-attempt-1','driver.json')))});
  assert.equal(Date.parse(receipt.preparationFinishedAt)-Date.parse(receipt.preparationStartedAt),3700_000);
  assert.ok(receipt.cases.every((entry,i)=>i===0||Date.parse(entry.caseStartedAt)>Date.parse(receipt.cases[i-1].caseStartedAt)));
 } finally {globalThis.Date=RealDate;}
});
test('expired cross-load preparation refuses native execution',async t=>{
 const peers=fixture(t);
 for(const backend of ['mlx','cuda']) {const m=mockRun(t,{backend});m.options.output=path.join(peers,`capture-${backend}-attempt-1`);await run(m.options);}
 const m=mockRun(t);m.options.env.MOSS_CROSS_LOAD_JOB_STARTED_AT=new Date(Date.now()-BOUNDS.build-1).toISOString();
 const receipt=await run({...m.options,phase:'cross-load','peer-artifacts':peers});
 assert.equal(receipt.ok,false);assert.match(receipt.failures.join(),/preparation deadline/);assert.equal(m.count(),0);
});

test('inspection appends same-run backend retries with per-case workflow identity',async t=>{
 const first=mockRun(t,{previews:true}),second=mockRun(t,{previews:true,backend:'cuda'});
 second.options.env.GITHUB_RUN_ATTEMPT='2';
 await run({...first.options,'capture-cell':CELLS[0]});await run({...second.options,'capture-cell':CELLS[0]});
 const args=reviewArgs(first);t.after(()=>rmSync(args.output,{force:true}));
 const original=recordInspection(args);
 const mixed=recordInspection({...reviewArgs(second),output:args.output});
 assert.deepEqual(mixed.cases[0],original.cases[0]);assert.deepEqual(mixed.cases.map(c=>c.captureWorkflow.attempt),[1,2]);
 assert.deepEqual(mixed.cases.map(c=>c.captureWorkflow.jobId),['capture','capture']);
 assert.equal(mixed.captureRunId,'123');assert.equal(mixed.attempt,undefined);
});
test('inspection refuses wrong source/file/job and never invents legacy per-case provenance',async t=>{
 const first=mockRun(t,{previews:true}),second=mockRun(t,{previews:true,backend:'cuda'});
 await run({...first.options,'capture-cell':CELLS[0]});await run({...second.options,'capture-cell':CELLS[0]});
 const args=reviewArgs(first),original=readFileSync(args.preview),index=JSON.parse(original);t.after(()=>rmSync(args.output,{force:true}));
 for(const changes of [{sha:'0'.repeat(40)},{fileSha256:'0'.repeat(64)},{jobId:'other'}]) {
  writeFileSync(args.preview,JSON.stringify({...index,workflow:{...index.workflow,...changes}}));
  assert.throws(()=>recordInspection(args),/source\/file\/job/);
 }
 writeFileSync(args.preview,original);const document=recordInspection(args);delete document.cases[0].captureWorkflow;document.attempt=1;
 writeFileSync(args.output,JSON.stringify(document));
 assert.throws(()=>recordInspection({...reviewArgs(second),output:args.output}),/legacy inspections/);
});
test('cross-load refuses another run even for identical source and bytes before any command',async t=>{
 const peers=fixture(t);
 for(const backend of ['mlx','cuda']) {const m=mockRun(t,{backend});m.options.output=path.join(peers,`capture-${backend}-attempt-1`);m.options.env.GITHUB_RUN_ID='124';await run(m.options);}
 const m=mockRun(t);m.options.env.MOSS_CROSS_LOAD_JOB_STARTED_AT=new Date().toISOString();
 const receipt=await run({...m.options,phase:'cross-load','peer-artifacts':peers});
 assert.equal(receipt.ok,false);assert.match(receipt.failures.join(),/another capture run/);assert.equal(m.count(),0);
});

test('cache inventory refuses missing selected files before compilation and retains its report',async t=>{
 const m=mockRun(t),report={ready:false,verification:'stat-only',missing:[{path:'/SCHEMA_TEST_ONLY/missing'}]};
 m.options.cacheInventory=()=>report;
 const receipt=await run(m.options);
 assert.equal(receipt.ok,false);assert.equal(m.count(),0);assert.deepEqual(receipt.cacheInventory,report);
 assert.match(receipt.failures.join(),/missing before compilation/);
});

test('MOSS dispatcher deliberately accepts cached primary real-weights and preserves unspecified refusal',t=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/memory-catalog-campaign.yml'),'utf8');
 const guard=workflow.split('      - name: Reject conflicting selectors before compiler or device jobs')[1].split("node --input-type=module <<'JS'\n")[1].split('          JS')[0].replace(/^          /gm,'');
 assert.ok(guard.includes("['real-weights','rw-llm','rw-mage','rw-krea']"));
 assert.match(workflow,/moss_options_json:[\s\S]*?default: "\{\}"/);
 for(const label of ['real-weights','rw-llm','rw-mage','rw-krea','unspecified','','nax']) {
  const result=spawnSync(process.execPath,['--input-type=module','-e',guard],{encoding:'utf8',env:{...process.env,GITHUB_OUTPUT:path.join(fixture(t),'output'),SELECTORS:JSON.stringify({moss_options_json:JSON.stringify({enabled:true,runner_label:label,phase:'capture'})}),DISPATCH_SHA:source.workflowSource,DISPATCH_EVENT:'workflow_dispatch',DISPATCH_REPOSITORY:'SceneWorks/SceneWorks'}});
  assert.equal(result.status,['real-weights','rw-llm','rw-mage','rw-krea'].includes(label)?0:1,result.stderr);
 }
 const reusable=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 assert.match(reusable,/pattern: capture-\*-attempt-\*/);assert.match(reusable,/pattern: cross-load-\*-attempt-\*/);
 for(const phase of ['capture','cross-load'])assert.ok(reusable.includes('name: '+phase+'-${{ matrix.backend }}-attempt-${{ github.run_attempt }}'));
 assert.doesNotMatch(reusable,/overwrite: true/);
});

test('actual CLI refuses non-Actions and wrong-source startup with no native commands',t=>{
 for(const changes of [{GITHUB_EVENT_NAME:'push'},{GITHUB_EVENT_NAME:'workflow_dispatch',MOSS_WORKFLOW_SOURCE:'a'.repeat(40),GITHUB_SHA:'b'.repeat(40)}]) {
  const output=fixture(t);
  const result=spawnSync(process.execPath,['scripts/moss-training-acceptance.mjs','--phase','capture','--root',root,'--output',output,'--backend','mlx','--manifest',path.join(root,'scripts/ci/moss-training-acceptance/campaign.json')],{cwd:root,encoding:'utf8',env:{...process.env,...changes}});
  assert.equal(result.status,1,result.stderr);
  const receipt=JSON.parse(readFileSync(path.join(output,'driver.json')));
  assert.equal(receipt.commands.length,0);assert.match(receipt.failures.join(),/workflow_dispatch|exact dispatched GitHub SHA/);
 }
});
test('assembled controller uses real stat inventory and refuses absent cache before first cargo',async t=>{
 const m=mockRun(t);m.options.cacheInventory=inventoryMossTrainingCache;
 m.options.env.HF_HUB_CACHE=path.join(fixture(t),'absent-hub');
 const receipt=await run(m.options);
 assert.equal(receipt.ok,false);assert.equal(m.count(),0);assert.equal(receipt.commands.length,0);
 assert.equal(receipt.cacheInventory.ready,false);assert.ok(receipt.cacheInventory.missing.length>0);
 assert.match(receipt.failures.join(),/missing before compilation/);
});

test('both MLX setup steps match cargo test and fall back only for absent exact prebuilts',async t=>{
 const receipt=await run(mockRun(t).options);
 const compile=receipt.commands.find(command=>command.command[0]==='cargo'&&command.command[1]==='test');
 assert.ok(compile);assert.equal(compile.command.includes('--release'),false);assert.equal(compile.command.includes('--profile'),false);
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 const bash=process.platform==='win32'?'C:/Program Files/Git/bin/bash.exe':'bash';
 for(const phase of ['capture','cross-load']) {
  const section=workflow.split(`  ${phase}:\n`)[1].split(/^  [a-z][a-z-]+:\n/m)[0];
  const step=section.split('      - name: Prepare MLX library for the test profile (not model weights)\n')[1].split('\n      - ')[0];
  assert.match(step,/if: \$\{\{ matrix\.backend == 'mlx' \}\}/);
  assert.match(step,/scripts\/fetch-prebuilt-mlx\.sh --build-type Debug --github-env/);
  const body=step.split('        run: |\n')[1].replace(/^          /gm,'');
  const directory=fixture(t),script=path.join(directory,'fixture-fetch.sh');
  writeFileSync(script,'if [ "$*" != "--build-type Debug --github-env" ]; then exit 2; fi\nif [ "$FIXTURE_FETCH_RC" = 0 ]; then\n  printf "%s\\n" PMETAL_MLX_PREBUILT_DIR=Debug PMETAL_METALLIB_PATH=Debug.metallib >> "$GITHUB_ENV"\nfi\nexit "$FIXTURE_FETCH_RC"\n');
  const mocked=body.replace('scripts/fetch-prebuilt-mlx.sh --build-type Debug --github-env','bash ./fixture-fetch.sh --build-type Debug --github-env');
  for(const rc of [0,1,2,3]) {
   const output=path.join(directory,`env-${rc}`);writeFileSync(output,'PMETAL_MLX_PREBUILT_DIR=stale\nPMETAL_METALLIB_PATH=stale.metallib\n');
   const result=spawnSync(bash,['-e','-c',mocked],{cwd:directory,encoding:'utf8',env:{...process.env,GITHUB_ENV:output.replaceAll('\\','/'),FIXTURE_FETCH_RC:String(rc)}});
   assert.equal(result.status,rc<2?0:rc,result.stderr);
   const lines=readFileSync(output,'utf8').trim().split('\n');
   assert.deepEqual(lines.slice(-2),rc===0?['PMETAL_MLX_PREBUILT_DIR=Debug','PMETAL_METALLIB_PATH=Debug.metallib']:rc===1?['PMETAL_MLX_PREBUILT_DIR=','PMETAL_METALLIB_PATH=']:['PMETAL_MLX_PREBUILT_DIR=stale','PMETAL_METALLIB_PATH=stale.metallib']);
  }
 }
});


test('every supervised native entry uses the production 8 MiB stack despite ambient overrides',async t=>{
 assert.equal(NATIVE_TEST_STACK_BYTES,8*1024*1024);
 const worker=readFileSync(path.join(root,'crates/sceneworks-worker/src/lib.rs'),'utf8');
 assert.match(worker,/pub const WORKER_ENTRY_STACK_BYTES: usize = 8 \* 1024 \* 1024;/);
 const peers=fixture(t);
 // Finish both source captures before testing either opposite-backend import.
 for(const phase of ['capture','cross-load']) {
  for(const backend of ['mlx','cuda']) {
   const m=mockRun(t,{backend}),execute=m.options.execute;let native=0,preparation=0;
   m.options.env.RUST_MIN_STACK='65536';
   m.options.execute=async(...args)=>{
    const command=args[1],env=args[5].env;
    if(command[1]?.startsWith(PREFIX)) {assert.equal(env.RUST_MIN_STACK,String(NATIVE_TEST_STACK_BYTES));native++;}
    else {assert.equal(env.RUST_MIN_STACK,'65536');preparation++;}
    return execute(...args);
   };
   if(phase==='capture')m.options.output=path.join(peers,`capture-${backend}-attempt-1`);
   else {m.options['peer-artifacts']=peers;m.options.env.MOSS_CROSS_LOAD_JOB_STARTED_AT=new Date().toISOString();}
   const receipt=await run({...m.options,phase});
   assert.equal(receipt.ok,true,receipt.failures.join());assert.equal(native,8);assert.equal(preparation,6);
   assert.equal(receipt.toolchain.libtestStackBytes,NATIVE_TEST_STACK_BYTES);
  }
 }
});

test('current dispatcher fits 25 inputs and derives exact source and finalize run without retired selectors',t=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/memory-catalog-campaign.yml'),'utf8');
 const inputs=workflow.split('    inputs:\n')[1].split('\n# `gh pr create`')[0];
 const names=[...inputs.matchAll(/^      ([A-Za-z_][A-Za-z0-9_]*):$/gm)].map(match=>match[1]);
 assert.equal(names.length,25);assert.equal(new Set(names).size,25);assert.ok(names.length<=25);
 for(const removed of ['moss_reference','moss_terminal','workflowSource','moss_evidence_run_id','run_moss_training_acceptance','moss_runner_label','moss_phase','moss_inspection_json'])assert.equal(names.includes(removed),false);
 assert.ok(names.includes('flux3_terminal_control_sha256'),'digits in delivered FLUX3 inputs must count');
 assert.match(workflow,/workflowSource: \$\{\{ github\.sha \}\}/);
 assert.match(workflow,/moss_evidence_run_id: \$\{\{ needs\.moss-selector-preflight\.outputs\.capture_run_id \}\}/);
 const guard=workflow.split('      - name: Reject conflicting selectors before compiler or device jobs')[1].split("node --input-type=module <<'JS'\n")[1].split('          JS')[0].replace(/^          /gm,'');
 const base={enabled:true,runner_label:'real-weights',phase:'capture'};
 const invoke=(options=base,changes={},environment={})=>{
  const output=path.join(fixture(t),'output');
  const result=spawnSync(process.execPath,['--input-type=module','-e',guard],{encoding:'utf8',env:{...process.env,GITHUB_OUTPUT:output,SELECTORS:JSON.stringify({moss_options_json:JSON.stringify(options),...changes}),DISPATCH_SHA:source.workflowSource,DISPATCH_EVENT:'workflow_dispatch',DISPATCH_REPOSITORY:'SceneWorks/SceneWorks',...environment}});
  return {...result,output:result.status===0?Object.fromEntries(readFileSync(output,'utf8').trim().split('\n').map(line=>{const i=line.indexOf('=');return [line.slice(0,i),line.slice(i+1)];})):null};
 };
 assert.deepEqual(invoke({}).output,{enabled:'false'});
 assert.deepEqual(invoke({enabled:false}).output,{enabled:'false'});
 assert.equal(invoke(base,{moss_options_json:'{broken'}).status,1);
 for(const options of [null,[],true,'text',{enabled:'true'},{enabled:false,runner_label:'real-weights'},{...base,unknown:true},{...base,runner_label:'unspecified'},{...base,inspection:[]},{...base,inspection:null}])assert.equal(invoke(options).status,1);
 assert.deepEqual(invoke().output,{enabled:'true',runner_label:'real-weights',phase:'capture',inspection_json:'',capture_run_id:''});
 assert.equal(invoke({...base,inspection:{}}).status,0);
 assert.equal(invoke({...base,phase:'bounded-product'}).output.phase,'bounded-product');
 assert.equal(invoke({...base,phase:'bounded-product',runner_label:'rw-mage'}).status,1);
 assert.equal(invoke({...base,phase:'bounded-product',inspection:{captureRunId:'123'}}).status,1);
 const inspection={captureRunId:'123',manifestSha256:source.manifestSha256,cases:[]};
 const finalized=invoke({...base,phase:'finalize',inspection});
 assert.equal(finalized.output.capture_run_id,'123');assert.deepEqual(JSON.parse(finalized.output.inspection_json),inspection);
 for(const captureRunId of [undefined,null,123,'0','01','-1','1\nother=2','1e3'])assert.equal(invoke({...base,phase:'finalize',inspection:{captureRunId}}).status,1);
 for(const inspection of [undefined,{},[],null,'{malformed'])assert.equal(invoke({...base,phase:'finalize',inspection}).status,1);
 assert.equal(invoke({...base,inspection}).status,1);
 for(const key of ['anchors','models'])assert.equal(invoke(base,{[key]:'conflicting'}).status,1);
 for(const profile of ['flux3-image-klein-app-lifecycle','flux3-image-klein-terminal-host','flux3-image-klein-terminal-queue'])assert.equal(invoke(base,{profile}).status,1);
 for(const [key,value] of Object.entries({flux3_app_runner_target:'primary',flux3_terminal_host:'nax-macos',flux3_terminal_step:'run',flux3_terminal_control_sha256:'a'.repeat(64),flux3_queue_control_sha256:'b'.repeat(64)}))assert.equal(invoke(base,{[key]:value}).status,1);
 for(const DISPATCH_SHA of ['', 'main', 'A'.repeat(40)])assert.equal(invoke(base,{}, {DISPATCH_SHA}).status,1);
 assert.equal(invoke(base,{ref:'b'.repeat(40)}).status,1);
 assert.equal(invoke(base,{}, {DISPATCH_EVENT:'push'}).status,1);
 assert.equal(invoke(base,{}, {DISPATCH_REPOSITORY:'another/repository'}).status,1);
 const reusable=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 assert.match(reusable,/MOSS_CAPTURE_RUN_ID: \$\{\{ inputs\.moss_evidence_run_id \}\}/);
 assert.match(reusable,/run-id: \$\{\{ inputs\.moss_evidence_run_id \}\}/);
 const verifier=readFileSync(path.join(root,'scripts/ci/moss-training-acceptance/verify_evidence.py'),'utf8');
 assert.ok(verifier.includes('inspection/capture run selection mismatch'));
});

test('native authentication uses the original cell budget, not the discovery timeout',async t=>{
 const m=mockRun(t,{previews:true}),execute=m.options.execute,RealDate=Date;let clock=RealDate.now(),preflightLimit,caseLimit;
 globalThis.Date=class extends RealDate {constructor(...args){super(...(args.length?args:[clock]));}static now(){return clock;}};
 try {
  m.options.execute=async(...args)=>{
   if(args[1].includes('--list'))assert.ok(args[3]<=BOUNDS.list);
   if(args[1][1]===PREFLIGHT){preflightLimit=args[3];clock+=BOUNDS.list+1000;}
   if(args[1][1]===testsFor('capture','mlx')[0])caseLimit=args[3];
   return execute(...args);
  };
  const r=await run({...m.options,'capture-cell':CELLS[0]});
  assert.equal(r.cellComplete,true,r.failures.join());assert.equal(preflightLimit,BOUNDS.case);
  assert.equal(caseLimit,BOUNDS.case-BOUNDS.list-1000);
  assert.equal(Date.parse(r.cases[0].caseDeadlineAt)-Date.parse(r.cases[0].caseStartedAt),BOUNDS.case);
 } finally {globalThis.Date=RealDate;}
});

test('an unreleased native writer leaves diagnostics unsealed and blocks continuation',async t=>{
 const m=mockRun(t),execute=m.options.execute;
 m.options.execute=async(...args)=>{
  const result=await execute(...args);
  if(args[1][1]===PREFLIGHT){args[4].cleanup={released:false};args[4].unsafe=true;throw new Error('owned processes did not release');}
  return result;
 };
 const r=await run(m.options);assert.equal(r.ok,false);assert.equal(r.cellComplete,false);assert.equal(m.count(),7);
 assert.deepEqual(r.members,[]);assert.match(r.failures.join(),/owned processes did not release/);assert.match(r.failures.join(),/unsealed/);
 // A late child write must not invalidate a claimed immutable member digest.
 writeFileSync(path.join(m.options.output,'base','lora','late-evidence.json'),'late write');
 assert.deepEqual(JSON.parse(readFileSync(path.join(m.options.output,'driver.json'))).members,[]);
});

test('a released native failure still seals stable failed-run diagnostics',async t=>{
 const m=mockRun(t),execute=m.options.execute;
 m.options.execute=async(...args)=>{const result=await execute(...args);if(args[1][1]===PREFLIGHT)result.exit=7;return result;};
 const r=await run(m.options);assert.equal(r.ok,false);assert.ok(r.members.some(m=>m.path==='preflight-base-lora.log'));assert.doesNotMatch(r.failures.join(),/unsealed/);
});

// Literal file bytes only; callbacks run on actual bounded read/yield boundaries.
test('independent weight hashing polls cancellation and original deadline through reads',async t=>{
 const dir=fixture(t),file=path.join(dir,'literal');writeFileSync(file,Buffer.alloc(1024*1024+17,7));
 assert.equal(await fileDigestChecked(file,()=>{}),digest(readFileSync(file)));
 for(const reason of ['owned cancellation','original deadline exceeded']) {
  let stop=false,chunks=0;
  await assert.rejects(fileDigestChecked(file,()=>{if(stop)throw new Error(reason);},()=>{chunks++;setImmediate(()=>{stop=true;});}),new RegExp(reason));
  assert.equal(chunks,1);
 }
});


// CPU-only literal receipts; no native telemetry or model executes.
function literalRootReceipt(values=[10,20]) {
 return {scope:'native-api-worker-root',pid:4242,measurement:'native-current-process-rss-sampled-maximum',
  metadataHelperAbsenceClaim:false,firstFailure:null,missingMeasurementCount:0,resolvedCeilingBytes:100,
  sampledPeakBytes:Math.max(0,...values),samples:values.map(bytes=>({at:new Date().toISOString(),bytes}))};
}
test('native root receipt binds positive sampled RSS, PID and both joins without descendant claims',()=>{
 const state={rootProcess:{pid:4242}},record={raw:{rootProcess:literalRootReceipt(),serverJoined:true,nativeBodyReturnedAtUnixMs:1}};
 assert.equal(validateNativeRoot(record,state,100).sampledPeakBytes,20);
 for(const change of [r=>r.raw.rootProcess.samples=[],r=>r.raw.rootProcess.samples[0].bytes=0,
  r=>r.raw.rootProcess.samples[0].bytes=101,r=>r.raw.rootProcess.missingMeasurementCount=1,
  r=>r.raw.rootProcess.firstFailure='first cap breach',r=>r.raw.rootProcess.pid=4243,
  r=>r.raw.serverJoined=false,r=>delete r.raw.nativeBodyReturnedAtUnixMs,
  r=>r.raw.rootProcess.resolvedCeilingBytes=101,r=>r.raw.rootProcess.sampledPeakBytes=19,
  r=>r.raw.rootProcess.metadataHelperAbsenceClaim=true]){
  const bad=structuredClone(record);change(bad);assert.throws(()=>validateNativeRoot(bad,state,100));
 }
});
test('native root completion waits for owned close; cancellation and deadline never create a second grace',async t=>{
 for(const mode of ['green','cancel','deadline','unclosed']){
  const dir=fixture(t),state={},child=Object.assign(new EventEmitter(),{pid:4242,stdout:new PassThrough(),stderr:new PassThrough()});
  const log=path.join(dir,`${mode}.log`);let clock=1000;
  const timer=setInterval(()=>{clock+=25;if(mode==='cancel'||mode==='unclosed')state.cancelled=true;
    if(mode!=='unclosed'&&clock>=1075)child.emit('close',0);},1);t.after(()=>clearInterval(timer));
  t.mock.method(Date,'now',()=>clock);
  const pending=executeSupervised(dir,['/mock/api'],log,mode==='deadline'?50:1000,state,{spawnChild:()=>child,releaseMs:100,executableDigest:()=> 'f'.repeat(64)});
  if(mode==='green'){assert.equal((await pending).exit,0);assert.equal(state.cleanup.released,true);}
  else{await assert.rejects(pending,/cancellation|deadline|cooperatively close/);assert.equal(state.cleanup.released,mode!=='unclosed');}
  assert.equal(state.cleanup.scope,'native-api-worker-root');assert.equal(state.cleanup.metadataHelperAbsenceClaim,false);
  assert.equal(state.unsafe,mode==='unclosed');clearInterval(timer);t.mock.restoreAll();
 }
});

// Real CPU child verifies the actual Node-owned exit/stdio seam, never model execution.
test('native root parent records actual child closure and exact executable SHA without OS census',async t=>{
 const dir=fixture(t);
 for(const code of [0,7]){
  const state={};const result=await executeSupervised(root,[process.execPath,'-e',`console.log('metadata only');process.exit(${code})`],path.join(dir,`child-${code}.log`),5000,state);
  assert.equal(result.exit,code);assert.match(result.output,/metadata only/);
  assert.ok(state.rootProcess.pid>0);assert.equal(state.rootProcess.binarySha256,digest(readFileSync(process.execPath)));
  assert.equal(state.cleanup.released,true);assert.equal(state.cleanup.stdioClosed,true);assert.equal(state.unsafe,false);
 }
});


const boundedEntry={name:BOUNDED_PRODUCT,kind:'test'};
function mockBoundedClaim() {
 return {...source,selector:BOUNDED_PRODUCT,deviceUuid:BOUNDED_GPU_UUID,physicalIndex:1,activityProfile:'non-isolated-correctness-headroom',childGpuCapBytes:BOUNDED_GPU_CAP_BYTES,gpuFreeReserveBytes:BOUNDED_GPU_RESERVE_BYTES,visibility:{CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:'1',NVIDIA_VISIBLE_DEVICES:'1'},ordinalBinding:{mapping:[{cuda_ordinal:0,nvml_index:1,uuid:BOUNDED_GPU_UUID}]}};
}
function readerLine(weight, selector=BOUNDED_PRODUCT) {
 return `MODEL_INFERENCE_INPUT_USED ${selector} ${JSON.stringify({schema_version:1,reader:'candle-gen-moss-image::loader::load_weights_cancellable',role:'model-weights',complete:true,items:[{absolute_lexical_path:{display:weight.canonicalPath},provenance:{kind:'successful-reader-binding',content_integrity:'not_attested'}}]})}\n`;
}
function mockBoundedRun(t,{missingReader=false,wrongEvidence=false,releaseFailure=false}={}) {
 const m=mockRun(t,{backend:'cuda'}),previous=m.options.execute;let releases=0;
 m.options.phase='bounded-product';
 m.options.deviceClaim=async()=>({admission:mockBoundedClaim(),claimFile:'/SCHEMA_TEST_ONLY/claim.json',alive:()=>true,release:async()=>{releases++;if(releaseFailure)throw new Error('SCHEMA_TEST_ONLY release refused');return {...mockBoundedClaim(),state:'released',postExecutionVerified:true,leaseReleased:true,nativeProcessExited:true,nativeRoot:{pid:4242,binarySha256:'f'.repeat(64)}};}});
 m.options.execute=async(...args)=>{
  const command=args[1];const result=await previous(...args);
  if(command.includes('--list'))result.output=`${BOUNDED_PRODUCT}: test`;
  if(command[1]===BOUNDED_PRODUCT) {
   const environment=args[5].env;
   assert.equal(environment.MODEL_INFERENCE_CASE_FQN,BOUNDED_PRODUCT);assert.equal(environment.CUDA_VISIBLE_DEVICES,'1');assert.equal(environment.NVIDIA_VISIBLE_DEVICES,'1');
   assert.equal(environment.RUST_MIN_STACK,String(NATIVE_TEST_STACK_BYTES));assert.equal(args[5].capBytes,BOUNDED_GPU_CAP_BYTES);
   const weight={canonicalPath:path.join(m.options.output,'SCHEMA_TEST_ONLY.safetensors'),sha256:'d'.repeat(64),expectedSha256:'d'.repeat(64),trustedHashMatches:true};
   const file=path.join(environment.MOSS_ACCEPTANCE_OUTPUT,'evidence.json');const evidence=JSON.parse(readFileSync(file));
   Object.assign(evidence,{recordKind:wrongEvidence?'campaign-acceptance-evidence':'bounded-product-acceptance-evidence',isCampaignEvidence:false,acceptanceDisposition:'accepted',selector:BOUNDED_PRODUCT,phase:'bounded-product',caseId:'cuda/base/lora',source:{...source,runId:'123',runAttempt:'1'}});
   evidence.raw.rootProcess.resolvedCeilingBytes=BOUNDED_GPU_CAP_BYTES;evidence.raw.weightFiles=[weight];
   const pixels=mockPng();writeFileSync(path.join(environment.MOSS_ACCEPTANCE_OUTPUT,'sample.png'),pixels);evidence.raw.sample={relativePath:'sample.png',sha256:digest(pixels)};
   writeFileSync(file,JSON.stringify(evidence));if(!missingReader)result.output+=readerLine(weight);
  }
  return result;
 };
 return {...m,releases:()=>releases};
}

test('bounded product selector is exact, ignored, CUDA-only and independent of the old campaign',()=>{
 assert.deepEqual(checkBoundedDiscovery([...entries('cuda'),boundedEntry],[...entries('cuda'),boundedEntry]),[BOUNDED_PRODUCT]);
 for(const broken of [[],[boundedEntry,boundedEntry],[{...boundedEntry,name:BOUNDED_PRODUCT+'_extra'}]])assert.throws(()=>checkBoundedDiscovery(broken,[boundedEntry]),/discovery/);
 assert.throws(()=>checkBoundedDiscovery([boundedEntry],[]),/ignored/);
 const args=['--phase','bounded-product','--backend','cuda','--root','/source','--output','/output','--manifest','/manifest'];
 assert.equal(parseArgs(args).phase,'bounded-product');
 assert.throws(()=>parseArgs(args.map(value=>value==='cuda'?'mlx':value)),/CUDA/);
 assert.throws(()=>parseArgs([...args,'--peer-artifacts','/old']),/campaign/);
});

test('bounded GPU claim rejects wrong physical identity, policy, mapping and source',()=>{
 const claim=mockBoundedClaim();validateBoundedClaim(claim,source);
 for(const patch of [{selector:PREFLIGHT},{physicalIndex:0},{deviceUuid:'GPU-other'},{activityProfile:'strict-zero-utilization'},{childGpuCapBytes:BOUNDED_GPU_CAP_BYTES+1},{gpuFreeReserveBytes:0},{workflowSource:'f'.repeat(40)},{ordinalBinding:{mapping:[{cuda_ordinal:1,nvml_index:1,uuid:BOUNDED_GPU_UUID}]}},{visibility:{...claim.visibility,NVIDIA_VISIBLE_DEVICES:'0'}}])assert.throws(()=>validateBoundedClaim({...claim,...patch},source));
});

test('bounded mock invocation compiles proof feature then runs one body and no learned CPU or campaign test',async t=>{
 const m=mockBoundedRun(t),receipt=await run(m.options);assert.equal(receipt.ok,true,receipt.failures.join());
 assert.equal(receipt.cases.length,1);assert.equal(receipt.cases[0].test,BOUNDED_PRODUCT);assert.equal(receipt.isCampaignEvidence,false);assert.equal(m.releases(),1);
 assert.equal(receipt.commands.length,6);assert.deepEqual(receipt.commands[2].command.slice(0,9),['cargo','test','--locked','-p','sceneworks-rust-api','--lib','--features','backend-candle,model-inference-proof','--no-run']);
 assert.deepEqual(receipt.commands[5].command,['/mock/api',BOUNDED_PRODUCT,'--ignored','--exact','--nocapture','--test-threads=1']);
 assert.equal(receipt.commands.some(row=>row.command.includes(PREFLIGHT)||row.command.includes(PRE_MODEL_CPU_TEST)),false);
 assert.equal(receipt.inputConsumption.consumedWeights.length,1);
});

test('bounded run refuses campaign-shaped evidence, missing actual readers and failed lease release',async t=>{
 for(const options of [{wrongEvidence:true},{missingReader:true},{releaseFailure:true}]){
  const m=mockBoundedRun(t,options),receipt=await run(m.options);assert.equal(receipt.ok,false);assert.equal(m.releases(),1);assert.equal(receipt.cases.length,1);
 }
});

test('bounded consumption binds successful source readers to every selected safetensors hash',t=>{
 const weight={canonicalPath:path.join(fixture(t),'SCHEMA_TEST_ONLY.safetensors'),sha256:'a'.repeat(64),expectedSha256:'a'.repeat(64),trustedHashMatches:true};
 assert.equal(boundedReaderBindings(readerLine(weight),[weight]).consumedWeights[0].sha256,weight.sha256);
 assert.throws(()=>boundedReaderBindings(readerLine(weight,'wrong::scope'),[weight]),/another selector/);
 assert.throws(()=>boundedReaderBindings('',[weight]),/every selected/);
 assert.throws(()=>boundedReaderBindings(readerLine(weight),[{...weight,trustedHashMatches:false}]),/published/);
 assert.throws(()=>boundedReaderBindings(readerLine({...weight,canonicalPath:weight.canonicalPath+'.wrong'}),[weight]),/absent/);
 assert.throws(()=>boundedReaderBindings(readerLine(weight),[weight,{...weight,canonicalPath:weight.canonicalPath+'2.safetensors'}]),/every selected/);
});

test('bounded manual workflow uses GPU1 and isolated phase, preserving the existing GPU0 campaign',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 const bounded=workflow.split('  bounded-product:\n')[1].split('  capture:\n')[0];
 assert.match(bounded,/inputs\.moss_phase == 'bounded-product'/);assert.match(bounded,/CUDA_VISIBLE_DEVICES: "1"/);assert.match(bounded,/NVIDIA_VISIBLE_DEVICES: "1"/);assert.match(bounded,/--phase bounded-product/);
 assert.doesNotMatch(bounded,/moss-capture-cell|--phase capture|--phase cross-load|--phase finalize/);
 const guard=readFileSync(path.join(root,'scripts/ci/moss-training-acceptance/bounded_product_claim.py'),'utf8');
 assert.match(guard,/require_stable_contexts=False/);assert.match(guard,/resources\.proof_cuda_idle/);assert.match(guard,/source_modules/);assert.doesNotMatch(guard,/bound_cuda_activity_profile|load_registry|configured_case/);
});


test('bounded sole served PNG must be present, authenticated, decoded and inside its output',t=>{
 const output=fixture(t),bytes=mockPng();writeFileSync(path.join(output,'sample.png'),bytes);
 const sample={relativePath:'sample.png',sha256:digest(bytes)};assert.equal(validateBoundedSample(sample,output).decoded.width,256);
 assert.throws(()=>validateBoundedSample({...sample,sha256:'f'.repeat(64)},output),/changed/);
 assert.throws(()=>validateBoundedSample({...sample,relativePath:'../sample.png'},output),/invalid/);
 writeFileSync(path.join(output,'sample.png'),'not an image');assert.throws(()=>validateBoundedSample({...sample,sha256:digest(Buffer.from('not an image'))},output),/not PNG/);
});


test('bounded loss of the lease cooperatively cancels the authenticated child before accepting closure',async t=>{
 const dir=fixture(t),state={};let bound;
 const code="const fs=require('fs');const timer=setInterval(()=>{if(fs.existsSync(process.env.MOSS_ACCEPTANCE_CANCEL_FILE)){clearInterval(timer);process.exit(0);}},5);";
 await assert.rejects(executeSupervised(root,[process.execPath,'-e',code],path.join(dir,'guard-loss.log'),5000,state,{guardAlive:()=>false,onSpawn:value=>{bound=value;}}),/claim helper exited/);
 assert.deepEqual(bound,state.rootProcess);assert.equal(state.cleanup.released,true);assert.equal(state.cleanup.stdioClosed,true);assert.equal(state.unsafe,false);
});
