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
import { MANIFEST_SHA256, PREFIX, PREFLIGHT, CELLS, BOUNDS, NATIVE_TEST_STACK_BYTES, testsFor, parseArgs, artifactMembers, verifyBundle, checkDiscovery, validatePreflight, observeOwned, processRows, executeSupervised, remainingCaseMs, decodePreviewPng, run } from './moss-training-acceptance.mjs';
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
test('preflight uses selected actual capacity and native provenance, not env asserted cap',t=>{
 const dir=fixture(t),weight=path.join(dir,'cached');writeFileSync(weight,'cached header');
 const p={...source,phase:'capture',observationSchema:{gradientObservations:true,fixedEvaluationPair:true,numericFactorDelta:true,preparedTensorSnapshots:true},backend:'mlx',caseId:'mlx/base/lora',verdict:'admitted',refusalReason:null,modelRevisions:{dit:'f'.repeat(40)},resolvedCapProvenance:'native host capacity + product rounding',headerDemandProvenance:'native estimate all phases',offline:true,weightsLoaded:false,deviceIdentity:'claimed device',gpuMemoryLimitFraction:.60,totalCapacityBytes:1000,resolvedCapBytes:600,headerDerivedDemandBytes:500,weights:[{path:weight,present:true,sha256:digest(readFileSync(weight))}]};
 assert.equal(validatePreflight(p,'mlx',source,p.caseId),600);
 assert.equal(validatePreflight({...p,phase:'cross-load',observationSchema:null},'mlx',source,p.caseId,'cross-load'),600);
 assert.throws(()=>validatePreflight({...p,phase:'cross-load'},'mlx',source,p.caseId),/phase/);
 for(const patch of [{resolvedCapBytes:601},{headerDerivedDemandBytes:601},{offline:false},{weightsLoaded:true},{deviceIdentity:''},{caseId:'mlx/sft/lora'},{modelRevisions:{}},{verdict:'refused'},{resolvedCapProvenance:''},{headerDemandProvenance:''},{observationSchema:{}},{observationSchema:{gradientObservations:true,fixedEvaluationPair:true,numericFactorDelta:true,preparedTensorSnapshots:false}}])assert.throws(()=>validatePreflight({...p,...patch},'mlx',source,p.caseId));
 assert.throws(()=>validatePreflight({...p,weights:[{...p.weights[0],present:false}]},'mlx',source,p.caseId));
 writeFileSync(weight,'changed');assert.throws(()=>validatePreflight(p,'mlx',source,p.caseId),/changed/);
});
test('owned RSS includes root, worker, grandchildren and refuses missing telemetry',()=>{
 const rows=new Map([[42,{pid:42,ppid:process.pid,birth:'one',executable:'/api',rssBytes:10}],[43,{pid:43,ppid:42,birth:'two',executable:'/worker',rssBytes:20}],[44,{pid:44,ppid:43,birth:'three',executable:'/child',rssBytes:30}],[45,{pid:45,ppid:1,birth:'other',executable:'/stranger',rssBytes:1000}]]);
 const known=new Map();assert.equal(observeOwned(rows,known,42).rss,60);assert.equal(known.size,3);
 rows.delete(42);rows.get(43).ppid=1;assert.equal(observeOwned(rows,known,42).rss,50);
 assert.throws(()=>observeOwned(null,known,42),/unavailable/);
 // A measured zero-RSS exiting child is valid while the aggregate stays positive.
 rows.get(43).rssBytes=0;assert.equal(observeOwned(rows,known,42).rss,30);
 rows.get(43).rssBytes=undefined;assert.throws(()=>observeOwned(rows,known,42),/missing/);
});
test('owned tree never adopts PID-reused descendant',()=>{
 const known=new Map([[2,{pid:2,birth:'old',executable:'/worker'}]]);
 const rows=new Map([[1,{pid:1,ppid:process.pid,birth:'root',executable:'/api',rssBytes:1}],[2,{pid:2,ppid:1,birth:'new',executable:'/stranger',rssBytes:1}]]);
 assert.throws(()=>observeOwned(rows,known,1),/reused/);
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
  executions++;if(isPreparationCommand(root,command))state.completion={scope:"root-and-stdio",completed:!unsafe};else state.cleanup={released:!unsafe};
  let text='';
  if(command[0]==='rustc')text='rustc 1.97.1 (SCHEMA_TEST_ONLY)\nhost: aarch64-apple-darwin\n';
  else if(command[0]==='cargo' && command[1]==='-V')text='cargo 1.97.1 (SCHEMA_TEST_ONLY)\n';
  else if(command[0]==='cargo')text=JSON.stringify({reason:'compiler-artifact',executable:'/mock/api',target:{name:'sceneworks_rust_api',kind:['lib']}});
  else if(command.includes('--list'))text=[...entries(backend),...(!command.includes('--ignored')?[{name:PRE_MODEL_CPU_TEST}]:[])].map(e=>`${e.name}: test`).join('\n');
  else if(command[1]===PRE_MODEL_CPU_TEST)text=goodOutput(PRE_MODEL_CPU_TEST);
  else if(command[1]===PREFLIGHT){writeFileSync(env.MOSS_ACCEPTANCE_DEVICE_PREFLIGHT,'{}');text=goodOutput(PREFLIGHT);}
  else {state.rssFirstBytes=missingRss?undefined:10;state.rssPeakBytes=20;text=badCounts?'test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out':goodOutput(command[1]);if(!missingEvidence) {
   const raw=previews?mockPreviewEvidence(env.MOSS_ACCEPTANCE_OUTPUT,env.MOSS_ACCEPTANCE_CASE_ID):{raw:'NOT ACCEPTANCE EVIDENCE'};
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
 for(const job of ['qwen-app-lifecycle','iris-mlx-real-weights','mlx','candle'])assert.match(dispatcher,new RegExp(`  ${job}:\\n    if: \\$\\{\\{ !inputs.run_moss_training_acceptance`));
 assert.match(dispatcher,/default: unspecified/);assert.match(dispatcher,/needs: moss-selector-preflight/);
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
test('both CUDA phases preserve established PCI device 1 exclusion and explicit Windows preparation workspace',()=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 const authoritative=readFileSync(path.join(root,'.github/workflows/windows-candle.yml'),'utf8');
 assert.match(authoritative,/group: windows-candle-gpu-real-weights\n      cancel-in-progress: false/);
 assert.match(authoritative,/CUDA_DEVICE_ORDER: PCI_BUS_ID\n      CUDA_VISIBLE_DEVICES: "1"/);
 for(const job of ['capture','cross-load']) {
  const section=workflow.split(`  ${job}:\n`)[1].split(/^  [a-z][a-z-]+:\n/m)[0];
  assert.match(section,/matrix\.backend == 'cuda' && 'windows-candle-gpu-real-weights' \|\| format\('moss-training-\{0\}-\{1\}-\{2\}-mlx', github\.run_id, github\.run_attempt, github\.job\)/);
  assert.match(section,/cancel-in-progress: false/);assert.match(section,/CUDA_DEVICE_ORDER: PCI_BUS_ID/);
  assert.match(section,/CUDA_VISIBLE_DEVICES: \$\{\{ matrix\.backend == 'cuda' && '1' \|\| '' \}\}/);
  assert.match(section,/NVIDIA_VISIBLE_DEVICES: \$\{\{ matrix\.backend == 'cuda' && '1' \|\| '' \}\}/);
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
test('Windows birth guards reject stale root PID and each stale ParentProcessId edge',()=>{
 const row=(pid,ppid,birthMs,rssBytes=1)=>({pid,ppid,birthMs,birth:String(birthMs),executable:`mock-${pid}`,rssBytes});
 const rows=new Map([[process.pid,row(process.pid,0,100)],[42,row(42,process.pid,200,10)],[43,row(43,42,150,1000)],[44,row(44,42,300,20)],[45,row(45,44,250,1000)],[46,row(46,44,400,0)]]);
 const known=new Map(),observed=observeOwned(rows,known,42);
 assert.equal(observed.rss,30);assert.deepEqual(observed.live.map(r=>r.pid).sort((a,b)=>a-b),[42,44,46]);assert.equal(known.has(43),false);assert.equal(known.has(45),false);
 rows.set(42,row(42,process.pid,50));assert.throws(()=>observeOwned(rows,new Map(),42),/root birth/);
 rows.set(42,row(42,process.pid,200));delete rows.get(44).birthMs;assert.throws(()=>observeOwned(rows,new Map(),42),/birth telemetry/);
});
test('Windows stale parent edges exclude blank unrelated paths before authenticating genuine children',()=>{
 const row=(pid,ppid,birthMs,executable=`owned-${pid}`)=>({pid,ppid,birthMs,birth:String(birthMs),executable,rssBytes:1});
 const rows=new Map([[process.pid,row(process.pid,0,100)],[42,row(42,process.pid,200)],[43,row(43,42,150,'')],[44,row(44,42,300)],[45,row(45,44,250,'')]]);
 const known=new Map();assert.deepEqual(observeOwned(rows,known,42,()=>assert.fail('stale rows are not owned identity failures')).live.map(r=>r.pid),[42,44]);
 assert.equal(known.has(43),false);assert.equal(known.has(45),false);
 let witness;rows.set(46,row(46,44,400,''));
 assert.throws(()=>observeOwned(rows,known,42,value=>{witness=value;}),/descendant identity missing/);
 assert.equal(witness.rootPid,42);assert.equal(witness.candidate.pid,46);assert.equal(witness.candidate.executable,'');assert.equal(witness.parent.pid,44);assert.equal(witness.parent.birthMs,300);
 rows.delete(46);const unknown=row(47,44,400,'');delete unknown.birthMs;rows.set(47,unknown);
 assert.throws(()=>observeOwned(rows,known,42),/Windows birth telemetry missing/);
});
test('owned identity refusal records only one bounded candidate relation',()=>{
 const parent={pid:42,ppid:process.pid,birth:'owned-birth',executable:'p'.repeat(1000),rssBytes:1};
 const rows=new Map([[42,parent],[43,{pid:43,ppid:42,birth:'',executable:'',rssBytes:1}],[99,{pid:99,ppid:1,secret:'unrelated process command line'}]]);let witness;
 assert.throws(()=>observeOwned(rows,new Map(),42,value=>{witness=value;}),/descendant identity missing/);
 assert.equal(witness.parent.executable.length,512);assert.equal(witness.candidate.pid,43);assert.equal(JSON.stringify(witness).includes('secret'),false);assert.equal(Object.hasOwn(witness,'rows'),false);
});
const winRow=(pid,ppid,birthMs,executable,rssBytes=1)=>({pid,ppid,birthMs,birth:String(birthMs),executable,rssBytes});
function actualWindowsWitness(){
 const parent=winRow(44816,process.pid,1793073157759,'d:/owned/native.exe',380723200);
 const candidate=winRow(57240,44816,1793073218066,'',28229632);
 return {parent,candidate,rows:new Map([[process.pid,winRow(process.pid,0,1793073150000,'d:/node.exe')],[44816,parent],[57240,candidate]])};
}
test('Windows exact-birth recheck resolves the actual young witness once without lowering its observed RSS',()=>{
 for(const outcome of ['resolved','absent','pid-reused']){
  const {rows,candidate}=actualWindowsWitness(),known=new Map(),pending=new Map();let resolution,calls=0;
  const fresh=outcome==='absent'?new Map():new Map([[57240,{...candidate,executable:'c:/metadata.exe',birthMs:candidate.birthMs+(outcome==='pid-reused'?1:0),birth:String(candidate.birthMs+(outcome==='pid-reused'?1:0)),rssBytes:1}]]);
  const observed=observeOwned(rows,known,44816,null,{pending,recheckWindows:pids=>{calls++;assert.deepEqual(pids,[57240]);return fresh;},onResolution:value=>{resolution=value;}});
  assert.equal(calls,1);assert.equal(resolution.outcome,outcome);assert.equal(observed.rss,380723200+28229632);
  assert.equal(known.has(57240),outcome==='resolved');assert.deepEqual(observed.live.map(r=>r.pid),outcome==='resolved'?[44816,57240]:[44816]);
 }
});
test('an initially unresolved Windows root cannot turn its own disappearance into initial identity admission',()=>{
 const {rows,parent}=actualWindowsWitness();rows.delete(57240);rows.set(44816,{...parent,executable:''});
 assert.throws(()=>observeOwned(rows,new Map(),44816,null,{recheckWindows:()=>new Map()}),/initial owned root identity unavailable/);
});
test('Windows live blank, denied and malformed exact-reader results never count as disappearance',()=>{
 const {candidate}=actualWindowsWitness();
 for(const fresh of [null,new Map([[57240,candidate]]),new Map([[57240,{...candidate,ppid:99}]]),new Map([[57240,{...candidate,birthMs:null}]]),new Map([[99,{...candidate,pid:99}]])]){
  const {rows}=actualWindowsWitness(),pending=new Map();let failure;
  assert.throws(()=>observeOwned(rows,new Map(),44816,value=>{failure=value;},{pending,recheckWindows:()=>fresh}),/descendant identity missing/);
  assert.equal(failure.candidate.pid,57240);assert.ok(failure.recheck);assert.equal(pending.get(57240).birth,candidate.birth);
 }
 const {rows}=actualWindowsWitness();assert.throws(()=>observeOwned(rows,new Map(),44816,null,{recheckWindows:()=>{throw new Error('denied');}}),/descendant identity missing/);
});
test('Windows selected reread batches only unresolved owned rows and retains witnessed offspring after parent disappearance',()=>{
 const {rows,candidate}=actualWindowsWitness();rows.set(57241,winRow(57241,57240,candidate.birthMs+1,'c:/child.exe',3));rows.set(99,winRow(99,1,100,'',999999));
 const observed=observeOwned(rows,new Map(),44816,null,{recheckWindows:pids=>{assert.deepEqual(pids,[57240]);return new Map();}});
 assert.deepEqual(observed.live.map(r=>r.pid),[44816,57241]);assert.equal(observed.rss,380723200+28229632+3);
});
test('known Windows blank paths require resolution and cannot become an exec; POSIX exec remains supported',()=>{
 for(const outcome of ['resolved','absent','live-blank','changed-image']){
  const {rows,candidate,parent}=actualWindowsWitness();const original={...candidate,executable:'c:/original.exe'};const known=new Map([[44816,parent],[57240,original]]);
  const current=outcome==='absent'?new Map():new Map([[57240,{...candidate,executable:outcome==='resolved'?'c:/original.exe':outcome==='changed-image'?'c:/different.exe':''}]]);
  const act=()=>observeOwned(rows,known,44816,null,{recheckWindows:()=>current});
  if(outcome==='live-blank'||outcome==='changed-image')assert.throws(act,/identity missing|Windows executable differs/);else act();
  assert.equal(known.get(57240).executable,'c:/original.exe');
 }
 const direct=actualWindowsWitness();direct.rows.set(57240,{...direct.candidate,executable:'c:/changed.exe'});
 assert.throws(()=>observeOwned(direct.rows,new Map([[44816,direct.parent],[57240,{...direct.candidate,executable:'c:/original.exe'}]]),44816),/Windows executable differs/);
 const old={pid:42,ppid:process.pid,birth:'same-birth',executable:'/old',rssBytes:10};const known=new Map([[42,old]]);
 const observed=observeOwned(new Map([[42,{...old,executable:'/new'}]]),known,42);assert.equal(observed.live[0].executable,'/new');assert.equal(known.get(42).executable,'/new');
});
test('POSIX census records bounded failure facts while preserving null telemetry and the exact query budget',()=>{
 for(const result of [{error:Object.assign(new Error('spawnSync ps ETIMEDOUT'),{code:'ETIMEDOUT'}),status:null,signal:'SIGTERM'}, {error:null,status:1,signal:null}]) {
  let failure;const rows=processRows('darwin',value=>{failure=value;},(command,args,options)=>{
   assert.equal(command,'ps');assert.deepEqual(args,['-axo','pid=,ppid=,rss=,lstart=,comm=']);assert.equal(options.timeout,3000);assert.equal(options.maxBuffer,16*1024*1024);assert.equal(options.env.LC_ALL,'C');
   return {...result,stdout:'PRIVATE PROCESS TABLE MUST NOT BE RECORDED',stderr:'diagnostic '.repeat(100)};
  });
  assert.equal(rows,null);assert.throws(()=>observeOwned(rows,new Map(),42),/owned process telemetry unavailable/);
  assert.equal(failure.command,'ps');assert.equal(failure.timeoutMs,3000);assert.ok(failure.elapsedMs>=0);assert.equal(failure.status,result.status);assert.equal(failure.signal,result.signal);assert.equal(failure.error?.code??null,result.error?.code??null);
  assert.equal(failure.stderr.length,512);assert.equal(JSON.stringify(failure).includes('PRIVATE PROCESS TABLE'),false);assert.equal(Object.hasOwn(failure,'env'),false);
 }
});
test('successful POSIX census preserves parsing without emitting an availability failure',()=>{
 for(const stdout of ['42 1 10 Mon Oct 26 20:08:34 2026 /owned/node\n','unparseable row\n']) {
  const rows=processRows('darwin',()=>assert.fail('successful ps is not unavailable telemetry'),()=>({status:0,stdout}));
  assert.ok(rows instanceof Map);assert.equal(rows.size,stdout.startsWith('42')?1:0);
 }
});
// Controlled census/close ordering: no host process queries, model execution or signals.
function exitRaceChild(samples) {
 const child=Object.assign(new EventEmitter(),{pid:4242,stdout:new PassThrough(),stderr:new PassThrough()});let queries=0;
 return {child,get queries(){return queries;},query:()=>{
  const sample=samples[Math.min(queries++,samples.length-1)];
  if(Object.hasOwn(sample,'close'))queueMicrotask(()=>child.emit('close',sample.close));
  return sample.rows===null?null:new Map(sample.rows.map(row=>[row.pid,row]));
 }};
}
const rssRoot=rssBytes=>({pid:4242,ppid:process.pid,birth:'owned-root-birth',executable:'/owned/native',rssBytes});
const rssDescendant=(rssBytes,ppid=4242)=>({pid:4243,ppid,birth:'owned-child-birth',executable:'/owned/helper',rssBytes});
test('RSS shutdown race preserves the cap refusal until close and fresh owned absence',async t=>{
 const dir=fixture(t),log=path.join(dir,'case.log'),state={};
 const probe=exitRaceChild([{rows:[rssRoot(10)]},{rows:[rssRoot(21)]},{rows:[rssRoot(0)],close:7},{rows:[]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],log,3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:1000}),/aggregate owned RSS exceeded cap \(21 > 20\)/);
 assert.equal(state.cancelled,true);assert.equal(state.rssFirstBytes,10);assert.equal(state.rssPeakBytes,21);
 assert.deepEqual(state.rssSamples.map(s=>s.aggregateBytes),[10,21]);assert.ok(probe.queries>=4);
 assert.equal(state.cleanup.released,true);assert.equal(state.unsafe,false);
 assert.match(JSON.parse(readFileSync(`${log}.cancel`)).reason,/aggregate owned RSS exceeded cap \(21 > 20\)/);
});
test('RSS shutdown race accepts exit zero only after close and a fresh empty owned census',async t=>{
 const dir=fixture(t),state={};const probe=exitRaceChild([{rows:[rssRoot(10)]},{rows:[rssRoot(0)],close:0},{rows:[]}]);
 const result=await executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:50});
 assert.equal(result.exit,0);assert.equal(state.cleanup.released,true);assert.ok(probe.queries>=3);
 assert.deepEqual(state.rssSamples.map(s=>s.aggregateBytes),[10]);
});
test('RSS shutdown race still refuses a true initial zero RSS sample',async t=>{
 const dir=fixture(t),state={};const probe=exitRaceChild([{rows:[rssRoot(0)],close:0},{rows:[]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:50}),/no positive initial aggregate owned RSS/);
 assert.equal(state.rssFirstBytes,undefined);
});
test('RSS shutdown race never turns unavailable telemetry into release proof',async t=>{
 const dir=fixture(t),state={};const probe=exitRaceChild([{rows:[rssRoot(10)]},{rows:null,close:0},{rows:null}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:1}),/owned process telemetry unavailable/);
 assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);
});
test('RSS shutdown race never releases a known surviving reparented descendant',async t=>{
 const dir=fixture(t),state={};const probe=exitRaceChild([{rows:[rssRoot(10),rssDescendant(1)]},{rows:[rssRoot(0),rssDescendant(0)],close:0},{rows:[rssDescendant(1,1)]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:50}),/owned processes did not cooperatively release/);
 assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);assert.ok(state.cleanup.observedIdentities.some(r=>r.pid===4243));
});
test('owned identity refusal persists the first candidate through later cleanup failures',async t=>{
 const dir=fixture(t),state={};const missing=pid=>({pid,ppid:4242,birth:'owned-child-birth',executable:'',rssBytes:1});
 const probe=exitRaceChild([{rows:[rssRoot(10)]},{rows:[rssRoot(10),missing(4243)]},{rows:[rssRoot(10),missing(4244)],close:7},{rows:[]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:1000}),/descendant identity missing/);
 assert.equal(state.identityFailure.candidate.pid,4243);assert.equal(state.identityFailure.parent.pid,4242);assert.equal(state.cleanup.released,true);assert.equal(state.unsafe,false);
});
test('RSS shutdown race retains the cap breach alongside unavailable cleanup telemetry',async t=>{
 const dir=fixture(t),log=path.join(dir,'case.log'),state={};const probe=exitRaceChild([{rows:[rssRoot(10)]},{rows:[rssRoot(21)]},{rows:null,close:7},{rows:null}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],log,3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:50}),error=>{
  assert.match(error.message,/owned process telemetry unavailable/);assert.match(error.message,/aggregate owned RSS exceeded cap \(21 > 20\)/);return true;
 });
 assert.equal(state.cancelled,true);assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);
 assert.match(JSON.parse(readFileSync(`${log}.cancel`)).reason,/aggregate owned RSS exceeded cap \(21 > 20\)/);
});
test('RSS shutdown race retains the cap breach alongside a surviving descendant refusal',async t=>{
 const dir=fixture(t),state={};const probe=exitRaceChild([{rows:[rssRoot(21),rssDescendant(1)]},{rows:[rssRoot(0),rssDescendant(0)],close:7},{rows:[rssDescendant(1,1)]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,releaseMs:400}),error=>{
  assert.match(error.message,/owned processes did not cooperatively release/);assert.match(error.message,/aggregate owned RSS exceeded cap \(22 > 20\)/);return true;
 });
 assert.equal(state.cancelled,true);assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);
});
test('unresolved Windows candidates and reparented observed offspring block cleanup after root close',async t=>{
 for(const keepCandidate of [true,false]){
  const dir=fixture(t),state={},parent=winRow(4242,process.pid,200,'c:/native.exe',10),candidate=winRow(4243,4242,300,'',20),leaf=winRow(4244,4243,400,'c:/leaf.exe',3);
  const control=winRow(process.pid,0,100,'c:/node.exe');const survivors=[{...leaf,ppid:1}];if(keepCandidate)survivors.push({...candidate,ppid:1});
  const probe=exitRaceChild([{rows:[control,parent,candidate,leaf]},{rows:survivors,close:7},{rows:survivors}]);
  await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{query:probe.query,spawnChild:()=>probe.child,recheckWindows:()=>null,releaseMs:1}),/descendant identity missing/);
  assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);assert.equal(state.identityFailure.candidate.pid,4243);
  if(keepCandidate)assert.ok(state.cleanup.pendingIdentities.some(row=>row.pid===4243));else assert.ok(state.cleanup.observedIdentities.some(row=>row.pid===4244));
 }
});
test('unresolved Windows cleanup requires fresh actual absence, never null telemetry',async t=>{
 for(const endRows of [[],null]){
  const dir=fixture(t),state={},control=winRow(process.pid,0,100,'c:/node.exe'),parent=winRow(4242,process.pid,200,'c:/native.exe',10),candidate=winRow(4243,4242,300,'',20);
  const probe=exitRaceChild([{rows:[control,parent,candidate]},{rows:endRows,close:7},{rows:endRows}]);
  await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{query:probe.query,spawnChild:()=>probe.child,recheckWindows:()=>null,releaseMs:endRows===null?1:1000}),/descendant identity missing/);
  assert.equal(state.cleanup.released,endRows!==null);assert.equal(state.unsafe,endRows===null);
 }
});
test('resolving a vanished Windows row cannot erase the original owned RSS cap breach',async t=>{
 const dir=fixture(t),state={},control=winRow(process.pid,0,100,'c:/node.exe'),parent=winRow(4242,process.pid,200,'c:/native.exe',10),candidate=winRow(4243,4242,300,'',30);
 const probe=exitRaceChild([{rows:[control,parent,candidate],close:7},{rows:[]}]);
 await assert.rejects(executeSupervised(dir,['controlled-native'],path.join(dir,'case.log'),3000,state,{capBytes:20,query:probe.query,spawnChild:()=>probe.child,recheckWindows:()=>new Map(),releaseMs:1000}),/aggregate owned RSS exceeded cap \(40 > 20\)/);
 assert.equal(state.rssPeakBytes,40);assert.equal(state.cancelled,true);assert.equal(state.cleanup.released,true);assert.equal(state.identityResolution.outcome,'absent');
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
 assert.match(workflow,/options: \[unspecified, real-weights, rw-llm, rw-mage, rw-krea\]/);
 assert.match(workflow,/moss_runner_label:[\s\S]*?default: unspecified/);
 for(const label of ['real-weights','rw-llm','rw-mage','rw-krea','unspecified','','nax']) {
  const result=spawnSync(process.execPath,['--input-type=module','-e',guard],{encoding:'utf8',env:{...process.env,GITHUB_OUTPUT:path.join(fixture(t),'output'),SELECTORS:JSON.stringify({moss_runner_label:label,moss_phase:'capture',moss_reference:'{}',moss_terminal:'{}'}),DISPATCH_SHA:source.workflowSource,DISPATCH_EVENT:'workflow_dispatch',DISPATCH_REPOSITORY:'SceneWorks/SceneWorks'}});
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

test('merged dispatcher fits 25 inputs and derives exact source and finalize run without redundant selectors',t=>{
 const workflow=readFileSync(path.join(root,'.github/workflows/memory-catalog-campaign.yml'),'utf8');
 const inputs=workflow.split('    inputs:\n')[1].split('\n# `gh pr create`')[0];
 const names=[...inputs.matchAll(/^      ([A-Za-z_]+):$/gm)].map(match=>match[1]);
 assert.equal(names.length,25);assert.equal(new Set(names).size,25);
 assert.equal(names.includes('workflowSource'),false);assert.equal(names.includes('moss_evidence_run_id'),false);
 assert.match(workflow,/workflowSource: \$\{\{ github\.sha \}\}/);
 assert.match(workflow,/moss_evidence_run_id: \$\{\{ needs\.moss-selector-preflight\.outputs\.capture_run_id \}\}/);
 const guard=workflow.split('      - name: Reject conflicting selectors before compiler or device jobs')[1].split("node --input-type=module <<'JS'\n")[1].split('          JS')[0].replace(/^          /gm,'');
 const base={moss_runner_label:'real-weights',moss_phase:'capture',moss_reference:'{}',moss_terminal:'{}'};
 const invoke=(changes={},environment={})=>{
  const output=path.join(fixture(t),'output');
  const result=spawnSync(process.execPath,['--input-type=module','-e',guard],{encoding:'utf8',env:{...process.env,GITHUB_OUTPUT:output,SELECTORS:JSON.stringify({...base,...changes}),DISPATCH_SHA:source.workflowSource,DISPATCH_EVENT:'workflow_dispatch',DISPATCH_REPOSITORY:'SceneWorks/SceneWorks',...environment}});
  return {...result,output:result.status===0?readFileSync(output,'utf8'):''};
 };
 assert.equal(invoke().output,'capture_run_id=\n');
 const inspection=JSON.stringify({captureRunId:'123',manifestSha256:source.manifestSha256,cases:[]});
 assert.equal(invoke({moss_phase:'finalize',moss_inspection_json:inspection}).output,'capture_run_id=123\n');
 for(const captureRunId of [undefined,null,123,'0','01','-1','1\nother=2','1e3']) {
  assert.equal(invoke({moss_phase:'finalize',moss_inspection_json:JSON.stringify({captureRunId})}).status,1);
 }
 for(const moss_inspection_json of ['', '{}', '[]', 'null', '{malformed'])assert.equal(invoke({moss_phase:'finalize',moss_inspection_json}).status,1);
 assert.equal(invoke({moss_inspection_json:inspection}).status,1);
 for(const key of ['moss_terminal','moss_reference'])for(const value of ['null','[]','broken',JSON.stringify({phase:'execute'})])assert.equal(invoke({[key]:value}).status,1);
 for(const DISPATCH_SHA of ['', 'main', 'A'.repeat(40)])assert.equal(invoke({}, {DISPATCH_SHA}).status,1);
 assert.equal(invoke({ref:'b'.repeat(40)}).status,1);
 assert.equal(invoke({}, {DISPATCH_EVENT:'push'}).status,1);
 assert.equal(invoke({}, {DISPATCH_REPOSITORY:'another/repository'}).status,1);
 const reusable=readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'),'utf8');
 assert.match(reusable,/MOSS_CAPTURE_RUN_ID: \$\{\{ inputs\.moss_evidence_run_id \}\}/);
 assert.match(reusable,/run-id: \$\{\{ inputs\.moss_evidence_run_id \}\}/);
 const verifier=readFileSync(path.join(root,'scripts/ci/moss-training-acceptance/verify_evidence.py'),'utf8');
 assert.ok(verifier.includes('inspection/capture run selection mismatch'));
});
