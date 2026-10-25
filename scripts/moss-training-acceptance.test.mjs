import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { digest } from './moss-consumer-assertions.mjs';
import { MANIFEST_SHA256, PREFIX, PREFLIGHT, CELLS, BOUNDS, testsFor, parseArgs, artifactMembers, verifyBundle, checkDiscovery, validatePreflight, observeOwned, remainingCaseMs, run } from './moss-training-acceptance.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const fixture=t=>{const dir=mkdtempSync(path.join(os.tmpdir(),'moss-harness-unit-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));return dir;};
const source={workflowSource:'a'.repeat(40),inferencePin:'b'.repeat(40),manifestSha256:MANIFEST_SHA256,cargoLockSha256:'c'.repeat(64),driverSha256:'d'.repeat(64),workflowSha256:'e'.repeat(64)};
const entries=()=>[...testsFor('capture'),...testsFor('cross-load'),PREFLIGHT].map(name=>({name,kind:'test'}));
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
 const p={...source,observationSchema:{gradientObservations:true,fixedEvaluationPair:true,numericFactorDelta:true,preparedTensorSnapshots:true},backend:'mlx',caseId:'mlx/base/lora',verdict:'admitted',refusalReason:null,modelRevisions:{dit:'f'.repeat(40)},resolvedCapProvenance:'native host capacity + product rounding',headerDemandProvenance:'native estimate all phases',offline:true,weightsLoaded:false,deviceIdentity:'claimed device',gpuMemoryLimitFraction:.60,totalCapacityBytes:1000,resolvedCapBytes:600,headerDerivedDemandBytes:500,weights:[{path:weight,present:true,sha256:digest(readFileSync(weight))}]};
 assert.equal(validatePreflight(p,'mlx',source,p.caseId),600);
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
 const r={ok:true,phase:'capture',backend:'mlx',source,members:artifactMembers(dir)};
 writeFileSync(path.join(dir,'driver.json'),JSON.stringify(r));assert.equal(verifyBundle(dir,source,'capture','mlx').ok,true);
 assert.throws(()=>verifyBundle(dir,{...source,inferencePin:'f'.repeat(40)},'capture','mlx'),/source mismatch/);
 writeFileSync(path.join(dir,'raw.log'),'changed');assert.throws(()=>verifyBundle(dir,source,'capture','mlx'),/members/);
});

function mockRun(t, {badCounts=false,missingEvidence=false,missingRss=false,unsafe=false}={}) {
 const output=fixture(t);let executions=0;
 const execute=async(_root,command,log,_limit,state,{env})=>{
  executions++;state.cleanup={released:!unsafe};
  let text='';
  if(command[0]==='rustc')text='rustc 1.97.1 (SCHEMA_TEST_ONLY)\nhost: aarch64-apple-darwin\n';
  else if(command[0]==='cargo' && command[1]==='-V')text='cargo 1.97.1 (SCHEMA_TEST_ONLY)\n';
  else if(command[0]==='cargo')text=JSON.stringify({reason:'compiler-artifact',executable:'/mock/api',target:{name:'sceneworks_rust_api',kind:['lib']}});
  else if(command.includes('--list'))text=entries().map(e=>`${e.name}: test`).join('\n');
  else if(command[1]===PREFLIGHT){writeFileSync(env.MOSS_ACCEPTANCE_DEVICE_PREFLIGHT,'{}');text=goodOutput(PREFLIGHT);}
  else {state.rssFirstBytes=missingRss?undefined:10;state.rssPeakBytes=20;text=badCounts?'test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out':goodOutput(command[1]);if(!missingEvidence)writeFileSync(path.join(env.MOSS_ACCEPTANCE_OUTPUT,'evidence.json'),JSON.stringify({raw:'NOT ACCEPTANCE EVIDENCE'}));}
  writeFileSync(log,text);return {exit:0,output:text};
 };
 return {options:{root,output,backend:'mlx',phase:'capture',manifest:path.join(root,'scripts/ci/moss-training-acceptance/campaign.json'),testPlatform:true,env:{GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_REF:'refs/heads/SCHEMA_TEST_ONLY',GITHUB_SHA:source.workflowSource,GITHUB_RUN_ID:'123',GITHUB_RUN_ATTEMPT:'1',GITHUB_JOB:'SCHEMA_TEST_ONLY'},sourceProvider:()=>source,execute,statExecutable:()=>true,executableDigest:()=> 'f'.repeat(64),preflightValidator:()=>100},count:()=>executions};
}
test('mock capture performs sequential exact native tests and stores raw evidence without asserting learning',async t=>{
 const m=mockRun(t);const r=await run(m.options);assert.equal(r.ok,true);assert.equal(r.cases.length,4);assert.ok(r.cases.every(c=>c.executionPassed));assert.equal(m.count(),13);assert.equal(r.toolchain.binarySha256,'f'.repeat(64));assert.match(r.toolchain.rustc,/SCHEMA_TEST_ONLY/);assert.equal(r.workflow.runId,'123');
 assert.equal(r.accepted,undefined);assert.ok(r.members.some(m=>m.path.endsWith('evidence.json')));
 for(const command of r.commands.filter(c=>c.command[1]?.startsWith(PREFIX)))assert.ok(command.command.includes('--exact')&&command.command.includes('--ignored'));
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
