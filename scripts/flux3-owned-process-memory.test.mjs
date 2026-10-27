// DECLARED SYNTHETIC CPU fixtures. No process/hardware/native/model activity.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {EventEmitter} from 'node:events';
import {OwnedProcessMemoryCapture,ownedProcessCounterScript} from './lib/flux3-owned-process-memory.mjs';
import {declaredSyntheticOwnedAccounting} from './lib/flux3-owned-accounting-fixture.mjs';
import {validateOwnedProcessAccountingRecord} from './lib/flux3-owned-accounting-contract.mjs';
import {startFlux3OwnedHardwareMonitor,ownedMemoryFailure,createFlux3StartupProbeBudget} from './flux3-owned-process-memory.mjs';
const identity={pid:12345,startedAt:'2026-10-23T00:00:00.000Z',executable:'D:\\DECLARED-fixture\\memory-candle-adapter.exe',luid:'0x00000000_0x0001f363',physicalGpuId:'0',gpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',pciBusId:'00000000:21:00.0',logicalCudaOrdinal:0};
function samples() {let sequence=0;return()=>({ ...identity,sequence:++sequence,elapsedMs:sequence*200,readStartedMs:sequence*200-5,expectedBefore:[0,1],expectedAfter:[1,0],
  rows:[0,1].map(physicalNode=>({pid:identity.pid,luid:identity.luid,physicalNode,statuses:[0,1,0],dedicatedBytes:physicalNode?200:100,sharedBytes:physicalNode?20:10,totalCommittedBytes:physicalNode?250:150}))});}
function complete(capture=new OwnedProcessMemoryCapture(identity)) {
  const sample=samples();capture.sample(sample());
  for(const phase of ['load','promptUpsampling','conditioning','denoise','decode']) {capture.boundary(phase,sample());capture.sample(sample());}
  capture.boundary(null,sample());return capture.finish();
}
test('sampled whole-owned-process accounting sums every physical node without allocator/residency aliases',()=>{
  const evidence=complete();assert.deepEqual(evidence.unavailableMetrics,{physicalResidentBytes:null,activeBytes:null,allocatorBytes:null});assert.equal(evidence.sampling.maxObservedReadSkewMs,5);assert.equal(evidence.overall.dedicatedPeakBytes,300);assert.equal(evidence.overall.sharedPeakBytes,30);assert.equal(evidence.overall.totalCommittedPeakBytes,400);
  assert.deepEqual(evidence.physicalNodes,[0,1]);assert.equal(evidence.sampling.losslessPeakClaim,false);
  for(const key of ['physicalResidencyClaim','allocatorActivityClaim','deviceTotalDeltaAttributionClaim','exclusiveThroughputClaim'])assert.equal(evidence[key],false);
  for(const phase of Object.values(evidence.phases)){assert(phase.samples>=2);assert(phase.lastSampleMs>phase.firstSampleMs);assert.equal(phase.activeBytes,undefined);assert.equal(phase.allocatorBytes,undefined);}
});
test('actual reducer output satisfies the shared strict operating-envelope consumer without aliases',()=>{
  const record=declaredSyntheticOwnedAccounting({strategy:{fixtureOnly:true}});record.ownedProcessMemory=complete();validateOwnedProcessAccountingRecord(record);
});
test('unknown, missing, duplicate, negative, cross-PID/LUID and changed physical-node counter sets fail closed',()=>{
  const mutants=[s=>s.rows.pop(),s=>s.rows.push({...s.rows[0]}),s=>s.expectedBefore=[],s=>s.expectedBefore.push(0),s=>s.expectedAfter=[0],
    s=>s.rows[0].physicalNode=-1,s=>s.rows[0].physicalNode=32,s=>s.rows[0].pid++,s=>s.rows[0].luid='0x00000000_0x00000000',
    s=>s.rows[0].statuses=[0,2,0],s=>s.rows[0].statuses=[0],s=>s.rows[0].statuses=['0',0,0],s=>s.rows[0].dedicatedBytes='unknown',s=>s.rows[0].sharedBytes=-1,s=>s.rows[0].totalCommittedBytes=NaN,
    s=>s.rows[0].totalCommittedBytes=Number.MAX_SAFE_INTEGER];
  for(const mutate of mutants){const capture=new OwnedProcessMemoryCapture(identity),sample=samples()();mutate(sample);assert.throws(()=>capture.sample(sample),/owned memory/);}
  const capture=new OwnedProcessMemoryCapture(identity),sample=samples();capture.sample(sample());const changed=sample();changed.expectedBefore=[0];changed.expectedAfter=[0];changed.rows.pop();assert.throws(()=>capture.sample(changed),/nodes changed during capture/);
});
test('PID reuse, executable substitution, wrong LUID, reordered samples and missing intervals fail',()=>{
  for(const mutate of [s=>s.pid++,s=>s.startedAt='2026-10-23T00:00:01.000Z',s=>s.executable='D:\\wrong.exe',s=>s.luid='0x00000000_0x00000000',s=>s.sequence++,s=>s.elapsedMs+=1501,s=>s.readStartedMs-=251,s=>s.startedAt='2026-10-23T00:00:00.0000001Z']){
    const capture=new OwnedProcessMemoryCapture(identity),sample=samples();capture.sample(sample());const changed=sample();mutate(changed);assert.throws(()=>capture.sample(changed),/owned memory/);
  }
});
test('synchronized endpoints, all native phase intervals and unique phase boundaries are mandatory',()=>{
  const capture=new OwnedProcessMemoryCapture(identity),sample=samples();capture.sample(sample());assert.throws(()=>capture.finish(),/missing load/);
  capture.boundary('load',sample());assert.throws(()=>capture.finish(),/terminal endpoint missing/);assert.throws(()=>capture.boundary('load',null),/repeated phase/);assert.throws(()=>capture.boundary('imaginary',null),/unknown/);
  capture.boundary(null,sample());assert.throws(()=>capture.finish(),/missing conditioning/);
});
test('no shared-memory accounting subtraction or physical-spill inference is made',()=>{
  const result=complete();assert(result.overall.sharedPeakBytes>0);assert.equal(result.overall.dedicatedPeakBytes,300);assert.match(result.semantics,/not physical residency/);assert.match(result.attribution,/cross-process-shared/);
});
test('expired startup deadlines launch zero probes and never install a fresh timeout budget',()=>{
  let probes=0,timers=0;
  for(const runDeadline of [0,1000,999,61001])assert.throws(()=>createFlux3StartupProbeBudget({timeoutSeconds:60,runDeadline},{now:()=>1000,execute:()=>{probes++;},schedule:()=>{timers++;}}),/startup deadline/);
  assert.equal(probes,0);assert.equal(timers,0);
});
test('startup expiry aborts owned probes and awaits child close before failure cleanup completes',async()=>{
  let clock=1000,expire,signal,probes=0;const child=new EventEmitter();
  const budget=createFlux3StartupProbeBudget({timeoutSeconds:60,runDeadline:61000},{now:()=>clock,schedule:(callback,ms)=>{assert.equal(ms,60000);expire=callback;return 1;},cancel:()=>{},
    execute:(_command,_args,options)=>{probes++;signal=options.signal;const operation=new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(Error('DECLARED abort'),{name:'AbortError'})),{once:true}));operation.child=child;return operation;}});
  const operation=budget.exec('DECLARED-NOT-EXECUTED',[]);const rejected=assert.rejects(operation,{name:'AbortError'});
  clock=61000;expire();assert.equal(signal.aborted,true);
  let cleaned=false;const cleanup=budget.close().then(()=>{cleaned=true;});await Promise.resolve();await Promise.resolve();assert.equal(cleaned,false,'AbortError alone is not owned child cleanup');
  child.emit('close');await rejected;await cleanup;assert.equal(cleaned,true);
  await assert.rejects(budget.exec('DECLARED-NOT-EXECUTED',[]),/startup deadline/);assert.equal(probes,1);
});
test('a late successful startup callback cannot produce READY after the original deadline',async()=>{
  let clock=1000,expire,complete;
  const budget=createFlux3StartupProbeBudget({timeoutSeconds:60,runDeadline:61000},{now:()=>clock,schedule:callback=>{expire=callback;return 1;},cancel:()=>{},execute:()=>new Promise(resolve=>{complete=resolve;})});
  const operation=budget.exec('DECLARED-NOT-EXECUTED',[]);clock=61000;expire();complete({stdout:'DECLARED LATE SUCCESS'});await assert.rejects(operation,/startup deadline/);await budget.close();
});
test('all startup probes share the original deadline and cancellation signal',async()=>{
  let clock=1000;const options=[],timers=[];
  const budget=createFlux3StartupProbeBudget({timeoutSeconds:60,runDeadline:61000},{now:()=>clock,schedule:(callback,ms)=>{timers.push(ms);return 1;},cancel:()=>{},execute:async(_command,_args,value)=>{options.push(value);return {stdout:'DECLARED CPU'};}});
  await budget.exec('DECLARED-NOT-EXECUTED',[],{timeout:30000});clock=51000;await budget.exec('DECLARED-NOT-EXECUTED',[],{timeout:30000});
  assert.deepEqual(options.map(value=>value.timeout),[30000,10000]);assert.equal(options[0].signal,options[1].signal);assert.deepEqual(timers,[60000]);await budget.close();assert.equal(options[0].signal.aborted,true);
});
test('ongoing foreign-allocation hardware guards never overlap and stop aborts only owned pending probes',async()=>{
  let tick,calls=0,release,signal,cancelled=0;const failures=[],records=[];
  const stop=startFlux3OwnedHardwareMonitor({runDeadline:61000},12345,{now:()=>1000,schedule:(callback,ms)=>{assert.equal(ms,2000);tick=callback;return 7;},cancel:id=>{assert.equal(id,7);cancelled++;},
    writeRecord:record=>records.push(record),onFailure:error=>failures.push(error),execute:async(_command,_args,options)=>{signal=options.signal;assert(options.timeout<=60000);},
    check:async(_admission,pids,{exec})=>{assert.deepEqual(pids,[12345]);calls++;await exec('DECLARED-NOT-EXECUTED',[],{timeout:30000});await new Promise(resolve=>{release=resolve;});}});
  const first=tick();await Promise.resolve();await tick();assert.equal(calls,1,'slow hardware guard cannot overlap next scheduled tick');
  stop();assert.equal(signal.aborted,true);assert.equal(cancelled,1);release();await first;await tick();assert.equal(calls,1);assert.deepEqual(failures,[]);assert.equal(records.at(-1).completedChecks,0);assert.equal(records.at(-1).periodicCoverageObserved,false);assert.equal(records.at(-1).pendingCheckAborted,true);
});
test('periodic guard logs retain only safe measured numeric snapshots and honest completion summary',async()=>{
  let tick;const records=[],failures=[];
  const stop=startFlux3OwnedHardwareMonitor({runDeadline:5000},12345,{now:()=>1000,schedule:callback=>{tick=callback;return 1;},cancel:()=>{},writeRecord:record=>records.push(record),onFailure:error=>failures.push(error),
    check:async()=>({physicalGpuId:identity.physicalGpuId,gpuUuid:identity.gpuUuid,gpuHeadroom:{totalBytes:1000,freeBytes:900},foreignGraphics:{totalBytes:50,maxProcessBytes:30,processes:2},unretainedSecret:'DECLARED-NOT-TO-LOG'})});
  await tick();stop();assert.deepEqual(failures,[]);assert.equal(records.length,2);assert.equal(records[0].foreignDedicatedBytes,50);assert.equal(records[0].unretainedSecret,undefined);assert.equal(records[1].completedChecks,1);assert.equal(records[1].periodicCoverageObserved,true);assert.equal(records[1].pendingCheckAborted,false);assert.equal(records[1].physicalResidencyClaim,false);
});
test('foreign guard cap/identity faults and expired deadlines latch stop without retries',async()=>{
  for(const expired of [false,true]) {let tick,calls=0,cancelled=0;const failures=[];
    startFlux3OwnedHardwareMonitor({runDeadline:2000},12345,{now:()=>expired?2000:1000,schedule:callback=>{tick=callback;return 1;},cancel:()=>cancelled++,writeRecord:()=>{},onFailure:error=>failures.push(error.message),
      check:async()=>{calls++;throw Error('DECLARED foreign allocation cap/start identity fault');}});
    await tick();await tick();assert.equal(calls,expired?0:1);assert.equal(cancelled,1);assert.equal(failures.length,1);assert.match(failures[0],expired?/deadline/:/foreign allocation/);
  }
});
test('successful finalization awaits an unresolved foreign guard and rejects terminal cap/unknown results',async()=>{
  const proof={physicalGpuId:identity.physicalGpuId,gpuUuid:identity.gpuUuid,gpuHeadroom:{totalBytes:1000,freeBytes:900},foreignGraphics:{totalBytes:50,maxProcessBytes:30,processes:2}};
  for(const result of ['valid','missing','capfault']) {
    let tick,release,reject,done=false;const records=[],failures=[];
    const stop=startFlux3OwnedHardwareMonitor({runDeadline:10000},12345,{now:()=>1000,schedule:callback=>{tick=callback;return 1;},cancel:()=>{},writeRecord:record=>records.push(record),onFailure:error=>failures.push(error),
      check:()=>new Promise((resolve,refuse)=>{release=resolve;reject=refuse;})});
    const checking=tick();const finishing=stop.finish().then(()=>{done=true;});await Promise.resolve();assert.equal(done,false);assert.equal(records.length,0,'unresolved guard cannot be aborted into a successful summary');
    if(result==='capfault')reject(Error('DECLARED final foreign cap fault'));else release(result==='valid'?proof:undefined);
    await checking;
    if(result==='valid'){await finishing;assert.equal(done,true);assert.equal(records.at(-1).pendingCheckAborted,false);assert.equal(records.at(-1).completedChecks,1);assert.deepEqual(failures,[]);}
    else {await assert.rejects(finishing,/cap fault|numeric observation/);assert.equal(done,false);assert.equal(records.at(-1).faulted,true);assert.equal(failures.length,1);}
  }
});
test('terminal boundary makes one fresh non-retry guard when stale, while short capture retains honest initial-only scope',async()=>{
  for(const elapsed of [500,2500]) {let clock=1000,calls=0;const records=[];
    const stop=startFlux3OwnedHardwareMonitor({runDeadline:10000},12345,{now:()=>clock,schedule:()=>1,cancel:()=>{},writeRecord:record=>records.push(record),onFailure:error=>{throw error;},
      check:async()=>{calls++;return {physicalGpuId:identity.physicalGpuId,gpuUuid:identity.gpuUuid,gpuHeadroom:{totalBytes:1000,freeBytes:900},foreignGraphics:{totalBytes:50,maxProcessBytes:30,processes:2}};}});
    clock+=elapsed;await stop.finish();assert.equal(calls,elapsed>=2000?1:0);assert.equal(records.at(-1).periodicCoverageObserved,elapsed>=2000);assert.equal(records.at(-1).pendingCheckAborted,false);
  }
});
test('guard failures never serialize raw captions, tokens, process records, arguments or exception stacks',async()=>{
  const sentinel='DECLARED_PRIVATE_CAPTION_TOKEN_COMMANDLINE_SENTINEL',records=[];let tick;
  const error=Error(sentinel);error.name=sentinel;error.stack=sentinel;error.actual={commandLine:sentinel,caption:sentinel};
  startFlux3OwnedHardwareMonitor({runDeadline:10000},12345,{now:()=>1000,schedule:callback=>{tick=callback;return 1;},cancel:()=>{},writeRecord:record=>records.push(record),
    onFailure:cause=>records.push(ownedMemoryFailure(cause,'hardware_monitor')),check:async()=>{throw error;}});
  await tick();const output=JSON.stringify(records);assert(!output.includes(sentinel));assert.equal(records.at(-1).errorClass,'Error');assert.equal(records.at(-1).failureCode,'FLUX3_OWNED_MEMORY_CAPTURE_REFUSED');
  assert(!JSON.stringify(ownedMemoryFailure(error,sentinel)).includes(sentinel));
});
test('Windows native integration establishes context before READY/load and never aliases accounting to old metrics',async()=>{
  const native=await readFile(new URL('../crates/sceneworks-memory-adapter/src/bin/candle_flux3.rs',import.meta.url),'utf8');
  const owned=native.slice(native.indexOf('fn run_owned_process'));
  assert(owned.indexOf('default_device()')<owned.indexOf('OwnedProcessMemory::start()'));
  assert(owned.indexOf('OwnedProcessMemory::start()')<owned.indexOf('.load(plan.checkpoint.provider'));
  assert(owned.includes('render_monitored'));assert(owned.includes('check_health'));
  assert(!/cuda_phase_metrics|DevicePeakDelta|certifying_vram_probe/.test(owned.split('#[cfg(test)]')[0]));
  const common=await readFile(new URL('../crates/sceneworks-memory-adapter/src/bin/flux3_common.rs',import.meta.url),'utf8');
  assert(common.includes('observedOperatingConditions'));assert(common.includes('measuredStrategySha256'));assert(common.includes('object.remove("observedMemory")'));
});
test('collector script has one persistent process with complete independent metadata and process identity bracketing',()=>{
  const script=ownedProcessCounterScript(identity);for(const expected of ["'Dedicated Usage'","'Shared Usage'","'Total Committed'",'GetInstanceNames','ConcurrentQueue<string>','StartTime','MainModule.FileName','PdhGetRawCounterValue','CStatus','Get-Nodes','Read-Sample'])assert(script.includes(expected));
  assert(script.includes('type != 0x00010100'));assert(script.includes('CStatus>1'));assert(!/Start-Process|nvidia-smi|memory\.used|Get-CimInstance/.test(script));assert.throws(()=>ownedProcessCounterScript({...identity,pid:0}));assert.throws(()=>ownedProcessCounterScript({...identity,luid:'unknown'}));
});
