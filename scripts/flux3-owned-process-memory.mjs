// Private FLUX3 resource-adapter measurement transport. Never launches a render.
import assert from 'node:assert/strict';
import {spawn,execFile as execCb} from 'node:child_process';
import {promisify} from 'node:util';
import {readFile} from 'node:fs/promises';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {admittedCandleDevice,guardHardware,observeProcessTable} from './flux3-image-klein-terminal-session.mjs';
import {OwnedProcessMemoryCapture,ownedProcessCounterScript,OWNED_MEMORY_MAX_GAP_MS} from './lib/flux3-owned-process-memory.mjs';
export function ownedMemoryFailure(error,phase='capture') {
  const phases=['startup','capture','hardware_monitor','counter_sampler','terminal_guard'];
  return {ok:false,error:'BLOCKED: owned-process memory capture refused',failureCode:'FLUX3_OWNED_MEMORY_CAPTURE_REFUSED',
    phase:phases.includes(phase)?phase:'capture',errorClass:['Error','AssertionError','AbortError','TypeError','SyntaxError'].includes(error?.name)?error.name:'Error'};
}
// Startup has the SAME fixed deadline as capture. Abortable execFile promises can
// reject before child close, so cleanup explicitly waits for every owned close.
export function createFlux3StartupProbeBudget(admission,{execute=promisify(execCb),now=Date.now,schedule=setTimeout,cancel=clearTimeout}={}) {
  const deadline=admission.runDeadline,started=now();
  assert(Number.isSafeInteger(admission.timeoutSeconds)&&admission.timeoutSeconds>=60&&admission.timeoutSeconds<=10800,'invalid fixed startup timeout');
  assert(Number.isSafeInteger(deadline)&&deadline>started&&deadline-started<=admission.timeoutSeconds*1000,'expired or extended fixed startup deadline');
  const abort=new AbortController(),pending=new Set();let closed=false;
  const check=()=>{assert(!closed&&!abort.signal.aborted&&now()<deadline,'fixed startup deadline expired or startup closed');};
  const timer=schedule(()=>abort.abort(),deadline-started);
  const exec=async(command,args,options={})=>{
    check();
    const operation=execute(command,args,{...options,timeout:Math.min(options.timeout??30000,deadline-now()),signal:abort.signal});
    const child=operation?.child;
    const cleanup=child?new Promise(resolve=>child.once('close',resolve)):Promise.resolve(operation).then(()=>{},()=>{});
    const tracked=(async()=>{try {const result=await operation;check();return result;} finally {await cleanup;}})();
    pending.add(tracked);tracked.then(()=>pending.delete(tracked),()=>pending.delete(tracked));
    return tracked;
  };
  return {exec,check,close:async()=>{closed=true;cancel(timer);abort.abort();while(pending.size)await Promise.allSettled([...pending]);}};
}
// This monitor is specific to this owned FLUX3 adapter. Expensive hardware guards
// never overlap each other or replace the independent 200ms process counter sampler.
export function startFlux3OwnedHardwareMonitor(admission,pid,{check=guardHardware,execute=promisify(execCb),onFailure,schedule=setInterval,cancel=clearInterval,now=Date.now,writeRecord=record=>process.stderr.write(JSON.stringify(record)+'\n')}={}) {
  assert(typeof onFailure==='function' && Number.isSafeInteger(admission.runDeadline),'bounded owned hardware monitor required');
  const abort=new AbortController(),startedMs=now();let stopped=false,running=false,finishing=false,checks=0,firstObservedAt=null,lastObservedAt=null,lastCompletedMs=null,faulted=false,failure=null,inFlight=null;
  const stop=()=>{if(!stopped){stopped=true;cancel(timer);abort.abort();writeRecord({schemaVersion:1,kind:'flux3-owned-resource-hardware-monitor',event:'summary',intervalMs:2000,completedChecks:checks,firstObservedAt,lastObservedAt,periodicCoverageObserved:checks>0,pendingCheckAborted:running,faulted,exclusiveThroughputClaim:false,physicalResidencyClaim:false});}};
  const performCheck=()=>{
    if(stopped||running)return inFlight;
    running=true;
    inFlight=(async()=>{try {
      assert(now()<admission.runDeadline,'owned hardware monitor deadline exceeded');
      const proof=await check(admission,[pid],{exec:(command,args,options={})=>{
        const remaining=admission.runDeadline-now();assert(remaining>0,'owned hardware monitor fixed deadline expired');
        return execute(command,args,{...options,timeout:Math.min(options.timeout??30000,remaining),signal:abort.signal});
      }});
      if(stopped)return;
      assert(now()<admission.runDeadline,'owned hardware monitor deadline exceeded');
      const values={totalDeviceBytes:proof?.gpuHeadroom?.totalBytes,freeDeviceBytes:proof?.gpuHeadroom?.freeBytes,
        foreignDedicatedBytes:proof?.foreignGraphics?.totalBytes,foreignMaxProcessBytes:proof?.foreignGraphics?.maxProcessBytes,foreignProcessCount:proof?.foreignGraphics?.processes};
      assert(Object.values(values).every(value=>Number.isSafeInteger(value)&&value>=0),'owned hardware monitor numeric observation missing');
      assert(['0','1'].includes(proof.physicalGpuId)&&/^GPU-[0-9a-f-]+$/i.test(proof.gpuUuid),'owned hardware monitor physical identity missing');
      lastCompletedMs=now();lastObservedAt=new Date(lastCompletedMs).toISOString();firstObservedAt??=lastObservedAt;checks++;
      writeRecord({schemaVersion:1,kind:'flux3-owned-resource-hardware-monitor',event:'sample',check:checks,observedAt:lastObservedAt,
        physicalGpuId:proof.physicalGpuId,gpuUuid:proof.gpuUuid,...values,exclusiveThroughputClaim:false,physicalResidencyClaim:false});
    }catch(error){if(!stopped){running=false;failure=error;faulted=true;stop();onFailure(error);}}
    finally{running=false;}})();
    return inFlight;
  };
  const timer=schedule(()=>finishing||running||stopped?undefined:performCheck(),2000);
  stop.finish=async()=>{
    assert(!finishing&&!stopped,'owned hardware monitor cannot finalize a stopped/faulted capture');
    finishing=true;cancel(timer);
    if(inFlight)await inFlight;
    if(failure)throw failure;
    assert(!stopped&&now()<admission.runDeadline,'owned hardware monitor final deadline/stopped state');
    // One normal terminal checkpoint when no recent periodic result exists. This
    // is not a retry and never substitutes an unresolved/aborted guard with success.
    if(lastCompletedMs===null?now()-startedMs>=2000:now()-lastCompletedMs>=2000)await performCheck();
    if(failure)throw failure;
    assert(!stopped&&now()<admission.runDeadline,'owned hardware monitor final deadline/stopped state');
    stop();
  };
  return stop;
}
export async function runOwnedProcessMemoryCollector() {
  assert.equal(process.platform,'win32','FLUX3 owned-process WDDM accounting is Windows-only');
  const pid=Number(process.argv[2]);assert(Number.isSafeInteger(pid)&&pid>0,'explicit owned adapter PID required');
  const invocation=JSON.parse(process.env.SCENEWORKS_FLUX3_INVOCATION??'null');
  assert(invocation?.admissionFile,'native admitted invocation required');
  const admission=JSON.parse(await readFile(invocation.admissionFile,'utf8'));
  assert(admission.backend==='candle' && admission.actualRun===true,'actual Candle resource admission required');
  const startup=createFlux3StartupProbeBudget(admission);
  let device,proof,parent,nativeIdentity;
  try {
    device=admittedCandleDevice(admission);
    proof=await guardHardware(admission,[pid],{exec:startup.exec});
    assert(proof.physicalGpuId===device.physicalGpuId && proof.gpuUuid.toLowerCase()===device.gpuUuid.toLowerCase(),'owned sampler approved device mismatch');
    const table=await observeProcessTable(startup.exec,'win32'),self=table.find(p=>p.pid===process.pid);parent=table.find(p=>p.pid===pid);
    assert(parent && self?.ppid===pid && parent.executable===admission.binary,'sampler must be immediate child of sealed owned adapter');
    nativeIdentity=JSON.parse((await startup.exec('powershell.exe',['-NoProfile','-NonInteractive','-Command',`$ErrorActionPreference='Stop';$p=[Diagnostics.Process]::GetProcessById(${pid});@{pid=$p.Id;startedAt=$p.StartTime.ToUniversalTime().ToString('o');executable=$p.MainModule.FileName}|ConvertTo-Json -Compress`],{timeout:5000})).stdout);
    assert(nativeIdentity.pid===pid && nativeIdentity.executable===parent.executable && Date.parse(nativeIdentity.startedAt)===Date.parse(parent.startedAt),'native owned process start identity differs from guarded parent');
    startup.check();
  } finally {await startup.close();}
  const identity={pid,startedAt:nativeIdentity.startedAt,executable:nativeIdentity.executable,luid:proof.foreignGraphics.luid,
    physicalGpuId:device.physicalGpuId,gpuUuid:device.gpuUuid,pciBusId:proof.pciBusId,logicalCudaOrdinal:0};
  const capture=new OwnedProcessMemoryCapture(identity);
  const deadline=admission.runDeadline;
  assert(Number.isSafeInteger(deadline)&&deadline>Date.now(),'owned sampler fixed deadline required');
  const child=spawn('powershell.exe',['-NoProfile','-NonInteractive','-Command',ownedProcessCounterScript(identity)],{windowsHide:true,stdio:['pipe','pipe','pipe'],env:{...process.env}});
  let closed=false,ready=false,finishing=false,pending=null,lastSampleAt=Date.now(),stderr='',stopHardware=()=>{};
  const send=value=>process.stdout.write(JSON.stringify(value)+'\n');
  const stop=()=>{if(!closed){closed=true;clearInterval(watchdog);stopHardware();child.kill();}};
  const fail=(error,phase='capture')=>{if(!closed){send(ownedMemoryFailure(error,phase));stop();process.exitCode=1;process.stdin.destroy();}};
  const watchdog=setInterval(()=>{const now=Date.now();if(now>=deadline)fail(Error('fixed deadline exceeded'));else if(!finishing&&now-lastSampleAt>(ready?OWNED_MEMORY_MAX_GAP_MS:15000))fail(Error('collector sample/READY coverage timeout'));},100);
  stopHardware=startFlux3OwnedHardwareMonitor(admission,pid,{onFailure:error=>fail(error,'hardware_monitor')});
  child.once('error',fail);child.stderr.setEncoding('utf8');child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-8192);});
  child.stdin.on('error',fail);
  child.once('close',code=>{if(!closed&&!finishing)fail(Error(`counter collector exited ${code}: ${stderr}`));});
  const lines=createInterface({input:child.stdout});
  lines.on('line',async line=>{try {
    if(closed||finishing)return;
    assert(line.length<=1024*1024,'owned sample output bound exceeded');
    const {sample,boundary}=JSON.parse(line);lastSampleAt=Date.now();
    if(boundary) {
      assert(ready&&pending&&boundary.id===pending.id&&boundary.phase===pending.phase,'unexpected/reordered owned phase boundary');
      capture.boundary(boundary.phase,sample);
      const result=boundary.phase===null?capture.finish():null;
      if(result) {
        finishing=true;child.kill(); // own phase endpoints complete; no invented samples during final guard.
        await stopHardware.finish(); // never turn an unresolved foreign check into successful evidence.
        assert(!closed,'owned capture failed during terminal hardware guard');
        stop();
        await new Promise((resolve,reject)=>process.stderr.write('',error=>error?reject(error):resolve()));
      }
      send({ok:true,id:boundary.id,evidence:result});pending=null;
      if(result) process.stdin.destroy();
    } else {
      capture.sample(sample);
      if(!ready){ready=true;send({ok:true,ready:true,identity,runDeadline:deadline});}
    }
  }catch(error){fail(error,finishing?'terminal_guard':'counter_sampler');}});
  const commands=createInterface({input:process.stdin});
  commands.on('line',line=>{try {
    assert(line.length<4096,'owned command bound exceeded');
    const command=JSON.parse(line);
    assert(ready&&!pending&&command.op==='boundary'&&Number.isSafeInteger(command.id)&&command.id>0,'invalid/unready owned phase command');
    pending=command;child.stdin.write(JSON.stringify(command)+'\n');
  }catch(error){fail(error);}});
  commands.once('close',()=>{if(!closed)fail(Error('native controller closed without terminal endpoint'));});
  process.once('exit',stop);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)runOwnedProcessMemoryCollector().catch(error=>{process.stdout.write(JSON.stringify(ownedMemoryFailure(error,'startup'))+'\n');process.exitCode=1;});
