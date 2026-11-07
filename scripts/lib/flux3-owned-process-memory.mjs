// FLUX3-only sampled WDDM PROCESS accounting. Never device residency, CUDA allocator
// activity, an idle-device delta, or a canonical calibration metric.
import assert from 'node:assert/strict';
export const OWNED_MEMORY_INTERVAL_MS = 200;
export const OWNED_MEMORY_MAX_GAP_MS = 1500;
const integer = (v, name) => { assert(Number.isSafeInteger(v) && v >= 0, `invalid owned memory ${name}`); return v; };
const sortedNodes = values => [...values].sort((a,b)=>a-b);
export class OwnedProcessMemoryCapture {
  constructor(identity, { intervalMs=OWNED_MEMORY_INTERVAL_MS, maxGapMs=OWNED_MEMORY_MAX_GAP_MS }={}) {
    assert(Number.isSafeInteger(identity.pid) && identity.pid>0 && /^0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i.test(identity.luid));
    assert(typeof identity.startedAt==='string' && Number.isFinite(Date.parse(identity.startedAt)) && typeof identity.executable==='string' && identity.executable.length);
    assert(['0','1'].includes(identity.physicalGpuId) && /^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(identity.gpuUuid) && /^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$/i.test(identity.pciBusId) && identity.logicalCudaOrdinal===0,'owned memory complete approved device identity required');
    assert(Number.isSafeInteger(intervalMs) && intervalMs>=50 && intervalMs<=250 && Number.isSafeInteger(maxGapMs) && maxGapMs>=intervalMs && maxGapMs<=1500);
    this.identity={...identity,luid:identity.luid.toLowerCase()};this.intervalMs=intervalMs;this.maxGapMs=maxGapMs;
    this.phases={};this.current=null;this.last=null;this.finished=false;this.nodes=null;this.sequence=0;this.maxObservedReadSkewMs=0;
    this.overall=this.empty();
  }
  empty() { return {dedicatedPeakBytes:0,sharedPeakBytes:0,totalCommittedPeakBytes:0,samples:0,firstSampleMs:null,lastSampleMs:null,maxGapMs:0}; }
  add(phase,sample,totals) {
    if(phase.lastSampleMs!==null) phase.maxGapMs=Math.max(phase.maxGapMs,sample.elapsedMs-phase.lastSampleMs);
    phase.firstSampleMs??=sample.elapsedMs;phase.lastSampleMs=sample.elapsedMs;phase.samples++;
    for(const [key,value] of Object.entries(totals)) phase[key]=Math.max(phase[key],value);
  }
  sample(value) {
    assert(!this.finished,'owned memory capture already finished');
    assert(value.pid===this.identity.pid && value.startedAt===this.identity.startedAt && typeof value.executable==='string' && value.executable.toLowerCase()===this.identity.executable.toLowerCase(),'owned memory process identity changed/reused');
    assert(value.luid===this.identity.luid,'owned memory wrong adapter LUID');
    integer(value.elapsedMs,'sample timestamp');integer(value.readStartedMs,'read start');
    assert(value.elapsedMs>=value.readStartedMs && value.elapsedMs-value.readStartedMs<=250,'owned memory sample skew exceeded');
    this.maxObservedReadSkewMs=Math.max(this.maxObservedReadSkewMs,value.elapsedMs-value.readStartedMs);
    assert(value.sequence===++this.sequence,'owned memory sample sequence missing/reordered');
    if(this.last) { const gap=value.elapsedMs-this.last.elapsedMs;assert(gap>0 && gap<=this.maxGapMs,'owned memory sample interval missing/excessive'); }
    assert(Array.isArray(value.expectedBefore) && Array.isArray(value.expectedAfter) && Array.isArray(value.rows) && value.rows.length>0,'owned memory complete node metadata required');
    const check=nodes=>{ assert(nodes.length>0 && nodes.every(n=>Number.isSafeInteger(n)&&n>=0&&n<=31) && new Set(nodes).size===nodes.length,'owned memory invalid/duplicate physical nodes');return sortedNodes(nodes); };
    const nodes=check(value.expectedBefore);assert.deepEqual(check(value.expectedAfter),nodes,'owned memory physical nodes changed during sample');
    if(this.nodes) assert.deepEqual(nodes,this.nodes,'owned memory physical nodes changed during capture'); else this.nodes=nodes;
    const seen=new Set(),totals={dedicatedPeakBytes:0,sharedPeakBytes:0,totalCommittedPeakBytes:0};
    for(const row of value.rows) {
      assert(row.pid===this.identity.pid && row.luid===this.identity.luid && nodes.includes(row.physicalNode) && !seen.has(row.physicalNode),'owned memory wrong/missing/duplicate node identity');seen.add(row.physicalNode);
      assert(Array.isArray(row.statuses) && row.statuses.length===3 && row.statuses.every(status=>status===0||status===1),'owned memory invalid counter status');
      for(const [metric,key] of [['dedicatedBytes','dedicatedPeakBytes'],['sharedBytes','sharedPeakBytes'],['totalCommittedBytes','totalCommittedPeakBytes']]) {
        totals[key]+=integer(row[metric],metric);integer(totals[key],`aggregate ${metric}`);
      }
    }
    assert.equal(seen.size,nodes.length,'owned memory missing physical node counter');
    // Counters are retained independently: shared accounting is NOT subtracted, treated
    // as VRAM, or silently interpreted as a reliable physical-spill detector.
    this.add(this.overall,value,totals);if(this.current) this.add(this.phases[this.current],value,totals);
    this.last={...value,totals};return totals;
  }
  boundary(phase,sample) {
    assert(phase===null || ['load','promptUpsampling','conditioning','denoise','decode'].includes(phase),'unknown owned memory phase');
    if(this.current) assert(phase===null || !Object.hasOwn(this.phases,phase),'owned memory repeated phase');
    this.sample(sample); // synchronized endpoint belongs to the ending AND starting windows.
    if(this.current) assert(this.phases[this.current].samples>=2 && this.phases[this.current].lastSampleMs>this.phases[this.current].firstSampleMs,'owned memory phase lacks interval coverage');
    this.current=phase;
    if(phase!==null) { assert(!Object.hasOwn(this.phases,phase),'owned memory repeated phase');this.phases[phase]=this.empty();this.add(this.phases[phase],sample,this.last.totals); }
  }
  finish() {
    assert(this.current===null && !this.finished,'owned memory terminal endpoint missing');
    for(const phase of ['load','conditioning','denoise','decode']) assert(this.phases[phase]?.samples>=2,`owned memory missing ${phase} interval`);
    this.finished=true;
    return {schemaVersion:1,kind:'flux3_owned_process_memory_accounting_v1',source:'Windows GPU Process Memory performance counters',
      semantics:'sampled per-process dedicated/shared/total-committed accounting; not physical residency or CUDA active/reserved allocator bytes',
      attribution:'own PID/start-time/executable and CUDA adapter LUID; includes any cross-process-shared allocations charged to this process',
      physicalResidencyClaim:false,allocatorActivityClaim:false,deviceTotalDeltaAttributionClaim:false,exclusiveThroughputClaim:false,
      unavailableMetrics:{physicalResidentBytes:null,activeBytes:null,allocatorBytes:null},
      sampling:{intervalMs:this.intervalMs,maxAllowedGapMs:this.maxGapMs,maxAllowedReadSkewMs:250,maxObservedReadSkewMs:this.maxObservedReadSkewMs,losslessPeakClaim:false},
      identity:this.identity,physicalNodes:this.nodes,phases:this.phases,overall:this.overall};
  }
}

// One persistent read-only counter process: no per-sample process launches. Independent
// instance discovery brackets every sample; every selected-node counter must exist.
export function ownedProcessCounterScript({pid,luid,intervalMs=OWNED_MEMORY_INTERVAL_MS}) {
  assert(Number.isSafeInteger(pid)&&pid>0 && /^0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i.test(luid));
  assert(Number.isSafeInteger(intervalMs)&&intervalMs>=50&&intervalMs<=250);
  return `$ErrorActionPreference='Stop'; $ownedPid=${pid}; $luid='${luid.toLowerCase()}'; $sequence=0;
$clock=[Diagnostics.Stopwatch]::StartNew(); $category=New-Object Diagnostics.PerformanceCounterCategory('GPU Process Memory');
Add-Type -TypeDefinition @'
using System;
using System.Threading;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
public static class Flux3CounterCommands {
 public static ConcurrentQueue<string> Lines = new ConcurrentQueue<string>(); public static volatile bool Ended;
 public static void Start() { var thread = new Thread(() => { string line; while ((line = Console.In.ReadLine()) != null) Lines.Enqueue(line); Ended = true; }); thread.IsBackground = true; thread.Start(); }
}
public static class Flux3PdhCounters {
 [StructLayout(LayoutKind.Sequential)] public struct RawCounter { public uint CStatus; public System.Runtime.InteropServices.ComTypes.FILETIME TimeStamp; public long FirstValue; public long SecondValue; public uint MultiCount; }
 [DllImport("pdh.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] static extern uint PdhOpenQueryW(string source, UIntPtr data, out IntPtr query);
 [DllImport("pdh.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] static extern uint PdhAddEnglishCounterW(IntPtr query, string path, UIntPtr data, out IntPtr counter);
 [DllImport("pdh.dll")] static extern uint PdhCollectQueryData(IntPtr query);
 [DllImport("pdh.dll")] static extern uint PdhGetRawCounterValue(IntPtr counter, out uint type, out RawCounter value);
 [DllImport("pdh.dll")] static extern uint PdhCloseQuery(IntPtr query);
 static IntPtr query; static IntPtr[] counters;
 static void Check(uint status) { if(status != 0) throw new InvalidOperationException("PDH API status 0x"+status.ToString("x8")); }
 public static void Open(string[] paths) { Check(PdhOpenQueryW(null, UIntPtr.Zero, out query)); counters=new IntPtr[paths.Length]; for(int i=0;i<paths.Length;i++) Check(PdhAddEnglishCounterW(query,paths[i],UIntPtr.Zero,out counters[i])); }
 public static RawCounter[] Sample() { Check(PdhCollectQueryData(query)); var values=new RawCounter[counters.Length]; for(int i=0;i<counters.Length;i++) { uint type; Check(PdhGetRawCounterValue(counters[i],out type,out values[i])); if(type != 0x00010100 || values[i].CStatus>1 || values[i].FirstValue<0) throw new InvalidOperationException("PDH invalid raw byte counter type/CStatus/value"); } return values; }
 public static void Close() { if(query!=IntPtr.Zero) { PdhCloseQuery(query);query=IntPtr.Zero; } }
}
'@;
[Flux3CounterCommands]::Start();
function Get-Nodes {
  param([switch]$AllowMissing)
  $nodes=@(); foreach($instance in $category.GetInstanceNames()) {
    if($instance -like ('pid_'+$ownedPid+'_luid_'+$luid+'_*')) {
      if($instance -notmatch ('^pid_'+$ownedPid+'_luid_'+$luid+'_phys_([0-9]+)$')) {throw 'malformed owned counter instance'};
      $nodes+= [int]$Matches[1]
    }
  }; if($nodes.Count -eq 0) {if($AllowMissing) {return};throw 'owned physical node metadata absent'};
  if(@($nodes|Select-Object -Unique).Count -ne $nodes.Count) {throw 'owned physical node metadata duplicate'};
  @($nodes|Sort-Object)
}
$initialNodes=@(Get-Nodes -AllowMissing);
while($initialNodes.Count -eq 0 -and $clock.ElapsedMilliseconds -lt 10000) {
  Start-Sleep -Milliseconds ${intervalMs};$initialNodes=@(Get-Nodes -AllowMissing)
};
if($initialNodes.Count -eq 0) {throw 'owned CUDA context counters unavailable before load'};
$counterPaths=@();
foreach($node in $initialNodes) {
  foreach($counterName in @('Dedicated Usage','Shared Usage','Total Committed')) {
    $counterPaths+= '\\GPU Process Memory(pid_'+$ownedPid+'_luid_'+$luid+'_phys_'+$node+')\\'+$counterName
  }
};
[Flux3PdhCounters]::Open([string[]]$counterPaths);
function Read-Sample {
  $readStarted=[long]$clock.ElapsedMilliseconds;
  $process=[Diagnostics.Process]::GetProcessById($ownedPid); $started=$process.StartTime.ToUniversalTime().ToString('o'); $exe=$process.MainModule.FileName;
  $before=@(Get-Nodes);$rows=@();$values=[Flux3PdhCounters]::Sample();$counterIndex=0;
  foreach($node in $initialNodes) {
    $dedicated=$values[$counterIndex];$shared=$values[$counterIndex+1];$committed=$values[$counterIndex+2];$counterIndex+=3;
    $rows+= @{pid=$ownedPid;luid=$luid;physicalNode=$node;statuses=@([long]$dedicated.CStatus,[long]$shared.CStatus,[long]$committed.CStatus);dedicatedBytes=$dedicated.FirstValue;sharedBytes=$shared.FirstValue;totalCommittedBytes=$committed.FirstValue}
  };
  $after=@(Get-Nodes); $again=[Diagnostics.Process]::GetProcessById($ownedPid);
  if($again.StartTime.ToUniversalTime().ToString('o') -cne $started -or $again.MainModule.FileName -cne $exe) {throw 'owned process identity changed'};
  $script:sequence++; @{pid=$ownedPid;startedAt=$started;executable=$exe;luid=$luid;sequence=$script:sequence;readStartedMs=$readStarted;elapsedMs=[long]$clock.ElapsedMilliseconds;expectedBefore=$before;expectedAfter=$after;rows=$rows}
}
try {
  while($true) {
    $command=$null;
    $line=$null;
    if([Flux3CounterCommands]::Lines.TryDequeue([ref]$line)) {
      $command=$line|ConvertFrom-Json;
      if($command.op -cne 'boundary') {throw 'unknown counter command'}
    } elseif([Flux3CounterCommands]::Ended) {break};
    $sample=Read-Sample; @{sample=$sample;boundary=$command}|ConvertTo-Json -Depth 8 -Compress|ForEach-Object {[Console]::Out.WriteLine($_)};
    Start-Sleep -Milliseconds ${intervalMs}
  }
} catch {[Console]::Error.WriteLine($_.Exception.Message);exit 1} finally {[Flux3PdhCounters]::Close()}
`;
}
