import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import test from "node:test";
import { POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES, SC20671_MODEL_CONTRACTS, buildReceipt, buildVerifiedReceipt, cancellationSafe, compareReceipts, detectFullCacheTemporary, inventoryModelArtifact, numericSemanticSha256, readDarwinMemory, readReceiptSet, renderReceiptMarkdown, sha256, validateCampaign, validateFixtureArtifact, validateReceipt, writeCampaignSet, writeReceiptSet, writeSealedJson } from "./kv-baseline-harness.mjs";
const run = promisify(execFile);
const phases = ["process-start","weights-loaded","prefill-peak","first-token","decode-steady","prompt-cache-reuse","cancellation-cleanup","post-run-release"];
const qualityFixtures = ["kernel-fp32-reference","structured-tool-call","long-context-needle","multi-turn-prompt-cache"];
const lifecycle = Object.fromEntries(["append","chunkedPrefill","singleShotPrefill","promptCacheReuse","trim","rollback","clear","cancel","clone","batchSplit","batchMerge","prefixCopyOnWrite","pageImport","pageExport","serialization","restore","denseFallback","postRunRelease"].map(k => [k,true]));
const formula = (batch=1, capacity=4096) => batch*2*2*8*capacity*128*2;
function sealedModelId(family, role, inventory) {
  const spec = SC20671_MODEL_CONTRACTS[family][role];
  return `${spec.repository}@${spec.revision};architecture=${spec.architecture};inventory=${inventory}`;
}
function memoryPhases(persistent) {
  const active = [100,1100,1100+persistent,1100+persistent,1100+persistent,1100+persistent,1100,100];
  let mlxPeak=0,footprintPeak=0;
  return phases.map((phase,i)=>{
    mlxPeak=Math.max(mlxPeak,active[i]);
    if(phase==="prefill-peak")mlxPeak=active[i]+100;
    const physFootprintBytes=active[i]+1000;
    footprintPeak=Math.max(footprintPeak,physFootprintBytes);
    return {phase,pid:9,source:"footprint -p",timestamp:"2026-08-29T12:00:0"+i+".000Z",physFootprintBytes,physFootprintPeakBytes:footprintPeak,mlx:{source:"mlx_rs::memory",activeBytes:active[i],cacheBytes:10,peakBytes:mlxPeak}};
  });
}
function fixture(mode="dense", coordinate={}, extra={}) {
  const family = coordinate.family || "llama";
  const contract = SC20671_MODEL_CONTRACTS[family];
  const contextWindowTokens = contract.candidate.nativeContextTokens;
  const contextBand = coordinate.contextBand || "short";
  const contextTargetTokens = {short:32,medium:Math.min(1024,Math.floor(contextWindowTokens/16)),"memory-material":Math.floor(contextWindowTokens/4),"fit-boundary":Math.max(contextWindowTokens-512,Math.ceil(contextWindowTokens*.9))}[contextBand];
  const capacity = ["memory-material","fit-boundary"].includes(contextBand) ? contextTargetTokens : 4096;
  const matrix = {family,contextBand,requestMode:"single",prefillMode:"single-shot",processTemperature:"cold",...coordinate};
  const batch = matrix.requestMode === "supported-batch" ? 2 : 1; const dense = formula(batch,capacity); const persistent = mode === "dense" ? dense : Math.floor(dense/2);
  const samples = Array.from({length:5}, (_,i) => ({loadMs:10+i,prefillMs:20+i,ttftMs:25+i,firstTokenMs:30+i,decodeTokensPerSecond:100+i}));
  const operation = matrix.requestMode === "supported-batch" ? "supported-batch" : matrix.prefillMode === "chunked" ? "chunked-prefix-reuse" : "single-shot-generation";
  const probeDurationsMs = matrix.processTemperature === "cold" ? [50,6,8,7,5] : [12,7];
  const steadyDispatchMs = matrix.processTemperature === "cold" ? 6.5 : 7;
  const source = matrix.processTemperature === "cold" ? "measured-repeats" : "warmup-suites";
  const matrixCoordinate = [matrix.family,matrix.contextBand,matrix.requestMode,matrix.prefillMode,matrix.processTemperature].join("-");
  const probeEvidence = probeDurationsMs.map((dispatchMs,index)=>({index,operation,source,matrixCoordinate,setupMs:operation === "chunked-prefix-reuse" ? 1 : 0,dispatchMs,operationEvidenceSha256:sha256(`${matrixCoordinate}:${index}:${dispatchMs}`)}));
  const compileAttribution = {method:"first-dispatch-minus-steady-v1",operation,source,probeDurationsMs,probeEvidence,firstDispatchMs:probeDurationsMs[0],steadyDispatchMs,firstDispatchExcessMs:probeDurationsMs[0]-steadyDispatchMs};
  const campaignSessionId = "c".repeat(64);
  const warmupSuiteSha256 = matrix.processTemperature === "warm" ? numericSemanticSha256({probeEvidence,sessionId:campaignSessionId,workerPid:9}) : "";
  return buildReceipt({runId: mode+"-"+(coordinate.family||"llama")+"-"+Math.random(),capturedAt:"2026-08-29T12:00:00.000Z",mode,status:"complete",
    provenance:{sceneWorksRepository:"github.com/SceneWorks/SceneWorks",inferenceRepository:"github.com/SceneWorks/inference",sceneWorksRevision:"a".repeat(40),inferenceRevision:"b".repeat(40),mlxVersion:"0.25.8",mlxSource:"git+https://github.com/michaeltrefry/mlx-rs?rev="+"1".repeat(40)+"#"+"1".repeat(40),mlxRevision:"1".repeat(40),dependencyLockSha256:"e".repeat(64),os:"macOS",xcode:"Xcode",hardware:"Apple",modelId:sealedModelId(family,"candidate","d".repeat(64)),modelFileSha256:"d".repeat(64),modelFileBytes:1000,referenceModelId:sealedModelId(family,"reference","9".repeat(64)),referenceModelSha256:"9".repeat(64),referenceModelBytes:2000,powerMode:"AC",thermalState:"nominal",commandTemplate:"runner --mode {mode}",command:"runner --mode "+mode,campaignSessionId, campaignCacheStateVersion:2,coordinateOperationSha256:"f".repeat(64)},
    matrix,
    geometry:{batch,queryHeads:8,kvHeads:8,headDimension:128,queryLength:1,kvLength:capacity,layers:2,elementBytes:2,capacity,contextWindowTokens,contextTargetTokens,contextPayloadTokens:contextTargetTokens},
  memory:{modelWeightsBytes:1000,persistentKvBytes:persistent,transientWorkspaceBytes:100,denseTheoreticalKvBytes:dense,phaseSamples:memoryPhases(persistent),prefillPeakWindow:{startedAt:"2026-08-29T12:00:01.500Z",baselineActiveBytes:1100,resetPeakBytes:0},allocationEvents:[{kind:"model-weights",role:"weights",lifetime:"persistent",phase:"weights-loaded",timestamp:"2026-08-29T12:00:01.100Z",bytes:1000},{kind:"kv-cache",role:"cache",lifetime:"persistent",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.100Z",bytes:persistent},{kind:"attention-scratch",role:"attention-workspace",lifetime:"transient",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.200Z",bytes:100},{kind:"kv-cache",role:"cache",lifetime:"persistent",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.100Z",bytes:persistent},{kind:"product-cache_release",role:"cache",lifetime:"released",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.500Z",bytes:persistent}],reconciliation:{expectedDenseKvBytes:dense,observedPersistentKvBytes:persistent,toleranceBytes:0},release:{verified:true,physFootprintToleranceBytes:POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES,mlxActiveToleranceBytes:0,mlxCacheToleranceBytes:0}},
    timings:{loadMs:12,prefillMs:22,ttftMs:27,firstTokenMs:32,decodeTokensPerSecond:102,coldCompileMs:compileAttribution.firstDispatchExcessMs,warmCompileMs:compileAttribution.steadyDispatchMs,compileAttribution,samples,summary:{decodeTokensPerSecondMean:102,decodeTokensPerSecondP95:104,decodeTokensPerSecondVariance:2,decodeTokensPerSecondCoefficientOfVariation:Math.sqrt(2)/102,confidenceIntervalLow:100,confidenceIntervalHigh:104}},
    quality:{parityMaxError:0,perplexityDelta:-0.1,greedyTokenAgreement:1,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:1,statistics:{repeats:5,warmups:2,confidenceInterval:"95% bootstrap",outlierPolicy:"report all samples; no silent deletion",variancePolicy:"all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum",maxCoefficientOfVariation:0.05},fixtureEvidence:Object.fromEntries(qualityFixtures.map(f=>{const artifactName=`fixtures/${f}.json`,artifactSha256="f".repeat(64);return [f,{passed:true,artifactName,artifactSha256,artifactSidecarSha256:sha256(`${artifactSha256}  ${artifactName}\n`),independentReference:"ref"}];}))},lifecycle,cancellation:{cleanupVerified:true},warmup:{required:matrix.processTemperature==="warm",completed:matrix.processTemperature==="warm",workerPid:9,suiteSha256:warmupSuiteSha256,sessionId:matrix.processTemperature==="warm"?campaignSessionId:"",cacheStateVersion:matrix.processTemperature==="warm"?1:0},...extra});
}

function artifactFixture(raw,name,repeat=0,{includeModel=false}={}) {
  const evidence = name === "kernel-fp32-reference"
    ? {candidatePerplexity:1,referencePerplexity:1,parityErrors:[0],greedyMatches:1,greedyTotal:1}
    : {matches:1,total:1};
  const probe = raw.timings.compileAttribution.probeEvidence[Math.min(repeat,raw.timings.compileAttribution.probeEvidence.length-1)];
  return {
    fixture:name,
    independentReference:"ref",
    binding:{
      coordinate:probe.matrixCoordinate,
      repeat,
      candidate:{
        ...(includeModel?{model:raw.provenance.modelId}:{}),
        operation:probe.operation,
        compileSetupMs:probe.setupMs,
        compileDispatchMs:probe.dispatchMs,
        operationEvidenceSha256:probe.operationEvidenceSha256,
      },
      reference:{...(includeModel?{model:raw.provenance.referenceModelId}:{})},
    },
    evidence,
    metrics:{parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:1,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:1},
  };
}

async function verifiedFixture(root, mode="dense", coordinate={}, { inferenceShaped=false }={}) {
  await mkdir(root, { recursive: true });
  const model = path.join(root, "model.safetensors");
  await writeFile(model, "weights");
  const raw = fixture(mode, coordinate);
  for (const key of ["schemaVersion", "harnessVersion", "contractHash", "receiptSha256"]) delete raw[key];
  raw.provenance = {
    ...raw.provenance,
    modelFilePath: model,
    modelFileBytes: 7,
    modelFileSha256: sha256("weights"),
    modelId: sealedModelId(raw.matrix.family, "candidate", sha256("weights")),
  };
  for (const name of qualityFixtures) {
    const artifactPath = path.join(root, `${name}.json`);
    const artifact = artifactFixture(raw,name,0,{includeModel:inferenceShaped});
    const bytes = `${JSON.stringify(artifact)}\n`;
    await writeFile(artifactPath, bytes);
    raw.quality.fixtureEvidence[name] = {
      ...raw.quality.fixtureEvidence[name],
      artifactPath,
      artifactSha256: sha256(bytes),
    };
    if (name === "kernel-fp32-reference" && raw.matrix.processTemperature === "cold") {
      const repeatArtifactPaths = [artifactPath];
      for (let repeat=1;repeat<5;repeat+=1) {
        const repeatPath=path.join(root,`repeat-${repeat}-kernel-fp32-reference.json`);
        await writeFile(repeatPath,`${JSON.stringify(artifactFixture(raw,name,repeat,{includeModel:inferenceShaped}))}\n`);
        repeatArtifactPaths.push(repeatPath);
      }
      raw.quality.fixtureEvidence[name].repeatArtifactPaths=repeatArtifactPaths;
    }
  }
  return buildVerifiedReceipt(raw);
}

function rebuildReceipt(receipt, edit) {
  const raw = structuredClone(receipt);
  for (const key of ["schemaVersion", "harnessVersion", "contractHash", "receiptSha256"]) delete raw[key];
  edit(raw);
  return buildReceipt(raw);
}
test("contract sidecar and valid comparison",async()=>{const raw=await readFile("config/kv-baseline-quality-contract.json");assert.equal(await readFile("config/kv-baseline-quality-contract.json.sha256","utf8"),`${sha256(raw)}  kv-baseline-quality-contract.json\n`);assert.equal(compareReceipts(fixture(),fixture("compressed")).persistentKvReduction,.5);});
test("numeric semantic seals normalize exact f64 bits and object order",()=>{
  const first={z:91.014,a:{workerPid:9,values:[91.01400000000001,1]}};
  const reordered={a:{values:[91.01400000000001,1],workerPid:9},z:91.014};
  assert.equal(numericSemanticSha256(first),numericSemanticSha256(reordered));
  assert.equal(numericSemanticSha256({value:91.014}),sha256('{\n  "value": "f64:4056c0e560418937"\n}'));
  assert.equal(numericSemanticSha256({value:91.01400000000001}),sha256('{\n  "value": "f64:4056c0e560418938"\n}'));
  assert.equal(numericSemanticSha256({value:9}),sha256('{\n  "value": "f64:4022000000000000"\n}'));
  assert.notEqual(numericSemanticSha256({value:91.014}),numericSemanticSha256({value:91.01400000000001}));
});
test("compile attribution v4 accepts cold repeats and warmup suites",()=>{
  const cold=fixture();
  assert.equal(cold.schemaVersion,4);
  assert.equal(cold.harnessVersion,"sc-20671-kv-baseline-v4");
  const {probeEvidence,...coldAttribution}=cold.timings.compileAttribution;
  assert.deepEqual(coldAttribution,{
    method:"first-dispatch-minus-steady-v1",
    operation:"single-shot-generation",
    source:"measured-repeats",
    probeDurationsMs:[50,6,8,7,5],
    firstDispatchMs:50,
    steadyDispatchMs:6.5,
    firstDispatchExcessMs:43.5,
  });
  assert.equal(probeEvidence.length,5);
  assert.deepEqual(probeEvidence.map(({index,operation,source,matrixCoordinate,setupMs,dispatchMs})=>({index,operation,source,matrixCoordinate,setupMs,dispatchMs})),[
    {index:0,operation:"single-shot-generation",source:"measured-repeats",matrixCoordinate:"llama-short-single-single-shot-cold",setupMs:0,dispatchMs:50},
    {index:1,operation:"single-shot-generation",source:"measured-repeats",matrixCoordinate:"llama-short-single-single-shot-cold",setupMs:0,dispatchMs:6},
    {index:2,operation:"single-shot-generation",source:"measured-repeats",matrixCoordinate:"llama-short-single-single-shot-cold",setupMs:0,dispatchMs:8},
    {index:3,operation:"single-shot-generation",source:"measured-repeats",matrixCoordinate:"llama-short-single-single-shot-cold",setupMs:0,dispatchMs:7},
    {index:4,operation:"single-shot-generation",source:"measured-repeats",matrixCoordinate:"llama-short-single-single-shot-cold",setupMs:0,dispatchMs:5},
  ]);
  assert.ok(probeEvidence.every((probe)=>/^[0-9a-f]{64}$/.test(probe.operationEvidenceSha256)));
  assert.equal(cold.timings.coldCompileMs,43.5);
  assert.equal(cold.timings.warmCompileMs,6.5);
  assert.doesNotThrow(()=>validateReceipt(cold));
  const warmChunked=fixture("dense",{prefillMode:"chunked",processTemperature:"warm"});
  assert.equal(warmChunked.timings.compileAttribution.operation,"chunked-prefix-reuse");
  assert.equal(warmChunked.timings.compileAttribution.source,"warmup-suites");
  assert.deepEqual(warmChunked.timings.compileAttribution.probeDurationsMs,[12,7]);
  assert.equal(warmChunked.timings.compileAttribution.firstDispatchMs,12);
  assert.equal(warmChunked.timings.compileAttribution.steadyDispatchMs,7);
  assert.equal(warmChunked.timings.compileAttribution.firstDispatchExcessMs,5);
  assert.equal(warmChunked.timings.compileAttribution.probeEvidence.length,2);
  assert.doesNotThrow(()=>validateReceipt(warmChunked));
  assert.equal(fixture("dense",{requestMode:"supported-batch",prefillMode:"chunked"}).timings.compileAttribution.operation,"supported-batch");
  assert.ok(cold.timings.samples.every((sample)=>!("coldCompileMs" in sample)&&!("warmCompileMs" in sample)));
  assert.match(renderReceiptMarkdown(cold),/Compile attribution: first-dispatch-minus-steady-v1 \/ single-shot-generation \/ measured-repeats/);
});
test("compile attribution v4 fails closed on malformed or tampered evidence",()=>{
  const rejects=(coordinate,edit,pattern=/schema validation|compile attribution|probe durations|probeEvidence|matrix coordinate|process temperature|timing aliases|derived values|not sequential|not bound/)=>assert.throws(()=>rebuildReceipt(fixture("dense",coordinate),(raw)=>edit(raw.timings)),pattern);
  rejects({},(timings)=>{delete timings.compileAttribution.method;});
  rejects({},(timings)=>{timings.compileAttribution.probeDurationsMs[0]=Number.NaN;});
  rejects({},(timings)=>{timings.compileAttribution.firstDispatchMs=Number.POSITIVE_INFINITY;});
  rejects({},(timings)=>{timings.compileAttribution.probeDurationsMs.pop();});
  rejects({processTemperature:"warm"},(timings)=>{timings.compileAttribution.probeDurationsMs.push(6);});
  rejects({},(timings)=>{timings.compileAttribution.source="warmup-suites";});
  rejects({processTemperature:"warm"},(timings)=>{timings.compileAttribution.source="measured-repeats";});
  rejects({},(timings)=>{timings.compileAttribution.operation="chunked-prefix-reuse";});
  rejects({prefillMode:"chunked"},(timings)=>{timings.compileAttribution.operation="single-shot-generation";});
  rejects({requestMode:"supported-batch"},(timings)=>{timings.compileAttribution.operation="single-shot-generation";});
  rejects({},(timings)=>{timings.compileAttribution.steadyDispatchMs+=1;});
  rejects({},(timings)=>{timings.compileAttribution.firstDispatchExcessMs+=1;});
  rejects({},(timings)=>{timings.coldCompileMs+=1;});
  rejects({},(timings)=>{timings.warmCompileMs+=1;});
  rejects({},(timings)=>{timings.compileAttribution.probeDurationsMs=[6.5,6,8,7,5];timings.compileAttribution.firstDispatchMs=6.5;timings.compileAttribution.steadyDispatchMs=6.5;timings.compileAttribution.firstDispatchExcessMs=0;timings.coldCompileMs=0;timings.warmCompileMs=6.5;});
  rejects({processTemperature:"warm"},(timings)=>{timings.compileAttribution.probeDurationsMs=[6,7];timings.compileAttribution.firstDispatchMs=6;timings.compileAttribution.steadyDispatchMs=7;timings.compileAttribution.firstDispatchExcessMs=-1;timings.coldCompileMs=-1;timings.warmCompileMs=7;});
  rejects({},(timings)=>{timings.samples[0].coldCompileMs=1;});
  rejects({},(timings)=>{delete timings.compileAttribution.probeEvidence;});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence.pop();});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[1].index=0;});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].operation="chunked-prefix-reuse";});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].source="warmup-suites";});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].matrixCoordinate="other";});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].setupMs=-1;});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].setupMs=Number.NaN;});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].dispatchMs+=1e-12;});
  rejects({},(timings)=>{timings.compileAttribution.probeEvidence[0].operationEvidenceSha256="A".repeat(64);});
  rejects({processTemperature:"warm"},(timings)=>{timings.compileAttribution.probeEvidence.push(structuredClone(timings.compileAttribution.probeEvidence[1]));});
  assert.throws(()=>rebuildReceipt(fixture("dense",{processTemperature:"warm"}),(raw)=>{raw.warmup.suiteSha256="0".repeat(64);}),/warmup suite seal/);
});
test("Darwin footprint requires current and lifetime peak units",()=>{const timestamp="2026-08-29T12:00:00.000Z";assert.deepEqual(readDarwinMemory(9,()=>"phys_footprint: 1664 KB\nphys_footprint_peak: 2 MB\n","process-start",timestamp),{phase:"process-start",pid:9,source:"footprint -p",timestamp,physFootprintBytes:1703936,physFootprintPeakBytes:2097152});assert.throws(()=>readDarwinMemory(9,()=>"phys_footprint: 1664 KB\n","process-start",timestamp),/phys_footprint_peak/);});
test("tamper, quality, formula and reconciliation fail closed",()=>{const r=fixture();r.memory.persistentKvBytes++;assert.throws(()=>validateReceipt(r),/receiptSha256/);assert.doesNotThrow(()=>fixture("dense",{}, {quality:{...fixture().quality,parityMaxError:1,perplexityDelta:1,greedyTokenAgreement:0,multiTurnPromptCache:0}}));assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,parityMaxError:1}}),/frozen maximum/);assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,perplexityDelta:1}}),/frozen maximum/);assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,greedyTokenAgreement:0}}),/below the frozen minimum/);assert.throws(()=>fixture("dense",{}, {memory:{...fixture().memory,denseTheoreticalKvBytes:formula()*2}}),/does not reconcile/);assert.throws(()=>fixture("dense",{}, {memory:{...fixture().memory,reconciliation:{...fixture().memory.reconciliation,observedPersistentKvBytes:1}}}),/does not reconcile/);});
test("persistent cache allowed; sequential transient snapshots never sum",()=>{assert.doesNotThrow(()=>fixture());const split=[{kind:"k-temp",role:"cache",lifetime:"transient",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.000Z",bytes:formula()/2},{kind:"v-temp",role:"output",lifetime:"transient",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.100Z",bytes:formula()/2}];assert.equal(detectFullCacheTemporary(split,formula()).detected,false);assert.equal(detectFullCacheTemporary([{...split[0],kind:"full_cache_materialization"}],formula()).detected,true);assert.throws(()=>fixture("dense",{}, {memory:{...fixture().memory,allocationEvents:[{kind:"hidden",role:"other",lifetime:"transient",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.000Z",bytes:formula()}]}}),/schema validation/);});
test("prefill peak window admits released transient workspace and one boundary reset",()=>{const base=fixture(),prefill=base.memory.phaseSamples[2],window=base.memory.prefillPeakWindow;assert.equal(prefill.mlx.activeBytes,window.baselineActiveBytes+base.memory.persistentKvBytes);assert.equal(prefill.mlx.peakBytes,prefill.mlx.activeBytes+base.memory.transientWorkspaceBytes);assert.doesNotThrow(()=>rebuildReceipt(base,(raw)=>{raw.memory.phaseSamples[1].mlx.peakBytes=raw.memory.phaseSamples[2].mlx.peakBytes+1;}));assert.throws(()=>rebuildReceipt(base,(raw)=>{raw.memory.phaseSamples[3].mlx.peakBytes=raw.memory.phaseSamples[3].mlx.activeBytes;}),/decreased without a declared reset boundary/);});
test("prefill peak window is required and exact",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{delete raw.memory.prefillPeakWindow;}),/schema validation/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.prefillPeakWindow.note="unexpected";}),/schema validation/);});
test("prefill active must contain baseline plus persistent KV",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{const window=raw.memory.prefillPeakWindow;raw.memory.phaseSamples[2].mlx.activeBytes=window.baselineActiveBytes+raw.memory.persistentKvBytes-1;}),/active bytes do not contain/);});
test("prefill peak must contain baseline, persistent KV, and transient workspace",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{const window=raw.memory.prefillPeakWindow;raw.memory.phaseSamples[2].mlx.peakBytes=window.baselineActiveBytes+raw.memory.persistentKvBytes+raw.memory.transientWorkspaceBytes-1;}),/peak bytes do not contain/);});
test("prefill peak reset marker must be exactly zero",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.prefillPeakWindow.resetPeakBytes=1;}),/schema validation|must be zero/);});
test("prefill peak window must be phase-local",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.prefillPeakWindow.startedAt=raw.memory.phaseSamples[1].timestamp;}),/strictly after weights-loaded/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.prefillPeakWindow.startedAt=raw.memory.phaseSamples[2].timestamp;}),/before prefill-peak/);});
test("prefill peak window baseline must cover weights-loaded active bytes",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.prefillPeakWindow.baselineActiveBytes=raw.memory.phaseSamples[1].mlx.activeBytes-1;}),/baseline is below weights-loaded/);});
test("allocation events must be strictly inside their declared phase",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents.find((event)=>event.phase==="prefill-peak").timestamp=raw.memory.phaseSamples[2].timestamp;}),/outside its declared phase/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents.find((event)=>event.phase==="prefill-peak").timestamp=raw.memory.phaseSamples[3].timestamp;}),/outside its declared phase/);});
test("timestamp ordering preserves producer microseconds and normalizes equivalent fractions",()=>{assert.doesNotThrow(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.phaseSamples[2].timestamp="2026-08-29T12:00:02.123456Z";raw.memory.allocationEvents.find((event)=>event.phase==="prefill-peak"&&event.role==="cache").timestamp="2026-08-29T12:00:02.123457Z";}));assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.phaseSamples[2].timestamp="2026-08-29T12:00:02.0000Z";raw.memory.allocationEvents.find((event)=>event.phase==="prefill-peak"&&event.role==="cache").timestamp="2026-08-29T12:00:02.000Z";}),/outside its declared phase/);});
test("prefill containment uses the live prefill cache rather than the larger decode cache",()=>{assert.doesNotThrow(()=>rebuildReceipt(fixture(),(raw)=>{const prefillKv=raw.memory.persistentKvBytes-10;raw.memory.allocationEvents.find((event)=>event.phase==="prefill-peak"&&event.role==="cache"&&event.lifetime==="persistent").bytes=prefillKv;raw.memory.phaseSamples[2].mlx.activeBytes=raw.memory.prefillPeakWindow.baselineActiveBytes+prefillKv;raw.memory.phaseSamples[2].mlx.peakBytes=raw.memory.phaseSamples[2].mlx.activeBytes+raw.memory.transientWorkspaceBytes;}));});
test("phase-local KV snapshots and decode containment fail closed",()=>{const findCache=(raw,phase)=>raw.memory.allocationEvents.find((event)=>event.phase===phase&&event.role==="cache"&&event.lifetime==="persistent");assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{findCache(raw,"prefill-peak").bytes=raw.memory.persistentKvBytes+1;}),/reconcile/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{findCache(raw,"decode-steady").bytes-=1;raw.memory.allocationEvents.find((event)=>event.lifetime==="released").bytes-=1;}),/reconcile/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.phaseSamples[4].mlx.activeBytes=raw.memory.prefillPeakWindow.baselineActiveBytes+raw.memory.persistentKvBytes-1;}),/decode MLX active bytes/);assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents.push({kind:"decode-workspace",role:"attention-workspace",lifetime:"transient",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.200Z",bytes:200});raw.memory.transientWorkspaceBytes=200;const below=raw.memory.prefillPeakWindow.baselineActiveBytes+raw.memory.persistentKvBytes+199;for(const sample of raw.memory.phaseSamples.slice(4))sample.mlx.peakBytes=below;}),/decode MLX peak bytes/);});
test("every phase with transient evidence is bound to its MLX peak",()=>{const cancellation=[{kind:"cancellation-cache",role:"cache",lifetime:"persistent",phase:"cancellation-cleanup",timestamp:"2026-08-29T12:00:06.100Z",bytes:formula()},{kind:"cancellation-workspace",role:"output",lifetime:"transient",phase:"cancellation-cleanup",timestamp:"2026-08-29T12:00:06.200Z",bytes:200},{kind:"product-cache_release",role:"cache",lifetime:"released",phase:"cancellation-cleanup",timestamp:"2026-08-29T12:00:06.300Z",bytes:formula()}];assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents.push(...cancellation);raw.memory.transientWorkspaceBytes=200;}),/cancellation-cleanup MLX peak bytes/);assert.doesNotThrow(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents.push(...cancellation);raw.memory.transientWorkspaceBytes=200;const peak=raw.memory.prefillPeakWindow.baselineActiveBytes+raw.memory.persistentKvBytes+200;for(const sample of raw.memory.phaseSamples.slice(6))sample.mlx.peakBytes=peak;}));});
test("release, exact phase order, PID and sources fail closed",()=>{const base=fixture(),weightsLoaded=base.memory.phaseSamples[1],excessPhys=weightsLoaded.physFootprintBytes+POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES+1;assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===7?{...s,physFootprintBytes:excessPhys,physFootprintPeakBytes:excessPhys}:s)}}),/post-run footprint/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===7?{...s,physFootprintBytes:weightsLoaded.mlx.activeBytes+1,mlx:{...s.mlx,activeBytes:weightsLoaded.mlx.activeBytes+1}}:s)}}),/MLX allocator/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,release:{...base.memory.release,physFootprintToleranceBytes:POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES+1}}}),/frozen release tolerances/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,pid:10}:s)}}),/PID/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,phase:"decode-steady"}:s)}}),/phase must/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,source:"rss"}:s)}}),/source/);});
test("identity, command pairing, thermal state and cancellation cleanup",async()=>{assert.throws(()=>compareReceipts(fixture(),fixture("compressed",{}, {provenance:{...fixture("compressed").provenance,hardware:"Other"}})),/identity/);assert.throws(()=>fixture("compressed",{}, {provenance:{...fixture("compressed").provenance,command:"different"}}),/mode-substitution/);assert.throws(()=>fixture("dense",{}, {provenance:{...fixture().provenance,thermalState:"serious"}}),/schema validation/);let cleaned=0;const ac=new AbortController();ac.abort();await assert.rejects(cancellationSafe(async()=>{},async()=>{cleaned++},ac.signal));assert.equal(cleaned,1);});
test("sealed inference model contracts reject revision family and reference substitution",()=>{const llama=fixture();const qwen=fixture("dense",{family:"qwen"});assert.doesNotThrow(()=>validateReceipt(llama));assert.doesNotThrow(()=>validateReceipt(qwen));assert.throws(()=>rebuildReceipt(llama,(raw)=>{raw.provenance.modelId=raw.provenance.modelId.replace("7f0dc925e0d0afb0322d96f9255cfddf2ba5636e","0".repeat(40));}),/sealed llama contract|schema validation/);assert.throws(()=>rebuildReceipt(llama,(raw)=>{raw.provenance.modelId=qwen.provenance.modelId;}),/sealed llama contract|schema validation/);assert.throws(()=>rebuildReceipt(llama,(raw)=>{raw.provenance.referenceModelId=llama.provenance.modelId;}),/sealed llama contract|schema validation/);assert.throws(()=>rebuildReceipt(llama,(raw)=>{raw.geometry.contextWindowTokens=32768;raw.geometry.contextTargetTokens=32;raw.geometry.contextPayloadTokens=32;}),/sealed llama contract/);});
test("session/cache evidence rejects omission and warm-state mutation",()=>{const base=fixture("dense",{processTemperature:"warm"});assert.throws(()=>fixture("dense",{processTemperature:"warm"},{provenance:(({campaignSessionId,...rest})=>rest)(base.provenance)}),/schema validation|keys/);assert.throws(()=>fixture("dense",{processTemperature:"warm"},{provenance:(({coordinateOperationSha256,...rest})=>rest)(base.provenance)}),/schema validation|keys/);assert.throws(()=>fixture("dense",{processTemperature:"warm"},{warmup:{...base.warmup,sessionId:"d".repeat(64)}}),/warmup session/);assert.throws(()=>fixture("dense",{processTemperature:"warm"},{warmup:{...base.warmup,cacheStateVersion:3}}),/warmup session/);});
test("fixture receipt evidence binds exact artifact and sidecar names",()=>{const quality=structuredClone(fixture().quality);quality.fixtureEvidence["kernel-fp32-reference"].artifactName="fixtures/wrong.json";assert.throws(()=>fixture("dense",{},{quality}),/schema validation|artifact name/);const second=structuredClone(fixture().quality);second.fixtureEvidence["kernel-fp32-reference"].artifactSidecarSha256="0".repeat(64);assert.throws(()=>fixture("dense",{},{quality:second}),/sidecar binding/);});
test("verified record hashes model and fixture artifacts and publishes one receipt set",async()=>{
  const dir=await mkdtemp("/tmp/kv20671-record-"),model=path.join(dir,"model.safetensors"),input=path.join(dir,"input.json"),out=path.join(dir,"dense.receipt");
  await writeFile(model,"weights");
  const raw=fixture();
  for(const key of ["schemaVersion","harnessVersion","contractHash","receiptSha256"])delete raw[key];
  raw.provenance={...raw.provenance,modelFilePath:model,modelFileBytes:7,modelFileSha256:sha256("weights"),modelId:sealedModelId("llama","candidate",sha256("weights"))};
  for(const name of qualityFixtures){
    const artifactPath=path.join(dir,`${name}.json`),bytes=`${JSON.stringify(artifactFixture(raw,name))}\n`;
    await writeFile(artifactPath,bytes);
    raw.quality.fixtureEvidence[name]={...raw.quality.fixtureEvidence[name],artifactPath,artifactSha256:sha256(bytes)};
    if(name==="kernel-fp32-reference"){
      const repeatArtifactPaths=[artifactPath];
      for(let repeat=1;repeat<5;repeat+=1){const repeatPath=path.join(dir,`repeat-${repeat}-kernel.json`);await writeFile(repeatPath,`${JSON.stringify(artifactFixture(raw,name,repeat))}\n`);repeatArtifactPaths.push(repeatPath);}
      raw.quality.fixtureEvidence[name].repeatArtifactPaths=repeatArtifactPaths;
    }
  }
  await writeFile(input,JSON.stringify(raw));
  await run(process.execPath,["scripts/kv-baseline-harness.mjs","record",input,out]);
  assert.equal((await readReceiptSet(out)).provenance.modelFileSha256,sha256("weights"));
  for(const file of [path.join(out,"receipt.json"),path.join(out,"receipt.md")]){const bytes=await readFile(file,"utf8");assert.equal(await readFile(file+".sha256","utf8"),`${sha256(bytes)}  ${path.basename(file)}\n`);}
  await assert.rejects(buildVerifiedReceipt({...raw,provenance:{...raw.provenance,modelFilePath:path.join(dir,"missing")}}),/unavailable/);
  await rm(dir,{recursive:true,force:true});
});
test("fixture artifacts accept zero raw matches but reject aggregate or wrong-shape evidence",()=>{const row={independentReference:"ref"},metrics={parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:0,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:0};assert.doesNotThrow(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{matches:0,total:8},metrics},"multi-turn-prompt-cache",row));assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{matches:-1,total:8},metrics},"multi-turn-prompt-cache",row),/non-negative/);assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{matches:0,total:0},metrics},"multi-turn-prompt-cache",row),/positive/);assert.throws(()=>validateFixtureArtifact({fixture:"structured-tool-call",independentReference:"ref",evidence:{matches:1},metrics:{}},"structured-tool-call",row),/must be finite/);assert.throws(()=>validateFixtureArtifact({fixture:"long-context-needle",independentReference:"other",evidence:{matches:1,total:1},metrics:{...metrics,greedyTokenAgreement:1,multiTurnPromptCache:1}},"long-context-needle",row),/reference mismatch/);});
test("sharded model identity covers every resolved snapshot file",async()=>{const dir=await mkdtemp("/tmp/kv20671-snapshot-"),snapshot=path.join(dir,"snapshot");await mkdir(path.join(snapshot,"nested"),{recursive:true});await writeFile(path.join(snapshot,"config.json"),"config");await writeFile(path.join(snapshot,"nested","model-00001-of-00002.safetensors"),"first");await writeFile(path.join(snapshot,"nested","model-00002-of-00002.safetensors"),"second");const first=await inventoryModelArtifact(snapshot);assert.equal(first.bytes,17);assert.equal(first.files,3);await writeFile(path.join(snapshot,"nested","model-00002-of-00002.safetensors"),"changed");const second=await inventoryModelArtifact(snapshot);assert.notEqual(first.sha256,second.sha256);await writeFile(path.join(snapshot,"empty.safetensors"),"");await assert.rejects(inventoryModelArtifact(snapshot),/empty or unsupported/);await rm(dir,{recursive:true,force:true});});
test("receipt sets reject mixed generations and CLI comparison preserves inputs",async()=>{const dir=await mkdtemp("/tmp/kv20671-"),dense=path.join(dir,"dense"),compressed=path.join(dir,"compressed"),out=path.join(dir,"comparison.json");const denseReceipt=await verifiedFixture(path.join(dir,"dense-source")),compressedReceipt=await verifiedFixture(path.join(dir,"compressed-source"),"compressed");await writeReceiptSet(dense,denseReceipt);await writeReceiptSet(compressed,compressedReceipt);const before=await readFile(path.join(compressed,"receipt.json"),"utf8");await run(process.execPath,["scripts/kv-baseline-harness.mjs","compare",dense,compressed,out]);assert.equal(await readFile(path.join(compressed,"receipt.json"),"utf8"),before);assert.match(await readFile(path.join(dir,"comparison.md"),"utf8"),/Persistent KV reduction: 50.00%/);await writeFile(path.join(dense,"receipt.md"),"# forged\n");await assert.rejects(readReceiptSet(dense),/sidecar hash mismatch|not bound/);await rm(dir,{recursive:true,force:true});});
test("published inference fixture bundles are required and bind cold probe evidence exactly",async()=>{
  const dir=await mkdtemp("/tmp/kv20671-fixtures-"),published=path.join(dir,"published"),receipt=await verifiedFixture(path.join(dir,"source"),"dense",{}, {inferenceShaped:true});
  await writeReceiptSet(published,receipt);
  const loaded=await readReceiptSet(published);
  assert.doesNotThrow(()=>validateReceipt(loaded));
  const repeatName="fixtures/repeat-2/kernel-fp32-reference.json",repeatFile=path.join(published,repeatName),repeatOriginal=await readFile(repeatFile,"utf8"),repeat=JSON.parse(repeatOriginal);
  repeat.binding.candidate.compileSetupMs+=Number.EPSILON;
  const tampered=`${JSON.stringify(repeat)}\n`;
  await writeFile(repeatFile,tampered);
  await writeFile(`${repeatFile}.sha256`,`${sha256(tampered)}  ${repeatName}\n`);
  await assert.rejects(readReceiptSet(published),/candidate compile evidence mismatch/);
  await writeFile(repeatFile,repeatOriginal);
  await writeFile(`${repeatFile}.sha256`,`${sha256(repeatOriginal)}  ${repeatName}\n`);
  const artifact=path.join(published,"fixtures","structured-tool-call.json");
  const artifactOriginal=await readFile(artifact);
  await writeFile(artifact,"{}\n");
  await assert.rejects(readReceiptSet(published),/not bound/);
  await writeFile(artifact,artifactOriginal);
  await rm(path.join(published,"fixtures","long-context-needle.json"));
  await assert.rejects(readReceiptSet(published),/ENOENT/);
  await rm(dir,{recursive:true,force:true});
});
test("volatile sessions do not change stable model identity while invariant drift fails",()=>{const rows=[];for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"]){const row=fixture("dense",{family,contextBand,requestMode,prefillMode,processTemperature});rows.push(rebuildReceipt(row,(raw)=>{raw.provenance.campaignSessionId=sha256(raw.runId);if(raw.warmup.required){raw.warmup.sessionId=raw.provenance.campaignSessionId;raw.warmup.suiteSha256=numericSemanticSha256({probeEvidence:raw.timings.compileAttribution.probeEvidence,sessionId:raw.warmup.sessionId,workerPid:raw.warmup.workerPid});}}));}assert.equal(validateCampaign(rows).coordinates,64);assert.equal(compareReceipts(rows[0],rebuildReceipt(fixture("compressed",rows[0].matrix),(raw)=>{raw.provenance.campaignSessionId="7".repeat(64);raw.provenance.modelId=rows[0].provenance.modelId;raw.provenance.referenceModelId=rows[0].provenance.referenceModelId;})).persistentKvReduction,.5);const refDrift=[...rows];refDrift[1]=rebuildReceipt(refDrift[1],(raw)=>{raw.provenance.referenceModelSha256="8".repeat(64);raw.provenance.referenceModelId=sealedModelId(raw.matrix.family,"reference",raw.provenance.referenceModelSha256);});assert.throws(()=>validateCampaign(refDrift),/model identity drift/);const sourceDrift=[...rows];sourceDrift[1]=rebuildReceipt(sourceDrift[1],(raw)=>{raw.provenance.inferenceRevision="8".repeat(40);});assert.throws(()=>validateCampaign(sourceDrift),/identity drift/);assert.throws(()=>rebuildReceipt(rows[0],(raw)=>{raw.provenance.mlxSource=`git+https://github.com/fork/mlx-rs?rev=${raw.provenance.mlxRevision}#${raw.provenance.mlxRevision}`;}),/exact Git revision/);});
test("measured context bands reject zero, non-material, and non-boundary receipts",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.geometry.contextWindowTokens=0;}),/must be >= 1|positive integer/);assert.throws(()=>rebuildReceipt(fixture("dense",{contextBand:"memory-material"}),(raw)=>{raw.memory.phaseSamples=raw.memory.phaseSamples.map((sample)=>({...sample,physFootprintBytes:10_000_000_000,physFootprintPeakBytes:10_000_000_000}));}),/memory-material/);assert.throws(()=>rebuildReceipt(fixture("dense",{contextBand:"fit-boundary"}),(raw)=>{raw.geometry.contextWindowTokens=8192;raw.geometry.contextTargetTokens=7680;raw.geometry.contextPayloadTokens=7680;}),/fit-boundary|sealed llama contract/);});
test("full 64-coordinate dense campaign and invalid sets",()=>{const rows=[];for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"])rows.push(fixture("dense",{family,contextBand,requestMode,prefillMode,processTemperature}));assert.equal(validateCampaign(rows).coordinates,64);assert.throws(()=>validateCampaign([rows[0]]),/incomplete/);assert.throws(()=>validateCampaign([fixture("compressed")]),/non-dense/);assert.throws(()=>validateCampaign([...rows,rows[0]]),/duplicate/);});
test("complete campaign publication is one atomic 64-coordinate set",async()=>{const root=await mkdtemp("/tmp/kv20671-campaign-"),inputs=path.join(root,"workers"),destination=path.join(root,"published"),coordinates=[];await mkdir(inputs);for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"])coordinates.push({family,contextBand,requestMode,prefillMode,processTemperature});const sets=[];for(let index=0;index<coordinates.length;index+=1){const source=path.join(inputs,`source-${index}`),set=path.join(inputs,String(index)),receipt=await verifiedFixture(source,"dense",coordinates[index]);await writeReceiptSet(set,receipt);sets.push(set);}await assert.rejects(writeCampaignSet(destination,sets.slice(1)),/exactly 64/);assert.equal((await rm(destination,{recursive:true,force:true})),undefined);await writeCampaignSet(destination,sets);const manifest=JSON.parse(await readFile(path.join(destination,"campaign.json"),"utf8"));assert.equal(manifest.coordinates,64);assert.equal(manifest.coordinateReceipts.length,64);assert.equal(await readFile(path.join(destination,"campaign.json.sha256"),"utf8"),`${sha256(await readFile(path.join(destination,"campaign.json"),"utf8"))}  campaign.json\n`);await rm(root,{recursive:true,force:true});});

test("output-role concat coexistence participates in transient reconciliation",()=>{
  const base=fixture();
  base.memory.allocationEvents.push(
    {kind:"dense_concat_coexistence",role:"output",lifetime:"transient",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.500Z",bytes:200},
    {kind:"dense_concat_coexistence",role:"output",lifetime:"transient",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.600Z",bytes:150},
    {kind:"dense_concat_coexistence",role:"output",lifetime:"transient",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.250Z",bytes:175},
  );
  base.memory.transientWorkspaceBytes=200;
  for(const sample of base.memory.phaseSamples.slice(2))sample.mlx.peakBytes+=100;
  assert.doesNotThrow(()=>buildReceipt(base));
  base.memory.transientWorkspaceBytes=199;
  assert.throws(()=>buildReceipt(base),/reconcile/);
});

test("cache release is exact lifecycle evidence and never allocation",()=>{
  const base=fixture();
  assert.equal(detectFullCacheTemporary(base.memory.allocationEvents,base.memory.denseTheoreticalKvBytes).detected,false);
  assert.match(renderReceiptMarkdown(base),new RegExp(`Released cache ownership bytes: ${base.memory.persistentKvBytes}`));
  assert.throws(()=>rebuildReceipt(base,(raw)=>{raw.memory.allocationEvents=raw.memory.allocationEvents.filter((event)=>event.lifetime!=="released");}),/explicitly released/);
  assert.throws(()=>rebuildReceipt(base,(raw)=>{raw.memory.allocationEvents.find((event)=>event.lifetime==="released").bytes-=1;}),/release bytes/);
  assert.throws(()=>rebuildReceipt(base,(raw)=>{raw.memory.allocationEvents.find((event)=>event.lifetime==="released").kind="reset";}),/cache release|schema validation/);
});

test("allocation accounting fails closed outside the safe integer range",()=>{
  assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.memory.allocationEvents[0].bytes=Number.MAX_SAFE_INTEGER+1;}),/safe integer|schema validation/);
  assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.geometry.capacity=Number.MAX_SAFE_INTEGER;}),/overflow|schema validation/);
});
