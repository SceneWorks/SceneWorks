import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import test from "node:test";
import { FORCED_CONTINUATION_TOKENS, MULTI_TURN_PROMPT_CACHE_METHOD, qualityGateFromRepeats, qualityGateSummary, sealedRepeatQualityMetrics, validateSealedQualityGate, campaignQualityGatePassed, GREEDY_AGREEMENT_METHOD, POST_RELEASE_MLX_SLACK_FLOOR_BYTES, POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES, campaignHostStateVaried, hostStateThrottled, pmsetCpuSpeedLimit, postReleaseMlxSlackBytes, SC20671_COVERING_SCHEDULE, SC20671_MODEL_CONTRACTS, buildReceipt, checkContract, renderComparisonMarkdown, validateAdmissionPolicy, validateFixtureOutcomes, validatePrimaryDiscrimination, validateRepeatDiscrimination, sameWeightsFixtureReference, buildVerifiedReceipt, campaignPolicySha256, campaignResumeIdentitySha256, canonicalJson, cancellationSafe, compareReceipts, detectFullCacheTemporary, inventoryModelArtifact, kernelPathValid, numericSemanticSha256, readCampaignSet, readDarwinMemory, readReceiptSet, renderReceiptMarkdown, sha256, validateCampaign, validateFixtureArtifact, validateReceipt, writeCampaignSet, writeReceiptSet, writeSealedJson } from "./kv-baseline-harness.mjs";
const run = promisify(execFile);
const phases = ["process-start","weights-loaded","prefill-peak","first-token","decode-steady","prompt-cache-reuse","cancellation-cleanup","post-run-release"];
const qualityFixtures = ["kernel-fp32-reference","structured-tool-call","long-context-needle","multi-turn-prompt-cache"];
const lifecycle = Object.fromEntries(["append","chunkedPrefill","singleShotPrefill","promptCacheReuse","trim","rollback","clear","cancel","clone","batchSplit","batchMerge","prefixCopyOnWrite","pageImport","pageExport","serialization","restore","denseFallback","postRunRelease"].map(k => [k,true]));
const formula = (batch=1, capacity=4096) => batch*2*2*8*capacity*128*2;
function sealedModelId(family, role, inventory) {
  const spec = SC20671_MODEL_CONTRACTS[family][role];
  return `${spec.repository}@${spec.revision};architecture=${spec.architecture};inventory=${inventory}`;
}
const PMSET_NOMINAL = "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded";
function hostState(boundary, capturedAt, overrides={}) {
  return {boundary,capturedAt,powerMode:"automatic",thermalState:"nominal",cpuSpeedLimit:null,pmsetThermalRaw:PMSET_NOMINAL,throttled:false,...overrides};
}
// v6 derived evidence, recomputed from a raw receipt's own phases, samples and host states.
const DERIVED_PATHS = [
  ["memory","release","mlxActiveToleranceBytes"],["memory","release","mlxCacheToleranceBytes"],
  ["memory","release","mlxActiveResidualBytes"],["memory","release","mlxCacheResidualBytes"],
  ["memory","denseKvShareBps"],["memory","belowMemoryMaterialShare"],
  ["provenance","thermalChangedDuringRow"],["provenance","powerModeChangedDuringRow"],
];
function derivedV6(raw) {
  const start = raw.memory.phaseSamples[1], released = raw.memory.phaseSamples.at(-1);
  const prefill = raw.memory.phaseSamples[2].physFootprintBytes, dense = raw.memory.denseTheoreticalKvBytes;
  const [rowStart, rowEnd] = raw.provenance.hostStates;
  const later = [...raw.timings.samples.map((sample)=>sample.hostState), rowEnd].filter(Boolean);
  return [
    postReleaseMlxSlackBytes(start.mlx.activeBytes), postReleaseMlxSlackBytes(start.mlx.cacheBytes),
    Math.max(0, released.mlx.activeBytes - start.mlx.activeBytes),
    Math.max(0, released.mlx.cacheBytes - start.mlx.cacheBytes),
    Number(BigInt(dense) * 10_000n / BigInt(prefill)),
    raw.matrix.contextBand === "memory-material" && BigInt(dense) * 10_000n < BigInt(prefill) * 1_000n,
    later.some((state)=>state.thermalState !== rowStart.thermalState || state.throttled),
    later.some((state)=>state.powerMode !== rowStart.powerMode),
  ];
}
const getPath = (raw, keys) => keys.reduce((value, key) => value?.[key], raw);
const setPath = (raw, keys, value) => { keys.slice(0, -1).reduce((node, key) => node[key], raw)[keys.at(-1)] = value; };
function withDerived(raw) {
  derivedV6(raw).forEach((value, index) => setPath(raw, DERIVED_PATHS[index], value));
  return raw;
}
function memoryPhases(persistent) {
  const active = [100,1100,1100+persistent,1100+persistent,1100+persistent,1100+persistent,1100,100];
  let mlxPeak=0,footprintPeak=0;
  return phases.map((phase,i)=>{
    mlxPeak=Math.max(mlxPeak,active[i]);
    if(phase==="prefill-peak")mlxPeak=active[i]+100;
    const physFootprintBytes=active[i]+1000;
    footprintPeak=Math.max(footprintPeak,physFootprintBytes);
    return {phase,pid:9,source:"proc_pid_rusage",timestamp:"2026-08-29T12:00:0"+i+".000Z",physFootprintBytes,physFootprintPeakBytes:footprintPeak,mlx:{source:"mlx_rs::memory",activeBytes:active[i],cacheBytes:10,peakBytes:mlxPeak}};
  });
}
// The coordinate's measured storage: its device share is the receipt's persistent KV and its host
// copy/staged tail (half the device share here) is counted into the physical bytes.
const compressionFixture = ({ persistent, kvLength }) => {
  const deviceMetadataBytes = Math.floor(persistent / 4), hostPayloadBytes = Math.floor(persistent / 2);
  return {
    method:"group-affine",representationIdentity:"sc-20676-packed-group-affine-v1",representationVersion:2,bits:2,quantizationGroupSize:32,
    kernelGpuFamily:"apple7-or-newer",kernelPaths:[
      {kernel:"sc20676_nax_tiled_matmul2d",selection:"nax-selected",reason:"mlx::core::metal::is_nax_available() and 16-bit queries at D 64/128",queryDtype:"bfloat16",calls:2},
      {kernel:"sc20676_split_kv_simdgroup",selection:"below-multi-row-threshold",reason:"fewer query rows than packed_tiled_min_query_tokens",queryDtype:"bfloat16",calls:8},
    ],
    deviceCodeBytes:persistent-deviceMetadataBytes,deviceMetadataBytes,hostPayloadBytes,physicalKvBytes:persistent+hostPayloadBytes,storageTokens:kvLength,
    persistentKvRepresentation:"compressed",
    fusedCalls:10,fallbackCalls:1,fallbacks:[{operation:"prompt-cache-reuse",reason:"the provider prefix cache stores dense contiguous K/V",calls:1}],
    fullCacheDequantizations:0,failedDispatches:0,
  };
};
// A coordinate that itself ran on the explicit dense path (batch / prefix reuse) claims no storage.
const denseFallbackCompression = () => ({
  ...compressionFixture({ persistent: 0, kvLength: 0 }),
  persistentKvRepresentation:"dense-fallback",
  fallbackCalls:2,fallbacks:[
    {operation:"prompt-cache-reuse",reason:"the provider prefix cache stores dense contiguous K/V",calls:1},
    {operation:"supported-batch",reason:"batched prefill attends through additive padding masks",calls:1},
  ],
});
// The inference supervisor's macOS admission measurement (campaign_supervisor::HostMemory).
function hostMemoryFor(requiredBytes, counters) {
  const pageSizeBytes=16384;
  const pages=counters ?? {freePages:Math.ceil(requiredBytes/pageSizeBytes),speculativePages:0,purgeablePages:0,inactivePages:0,fileBackedPages:0,anonymousPages:0,throttledPages:0,activePages:0};
  const reclaimableFilePages=Math.max(0,pages.fileBackedPages-pages.speculativePages);
  const availableBytes=(pages.freePages+pages.speculativePages+pages.purgeablePages+reclaimableFilePages)*pageSizeBytes;
  return {metric:"darwin-vm-stat-available-v3",pageSizeBytes,...pages,reclaimableFilePages,availableBytes};
}
// A compressed row's forced continuation over `tokens` positions with `matches` agreeing.
function forcedContinuation(matches=FORCED_CONTINUATION_TOKENS, tokens=FORCED_CONTINUATION_TOKENS) {
  const flips=Array.from({length:tokens-matches},(_,i)=>matches+i);
  return {method:"dense-kv-same-weights-greedy-continuation-eos-ignored-teacher-forced",tokens,matches,agreement:matches/tokens,flipCount:tokens-matches,firstFlipPositions:flips.slice(0,32),referenceStreamSha256:sha256("reference"),candidateChoicesSha256:sha256(`choices-${matches}`)};
}
// A compressed row's contract v4 turn-2 forced continuation (multiTurnPromptCache).
function multiTurnForcedContinuation(matches=FORCED_CONTINUATION_TOKENS, tokens=FORCED_CONTINUATION_TOKENS) {
  return {...forcedContinuation(matches,tokens),method:"dense-kv-same-weights-turn-2-prompt-cache-hit-greedy-continuation-eos-ignored-teacher-forced",candidateChoicesSha256:sha256(`turn-2-choices-${matches}`)};
}
// Both arms' multi-turn prompt-cache records: turn 1 misses, turn 2 reuses turn 1's prefix.
const MULTI_TURN_TURNS={turn1:{promptTokens:40,promptSha256:"1".repeat(64),cacheHit:false,reusedPrefixTokens:0},turn2:{promptTokens:70,promptSha256:"2".repeat(64),cacheHit:true,reusedPrefixTokens:45}};
const MULTI_TURN_FORCED_PASS={reference:MULTI_TURN_TURNS,candidate:MULTI_TURN_TURNS};
const MULTI_TURN_QUALITY={multiTurnPromptCacheMethod:MULTI_TURN_PROMPT_CACHE_METHOD,multiTurnFreeRunningFirstDivergence:null,multiTurnMatchedPrefixTokens:8,multiTurnCache:{candidate:MULTI_TURN_TURNS,reference:MULTI_TURN_TURNS}};
// Compressed quality records its forced continuations and the gate of its receipt-level values
// (every repeat measured the same values).
function gatedQuality(mode, quality, continuation=forcedContinuation(), multiTurn=multiTurnForcedContinuation()) {
  if (mode !== "compressed") return quality;
  const gated={...quality,greedyTokenAgreement:continuation.agreement,greedyTokenAgreementByRepeat:Array(5).fill(continuation.agreement),forcedContinuation:continuation,multiTurnPromptCache:multiTurn.agreement,multiTurnForcedContinuation:multiTurn,multiTurnForcedPass:MULTI_TURN_FORCED_PASS};
  return {...gated,qualityGate:qualityGateFromRepeats(Array(5).fill(gated))};
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
  const samples = Array.from({length:5}, (_,i) => ({loadMs:10+i,prefillMs:20+i,ttftMs:25+i,firstTokenMs:30+i,decodeTokensPerSecond:100+i,steadyDecodePromptTokens:48,steadyDecodeGeneratedTokens:256,steadyDecodeTimedTokens:255,steadyDecodeMs:255000/(100+i),steadyDecodeForcedStopTokens:2,hostState:hostState("timing-sample",`2026-08-29T12:00:08.${100+i*10}Z`)}));
  const operation = matrix.requestMode === "supported-batch" ? "supported-batch" : matrix.prefillMode === "chunked" ? "chunked-prefix-reuse" : "single-shot-generation";
  const probeDurationsMs = matrix.processTemperature === "cold" ? [50,6,8,7,5] : [12,7];
  const steadyDispatchMs = matrix.processTemperature === "cold" ? 6.5 : 7;
  const source = matrix.processTemperature === "cold" ? "measured-repeats" : "warmup-suites";
  const matrixCoordinate = [matrix.family,matrix.contextBand,matrix.requestMode,matrix.prefillMode,matrix.processTemperature].join("-");
  const probeEvidence = probeDurationsMs.map((dispatchMs,index)=>({index,operation,source,matrixCoordinate,setupMs:operation === "chunked-prefix-reuse" ? 1 : 0,dispatchMs,operationEvidenceSha256:sha256(`${matrixCoordinate}:${index}:${dispatchMs}`)}));
  const noiseSamplesMs = matrix.processTemperature === "cold" ? probeDurationsMs.slice(1) : [7,7.5,7.2,7.1,7.4];
  const noiseBandMs = Math.max(...noiseSamplesMs)-Math.min(...noiseSamplesMs);
  const compileAttribution = {method:"first-dispatch-minus-steady-v2",operation,source,probeDurationsMs,probeEvidence,firstDispatchMs:probeDurationsMs[0],steadyDispatchMs,firstDispatchExcessMs:probeDurationsMs[0]-steadyDispatchMs,noiseSamplesMs,noiseBandMs,compileCostResolved:true,compileCostMs:probeDurationsMs[0]-steadyDispatchMs};
  const campaignSessionId = "c".repeat(64);
  const warmupSuiteSha256 = matrix.processTemperature === "warm" ? numericSemanticSha256({probeEvidence,sessionId:campaignSessionId,workerPid:9}) : "";
  return buildReceipt(withDerived({runId: mode+"-"+(coordinate.family||"llama")+"-"+Math.random(),capturedAt:"2026-08-29T12:00:00.000Z",mode,status:"complete",
    provenance:{sceneWorksRepository:"github.com/SceneWorks/SceneWorks",inferenceRepository:"github.com/SceneWorks/inference",sceneWorksRevision:"a".repeat(40),inferenceRevision:"b".repeat(40),mlxVersion:"0.25.8",mlxSource:"git+https://github.com/michaeltrefry/mlx-rs?rev="+"1".repeat(40)+"#"+"1".repeat(40),mlxRevision:"1".repeat(40),dependencyLockSha256:"e".repeat(64),os:"macOS",xcode:"Xcode",hardware:"Apple",modelId:sealedModelId(family,"candidate","d".repeat(64)),modelFileSha256:"d".repeat(64),modelFileBytes:1000,referenceModelId:sealedModelId(family,"reference","9".repeat(64)),referenceModelSha256:"9".repeat(64),referenceModelBytes:2000,powerMode:"automatic",thermalState:"nominal",hostStates:[hostState("row-start","2026-08-29T11:59:59.000Z"),hostState("row-end","2026-08-29T12:00:09.000Z")],thermalChangedDuringRow:false,powerModeChangedDuringRow:false,commandTemplate:"runner --mode {mode}",command:"runner --mode "+mode,campaignSessionId, campaignCacheStateVersion:2,coordinateOperationSha256:"f".repeat(64)},
    matrix,
    geometry:{batch,queryHeads:8,kvHeads:8,headDimension:128,queryLength:1,kvLength:capacity,layers:2,elementBytes:2,capacity,contextWindowTokens,contextTargetTokens,contextPayloadTokens:contextTargetTokens},
  memory:{modelWeightsBytes:1000,persistentKvBytes:persistent,transientWorkspaceBytes:100,denseTheoreticalKvBytes:dense,phaseSamples:memoryPhases(persistent),prefillPeakWindow:{startedAt:"2026-08-29T12:00:01.500Z",baselineActiveBytes:1100,resetPeakBytes:0},allocationEvents:[{kind:"model-weights",role:"weights",lifetime:"persistent",phase:"weights-loaded",timestamp:"2026-08-29T12:00:01.100Z",bytes:1000},{kind:"kv-cache",role:"cache",lifetime:"persistent",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.100Z",bytes:persistent},{kind:"attention-scratch",role:"attention-workspace",lifetime:"transient",phase:"prefill-peak",timestamp:"2026-08-29T12:00:02.200Z",bytes:100},{kind:"kv-cache",role:"cache",lifetime:"persistent",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.100Z",bytes:persistent},{kind:"product-cache_release",role:"cache",lifetime:"released",phase:"decode-steady",timestamp:"2026-08-29T12:00:04.500Z",bytes:persistent}],reconciliation:{expectedDenseKvBytes:dense,observedPersistentKvBytes:persistent,toleranceBytes:0},release:{verified:true,physFootprintToleranceBytes:POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES,mlxActiveToleranceBytes:0,mlxCacheToleranceBytes:0,mlxActiveResidualBytes:0,mlxCacheResidualBytes:0},admission:{mode:"runtime-guarded",rule:"estimate-plus-reserve-v1",childFootprintCapBytes:1<<30,hostFreeReserveBytes:1<<30,staticFootprintFloorBytes:1<<20,estimateSource:"sc20671-static-row-footprint-budget",estimateBytes:1<<20,hostMemoryComponents:hostMemoryFor(2**31)},denseKvShareBps:0,belowMemoryMaterialShare:false},
    timings:{loadMs:12,prefillMs:22,ttftMs:27,firstTokenMs:32,decodeTokensPerSecond:102,coldCompileMs:compileAttribution.firstDispatchExcessMs,warmCompileMs:compileAttribution.steadyDispatchMs,compileAttribution,samples,summary:{decodeTokensPerSecondMean:102,decodeTokensPerSecondP95:104,decodeTokensPerSecondVariance:2,decodeTokensPerSecondCoefficientOfVariation:Math.sqrt(2)/102,confidenceIntervalLow:100,confidenceIntervalHigh:104}},
    quality:gatedQuality(mode,{parityMaxError:0,perplexityDelta:-0.1,greedyTokenAgreement:1,greedyAgreementMethod:GREEDY_AGREEMENT_METHOD,freeRunningFirstDivergence:null,greedyTokenAgreementByRepeat:[1,1,1,1,1],structuredToolAgreement:1,needleRetrieval:1,needleDiscriminating:true,toolDiscriminating:true,multiTurnPromptCache:1,...MULTI_TURN_QUALITY,statistics:{repeats:5,warmups:2,confidenceInterval:"95% bootstrap",outlierPolicy:"report all samples; no silent deletion",variancePolicy:"all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum",maxCoefficientOfVariation:0.05},fixtureEvidence:Object.fromEntries(qualityFixtures.map(f=>{const artifactName=`fixtures/${f}.json`,artifactSha256="f".repeat(64);return [f,{passed:true,artifactName,artifactSha256,artifactSidecarSha256:sha256(`${artifactSha256}  ${artifactName}\n`),independentReference:mode==="compressed"?sameWeightsFixtureReference(f,"d".repeat(64)):"ref"}];}))}),lifecycle,cancellation:{cleanupVerified:true},warmup:{required:matrix.processTemperature==="warm",completed:matrix.processTemperature==="warm",workerPid:9,suiteSha256:warmupSuiteSha256,sessionId:matrix.processTemperature==="warm"?campaignSessionId:"",cacheStateVersion:matrix.processTemperature==="warm"?1:0},...(mode==="compressed"?{compression:compressionFixture({persistent,kvLength:capacity})}:{}),...extra}));
}

function artifactFixture(raw,name,repeat=0,{includeModel=false}={}) {
  const continuation = raw.quality.forcedContinuation;
  const evidence = name === "kernel-fp32-reference"
    ? (continuation
      // Compressed: 0.1 - 0.2 is exactly the fixture's perplexityDelta of -0.1.
      ? {candidatePerplexity:0.1,referencePerplexity:0.2,parityErrors:[0],greedyMatches:continuation.matches,greedyTotal:continuation.tokens,freeRunningFirstDivergence:null,forcedContinuation:continuation}
      : {candidatePerplexity:1,referencePerplexity:1,parityErrors:[0],greedyMatches:1,greedyTotal:1,freeRunningFirstDivergence:null})
    : name === "structured-tool-call"
      ? {matches:1,total:1,candidateValid:true,referenceValid:true,outputsMatch:true,discriminating:true}
      : name === "long-context-needle"
        ? {matches:1,total:1,candidateRecovered:true,referenceRecovered:true,outputsMatch:true,discriminating:true}
        : {...(raw.quality.multiTurnForcedContinuation
          ? {matches:raw.quality.multiTurnForcedContinuation.matches,total:raw.quality.multiTurnForcedContinuation.tokens,forcedContinuation:raw.quality.multiTurnForcedContinuation,forcedPass:raw.quality.multiTurnForcedPass}
          : {matches:1,total:1}),method:MULTI_TURN_PROMPT_CACHE_METHOD,freeRunningFirstDivergence:null,matchedPrefixTokens:8,turns:{candidate:MULTI_TURN_TURNS,reference:MULTI_TURN_TURNS}};
  const probe = raw.timings.compileAttribution.probeEvidence[Math.min(repeat,raw.timings.compileAttribution.probeEvidence.length-1)];
  return {
    fixture:name,
    independentReference:raw.quality.fixtureEvidence[name].independentReference,
    binding:{
      coordinate:probe.matrixCoordinate,
      repeat,
      candidate:{
        ...(includeModel?{
          model:raw.provenance.modelId,
          coordinateInventorySha256:raw.provenance.modelFileSha256,
          operationPromptTokens:raw.geometry.contextPayloadTokens,
        }:{}),
        operation:probe.operation,
        compileSetupMs:probe.setupMs,
        compileDispatchMs:probe.dispatchMs,
        operationEvidenceSha256:probe.operationEvidenceSha256,
      },
      reference:{...(includeModel?{
        model:raw.provenance.referenceModelId,
        coordinateInventorySha256:raw.provenance.referenceModelSha256,
        operationPromptTokens:raw.geometry.contextPayloadTokens,
      }:{}),...(raw.mode==="compressed"?{
        coordinateInventorySha256:raw.provenance.modelFileSha256,
        qualityInventorySha256:raw.provenance.modelFileSha256,
      }:{})},
    },
    evidence,
    metrics:{parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:1,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:1},
  };
}

async function verifiedFixture(root, mode="dense", coordinate={}, { inferenceShaped=false, pid=9, policy, continuation }={}) {
  await mkdir(root, { recursive: true });
  const model = path.join(root, "model.safetensors");
  await writeFile(model, "weights");
  const raw = fixture(mode, coordinate);
  if (continuation) {
    const { qualityGate: _gate, forcedContinuation: _forced, multiTurnForcedContinuation: _turn2, multiTurnForcedPass: _pass, ...measured } = raw.quality;
    raw.quality = gatedQuality(mode, measured, continuation);
  }
  for (const key of ["schemaVersion", "harnessVersion", "contractHash", "receiptSha256"]) delete raw[key];
  raw.memory.phaseSamples = raw.memory.phaseSamples.map((sample) => ({ ...sample, pid }));
  if (policy) {
    raw.memory.admission = {
      ...raw.memory.admission,
      childFootprintCapBytes: policy.childFootprintCapBytes,
      hostFreeReserveBytes: policy.hostFreeReserveBytes,
      staticFootprintFloorBytes: Math.min(raw.memory.admission.staticFootprintFloorBytes, policy.childFootprintCapBytes),
      estimateBytes: Math.min(raw.memory.admission.estimateBytes, policy.childFootprintCapBytes),
      hostMemoryComponents: hostMemoryFor(Math.min(raw.memory.admission.estimateBytes, policy.childFootprintCapBytes) + policy.hostFreeReserveBytes),
    };
  }
  raw.warmup.workerPid = pid;
  if (raw.warmup.required) {
    raw.warmup.suiteSha256 = numericSemanticSha256({
      probeEvidence: raw.timings.compileAttribution.probeEvidence,
      sessionId: raw.warmup.sessionId,
      workerPid: pid,
    });
  }
  raw.provenance = {
    ...raw.provenance,
    modelFilePath: model,
    modelFileBytes: 7,
    modelFileSha256: sha256("weights"),
    modelId: sealedModelId(raw.matrix.family, "candidate", sha256("weights")),
  };
  if (mode === "compressed") {
    for (const name of qualityFixtures) {
      raw.quality.fixtureEvidence[name].independentReference =
        sameWeightsFixtureReference(name, raw.provenance.modelFileSha256);
    }
  }
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
  // Derived v6 evidence follows the edit unless the edit set that field itself (a tamper).
  const prior = DERIVED_PATHS.map((keys) => structuredClone(getPath(raw, keys)));
  edit(raw);
  const derived = (() => { try { return derivedV6(raw); } catch { return null; } })();
  if (derived) {
    DERIVED_PATHS.forEach((keys, index) => {
      if (canonicalJson(getPath(raw, keys) ?? null) === canonicalJson(prior[index] ?? null)) setPath(raw, keys, derived[index]);
    });
  }
  return buildReceipt(raw);
}
function denseWithAllocatedCapacity(base, capacity, kvLength) {
  return rebuildReceipt(base, (raw) => {
    const physicalBytes = formula(raw.geometry.batch, capacity);
    raw.geometry.capacity = capacity;
    raw.geometry.kvLength = kvLength;
    raw.memory.persistentKvBytes = physicalBytes;
    raw.memory.denseTheoreticalKvBytes = physicalBytes;
    raw.memory.reconciliation.expectedDenseKvBytes = physicalBytes;
    raw.memory.reconciliation.observedPersistentKvBytes = physicalBytes;
    raw.memory.phaseSamples = memoryPhases(physicalBytes);
    for (const event of raw.memory.allocationEvents) {
      if (event.role === "cache") event.bytes = physicalBytes;
    }
  });
}
function withCampaignPid(base, pid) {
  return rebuildReceipt(base, (raw) => {
    raw.memory.phaseSamples = raw.memory.phaseSamples.map((sample) => ({ ...sample, pid }));
    raw.warmup.workerPid = pid;
    if (raw.warmup.required) {
      raw.warmup.suiteSha256 = numericSemanticSha256({
        probeEvidence: raw.timings.compileAttribution.probeEvidence,
        sessionId: raw.warmup.sessionId,
        workerPid: pid,
      });
    }
  });
}
const safetyPolicy = Object.freeze({
  schemaVersion: 1, rowDeadlineSeconds: 600, pollMillis: 250,
  termGraceMillis: 5000, hostFreeReserveBytes: 8_000_000_000,
  childFootprintCapBytes: 16_000_000_000, maxContextTokens: 131_072,
  maxRequestTokens: 131_072, stdoutCapBytes: 1_000_000, stderrCapBytes: 1_000_000,
});
async function writeCampaignManifest(directory, manifest) {
  const bytes = `${canonicalJson(manifest)}\n`;
  await writeFile(path.join(directory, "campaign.json"), bytes);
  await writeFile(path.join(directory, "campaign.json.sha256"), `${sha256(bytes)}  campaign.json\n`);
}
function sampleResumeIdentity(policy) {
  const candidate = { sha256: sha256("weights"), bytes: 7 };
  const reference = { sha256: "9".repeat(64), bytes: 2000 };
  return {
    schemaVersion: 1, kind: "sc-20671-resume-identity", scheduleVersion: 2,
    coordinates: SC20671_COVERING_SCHEDULE.map((row) => row.join("-")),
    inferenceRevision: "b".repeat(40), sceneWorksRevision: "a".repeat(40),
    executableSha256: "d".repeat(64), promptSha256: "c".repeat(64),
    policySha256: campaignPolicySha256(policy),
    llamaCandidate: candidate, qwenCandidate: candidate,
    llamaReference: reference, qwenReference: reference,
  };
}
async function writeEightCampaign(root, policy = safetyPolicy, { rowPolicy = policy, mode = "dense", continuations = {} } = {}) {
  const directory = path.join(root, "campaign");
  await mkdir(directory);
  const resumeIdentity = {
    ...sampleResumeIdentity(policy),
    ...(mode === "compressed" ? { mode, kvMethod: "group-affine" } : {}),
  };
  const resumeIdentitySha256 = campaignResumeIdentitySha256(resumeIdentity, campaignPolicySha256(policy));
  const rows = [];
  const receipts = [];
  for (const [index, entry] of SC20671_COVERING_SCHEDULE.entries()) {
    const coordinate = {
      family: entry[0], contextBand: entry[1], requestMode: entry[2],
      prefillMode: entry[3], processTemperature: entry[4],
    };
    const slug = entry.join("-");
    const receipt = await verifiedFixture(path.join(root, `source-${index}`), mode, coordinate, {
      pid: index + 100, inferenceShaped: true, policy: rowPolicy, continuation: continuations[index],
    });
    const receiptDirectory = path.join(directory, slug);
    await writeReceiptSet(receiptDirectory, receipt);
    for (let repeat = 1; repeat < 5; repeat += 1) {
      for (const fixture of qualityFixtures) {
        if (fixture === "kernel-fp32-reference" && coordinate.processTemperature === "cold") continue;
        const name = `fixtures/repeat-${repeat}/${fixture}.json`;
        const file = path.join(receiptDirectory, name);
        await mkdir(path.dirname(file), { recursive: true });
        const bytes = `${JSON.stringify(artifactFixture(receipt, fixture, repeat, { includeModel: true }))}\n`;
        await writeFile(file, bytes);
        await writeFile(`${file}.sha256`, `${sha256(bytes)}  ${name}\n`);
      }
    }
    const names = (await readdir(receiptDirectory, { recursive: true, withFileTypes: true }))
      .filter((entry) => entry.isFile() && !entry.name.endsWith(".sha256"))
      .map((entry) => path.relative(receiptDirectory, path.join(entry.parentPath, entry.name)))
      .sort();
    const files = await Promise.all(names.map(async (name) => ({
      name,
      sha256: sha256(await readFile(path.join(receiptDirectory, name))),
      sidecarSha256: sha256(await readFile(path.join(receiptDirectory, `${name}.sha256`))),
    })));
    rows.push({
      coordinate: slug, receiptSha256: receipt.receiptSha256, workerPid: index + 100, files,
      ...(receipt.quality.qualityGate ? { qualityGatePassed: receipt.quality.qualityGate.passed } : {}),
    });
    receipts.push(receipt);
  }
  const manifest = {
    schemaVersion: 2, kind: "sc-20671-complete-covering-set", scheduleVersion: 2,
    policySha256: campaignPolicySha256(policy), resumeIdentitySha256,
    hostStateVaried: campaignHostStateVaried(receipts),
    coordinates: rows,
    ...(mode === "compressed" ? { qualityGatePassed: campaignQualityGatePassed(receipts) } : {}),
  };
  await writeFile(path.join(directory, "safety-policy.json"), canonicalJson(policy));
  await writeFile(path.join(directory, "resume-identity.json"), canonicalJson(resumeIdentity));
  await writeFile(path.join(directory, "resume-identity.json.sha256"),
    `${resumeIdentitySha256}  resume-identity.json\n`);
  await writeCampaignManifest(directory, manifest);
  return { directory, manifest, resumeIdentity };
}
test("contract sidecar and valid comparison",async()=>{const raw=await readFile("config/kv-baseline-quality-contract.json");assert.equal(await readFile("config/kv-baseline-quality-contract.json.sha256","utf8"),`${sha256(raw)}  kv-baseline-quality-contract.json\n`);assert.equal(compareReceipts(fixture(),fixture("compressed")).persistentKvReduction,.25);});
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
  assert.equal(cold.schemaVersion,7);
  assert.equal(cold.harnessVersion,"sc-20671-kv-baseline-v7");
  const {probeEvidence,...coldAttribution}=cold.timings.compileAttribution;
  assert.deepEqual(coldAttribution,{
    method:"first-dispatch-minus-steady-v2",
    operation:"single-shot-generation",
    source:"measured-repeats",
    probeDurationsMs:[50,6,8,7,5],
    firstDispatchMs:50,
    steadyDispatchMs:6.5,
    firstDispatchExcessMs:43.5,
    noiseSamplesMs:[6,8,7,5],
    noiseBandMs:3,
    compileCostResolved:true,
    compileCostMs:43.5,
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
  assert.match(renderReceiptMarkdown(cold),/Compile attribution: first-dispatch-minus-steady-v2 \/ single-shot-generation \/ measured-repeats/);
  assert.match(renderReceiptMarkdown(cold),/Compile cost: 43.5 ms \(noise band 3 ms\)/);
});
test("compile cost below the steady noise band is recorded, not refused (W1 row 3)",()=>{
  // W1 row 3 (llama memory-material 32k warm): first 9017 ms, steady 10585 ms.
  const row3=rebuildReceipt(fixture("dense",{processTemperature:"warm"}),(raw)=>{
    const a=raw.timings.compileAttribution;
    a.probeDurationsMs=[9017.437917,10585.460208];
    a.probeEvidence.forEach((evidence,index)=>{evidence.dispatchMs=a.probeDurationsMs[index];});
    a.firstDispatchMs=9017.437917;a.steadyDispatchMs=10585.460208;a.firstDispatchExcessMs=9017.437917-10585.460208;
    a.noiseSamplesMs=[10510,10590,10555,10620,10575];a.noiseBandMs=110;
    a.compileCostResolved=false;delete a.compileCostMs;a.compileCostUnresolvedReason="first-dispatch-not-slower-than-steady";
    raw.timings.coldCompileMs=null;raw.timings.warmCompileMs=10585.460208;
    raw.warmup.suiteSha256=numericSemanticSha256({probeEvidence:a.probeEvidence,sessionId:raw.warmup.sessionId,workerPid:raw.warmup.workerPid});
  });
  assert.doesNotThrow(()=>validateReceipt(row3));
  assert.equal(row3.timings.coldCompileMs,null);
  assert.match(renderReceiptMarkdown(row3),/Compile cost: unresolved: first-dispatch-not-slower-than-steady \(noise band 110 ms\)/);
  const within=rebuildReceipt(fixture("dense"),(raw)=>{
    const a=raw.timings.compileAttribution;
    a.probeDurationsMs=[8,6,8,7,5];a.probeEvidence[0].dispatchMs=8;a.firstDispatchMs=8;a.firstDispatchExcessMs=1.5;
    a.compileCostResolved=false;delete a.compileCostMs;a.compileCostUnresolvedReason="excess-within-steady-noise-band";
    raw.timings.coldCompileMs=null;
  });
  assert.doesNotThrow(()=>validateReceipt(within));
  assert.throws(()=>rebuildReceipt(within,(raw)=>{raw.timings.coldCompileMs=1.5;}),/timing aliases/);
  // The retired v1 method (positive excess, no band) is refused.
  assert.throws(()=>rebuildReceipt(fixture("dense"),(raw)=>{
    const a=raw.timings.compileAttribution;
    a.method="first-dispatch-minus-steady-v1";
    for (const key of ["noiseSamplesMs","noiseBandMs","compileCostResolved","compileCostMs"]) delete a[key];
  }),/schema validation|frozen attribution method/);
  // Greedy agreement is the minimum over the recorded repeats.
  const weakest=rebuildReceipt(fixture(),(raw)=>{raw.quality.greedyTokenAgreementByRepeat=[1,0.9995,1,1,1];raw.quality.greedyTokenAgreement=0.9995;});
  assert.equal(weakest.quality.greedyTokenAgreement,0.9995);
  assert.throws(()=>rebuildReceipt(weakest,(raw)=>{raw.quality.greedyTokenAgreement=1;}),/minimum over the recorded repeats/);
  assert.throws(()=>rebuildReceipt(weakest,(raw)=>{raw.quality.greedyTokenAgreementByRepeat.pop();}),/schema validation|minimum over the recorded repeats/);
});
test("compile attribution v4 fails closed on malformed or tampered evidence",()=>{
  const rejects=(coordinate,edit,pattern=/schema validation|compile attribution|compile cost|noiseSamplesMs|probe durations|probeEvidence|matrix coordinate|process temperature|timing aliases|derived values|not sequential|not bound/)=>assert.throws(()=>rebuildReceipt(fixture("dense",coordinate),(raw)=>edit(raw.timings)),pattern);
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
  // An unresolved compile cost must be recorded as unresolved, never claimed or aliased.
  rejects({},(timings)=>{timings.compileAttribution.probeDurationsMs=[8,6,8,7,5];timings.compileAttribution.probeEvidence[0].dispatchMs=8;timings.compileAttribution.firstDispatchMs=8;timings.compileAttribution.firstDispatchExcessMs=1.5;timings.compileAttribution.compileCostMs=1.5;timings.coldCompileMs=1.5;});
  rejects({processTemperature:"warm"},(timings)=>{Object.assign(timings.compileAttribution,{compileCostResolved:false,compileCostUnresolvedReason:"first-dispatch-not-slower-than-steady"});delete timings.compileAttribution.compileCostMs;timings.coldCompileMs=null;});
  rejects({},(timings)=>{timings.coldCompileMs=null;});
  rejects({},(timings)=>{timings.compileAttribution.noiseBandMs+=1;});
  rejects({},(timings)=>{timings.compileAttribution.noiseSamplesMs[0]+=1;});
  rejects({},(timings)=>{delete timings.compileAttribution.noiseSamplesMs;});
  rejects({processTemperature:"warm"},(timings)=>{timings.compileAttribution.noiseSamplesMs.pop();});
  rejects({},(timings)=>{delete timings.compileAttribution.compileCostMs;});
  rejects({},(timings)=>{timings.compileAttribution.compileCostResolved=false;});
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
test("tamper, quality, formula and reconciliation fail closed",()=>{const r=fixture();r.memory.persistentKvBytes++;assert.throws(()=>validateReceipt(r),/receiptSha256/);assert.doesNotThrow(()=>fixture("dense",{}, {quality:{...fixture().quality,parityMaxError:1,perplexityDelta:1,greedyTokenAgreement:0,greedyTokenAgreementByRepeat:[1,0,1,1,1],multiTurnPromptCache:0}}));assert.throws(()=>fixture("compressed",{}, {quality:gatedQuality("compressed",{...fixture("compressed").quality,parityMaxError:1})}),/kernel parity failed: metric=parityMaxError value=1 threshold=0.0001/);assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,perplexityDelta:1}}),/quality gate does not record perplexityDelta repeat 0 = 1 against the frozen maximum 0.01/);assert.equal(fixture("compressed",{}, {quality:gatedQuality("compressed",{...fixture("compressed").quality,perplexityDelta:1})}).quality.qualityGate.passed,false);assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,greedyTokenAgreement:0,greedyTokenAgreementByRepeat:[1,0,1,1,1]}}),/forced-continuation agreement/);assert.throws(()=>fixture("dense",{}, {memory:{...fixture().memory,denseTheoreticalKvBytes:formula()*2}}),/does not reconcile/);assert.throws(()=>fixture("dense",{}, {memory:{...fixture().memory,reconciliation:{...fixture().memory.reconciliation,observedPersistentKvBytes:1}}}),/does not reconcile/);});
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
test("release, exact phase order, PID and sources fail closed",()=>{const base=fixture(),weightsLoaded=base.memory.phaseSamples[1],excessPhys=weightsLoaded.physFootprintBytes+POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES+1;assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===7?{...s,physFootprintBytes:excessPhys,physFootprintPeakBytes:excessPhys}:s)}}),/post-run footprint/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===7?{...s,physFootprintBytes:weightsLoaded.mlx.activeBytes+POST_RELEASE_MLX_SLACK_FLOOR_BYTES+1,physFootprintPeakBytes:Math.max(s.physFootprintPeakBytes,weightsLoaded.mlx.activeBytes+POST_RELEASE_MLX_SLACK_FLOOR_BYTES+1),mlx:{...s.mlx,activeBytes:weightsLoaded.mlx.activeBytes+POST_RELEASE_MLX_SLACK_FLOOR_BYTES+1}}:s)}}),/MLX allocator/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,release:{...base.memory.release,physFootprintToleranceBytes:POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES+1}}}),/tolerances or residuals/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,pid:10}:s)}}),/PID/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,phase:"decode-steady"}:s)}}),/phase must/);assert.throws(()=>fixture("dense",{}, {memory:{...base.memory,phaseSamples:base.memory.phaseSamples.map((s,i)=>i===3?{...s,source:"rss"}:s)}}),/source/);});
test("v5 decode throughput is the fixed-length steady decode; host power/thermal state brackets the row",()=>{
  const receipt=fixture();
  assert.deepEqual(receipt.timings.samples.map((s)=>[s.steadyDecodeGeneratedTokens,s.steadyDecodeTimedTokens]),Array(5).fill([256,255]));
  assert.deepEqual(receipt.provenance.hostStates.map((s)=>s.boundary),["row-start","row-end"]);
  const rejects=(edit,pattern)=>assert.throws(()=>rebuildReceipt(fixture(),edit),pattern);
  // Each sample's throughput is exactly its own recorded steady decode.
  rejects((raw)=>{raw.timings.samples[2].steadyDecodeMs*=2;},/does not derive from its steady decode/);
  // A short, EOS-terminated generation is not a steady-decode sample, even when self-consistent.
  rejects((raw)=>{const s=raw.timings.samples[2];s.steadyDecodeGeneratedTokens=24;s.steadyDecodeTimedTokens=23;s.decodeTokensPerSecond=23000/s.steadyDecodeMs;},/schema validation|fixed-length/);
  rejects((raw)=>{raw.timings.samples[2].steadyDecodePromptTokens=raw.geometry.contextWindowTokens;},/native context window/);
  rejects((raw)=>{delete raw.timings.samples[2].steadyDecodeForcedStopTokens;},/schema validation|steadyDecodeForcedStopTokens/);
  // Power mode and thermal state: recorded at row start, each timing sample and row end; only a
  // throttled row start refuses, and later changes are recorded as flags.
  const changed=rebuildReceipt(fixture(),(raw)=>{Object.assign(raw.provenance.hostStates[1],{thermalState:"serious",throttled:true,powerMode:"low-power"});});
  assert.equal(changed.provenance.thermalChangedDuringRow,true);
  assert.equal(changed.provenance.powerModeChangedDuringRow,true);
  const sampleChange=rebuildReceipt(fixture(),(raw)=>{raw.timings.samples[3].hostState.thermalState="fair";});
  assert.equal(sampleChange.provenance.thermalChangedDuringRow,true);
  assert.throws(()=>rebuildReceipt(rebuildReceipt(fixture(),(raw)=>{raw.provenance.hostStates[1].thermalState="fair";}),(raw)=>{raw.provenance.thermalChangedDuringRow=false;}),/change flags/);
  rejects((raw)=>{Object.assign(raw.provenance.hostStates[0],{thermalState:"critical",throttled:true});raw.provenance.thermalState="critical";},/schema validation|throttled/);
  rejects((raw)=>{Object.assign(raw.provenance.hostStates[0],{pmsetThermalRaw:"CPU_Speed_Limit \t= 80",cpuSpeedLimit:80,throttled:true});},/throttled/);
  rejects((raw)=>{raw.provenance.hostStates[1].throttled=true;},/recompute/);
  rejects((raw)=>{raw.timings.samples.reverse();},/ordered|derive/);
  rejects((raw)=>{delete raw.timings.samples[0].hostState;},/schema validation|hostState/);
  rejects((raw)=>{raw.provenance.hostStates.reverse();},/schema validation|boundary must be/);
  rejects((raw)=>{raw.provenance.hostStates.pop();},/schema validation|row start and row end/);
  rejects((raw)=>{raw.provenance.hostStates[0].capturedAt="2026-08-29T12:00:03.000Z";},/bracket/);
  rejects((raw)=>{raw.provenance.hostStates[1].capturedAt="2026-08-29T12:00:05.000Z";},/bracket|ordered/);
  rejects((raw)=>{raw.provenance.powerMode="AC";for(const s of raw.provenance.hostStates)s.powerMode="AC";},/schema validation|normalized energy mode/);
});
test("v6 records real-hardware observations instead of refusing the row",()=>{
  // pmset: unknown note lines are tolerated and recorded raw; only CPU_Speed_Limit is read.
  assert.equal(pmsetCpuSpeedLimit(PMSET_NOMINAL+"\nNote: something new"),null);
  assert.equal(pmsetCpuSpeedLimit("CPU_Scheduler_Limit \t= 100\nCPU_Speed_Limit \t= 70"),70);
  assert.throws(()=>pmsetCpuSpeedLimit("CPU_Speed_Limit = fast"),/not a number/);
  assert.throws(()=>pmsetCpuSpeedLimit("CPU_Speed_Limit = 100\nCPU_Speed_Limit = 90"),/contradictory/);
  assert.equal(hostStateThrottled("fair",100),false);
  assert.equal(hostStateThrottled("nominal",99),true);
  assert.equal(hostStateThrottled("serious",null),true);
  // Post-release MLX slack: max(1 MiB, 0.1% of baseline); residuals are recorded.
  assert.equal(postReleaseMlxSlackBytes(3),POST_RELEASE_MLX_SLACK_FLOOR_BYTES);
  assert.equal(postReleaseMlxSlackBytes(4_000_000_001),4_000_001);
  const residual=rebuildReceipt(fixture(),(raw)=>{const end=raw.memory.phaseSamples[7];end.mlx.activeBytes=raw.memory.phaseSamples[1].mlx.activeBytes+1000;end.physFootprintBytes=Math.max(end.physFootprintBytes,end.mlx.activeBytes);end.physFootprintPeakBytes=Math.max(end.physFootprintPeakBytes,end.physFootprintBytes);});
  assert.equal(residual.memory.release.mlxActiveResidualBytes,1000);
  assert.throws(()=>rebuildReceipt(residual,(raw)=>{raw.memory.release.mlxActiveResidualBytes=0;}),/residuals/);
  // Dense-KV share of the prefill footprint: recorded and flagged, never refused.
  const low=rebuildReceipt(fixture("dense",{contextBand:"memory-material"}),(raw)=>{const footprint=raw.memory.denseTheoreticalKvBytes*20;raw.memory.phaseSamples[2].physFootprintBytes=footprint;for(const sample of raw.memory.phaseSamples.slice(2))sample.physFootprintPeakBytes=Math.max(sample.physFootprintPeakBytes,footprint);});
  assert.equal(low.memory.belowMemoryMaterialShare,true);
  assert.equal(low.memory.denseKvShareBps,500);
  assert.throws(()=>rebuildReceipt(low,(raw)=>{raw.memory.belowMemoryMaterialShare=false;}),/dense KV share/);
  // Greedy agreement is teacher-forced; free-running divergence is an observation.
  const diverged=rebuildReceipt(fixture(),(raw)=>{raw.quality.freeRunningFirstDivergence=3;});
  assert.equal(diverged.quality.freeRunningFirstDivergence,3);
  assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.quality.greedyAgreementMethod="free-running";}),/schema validation|teacher-forced/);
  // Campaign host-state variation is a recorded flag, not an identity drift.
  const base=fixture(),lowPower=rebuildReceipt(fixture(),(raw)=>{raw.provenance.powerMode="low-power";for(const state of [...raw.provenance.hostStates,...raw.timings.samples.map((s)=>s.hostState)])state.powerMode="low-power";});
  assert.equal(campaignHostStateVaried([base,base]),false);
  assert.equal(campaignHostStateVaried([base,lowPower]),true);
  assert.equal(campaignHostStateVaried([changedFlag(base)]),true);
});
function changedFlag(receipt){return {...receipt,provenance:{...receipt.provenance,thermalChangedDuringRow:true}};}
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
test("contract v4 gates compressed quality only against the same-weights dense-KV run",()=>{
  const contract=JSON.parse(readFileSync("config/kv-baseline-quality-contract.json","utf8"));
  assert.equal(contract.version,4);
  assert.equal(contract.gate.compressedReference,"dense-kv-same-weights");
  // v4 changed only multiTurnPromptCache (now teacher-forced on a cache-hit turn 2, aligned with
  // greedy agreement); every other threshold is v3's.
  assert.deepEqual(contract.thresholds,{parityMaxError:0.0001,perplexityDelta:0.01,greedyTokenAgreement:0.999,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:0.999});
  assert.equal(contract.multiTurnFixture.followUp,"Now repeat that same stable baseline fact once more, in one short sentence.");
  assert.throws(()=>checkContract({...contract,version:3}),/unsupported quality contract version/);
  assert.throws(()=>checkContract({...contract,multiTurnFixture:{...contract.multiTurnFixture,metric:""}}),/multiTurnFixture.metric/);
  assert.doesNotMatch(JSON.stringify(contract.needleFixture),/passphrase/i);
  assert.doesNotThrow(()=>checkContract(contract));
  assert.throws(()=>checkContract({...contract,gate:{...contract.gate,compressedReference:"bf16-model"}}),/same-weights dense-KV/);
  assert.throws(()=>checkContract({...contract,needleFixture:{...contract.needleFixture,statement:"The special magic identifier."}}),/exact needle token/);
  assert.throws(()=>checkContract({...contract,version:2}),/unsupported quality contract version/);
  // A compressed receipt whose denominator is not the same-weights dense-KV run is refused.
  assert.throws(()=>fixture("compressed",{}, {quality:gatedQuality("compressed",{...fixture().quality,needleDiscriminating:true})}),/contract v3 denominator/);
  // A measured needle miss is a recorded gate failure, never an unrecorded or refused row.
  assert.throws(()=>fixture("compressed",{}, {quality:{...fixture("compressed").quality,needleRetrieval:0}}),/quality gate does not record needleRetrieval repeat 0 = 0/);
  assert.equal(fixture("compressed",{}, {quality:gatedQuality("compressed",{...fixture("compressed").quality,needleRetrieval:0})}).quality.qualityGate.passed,false);
  // A shared dense miss is accepted only when flagged non-discriminating, and the comparison says so.
  const sharedMiss=fixture("compressed",{}, {quality:{...fixture("compressed").quality,needleDiscriminating:false}});
  const comparison=compareReceipts(fixture(),sharedMiss);
  assert.equal(comparison.quality.needleDiscriminating,false);
  assert.match(renderComparisonMarkdown(comparison),/NON-DISCRIMINATING/);
  assert.doesNotMatch(renderComparisonMarkdown(compareReceipts(fixture(),fixture("compressed"))),/NON-DISCRIMINATING/);
  // Dense rows are characterization: a needle/tool miss is recorded, never rejected, but not hidden.
  assert.doesNotThrow(()=>fixture("dense",{}, {quality:{...fixture().quality,needleRetrieval:0,needleDiscriminating:false,structuredToolAgreement:0}}));
  assert.throws(()=>fixture("dense",{}, {quality:{...fixture().quality,needleRetrieval:0}}),/needle discrimination/);
  const { qualityGate: _gate, forcedContinuation: _forced, multiTurnForcedContinuation: _turn2, multiTurnForcedPass: _pass, ...compressedQuality } = fixture("compressed").quality;
  assert.throws(()=>fixture("dense",{}, {quality:compressedQuality}),/contract v3 denominator/);
  // A dense row is characterization: it can carry neither a gate nor a forced continuation.
  assert.throws(()=>fixture("dense",{}, {quality:{...fixture().quality,qualityGate:{passed:true,failures:[]}}}),/schema validation failed/);
});
test("rows record runtime-guarded admission with the stated cap and estimate",()=>{
  const memory=fixture().memory;
  assert.equal(memory.admission.mode,"runtime-guarded");
  assert.throws(()=>fixture("dense",{}, {memory:{...memory,admission:{...memory.admission,mode:"static-proof"}}}),/schema validation|runtime-guarded/);
  assert.throws(()=>fixture("dense",{}, {memory:{...memory,admission:{...memory.admission,staticFootprintFloorBytes:memory.admission.childFootprintCapBytes+1}}}),/exceeds the stated child cap/);
  const {admission,...unadmitted}=memory;
  assert.ok(admission);
  assert.throws(()=>fixture("dense",{}, {memory:unadmitted}),/schema validation|memory/);
});
test("rows record the estimate-plus-reserve decision and refuse records lacking it",()=>{
  const memory=fixture().memory,admission=memory.admission;
  assert.equal(admission.rule,"estimate-plus-reserve-v1");
  const withAdmission=(changes)=>({memory:{...memory,admission:{...admission,...changes}}});
  for (const field of ["rule","estimateSource","estimateBytes"]) {
    const {[field]:_omitted,...missing}=admission;
    assert.throws(()=>fixture("dense",{}, {memory:{...memory,admission:missing}}),/schema validation|memory\.admission/,field);
  }
  for (const changes of [
    {rule:"cap-plus-reserve"},
    {estimateSource:""},
    {estimateBytes:admission.staticFootprintFloorBytes-1},
    {estimateBytes:admission.childFootprintCapBytes+1},
    // the cap fallback must be the cap itself
    {estimateSource:"child-footprint-cap-fallback"},
  ]) assert.throws(()=>fixture("dense",{}, withAdmission(changes)),/schema validation|memory\.admission/,JSON.stringify(changes));
  // Exactly estimate plus reserve is admitted; one page less is not, although both are far below
  // the former cap plus reserve. The cap fallback needs cap plus reserve.
  const required=admission.estimateBytes+admission.hostFreeReserveBytes;
  fixture("dense",{}, withAdmission({hostMemoryComponents:hostMemoryFor(required)}));
  assert.ok(hostMemoryFor(required).availableBytes<admission.childFootprintCapBytes+admission.hostFreeReserveBytes);
  assert.throws(()=>fixture("dense",{}, withAdmission({hostMemoryComponents:hostMemoryFor(required-16384)})),/below reserve plus estimate/);
  const fallback={estimateSource:"child-footprint-cap-fallback",estimateBytes:admission.childFootprintCapBytes};
  assert.throws(()=>fixture("dense",{}, withAdmission({...fallback,hostMemoryComponents:hostMemoryFor(required)})),/below reserve plus estimate/);
  fixture("dense",{}, withAdmission({...fallback,hostMemoryComponents:hostMemoryFor(admission.childFootprintCapBytes+admission.hostFreeReserveBytes)}));
});
test("admission records the macOS host measurement, which must recompute and cover estimate plus reserve",()=>{
  const memory=fixture().memory,host=memory.admission.hostMemoryComponents;
  assert.equal(host.metric,"darwin-vm-stat-available-v3");
  const withHost=(changes)=>({memory:{...memory,admission:{...memory.admission,hostMemoryComponents:{...host,...changes}}}});
  // Heavy clean file cache: inactive file-backed pages are credited.
  const cached=hostMemoryFor(2**31,{freePages:1000,speculativePages:500,purgeablePages:0,inactivePages:200000,fileBackedPages:200500,anonymousPages:2000,throttledPages:0,activePages:2000});
  assert.equal(cached.reclaimableFilePages,200000);
  assert.ok((cached.freePages+cached.speculativePages)*cached.pageSizeBytes<2**31);
  fixture("dense",{}, {memory:{...memory,admission:{...memory.admission,hostMemoryComponents:cached}}});
  const {hostMemoryComponents:_omitted,...unmeasured}=memory.admission;
  assert.throws(()=>fixture("dense",{}, {memory:{...memory,admission:unmeasured}}),/schema validation|hostMemoryComponents/);
  // Tampered derived values, a foreign metric, and the plausible wrong definitions all fail.
  for (const changes of [
    {availableBytes:host.availableBytes+host.pageSizeBytes},
    {metric:"darwin-vm-stat-available-v2"},
    // anonymous inactive pages credited instead of the (empty) file cache
    {inactivePages:40000,anonymousPages:40000,reclaimableFilePages:40000,availableBytes:host.availableBytes+40000*host.pageSizeBytes},
    {pageSizeBytes:12288},
    // min with inactive (v1): active file cache left uncredited
    {inactivePages:10,fileBackedPages:100,activePages:90,reclaimableFilePages:10,availableBytes:host.availableBytes+10*host.pageSizeBytes},
    // speculative counted twice: file-backed not reduced by the speculative pages inside it
    {speculativePages:40,inactivePages:100,fileBackedPages:100,reclaimableFilePages:100,availableBytes:host.availableBytes+140*host.pageSizeBytes},
  ]) assert.throws(()=>fixture("dense",{}, withHost(changes)),/schema validation|hostMemoryComponents/);
  // A measurement below estimate plus reserve is not an admitted row.
  const short=hostMemoryFor(memory.admission.estimateBytes+memory.admission.hostFreeReserveBytes-16384);
  assert.throws(()=>fixture("dense",{}, {memory:{...memory,admission:{...memory.admission,hostMemoryComponents:short}}}),/below reserve plus estimate/);
});
test("fixture outcomes are bound to the same weights and re-derived, and flags AND across repeats",()=>{
  const dense=fixture(),compressed=fixture("compressed");
  const needle=(candidateRecovered,referenceRecovered,outputsMatch,matches,discriminating)=>({evidence:{matches,total:1,candidateRecovered,referenceRecovered,outputsMatch,discriminating}});
  const tool=(candidateValid,referenceValid,outputsMatch,discriminating)=>({evidence:{matches:outputsMatch?1:0,total:1,candidateValid,referenceValid,outputsMatch,discriminating}});
  const sameWeights={binding:{reference:{coordinateInventorySha256:compressed.provenance.modelFileSha256,qualityInventorySha256:compressed.provenance.modelFileSha256}}};
  const bf16={binding:{reference:{coordinateInventorySha256:compressed.provenance.referenceModelSha256,qualityInventorySha256:compressed.provenance.referenceModelSha256}}};
  // Compressed denominators are bound, not labelled.
  assert.equal(validateFixtureOutcomes({...sameWeights,...needle(true,true,true,1,true)},"long-context-needle",compressed),true);
  assert.throws(()=>validateFixtureOutcomes({...bf16,...needle(true,true,true,1,true)},"long-context-needle",compressed),/same weights/);
  for (const key of ["coordinateInventorySha256","qualityInventorySha256"]) {
    const mixed={binding:{reference:{...sameWeights.binding.reference,[key]:compressed.provenance.referenceModelSha256}}};
    assert.throws(()=>validateFixtureOutcomes({...mixed,...needle(true,true,true,1,true)},"long-context-needle",compressed),/same weights/,key);
  }
  assert.throws(()=>validateFixtureOutcomes({evidence:{}},"multi-turn-prompt-cache",compressed),/binding/);
  assert.equal(validateFixtureOutcomes({...sameWeights,...needle(false,false,true,1,false)},"long-context-needle",compressed),false);
  assert.equal(validateFixtureOutcomes({...sameWeights,...tool(false,false,true,false)},"structured-tool-call",compressed),false);
  // A validly measured compressed miss is quality-gate evidence, not a malformed artifact.
  assert.equal(validateFixtureOutcomes({...sameWeights,...needle(false,true,false,0,true)},"long-context-needle",compressed),true);
  assert.equal(validateFixtureOutcomes({...sameWeights,...tool(true,true,false,true)},"structured-tool-call",compressed),true);
  for (const [fixture,forged] of [
    ["long-context-needle",needle(false,false,false,1,false)],
    ["long-context-needle",needle(true,true,true,1,false)],
    ["long-context-needle",needle(false,true,false,1,true)],
    ["structured-tool-call",tool(true,false,true,true)],
  ]) assert.throws(()=>validateFixtureOutcomes({...sameWeights,...forged},fixture,compressed),/does not derive/,fixture);
  // Dense rows derive discrimination from their own run and are never gated on outcomes.
  assert.equal(validateFixtureOutcomes(needle(false,true,false,0,false),"long-context-needle",dense),false);
  assert.equal(validateFixtureOutcomes(tool(false,true,false,false),"structured-tool-call",dense),false);
  assert.throws(()=>validateFixtureOutcomes(needle(false,true,false,0,true),"long-context-needle",dense),/does not derive/);
  // AND across all repeats in a campaign; primary-only receipt sets may not over-claim.
  validateRepeatDiscrimination(dense,{"long-context-needle":[true,true],"structured-tool-call":[true]});
  assert.throws(()=>validateRepeatDiscrimination(dense,{"long-context-needle":[true,false],"structured-tool-call":[true]}),/needleDiscriminating is not the AND/);
  assert.throws(()=>validateRepeatDiscrimination(dense,{"long-context-needle":[true],"structured-tool-call":[true,false]}),/toolDiscriminating is not the AND/);
  const primary={"kernel-fp32-reference":{},"multi-turn-prompt-cache":{},"long-context-needle":needle(false,true,false,0,false),"structured-tool-call":tool(true,true,true,true)};
  assert.throws(()=>validatePrimaryDiscrimination(dense,primary),/needleDiscriminating is not the AND/);
  validatePrimaryDiscrimination(fixture("dense",{}, {quality:{...dense.quality,needleRetrieval:0,needleDiscriminating:false}}),primary);
  // Admission must carry the captured policy's cap and reserve.
  const policy={childFootprintCapBytes:dense.memory.admission.childFootprintCapBytes,hostFreeReserveBytes:dense.memory.admission.hostFreeReserveBytes};
  validateAdmissionPolicy(dense,policy);
  assert.throws(()=>validateAdmissionPolicy(dense,{...policy,childFootprintCapBytes:policy.childFootprintCapBytes+1}),/captured safety policy/);
  assert.throws(()=>validateAdmissionPolicy(dense,{...policy,hostFreeReserveBytes:policy.hostFreeReserveBytes+1}),/captured safety policy/);
  // The comparison flags a non-discriminating tool check.
  const sharedTool=fixture("compressed",{}, {quality:{...compressed.quality,toolDiscriminating:false}});
  assert.match(renderComparisonMarkdown(compareReceipts(dense,sharedTool)),/Tool check: NON-DISCRIMINATING/);
});
test("fixture artifacts record both arms' tool and needle outcomes as booleans",()=>{
  const row={independentReference:"ref"},metrics={parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:1,structuredToolAgreement:0,needleRetrieval:0,multiTurnPromptCache:1};
  const needle={matches:0,total:1,candidateRecovered:false,referenceRecovered:true,outputsMatch:false,discriminating:false};
  assert.doesNotThrow(()=>validateFixtureArtifact({fixture:"long-context-needle",independentReference:"ref",evidence:needle,metrics},"long-context-needle",row));
  assert.throws(()=>validateFixtureArtifact({fixture:"long-context-needle",independentReference:"ref",evidence:{matches:0,total:1},metrics},"long-context-needle",row),/must be boolean/);
  assert.throws(()=>validateFixtureArtifact({fixture:"long-context-needle",independentReference:"ref",evidence:{...needle,referenceRecovered:1},metrics},"long-context-needle",row),/must be boolean/);
  assert.doesNotThrow(()=>validateFixtureArtifact({fixture:"structured-tool-call",independentReference:"ref",evidence:{matches:0,total:1,candidateValid:false,referenceValid:true,outputsMatch:false,discriminating:false},metrics},"structured-tool-call",row));
});
test("fixture artifacts accept zero raw matches but reject aggregate or wrong-shape evidence",()=>{const cacheEvidence={matches:1,total:1,method:MULTI_TURN_PROMPT_CACHE_METHOD,freeRunningFirstDivergence:null,matchedPrefixTokens:0,turns:{candidate:MULTI_TURN_TURNS,reference:MULTI_TURN_TURNS}};const row={independentReference:"ref"},metrics={parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:0,structuredToolAgreement:1,needleRetrieval:1,multiTurnPromptCache:0};assert.doesNotThrow(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{...cacheEvidence,matches:0,total:8},metrics},"multi-turn-prompt-cache",row));assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{...cacheEvidence,matches:-1,total:8},metrics},"multi-turn-prompt-cache",row),/non-negative/);assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{...cacheEvidence,matches:0,total:0},metrics},"multi-turn-prompt-cache",row),/positive/);assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{matches:0,total:8},metrics},"multi-turn-prompt-cache",row),/lacks method/);assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{...cacheEvidence,method:"free-running"},metrics},"multi-turn-prompt-cache",row),/not teacher-forced on a cache-hit turn 2/);assert.throws(()=>validateFixtureArtifact({fixture:"multi-turn-prompt-cache",independentReference:"ref",evidence:{...cacheEvidence,turns:{...cacheEvidence.turns,reference:{...MULTI_TURN_TURNS,turn2:{...MULTI_TURN_TURNS.turn2,cacheHit:false,reusedPrefixTokens:0}}}},metrics},"multi-turn-prompt-cache",row),/did not serve turn 2/);assert.throws(()=>validateFixtureArtifact({fixture:"structured-tool-call",independentReference:"ref",evidence:{matches:1},metrics:{}},"structured-tool-call",row),/must be finite/);assert.throws(()=>validateFixtureArtifact({fixture:"long-context-needle",independentReference:"other",evidence:{matches:1,total:1},metrics:{...metrics,greedyTokenAgreement:1,multiTurnPromptCache:1}},"long-context-needle",row),/reference mismatch/);});
test("sharded model identity covers every resolved snapshot file",async()=>{const dir=await mkdtemp("/tmp/kv20671-snapshot-"),snapshot=path.join(dir,"snapshot");await mkdir(path.join(snapshot,"nested"),{recursive:true});await writeFile(path.join(snapshot,"config.json"),"config");await writeFile(path.join(snapshot,"nested","model-00001-of-00002.safetensors"),"first");await writeFile(path.join(snapshot,"nested","model-00002-of-00002.safetensors"),"second");const first=await inventoryModelArtifact(snapshot);assert.equal(first.bytes,17);assert.equal(first.files,3);await writeFile(path.join(snapshot,"nested","model-00002-of-00002.safetensors"),"changed");const second=await inventoryModelArtifact(snapshot);assert.notEqual(first.sha256,second.sha256);await writeFile(path.join(snapshot,"empty.safetensors"),"");await assert.rejects(inventoryModelArtifact(snapshot),/empty or unsupported/);await rm(dir,{recursive:true,force:true});});
test("receipt sets reject mixed generations and CLI comparison preserves inputs",async()=>{const dir=await mkdtemp("/tmp/kv20671-"),dense=path.join(dir,"dense"),compressed=path.join(dir,"compressed"),out=path.join(dir,"comparison.json");const denseReceipt=await verifiedFixture(path.join(dir,"dense-source")),compressedReceipt=await verifiedFixture(path.join(dir,"compressed-source"),"compressed");await writeReceiptSet(dense,denseReceipt);await writeReceiptSet(compressed,compressedReceipt);const before=await readFile(path.join(compressed,"receipt.json"),"utf8");await run(process.execPath,["scripts/kv-baseline-harness.mjs","compare",dense,compressed,out]);assert.equal(await readFile(path.join(compressed,"receipt.json"),"utf8"),before);assert.match(await readFile(path.join(dir,"comparison.md"),"utf8"),/Persistent KV reduction: 25.00%/);await writeFile(path.join(dense,"receipt.md"),"# forged\n");await assert.rejects(readReceiptSet(dense),/sidecar hash mismatch|not bound/);await rm(dir,{recursive:true,force:true});});
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
test("volatile sessions do not change stable model identity while invariant drift fails",()=>{const rows=[];for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"]){const row=fixture("dense",{family,contextBand,requestMode,prefillMode,processTemperature});rows.push(rebuildReceipt(row,(raw)=>{raw.provenance.campaignSessionId=sha256(raw.runId);if(raw.warmup.required){raw.warmup.sessionId=raw.provenance.campaignSessionId;raw.warmup.suiteSha256=numericSemanticSha256({probeEvidence:raw.timings.compileAttribution.probeEvidence,sessionId:raw.warmup.sessionId,workerPid:raw.warmup.workerPid});}}));}rows.forEach((row,index)=>{rows[index]=withCampaignPid(row,index+10);});assert.equal(validateCampaign(rows, { scheduleVersion: 1 }).coordinates,64);assert.equal(compareReceipts(rows[0],rebuildReceipt(fixture("compressed",rows[0].matrix),(raw)=>{raw.provenance.campaignSessionId="7".repeat(64);raw.provenance.modelId=rows[0].provenance.modelId;raw.provenance.referenceModelId=rows[0].provenance.referenceModelId;})).persistentKvReduction,.25);const refDrift=[...rows];refDrift[1]=rebuildReceipt(refDrift[1],(raw)=>{raw.provenance.referenceModelSha256="8".repeat(64);raw.provenance.referenceModelId=sealedModelId(raw.matrix.family,"reference",raw.provenance.referenceModelSha256);});assert.throws(()=>validateCampaign(refDrift, { scheduleVersion: 1 }),/model identity drift/);const sourceDrift=[...rows];sourceDrift[1]=rebuildReceipt(sourceDrift[1],(raw)=>{raw.provenance.inferenceRevision="8".repeat(40);});assert.throws(()=>validateCampaign(sourceDrift, { scheduleVersion: 1 }),/identity drift/);const powerChange=[...rows];powerChange[1]=rebuildReceipt(powerChange[1],(raw)=>{raw.provenance.powerMode="low-power";for(const state of [...raw.provenance.hostStates,...raw.timings.samples.map((sample)=>sample.hostState)])state.powerMode="low-power";});assert.equal(validateCampaign(powerChange, { scheduleVersion: 1 }).coordinates,64,"a different power mode is recorded, not an identity drift");assert.equal(campaignHostStateVaried(powerChange),true);assert.throws(()=>rebuildReceipt(rows[0],(raw)=>{raw.provenance.mlxSource=`git+https://github.com/fork/mlx-rs?rev=${raw.provenance.mlxRevision}#${raw.provenance.mlxRevision}`;}),/exact Git revision/);});
test("measured context bands reject zero, non-material, and non-boundary receipts",()=>{assert.throws(()=>rebuildReceipt(fixture(),(raw)=>{raw.geometry.contextWindowTokens=0;}),/must be >= 1|positive integer/);assert.equal(rebuildReceipt(fixture("dense",{contextBand:"memory-material"}),(raw)=>{raw.memory.phaseSamples=raw.memory.phaseSamples.map((sample)=>({...sample,physFootprintBytes:10_000_000_000,physFootprintPeakBytes:10_000_000_000}));}).memory.belowMemoryMaterialShare,true,"a non-material share is recorded, not refused");assert.throws(()=>rebuildReceipt(fixture("dense",{contextBand:"fit-boundary"}),(raw)=>{raw.geometry.contextWindowTokens=8192;raw.geometry.contextTargetTokens=7680;raw.geometry.contextPayloadTokens=7680;}),/fit-boundary|sealed llama contract/);});
test("dense receipts distinguish live KV length from observed allocated capacity", () => {
  const padded = denseWithAllocatedCapacity(fixture(), 256, 32);
  assert.equal(padded.geometry.kvLength, 32);
  assert.equal(padded.geometry.capacity, 256);
  assert.equal(padded.memory.persistentKvBytes, formula(1, 256));
  assert.doesNotThrow(() => validateReceipt(padded));
  // A restored or trimmed cache may retain more allocation than its live context.
  const boundary = fixture("dense", { family: "qwen", contextBand: "fit-boundary" });
  assert.doesNotThrow(() => validateReceipt(denseWithAllocatedCapacity(
    boundary, boundary.geometry.contextWindowTokens + 1, boundary.geometry.kvLength,
  )));
  assert.throws(() => rebuildReceipt(boundary, (raw) => {
    raw.geometry.kvLength = Math.floor(raw.geometry.contextWindowTokens * 0.89);
  }), /fit-boundary live context/);
  // Existing v4 receipts with exact-length capacity remain readable.
  assert.doesNotThrow(() => validateReceipt(boundary));
  assert.throws(() => rebuildReceipt(padded, (raw) => {
    raw.geometry.capacity = raw.geometry.kvLength - 1;
  }), /capacity is below kvLength/);
});
test("dense physical KV bytes reconcile exactly without padding tolerance", () => {
  const padded = denseWithAllocatedCapacity(fixture(), 256, 32);
  assert.throws(() => rebuildReceipt(padded, (raw) => {
    raw.memory.reconciliation.toleranceBytes = formula(1, 224);
  }), /tolerance must be zero/);
  assert.throws(() => rebuildReceipt(padded, (raw) => {
    raw.memory.persistentKvBytes -= 1;
    raw.memory.reconciliation.observedPersistentKvBytes -= 1;
    for (const event of raw.memory.allocationEvents) {
      if (event.role === "cache") event.bytes -= 1;
    }
  }), /do not equal allocated capacity bytes/);
});
test("eight-row covering campaign requires the exact family, band, and process schedule", () => {
  const rows = SC20671_COVERING_SCHEDULE.map((entry, index) => withCampaignPid(fixture("dense", {
    family: entry[0], contextBand: entry[1], requestMode: entry[2],
    prefillMode: entry[3], processTemperature: entry[4],
  }), index + 100));
  assert.equal(validateCampaign(rows).coordinates, 8);
  assert.equal(validateCampaign(rows).scheduleVersion, 2);
  assert.throws(() => validateCampaign(rows.slice(1)), /incomplete/);
  assert.throws(() => validateCampaign([...rows.slice(0, 7), rows[0]]), /duplicate/);
  const wrongSchedule = [...rows];
  wrongSchedule[0] = withCampaignPid(fixture("dense", {
    family: "llama", contextBand: "short", requestMode: "single",
    prefillMode: "single-shot", processTemperature: "cold",
  }), 999);
  assert.throws(() => validateCampaign(wrongSchedule), /frozen schedule/);
  const reusedColdPid = [...rows];
  reusedColdPid[3] = withCampaignPid(rows[3], rows[0].memory.phaseSamples[0].pid);
  assert.throws(() => validateCampaign(reusedColdPid), /reused a worker PID/);
  const sourceDrift = [...rows];
  sourceDrift[1] = rebuildReceipt(rows[1], (raw) => {
    raw.provenance.inferenceRevision = "8".repeat(40);
  });
  assert.throws(() => validateCampaign(sourceDrift), /identity drift/);
  const familyDrift = [...rows];
  familyDrift[2] = rebuildReceipt(rows[2], (raw) => {
    raw.provenance.referenceModelSha256 = "8".repeat(64);
    raw.provenance.referenceModelId = sealedModelId("llama", "reference", "8".repeat(64));
  });
  assert.throws(() => validateCampaign(familyDrift), /model identity drift/);
});
test("campaign policy and resume identity seals match the Rust canonical parity fixtures", async () => {
  const policyBytes = await readFile("scripts/fixtures/sc20671-safety-policy-parity.json", "utf8");
  const identityBytes = await readFile("scripts/fixtures/sc20671-resume-identity-parity.json", "utf8");
  const policy = JSON.parse(policyBytes);
  const identity = JSON.parse(identityBytes);
  const policySeal = "03b18fb4b6e8e729189ad1243fecc31934ff1b4aa011cdf00fb6489029a17d2a";
  const identitySeal = "a9bcc30bc11a2c86a4057489e1e0c9bc1519f0802b6f576200562406a8d02112";
  assert.equal(policyBytes, canonicalJson(policy));
  assert.equal(identityBytes, canonicalJson(identity));
  assert.equal(campaignPolicySha256(policy), policySeal);
  assert.equal(campaignResumeIdentitySha256(identity, policySeal), identitySeal);
  assert.throws(() => campaignPolicySha256({ ...policy, maxContextTokens: 0 }), /positive safe integer/);
  assert.throws(() => campaignPolicySha256({ ...policy, maxContextTokens: Number.MAX_SAFE_INTEGER + 1 }), /positive safe integer/);
  assert.throws(() => campaignPolicySha256({ ...policy, pollMillis: 10_000 }), /below row deadline/);
  assert.throws(() => campaignResumeIdentitySha256({ ...identity, coordinates: identity.coordinates.slice(1) }, policySeal), /schedule/);
});
test("v2 campaign reader binds every row and artifact to trusted policy and resume inputs", async () => {
  const root = await mkdtemp("/tmp/kv20671-covering-");
  try {
    const { directory, manifest, resumeIdentity } = await writeEightCampaign(root, safetyPolicy);
    const trusted = { safetyPolicy, resumeIdentity };
    assert.equal((await readCampaignSet(directory, trusted)).summary.coordinates, 8);
    // Host-state variation across the campaign is a recorded flag that must recompute.
    await writeCampaignManifest(directory, { ...manifest, hostStateVaried: !manifest.hostStateVaried });
    await assert.rejects(readCampaignSet(directory, trusted), /host-state variation flag/);
    await writeCampaignManifest(directory, manifest);
    // An inference `--only-coordinate` run (kind sc-20671-partial-coordinate-run, marked partial
    // and non-publishable, one row) is never read as a campaign.
    const only = manifest.coordinates[0].coordinate;
    await writeCampaignManifest(directory, {
      ...manifest, kind: "sc-20671-partial-coordinate-run", partial: true, publishable: false,
      onlyCoordinate: only, coordinates: [manifest.coordinates[0]],
    });
    await assert.rejects(readCampaignSet(directory, trusted), /campaign manifest/);
    await writeCampaignManifest(directory, { ...manifest, kind: "sc-20671-partial-coordinate-run" });
    await assert.rejects(readCampaignSet(directory, trusted), /mismatched kind/);
    await writeCampaignManifest(directory, manifest);
    // Rows that ran under a different cap than the captured policy are refused.
    await mkdir(path.join(root, "drifted"));
    const drifted = await writeEightCampaign(path.join(root, "drifted"), safetyPolicy, {
      rowPolicy: { ...safetyPolicy, childFootprintCapBytes: safetyPolicy.childFootprintCapBytes + 1 },
    });
    await assert.rejects(
      readCampaignSet(drifted.directory, { safetyPolicy, resumeIdentity: drifted.resumeIdentity }),
      /captured safety policy/,
    );
    const policyFile = path.join(root, "policy.json");
    const trustedResume = path.join(root, "resume");
    const summaryFile = path.join(root, "summary.json");
    await mkdir(trustedResume);
    await writeFile(policyFile, canonicalJson(safetyPolicy));
    await writeFile(path.join(trustedResume, "identity.json"), canonicalJson(resumeIdentity));
    await writeFile(path.join(trustedResume, "identity.json.sha256"),
      `${manifest.resumeIdentitySha256}  identity.json\n`);
    await run(process.execPath, ["scripts/kv-baseline-harness.mjs", "campaign", directory,
      summaryFile, "--safety-policy", policyFile, "--resume-identity",
      path.join(trustedResume, "identity.json")]);
    assert.equal(JSON.parse(await readFile(summaryFile, "utf8")).coordinates, 8);
    await assert.rejects(run(process.execPath, ["scripts/kv-baseline-harness.mjs", "campaign",
      directory, path.join(root, "missing-policy-summary.json")]), /usage:/);
    await assert.rejects(readCampaignSet(directory), /requires a safety policy/);
    await assert.rejects(readCampaignSet(directory, { safetyPolicy }), /requires a trusted resume identity/);
    await assert.rejects(readCampaignSet(directory, {
      ...trusted, resumeIdentity: { ...resumeIdentity, executableSha256: "b".repeat(64) },
    }), /resume identity differs/);
    await assert.rejects(readCampaignSet(directory, {
      ...trusted, safetyPolicy: { ...safetyPolicy, rowDeadlineSeconds: 601 },
    }), /safety policy identity differs/);
    const constrainedPolicy = { ...safetyPolicy, maxContextTokens: 4096, maxRequestTokens: 4096 };
    const constrainedIdentity = {
      ...resumeIdentity, policySha256: campaignPolicySha256(constrainedPolicy),
    };
    const constrainedIdentitySha256 = campaignResumeIdentitySha256(
      constrainedIdentity, constrainedIdentity.policySha256,
    );
    await writeFile(path.join(directory, "safety-policy.json"), canonicalJson(constrainedPolicy));
    await writeFile(path.join(directory, "resume-identity.json"), canonicalJson(constrainedIdentity));
    await writeFile(path.join(directory, "resume-identity.json.sha256"),
      `${constrainedIdentitySha256}  resume-identity.json\n`);
    await writeCampaignManifest(directory, {
      ...manifest, policySha256: constrainedIdentity.policySha256,
      resumeIdentitySha256: constrainedIdentitySha256,
    });
    await assert.rejects(readCampaignSet(directory, {
      safetyPolicy: constrainedPolicy, resumeIdentity: constrainedIdentity,
    }), /geometry exceeds mandatory safety policy/);
    await writeFile(path.join(directory, "safety-policy.json"), canonicalJson(safetyPolicy));
    await writeFile(path.join(directory, "resume-identity.json"), canonicalJson(resumeIdentity));
    await writeFile(path.join(directory, "resume-identity.json.sha256"),
      `${manifest.resumeIdentitySha256}  resume-identity.json\n`);
    await writeCampaignManifest(directory, { ...manifest, coordinates: manifest.coordinates.slice(1) });
    await assert.rejects(readCampaignSet(directory, trusted), /exactly 8 scheduled rows/);
    const unscheduled = structuredClone(manifest);
    unscheduled.coordinates[0].coordinate = "llama-short-single-single-shot-cold";
    await writeCampaignManifest(directory, unscheduled);
    await assert.rejects(readCampaignSet(directory, trusted), /unscheduled coordinate/);
    const duplicate = structuredClone(manifest);
    duplicate.coordinates[7] = structuredClone(duplicate.coordinates[0]);
    await writeCampaignManifest(directory, duplicate);
    await assert.rejects(readCampaignSet(directory, trusted), /repeats a coordinate/);
    const wrongArtifact = structuredClone(manifest);
    wrongArtifact.coordinates[0].files[0].sha256 = "f".repeat(64);
    await writeCampaignManifest(directory, wrongArtifact);
    await assert.rejects(readCampaignSet(directory, trusted), /artifact bytes or sidecar differ/);
    const forgedIdentity = { ...manifest, resumeIdentitySha256: "b".repeat(64) };
    await writeCampaignManifest(directory, forgedIdentity);
    await assert.rejects(readCampaignSet(directory, trusted), /resume identity differs/);
    const forgedModel = structuredClone(resumeIdentity);
    forgedModel.llamaCandidate.sha256 = "f".repeat(64);
    await writeCampaignManifest(directory, {
      ...manifest,
      resumeIdentitySha256: campaignResumeIdentitySha256(forgedModel, manifest.policySha256),
    });
    await writeFile(path.join(directory, "resume-identity.json"), canonicalJson(forgedModel));
    await writeFile(path.join(directory, "resume-identity.json.sha256"),
      `${campaignResumeIdentitySha256(forgedModel, manifest.policySha256)}  resume-identity.json\n`);
    await assert.rejects(readCampaignSet(directory, { ...trusted, resumeIdentity: forgedModel }),
      /source or model inventory differs/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("full 64-coordinate dense campaign and invalid sets",()=>{const rows=[];for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"])rows.push(fixture("dense",{family,contextBand,requestMode,prefillMode,processTemperature}));rows.forEach((row,index)=>{rows[index]=withCampaignPid(row,index+10);});assert.equal(validateCampaign(rows, { scheduleVersion: 1 }).coordinates,64);assert.throws(()=>validateCampaign([rows[0]], { scheduleVersion: 1 }),/incomplete/);assert.throws(()=>validateCampaign([fixture("compressed")], { scheduleVersion: 1 }),/incomplete|non-dense/);assert.throws(()=>validateCampaign([...rows,rows[0]], { scheduleVersion: 1 }),/incomplete|duplicate/);});
test("complete campaign publication is one atomic 64-coordinate set",async()=>{const root=await mkdtemp("/tmp/kv20671-campaign-"),inputs=path.join(root,"workers"),destination=path.join(root,"published"),coordinates=[];await mkdir(inputs);for(const family of ["llama","qwen"])for(const contextBand of ["short","medium","memory-material","fit-boundary"])for(const requestMode of ["single","supported-batch"])for(const prefillMode of ["chunked","single-shot"])for(const processTemperature of ["cold","warm"])coordinates.push({family,contextBand,requestMode,prefillMode,processTemperature});const sets=[];for(let index=0;index<coordinates.length;index+=1){const source=path.join(inputs,`source-${index}`),set=path.join(inputs,String(index)),receipt=await verifiedFixture(source,"dense",coordinates[index],{pid:index+10});await writeReceiptSet(set,receipt);sets.push(set);}await assert.rejects(writeCampaignSet(destination,sets.slice(1)),/exactly 64/);assert.equal((await rm(destination,{recursive:true,force:true})),undefined);await writeCampaignSet(destination,sets);const manifest=JSON.parse(await readFile(path.join(destination,"campaign.json"),"utf8"));assert.equal(manifest.schemaVersion,1);assert.equal(manifest.coordinates.length,64);assert.equal((await readCampaignSet(destination,{allowLegacy:true})).summary.coordinates,64);await assert.rejects(readCampaignSet(destination),/legacy/);assert.equal(await readFile(path.join(destination,"campaign.json.sha256"),"utf8"),`${sha256(await readFile(path.join(destination,"campaign.json"),"utf8"))}  campaign.json\n`);await rm(root,{recursive:true,force:true});});

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

test("SC-20676 compressed receipts carry reasoned fused/fallback evidence and no dense reconstruction", () => {
  const accepted = fixture("compressed");
  assert.equal(accepted.compression.method, "group-affine");
  assert.equal(validateReceipt(accepted), accepted);
  const withCompression = (edit, base = accepted.compression) => {
    const compression = structuredClone(base);
    edit(compression);
    return () => fixture("compressed", {}, { compression });
  };
  // The block belongs exactly to compressed rows.
  assert.throws(() => fixture("dense", {}, { compression: accepted.compression }), /schema validation|present exactly/);
  assert.throws(() => rebuildReceipt(accepted, (raw) => { delete raw.compression; }), /schema validation|present exactly/);
  // Silent fallback: uncounted, unreasoned, or unordered dense execution, or no fused execution.
  assert.throws(withCompression((c) => { c.fallbackCalls = 2; }), /not fully reasoned/);
  assert.throws(withCompression((c) => { c.fallbacks[0].reason = " "; }), /unreasoned/);
  assert.throws(withCompression((c) => {
    c.fallbacks = [{ operation: "z", reason: "r", calls: 1 }, { operation: "a", reason: "r", calls: 1 }];
    c.fallbackCalls = 2;
  }), /unordered/);
  assert.throws(withCompression((c) => { c.fusedCalls = 0; }), /never executed the fused/);
  // Kernel paths: the NAX kernel exactly with the NAX selection, every fused call attributed.
  const pathRejected = /disagrees with its selection or is unordered|schema validation/;
  assert.throws(withCompression((c) => { c.kernelPaths[0].selection = "nax-unavailable"; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths[0].kernel = "sc20676_tiled_multi_row_simdgroup_matrix"; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths[1].selection = "nax-selected"; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths[0].queryDtype = "float32"; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths[0].reason = " "; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths.reverse(); }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelGpuFamily = "conservative-unknown-apple"; }), pathRejected);
  assert.throws(withCompression((c) => { c.kernelPaths[1].calls -= 1; }), /not all attributed to a kernel path/);
  assert.throws(withCompression((c) => { c.kernelPaths = []; }), /not all attributed|schema validation/);
  assert.throws(withCompression((c) => { delete c.kernelPaths; }), /schema validation|kernelPaths/);
  for (const [family, kernel, selection, dtype, valid] of [
    ["apple7-or-newer", "sc20676_nax_tiled_matmul2d", "nax-selected", "float16", true],
    ["apple7-or-newer", "sc20676_tiled_multi_row_simdgroup_matrix", "f32-query", "float32", true],
    ["apple7-or-newer", "sc20676_tiled_multi_row_simdgroup_matrix", "nax-head-dimension", "bfloat16", true],
    ["apple7-or-newer", "sc20676_tiled_multi_row_simdgroup_matrix", "nax-unavailable", "float32", true],
    ["conservative-unknown-apple", "sc20676_split_kv_simdgroup", "conservative-family", "bfloat16", true],
    ["apple7-or-newer", "sc20676_nax_tiled_matmul2d", "nax-selected", "float32", false],
    ["apple7-or-newer", "sc20676_tiled_multi_row_simdgroup_matrix", "f32-query", "bfloat16", false],
    ["apple7-or-newer", "sc20676_tiled_multi_row_simdgroup_matrix", "nax-head-dimension", "float32", false],
    ["apple7-or-newer", "sc20676_split_kv_simdgroup", "conservative-family", "bfloat16", false],
    ["conservative-unknown-apple", "sc20676_split_kv_simdgroup", "below-multi-row-threshold", "bfloat16", false],
  ]) {
    assert.equal(kernelPathValid(family, kernel, selection, dtype), valid, `${family} ${kernel} ${selection} ${dtype}`);
  }
  assert.throws(withCompression((c) => {
    c.persistentKvRepresentation = "dense-fallback"; c.fallbacks = []; c.fallbackCalls = 0;
  }), /persistent KV representation/);
  assert.throws(() => fixture("compressed", {}, { lifecycle: { ...lifecycle, denseFallback: false, denseFallbackFallbackReason: "unsupported" } }), /not fully reasoned/);
  // Dense reconstruction: counted by the cache, or witnessed by an explicit allocation event.
  assert.throws(withCompression((c) => { c.fullCacheDequantizations = 1; }), /reconstructed a dense full cache/);
  assert.throws(() => rebuildReceipt(accepted, (raw) => {
    raw.memory.allocationEvents.push({ kind: "full_cache_materialization", role: "cache", lifetime: "transient", phase: "prefill-peak", timestamp: "2026-08-29T12:00:02.300Z", bytes: 1 });
  }), /full-cache temporary/);
  // Physical bytes are the measured components, not bit accounting.
  assert.throws(withCompression((c) => { c.hostPayloadBytes += 1; }), /do not reconcile with measured storage/);
  // A compressed coordinate: persistent KV is exactly its device share at the receipt's KV length,
  // and the whole physical representation (host copy + staged tail included) is below dense.
  assert.throws(withCompression((c) => { c.deviceCodeBytes += 1; c.hostPayloadBytes -= 1; }), /coordinate's measured storage/);
  assert.throws(withCompression((c) => { c.storageTokens += 1; }), /coordinate's measured storage/);
  const dense = accepted.memory.denseTheoreticalKvBytes;
  assert.throws(withCompression((c) => {
    c.hostPayloadBytes = dense - c.deviceCodeBytes - c.deviceMetadataBytes; c.physicalKvBytes = dense;
  }), /persistent KV representation/);
  // A dense-fallback coordinate is a valid row that claims no compressed storage.
  const denseRow = fixture("compressed", {}, { compression: denseFallbackCompression() });
  assert.equal(validateReceipt(denseRow), denseRow);
  assert.throws(withCompression((c) => { c.persistentKvRepresentation = "dense-fallback"; }), /persistent KV representation/);
  assert.throws(withCompression((c) => { c.storageTokens = 1; }, denseFallbackCompression()), /persistent KV representation/);
  assert.throws(withCompression((c) => {
    c.deviceCodeBytes = 1; c.physicalKvBytes = 1;
  }, denseFallbackCompression()), /persistent KV representation/);
});

test("SC-20676 reductions are claimed only for coordinates that ran compressed, from physical bytes", () => {
  const compressed = fixture("compressed");
  const eligible = compareReceipts(fixture(), compressed);
  assert.equal(eligible.persistentKvRepresentation, "compressed");
  assert.equal(eligible.persistentKvReductionEligible, true);
  assert.equal(eligible.persistentKvReductionIneligibleReason, null);
  // Device share alone would read 50%; the host copy and staged tail make the honest claim 25%.
  assert.equal(
    eligible.persistentKvReduction,
    1 - compressed.compression.physicalKvBytes / fixture().memory.persistentKvBytes,
  );
  assert.equal(eligible.persistentKvReduction, .25);
  assert.match(renderComparisonMarkdown(eligible), /Persistent KV reduction: 25\.00%/);
  for (const coordinate of [
    { requestMode: "supported-batch" },
    { prefillMode: "chunked" },
    { contextBand: "fit-boundary", requestMode: "supported-batch", prefillMode: "chunked" },
  ]) {
    const ineligible = compareReceipts(
      fixture("dense", coordinate),
      fixture("compressed", coordinate, { compression: denseFallbackCompression() }),
    );
    assert.equal(ineligible.persistentKvReductionEligible, false);
    assert.equal(ineligible.persistentKvReduction, null);
    assert.match(ineligible.persistentKvReductionIneligibleReason, /dense-fallback/);
    assert.match(renderComparisonMarkdown(ineligible), /Persistent KV reduction: not claimed \(.*dense-fallback/);
  }
  // Eligibility keys off the recorded representation, not the row name: a chunked row whose
  // coordinate ran compressed is eligible.
  const chunked = { prefillMode: "chunked" };
  assert.equal(compareReceipts(fixture("dense", chunked), fixture("compressed", chunked)).persistentKvReductionEligible, true);
});

test("SC-20676 compressed campaigns are uniform and bound to their resume identity mode", () => {
  const identity = sampleResumeIdentity(safetyPolicy);
  const policySha256 = campaignPolicySha256(safetyPolicy);
  const denseSha = campaignResumeIdentitySha256(identity, policySha256);
  const compressedSha = campaignResumeIdentitySha256({ ...identity, mode: "compressed", kvMethod: "group-affine" }, policySha256);
  assert.notEqual(denseSha, compressedSha);
  assert.throws(() => campaignResumeIdentitySha256({ ...identity, mode: "compressed" }, policySha256), /compressed mode binding/);
  assert.throws(() => campaignResumeIdentitySha256({ ...identity, mode: "dense", kvMethod: "group-affine" }, policySha256), /compressed mode binding/);
  const rows = SC20671_COVERING_SCHEDULE.map(([family, contextBand, requestMode, prefillMode, processTemperature], index) =>
    withCampaignPid(fixture("compressed", { family, contextBand, requestMode, prefillMode, processTemperature }), index + 10));
  const summary = validateCampaign(rows);
  assert.equal(summary.mode, "compressed");
  assert.equal(summary.kvMethod, "group-affine");
  const mixed = [...rows];
  mixed[0] = withCampaignPid(fixture("dense", rows[0].matrix), 10);
  assert.throws(() => validateCampaign(mixed), /mixes dense and compressed/);
});

test("SC-20676 group-affine-4 and -8 rows validate and bind their own method", () => {
  const fourBit = (row) => ({ ...row.compression, method: "group-affine-4",
    representationIdentity: "sc-20676-packed-group-affine-b4-v1", bits: 4 });
  const accepted = fixture("compressed", {}, { compression: fourBit(fixture("compressed")) });
  assert.equal(validateReceipt(accepted), accepted);
  assert.equal(accepted.compression.bits, 4);
  const identity = sampleResumeIdentity(safetyPolicy);
  const policySha256 = campaignPolicySha256(safetyPolicy);
  assert.notEqual(
    campaignResumeIdentitySha256({ ...identity, mode: "compressed", kvMethod: "group-affine-4" }, policySha256),
    campaignResumeIdentitySha256({ ...identity, mode: "compressed", kvMethod: "group-affine" }, policySha256));
  const rows = SC20671_COVERING_SCHEDULE.map(([family, contextBand, requestMode, prefillMode, processTemperature], index) => {
    const matrix = { family, contextBand, requestMode, prefillMode, processTemperature };
    return withCampaignPid(fixture("compressed", matrix, { compression: fourBit(fixture("compressed", matrix)) }), index + 10);
  });
  assert.equal(validateCampaign(rows).kvMethod, "group-affine-4");
  // Each method is bound to its width and reader identity; unknown methods are refused.
  // Every (method, bits, identity) combination: only the method's own pair validates.
  const table = [
    ["group-affine", 2, "sc-20676-packed-group-affine-v1"],
    ["group-affine-4", 4, "sc-20676-packed-group-affine-b4-v1"],
    ["group-affine-8", 8, "sc-20676-packed-group-affine-b8-v1"],
  ];
  for (const [method, ownBits, ownIdentity] of table) {
    for (const [, bits] of table) {
      for (const [, , representationIdentity] of table) {
        const compression = { ...accepted.compression, method, bits, representationIdentity };
        if (bits === ownBits && representationIdentity === ownIdentity) {
          const row = fixture("compressed", {}, { compression });
          assert.equal(validateReceipt(row), row);
        } else {
          assert.throws(() => fixture("compressed", {}, { compression }), new RegExp(`compressed method ${method} is`), `${method}/${bits}/${representationIdentity}`);
        }
      }
    }
  }
  // Unknown methods and widths are refused by the schema and by the validator's table.
  assert.throws(() => fixture("compressed", {}, { compression: { ...accepted.compression, method: "rvq-unwired" } }), /schema validation|unknown compressed KV method/);
  assert.throws(() => fixture("compressed", {}, { compression: { ...accepted.compression, bits: 3 } }), /schema validation|compressed method/);
  // A campaign is one method: a 2-bit row cannot join a 4-bit campaign.
  const mixed = [...rows];
  mixed[0] = withCampaignPid(fixture("compressed", rows[0].matrix), 10);
  assert.throws(() => validateCampaign(mixed), /compressed-method/);
});
test("SC-20671 compressed quality misses are recorded as a failed gate, never a pass", () => {
  const base = fixture("compressed").quality;
  const miss = fixture("compressed", {}, { quality: gatedQuality("compressed", { ...base, needleRetrieval: 0, perplexityDelta: 0.5 }, forcedContinuation(1022)) });
  const gate = miss.quality.qualityGate;
  assert.equal(gate.passed, false);
  assert.equal(gate.failures.length, 15, "three misses in each of five repeats");
  assert.deepEqual(gate.failures.slice(0, 3), [
    { metric: "greedyTokenAgreement", fixture: "kernel-fp32-reference", repeat: 0, value: 1022 / 1024, threshold: 0.999, comparison: "minimum" },
    { metric: "perplexityDelta", fixture: "kernel-fp32-reference", repeat: 0, value: 0.5, threshold: 0.01, comparison: "maximum" },
    { metric: "needleRetrieval", fixture: "long-context-needle", repeat: 0, value: 0, threshold: 1, comparison: "minimum" },
  ]);
  assert.deepEqual(miss.quality.forcedContinuation.firstFlipPositions, [1022, 1023]);
  // One flip in 1024 clears 0.999; two do not.
  assert.equal(qualityGateFromRepeats([{ ...base, greedyTokenAgreement: 1023 / 1024 }]).passed, true);
  assert.equal(qualityGateFromRepeats([{ ...base, greedyTokenAgreement: 1022 / 1024 }]).passed, false);
  // Comparison and its human record report the failed gate.
  const comparison = compareReceipts(fixture(), miss);
  assert.equal(comparison.qualityGatePassed, false);
  assert.deepEqual(comparison.qualityGate, gate);
  assert.equal(compareReceipts(fixture(), fixture("compressed")).qualityGatePassed, true);
  assert.match(renderComparisonMarkdown(comparison), /Quality gate: FAILED: greedyTokenAgreement repeat 0 = 0\.998046875 \(minimum 0\.999, fixture kernel-fp32-reference\); perplexityDelta repeat 0 = 0\.5 \(maximum 0\.01/);
  assert.match(renderReceiptMarkdown(miss), /- Quality gate: FAILED: /);
  assert.match(renderReceiptMarkdown(fixture("compressed")), /- Quality gate: passed\n/);
  assert.doesNotMatch(renderReceiptMarkdown(fixture()), /Quality gate/);
  // A pass can never be claimed over a failing value.
  const forged = (edit) => {
    const quality = structuredClone(miss.quality);
    edit(quality.qualityGate);
    return () => fixture("compressed", {}, { quality });
  };
  assert.throws(forged((g) => { g.passed = true; g.failures = []; }), /quality gate does not record greedyTokenAgreement repeat 0 = 0\.998046875 against the frozen minimum 0\.999/);
  assert.throws(forged((g) => { g.passed = true; }), /claims passed=true with 15 recorded failure/);
  assert.throws(forged((g) => { g.failures[0].value = 0.5; }), /does not record greedyTokenAgreement repeat 0/);
  assert.throws(forged((g) => { g.failures[1].threshold = 0.6; }), /not an ordered frozen-threshold miss/);
  assert.throws(forged((g) => { g.failures.reverse(); }), /not an ordered frozen-threshold miss/);
  assert.throws(forged((g) => { g.failures[2].fixture = "kernel-fp32-reference"; }), /not an ordered frozen-threshold miss/);
  // Integrity still refuses: a gated row with a broken kernel, or a missing measurement.
  assert.throws(() => fixture("compressed", {}, { quality: gatedQuality("compressed", { ...base, parityMaxError: 0.5 }) }), /kernel parity failed/);
  const { forcedContinuation: _forced, ...unmeasured } = base;
  assert.throws(() => fixture("compressed", {}, { quality: unmeasured }), /schema validation failed/);
  assert.throws(() => fixture("compressed", {}, { quality: { ...base, forcedContinuation: { ...base.forcedContinuation, matches: 1000 } } }), /forced continuation evidence is inconsistent/);
  // The sealed repeats bind the gate: a miss visible only in repeat 3's artifact must be recorded.
  const evidence = (raw, repeat, overrides = {}) => Object.fromEntries(qualityFixtures.map((name) => {
    const artifact = artifactFixture(raw, name, repeat);
    return [name, { ...artifact, evidence: { ...artifact.evidence, ...(overrides[name] ?? {}) } }];
  }));
  const clean = fixture("compressed");
  const sealed = [0, 1, 2, 3, 4].map((repeat) => evidence(clean, repeat, repeat === 3 ? { "long-context-needle": { matches: 0 } } : {}));
  const repeats = sealedRepeatQualityMetrics(clean, sealed);
  assert.equal(repeats[3].needleRetrieval, 0);
  assert.throws(() => validateSealedQualityGate(clean, repeats), /is not the gate of its sealed repeats \(FAILED: needleRetrieval repeat 3 = 0/);
  const recorded = { ...clean, quality: { ...clean.quality, qualityGate: qualityGateFromRepeats(repeats) } };
  validateSealedQualityGate(recorded, repeats);
  assert.throws(() => sealedRepeatQualityMetrics(clean, sealed.map((fixtures, repeat) => repeat === 1
    ? { ...fixtures, "kernel-fp32-reference": { ...fixtures["kernel-fp32-reference"], evidence: { ...fixtures["kernel-fp32-reference"].evidence, forcedContinuation: forcedContinuation(1000) } } }
    : fixtures)), /repeat 1 kernel fixture forced continuation is not the receipt's/);
  // The receipt's multiTurnCache and forced-pass records are the sealed artifacts' own.
  const otherTurns = { ...MULTI_TURN_TURNS, turn2: { ...MULTI_TURN_TURNS.turn2, reusedPrefixTokens: 46 } };
  const cacheEdit = (repeat, edit) => sealed.map((fixtures, index) => index === repeat
    ? { ...fixtures, "multi-turn-prompt-cache": { ...fixtures["multi-turn-prompt-cache"], evidence: edit({ ...fixtures["multi-turn-prompt-cache"].evidence }) } }
    : fixtures);
  assert.throws(() => sealedRepeatQualityMetrics(clean, cacheEdit(0, (e) => ({ ...e, turns: { candidate: otherTurns, reference: MULTI_TURN_TURNS } }))), /receipt multiTurnCache is not the primary repeat's sealed turn records/);
  assert.throws(() => sealedRepeatQualityMetrics(clean, cacheEdit(2, (e) => ({ ...e, forcedPass: { reference: otherTurns, candidate: otherTurns } }))), /repeat 2 multi-turn forced-pass turn records are not the receipt's/);
  assert.throws(() => validateSealedQualityGate(clean, [0, 1, 2, 3, 4].map((repeat) => ({ ...repeats[0], parityMaxError: repeat === 2 ? 0.5 : 0 }))), /kernel parity failed: metric=parityMaxError value=0\.5 .*repeat=2/);
  assert.equal(campaignQualityGatePassed([clean, clean]), true);
  assert.equal(campaignQualityGatePassed([clean, miss]), false);
  assert.equal(campaignQualityGatePassed([fixture(), fixture()]), undefined);
  assert.throws(() => campaignQualityGatePassed([fixture(), clean]), /mixes gated and ungated/);
  assert.equal(qualityGateSummary(clean.quality.qualityGate), "passed");
});
test("SC-20671 compressed campaign completes with every row and reports a failed gate verdict", async () => {
  const root = await mkdtemp("/tmp/kv20671-qgate-");
  try {
    const { directory, manifest, resumeIdentity } = await writeEightCampaign(root, safetyPolicy, {
      mode: "compressed", continuations: { 0: forcedContinuation(1022) },
    });
    const trusted = { safetyPolicy, resumeIdentity };
    const { summary } = await readCampaignSet(directory, trusted);
    assert.equal(summary.mode, "compressed");
    assert.equal(summary.receipts, 8);
    assert.equal(summary.qualityGatePassed, false);
    assert.equal(manifest.qualityGatePassed, false);
    assert.deepEqual(summary.qualityGates.map((row) => row.passed), [false, true, true, true, true, true, true, true]);
    assert.equal(summary.qualityGates[0].coordinate, "llama-short-single-chunked-cold");
    assert.deepEqual(summary.qualityGates[0].failures.map((failure) => [failure.metric, failure.repeat, failure.value]),
      [0, 1, 2, 3, 4].map((repeat) => ["greedyTokenAgreement", repeat, 1022 / 1024]));
    // The manifest verdicts must recompute from the rows.
    await writeCampaignManifest(directory, { ...manifest, qualityGatePassed: true });
    await assert.rejects(readCampaignSet(directory, trusted), /qualityGatePassed does not recompute from its rows/);
    const { qualityGatePassed: _omitted, ...unstated } = manifest;
    await writeCampaignManifest(directory, unstated);
    await assert.rejects(readCampaignSet(directory, trusted), /qualityGatePassed does not recompute from its rows/);
    await writeCampaignManifest(directory, { ...manifest, coordinates: manifest.coordinates.map((row, index) => index === 0 ? { ...row, qualityGatePassed: true } : row) });
    await assert.rejects(readCampaignSet(directory, trusted), /row llama-short-single-chunked-cold qualityGatePassed does not recompute/);
    await writeCampaignManifest(directory, manifest);
    assert.equal((await readCampaignSet(directory, trusted)).summary.qualityGatePassed, false);
    // Every row meeting its thresholds passes the campaign.
    await mkdir(path.join(root, "clean"));
    const clean = await writeEightCampaign(path.join(root, "clean"), safetyPolicy, { mode: "compressed" });
    const cleanSummary = (await readCampaignSet(clean.directory, { safetyPolicy, resumeIdentity: clean.resumeIdentity })).summary;
    assert.equal(cleanSummary.qualityGatePassed, true);
    assert.ok(cleanSummary.qualityGates.every((row) => row.passed && row.failures.length === 0));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("contract v4 multiTurnPromptCache is teacher-forced on a cache-hit turn 2", () => {
  const compressed = fixture("compressed");
  assert.equal(compressed.quality.multiTurnPromptCacheMethod, MULTI_TURN_PROMPT_CACHE_METHOD);
  assert.equal(compressed.quality.multiTurnPromptCache, compressed.quality.multiTurnForcedContinuation.agreement);
  const measured = (() => { const { qualityGate: _g, ...rest } = compressed.quality; return rest; })();
  // One flip in 1024 meets the 0.999 minimum (aligned with greedy agreement); two do not.
  const oneFlip = fixture("compressed", {}, { quality: gatedQuality("compressed", measured, forcedContinuation(), multiTurnForcedContinuation(1023)) });
  assert.equal(oneFlip.quality.qualityGate.passed, true);
  const twoFlips = fixture("compressed", {}, { quality: gatedQuality("compressed", measured, forcedContinuation(), multiTurnForcedContinuation(1022)) });
  assert.equal(twoFlips.quality.qualityGate.passed, false);
  assert.deepEqual(twoFlips.quality.qualityGate.failures.map((failure) => [failure.metric, failure.threshold]),
    Array(5).fill(["multiTurnPromptCache", 0.999]));
  // The recorded value must be the turn-2 forced-continuation agreement, never a free-running one.
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnPromptCache: 0.5625 } }), /not the row's turn-2 forced-continuation agreement/);
  const { multiTurnForcedContinuation: _turn2, ...unmeasured } = compressed.quality;
  assert.throws(() => fixture("compressed", {}, { quality: unmeasured }), /schema validation|turn-2 forced continuation/);
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnForcedContinuation: { ...compressed.quality.multiTurnForcedContinuation, method: "dense-kv-same-weights-greedy-continuation-eos-ignored-teacher-forced" } } }), /schema validation|forced continuation evidence is inconsistent/);
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnPromptCacheMethod: "free-running" } }), /schema validation|cache-hit turn 2/);
  // Both arms' turn 2 must have been served by the prompt-cache hit.
  const missed = { ...MULTI_TURN_TURNS, turn2: { ...MULTI_TURN_TURNS.turn2, cacheHit: false, reusedPrefixTokens: 0 } };
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnCache: { candidate: MULTI_TURN_TURNS, reference: missed } } }), /did not serve turn 2/);
  // The forced pass must be sealed, over one turn-2 prompt shared by both sessions and the fixture.
  const { multiTurnForcedPass: _unsealed, ...unsealedPass } = compressed.quality;
  assert.throws(() => fixture("compressed", {}, { quality: unsealedPass }), /schema validation|no turn records/);
  const otherPrompt = { ...MULTI_TURN_TURNS, turn2: { ...MULTI_TURN_TURNS.turn2, promptSha256: "3".repeat(64) } };
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnForcedPass: { reference: MULTI_TURN_TURNS, candidate: otherPrompt } } }), /turn-2 prompts differ/);
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnForcedPass: { reference: otherPrompt, candidate: otherPrompt } } }), /not the multi-turn fixture's same-weights turn-2 prompt/);
  assert.throws(() => fixture("compressed", {}, { quality: { ...compressed.quality, multiTurnCache: { candidate: otherPrompt, reference: MULTI_TURN_TURNS } } }), /not the multi-turn fixture's same-weights turn-2 prompt/);
  // The multi-turn continuation is never shortened, even on a fit-boundary row.
  assert.throws(() => fixture("compressed", {}, { quality: gatedQuality("compressed", measured, forcedContinuation(), multiTurnForcedContinuation(300, 300)) }), /schema validation|forced continuation evidence is inconsistent/);
  // Dense rows carry no turn-2 forced continuation; a v3-contract receipt is refused outright.
  assert.throws(() => fixture("dense", {}, { quality: { ...fixture().quality, multiTurnForcedContinuation: multiTurnForcedContinuation() } }), /schema validation/);
  const v3 = structuredClone(fixture());
  v3.contractHash = "58eaa007c35084c8acac2b35b5a2a1dff5af55832a7533557741d43a05944b49";
  assert.throws(() => validateReceipt(v3), /quality contract hash mismatch/);
});
