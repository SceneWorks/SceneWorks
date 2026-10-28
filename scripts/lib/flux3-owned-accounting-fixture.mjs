// DECLARED SYNTHETIC CPU fixture, not user approval, hardware evidence or a fit anchor.
import { ownedAccountingDigest, OWNED_ACCOUNTING_KIND } from './flux3-owned-accounting-contract.mjs';
export function declaredSyntheticOwnedAccounting(record) {
  const value = structuredClone(record);
  value.measurementKind = OWNED_ACCOUNTING_KIND;
  value.backend = 'candle'; value.target = { ...value.target, provider: 'flux3_klein_base', tier: 'bf16' };
  delete value.observedMemory;
  if (value.diagnostics?.metrics) for (const key of ['conditioningDevicePeakDelta','denoiseDevicePeakDelta','decodeDevicePeakDelta','overallDevicePeakDelta']) delete value.diagnostics.metrics[key];
  const phase = (first,last) => ({dedicatedPeakBytes:100,sharedPeakBytes:20,totalCommittedPeakBytes:120,samples:2,firstSampleMs:first,lastSampleMs:last,maxGapMs:last-first});
  value.ownedProcessMemory = {schemaVersion:1,kind:'flux3_owned_process_memory_accounting_v1',source:'Windows GPU Process Memory performance counters',semantics:'sampled per-process dedicated/shared/total-committed accounting; not physical residency or CUDA active/reserved allocator bytes',attribution:'own PID/start-time/executable and CUDA adapter LUID; includes any cross-process-shared allocations charged to this process',
    physicalResidencyClaim:false,allocatorActivityClaim:false,deviceTotalDeltaAttributionClaim:false,exclusiveThroughputClaim:false,
    unavailableMetrics:{physicalResidentBytes:null,activeBytes:null,allocatorBytes:null},
    sampling:{intervalMs:200,maxAllowedGapMs:1500,maxAllowedReadSkewMs:250,maxObservedReadSkewMs:5,losslessPeakClaim:false},
    identity:{pid:12345,startedAt:'2026-10-23T00:00:00.0000000Z',executable:'DECLARED_SYNTHETIC_NOT_AN_EXECUTABLE',luid:'0x00000000_0x00020806',physicalGpuId:'0',gpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',pciBusId:'00000000:21:00.0',logicalCudaOrdinal:0},physicalNodes:[0,1],
    phases:{load:phase(0,200),conditioning:phase(200,400),denoise:phase(400,600),decode:phase(600,800)},overall:{...phase(0,800),samples:5,maxGapMs:200}};
  value.observedExecutionRecipe = {schemaVersion:1,backend:'candle',provider:value.target.provider,tier:value.target.tier,actualLoadSpecSha256:'a'.repeat(64),actualGenerationRequestSha256:'b'.repeat(64),invocationRequestSha256:'c'.repeat(64),measuredStrategySha256:ownedAccountingDigest('flux3_observed_memory_strategy_v1\0',value.strategy)};
  value.observedExecutionRecipeSha256 = ownedAccountingDigest('flux3_observed_execution_recipe_v1\0',value.observedExecutionRecipe);
  value.observedOperatingConditions = {schemaVersion:1,sharingMode:'bounded-shared',physicalGpuId:'0',gpuUuid:value.ownedProcessMemory.identity.gpuUuid,pciBusId:value.ownedProcessMemory.identity.pciBusId,observedAt:'2026-10-23T00:00:00.000Z',totalDeviceBytes:100*1024**3,freeDeviceBytes:99*1024**3,foreignDedicatedBytes:604*1024**2,foreignProcessCount:1,foreignMaxProcessBytes:604*1024**2,foreignProcessLimitBytes:1024**3,foreignTotalLimitBytes:3*1024**3,observationScope:'producer-reservation-admission',fitAuthorization:false,exclusiveThroughputClaim:false,physicalResidencyClaim:false};
  return value;
}
