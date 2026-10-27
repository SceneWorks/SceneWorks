// Typed sampled Windows process accounting: never a physical-residency/allocator alias
// or predictive fit authorization. Pure CPU validation; no telemetry or model access.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
export const OWNED_ACCOUNTING_KIND = 'windows-owned-process-accounting-v1';
const sha = /^[a-f0-9]{64}$/;
const exact = (value, keys, label) => { assert(value && typeof value === 'object' && !Array.isArray(value), `${label} object required`); assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), `${label} unknown/missing fields`); };
const natural = (v, label) => assert(Number.isSafeInteger(v) && v >= 0, `${label} finite nonnegative safe integer required`);
const text = (v, label) => assert(typeof v === 'string' && v.length > 0, `${label} required`);
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])])) : value;
export const ownedAccountingDigest = (domain, value) => createHash('sha256').update(domain).update(JSON.stringify(canonical(value))).digest('hex');
const unavailable = value => { exact(value, ['physicalResidentBytes','activeBytes','allocatorBytes'], 'unavailable accounting metrics'); for (const v of Object.values(value)) assert.equal(v, null, 'unavailable residency/allocator metrics cannot become numbers'); };
export function validateOwnedProcessAccountingRecord(record) {
  assert.equal(record.measurementKind, OWNED_ACCOUNTING_KIND, 'unknown owned-process measurement kind');
  assert.equal(record.backend, 'candle');
  assert(['flux3_klein','flux3_klein_base'].includes(record.target?.provider), 'owned accounting is scoped to native FLUX3');
  assert.equal(record.observedMemory, undefined, 'owned accounting cannot alias legacy residency/allocator observedMemory');
  for (const name of ['conditioningDevicePeakDelta','denoiseDevicePeakDelta','decodeDevicePeakDelta','overallDevicePeakDelta']) {
    assert.equal(record.diagnostics?.metrics?.[name], undefined, 'owned accounting cannot alias legacy device delta');
    assert(!record.diagnostics?.measurements?.some(metric => metric.name === name), 'owned accounting cannot alias legacy diagnostic measurement');
  }
  if (record.unavailableMetrics !== undefined) unavailable(record.unavailableMetrics);
  const memory = record.ownedProcessMemory;
  exact(memory, ['schemaVersion','kind','source','semantics','attribution','physicalResidencyClaim','allocatorActivityClaim','deviceTotalDeltaAttributionClaim','exclusiveThroughputClaim','unavailableMetrics','sampling','identity','physicalNodes','phases','overall'], 'owned process memory');
  assert.equal(memory.schemaVersion, 1); assert.equal(memory.kind, 'flux3_owned_process_memory_accounting_v1');
  assert.equal(memory.source, 'Windows GPU Process Memory performance counters');
  assert.equal(memory.semantics, 'sampled per-process dedicated/shared/total-committed accounting; not physical residency or CUDA active/reserved allocator bytes');
  assert.equal(memory.attribution, 'own PID/start-time/executable and CUDA adapter LUID; includes any cross-process-shared allocations charged to this process');
  for (const name of ['physicalResidencyClaim','allocatorActivityClaim','deviceTotalDeltaAttributionClaim','exclusiveThroughputClaim']) assert.equal(memory[name], false, `unsupported accounting ${name}`);
  unavailable(memory.unavailableMetrics);
  const sampling = memory.sampling;
  exact(sampling, ['intervalMs','maxAllowedGapMs','maxAllowedReadSkewMs','maxObservedReadSkewMs','losslessPeakClaim'], 'accounting sampling');
  assert.equal(sampling.intervalMs, 200);
  assert.equal(sampling.maxAllowedGapMs, 1500);
  assert.equal(sampling.maxAllowedReadSkewMs, 250); natural(sampling.maxObservedReadSkewMs, 'observed counter read skew');
  assert(sampling.maxObservedReadSkewMs <= sampling.maxAllowedReadSkewMs); assert.equal(sampling.losslessPeakClaim, false);
  const identity = memory.identity;
  exact(identity, ['pid','startedAt','executable','luid','physicalGpuId','gpuUuid','pciBusId','logicalCudaOrdinal'], 'accounting identity');
  assert(Number.isSafeInteger(identity.pid) && identity.pid > 0); assert(Number.isFinite(Date.parse(identity.startedAt))); text(identity.executable, 'accounting executable');
  assert(/^0x[0-9a-f]{8}_0x[0-9a-f]{8}$/i.test(identity.luid));
  assert(['0','1'].includes(identity.physicalGpuId)); assert(/^GPU-[a-f0-9-]+$/i.test(identity.gpuUuid)); assert(/^[a-f0-9]{8}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]$/i.test(identity.pciBusId)); assert.equal(identity.logicalCudaOrdinal, 0);
  assert(Array.isArray(memory.physicalNodes) && memory.physicalNodes.length > 0 && memory.physicalNodes.every(n => Number.isSafeInteger(n) && n >= 0 && n <= 31) && new Set(memory.physicalNodes).size === memory.physicalNodes.length, 'complete unique physical nodes required');
  const phase = (value, label) => {
    exact(value, ['dedicatedPeakBytes','sharedPeakBytes','totalCommittedPeakBytes','samples','firstSampleMs','lastSampleMs','maxGapMs'], label);
    for (const key of Object.keys(value)) natural(value[key], `${label}.${key}`);
    assert(value.samples >= 2 && value.lastSampleMs > value.firstSampleMs && value.maxGapMs > 0 && value.maxGapMs <= sampling.maxAllowedGapMs, `${label} missing interval coverage`);
    const span = value.lastSampleMs - value.firstSampleMs;
    assert(value.maxGapMs <= span && span <= value.maxGapMs * (value.samples - 1), `${label} impossible sample coverage`);
    // Each counter has its own sampled maximum; no additive/simultaneous/residency inference.
  };
  exact(memory.phases, ['load','conditioning','denoise','decode', ...(Object.hasOwn(memory.phases ?? {}, 'promptUpsampling') ? ['promptUpsampling'] : [])], 'accounting phases');
  phase(memory.overall, 'accounting overall');
  const order = ['load', ...(memory.phases.promptUpsampling ? ['promptUpsampling'] : []), 'conditioning','denoise','decode'];
  let previous;
  for (const name of order) {
    const value = memory.phases[name]; phase(value, `accounting ${name}`);
    assert(memory.overall.firstSampleMs <= value.firstSampleMs && memory.overall.lastSampleMs >= value.lastSampleMs);
    if (previous) assert.equal(previous.lastSampleMs, value.firstSampleMs, 'accounting phase boundaries must share an actual synchronized endpoint');
    for (const key of ['dedicatedPeakBytes','sharedPeakBytes','totalCommittedPeakBytes']) assert(memory.overall[key] >= value[key], 'overall accounting must cover every phase peak');
    assert(memory.overall.samples >= value.samples); previous = value;
  }
  const recipe = record.observedExecutionRecipe;
  exact(recipe, ['schemaVersion','backend','provider','tier','actualLoadSpecSha256','actualGenerationRequestSha256','invocationRequestSha256','measuredStrategySha256'], 'observed execution recipe');
  assert.equal(recipe.schemaVersion, 1); assert.equal(recipe.backend, record.backend); assert.equal(recipe.provider, record.target.provider); assert.equal(recipe.tier, record.target.tier);
  for (const key of ['actualLoadSpecSha256','actualGenerationRequestSha256','invocationRequestSha256','measuredStrategySha256']) assert(sha.test(recipe[key] ?? ''), `recipe ${key} required`);
  assert.equal(recipe.measuredStrategySha256, ownedAccountingDigest('flux3_observed_memory_strategy_v1\0', record.strategy), 'actual measured execution strategy changed');
  assert.equal(record.observedExecutionRecipeSha256, ownedAccountingDigest('flux3_observed_execution_recipe_v1\0', recipe), 'observed recipe binding changed');
  const c = record.observedOperatingConditions;
  exact(c, ['schemaVersion','sharingMode','physicalGpuId','gpuUuid','pciBusId','observedAt','totalDeviceBytes','freeDeviceBytes','foreignDedicatedBytes','foreignProcessCount','foreignMaxProcessBytes','foreignProcessLimitBytes','foreignTotalLimitBytes','observationScope','fitAuthorization','exclusiveThroughputClaim','physicalResidencyClaim'], 'observed operating conditions');
  assert.equal(c.schemaVersion, 1); assert.equal(c.sharingMode, 'bounded-shared'); assert.equal(c.observationScope, 'producer-reservation-admission');
  for (const key of ['fitAuthorization','exclusiveThroughputClaim','physicalResidencyClaim']) assert.equal(c[key], false);
  for (const key of ['physicalGpuId','gpuUuid','pciBusId']) assert.equal(c[key], identity[key], 'observed conditions differ from measured physical adapter');
  assert(Number.isFinite(Date.parse(c.observedAt)));
  for (const key of ['totalDeviceBytes','freeDeviceBytes','foreignDedicatedBytes','foreignProcessCount','foreignMaxProcessBytes','foreignProcessLimitBytes','foreignTotalLimitBytes']) natural(c[key], `conditions ${key}`);
  assert(c.totalDeviceBytes > 0 && c.freeDeviceBytes > 0 && c.freeDeviceBytes <= c.totalDeviceBytes);
  assert.equal(c.foreignProcessLimitBytes, 1024 ** 3); assert.equal(c.foreignTotalLimitBytes, 3 * 1024 ** 3);
  assert(c.foreignMaxProcessBytes <= c.foreignProcessLimitBytes && c.foreignDedicatedBytes <= c.foreignTotalLimitBytes && c.foreignMaxProcessBytes <= c.foreignDedicatedBytes);
  assert(c.foreignDedicatedBytes < c.totalDeviceBytes);
  // Match the strict Rust accounting context: an empty foreign set cannot own bytes.
  assert(c.foreignProcessCount !== 0 || (c.foreignDedicatedBytes === 0 && c.foreignMaxProcessBytes === 0), 'zero foreign process count cannot carry positive allocations');
}
