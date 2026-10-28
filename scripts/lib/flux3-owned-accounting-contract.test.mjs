// DECLARED SYNTHETIC CPU-only evidence contracts, not hardware or acceptance evidence.
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateOwnedProcessAccountingRecord, ownedAccountingDigest } from './flux3-owned-accounting-contract.mjs';
import { declaredSyntheticOwnedAccounting } from './flux3-owned-accounting-fixture.mjs';
const fixture = () => declaredSyntheticOwnedAccounting({target:{provider:'flux3_klein_base',tier:'bf16'},strategy:{name:'DECLARED_SYNTHETIC',parameters:{offload:'none'}}});
test('typed sampled accounting validates without claiming allocator/residency/fit or reusing old metric names', () => {
  const record=fixture(); validateOwnedProcessAccountingRecord(record);
  assert.equal(record.observedMemory,undefined);assert.equal(record.observedOperatingConditions.fitAuthorization,false);
  assert.equal(record.ownedProcessMemory.unavailableMetrics.physicalResidentBytes,null);
  assert.equal(record.observedExecutionRecipe.measuredStrategySha256,ownedAccountingDigest('flux3_observed_memory_strategy_v1\0',record.strategy));
});
test('foreign count and allocation consistency matches Rust without rejecting valid zero accounting', () => {
  const empty=fixture();Object.assign(empty.observedOperatingConditions,{foreignProcessCount:0,foreignDedicatedBytes:0,foreignMaxProcessBytes:0});validateOwnedProcessAccountingRecord(empty);
  const context=fixture();Object.assign(context.observedOperatingConditions,{foreignProcessCount:1,foreignDedicatedBytes:0,foreignMaxProcessBytes:0});validateOwnedProcessAccountingRecord(context);
  const reported=fixture();reported.observedOperatingConditions.foreignProcessCount=0;
  assert.equal(reported.observedOperatingConditions.foreignDedicatedBytes,633339904);
  assert.throws(()=>validateOwnedProcessAccountingRecord(reported),/zero foreign process count/);
});
const mutations = [
  ['unknown kind',r=>r.measurementKind='other'],['wrong backend',r=>r.backend='mlx'],['other family',r=>r.target.provider='flux2'],
  ['legacy observedMemory',r=>r.observedMemory={overall:{activeBytes:100}}],['legacy diagnostic alias',r=>r.diagnostics={metrics:{denoiseDevicePeakDelta:100}}],
  ['physical residency claim',r=>r.ownedProcessMemory.physicalResidencyClaim=true],['allocator claim',r=>r.ownedProcessMemory.allocatorActivityClaim=true],
  ['allocator fake number',r=>r.ownedProcessMemory.unavailableMetrics.allocatorBytes=100],['unknown memory field',r=>r.ownedProcessMemory.unknown=true],
  ['missing phase',r=>delete r.ownedProcessMemory.phases.denoise],['unbounded cadence',r=>r.ownedProcessMemory.sampling.intervalMs=5000],
  ['missing skew evidence',r=>delete r.ownedProcessMemory.sampling.maxObservedReadSkewMs],['excess skew',r=>r.ownedProcessMemory.sampling.maxObservedReadSkewMs=251],
  ['lossless peak claim',r=>r.ownedProcessMemory.sampling.losslessPeakClaim=true],['duplicate nodes',r=>r.ownedProcessMemory.physicalNodes=[0,0]],
  ['wrong LUID',r=>r.ownedProcessMemory.identity.luid='invalid'],['wrong ordinal',r=>r.ownedProcessMemory.identity.logicalCudaOrdinal=1],
  ['invalid PID',r=>r.ownedProcessMemory.identity.pid=0],['negative bytes',r=>r.ownedProcessMemory.phases.denoise.dedicatedPeakBytes=-1],
  ['missing intervals',r=>r.ownedProcessMemory.phases.denoise.samples=1],['excess gap',r=>r.ownedProcessMemory.phases.denoise.maxGapMs=1501],
  ['boundary mixup',r=>r.ownedProcessMemory.phases.denoise.firstSampleMs=401],['overall missing phase peak',r=>r.ownedProcessMemory.overall.dedicatedPeakBytes=99],
  ['unknown recipe',r=>r.observedExecutionRecipe.unknown=true],['strategy choice changed',r=>r.strategy.parameters.offload='cpu'],
  ['strategy digest forged',r=>r.observedExecutionRecipe.measuredStrategySha256='f'.repeat(64)],['recipe digest changed',r=>r.observedExecutionRecipeSha256='f'.repeat(64)],
  ['recipe provider changed',r=>r.observedExecutionRecipe.provider='flux3_klein'],['conditions device changed',r=>r.observedOperatingConditions.physicalGpuId='1'],
  ['fit authorization',r=>r.observedOperatingConditions.fitAuthorization=true],['exclusive performance',r=>r.observedOperatingConditions.exclusiveThroughputClaim=true],
  ['phase headroom claim',r=>r.observedOperatingConditions.observationScope='whole-capture-minimum'],['no headroom',r=>r.observedOperatingConditions.freeDeviceBytes=0],
  ['foreign process over cap',r=>r.observedOperatingConditions.foreignMaxProcessBytes=1024**3+1],['foreign total over cap',r=>r.observedOperatingConditions.foreignDedicatedBytes=3*1024**3+1],
  ['zero foreign count with positive total and maximum',r=>r.observedOperatingConditions.foreignProcessCount=0],
  ['zero foreign count with positive total only',r=>{r.observedOperatingConditions.foreignProcessCount=0;r.observedOperatingConditions.foreignMaxProcessBytes=0;}],
  ['unknown conditions',r=>r.observedOperatingConditions.relax=true],
];
for(const [name,mutate]of mutations)test(`typed accounting rejects ${name}`,()=>{const r=fixture();mutate(r);assert.throws(()=>validateOwnedProcessAccountingRecord(r));});
