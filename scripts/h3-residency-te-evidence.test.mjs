import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { sha256 } from './lib/file-sha256.mjs';
import { validateResidencyTextEncoderCoverage } from './h3-residency-te-evidence.mjs';
import { RESIDENCY_COMPONENT_SOURCE as SOURCE } from './h3-residency-component-evidence.mjs';
// Actual metadata fixture from accepted run42295556319/source28329; no tensor/model payloads.
const raw=await readFile(new URL('./fixtures/h3-residency-te-C4.json',import.meta.url));
const actual=JSON.parse(raw);
// Exact fields consumed from the actual generic receipts. Full immutable originals remain at
// proof-residency-TE-C4/{receipt.json,00000/result.json}; these false stronger claims are deliberate.
const runReceipt={"accepted":true,"selected":1,"completed":1,"source":{"revision":"28329ef9ec9d58e34ea3fb65d7d40d8068aa2f52","tree":"77670bdf615c0991dfccff91e489d5f2c81bc53b"},"acceptance_scope":"source-bound-case-execution","input_grade":"successful-reader-binding","required_input_content_attested":false,"checkpoint_byte_integrity_attested":false,"request_selection":"candle-gen-minimax-h3/candle_gen_minimax_h3/text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference","selected_bindings":["candle-gen-minimax-h3/candle_gen_minimax_h3/text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference"],"cases":[{"binding":"candle-gen-minimax-h3/candle_gen_minimax_h3/text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference","accepted":true}]};
const caseReceipt={"accepted":true,"binding":"candle-gen-minimax-h3/candle_gen_minimax_h3/text_encoder::encoder::residency_te::production_prompt_resident_window_first_difference","source_file_sha256":"5dba681a3dde8d83f6974358688da24811f3cd1e1628002342ff936ecc15a56a","actual_devices":["candle-cuda"],"body_completed":1,"successful_libtest_summaries":1,"resource_verification_pending":false,"physical_execution":{"physical_index":0,"device_uuid":"GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0","post_execution_verified":true},"input_grade":"successful-reader-binding","required_input_content_attested":false,"checkpoint_byte_integrity_attested":false,"acceptance_scope":"source-bound-case-execution"};
const options={proofRevision:SOURCE.proof,runReceipt,caseReceipt};
const fixture=()=>structuredClone(actual);
test('actual C4 TE raw artifact covers exact production prompt/layers/context with honest input scope',()=>{
 assert.equal(sha256(raw),'e389eafbc14c615092306df39441793611c46e943999614b050821e07cb0b7a3');
 const coverage=validateResidencyTextEncoderCoverage(actual,options);
 assert.equal(coverage.passed,undefined);assert.equal(coverage.accepted,undefined);assert.ok(coverage.exclusions.includes('overall E5 acceptance'));
 assert.deepEqual(coverage.inputScope,{grade:'successful-reader-binding',requiredInputContentAttested:false,checkpointByteIntegrityAttested:false});
 assert.equal(coverage.observations.defaultWorkspaceContextExact,true);assert.deepEqual(coverage.observations.discardedCpuTablesExact,[false,false]);
});
test('discarded CPU tables and default-workspace contrasts remain finite observations, not runtime-input tolerances',()=>{
 const value=fixture();
 value.same_policy.tables.discardedCpuCosVsDirectGpu.maxAbs=0.5;value.same_policy.tables.discardedCpuCosVsDirectGpu.rms=0.2;
 Object.assign(value.workspace_default_vs_32m,{leftSha256:'b'.repeat(64),exact:false,maxAbs:1,relativePeak:0.01,rms:0.1});
 const coverage=validateResidencyTextEncoderCoverage(value,options);assert.equal(coverage.observations.defaultWorkspaceContextExact,false);
 value.same_policy.tables.runtimeCos.exact=false;assert.throws(()=>validateResidencyTextEncoderCoverage(value,options),'runtime tables still require exact parity');
});
test('accepted flag cannot replace every layer, runtime table, packed factor/norm, mask or final-context coverage',()=>{
 const mutations=[
 v=>v.kind='other',v=>v.schemaVersion=2,v=>v.accepted=false,v=>v.completedLayerPairs=0,v=>v.completedDefaultResidentLayers=49,
 v=>v.materializedAttentionCalls=149,v=>v.actualF32ReducedPrecision=true,v=>delete v.readback,v=>v.readback.elementBytes=2,
 v=>v.readback.maximumAllowedChunkElements=1048577,v=>v.readback.everyChunkLayoutStartOffset=1,v=>v.readback.everyStorageElementCountEqualsChunk=false,
 v=>v.source.manifestSha256='f'.repeat(64),v=>v.source.manifest.acceptedSourcePair.inference=SOURCE.proof,v=>v.source.requestSha256='f'.repeat(64),
 v=>v.source.verifiedAssets=[],v=>v.source.verifiedAssets[0].bytes=0,v=>v.source.verifiedAssets[0].sha256='f'.repeat(64),v=>v.source.verifiedAssets[0].path='D:/foreign',
 v=>v.input.idsShape=[1,587],v=>v.input.idsSha256='f'.repeat(64),v=>v.input.attentionMask.dtype='F32',v=>v.input.tokens=587,
 v=>v.planning.residentWeights--,v=>v.planning.plannedDevicePeakBytes--,v=>v.planning.workingTerms.hiddenCopies=1,
 v=>v.same_policy.exact=false,v=>v.same_policy.workspaceBytes=12582912,v=>v.same_policy.firstDifference={boundary:'embedding'},
 v=>v.same_policy.embeddings.dtype='BF16',v=>v.same_policy.embeddings.elements=1,
 v=>v.same_policy.tables.runtimeCos.rightSha256='b'.repeat(64),v=>v.same_policy.tables.runtimeSin.rms=0.01,
 v=>v.same_policy.tables.discardedCpuSinVsDirectGpu.nonfinite=1,v=>v.same_policy.masks.streamed.sha256='b'.repeat(64),
 v=>v.same_policy.layers.pop(),v=>v.same_policy.layers[49].index=48,
 v=>v.same_policy.layers[1].input.leftSha256='b'.repeat(64),v=>v.same_policy.layers[0].preparation.exact=false,
 v=>v.same_policy.layers[0].preparation.packedProjections.pop(),v=>v.same_policy.layers[0].preparation.packedProjections[0].preparedSha256='b'.repeat(64),
 v=>v.same_policy.layers[0].preparation.norms.pop(),v=>v.same_policy.layers[0].preparation.norms[0].metrics.elements=1,
 v=>v.same_policy.layers[49].output.shape=[1,1,5120],v=>v.same_policy.context.rightSha256='b'.repeat(64),
 v=>v.workspace_default_vs_32m.rms=NaN,v=>v.workspace_default_vs_32m.referencePeak=Infinity,
 ];
 for(const [index,mutate] of mutations.entries()){const value=fixture();mutate(value);assert.throws(()=>validateResidencyTextEncoderCoverage(value,options),`mutation${index} must refuse`);}
 // Internally exact hashes still must connect embedding -> all50layers -> context.
 for(const key of ['input','output']){const value=fixture(),metric=value.same_policy.layers[1][key];metric.leftSha256=metric.rightSha256='b'.repeat(64);assert.throws(()=>validateResidencyTextEncoderCoverage(value,options));}
});
test('existing generic execution/source/device/post-idle bindings and false global attestations stay strict',()=>{
 for(const mutation of [o=>o.proofRevision=SOURCE.originalProof,o=>o.runtimeSourcePair={sceneworks:SOURCE.consumer,inference:SOURCE.proof},
 o=>delete o.runReceipt,o=>o.runReceipt.accepted=false,o=>o.runReceipt.completed=0,o=>o.runReceipt.source.revision=SOURCE.originalProof,
 o=>o.runReceipt.source.tree='f'.repeat(40),o=>o.runReceipt.acceptance_scope='whole-feature',o=>o.runReceipt.required_input_content_attested=true,
 o=>o.runReceipt.request_selection='other',o=>o.runReceipt.selected_bindings=[],o=>o.runReceipt.cases[0].accepted=false,
 o=>delete o.caseReceipt,o=>o.caseReceipt.accepted=false,o=>o.caseReceipt.binding='other',o=>o.caseReceipt.acceptance_scope='whole-feature',o=>o.caseReceipt.source_file_sha256='f'.repeat(64),
 o=>o.caseReceipt.actual_devices=['cpu'],o=>o.caseReceipt.body_completed=0,o=>o.caseReceipt.successful_libtest_summaries=0,
 o=>o.caseReceipt.resource_verification_pending=true,o=>o.caseReceipt.physical_execution.physical_index=1,
 o=>o.caseReceipt.physical_execution.device_uuid='GPU-e4b79931-7be6-f216-460a-f5405cfafffe',o=>o.caseReceipt.physical_execution.post_execution_verified=false,
 o=>o.caseReceipt.input_grade='none',o=>o.caseReceipt.checkpoint_byte_integrity_attested=true]){
  const changed=structuredClone(options);mutation(changed);assert.throws(()=>validateResidencyTextEncoderCoverage(actual,changed));
 }
});
