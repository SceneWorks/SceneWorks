// Coverage of one registered component witness, never a campaign/E5 acceptance gate.
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { canonicalJson } from './lib/canonical-json.mjs';
import { sha256 } from './lib/file-sha256.mjs';

export const RESIDENCY_COMPONENT_SOURCE=Object.freeze({
  proof:'c5df0dc4aa5b264bca7690af0a2f4c26c9aae3a9',
  runtime:'2ff0353820e370d004305ae99a035ab38b93de30',
  consumer:'c8815c7cd46feba2f82562a315d1a44a6c88a9e8',
  manifestSha256:'9df0a1203b8a20d87b81888cb4edeaa34537f50c558aba239a79af5b3c443b01',
});
const PAIR=Object.freeze({sceneworks:RESIDENCY_COMPONENT_SOURCE.consumer,inference:RESIDENCY_COMPONENT_SOURCE.runtime});
const PREFIX='transformer_blocks.0.attn.to_q',QK=[1,104554,56,128],PROJECTION=[1,104554,7168];
const UUID='GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0';
const hash=value=>assert.match(value??'',/^[a-f0-9]{64}$/,'complete SHA256 required');
const finite=value=>assert.ok(typeof value==='number'&&Number.isFinite(value)&&value>=0,'finite nonnegative metric required');
function identity(value,shape){assert.equal(value?.dtype,'BF16');assert.deepEqual(value.shape,shape);hash(value.sha256);}
function metric(value,shape,exact){
  assert.equal(value?.dtype,'BF16');assert.deepEqual(value.shape,shape);
  assert.equal(value.elements,shape.reduce((a,b)=>a*b,1),'every element must be compared');
  hash(value.leftSha256);hash(value.rightSha256);assert.equal(value.nonfinite,0);
  for(const key of ['maxAbs','referencePeak','relativePeak','rms'])finite(value[key]);
  assert.equal(typeof value.exact,'boolean');assert.equal(value.exact,value.leftSha256===value.rightSha256,'hash/exact consistency');
  if(exact){assert.equal(value.exact,true,'required exact component comparison');for(const key of ['maxAbs','relativePeak','rms'])assert.equal(value[key],0);}
}

export function validateResidencyComponentCoverage(receipt,{proofRevision,runtimeSourcePair=PAIR}={}){
  assert.equal(proofRevision,RESIDENCY_COMPONENT_SOURCE.proof,'unrecognized component proof source');
  assert.deepEqual(runtimeSourcePair,PAIR,'unrecognized runtime source pair');
  assert.equal(receipt?.kind,'h3-residency-component/v1');assert.equal(receipt.schemaVersion,1);assert.equal(receipt.accepted,true);
  assert.equal(receipt.completedLearnedProjectionCalls,6);assert.equal(receipt.completedResidualOnlyScopes,2);
  assert.deepEqual(receipt.readback,{method:'narrow-force_contiguous-to_vec1',maximumAllowedChunkElements:1048576,elementBytes:2,everyChunkLayoutStartOffset:0,everyStorageElementCountEqualsChunk:true},'bounded readback storage/layout contract required');
  const device=receipt.device;assert.equal(device?.logicalOrdinal,0);assert.deepEqual(device.computeCapability,[12,0]);
  assert.equal(device.cudaVisibleDevices,UUID);assert.equal(device.proofPoolInstalled,false);
  assert.match(device.stream??'',/^0x[0-9a-f]+$/i);assert.notEqual(BigInt(device.stream),0n);
  const source=receipt.source;assert.deepEqual(source?.acceptedSourcePair,PAIR);
  assert.equal(source.manifestSha256,RESIDENCY_COMPONENT_SOURCE.manifestSha256);assert.equal(source.expectedManifestSha256,source.manifestSha256);
  assert.equal(sha256(canonicalJson(source.manifest)),'695d7d9f9d397a69b5369bc4dc3118fb2153c931e86969888579d7bddb64985f','frozen manifest content differs');
  const manifest=source.manifest;assert.equal(source.projection,PREFIX);
  assert.equal(source.configSha256,manifest.expectedConfig.sha256);assert.equal(source.turboSha256,manifest.expectedTurbo.sha256);
  const assets=source.verifiedAssets;assert.ok(Array.isArray(assets));assert.equal(assets.length,16);assert.equal(new Set(assets.map(a=>a.path)).size,16);
  const expected=[
    ['SceneWorks/minimax-h3-mlx',manifest.modelRevision,'q4/transformer/config.json',manifest.transformerDir+'\\config.json',manifest.expectedConfig],
    ['lightx2v/Minimax-h3-Turbo',manifest.turboRevision,'minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors',manifest.turboPath,manifest.expectedTurbo],
    ...manifest.expectedShards.map(s=>['SceneWorks/minimax-h3-mlx',manifest.modelRevision,'q4/transformer/'+s.name,manifest.transformerDir+'\\'+s.name,s]),
  ];
  // Canonical Windows asset paths may carry the verbatim prefix; their namespace stays exact.
  const windows=value=>{assert.equal(typeof value,'string');return value.replace(/^\\\\\?\\/,'').replaceAll('/','\\').toLowerCase();};
  for(const [repo,revision,relative,path,commitment] of expected){
    const rows=assets.filter(a=>a.repo===repo&&a.revision===revision&&a.relative===relative);assert.equal(rows.length,1,'complete unique verified asset binding required');
    const asset=rows[0];assert.equal(windows(asset.path),windows(path));assert.equal(asset.bytes,commitment.bytes);assert.ok(Number.isSafeInteger(asset.bytes)&&asset.bytes>0);assert.equal(asset.sha256,commitment.sha256);hash(asset.sha256);
  }
  const headers=[
    {name:'transformer_blocks.0.attn.norm_q.weight',dtype:'BF16',shape:[128]},
    {name:PREFIX+'.biases',dtype:'BF16',shape:[7168,84]},
    {name:PREFIX+'.scales',dtype:'BF16',shape:[7168,84]},
    {name:PREFIX+'.weight',dtype:'U32',shape:[7168,672]},
  ];
  assert.deepEqual([...source.selectedSourceHeader].sort((a,b)=>a.name.localeCompare(b.name)),headers);
  const packed=source.runtimePacked;for(const key of ['hostSha256','directSha256','copiedSha256'])hash(packed[key]);
  assert.equal(packed.hostSha256,packed.directSha256);assert.equal(packed.hostSha256,packed.copiedSha256);
  assert.equal(source.canonicalTurboFactors?.length,1);const factor=source.canonicalTurboFactors[0];
  for(const key of ['hostScale','directScale','copiedScale','strength'])assert.equal(factor[key],1);
  assert.deepEqual(factor.aShape,[5376,128]);assert.deepEqual(factor.bShape,[128,7168]);
  for(const [key,shape] of [['hostVsDirectA',factor.aShape],['hostVsCopiedA',factor.aShape],['hostVsDirectB',factor.bShape],['hostVsCopiedB',factor.bShape]])metric(factor[key],shape,true);
  assert.equal(factor.hostVsDirectA.leftSha256,factor.hostVsCopiedA.leftSha256);assert.equal(factor.hostVsDirectB.leftSha256,factor.hostVsCopiedB.leftSha256);
  identity(source.normQ,[128]);assert.equal(receipt.input?.identity,'integer-lcg-mod65521/seed42/bf16-once/v1');identity(receipt.input.tensor,[1,104554,5376]);
  const preparation=receipt.preparation_same_policy;assert.equal(preparation?.userWorkspaceBytes,33554432);
  for(const key of ['base','residual','full'])metric(preparation[key],PROJECTION,true);
  assert.deepEqual(receipt.workspaceControls?.explicitUserBytes,[12582912,33554432]);assert.equal(receipt.workspaceControls.libraryDefaultUserBytes,null);
  for(const key of ['workspace_default_vs_12m','workspace_default_vs_32m','workspace_12m_vs_32m'])metric(receipt[key],PROJECTION,false);
  assert.equal(receipt.workspace_default_vs_12m.leftSha256,receipt.workspace_default_vs_32m.leftSha256);
  assert.equal(receipt.workspace_default_vs_12m.rightSha256,receipt.workspace_12m_vs_32m.leftSha256);
  assert.equal(receipt.workspace_12m_vs_32m.rightSha256,receipt.workspace_default_vs_32m.rightSha256);
  assert.equal(preparation.full.leftSha256,receipt.workspace_default_vs_32m.rightSha256);
  const qk=receipt.qk_full_vs_bounded;assert.deepEqual(qk?.shape,QK);
  for(const [key,value] of Object.entries({budgetElements:33554432,normChunks:23,rotaryChunks:23,tokensPerChunk:4681,tailTokens:1572,maximumChunkElements:33553408}))assert.equal(qk[key],value);
  metric(qk.norm,QK,true);metric(qk.rotary,QK,true);
  assert.equal(qk.positionIdentity,'deterministic-three-axis-mod102x24x42/v1');
  identity(qk.effectiveCos,[104554,96]);identity(qk.effectiveSin,[104554,96]);
  for(const key of ['samePolicyPreparationExact','fullBoundedQkExact','workspaceObservationsFinite','noAttentionPredictorDenoiseVaeOrClip'])assert.equal(receipt.assertions?.[key],true);
  return {kind:'h3-residency-component-covered-boundaries/v1',proofRevision,runtimeSourcePair:PAIR,manifestSha256:source.manifestSha256,
    boundaries:['actual q4 packed bytes and frozen Turbo factors/scales: host/direct/copied','same32MiB base/residual/full BF16 projection: exact','full-shape original/bounded BF16 Q/K norm and partial RoPE: exact'],
    workspaceContrasts:'finite observations only; no projection tolerance or causal conclusion',
    exclusions:['overall E5 acceptance','full TE/DiT/FFN/audio-path parity','final-media drift acceptance or cause','full-model24GiB fit','primitive whole-tree or rawVeda45 proof reuse']};
}

const DELTA=[
  ':100644 100644 edd41ea657b38a12f65360e91daf989defde97b8 bde0f9cea2fdc90ed4dc58f35fd427d6ad959189 M\tcrates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers.rs',
  ':000000 100644 0000000000000000000000000000000000000000 f4d3b99d82d687405506b048ab75c41561803f43 A\tcrates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers/residency_component.rs',
  ':100644 100644 89dc364c24bc3da9c0b086ac3ac70ecdd53511ea 17b5ca7d8dd09c3e2c72707f6f823e72dbd79045 M\tcrates/media/candle-gen/candle-gen/src/quant/adapt.rs',
  ':000000 100644 0000000000000000000000000000000000000000 c0c0b2ec92e3bf59d086f83b9a4520628d6aef5d A\tdocs/testing/h3-residency-component.md',
  ':100644 100644 218cfd06259b2008a1fa7f4292df52f8289d39fa e3f47600c77cfd034fb8413f248176e31c7d4de9 M\tscripts/ci/model-inference-proofs.json',
].join('\n');
export async function residencyComponentSourceApplicability(inferenceRoot,{proofRevision,runtimeRevision},command=promisify(execFile)){
  assert.equal(proofRevision,RESIDENCY_COMPONENT_SOURCE.proof,'unrecognized component proof source');assert.equal(runtimeRevision,RESIDENCY_COMPONENT_SOURCE.runtime,'unrecognized runtime source');
  const git=async args=>(await command('git',['-C',inferenceRoot,...args])).stdout.trim();
  const ancestry=[
    [proofRevision,'a43054848a1d6f915471dc8625a56e4406243887'],
    ['255322bd1f430696fb1a21d81653a58bf1ac129a','d793f71c69ec07eba061fb0b2b4b975f18dd9e38'],
    ['4986a685c256e1966b97f9538e754a46e4246d47','effe9c1152f534b6f746713de59083e6c17a5035'],
    [runtimeRevision,'75d392586953d990cf346c7b82f69f7ef246b133'],
  ];
  for(const [index,[revision,tree]] of ancestry.entries()){
    assert.equal(await git(['rev-parse',revision+'^{tree}']),tree,'exact reviewed ancestry tree required');
    if(index<ancestry.length-1)assert.equal(await git(['rev-list','--parents','-n','1',revision]),`${revision} ${ancestry[index+1][0]}`,'exact reviewed three-commit ancestry required');
  }
  assert.equal(await git(['diff-tree','--no-commit-id','-r','--raw','--no-abbrev',runtimeRevision,proofRevision]),DELTA,'only the exact reviewed five-file diagnostic delta may map');
  return {kind:'h3-residency-component-runtime-applicability/v1',proofRevision,runtimeRevision,directParent:false,reviewedAncestry:ancestry.map(([revision])=>revision),
    reviewedExactDeltaSha256:sha256(DELTA),productionScope:'Exact reviewed objects: layers.rs adds only the equivalent proof-gated CUDA test child declaration; adapt.rs adds only proof-gated frozen-factor observation. Existing production functions/math unchanged.',
    exclusions:['whole-tree equivalence','primitive whole-tree proof reuse','rawVeda45 proof reuse','full-campaign acceptance']};
}
