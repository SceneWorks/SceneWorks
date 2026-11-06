import assert from 'node:assert/strict';
import test from 'node:test';
import { validateResidencyComponentCoverage,residencyComponentSourceApplicability,RESIDENCY_COMPONENT_SOURCE as SOURCE } from './h3-residency-component-evidence.mjs';
// Pure receipt-shape fixtures only: these hashes/metrics are not measured GPU evidence.
const manifest={"schemaVersion":1,"modelRevision":"137ce668c55a20bc0935fd1cf2a3de8448abb7f4","turboRevision":"5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1","transformerDir":"D:\\Codex\\work\\sw-f329-acceptance-42243645213-1\\staged\\hub\\models--SceneWorks--minimax-h3-mlx\\snapshots\\137ce668c55a20bc0935fd1cf2a3de8448abb7f4\\q4\\transformer","turboPath":"D:\\Codex\\work\\sw-f329-acceptance-42243645213-1\\staged\\hub\\models--lightx2v--Minimax-h3-Turbo\\snapshots\\5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1\\minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors","acceptedStagingReceipt":{"path":"D:\\Codex\\work\\sw-f329-acceptance-42243645213-1\\staged-models.json","sha256":"a58d6f7e2d2655a5537b5f7deac02fb95694d5e0a1ec1d47161a14801223d6dd"},"expectedConfig":{"bytes":613,"sha256":"9208a8892c74a15dcac6828039a528ba13ba2eac0b708ee7d26caef59237ee8d"},"expectedTurbo":{"bytes":1383677768,"sha256":"e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"},"expectedShards":[{"name":"diffusion_pytorch_model-00001-of-00014.safetensors","bytes":1488457995,"sha256":"6d5ba9b4acfc9246cd1df7b3e157a9b101c4458b7ef788ff0e1ec76321831a71"},{"name":"diffusion_pytorch_model-00002-of-00014.safetensors","bytes":1323111315,"sha256":"95321756ababc90fb758cd39fcd2d0366d326b45022cda5cd310236a3344da9b"},{"name":"diffusion_pytorch_model-00003-of-00014.safetensors","bytes":1388138401,"sha256":"f5e70d6d8bc871455944e8f9da7f4d9dd3a0574b124c39226d153edebb2c0797"},{"name":"diffusion_pytorch_model-00004-of-00014.safetensors","bytes":1284962984,"sha256":"5e16bc143946dd2a3a7d6aac4a3277e9ac34431f149f80aee51e88c8b6df05f0"},{"name":"diffusion_pytorch_model-00005-of-00014.safetensors","bytes":1323111403,"sha256":"601fef51b7e0538a425b441f77c38be8b18952442a19324ce03d0d28574f5c8f"},{"name":"diffusion_pytorch_model-00006-of-00014.safetensors","bytes":1388138485,"sha256":"d12865ea0ffd48cfc7d311f8fa8953ec832daf9048f27b8564f22a691bd2f7ad"},{"name":"diffusion_pytorch_model-00007-of-00014.safetensors","bytes":1284962964,"sha256":"a3d7a8aee94ecdd06bb072b635a7c56b53aba7fbb7dd19a87f0c96dcedf1907f"},{"name":"diffusion_pytorch_model-00008-of-00014.safetensors","bytes":1323111415,"sha256":"62ba99fc9679e154a328aaa6170cf74d0c47bdd12dfe6407a92367feb6d16960"},{"name":"diffusion_pytorch_model-00009-of-00014.safetensors","bytes":1388138421,"sha256":"e3fe77de684132940effbfc2479d6e123e40702606b11637366512b2d0a104f1"},{"name":"diffusion_pytorch_model-00010-of-00014.safetensors","bytes":1284962958,"sha256":"580fe2760e245184c6a14c3f842da4aa8e08be2cad88ad352f5bd17f7b52a374"},{"name":"diffusion_pytorch_model-00011-of-00014.safetensors","bytes":1323111395,"sha256":"186000b10654fdbf6c4edb0bdfa406b1847e5d0bf691879d8fe507fe6a37f2b8"},{"name":"diffusion_pytorch_model-00012-of-00014.safetensors","bytes":1388138445,"sha256":"bce82195aaae6c4984f51f1adf2ab32d81da0f41ec228b014c6327d79a533611"},{"name":"diffusion_pytorch_model-00013-of-00014.safetensors","bytes":1284963008,"sha256":"d15dcbc77a87acf2ee3a140a1b259c0255a99a4cdff544dfcd702f32aee1f5e3"},{"name":"diffusion_pytorch_model-00014-of-00014.safetensors","bytes":1306661489,"sha256":"bbbdefdbafe5feb98d48841f9601772df0e1eeefd0a6d26b2cd4b214de21bd71"}]};
const reviewedDelta=":100644 100644 edd41ea657b38a12f65360e91daf989defde97b8 f2c834715278eee0a87ee7e4ff263ecb963b5e65 M\tcrates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers.rs\n:000000 100644 0000000000000000000000000000000000000000 11544ba01c6526e937641b38925bb045df77b9ff A\tcrates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers/residency_component.rs\n:100644 100644 89dc364c24bc3da9c0b086ac3ac70ecdd53511ea 17b5ca7d8dd09c3e2c72707f6f823e72dbd79045 M\tcrates/media/candle-gen/candle-gen/src/quant/adapt.rs\n:000000 100644 0000000000000000000000000000000000000000 c0c0b2ec92e3bf59d086f83b9a4520628d6aef5d A\tdocs/testing/h3-residency-component.md\n:100644 100644 218cfd06259b2008a1fa7f4292df52f8289d39fa e3f47600c77cfd034fb8413f248176e31c7d4de9 M\tscripts/ci/model-inference-proofs.json";
const SHA='a'.repeat(64),QK=[1,104554,56,128],PROJECTION=[1,104554,7168];
const options={proofRevision:SOURCE.proof};
function metric(shape){return {shape:[...shape],dtype:'BF16',elements:shape.reduce((a,b)=>a*b,1),leftSha256:SHA,rightSha256:SHA,exact:true,maxAbs:0,referencePeak:1,relativePeak:0,rms:0,nonfinite:0};}
const identity=shape=>({shape,dtype:'BF16',sha256:SHA});
function fixture(){
 const m=structuredClone(manifest),asset=(repo,revision,relative,path,x)=>({repo,revision,relative,path,bytes:x.bytes,sha256:x.sha256});
 return {kind:'h3-residency-component/v1',schemaVersion:1,accepted:true,completedLearnedProjectionCalls:6,completedResidualOnlyScopes:2,
 device:{logicalOrdinal:0,computeCapability:[12,0],cudaVisibleDevices:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',proofPoolInstalled:false,stream:'0x12345'},
 source:{manifestSha256:SOURCE.manifestSha256,expectedManifestSha256:SOURCE.manifestSha256,manifest:m,
 configSha256:m.expectedConfig.sha256,turboSha256:m.expectedTurbo.sha256,acceptedSourcePair:{sceneworks:SOURCE.consumer,inference:SOURCE.runtime},
 verifiedAssets:[asset('SceneWorks/minimax-h3-mlx',m.modelRevision,'q4/transformer/config.json',m.transformerDir+'/config.json',m.expectedConfig),asset('lightx2v/Minimax-h3-Turbo',m.turboRevision,'minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors',m.turboPath,m.expectedTurbo),...m.expectedShards.map(s=>asset('SceneWorks/minimax-h3-mlx',m.modelRevision,'q4/transformer/'+s.name,m.transformerDir+'/'+s.name,s))],
 projection:'transformer_blocks.0.attn.to_q',selectedSourceHeader:[
 {name:'transformer_blocks.0.attn.norm_q.weight',dtype:'BF16',shape:[128]},
 {name:'transformer_blocks.0.attn.to_q.biases',dtype:'BF16',shape:[7168,84]},
 {name:'transformer_blocks.0.attn.to_q.scales',dtype:'BF16',shape:[7168,84]},
 {name:'transformer_blocks.0.attn.to_q.weight',dtype:'U32',shape:[7168,672]}],
 runtimePacked:{hostSha256:SHA,directSha256:SHA,copiedSha256:SHA},canonicalTurboFactors:[{
 hostVsDirectA:metric([5376,128]),hostVsCopiedA:metric([5376,128]),hostVsDirectB:metric([128,7168]),hostVsCopiedB:metric([128,7168]),hostScale:1,directScale:1,copiedScale:1,strength:1,aShape:[5376,128],bShape:[128,7168]}],normQ:identity([128])},
 input:{identity:'integer-lcg-mod65521/seed42/bf16-once/v1',tensor:identity([1,104554,5376])},
 preparation_same_policy:{userWorkspaceBytes:33554432,base:metric(PROJECTION),residual:metric(PROJECTION),full:metric(PROJECTION)},
 workspace_default_vs_12m:metric(PROJECTION),workspace_default_vs_32m:metric(PROJECTION),workspace_12m_vs_32m:metric(PROJECTION),
 workspaceControls:{libraryDefaultUserBytes:null,explicitUserBytes:[12582912,33554432]},
 qk_full_vs_bounded:{norm:metric(QK),rotary:metric(QK),shape:QK,budgetElements:33554432,normChunks:23,rotaryChunks:23,tokensPerChunk:4681,tailTokens:1572,maximumChunkElements:33553408,positionIdentity:'deterministic-three-axis-mod102x24x42/v1',effectiveCos:identity([104554,96]),effectiveSin:identity([104554,96])},
 assertions:{samePolicyPreparationExact:true,fullBoundedQkExact:true,workspaceObservationsFinite:true,noAttentionPredictorDenoiseVaeOrClip:true}};
}
test('fixed production component coverage is narrow and workspace contrasts remain observations',()=>{
 const value=fixture(),covered=validateResidencyComponentCoverage(value,options);
 assert.equal(covered.boundaries.length,3);assert.ok(covered.exclusions.includes('overall E5 acceptance'));assert.equal(covered.passed,undefined);
 // Nonzero workspace differences never become a fabricated BF16 or media threshold.
 value.workspace_default_vs_12m.leftSha256='b'.repeat(64);value.workspace_default_vs_32m.leftSha256='b'.repeat(64);
 for(const key of ['workspace_default_vs_12m','workspace_default_vs_32m'])Object.assign(value[key],{exact:false,maxAbs:0.5,referencePeak:1,relativePeak:0.5,rms:0.1});
 validateResidencyComponentCoverage(value,options);
 value.source.verifiedAssets[0].path='\\\\?\\'+value.source.verifiedAssets[0].path.replaceAll('/','\\');validateResidencyComponentCoverage(value,options);
});
test('accepted true cannot replace actual CUDA, complete counts, frozen sources or exact component results',()=>{
 const mutations=[
 v=>v.kind='other',v=>v.schemaVersion=2,v=>v.accepted=false,
 v=>v.completedLearnedProjectionCalls=0,v=>v.completedLearnedProjectionCalls=5,v=>v.completedResidualOnlyScopes=1,
 v=>v.device.logicalOrdinal=1,v=>v.device.computeCapability=[0,0],v=>v.device.proofPoolInstalled=true,v=>v.device.cudaVisibleDevices='cpu',v=>v.device.stream='0x0',
 v=>v.source.manifestSha256='f'.repeat(64),v=>v.source.acceptedSourcePair.inference=SOURCE.proof,v=>v.source.manifest.modelRevision='f'.repeat(40),
 v=>v.source.verifiedAssets=[],v=>v.source.verifiedAssets[0].bytes=0,v=>v.source.verifiedAssets[0].sha256=SHA,v=>v.source.verifiedAssets[0].path='C:/foreign/config.json',
 v=>v.source.runtimePacked.copiedSha256='b'.repeat(64),v=>delete v.source.canonicalTurboFactors[0].hostVsCopiedB,
 v=>v.source.canonicalTurboFactors[0].copiedScale=0,v=>v.source.canonicalTurboFactors[0].hostVsCopiedA.elements=1,
 v=>v.source.selectedSourceHeader[0].dtype='F32',v=>v.input.tensor.dtype='F32',
 v=>v.preparation_same_policy.userWorkspaceBytes=12582912,
 v=>Object.assign(v.preparation_same_policy.base,{exact:false,rightSha256:'b'.repeat(64),maxAbs:0.1,rms:0.1,relativePeak:0.1}),
 v=>v.preparation_same_policy.residual.rms=1,v=>v.preparation_same_policy.full.nonfinite=1,
 v=>v.qk_full_vs_bounded.norm.dtype='F16',v=>v.qk_full_vs_bounded.shape=[1,2,56,128],v=>v.qk_full_vs_bounded.normChunks=1,
 v=>v.qk_full_vs_bounded.rotaryChunks=0,v=>v.qk_full_vs_bounded.tailTokens=0,v=>v.qk_full_vs_bounded.maximumChunkElements=1,
 v=>v.qk_full_vs_bounded.rotary.elements=1,v=>v.qk_full_vs_bounded.rotary.rightSha256='b'.repeat(64),
 v=>Object.assign(v.qk_full_vs_bounded.norm,{exact:false,rightSha256:'b'.repeat(64),maxAbs:0.1,rms:0.1,relativePeak:0.1}),
 v=>v.workspace_default_vs_32m.rms=NaN,v=>v.workspace_12m_vs_32m.relativePeak=Infinity,
 v=>v.workspace_default_vs_12m.nonfinite=1,v=>v.workspaceControls.explicitUserBytes=[1],
 v=>v.assertions.noAttentionPredictorDenoiseVaeOrClip=false,
 ];
 for(const [index,mutate] of mutations.entries()){const value=fixture();mutate(value);assert.throws(()=>validateResidencyComponentCoverage(value,options),`mutation${index} must refuse`);}
 assert.throws(()=>validateResidencyComponentCoverage(fixture(),{proofRevision:'f'.repeat(40)}));
 assert.throws(()=>validateResidencyComponentCoverage(fixture(),{...options,runtimeSourcePair:{sceneworks:SOURCE.consumer,inference:SOURCE.proof}}));
});
function gitCommand(change=()=>{}){return async(program,args)=>{
 assert.equal(program,'git');assert.deepEqual(args.slice(0,2),['-C','fixture']);const command=args[2];
 let stdout=command==='rev-list'?`${SOURCE.proof} ${SOURCE.runtime}`:command==='rev-parse'?(args[3].startsWith(SOURCE.runtime)?'75d392586953d990cf346c7b82f69f7ef246b133':'effe9c1152f534b6f746713de59083e6c17a5035'):reviewedDelta;
 return {stdout:change(command,stdout)??stdout};
};}
test('only exact reviewed C1 child maps component production applicability, never other proof guards',async()=>{
 const revisions={proofRevision:SOURCE.proof,runtimeRevision:SOURCE.runtime};
 const result=await residencyComponentSourceApplicability('fixture',revisions,gitCommand());
 assert.equal(result.directParent,true);assert.match(result.reviewedExactDeltaSha256,/^[a-f0-9]{64}$/);assert.ok(result.exclusions.includes('whole-tree equivalence'));
 for(const [command,change] of [['rev-list',s=>s+' '+SOURCE.runtime],['rev-parse',s=>'f'.repeat(40)],['diff-tree',s=>s.replace('11544ba','21544ba')],['diff-tree',s=>s+'\n:unexpected production path']])await assert.rejects(residencyComponentSourceApplicability('fixture',revisions,gitCommand((c,s)=>c===command?change(s):s)));
 await assert.rejects(residencyComponentSourceApplicability('fixture',{...revisions,proofRevision:'f'.repeat(40)},gitCommand()));
 await assert.rejects(residencyComponentSourceApplicability('fixture',{...revisions,runtimeRevision:SOURCE.proof},gitCommand()));
});
