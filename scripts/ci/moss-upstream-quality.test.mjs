import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,rm} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve,matchesGlob} from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync,spawnSync} from 'node:child_process';
import {envelope,PRIOR_PAYLOAD,CASES_STAGED,CORRECTED_STAGED,CORRECTED_MANIFEST_SHA256,verifyCasePayload,authenticateModels} from './moss-upstream-quality.mjs';

const ref='a'.repeat(40),upload={phase:'upload-cases',artifact_run_id:null,artifact_id:null};
test('three-case suite accepts only one closed upload/execute artifact handoff',()=>{
  assert.deepEqual(envelope(JSON.stringify(upload),ref,ref,'mlx'),upload);
  assert.equal(envelope(JSON.stringify({phase:'execute-cases',artifact_run_id:'40191507328',artifact_id:'12598132624'}),ref,ref,'mlx').phase,'execute-cases');
  for(const bad of [{...upload,case:'sft-four'},{...upload,phase:'execute-cases'},{...upload,artifact_id:'12598132624'},{...upload,phase:'execute-cases',artifact_run_id:'1;id',artifact_id:'12598132624'},{...upload,phase:'execute'}])assert.throws(()=>envelope(JSON.stringify(bad),ref,ref,'mlx'));
  for(const args of [[ref,'b'.repeat(40),'mlx'],['main',ref,'mlx'],[ref,ref,'cuda']])assert.throws(()=>envelope(JSON.stringify(upload),...args));
});
test('new suite keeps original frozen learned payload and sealed new data separate',()=>{
  assert.equal(PRIOR_PAYLOAD,'/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-actions-r15/payload');
  assert.equal(CASES_STAGED,'/Users/michael/.codex/epic-runs/sc-26752/h22-upstream-comparison-prep/payload');
});
test('sealed case data binds both actual Base/SFT model views without rehashing retained weights',{skip:!existsSync(CASES_STAGED)},async()=>{
  const {cases,models}=await verifyCasePayload(CASES_STAGED);
  assert.deepEqual(cases.cases.map(item=>item.id),['base-four','sft-single','sft-four']);
  const root=await mkdtemp(resolve(tmpdir(),'moss-h22-auth-'));
  try {
    const result=await authenticateModels(root,CASES_STAGED,{base:models.base.plannedMappings,sft:models.sft});
    assert.deepEqual(Object.keys(result.views),['base','sft']);
    assert.equal(result.identities.length,25);
    await result.assertStable();
  } finally {await rm(root,{recursive:true,force:true});}
});
test('real CLI upload-cases validate completes with the sealed payload and Actions outputs',{skip:!existsSync(CASES_STAGED)},async()=>{
  const source=resolve(new URL('../../',import.meta.url).pathname),head=execFileSync('git',['-C',source,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  const root=await mkdtemp(resolve(tmpdir(),'moss-h23-cli-'));
  try {
    const output=resolve(root,'github-output'),result=spawnSync(process.execPath,[resolve(source,'scripts/ci/moss-upstream-reference.mjs'),'validate'],{
      cwd:source,encoding:'utf8',timeout:15000,env:{...process.env,GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:head,SOURCE_REF:head,INPUT_BACKEND:'mlx',GITHUB_RUN_ID:'40217574855',GITHUB_RUN_ATTEMPT:'1',GITHUB_OUTPUT:output,MOSS_REFERENCE:JSON.stringify(upload)},
    });
    assert.equal(result.status,0,`reference CLI did not settle: ${result.stderr}`);
    assert.match(await readFile(output,'utf8'),/^phase=upload-cases\n[\s\S]*payload=\/Users\/michael\/\.codex\/epic-runs\/sc-26752\/h22-upstream-comparison-prep\/payload\n$/);
  } finally {await rm(root,{recursive:true,force:true});}
});
test('registered MPS route queues one three-case execute suite and retains each case result',async()=>{
  const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8');
  const route=workflow.slice(workflow.indexOf('  moss-upstream-reference:'),workflow.indexOf('  moss-dispatch:'));
  assert.match(route,/group: memory-catalog-campaign-mlx-nax/);
  assert.match(route,/cancel-in-progress: false/);
  assert.match(route,/phase == 'upload-cases'/);
  assert.match(route,/phase == 'execute-cases'/);
  for(const id of ['base-four','sft-single','sft-four'])assert.match(route,new RegExp(`upstream-${id}\\.png`));
  assert.doesNotMatch(route,/nvidia-smi|git push|gh pr|cargo build/);
});


test('corrected four-case diagnostic has a separate closed phase pair',()=>{
  const value={phase:'upload-corrected-cases',artifact_id:null,artifact_run_id:null};
  assert.deepEqual(envelope(JSON.stringify(value),ref,ref,'mlx',true),value);
  assert.equal(envelope(JSON.stringify({phase:'execute-corrected-cases',artifact_id:'12701291893',artifact_run_id:'40428788379'}),ref,ref,'mlx',true).phase,'execute-corrected-cases');
  for(const bad of [upload,{...value,proof:'unbound'},{...value,artifact_id:'12701291893'},{...value,phase:'execute-corrected-cases'},{...value,phase:'execute-corrected-cases',artifact_id:'future',artifact_run_id:'40428788379'}])assert.throws(()=>envelope(JSON.stringify(bad),ref,ref,'mlx',true));
  assert.throws(()=>envelope(JSON.stringify(value),ref,ref,'mlx'));
  assert.throws(()=>envelope(JSON.stringify(value),ref,'b'.repeat(40),'mlx',true));
});
test('corrected sealed four cases preserve original cases and unchanged source/model/runtime inputs',{skip:!existsSync(CORRECTED_STAGED)},async()=>{
  const old=await verifyCasePayload(CASES_STAGED),current=await verifyCasePayload(CORRECTED_STAGED,CORRECTED_MANIFEST_SHA256);
  assert.deepEqual(current.cases.cases.map(item=>item.id),['base-single','base-four','sft-single','sft-four']);
  assert.deepEqual(current.cases.cases.slice(1),old.cases.cases);
  assert.deepEqual(current.cases.cases[0],{...old.cases.cases[1],id:'base-single',rowId:'quality-mlx-base-single',variant:'base'});
  assert.deepEqual(current.models,old.models);
  for(const item of old.manifest.files.filter(item=>item.file!=='cases.json'))assert.deepEqual(current.manifest.files.find(next=>next.file===item.file),item);
});
test('actual Python scoped encoder correction constructs Bool causal mask and restores on every exit',()=>{
  const wrapper=new URL('./moss-upstream-quality.py',import.meta.url).pathname;
  const python=process.env.MOSS_TEST_PYTHON??'python3';
  const result=spawnSync(python,['-B','-c',`
import importlib.util, types
spec=importlib.util.spec_from_file_location('quality', ${JSON.stringify(wrapper)})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tensor:
    def __init__(self, shape, tag='v0', dtype='bf16'):
        self.shape=shape;self.tag=tag;self.dtype=dtype;self.ndim=len(shape);self.device=types.SimpleNamespace(type='mps')
    def tril(self):
        self.triangle=True;return self
    def repeat_interleave(self, count, dim):
        assert dim==-3 and count==4;return Tensor((1,16,71,256),self.tag)
    def __getitem__(self, key): return self.tag
q=Tensor((1,16,71,256));k=Tensor((1,4,71,256));v=Tensor((1,4,71,256))
observed=[]
def sdpa(query,key,value,**kw):
    observed.append(kw)
    if kw['is_causal'] is False:
        assert kw['attn_mask'].triangle and kw['attn_mask'].shape==(71,71)
    return Tensor(query.shape)
f=types.SimpleNamespace(scaled_dot_product_attention=sdpa)
torch=types.SimpleNamespace(bfloat16='bf16',bool='bool',equal=lambda a,b:a==b,
    ones=lambda shape,**kw:Tensor(shape,dtype=kw['dtype']))
retained=[]
def encode():
    return f.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
a=types.SimpleNamespace(encode_prompts=encode)
with m.corrected_prompt_encoding(a,f,torch,lambda result,calls:retained.extend(calls)):
    a.encode_prompts()
    assert f.scaled_dot_product_attention is sdpa # restored before denoiser
assert a.encode_prompts is encode and f.scaled_dot_product_attention is sdpa
assert len(observed)==len(retained)==1 and retained[0]['firstQueryEqualsV0']
assert observed[0]['enable_gqa'] is True and observed[0]['is_causal'] is False
# Reject invalid input instead of silently altering any different attention contract.
for field,value in [('dtype','f32'),('shape',(1,16,70,256))]:
    original=getattr(q,field);setattr(q,field,value)
    try:
        with m.corrected_prompt_encoding(a,f,torch,lambda *_:None):a.encode_prompts()
        raise AssertionError('bad input accepted')
    except ValueError:pass
    finally:setattr(q,field,original)
    assert a.encode_prompts is encode and f.scaled_dot_product_attention is sdpa
# Encoder and retention failures each restore both patch points.
for fail_encode in (True,False):
    def failing():
        if fail_encode:raise RuntimeError('encode')
        return encode()
    a.encode_prompts=failing
    def retain(*_):raise RuntimeError('retain')
    try:
        with m.corrected_prompt_encoding(a,f,torch,retain):a.encode_prompts()
        raise AssertionError('failure swallowed')
    except RuntimeError:pass
    assert a.encode_prompts is failing and f.scaled_dot_product_attention is sdpa
# Actual first-query counterexample must fail even though attention returned.
a.encode_prompts=encode
torch.equal=lambda *_:False
try:
    with m.corrected_prompt_encoding(a,f,torch,lambda *_:None):a.encode_prompts()
    raise AssertionError('causal violation accepted')
except ValueError as error:assert 'V0' in str(error)
assert a.encode_prompts is encode and f.scaled_dot_product_attention is sdpa
print('scoped correction CPU stubs PASS; no Torch/model import')
`],{encoding:'utf8',timeout:10000});
  assert.equal(result.status,0,result.stderr);
  assert.match(result.stdout,/CPU stubs PASS/);
});
test('corrected upload validates via the existing quality CLI with no extra inputs',{skip:!existsSync(CORRECTED_STAGED)},async()=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-corrected-cli-'));
  try {
    const output=resolve(root,'github-output'),value={phase:'upload-corrected-cases',artifact_id:null,artifact_run_id:null};
    const result=spawnSync(process.execPath,[new URL('./moss-upstream-quality.mjs',import.meta.url).pathname,'validate'],{encoding:'utf8',timeout:15000,env:{...process.env,GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:ref,SOURCE_REF:ref,INPUT_BACKEND:'mlx',GITHUB_RUN_ID:'40428788379',GITHUB_RUN_ATTEMPT:'1',GITHUB_OUTPUT:output,MOSS_REFERENCE:JSON.stringify(value)}});
    assert.equal(result.status,0,result.stderr);
    assert.match(await readFile(output,'utf8'),/phase=upload-corrected-cases/);
    assert.match(await readFile(output,'utf8'),/corrected-upstream-quality-prep\/payload/);
  } finally {await rm(root,{recursive:true,force:true});}
});


test('workflow retains actual helper-generated conditioning wires while excluding weights and caches',async()=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-conditioning-retention-'));
  try {
    const wrapper=new URL('./moss-upstream-quality.py',import.meta.url).pathname;
    const result=spawnSync(process.env.MOSS_TEST_PYTHON??'python3',['-B','-c',`
import importlib.util, types, struct
from pathlib import Path
spec=importlib.util.spec_from_file_location('quality', ${JSON.stringify(wrapper)})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tensor:
    def __init__(self, shape, payload):self.shape=shape;self.payload=payload
    def detach(self):return self
    def to(self, **kw):return self
    def contiguous(self):return self
    def numpy(self):return self
    def tobytes(self, order):assert order=='C';return self.payload
condition=types.SimpleNamespace(embeds=Tensor((1,2,2),struct.pack('<4f',1,2,3,4)),
    attention_mask=Tensor((1,2),bytes([1,1])))
computed=types.SimpleNamespace(positive=condition,negative=condition)
torch=types.SimpleNamespace(float32='f32',uint8='u8')
for case in ('base-single','base-four','sft-single','sft-four'):
    m.retain_conditioning(computed,[],Path(${JSON.stringify(root)})/f'upstream-{case}.png',torch)
assert 'torch' not in __import__('sys').modules
`],{encoding:'utf8',timeout:10000});
    assert.equal(result.status,0,result.stderr);
    const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8');
    const route=workflow.slice(workflow.indexOf('  moss-upstream-reference:'),workflow.indexOf('  moss-dispatch:'));
    const patterns=route.split('\n').map(line=>line.trim()).filter(line=>line.includes('/upstream-*.conditioning/')).map(line=>line.replace('${{ steps.reference.outputs.root }}',root));
    assert.deepEqual(patterns.map(pattern=>pattern.slice(root.length)),['/upstream-*.conditioning/*.json','/upstream-*.conditioning/*.f32le','/upstream-*.conditioning/*.u8']);
    for(const id of ['base-single','base-four','sft-single','sft-four']){
      const folder=resolve(root,`upstream-${id}.conditioning`),wire=JSON.parse(await readFile(resolve(folder,'wire.json')));
      assert.deepEqual(wire.files.map(item=>item.file),['positive.f32le','positive.u8','negative.f32le','negative.u8']);
      for(const file of ['wire.json',...wire.files.map(item=>item.file)])assert.ok(patterns.some(pattern=>matchesGlob(resolve(folder,file),pattern)),`${id}/${file} not retained`);
      for(const item of wire.files){
        const bytes=await readFile(resolve(folder,item.file));
        assert.equal(bytes.length,item.byteLength);
        assert.equal(createHash('sha256').update(bytes).digest('hex'),item.sha256);
      }
    }
    for(const file of ['weights/model.safetensors','cache/tensor.f32le','upstream-base-single.conditioning/weights.safetensors','upstream-base-single.conditioning/cache/tensor.f32le','upstream-base-single.conditioning/hf/model.bin'])assert.ok(!patterns.some(pattern=>matchesGlob(resolve(root,file),pattern)),`${file} must not be retained`);
  } finally {await rm(root,{recursive:true,force:true});}
});
