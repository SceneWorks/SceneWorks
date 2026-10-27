import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,symlink,rm,realpath} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {envelope,safe,verifyPayload,validateRuntime,PAYLOAD_SHA} from './moss-upstream-reference.mjs';
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const ref='a'.repeat(40),upload={phase:'upload',artifact_id:null,artifact_run_id:null};
test('fixed reference accepts only exact source/mlx and closed typed upload/execute artifact identities',()=>{
 assert.deepEqual(envelope(JSON.stringify(upload),ref,ref,'mlx'),upload);
 assert.equal(envelope(JSON.stringify({phase:'execute',artifact_id:'1234567',artifact_run_id:'12345678'}),ref,ref,'mlx').phase,'execute');
 for(const bad of [{...upload,command:'id'},{...upload,path:'/tmp/x'},{...upload,phase:'cpu'},{...upload,artifact_id:'1234567'},{phase:'execute',artifact_id:null,artifact_run_id:null},{phase:'execute',artifact_id:'1;id',artifact_run_id:'12345678'}])assert.throws(()=>envelope(JSON.stringify(bad),ref,ref,'mlx'));
 for(const args of [[ref,'b'.repeat(40),'mlx'],['main',ref,'mlx'],[ref,ref,'candle']])assert.throws(()=>envelope(JSON.stringify(upload),...args));
 for(const path of ['../escape','/absolute','x/../y','x\\y','x//y'])assert.throws(()=>safe('/owned',path));
});
async function fixture(t){const root=await mkdtemp(resolve(await realpath(tmpdir()),'moss-reference-'));t.after(()=>rm(root,{recursive:true,force:true}));const files=[];for(let index=0;index<45;index++){const file=`f${index}`,bytes=Buffer.from(`frozen ${index}`);await writeFile(resolve(root,file),bytes);files.push({file,byteLength:bytes.length,sha256:digest(bytes)});}const manifest={kind:'sc-26752-frozen-upstream-base-single-reference/v1',actualNativeSource:'a0692191b09813fab859659bf382bd7c91a60140',nativePin:'857f7a1996e3678a0773c3509ac0087522c19ce6',files},bytes=JSON.stringify(manifest);await writeFile(resolve(root,'payload-manifest.json'),bytes);return {root,manifest,expected:digest(bytes)};}
test('authenticated payload refuses changed code, unknown files, symlinks and manifest substitution',async t=>{
 const f=await fixture(t);await verifyPayload(f.root,f.expected);await assert.rejects(()=>verifyPayload(f.root,'0'.repeat(64)),/manifest SHA/);
 await writeFile(resolve(f.root,'f0'),'changed');await assert.rejects(()=>verifyPayload(f.root,f.expected),/bytes/);await writeFile(resolve(f.root,'f0'),'frozen 0');
 await writeFile(resolve(f.root,'unlisted.py'),'untrusted');await assert.rejects(()=>verifyPayload(f.root,f.expected),/unlisted/);await rm(resolve(f.root,'unlisted.py'));
 await rm(resolve(f.root,'f0'));await symlink(resolve(f.root,'f1'),resolve(f.root,'f0'));await assert.rejects(()=>verifyPayload(f.root,f.expected),/linked/);
});
test('reviewed payload manifest fixture pins original native provenance, wrapper, noise and source closure',async()=>{
 const bytes=await readFile(new URL('../fixtures/moss-upstream-reference/payload-manifest.json',import.meta.url)),manifest=JSON.parse(bytes);assert.equal(digest(bytes),PAYLOAD_SHA);assert.equal(manifest.wrapperSha256,'ffe0e18dbf021ca5f8f62cf7cacb54f4b0c2a40c4d6142c862bd4df9d42f248d');assert.equal(manifest.files.length,45);assert.equal(manifest.files.find(x=>x.file==='native-seed173-f32le.bin').sha256,'92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82');
});
test('host runtime attestation refuses other interpreter versions and path escapes',()=>{
 const valid={launcher:'/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-prep-r15/venv/bin/python',versions:{torch:'2.9.1',transformers:'5.5.3',diffusers:'0.38.0'},files:Array.from({length:1001},()=>({path:'/opt/homebrew/example.py',byteLength:1,sha256:'a'.repeat(64)}))};validateRuntime(valid);
 for(const value of [{...valid,launcher:'/tmp/python'},{...valid,versions:{...valid.versions,torch:'other'}},{...valid,files:[{path:'/etc/passwd'}]},{...valid,files:valid.files.map(x=>({...x,path:'/opt/homebrew/../../etc/passwd'}))}])assert.throws(()=>validateRuntime(value));
});
test('registered fixed route preserves queueing/read permissions and excludes native/catalog publication paths',async()=>{
 const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8'),route=workflow.slice(workflow.indexOf('  moss-upstream-reference:'),workflow.indexOf('  moss-dispatch:'));
 assert.match(route,/runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);assert.match(route,/group: memory-catalog-campaign-mlx-nax/);assert.match(route,/cancel-in-progress: false/);assert.match(route,/contents: read/);assert.match(route,/artifact-ids: \$\{\{ steps.reference.outputs.artifact_id \}\}/);
 assert.doesNotMatch(route,/nvidia-smi|pgrep|git push|gh pr|snapshot-view\/|uses:.*moss-edit-terminal/);
 const code=await readFile(new URL('./moss-upstream-reference.mjs',import.meta.url),'utf8');assert.match(code,/GITHUB_ACTIONS==='true'/);assert.match(code,/'--device','mps'/);assert.match(code,/PYTORCH_ENABLE_MPS_FALLBACK:'0'/);assert.doesNotMatch(code,/shell:true|execSync|COPYFILE_FICLONE|nvidia-smi/);
});

test('reference-only Windows recovery uses no Actions dependencies/native build and always selects HF-only mode',async()=>{
 const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8'),job=workflow.slice(workflow.indexOf('  moss-reference-maintenance:'),workflow.indexOf('  moss-upstream-reference:'));
 assert.match(job,/runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);assert.match(job,/--moss-disk-hf-only/);assert.match(job,/MOSS_SOURCE_SHA: \$\{\{ inputs.ref \}\}/);assert.doesNotMatch(job,/\buses:|\bneeds:|cargo|nvidia-smi/);
});
