import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdtemp,rm,symlink,realpath,mkdir,stat,link,cp,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {envelope,PAYLOAD_SHA} from './moss-upstream-reference.mjs';
import {verifyProbePayload,verifyPriorBinding,probeBinary,PROBE_SHA,ENCODING_STAGED,ENCODING_SHA,verifyEncodingPayload,encodingEnvelope,encodingPrecisionProfile,cloneEncodingTarget,encodingMetrics} from './moss-first-step-probe.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex'),ref='a'.repeat(40),pin='857f7a1996e3678a0773c3509ac0087522c19ce6';
test('probe dispatch is closed, immutable-artifact-only, and cannot select backend/source/commands',()=>{
 for(const phase of ['upload-probe','execute-probe']){const value={phase,artifact_id:phase==='upload-probe'?null:'123456',artifact_run_id:phase==='upload-probe'?null:'1234567'};assert.deepEqual(envelope(JSON.stringify(value),ref,ref,'mlx'),value);for(const bad of [{...value,command:'cargo test'},{...value,payload:'/tmp/x'},{...value,device:'cpu'}])assert.throws(()=>envelope(JSON.stringify(bad),ref,ref,'mlx'));assert.throws(()=>envelope(JSON.stringify(value),ref,ref,'candle'));assert.throws(()=>envelope(JSON.stringify(value),ref,'b'.repeat(40),'mlx'));}
 assert.throws(()=>envelope(JSON.stringify({phase:'execute-probe',artifact_id:null,artifact_run_id:null}),ref,ref,'mlx'));
});
test('probe payload verification uses real files and rejects altered closure, links and substituted manifest',async t=>{
 const root=await mkdtemp(resolve(await realpath(tmpdir()),'moss-probe-'));t.after(()=>rm(root,{force:true,recursive:true}));const files=[];for(let i=0;i<4;i++){const file='f'+i,data=Buffer.from('fixed '+i);await writeFile(resolve(root,file),data);files.push({file,byteLength:data.length,sha256:sha(data)});}const bytes=JSON.stringify({kind:'sc-26752-first-step-probe/v1',nativePin:pin,originalReferencePayloadSha256:PAYLOAD_SHA,files});await writeFile(resolve(root,'payload-manifest.json'),bytes);await verifyProbePayload(root,sha(bytes));await assert.rejects(()=>verifyProbePayload(root,'0'.repeat(64)),/manifest SHA/);
 await writeFile(resolve(root,'f0'),'changed');await assert.rejects(()=>verifyProbePayload(root,sha(bytes)),/file mismatch/);await writeFile(resolve(root,'f0'),'fixed 0');await writeFile(resolve(root,'extra.py'),'unreviewed');await assert.rejects(()=>verifyProbePayload(root,sha(bytes)),/unlisted/);await rm(resolve(root,'extra.py'));await rm(resolve(root,'f0'));await symlink(resolve(root,'f1'),resolve(root,'f0'));await assert.rejects(()=>verifyProbePayload(root,sha(bytes)),/linked/);
});
function priorFixture(){const root='/Volumes/Models/sceneworks-terminal-sc-26752/moss-upstream-reference-40191507328-1',sources=Array.from({length:23},(_,i)=>({file:`hub/f${i}`,source:`/Volumes/Models/huggingface/hub/f${i}`,sha256:'a'.repeat(64),byteLength:12})),auth=Buffer.from(JSON.stringify({payloadManifestSha256:PAYLOAD_SHA,nativePin:pin,models:sources}));const binding={kind:'sc-26752-prior-reference-stat-binding/v1',priorRunId:'40191507328',priorAttempt:1,root,authenticationSha256:sha(auth),authenticationBirthtimeNs:'100',files:sources.map(x=>({...x,realPath:x.source,view:null,stat:{dev:'1',ino:'2',nlink:'1',size:'12',mtimeNs:'50',ctimeNs:'50'}}))};const io={readFile:async()=>auth,realpath:async path=>path,stat:async path=>path.endsWith('authentication.json')?{birthtimeNs:100n}:{dev:1n,ino:2n,nlink:1n,size:12n,mtimeNs:50n,ctimeNs:50n},readlink:async()=>{throw Error('unexpected link');}};return {binding,io};}
test('prior authentication continuity refuses replaced receipt, inode/time/length/target changes and wrong source',async()=>{
 const {binding,io}=priorFixture();await verifyPriorBinding(binding,io);
 for(const key of ['dev','ino','nlink','size','mtimeNs','ctimeNs'])await assert.rejects(()=>verifyPriorBinding(binding,{...io,stat:async path=>({...await io.stat(path),...(!path.endsWith('authentication.json')?{[key]:999n}:{})})}),/stat changed/);
 await assert.rejects(()=>verifyPriorBinding(binding,{...io,stat:async path=>({...await io.stat(path),...(path.endsWith('authentication.json')?{birthtimeNs:101n}:{})})}),/authentication file replaced/);
 await assert.rejects(()=>verifyPriorBinding(binding,{...io,readFile:async()=>Buffer.from('{}')}),/authentication changed/);await assert.rejects(()=>verifyPriorBinding(binding,{...io,realpath:async()=>'/elsewhere'}),/retargeted/);
 const escaped=structuredClone(binding);escaped.files[0].source='/etc/passwd';await assert.rejects(()=>verifyPriorBinding(escaped,io),/identity mismatch/);
 const future=structuredClone(binding);future.files.forEach(x=>x.stat.ctimeNs='101');await assert.rejects(()=>verifyPriorBinding(future,{...io,stat:async path=>({...await io.stat(path),...(!path.endsWith('authentication.json')?{ctimeNs:101n}:{})})}),/after prior authentication/);
});
test('native executable selection rejects old/external/multiple/non-test executables',()=>{
 const entry={reason:'compiler-artifact',target:{name:'mlx_gen_moss_image'},profile:{test:true},executable:'/owned/target/release/deps/probe'};assert.equal(probeBinary(JSON.stringify(entry),'/owned/target'),entry.executable);for(const output of ['',JSON.stringify({...entry,executable:'/old/target/release/deps/probe'}),JSON.stringify({...entry,profile:{test:false}}),JSON.stringify(entry)+'\n'+JSON.stringify(entry)])assert.throws(()=>probeBinary(output,'/owned/target'));
});
test('sealed identity pins exact two-file cfg(test) overlay and original payload, with no numerical gate',async()=>{
 const bytes=await readFile(new URL('../fixtures/moss-first-step-probe/payload-manifest.json',import.meta.url)),manifest=JSON.parse(bytes);assert.equal(sha(bytes),PROBE_SHA);assert.equal(manifest.originalReferencePayloadSha256,PAYLOAD_SHA);assert.equal(manifest.nativePin,pin);assert.equal(manifest.files.length,10);for(const file of ['capture_first_step.py','compare_first_step.py','prior-authentication-binding.json','overlay/lib.rs.patch','overlay/first_step_probe.rs','overlay/source-identity.json'])assert.ok(manifest.files.find(x=>x.file===file));
 const code=await readFile(new URL('./moss-first-step-probe.mjs',import.meta.url),'utf8');assert.match(code,/'rustup',\['run','1.98.0','cargo','test','--locked','--release'/);assert.match(code,/originalTargetRetained:true/);assert.match(code,/--ignored/);assert.match(code,/COPYFILE_EXCL/);assert.match(code,/copyCacheExclusive\(WARM,common.CARGO_TARGET_DIR\)/);assert.doesNotMatch(code,/\/bin\/cp/);assert.match(code,/unexpected native source delta/);assert.doesNotMatch(code.slice(0,code.indexOf('// Separate fixed diagnostic:')),/git push|nvidia-smi|FICLONE_FORCE|shell:true/);
 const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8'),job=workflow.slice(workflow.indexOf('  moss-upstream-reference:'),workflow.indexOf('  moss-dispatch:')),maintenance=workflow.slice(workflow.indexOf('  moss-reference-maintenance:'),workflow.indexOf('  moss-upstream-reference:'));assert.match(job,/runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);assert.match(job,/memory-catalog-campaign-mlx-nax/);assert.match(job,/phase == 'execute-probe'/);assert.match(job,/path: moss-first-step-native-\$\{\{ github.run_id \}\}-\$\{\{ github.run_attempt \}\}/);assert.match(job,/native\/\*\.f32le/);assert.doesNotMatch(job,/path:.*snapshot-view|path:.*target/);assert.doesNotMatch(maintenance,/upload-probe|execute-probe/);
});
test('encoding envelope is fixed, source-bound and artifact-only',()=>{
 const value={phase:'execute-encoding',artifact_id:'123456',artifact_run_id:'123457'};
 assert.deepEqual(encodingEnvelope(JSON.stringify(value),ref,ref,'mlx'),value);
 assert.deepEqual(encodingEnvelope(JSON.stringify({...value,phase:'upload-encoding',artifact_id:null,artifact_run_id:null}),ref,ref,'mlx').phase,'upload-encoding');
 for(const bad of [{...value,phase:'execute-probe'},{...value,command:'anything'},{...value,artifact_id:null},{...value,payload:'/tmp/elsewhere'}])assert.throws(()=>encodingEnvelope(JSON.stringify(bad),ref,ref,'mlx'));
 assert.throws(()=>encodingEnvelope(JSON.stringify(value),ref,ref,'candle'));
 assert.throws(()=>encodingEnvelope(JSON.stringify(value),ref,'b'.repeat(40),'mlx'));
});
test('precision encoding stays pending until an exact reviewed native and payload pair is frozen',()=>{
 const upload={phase:'upload-encoding-precision',artifact_id:null,artifact_run_id:null},execute={phase:'execute-encoding-precision',artifact_id:'123456',artifact_run_id:'123457'};
 assert.throws(()=>encodingPrecisionProfile(),/pending/);
 assert.throws(()=>encodingEnvelope(JSON.stringify(upload),ref,ref,'mlx'),/pending/);
 assert.throws(()=>encodingEnvelope(JSON.stringify(execute),ref,ref,'mlx'),/pending/);
 const profile=encodingPrecisionProfile('b'.repeat(40),'c'.repeat(64));
 assert.deepEqual(encodingEnvelope(JSON.stringify(upload),ref,ref,'mlx',profile),upload);
 assert.deepEqual(encodingEnvelope(JSON.stringify(execute),ref,ref,'mlx',profile),execute);
 for(const bad of [{...execute,native_pin:'b'.repeat(40)},{...execute,payload:'/tmp/elsewhere'},{...execute,artifact_id:null}])assert.throws(()=>encodingEnvelope(JSON.stringify(bad),ref,ref,'mlx',profile));
 assert.throws(()=>encodingEnvelope(JSON.stringify(execute),ref,'d'.repeat(40),'mlx',profile));
 assert.throws(()=>encodingEnvelope(JSON.stringify(execute),ref,ref,'candle',profile));
 assert.throws(()=>encodingPrecisionProfile(pin,'c'.repeat(64)),/pending/);
 assert.throws(()=>encodingPrecisionProfile('b'.repeat(40),'not-a-sha'),/pending/);
});
test('precision payload binds the exact historical tensors, request and B image independently of new overlay',async t=>{
 try{await access(resolve(ENCODING_STAGED,'payload-manifest.json'));}catch{t.skip('retained sealed encoding payload unavailable on this host');return;}
 const root=await mkdtemp(resolve(await realpath(tmpdir()),'moss-precision-payload-'));t.after(()=>rm(root,{recursive:true,force:true}));
 await cp(ENCODING_STAGED,resolve(root,'payload'),{recursive:true});const payload=resolve(root,'payload');
 const original=JSON.parse(await readFile(resolve(payload,'payload-manifest.json')));
 const manifest={...original,kind:'sc-26752-normal-encoding-precision/v1',nativePin:'b'.repeat(40),historicalNativePin:pin,historicalManifestSha256:ENCODING_SHA};
 const bytes=Buffer.from(JSON.stringify(manifest));await writeFile(resolve(payload,'payload-manifest.json'),bytes);
 const profile=encodingPrecisionProfile('b'.repeat(40),sha(bytes));await verifyEncodingPayload(payload,profile);
 const changed=Buffer.from(await readFile(resolve(payload,'request.json')));changed[0]^=1;await writeFile(resolve(payload,'request.json'),changed);
 manifest.files.find(x=>x.file==='request.json').sha256=sha(changed);const replaced=Buffer.from(JSON.stringify(manifest));await writeFile(resolve(payload,'payload-manifest.json'),replaced);
 await assert.rejects(()=>verifyEncodingPayload(payload,encodingPrecisionProfile(profile.pin,sha(replaced))),/frozen request or B image changed/);
});
test('encoding metrics separate observations from acceptance and reject invalid inputs',()=>{
 const same=encodingMetrics([1,2,3],[1,2,3]);assert.equal(same.relativeL2,0);assert.equal(same.equalValues,3);assert.equal(same.cosineSimilarity,1);
 const changed=encodingMetrics([1,2],[2,4]);assert.equal(changed.relativeL2,1);assert.equal(changed.maxAbsError,2);assert.equal(changed.cosineSimilarity,1);assert.ok(!('pass' in changed));
 assert.throws(()=>encodingMetrics([1],[1,2]));assert.throws(()=>encodingMetrics([Infinity],[1]));
});
test('encoding COW clone preserves original after destination writes and rejects link/overwrite/fallback',async t=>{
 const root=await mkdtemp(resolve(await realpath(tmpdir()),'moss-encoding-clone-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const source=resolve(root,'source');await mkdir(source);await mkdir(resolve(source,'nested'));await writeFile(resolve(source,'nested/proof'),'original');
 const before=await stat(resolve(source,'nested/proof'));
 if(process.platform==='darwin'){
  const result=await cloneEncodingTarget(source,resolve(root,'cloned'));assert.equal(result.files,1);assert.match(result.method,/clonefile/);
  await writeFile(resolve(root,'cloned/nested/proof'),'new');assert.equal(await readFile(resolve(source,'nested/proof'),'utf8'),'original');assert.equal((await stat(resolve(source,'nested/proof'))).ino,before.ino);
  await assert.rejects(()=>cloneEncodingTarget(source,resolve(root,'cloned')),/EEXIST/);
 }
 await assert.rejects(()=>cloneEncodingTarget(source,resolve(source,'inside')),/overlap/);
 let called=0;await assert.rejects(()=>cloneEncodingTarget(source,resolve(root,'unsupported'),async()=>{called++;throw Object.assign(Error('clone unsupported'),{code:'ENOTSUP'});}),/unsupported/);assert.equal(called,1);assert.equal(await readFile(resolve(source,'nested/proof'),'utf8'),'original');
 await assert.rejects(()=>cloneEncodingTarget(source,resolve(root,'hardlinked'),async(a,b)=>{await mkdir(resolve(b,'nested'));await link(resolve(a,'nested/proof'),resolve(b,'nested/proof'));}),/independent|changed/);
 await symlink(resolve(source,'nested/proof'),resolve(source,'alias'));
 await assert.rejects(()=>cloneEncodingTarget(source,resolve(root,'linked')),/link\/type/);
});
test('encoding route has no upstream execution and retains ordinary route behavior',async()=>{
 const code=await readFile(new URL('./moss-first-step-probe.mjs',import.meta.url),'utf8'),encoding=code.slice(code.indexOf('export async function executeEncoding('),code.indexOf('async function encodingMain('));
 assert.doesNotMatch(encoding,/capture_first_step|runtime.launcher|executeProbe|sample_latents/);
 assert.match(encoding,/cloneEncodingTarget\(ENCODING_WARM/);assert.ok(encoding.indexOf('cloneEncodingTarget(')<encoding.indexOf("run('rustup'"));assert.match(encoding,/encoding_probe::fixed_normal_encoding_only/);
 const workflow=await readFile(new URL('../../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8');assert.match(workflow,/moss-first-step-probe.mjs route validate/);assert.match(workflow,/phase == 'execute-encoding'/);
 assert.match(workflow,/phase == 'execute-encoding-precision'/);assert.match(workflow,/ref: \$\{\{ steps\.reference\.outputs\.native_pin \}\}/);
 assert.match(workflow,/ref: '857f7a1996e3678a0773c3509ac0087522c19ce6'/);
 assert.match(code,/moss-upstream-reference-common.mjs/);assert.doesNotMatch(code,/import .*from '.\/moss-upstream-reference.mjs'/);
});
