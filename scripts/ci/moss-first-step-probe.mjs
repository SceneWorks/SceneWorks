// Fixed, authenticated first-step comparison only. Actual compilation/model work is Actions-only.
import {readFile,writeFile,mkdir,lstat,stat,realpath,readlink,readdir,copyFile,statfs,appendFile,cp} from 'node:fs/promises';
import {createReadStream,createWriteStream,constants} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {safe,verifyPayload,validateRuntime,PAYLOAD_SHA} from './moss-upstream-reference-common.mjs';
import {copyCacheExclusive} from './moss-owned-cache-maintenance.mjs';
export const PROBE_STAGED='/Users/michael/.codex/epic-runs/sc-26752/first-step-probe-r15/payload';
export const PROBE_SHA='440a33b4ddb9cadaecb3b73291f648b1c03fe0239a7ea682cf29d56b7f4c8796';
const BASE='/Volumes/Models/sceneworks-terminal-sc-26752',PRIOR=BASE+'/moss-upstream-reference-40191507328-1';
const PIN='857f7a1996e3678a0773c3509ac0087522c19ce6';
const ADAPTER=BASE+'/moss-upstream-reference-r15-retained-adapter';
const HUB='/Volumes/Models/huggingface/hub/';
const MODEL_BASE=HUB+'models--OpenMOSS-Team--MOSS-Image-Base/snapshots/1ceede80badddd6a7812069619edaa5633587192';
const SHARED=HUB+'models--OpenMOSS-Team--MOSS-Image-SFT/snapshots/fc5de0d56fc910cbca6f6625a70f5f582b26deb1';
const WARM='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/moss-sc-26752/moss-build-40181253824-1-mlx/target';
const CARGO_SEED='/Users/michael/.codex/epic-runs/sc-26752/studio56-integration-cargo-home-v1';
const check=(ok,message)=>{if(!ok)throw Error(message);};
const digest=data=>createHash('sha256').update(data).digest('hex');
async function fileHash(path){const hash=createHash('sha256');for await(const bytes of createReadStream(path))hash.update(bytes);return hash.digest('hex');}
async function noLinks(path){for(path=resolve(path);;path=dirname(path)){check(!(await lstat(path)).isSymbolicLink(),'probe linked ancestor refused');if(dirname(path)===path)return;}}
export async function verifyProbePayload(root,expected=PROBE_SHA){
 await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(digest(bytes)===expected,'probe manifest SHA mismatch');const manifest=JSON.parse(bytes);
 check(manifest.kind==='sc-26752-first-step-probe/v1'&&manifest.nativePin===PIN&&manifest.originalReferencePayloadSha256===PAYLOAD_SHA,'probe provenance mismatch');
 check(Array.isArray(manifest.files)&&manifest.files.length>3&&manifest.files.length<20,'bounded probe closure required');const wanted=new Set(['payload-manifest.json']);
 for(const file of manifest.files){check(!wanted.has(file.file),'duplicate probe file');wanted.add(file.file);const path=safe(root,file.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.nlink===1&&info.size===file.byteLength&&await fileHash(path)===file.sha256,'probe file mismatch');}
 async function walk(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){check(!entry.isSymbolicLink(),'probe payload link refused');const file=prefix+entry.name;if(entry.isDirectory())await walk(resolve(path,entry.name),file+'/');else check(wanted.has(file),'unlisted probe file');}}await walk(root);return manifest;
}
export async function verifyPriorBinding(binding,io={readFile,realpath,stat,readlink}){
 check(binding.kind==='sc-26752-prior-reference-stat-binding/v1'&&binding.root===PRIOR&&binding.priorRunId==='40191507328'&&binding.priorAttempt===1&&binding.files.length===23,'wrong prior reference identity');
 const authPath=resolve(PRIOR,'authentication.json'),authStat=await io.stat(authPath,{bigint:true});check(String(authStat.birthtimeNs)===binding.authenticationBirthtimeNs,'original prior authentication file replaced');const bytes=await io.readFile(authPath);check(digest(bytes)===binding.authenticationSha256,'prior authentication changed');const auth=JSON.parse(bytes);check(auth.payloadManifestSha256===PAYLOAD_SHA&&auth.nativePin===PIN&&auth.models.length===23,'prior provenance mismatch');
 const seen=new Set();for(const item of binding.files){check(!seen.has(item.source),'duplicate prior model');seen.add(item.source);const prior=auth.models.find(x=>x.source===item.source);check(prior&&prior.sha256===item.sha256&&prior.byteLength===item.byteLength&&prior.file===item.file,'prior model identity mismatch');
  check(item.source===safe(item.source.startsWith(ADAPTER+'/')?ADAPTER:'/Volumes/Models/huggingface',item.source.startsWith(ADAPTER+'/')?item.file.split('/').at(-1):item.file),'prior model escaped fixed roots');
  check(await io.realpath(item.source)===item.realPath,'prior source retargeted');const info=await io.stat(item.source,{bigint:true});for(const key of ['dev','ino','nlink','size','mtimeNs','ctimeNs'])check(String(info[key])===item.stat[key],`prior stat changed: ${item.file} ${key}`);check(info.ctimeNs<=BigInt(binding.authenticationBirthtimeNs),'model changed after prior authentication');
  if(item.view){check(item.view.startsWith(PRIOR+'/snapshot-view/')&&await io.readlink(item.view)===item.viewTarget&&item.viewTarget===item.source,'prior snapshot view changed');}
 }return auth;
}
async function run(command,args,{cwd,env,log}){const stream=createWriteStream(log,{flags:'wx'});let output='',oversize=false;await new Promise((done,fail)=>{const child=spawn(command,args,{cwd,env,stdio:['ignore','pipe','pipe']});child.stdout.on('data',bytes=>{if(output.length+bytes.length<32*1024*1024)output+=bytes;else oversize=true;});child.stdout.pipe(stream,{end:false});child.stderr.pipe(stream,{end:false});child.on('error',error=>{stream.end();fail(error);});child.on('close',code=>{stream.end();code===0&&!oversize?done():fail(Error(`fixed probe subprocess failed (${code}, bounded output exceeded=${oversize}): ${command}`));});});return output;}

async function cacheInventory(root){let files=0,bytes=0,links=0;async function walk(path){for(const entry of await readdir(path,{withFileTypes:true})){const file=resolve(path,entry.name);if(entry.isDirectory())await walk(file);else if(entry.isSymbolicLink())links++;else{const info=await lstat(file);check(info.isFile(),'unsupported cache entry');files++;bytes+=info.size;}}}await walk(root);return {files,bytes,links};}
export function probeBinary(output,target){const matches=output.split('\n').filter(Boolean).flatMap(line=>{try{return[JSON.parse(line)];}catch{return[];}}).filter(x=>x.reason==='compiler-artifact'&&x.target?.name==='mlx_gen_moss_image'&&x.profile?.test&&typeof x.executable==='string');check(matches.length===1,'one exact native libtest executable required');const path=resolve(matches[0].executable);check(path.startsWith(resolve(target)+'/release/deps/'),'native executable escaped owned target');return path;}
export async function executeProbe(root,payload,env=process.env){
 check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&process.platform==='darwin','probe execution requires registered Mac Actions');
 await verifyProbePayload(payload);await verifyPayload(resolve(PRIOR,'payload'));const binding=JSON.parse(await readFile(resolve(payload,'prior-authentication-binding.json')));await verifyPriorBinding(binding);
 const runtime=JSON.parse(await readFile(resolve(PRIOR,'payload/runtime-files.json')));validateRuntime(runtime);check(await realpath(runtime.launcher)===runtime.executable,'prepared runtime changed');for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'prepared runtime changed');}
 const source=resolve(env.GITHUB_WORKSPACE,`moss-first-step-native-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`),nativeRoot=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752',`moss-first-step-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);await noLinks(source);await noLinks(dirname(nativeRoot));await mkdir(nativeRoot,{mode:0o700});
 const common={...env,CARGO_HOME:resolve(nativeRoot,'cargo-home'),CARGO_TARGET_DIR:resolve(nativeRoot,'target'),CARGO_BUILD_JOBS:'2',CMAKE_BUILD_PARALLEL_LEVEL:'2',CARGO_PROFILE_RELEASE_STRIP:'none',CARGO_INCREMENTAL:'0',PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'};
 const git=async args=>(await run('git',args,{cwd:source,env:common,log:resolve(root,`git-${args[0]}-${args.includes('--check')?'check':args.length}.log`)})).trimEnd();check(await git(['rev-parse','HEAD'])===PIN,'native checkout pin mismatch');check(await git(['status','--porcelain'])==='','native checkout must be clean');
 const identity=JSON.parse(await readFile(resolve(payload,'overlay/source-identity.json')));check(identity.baseHead===PIN&&identity.libPath==='crates/media/mlx-gen/mlx-gen-moss-image/src/lib.rs'&&identity.newModulePath==='crates/media/mlx-gen/mlx-gen-moss-image/src/first_step_probe.rs'&&identity.test==='first_step_probe::fixed_matched_conditioning_first_step','fixed cfg(test) overlay identity required');
 const lib=safe(source,identity.libPath);check(await fileHash(lib)===identity.originalLibSha256,'native original lib mismatch');await git(['apply','--check',resolve(payload,'overlay/lib.rs.patch')]);await git(['apply',resolve(payload,'overlay/lib.rs.patch')]);await copyFile(resolve(payload,'overlay/first_step_probe.rs'),safe(source,identity.newModulePath),constants.COPYFILE_EXCL);check(await fileHash(lib)===identity.overlayLibSha256,'native patched lib mismatch');
 const delta=await git(['status','--porcelain','--untracked-files=all']);check(delta.split('\n').sort().join('\n')===[' M '+identity.libPath,'?? '+identity.newModulePath].sort().join('\n'),'unexpected native source delta');
 await noLinks(WARM);await noLinks(CARGO_SEED);const warmInventory=await cacheInventory(WARM),cargoInventory=await cacheInventory(CARGO_SEED),volume=await statfs(nativeRoot,{bigint:true});check(volume.bavail*volume.bsize>BigInt(warmInventory.bytes+cargoInventory.bytes)+4n*1024n**3n,'insufficient space for private dependency copies and reserve');await copyCacheExclusive(WARM,common.CARGO_TARGET_DIR);await copyCacheExclusive(CARGO_SEED,common.CARGO_HOME);
 await writeFile(resolve(root,'probe-authentication.json'),JSON.stringify({probeManifestSha256:PROBE_SHA,priorReferenceRun:'40191507328',priorAuthenticationSha256:binding.authenticationSha256,priorStatBinding:'exact current identities match immutable capture; no new blanket model hash scan',nativePin:PIN,overlay:identity,warmTarget:WARM,originalTargetRetained:true,cacheCopy:'existing exclusive ordinary Node copy; fresh capacity checked, no clone-only requirement',warmInventory,cargoInventory,freeBytesBeforeCopy:String(volume.bavail*volume.bsize),target:common.CARGO_TARGET_DIR,nativeRoot,source,sourceDelta:delta,routeSource:env.GITHUB_SHA,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT},null,2)+'\n',{flag:'wx'});
 const build=await run('rustup',['run','1.98.0','cargo','test','--locked','--release','-p','mlx-gen-moss-image','--lib','--no-run','--message-format=json'],{cwd:source,env:common,log:resolve(root,'native-build.log')});const binary=probeBinary(build,common.CARGO_TARGET_DIR);const binarySha256=await fileHash(binary);
 await verifyPriorBinding(binding);await run(runtime.launcher,['-B',resolve(payload,'capture_first_step.py'),'--reference-payload',resolve(PRIOR,'payload'),'--snapshot-view',resolve(PRIOR,'snapshot-view'),'--adapter',ADAPTER,'--output-dir',resolve(root,'upstream')],{cwd:payload,env:common,log:resolve(root,'upstream.log')});
 const inputs=resolve(root,'upstream/inputs.json'),upstream=JSON.parse(await readFile(resolve(root,'upstream/upstream.json')));check(upstream.kind==='sc-26752-upstream-first-step/v1'&&upstream.status==='CAPTURED_FIRST_STEP_ONLY'&&upstream.predictions===2&&upstream.updates===1&&upstream.inputsSha256===await fileHash(inputs),'successful upstream first-step receipt required');await verifyPriorBinding(binding);await run(binary,[identity.test,'--exact','--ignored','--nocapture','--test-threads=1'],{cwd:source,env:{...common,MOSS_FIRST_STEP_ADMITTED:'1',MOSS_FIRST_STEP_INPUTS:inputs,MOSS_FIRST_STEP_INPUTS_SHA256:await fileHash(inputs),MOSS_FIRST_STEP_OUTPUT:resolve(root,'native'),MOSS_FIRST_STEP_BASE:MODEL_BASE,MOSS_FIRST_STEP_SHARED:SHARED,MOSS_FIRST_STEP_ADAPTER:ADAPTER},log:resolve(root,'native.log')});
 await run(runtime.launcher,['-B',resolve(payload,'compare_first_step.py'),'--upstream',resolve(root,'upstream'),'--native',resolve(root,'native'),'--output',resolve(root,'comparison.json')],{cwd:payload,env:common,log:resolve(root,'comparison.log')});
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({nativePin:PIN,binary,binarySha256,inputsSha256:await fileHash(inputs),upstreamSha256:await fileHash(resolve(root,'upstream/upstream.json')),nativeSha256:await fileHash(resolve(root,'native/native.json')),scope:'one original30-step schedule first update; metrics only, no quality tolerance gate'},null,2)+'\n',{flag:'wx'});
}

// Separate fixed diagnostic: reuse already encoded upstream inputs, never execute Python/models upstream.
export const ENCODING_STAGED='/Users/michael/.codex/epic-runs/sc-26752/encoding-only-prep/payload';
export const ENCODING_SHA='3004cb310e3e3e98de8ea0901ca848f2b67c81f9aebb7aff6ca974e19230ffd2';
export const ENCODING_PRECISION_STAGED='/Users/michael/.codex/epic-runs/sc-26752/encoding-precision-prep/payload';
// Reviewed protected native merge and its separately sealed cfg(test) payload.
export const ENCODING_PRECISION_PIN='7275f7c672eecabeba9342e99897ef484e992e14';
export const ENCODING_PRECISION_SHA='f819d901ce8caa815dc9bdd06cc781dee881f30ba1672a8f91cd4f3ab910cd5f';
const ENCODING_WARM='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/moss-sc-26752/moss-first-step-40195966134-1/target';
const ENCODING_HUB=BASE+'/moss-execute-40198829729-1-mlx/hf/hub';
const INPUT_SHA='47b07acccc6491dbc1f4c5c3135fe37d4d65f4afb496d28dafd8ca63a08cd47d';
const UPSTREAM_SHA='ade384909936c7b55326ce3bf7a804e827c9b6b1ffbdc9232e7e36b5016b9b57';
const REQUEST_SHA='f9337c3c4ee20d8fddc00451df4ef53f0bbe68e8c5bed7217b835af53f2781f3';
const B_SHA='c6ec194e6d7e04a2893449e530cf51d064c6adfd9847f0b028921b21e1c1bcb5';
const ENCODING_SHAPES={positive:[1,71,4096],negative:[1,10,4096],positive_mask:[1,71],negative_mask:[1,10],reference_0:[1,128,21,32]};
const OLD_ENCODING={pin:PIN,sha:ENCODING_SHA,kind:'sc-26752-normal-encoding/v1',staged:ENCODING_STAGED,precision:false};
export function encodingPrecisionProfile(pin=ENCODING_PRECISION_PIN,sha=ENCODING_PRECISION_SHA){
 check(typeof pin==='string'&&/^[a-f0-9]{40}$/.test(pin)&&pin!==PIN&&typeof sha==='string'&&/^[a-f0-9]{64}$/.test(sha),'reviewed encoding precision pin and payload are pending');
 return {pin,sha,kind:'sc-26752-normal-encoding-precision/v1',staged:ENCODING_PRECISION_STAGED,precision:true};
}
export function encodingEnvelope(raw,ref,sha,backend,precision){
 const value=JSON.parse(raw);
 check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase','closed encoding envelope required');
 check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact encoding source and mlx required');
 check(['upload-encoding','execute-encoding','upload-encoding-precision','execute-encoding-precision'].includes(value.phase),'unknown encoding phase');
 if(value.phase.endsWith('-precision'))check((precision??encodingPrecisionProfile()).precision===true,'precision profile required');
 for(const key of ['artifact_id','artifact_run_id'])check(value.phase.startsWith('upload-')?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact encoding artifact identity required');
 return value;
}
async function tensorValues(root,name,item,shape){
 check(item&&item.file===name+'.f32le'&&item.dtype==='f32le'&&JSON.stringify(item.shape)===JSON.stringify(shape),'encoding tensor contract mismatch');
 const length=shape.reduce((a,b)=>a*b,4),path=safe(root,item.file);await noLinks(path);
 const info=await lstat(path);check(length<=4*1024*1024&&item.byteLength===length&&info.isFile()&&info.size===length,'encoding tensor length mismatch');
 const bytes=await readFile(path);check(digest(bytes)===item.sha256,'encoding tensor SHA mismatch');
 const values=[];for(let i=0;i<bytes.length;i+=4){const value=bytes.readFloatLE(i);check(Number.isFinite(value),'nonfinite encoding tensor');if(name.endsWith('_mask'))check(value===1,'fixed unpadded mask must be all ones');values.push(value);}return values;
}
export async function verifyEncodingPayload(root,profile=OLD_ENCODING){
 await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(bytes.length<65536&&digest(bytes)===profile.sha,'encoding manifest SHA mismatch');
 const manifest=JSON.parse(bytes);check(manifest.kind===profile.kind&&manifest.nativePin===profile.pin&&manifest.upstreamRun==='40195966134'&&manifest.inputsSha256===INPUT_SHA&&manifest.upstreamSha256===UPSTREAM_SHA,'encoding provenance mismatch');
 if(profile.precision)check(manifest.historicalManifestSha256===ENCODING_SHA&&manifest.historicalNativePin===PIN,'historical encoding producer mismatch');
 check(Array.isArray(manifest.files)&&manifest.files.length===13,'fixed encoding payload closure required');const wanted=new Set(['payload-manifest.json']);
 for(const item of manifest.files){check(!wanted.has(item.file),'duplicate encoding file');wanted.add(item.file);const path=safe(root,item.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.nlink===1&&info.size===item.byteLength&&info.size<=8*1024*1024&&await fileHash(path)===item.sha256,'encoding payload file mismatch');}
 async function walk(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){check(!entry.isSymbolicLink(),'encoding payload link');const file=prefix+entry.name;if(entry.isDirectory())await walk(resolve(path,entry.name),file+'/');else check(wanted.has(file),'unlisted encoding file');}}await walk(root);
 const inputBytes=await readFile(resolve(root,'upstream/inputs.json')),upstreamBytes=await readFile(resolve(root,'upstream/upstream.json'));
 check(digest(inputBytes)===INPUT_SHA&&digest(upstreamBytes)===UPSTREAM_SHA,'retained upstream identity mismatch');
 if(profile.precision)check(await fileHash(resolve(root,'request.json'))===REQUEST_SHA&&await fileHash(resolve(root,'B.png'))===B_SHA,'frozen request or B image changed');
 const inputs=JSON.parse(inputBytes),upstream=JSON.parse(upstreamBytes);
 check(inputs.kind==='sc-26752-first-step-inputs/v1'&&inputs.nativeHead===PIN&&upstream.status==='CAPTURED_FIRST_STEP_ONLY'&&upstream.inputsSha256===INPUT_SHA&&upstream.predictions===2&&upstream.updates===1,'retained capture incomplete');
 check(Object.keys(inputs.tensors).length===6,'exact six retained inputs required');
 for(const [name,shape] of Object.entries({...ENCODING_SHAPES,noise:[1,128,32,48]}))await tensorValues(resolve(root,'upstream'),name,inputs.tensors[name],shape);
 check(await fileHash(resolve(root,'B.png'))===inputs.provenance.referenceSha256,'reference PNG identity mismatch');
 return manifest;
}
// macOS clonefile(2) has no ordinary-copy fallback. Node's FORCE flag is unsupported
// by this host's libuv; use the system interpreter only to invoke the native syscall.
const CLONE_TREE_PYTHON = String.raw`
import ctypes, os, pathlib, stat, sys
source, destination = map(pathlib.Path, sys.argv[1:])
libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
clone = libc.clonefile
clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
clone.restype = ctypes.c_int
for directory, dirs, files in os.walk(source, followlinks=False):
    relative = pathlib.Path(directory).relative_to(source)
    for name in dirs:
        child = pathlib.Path(directory) / name
        if not stat.S_ISDIR(child.lstat().st_mode): raise RuntimeError('linked clone directory')
        (destination / relative / name).mkdir(mode=0o700)
    for name in files:
        child = pathlib.Path(directory) / name
        if not stat.S_ISREG(child.lstat().st_mode): raise RuntimeError('nonregular clone file')
        target = destination / relative / name
        if clone(os.fsencode(child), os.fsencode(target), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(child))
`;
async function nativeCloneTree(source,destination){
 check(process.platform==='darwin','encoding clone requires macOS clonefile');
 await new Promise((done,fail)=>{let error='';const child=spawn('/usr/bin/python3',['-I','-B','-c',CLONE_TREE_PYTHON,source,destination],{stdio:['ignore','ignore','pipe'],env:{PATH:'/usr/bin:/bin'}});child.stderr.on('data',b=>{if(error.length<65536)error+=b.toString();});child.on('error',fail);child.on('close',code=>code===0?done():fail(Error(`clonefile failed without fallback (${code}): ${error}`)));});
}
// No hardlinks, symlinks, existing destination, fallback, or writes into retained target.
export async function cloneEncodingTarget(source,destination,cloneTree=nativeCloneTree){
 await noLinks(source);await noLinks(dirname(destination));
 const src=await lstat(source,{bigint:true}),parent=await lstat(dirname(destination),{bigint:true});
 check(src.isDirectory()&&src.dev===parent.dev,'encoding clone requires same-volume directories');
 check(!resolve(destination).startsWith(resolve(source)+'/')&&!resolve(source).startsWith(resolve(destination)+'/')&&resolve(source)!==resolve(destination),'encoding clone roots overlap');
 const entries=[];
 async function inspect(dir,prefix=''){for(const entry of await readdir(dir,{withFileTypes:true})){const file=prefix+entry.name,path=resolve(source,file),info=await lstat(path,{bigint:true});check(!info.isSymbolicLink()&&(info.isFile()||info.isDirectory()),'encoding cache contains unsupported link/type');if(info.isDirectory())await inspect(path,file+'/');else entries.push({file,info});}}await inspect(source);
 await mkdir(destination,{mode:0o700});await cloneTree(source,destination);let bytes=0n;
 for(const entry of entries){const after=await lstat(resolve(source,entry.file),{bigint:true}),cloned=await lstat(resolve(destination,entry.file),{bigint:true});
  for(const key of ['dev','ino','size','mtimeNs','ctimeNs'])check(after[key]===entry.info[key],'retained cache changed while cloning');
  check(cloned.isFile()&&cloned.dev===after.dev&&cloned.ino!==after.ino&&cloned.nlink===1n&&cloned.size===after.size&&(cloned.mode&0o777n)===(after.mode&0o777n),'encoding clone is not an independent file');bytes+=after.size;
 }
 return {files:entries.length,logicalBytes:String(bytes),method:'macOS clonefile(2) CLONE_NOFOLLOW; no fallback',sourceRetained:true};
}
export function encodingMetrics(reference,native){
 check(reference.length===native.length&&reference.length>0,'encoding metric cardinality');let error=0,ref=0,out=0,dot=0,absolute=0,signed=0,max=0,equal=0;
 for(let i=0;i<reference.length;i++){const a=reference[i],b=native[i];check(Number.isFinite(a)&&Number.isFinite(b),'nonfinite encoding metric');const d=b-a;error+=d*d;ref+=a*a;out+=b*b;dot+=a*b;absolute+=Math.abs(d);signed+=d;max=Math.max(max,Math.abs(d));equal+=Number(a===b);}
 return {elements:reference.length,maxAbsError:max,meanAbsError:absolute/reference.length,rmse:Math.sqrt(error/reference.length),relativeL2:ref?Math.sqrt(error/ref):null,cosineSimilarity:ref&&out?dot/Math.sqrt(ref*out):null,signedMeanError:signed/reference.length,equalValues:equal};
}
export async function compareEncoding(payload,nativeRoot,profile=OLD_ENCODING){
 const inputs=JSON.parse(await readFile(resolve(payload,'upstream/inputs.json'))),bytes=await readFile(resolve(nativeRoot,'native.json'));check(bytes.length<65536,'oversized native encoding receipt');const native=JSON.parse(bytes);
 check(native.kind==='sc-26752-native-normal-encoding/v1'&&native.status==='CAPTURED_ENCODING_ONLY'&&native.nativePin===profile.pin&&native.inputsSha256===INPUT_SHA,'incomplete native encoding');
 check(native.requestSha256===await fileHash(resolve(payload,'request.json'))&&native.referencePngSha256===inputs.provenance.referenceSha256,'native encoding input mismatch');
 check(Object.keys(native.tensors).sort().join(',')===Object.keys(ENCODING_SHAPES).sort().join(','),'unexpected native encoding tensor closure');
 const request=JSON.parse(await readFile(resolve(payload,'request.json')));
 for(const [key,prompt,length] of [['positiveReceipt',request.prompt,71],['negativeReceipt',request.negativePrompt,10]]){const receipt=native[key];check(receipt?.source==='text'&&receipt.sha256===digest(Buffer.from(prompt))&&receipt.sequence_length===length&&receipt.valid_tokens===length,'native text receipt mismatch');}
 for(const item of Object.values(native.tensors))check(typeof item.sourceDtype==='string'&&item.sourceDtype.length>0,'native source dtype missing');
 const metrics={};for(const [name,shape] of Object.entries(ENCODING_SHAPES))metrics[name]=encodingMetrics(await tensorValues(resolve(payload,'upstream'),name,inputs.tensors[name],shape),await tensorValues(nativeRoot,name,native.tensors[name],shape));
 check(native.referenceReceipts.length===1&&native.referenceReceipts[0].latent_sha256===native.tensors.reference_0.sha256,'reference receipt tensor mismatch');
 return {kind:'sc-26752-normal-encoding-comparison/v1',status:'METRICS_ONLY_NO_ACCEPTANCE_VERDICT',inputsSha256:INPUT_SHA,nativeReceiptSha256:digest(bytes),...(profile.precision?{historicalNativePin:PIN,nativePin:profile.pin}:{}),text:{positive:metrics.positive,negative:metrics.negative,positive_mask:metrics.positive_mask,negative_mask:metrics.negative_mask},vae:{reference_0:metrics.reference_0},note:'Normally encoded text and deterministic VAE only; no DiT, denoising, image generation or numerical acceptance threshold.'};
}
export async function executeEncoding(root,payload,env=process.env,profile=OLD_ENCODING){
 check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&process.platform==='darwin','encoding execution requires registered Mac Actions');
 const manifest=await verifyEncodingPayload(payload,profile),source=resolve(env.GITHUB_WORKSPACE,`moss-first-step-native-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`),nativeRoot=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752',`moss-encoding-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);
 await noLinks(source);await noLinks(dirname(nativeRoot));await mkdir(nativeRoot,{mode:0o700});
 const common={...env,CARGO_HOME:resolve(nativeRoot,'cargo-home'),CARGO_TARGET_DIR:resolve(nativeRoot,'target'),CARGO_BUILD_JOBS:'2',CMAKE_BUILD_PARALLEL_LEVEL:'2',CARGO_PROFILE_RELEASE_STRIP:'none',CARGO_INCREMENTAL:'0',HF_HOME:resolve(root,'hf'),HF_HUB_OFFLINE:'1',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'};
 let gitIndex=0;const git=async args=>(await run('git',args,{cwd:source,env:common,log:resolve(root,`encoding-git-${++gitIndex}.log`)})).trimEnd();
 check(await git(['rev-parse','HEAD'])===profile.pin&&await git(['status','--porcelain'])==='','encoding native checkout must be clean exact pin');
 const identity=JSON.parse(await readFile(resolve(payload,'overlay/source-identity.json')));
 check(identity.baseHead===profile.pin&&identity.libPath==='crates/media/mlx-gen/mlx-gen-moss-image/src/lib.rs'&&identity.newModulePath==='crates/media/mlx-gen/mlx-gen-moss-image/src/encoding_probe.rs'&&identity.test==='encoding_probe::fixed_normal_encoding_only','fixed encoding overlay identity required');
 if(profile.precision)check(identity.historicalNativePin===PIN&&identity.historicalManifestSha256===ENCODING_SHA,'encoding overlay historical binding mismatch');
 const lib=safe(source,identity.libPath);check(await fileHash(lib)===identity.originalLibSha256,'encoding original lib mismatch');
 await git(['apply','--check',resolve(payload,'overlay/lib.rs.patch')]);await git(['apply',resolve(payload,'overlay/lib.rs.patch')]);await copyFile(resolve(payload,'overlay/encoding_probe.rs'),safe(source,identity.newModulePath),constants.COPYFILE_EXCL);
 check(await fileHash(lib)===identity.overlayLibSha256,'encoding patched lib mismatch');
 const delta=await git(['status','--porcelain','--untracked-files=all']);check(delta.split('\n').sort().join('\n')===[' M '+identity.libPath,'?? '+identity.newModulePath].sort().join('\n'),'unexpected encoding source delta');
 const clone=await cloneEncodingTarget(ENCODING_WARM,common.CARGO_TARGET_DIR);await noLinks(CARGO_SEED);await copyCacheExclusive(CARGO_SEED,common.CARGO_HOME);
 const models={base:ENCODING_HUB+'/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/1ceede80badddd6a7812069619edaa5633587192',shared:ENCODING_HUB+'/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots/fc5de0d56fc910cbca6f6625a70f5f582b26deb1',adapter:ENCODING_HUB+'/models--chinoll--Moss-Image-Edit/snapshots/3be9c40037e1b89b17aa85ff6f8ee805872aa558'};
 await writeFile(resolve(root,'probe-authentication.json'),JSON.stringify({kind:'sc-26752-encoding-authentication/v1',manifestSha256:profile.sha,nativePin:profile.pin,...(profile.precision?{historicalNativePin:PIN,historicalManifestSha256:ENCODING_SHA}:{}),routeSource:env.GITHUB_SHA,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT,originalUpstreamRun:manifest.upstreamRun,inputsSha256:INPUT_SHA,upstreamSha256:UPSTREAM_SHA,overlay:identity,sourceDelta:delta,source,target:common.CARGO_TARGET_DIR,warmTarget:ENCODING_WARM,clone,models,modelAuthentication:'unchanged native Sources/ArtifactSeal authentication before learned tensor loading'},null,2)+'\n',{flag:'wx'});
 const build=await run('rustup',['run','1.98.0','cargo','test','--locked','--release','-p','mlx-gen-moss-image','--lib','--no-run','--message-format=json'],{cwd:source,env:common,log:resolve(root,'native-build.log')});const binary=probeBinary(build,common.CARGO_TARGET_DIR),binarySha256=await fileHash(binary);
 await verifyEncodingPayload(payload,profile);
 await run(binary,[identity.test,'--exact','--ignored','--nocapture','--test-threads=1'],{cwd:source,env:{...common,MOSS_ENCODING_ADMITTED:'1',MOSS_ENCODING_PAYLOAD:payload,MOSS_ENCODING_OUTPUT:resolve(root,'native'),MOSS_ENCODING_REQUEST_SHA:await fileHash(resolve(payload,'request.json')),MOSS_ENCODING_BASE:models.base,MOSS_ENCODING_SHARED:models.shared,MOSS_ENCODING_ADAPTER:models.adapter},log:resolve(root,'native.log')});
 const comparison=await compareEncoding(payload,resolve(root,'native'),profile);await writeFile(resolve(root,'comparison.json'),JSON.stringify(comparison,null,2)+'\n',{flag:'wx'});
 await verifyEncodingPayload(payload,profile);await cp(resolve(payload,'upstream'),resolve(root,'upstream'),{recursive:true,errorOnExist:true,force:false});
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({kind:'sc-26752-encoding-completion/v1',nativePin:profile.pin,...(profile.precision?{historicalNativePin:PIN,historicalManifestSha256:ENCODING_SHA}:{}),binary,binarySha256,inputsSha256:INPUT_SHA,upstreamSha256:UPSTREAM_SHA,nativeSha256:await fileHash(resolve(root,'native/native.json')),comparisonSha256:await fileHash(resolve(root,'comparison.json')),scope:'normal encoding only; upstream reused byte-for-byte; no DiT/sampling/image run'},null,2)+'\n',{flag:'wx'});
}
async function encodingMain(){
 const phase=JSON.parse(process.env.MOSS_REFERENCE??'{}').phase;
 if(!['upload-encoding','execute-encoding','upload-encoding-precision','execute-encoding-precision'].includes(phase)){
  check(process.argv[2]==='route'&&['validate','execute'].includes(process.argv[3]),'fixed reference route required');
  const code=await new Promise((done,fail)=>{const child=spawn(process.execPath,[fileURLToPath(new URL('./moss-upstream-reference.mjs',import.meta.url)),process.argv[3]],{stdio:'inherit',env:process.env});child.on('error',fail);child.on('close',done);});check(code===0,'original reference route failed');return;
 }
 check(process.env.GITHUB_ACTIONS==='true'&&process.env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&process.platform==='darwin','registered Mac encoding route only');
 const value=encodingEnvelope(process.env.MOSS_REFERENCE,process.env.SOURCE_REF,process.env.GITHUB_SHA,process.env.INPUT_BACKEND);
 const profile=value.phase.endsWith('-precision')?encodingPrecisionProfile():OLD_ENCODING;
 check(/^[1-9][0-9]+$/.test(process.env.GITHUB_RUN_ID)&&/^[1-9][0-9]*$/.test(process.env.GITHUB_RUN_ATTEMPT),'actual encoding Actions identity required');
 const root=resolve(BASE,`moss-encoding-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);
 check(process.argv[2]==='route','fixed route required');
 if(process.argv[3]==='validate'){
  if(value.phase.startsWith('upload-'))await verifyEncodingPayload(profile.staged,profile);
  await appendFile(process.env.GITHUB_OUTPUT,Object.entries({...value,root,payload:profile.staged,...(profile.precision?{native_pin:profile.pin}:{})}).map(([k,v])=>`${k}=${v??''}\n`).join(''));return;
 }
 check(process.argv[3]==='execute'&&value.phase.startsWith('execute-encoding'),'fixed encoding execute required');
 await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});
 const download=resolve(process.env.RUNNER_TEMP,`moss-upstream-reference-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);await verifyEncodingPayload(download,profile);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await executeEncoding(root,resolve(root,'payload'),process.env,profile);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await encodingMain();
