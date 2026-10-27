// Fixed, authenticated first-step comparison only. Actual compilation/model work is Actions-only.
import {readFile,writeFile,mkdir,lstat,stat,realpath,readlink,readdir,copyFile,statfs,appendFile,cp} from 'node:fs/promises';
import {createReadStream,createWriteStream,constants} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname} from 'node:path';
import {spawn,execFile} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual,promisify} from 'node:util';
import {safe,verifyPayload,validateRuntime,PAYLOAD_SHA} from './moss-upstream-reference-common.mjs';
import {copyCacheExclusive,actionsApi,stopped} from './moss-owned-cache-maintenance.mjs';
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
export const ENCODING_PRECISION_STAGED='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/encoding-gated-norm-prep/candidate-main-78100568/payload';
// Reviewed protected native merge and its separately sealed cfg(test) payload.
export const ENCODING_PRECISION_PIN='78100568d8e5b9fb5b199872dc57cb28f5b87340';
export const ENCODING_PRECISION_SHA='0dda6b031fa20f73059094d703e022620a9514cd14d828a347252d2309e30b9f';
const ENCODING_WARM='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/moss-sc-26752/moss-encoding-40327571347-1/target';
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

// Separate diagnostic, never a replacement for the sealed normal-encoding profiles above.
export const BLOCK_TRACE_STAGED='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/encoding-block-trace-prep/payload';
export const BLOCK_TRACE_SHA='6c2af571f051539d3a3156fdfd8352fe73fe057ac63d37b9a3d157d3731327b8';
export const BLOCK_TRACE_PIN='78100568d8e5b9fb5b199872dc57cb28f5b87340';
const BLOCK_TRACE_TREE='d73f2887baa5776b734ef955f1eca2608d288788';
const BLOCK_TRACE_WARM='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/moss-sc-26752/moss-encoding-40345250725-1/target';
const BLOCK_TRACE_PHASES=['upload-encoding-block-trace','execute-encoding-block-trace'];
const BLOCK_TRACE_FILES=['B.png','capture_block_trace.py','compare_block_traces.py','historical/native-h40/native.json','historical/native-h40/negative.f32le','historical/native-h40/positive.f32le','historical/native-h40/probe-authentication.json','historical/native-h40/probe-completion.json','historical/prior-authentication-binding.json','historical/runtime-source-authentication.json','historical/upstream/inputs.json','historical/upstream/negative.f32le','historical/upstream/negative_mask.f32le','historical/upstream/positive.f32le','historical/upstream/positive_mask.f32le','historical/upstream/upstream.json','overlay/encoding_block_trace.rs','overlay/source-identity.json','overlay/text_encoder.rs.patch','request.json','trace-spec.json'];
export function blockTraceProfile(){return {pin:BLOCK_TRACE_PIN,sha:BLOCK_TRACE_SHA,kind:'sc-26752-encoding-block-trace-inputs/v1',staged:BLOCK_TRACE_STAGED};}
export function blockTraceEnvelope(raw,ref,sha,backend){
 const value=JSON.parse(raw);
 check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&BLOCK_TRACE_PHASES.includes(value.phase),'closed block trace envelope required');
 check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact block trace source and mlx required');
 for(const key of ['artifact_id','artifact_run_id'])check(value.phase.startsWith('upload-')?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact block trace artifact identity required');
 return value;
}
export function blockTraceRun(env,platform=process.platform,arch=process.arch){
 check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&env.GITHUB_JOB==='moss-upstream-reference'&&platform==='darwin'&&arch==='arm64','block trace requires its registered dedicated Mac job');
 check(/^[a-f0-9]{40}$/.test(env.GITHUB_SHA??'')&&env.SOURCE_REF===env.GITHUB_SHA&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ID??'')&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ATTEMPT??''),'actual block trace source/run/attempt required');
 check(typeof env.GITHUB_WORKSPACE==='string'&&env.GITHUB_WORKSPACE.startsWith('/')&&resolve(env.GITHUB_WORKSPACE)===env.GITHUB_WORKSPACE,'exact block trace workspace required');
 return {toolingSha:env.GITHUB_SHA,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT,job:env.GITHUB_JOB};
}
export async function verifyBlockTracePayload(root,profile=blockTraceProfile()){
 await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(bytes.length<65536&&digest(bytes)===profile.sha,'block trace manifest SHA mismatch');const manifest=JSON.parse(bytes);
 check(manifest.kind===profile.kind&&manifest.nativePin===BLOCK_TRACE_PIN&&profile.pin===BLOCK_TRACE_PIN&&manifest.nativeTree===BLOCK_TRACE_TREE&&manifest.referencePayload===resolve(PRIOR,'payload')&&manifest.runtimeInventory?.path===resolve(PRIOR,'payload/runtime-files.json'),'block trace provenance mismatch');
 check(Array.isArray(manifest.files)&&manifest.files.map(x=>x.file).sort().join('\n')===[...BLOCK_TRACE_FILES,...(profile.upstreamOnly?['native-carry-inventory.json']:[])].sort().join('\n'),'exact block trace payload closure required');
 const wanted=new Set(['payload-manifest.json']);
 for(const item of manifest.files){const path=safe(root,item.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.nlink===1&&Number.isSafeInteger(item.byteLength)&&item.byteLength<=4*1024*1024&&info.size===item.byteLength&&await fileHash(path)===item.sha256,'block trace payload file mismatch');wanted.add(item.file);}
 async function walk(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){check(!entry.isSymbolicLink(),'block trace payload link');const file=prefix+entry.name;if(entry.isDirectory())await walk(resolve(path,entry.name),file+'/');else check(wanted.has(file),'unlisted block trace payload file');}}await walk(root);
 check(await fileHash(resolve(root,'request.json'))===REQUEST_SHA&&await fileHash(resolve(root,'B.png'))===B_SHA,'block trace original request or reference changed');
 const identity=JSON.parse(await readFile(resolve(root,'overlay/source-identity.json')));
 check(identity.baseHead===profile.pin&&identity.baseTree===BLOCK_TRACE_TREE&&identity.sourcePath==='crates/media/mlx-gen/mlx-gen-moss-image/src/text_encoder.rs'&&identity.newModulePath==='crates/media/mlx-gen/mlx-gen-moss-image/src/encoding_block_trace.rs'&&identity.test==='text_encoder::encoding_block_trace::fixed_block_trace','fixed block trace child-module identity required');
 return manifest;
}
export function blockTraceDelta(delta,identity){check(delta.split('\n').sort().join('\n')===[' M '+identity.sourcePath,'?? '+identity.newModulePath].sort().join('\n'),'unexpected block trace native source delta');}
export function blockTraceTestList(output,test){check(output.split('\n').filter(line=>line.endsWith(': test')).join('\n')===`${test}: test`,'exact compiled block trace test must exist before model execution');}
export function blockTraceProducer(trace,producer){check(trace.kind==='sc-26752-encoding-block-trace/v1'&&trace.backend==='native-mlx'&&trace.device==='metal'&&trace.nativePin===BLOCK_TRACE_PIN&&trace.manifestSha256===BLOCK_TRACE_SHA&&trace.actualToolingSha===producer.toolingSha&&trace.runId===producer.runId&&trace.attempt===producer.attempt,'native block trace changed actual producer');}
export async function executeBlockTrace(root,payload,env=process.env){
 const producer=blockTraceRun(env),profile=blockTraceProfile(),manifest=await verifyBlockTracePayload(payload,profile);
 await verifyPayload(resolve(PRIOR,'payload'));const binding=JSON.parse(await readFile(resolve(payload,'historical/prior-authentication-binding.json')));await verifyPriorBinding(binding);
 const runtimeBytes=await readFile(resolve(PRIOR,'payload/runtime-files.json'));check(digest(runtimeBytes)===manifest.runtimeInventory.sha256,'block trace original runtime inventory changed');const runtime=JSON.parse(runtimeBytes);validateRuntime(runtime);
 check(await realpath(runtime.launcher)===runtime.executable,'block trace runtime executable changed');for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'block trace installed runtime changed');}
 const source=resolve(env.GITHUB_WORKSPACE,`moss-first-step-native-${producer.runId}-${producer.attempt}`),nativeRoot=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752',`moss-encoding-block-trace-${producer.runId}-${producer.attempt}`);
 await noLinks(source);await noLinks(dirname(nativeRoot));await mkdir(nativeRoot,{mode:0o700});
 const common={...env,CARGO_HOME:resolve(nativeRoot,'cargo-home'),CARGO_TARGET_DIR:resolve(nativeRoot,'target'),CARGO_BUILD_JOBS:'2',CMAKE_BUILD_PARALLEL_LEVEL:'2',CARGO_PROFILE_RELEASE_STRIP:'none',CARGO_INCREMENTAL:'0',PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'};
 let index=0;const git=async args=>(await run('git',args,{cwd:source,env:common,log:resolve(root,`trace-git-${++index}.log`)})).trimEnd();
 check(await git(['rev-parse','HEAD'])===profile.pin&&await git(['rev-parse','HEAD^{tree}'])===BLOCK_TRACE_TREE&&await git(['status','--porcelain'])==='','block trace native checkout must be clean exact pin/tree');
 check((await run('git',['rev-parse','HEAD'],{cwd:env.GITHUB_WORKSPACE,env:common,log:resolve(root,'tooling-source.log')})).trim()===producer.toolingSha,'block trace actual tooling checkout changed');
 const identity=JSON.parse(await readFile(resolve(payload,'overlay/source-identity.json'))),text=safe(source,identity.sourcePath);
 check(await fileHash(text)===identity.originalSourceSha256,'block trace original text encoder changed');await git(['apply','--check',resolve(payload,'overlay/text_encoder.rs.patch')]);await git(['apply',resolve(payload,'overlay/text_encoder.rs.patch')]);await copyFile(resolve(payload,'overlay/encoding_block_trace.rs'),safe(source,identity.newModulePath),constants.COPYFILE_EXCL);check(await fileHash(text)===identity.overlaySourceSha256,'block trace patched text encoder changed');
 const delta=await git(['status','--porcelain','--untracked-files=all']);blockTraceDelta(delta,identity);
 const clone=await cloneEncodingTarget(BLOCK_TRACE_WARM,common.CARGO_TARGET_DIR);await noLinks(CARGO_SEED);await copyCacheExclusive(CARGO_SEED,common.CARGO_HOME);
 const models={base:ENCODING_HUB+'/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/1ceede80badddd6a7812069619edaa5633587192',shared:ENCODING_HUB+'/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots/fc5de0d56fc910cbca6f6625a70f5f582b26deb1',adapter:ENCODING_HUB+'/models--chinoll--Moss-Image-Edit/snapshots/3be9c40037e1b89b17aa85ff6f8ee805872aa558'};
 await writeFile(resolve(root,'probe-authentication.json'),JSON.stringify({kind:'sc-26752-block-trace-authentication/v1',...producer,nativePin:profile.pin,nativeTree:BLOCK_TRACE_TREE,manifestSha256:profile.sha,overlay:identity,sourceDelta:delta,source,nativeRoot,clone,warmTarget:BLOCK_TRACE_WARM,models,originalRuntimeInventory:manifest.runtimeInventory,originalModelAuthenticationSha256:binding.authenticationSha256,modelAuthentication:'unchanged Sources/ArtifactSeal plus original full-request native admission; frozen MPS runtime/fork/stat binding',runner:{name:env.RUNNER_NAME,os:env.RUNNER_OS,arch:env.RUNNER_ARCH},scope:'three native text forwards then three frozen MPS text forwards; no VAE/DiT/sampling/vision forward; metrics only'},null,2)+'\n',{flag:'wx'});
 await run('rustup',['run','1.98.0','rustc','--version','--verbose'],{cwd:source,env:common,log:resolve(root,'toolchain.log')});await run('/usr/sbin/system_profiler',['SPDisplaysDataType','-json'],{cwd:source,env:common,log:resolve(root,'device.log')});
 const build=await run('rustup',['run','1.98.0','cargo','test','--locked','--release','-p','mlx-gen-moss-image','--lib','--no-run','--message-format=json'],{cwd:source,env:common,log:resolve(root,'native-build.log')}),binary=probeBinary(build,common.CARGO_TARGET_DIR),binarySha256=await fileHash(binary);
 blockTraceTestList(await run(binary,[identity.test,'--exact','--ignored','--list'],{cwd:source,env:common,log:resolve(root,'native-test-list.log')}),identity.test);
 await verifyBlockTracePayload(payload);await verifyPriorBinding(binding);
 const nativeOutput=await run(binary,[identity.test,'--exact','--ignored','--nocapture','--test-threads=1'],{cwd:source,env:{...common,MOSS_BLOCK_TRACE_ADMITTED:'1',MOSS_TRACE_PAYLOAD:payload,MOSS_TRACE_MANIFEST_SHA:profile.sha,MOSS_TRACE_NATIVE_OUTPUT:resolve(root,'native'),MOSS_ENCODING_BASE:models.base,MOSS_ENCODING_SHARED:models.shared,MOSS_ENCODING_ADAPTER:models.adapter},log:resolve(root,'native.log')});check(/test result: ok\. 1 passed; 0 failed; 0 ignored/.test(nativeOutput),'block trace native test did not execute exactly once');
 blockTraceProducer(JSON.parse(await readFile(resolve(root,'native/trace.json'))),producer);
 await verifyBlockTracePayload(payload);await verifyPriorBinding(binding);
 await run(runtime.launcher,['-B',resolve(payload,'capture_block_trace.py'),'--payload',payload,'--manifest-sha',profile.sha,'--native-trace',resolve(root,'native'),'--reference-payload',resolve(PRIOR,'payload'),'--snapshot-view',resolve(PRIOR,'snapshot-view'),'--output-dir',resolve(root,'upstream')],{cwd:payload,env:common,log:resolve(root,'upstream.log')});
 await run(runtime.launcher,['-B',resolve(payload,'compare_block_traces.py'),'--native',resolve(root,'native'),'--upstream',resolve(root,'upstream'),'--output',resolve(root,'comparison.json')],{cwd:payload,env:common,log:resolve(root,'comparison.log')});
 await verifyBlockTracePayload(payload);await verifyPriorBinding(binding);
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({kind:'sc-26752-block-trace-completion/v1',...producer,nativePin:profile.pin,manifestSha256:profile.sha,binary,binarySha256,nativeTraceSha256:await fileHash(resolve(root,'native/trace.json')),upstreamTraceSha256:await fileHash(resolve(root,'upstream/trace.json')),comparisonSha256:await fileHash(resolve(root,'comparison.json')),scope:'three paired text traces; metrics and capture-drift observations only, no acceptance or causal verdict'},null,2)+'\n',{flag:'wx'});
}
async function blockTraceMain(){
 const env=process.env;blockTraceRun(env);const value=blockTraceEnvelope(env.MOSS_REFERENCE,env.SOURCE_REF,env.GITHUB_SHA,env.INPUT_BACKEND),profile=blockTraceProfile(),root=resolve(BASE,`moss-encoding-block-trace-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);
 check(process.argv[2]==='route','fixed block trace route required');
 if(process.argv[3]==='validate'){if(value.phase.startsWith('upload-'))await verifyBlockTracePayload(profile.staged);await appendFile(env.GITHUB_OUTPUT,Object.entries({...value,root,payload:profile.staged,native_pin:profile.pin}).map(([k,v])=>`${k}=${v??''}\n`).join(''));return;}
 check(process.argv[3]==='execute'&&value.phase==='execute-encoding-block-trace','fixed block trace execute required');await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});
 const download=resolve(env.RUNNER_TEMP,`moss-upstream-reference-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);await verifyBlockTracePayload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await executeBlockTrace(root,resolve(root,'payload'),env);
}

// Explicit upstream-only carry of one authenticated completed native capture. No native execution.
export const BLOCK_TRACE_UPSTREAM_STAGED='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/encoding-block-trace-portable-prep/payload';
export const BLOCK_TRACE_UPSTREAM_SHA='eeb7178ee939249729ad4049accd5f2a6593ada7592c0c27daa609716f658b9a';
const BLOCK_TRACE_UPSTREAM_PHASES=['upload-encoding-block-trace-upstream','execute-encoding-block-trace-upstream'];
const BLOCK_TRACE_NATIVE={runId:'40384275958',attempt:'1',toolingSha:'d2b5ff1a45b36a6b316974c4e8a080d31be0f407',artifactId:'12681908625',artifactName:'moss-upstream-reference-result-40384275958-1',artifactBytes:37379611,artifactSha256:'f33e02222aeb963b7d7a071cebb04607483d315c186097b362fb02a8ddddb1f1'};
export function blockTraceUpstreamProfile(){return {pin:BLOCK_TRACE_PIN,sha:BLOCK_TRACE_UPSTREAM_SHA,kind:'sc-26752-encoding-block-trace-upstream-inputs/v1',staged:BLOCK_TRACE_UPSTREAM_STAGED,upstreamOnly:true};}
export function blockTraceUpstreamEnvelope(raw,ref,sha,backend){
 const value=JSON.parse(raw);check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&BLOCK_TRACE_UPSTREAM_PHASES.includes(value.phase),'closed upstream-only envelope required');
 check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact upstream-only source and mlx required');
 for(const key of ['artifact_id','artifact_run_id'])check(value.phase.startsWith('upload-')?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact upstream-only payload artifact identity required');return value;
}
export function blockTraceCarryIdentity(carry){
 check(carry.kind==='sc-26752-block-trace-native-carry-inventory/v1'&&carry.nativePin===BLOCK_TRACE_PIN&&carry.nativeManifestSha256===BLOCK_TRACE_SHA&&carry.originalPayloadManifestSha256===BLOCK_TRACE_SHA&&carry.actualToolingSha===BLOCK_TRACE_NATIVE.toolingSha&&carry.runId===BLOCK_TRACE_NATIVE.runId&&carry.attempt===BLOCK_TRACE_NATIVE.attempt,'fixed original native carry producer required');
 for(const key of ['artifactId','artifactName','artifactBytes','artifactSha256'])check(carry[key]===BLOCK_TRACE_NATIVE[key],'fixed original native artifact required');
 check(/^[a-f0-9]{64}$/.test(carry.nativeTraceSha256??'')&&/^[a-f0-9]{64}$/.test(carry.probeAuthenticationSha256??''),'actual native authentication hashes required');
 const names=['trace.json',...['positive','negative','positive_last_token_variant'].flatMap(name=>Array.from({length:34},(_,i)=>`${name}/boundary-${String(i).padStart(2,'0')}.f32le`))].sort();
 check(Array.isArray(carry.nativeFiles)&&carry.nativeFiles.length===103&&carry.nativeFiles.map(x=>x.file).sort().join('\n')===names.join('\n')&&carry.nativeFiles.every(x=>Object.keys(x).sort().join(',')==='byteLength,file,sha256'&&Number.isSafeInteger(x.byteLength)&&x.byteLength>0&&x.byteLength<=4*1024*1024&&/^[a-f0-9]{64}$/.test(x.sha256)),'exact bounded103 native file inventory required');return carry;
}
export function blockTraceCarryArtifact(carry,artifact){
 blockTraceCarryIdentity(carry);check(String(artifact.id)===carry.artifactId&&artifact.name===carry.artifactName&&artifact.size_in_bytes===carry.artifactBytes&&artifact.digest===`sha256:${carry.artifactSha256}`&&artifact.expired===false&&String(artifact.workflow_run?.id)===carry.runId&&artifact.workflow_run?.head_sha===carry.actualToolingSha,'actual native artifact metadata changed');return artifact;
}
export async function verifyBlockTraceUpstreamPayload(root,profile=blockTraceUpstreamProfile()){
 const manifest=await verifyBlockTracePayload(root,profile),descriptor=manifest.nativeCarry;check(descriptor?.file==='native-carry-inventory.json'&&Number.isSafeInteger(descriptor.byteLength)&&descriptor.byteLength<65536&&/^[a-f0-9]{64}$/.test(descriptor.sha256??''),'sealed carry inventory descriptor required');
 const bytes=await readFile(resolve(root,descriptor.file));check(bytes.length===descriptor.byteLength&&digest(bytes)===descriptor.sha256,'sealed carry inventory changed');const carry=blockTraceCarryIdentity(JSON.parse(bytes));
 const original=await verifyBlockTracePayload(BLOCK_TRACE_STAGED),core=original.files.filter(x=>!['capture_block_trace.py','compare_block_traces.py'].includes(x.file)),sorted=items=>JSON.stringify(items.slice().sort((a,b)=>a.file.localeCompare(b.file)));
 check(carry.coreInputFiles?.length===19&&sorted(carry.coreInputFiles)===sorted(core)&&sorted(manifest.files.filter(x=>core.some(y=>y.file===x.file)))===sorted(core),'native/upstream common input overlay/runtime closure changed');return {manifest,carry};
}
export async function verifyBlockTraceNative(root,carry,payload){
 blockTraceCarryIdentity(carry);await noLinks(root);const wanted=new Set(carry.nativeFiles.map(x=>x.file));
 for(const item of carry.nativeFiles){const file=safe(root,item.file);await noLinks(file);const info=await lstat(file);check(info.isFile()&&info.nlink===1&&info.size===item.byteLength&&await fileHash(file)===item.sha256,'carried native output bytes changed');}
 async function walk(dir,prefix=''){for(const item of await readdir(dir,{withFileTypes:true})){check(!item.isSymbolicLink(),'carried native output link');const name=prefix+item.name;if(item.isDirectory())await walk(resolve(dir,item.name),name+'/');else check(wanted.has(name),'unlisted carried native output');}}await walk(root);
 check(await fileHash(resolve(root,'trace.json'))===carry.nativeTraceSha256,'carried trace identity changed');const trace=JSON.parse(await readFile(resolve(root,'trace.json')));blockTraceProducer(trace,{toolingSha:carry.actualToolingSha,runId:carry.runId,attempt:carry.attempt});
 const authentication=resolve(dirname(root),'probe-authentication.json');await noLinks(authentication);check(await fileHash(authentication)===carry.probeAuthenticationSha256,'carried native probe authentication changed');const auth=JSON.parse(await readFile(authentication)),identity=JSON.parse(await readFile(resolve(payload,'overlay/source-identity.json')));
 check(auth.kind==='sc-26752-block-trace-authentication/v1'&&auth.toolingSha===carry.actualToolingSha&&auth.runId===carry.runId&&auth.attempt===carry.attempt&&auth.nativePin===BLOCK_TRACE_PIN&&auth.nativeTree===BLOCK_TRACE_TREE&&auth.manifestSha256===BLOCK_TRACE_SHA&&JSON.stringify(auth.overlay)===JSON.stringify(identity),'carried native compiled source closure changed');return trace;
}
export function blockTraceCarryReceipt(carry,current,manifestSha256){return {kind:'sc-26752-block-trace-native-carry/v1',originalNative:{nativePin:carry.nativePin,manifestSha256:carry.nativeManifestSha256,toolingSha:carry.actualToolingSha,runId:carry.runId,attempt:carry.attempt,traceSha256:carry.nativeTraceSha256},currentUpstream:{...current,manifestSha256},artifact:{id:carry.artifactId,name:carry.artifactName,byteLength:carry.artifactBytes,sha256:carry.artifactSha256},nativeFilesSha256:digest(JSON.stringify(carry.nativeFiles)),probeAuthenticationSha256:carry.probeAuthenticationSha256};}
export function blockTraceUpstreamCommands(root,payload,manifestSha256){const receipt=resolve(root,'native-carry-authentication.json');return {capture:['-B',resolve(payload,'capture_block_trace.py'),'--payload',payload,'--manifest-sha',manifestSha256,'--native-trace',resolve(root,'native'),'--reference-payload',resolve(PRIOR,'payload'),'--snapshot-view',resolve(PRIOR,'snapshot-view'),'--output-dir',resolve(root,'upstream'),'--stat-node',process.execPath,'--native-carry-auth',receipt],compare:['-B',resolve(payload,'compare_block_traces.py'),'--native',resolve(root,'native'),'--upstream',resolve(root,'upstream'),'--output',resolve(root,'comparison.json'),'--native-carry-auth',receipt]};}
export async function executeBlockTraceUpstream(root,payload,nativeArtifact,env=process.env){
 const producer=blockTraceRun(env),profile=blockTraceUpstreamProfile(),{manifest,carry}=await verifyBlockTraceUpstreamPayload(payload);check(producer.runId!==carry.runId&&producer.toolingSha!==carry.actualToolingSha,'upstream carry requires separate actual new producer');
 check((await run('git',['rev-parse','HEAD'],{cwd:env.GITHUB_WORKSPACE,env,log:resolve(root,'tooling-source.log')})).trim()===producer.toolingSha,'actual upstream tooling checkout changed');
 const api=route=>actionsApi(env.MOSS_TRACE_TOKEN,route),stop=await stopped({run:carry.runId,source:carry.actualToolingSha},api);blockTraceCarryArtifact(carry,await api(`actions/artifacts/${carry.artifactId}`));
 await verifyBlockTraceNative(resolve(nativeArtifact,'native'),carry,payload);await cp(resolve(nativeArtifact,'native'),resolve(root,'native'),{recursive:true,errorOnExist:true,force:false});await copyFile(resolve(nativeArtifact,'probe-authentication.json'),resolve(root,'probe-authentication.json'),constants.COPYFILE_EXCL);await verifyBlockTraceNative(resolve(root,'native'),carry,payload);
 await verifyPayload(resolve(PRIOR,'payload'));const binding=JSON.parse(await readFile(resolve(payload,'historical/prior-authentication-binding.json')));await verifyPriorBinding(binding);
 const runtimeBytes=await readFile(resolve(PRIOR,'payload/runtime-files.json'));check(digest(runtimeBytes)===manifest.runtimeInventory.sha256,'upstream original runtime inventory changed');const runtime=JSON.parse(runtimeBytes);validateRuntime(runtime);check(await realpath(runtime.launcher)===runtime.executable,'upstream frozen runtime launcher changed');for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'upstream frozen runtime file changed');}
 const current={toolingSha:producer.toolingSha,runId:producer.runId,attempt:producer.attempt},receipt=blockTraceCarryReceipt(carry,current,profile.sha);await writeFile(resolve(root,'native-carry-authentication.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});await writeFile(resolve(root,'upstream-authentication.json'),JSON.stringify({kind:'sc-26752-block-trace-upstream-authentication/v1',...current,nativeCarry:receipt,stoppedOriginal:stop,manifestSha256:profile.sha,runtimeInventory:manifest.runtimeInventory,statNode:{path:process.execPath,version:process.version,sha256:await fileHash(process.execPath)},scope:'carried native capture; three upstream text forwards only; no native build/forward'},null,2)+'\n',{flag:'wx'});
 const common={...env,PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'},commands=blockTraceUpstreamCommands(root,payload,profile.sha);check(/^v24\./.test(process.version),'actual Node24 metadata bridge required');
 await verifyBlockTraceUpstreamPayload(payload);await verifyPriorBinding(binding);await run(runtime.launcher,commands.capture,{cwd:payload,env:common,log:resolve(root,'upstream.log')});await run(runtime.launcher,commands.compare,{cwd:payload,env:common,log:resolve(root,'comparison.log')});await verifyBlockTraceUpstreamPayload(payload);await verifyPriorBinding(binding);await verifyBlockTraceNative(resolve(root,'native'),carry,payload);
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({kind:'sc-26752-block-trace-upstream-completion/v1',...current,manifestSha256:profile.sha,nativeCarry:receipt,nativeTraceSha256:carry.nativeTraceSha256,upstreamTraceSha256:await fileHash(resolve(root,'upstream/trace.json')),comparisonSha256:await fileHash(resolve(root,'comparison.json')),scope:'original native producer retained; three fresh upstream text forwards; metrics only, no acceptance or causal verdict'},null,2)+'\n',{flag:'wx'});
}
async function blockTraceUpstreamMain(){
 const env=process.env;blockTraceRun(env);const value=blockTraceUpstreamEnvelope(env.MOSS_REFERENCE,env.SOURCE_REF,env.GITHUB_SHA,env.INPUT_BACKEND),profile=blockTraceUpstreamProfile(),root=resolve(BASE,`moss-encoding-block-trace-upstream-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);check(process.argv[2]==='route','fixed upstream-only route required');
 if(process.argv[3]==='validate'){const {carry}=await verifyBlockTraceUpstreamPayload(profile.staged);await appendFile(env.GITHUB_OUTPUT,Object.entries({...value,root,payload:profile.staged,native_artifact_id:carry.artifactId,native_artifact_run_id:carry.runId}).map(([k,v])=>`${k}=${v??''}\n`).join(''));return;}
 check(process.argv[3]==='execute'&&value.phase==='execute-encoding-block-trace-upstream','fixed upstream-only execute required');await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});
 const download=resolve(env.RUNNER_TEMP,`moss-upstream-reference-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);await verifyBlockTraceUpstreamPayload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await executeBlockTraceUpstream(root,resolve(root,'payload'),resolve(env.RUNNER_TEMP,`moss-native-trace-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`),env);
}
// Fixed same-input layer3 discriminator. The sealed overlay is test-only on native781.
export const LAYER3_STAGED='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/encoding-layer3-discriminator-cpuload-prep/payload';
export const LAYER3_SHA='eaef78ecfc806692b5b221a1a1daf9d7af0a22e1206ee334c7562a519d7c0210';
export const LAYER3_PIN=BLOCK_TRACE_PIN;
const LAYER3_PHASES=['upload-encoding-layer3','execute-encoding-layer3'];
const LAYER3_FILES=['capture_layer3.py','compare_layer3.py','frozen_runtime_bridge.py','historical/parent-artifact-authentication.json','historical/parent-artifact-metadata.json','historical/parent-carry-authentication.json','historical/parent-comparison.json','historical/parent-completion.json','historical/parent-native-authentication.json','historical/parent-native-trace.json','historical/parent-payload-manifest.json','historical/parent-upstream-authentication.json','historical/parent-upstream-trace.json','historical/prior-authentication-binding.json','historical/runtime-source-authentication.json','historical/upstream-positive-boundary-04.f32le','input/positive-boundary-03.f32le','layer3-contract.json','overlay/layer3_probe.rs','overlay/qwen35.rs.patch','overlay/source-identity.json'];
const LAYER3_INPUT_SHA='5bb3bb46137bf6d3773f309375ab1c58cb9fd3042c487b2e3d1f6adfb4a433e8',LAYER3_ORACLE_SHA='75ddf91405bc9135c909817676b708a1ce33d710eeff89bc31053e03ec7637e1';
export function layer3Profile(){return {pin:LAYER3_PIN,sha:LAYER3_SHA,kind:'sc-26752-encoding-layer3-inputs/v1',staged:LAYER3_STAGED};}
export function layer3Envelope(raw,ref,sha,backend){
 const value=JSON.parse(raw);check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&LAYER3_PHASES.includes(value.phase),'closed layer3 envelope required');check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact layer3 source and mlx required');
 for(const key of ['artifact_id','artifact_run_id'])check(value.phase.startsWith('upload-')?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact layer3 artifact identity required');return value;
}
export function layer3Binary(output,target){
 const matches=output.split('\n').filter(Boolean).flatMap(line=>{try{return[JSON.parse(line)];}catch{return[];}}).filter(x=>x.reason==='compiler-artifact'&&x.target?.name==='mlx_llm'&&x.profile?.test&&typeof x.executable==='string');check(matches.length===1,'one exact mlx_llm libtest executable required');const path=resolve(matches[0].executable);check(path.startsWith(resolve(target)+'/release/deps/'),'layer3 executable escaped owned target');return path;
}
export async function verifyLayer3Payload(root,profile=layer3Profile()){
 await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(bytes.length<65536&&digest(bytes)===profile.sha,'layer3 manifest SHA mismatch');const manifest=JSON.parse(bytes);
 check(profile.pin===LAYER3_PIN&&manifest.kind===profile.kind&&manifest.nativePin===LAYER3_PIN&&manifest.nativeTree===BLOCK_TRACE_TREE&&manifest.referencePayload===resolve(PRIOR,'payload')&&manifest.runtimeInventory?.path===resolve(PRIOR,'payload/runtime-files.json')&&manifest.runtimeInventory.sha256==='053fa7d3a714423a5ab82c9e7207768b4d3c53216ed9f175d22a907fde0e8cae','layer3 provenance mismatch');
 check(Array.isArray(manifest.files)&&manifest.files.map(x=>x.file).sort().join('\n')===LAYER3_FILES.slice().sort().join('\n'),'exact layer3 payload closure required');const wanted=new Set(['payload-manifest.json']);
 for(const item of manifest.files){const path=safe(root,item.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.nlink===1&&Number.isSafeInteger(item.byteLength)&&item.byteLength<=4*1024*1024&&info.size===item.byteLength&&await fileHash(path)===item.sha256,'layer3 payload file mismatch');wanted.add(item.file);}
 async function walk(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){check(!entry.isSymbolicLink(),'layer3 payload link');const file=prefix+entry.name;if(entry.isDirectory())await walk(resolve(path,entry.name),file+'/');else check(wanted.has(file),'unlisted layer3 payload file');}}await walk(root);
 const identity=JSON.parse(await readFile(resolve(root,'overlay/source-identity.json')));check(identity.nativePin===LAYER3_PIN&&identity.nativeTree===BLOCK_TRACE_TREE&&identity.sourcePath==='crates/llm/mlx-llm/src/models/qwen35.rs'&&identity.newModulePath==='crates/llm/mlx-llm/src/models/qwen35/layer3_probe.rs'&&identity.originalSourceSha256==='a5439b4046922d54529c0a26f5b5b6b2cf668f4ea3e180055c4a9dd79e268a87'&&identity.overlaySourceSha256==='61bfe40f19f07e0c2e6f3dc70d7382abf7925c0968ef6d3ac2693124c7de36cc'&&identity.package==='mlx-llm'&&identity.binaryStem==='mlx_llm'&&identity.test==='models::qwen35::layer3_probe::fixed_same_input_layer3','fixed layer3 overlay identity required');
 const contract=JSON.parse(await readFile(resolve(root,'layer3-contract.json'))),input=await readFile(resolve(root,'input/positive-boundary-03.f32le'));
 check(contract.kind==='sc-26752-same-input-layer3-contract/v1'&&contract.nativePin===LAYER3_PIN&&contract.layerIndex===3&&JSON.stringify(contract.shape)==='[1,71,4096]'&&contract.offset===0&&contract.freshCache===true&&JSON.stringify(contract.attentionMask)===JSON.stringify(Array(71).fill(1))&&JSON.stringify(contract.positionIds)===JSON.stringify(Array.from({length:71},(_,i)=>i))&&isDeepStrictEqual(contract.sourceInput,manifest.sourceInput)&&contract.sourceInput.upstreamRunId==='40393828906'&&contract.sourceInput.upstreamAttempt==='1'&&contract.sourceInput.upstreamToolingSha==='d6808ab06a2422fe50274aac7332c255eeeb37e9'&&contract.sourceInput.artifactId==='12686156941'&&contract.sourceInput.sha256===LAYER3_INPUT_SHA&&contract.layerCalls.native===2&&contract.layerCalls.upstream===1&&contract.fullEncoderForwards===0,'fixed layer3 input/call/source contract changed');
 check(input.length===1163264&&digest(input)===LAYER3_INPUT_SHA&&await fileHash(resolve(root,'historical/upstream-positive-boundary-04.f32le'))===LAYER3_ORACLE_SHA&&contract.historicalUpstreamLayer3Output.sha256===LAYER3_ORACLE_SHA,'layer3 historical input/oracle bytes changed');for(let offset=0;offset<input.length;offset+=4)check(Number.isFinite(input.readFloatLE(offset))&&(input.readUInt32LE(offset)&65535)===0,'layer3 input is not exact finite BF16 wire');return {manifest,identity,contract};
}
export function layer3Producer(trace,producer,backend='native-mlx',sourceInput){
 check(['native-mlx','upstream-mps'].includes(backend)&&trace.kind==='sc-26752-same-input-layer3/v1'&&trace.backend===backend&&trace.device===(backend==='native-mlx'?'metal':'mps')&&trace.nativePin===LAYER3_PIN&&trace.manifestSha256===LAYER3_SHA&&trace.actualToolingSha===producer.toolingSha&&trace.runId===producer.runId&&trace.attempt===producer.attempt&&trace.layerCalls===(backend==='native-mlx'?2:1)&&trace.fullEncoderForwards===0&&trace.inputSha256===LAYER3_INPUT_SHA&&trace.layerIndex===3&&trace.offset===0&&trace.stages?.final?.file==='final.f32le'&&/^[a-f0-9]{64}$/.test(trace.stages.final.sha256)&&trace.oracle?.bytesEqual===true&&(!sourceInput||isDeepStrictEqual(trace.sourceInput,sourceInput)),'layer3 actual producer/input/oracle changed');
 const oracle=trace.oracle;check((backend==='native-mlx'?oracle.normalLayer3Sha256===oracle.stagedLayer3Sha256&&oracle.stagedLayer3Sha256===trace.stages.final.sha256:oracle.historicalLayer3Sha256===LAYER3_ORACLE_SHA&&oracle.freshLayer3Sha256===LAYER3_ORACLE_SHA&&trace.stages.final.sha256===LAYER3_ORACLE_SHA),'layer3 exact oracle drift');return trace;
}
export function layer3Commands(root,payload){return {build:['run','1.98.0','cargo','test','--locked','--release','-p','mlx-llm','--lib','--no-run','--message-format=json'],capture:['-B',resolve(payload,'capture_layer3.py'),'--payload',payload,'--manifest-sha',LAYER3_SHA,'--reference-payload',resolve(PRIOR,'payload'),'--snapshot-view',resolve(PRIOR,'snapshot-view'),'--output-dir',resolve(root,'upstream'),'--stat-node',process.execPath],compare:['-B',resolve(payload,'compare_layer3.py'),'--payload',payload,'--manifest-sha',LAYER3_SHA,'--native',resolve(root,'native'),'--upstream',resolve(root,'upstream'),'--output',resolve(root,'comparison.json')]};}
export async function executeLayer3(root,payload,env=process.env){
 const producer=blockTraceRun(env),profile=layer3Profile(),{manifest,identity}=await verifyLayer3Payload(payload);check(/^v24\./.test(process.version),'actual Node24 layer3 stat bridge required');
 await verifyPayload(resolve(PRIOR,'payload'));const binding=JSON.parse(await readFile(resolve(payload,'historical/prior-authentication-binding.json')));await verifyPriorBinding(binding);
 const runtimeBytes=await readFile(resolve(PRIOR,'payload/runtime-files.json'));check(digest(runtimeBytes)===manifest.runtimeInventory.sha256,'layer3 original runtime inventory changed');const runtime=JSON.parse(runtimeBytes);validateRuntime(runtime);check(await realpath(runtime.launcher)===runtime.executable,'layer3 frozen runtime executable changed');for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'layer3 installed runtime changed');}
 const api=route=>actionsApi(env.MOSS_TRACE_TOKEN,route),artifact=await api('actions/artifacts/12686156941');check(String(artifact.id)==='12686156941'&&artifact.name===manifest.sourceInput.artifactName&&artifact.size_in_bytes===manifest.sourceInput.artifactBytes&&artifact.digest===`sha256:${manifest.sourceInput.artifactSha256}`&&artifact.expired===false&&String(artifact.workflow_run?.id)==='40393828906'&&artifact.workflow_run.head_sha===manifest.sourceInput.upstreamToolingSha,'layer3 original artifact producer changed');
 const source=resolve(env.GITHUB_WORKSPACE,`moss-first-step-native-${producer.runId}-${producer.attempt}`),nativeRoot=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752',`moss-encoding-layer3-${producer.runId}-${producer.attempt}`);await noLinks(source);await noLinks(dirname(nativeRoot));await mkdir(nativeRoot,{mode:0o700});
 const common={...env,CARGO_HOME:resolve(nativeRoot,'cargo-home'),CARGO_TARGET_DIR:resolve(nativeRoot,'target'),CARGO_BUILD_JOBS:'2',CMAKE_BUILD_PARALLEL_LEVEL:'2',CARGO_PROFILE_RELEASE_STRIP:'none',CARGO_INCREMENTAL:'0',PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'},commands=layer3Commands(root,payload);
 let index=0;const git=async args=>(await run('git',args,{cwd:source,env:common,log:resolve(root,`layer3-git-${++index}.log`)})).trimEnd();check(await git(['rev-parse','HEAD'])===profile.pin&&await git(['rev-parse','HEAD^{tree}'])===BLOCK_TRACE_TREE&&await git(['status','--porcelain'])==='','layer3 native checkout must be clean exact pin/tree');check((await run('git',['rev-parse','HEAD'],{cwd:env.GITHUB_WORKSPACE,env:common,log:resolve(root,'tooling-source.log')})).trim()===producer.toolingSha,'layer3 actual tooling checkout changed');
 const text=safe(source,identity.sourcePath),original=await readFile(text);check(digest(original)===identity.originalSourceSha256,'layer3 original qwen35 source changed');await git(['apply','--check',resolve(payload,'overlay/qwen35.rs.patch')]);await git(['apply',resolve(payload,'overlay/qwen35.rs.patch')]);await mkdir(dirname(safe(source,identity.newModulePath)),{recursive:true});await copyFile(resolve(payload,'overlay/layer3_probe.rs'),safe(source,identity.newModulePath),constants.COPYFILE_EXCL);
 const verifySource=async()=>{const current=await readFile(text);check(digest(current)===identity.overlaySourceSha256&&current.subarray(0,original.length).equals(original)&&await fileHash(safe(source,identity.newModulePath))===await fileHash(resolve(payload,'overlay/layer3_probe.rs')),'layer3 append-only native source changed');blockTraceDelta(await git(['status','--porcelain','--untracked-files=all']),identity);};await verifySource();
 const clone=await cloneEncodingTarget(BLOCK_TRACE_WARM,common.CARGO_TARGET_DIR);await noLinks(CARGO_SEED);await copyCacheExclusive(CARGO_SEED,common.CARGO_HOME);
 await writeFile(resolve(root,'probe-authentication.json'),JSON.stringify({kind:'sc-26752-layer3-authentication/v1',...producer,nativePin:profile.pin,nativeTree:BLOCK_TRACE_TREE,manifestSha256:profile.sha,overlay:identity,sourceInput:manifest.sourceInput,parentArtifact:artifact,source,nativeRoot,clone,warmTarget:BLOCK_TRACE_WARM,originalTargetRetained:true,originalRuntimeInventory:manifest.runtimeInventory,originalModelAuthenticationSha256:binding.authenticationSha256,statNode:{path:process.execPath,version:process.version,sha256:await fileHash(process.execPath)},loader:'Direct mlx-llm same four BF16 text shards/language_model filter/MossFrozenHf materialized constructor under original23stat auth; downstream product Sources loader did not run.',scope:'2native +1HF same-input layer3 calls only; zero full encoder/vision/VAE/DiT/sampling forwards'},null,2)+'\n',{flag:'wx'});
 await run('rustup',['run','1.98.0','rustc','--version','--verbose'],{cwd:source,env:common,log:resolve(root,'toolchain.log')});await run('/usr/sbin/system_profiler',['SPDisplaysDataType','-json'],{cwd:source,env:common,log:resolve(root,'device.log')});
 const preflight="import importlib.util,json,pathlib,sys;p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('sealed_layer3_compare',p/'compare_layer3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.validate_contract(p,json.loads((p/'layer3-contract.json').read_bytes()))";await run(runtime.launcher,['-B','-c',preflight,payload],{cwd:payload,env:common,log:resolve(root,'input-preflight.log')});
 const build=await run('rustup',commands.build,{cwd:source,env:common,log:resolve(root,'native-build.log')}),binary=layer3Binary(build,common.CARGO_TARGET_DIR),binarySha256=await fileHash(binary);blockTraceTestList(await run(binary,[identity.test,'--exact','--ignored','--list'],{cwd:source,env:common,log:resolve(root,'native-test-list.log')}),identity.test);
 await verifyLayer3Payload(payload);await verifyPriorBinding(binding);await verifySource();const nativeOutput=await run(binary,[identity.test,'--exact','--ignored','--nocapture','--test-threads=1'],{cwd:source,env:{...common,MOSS_LAYER3_ADMITTED:'1',MOSS_LAYER3_PAYLOAD:payload,MOSS_LAYER3_MANIFEST_SHA:profile.sha,MOSS_LAYER3_NATIVE_OUTPUT:resolve(root,'native')},log:resolve(root,'native.log')});check(/test result: ok\. 1 passed; 0 failed; 0 ignored/.test(nativeOutput),'layer3 native test did not execute exactly once');
 const native=layer3Producer(JSON.parse(await readFile(resolve(root,'native/trace.json'))),producer,'native-mlx',manifest.sourceInput);check(await fileHash(safe(resolve(root,'native'),native.stages.final.file))===native.stages.final.sha256,'layer3 native final oracle bytes changed');await verifyLayer3Payload(payload);await verifyPriorBinding(binding);await verifySource();
 await run(runtime.launcher,commands.capture,{cwd:payload,env:common,log:resolve(root,'upstream.log')});layer3Producer(JSON.parse(await readFile(resolve(root,'upstream/trace.json'))),producer,'upstream-mps',manifest.sourceInput);await run(runtime.launcher,commands.compare,{cwd:payload,env:common,log:resolve(root,'comparison.log')});await verifyLayer3Payload(payload);await verifyPriorBinding(binding);await verifySource();
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({kind:'sc-26752-layer3-completion/v1',...producer,nativePin:profile.pin,manifestSha256:profile.sha,binary,binarySha256,nativeTraceSha256:await fileHash(resolve(root,'native/trace.json')),upstreamTraceSha256:await fileHash(resolve(root,'upstream/trace.json')),comparisonSha256:await fileHash(resolve(root,'comparison.json')),scope:'2native +1HF one-layer calls; zero full encoder forwards; metrics/oracle observations only, no quality/causal verdict'},null,2)+'\n',{flag:'wx'});
}
async function layer3Main(){
 const env=process.env;blockTraceRun(env);const value=layer3Envelope(env.MOSS_REFERENCE,env.SOURCE_REF,env.GITHUB_SHA,env.INPUT_BACKEND),profile=layer3Profile(),root=resolve(BASE,`moss-encoding-layer3-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);check(process.argv[2]==='route','fixed layer3 route required');
 if(process.argv[3]==='validate'){if(value.phase.startsWith('upload-'))await verifyLayer3Payload(profile.staged);await appendFile(env.GITHUB_OUTPUT,Object.entries({...value,root,payload:profile.staged,native_pin:profile.pin}).map(([k,v])=>`${k}=${v??''}\n`).join(''));return;}
 check(process.argv[3]==='execute'&&value.phase==='execute-encoding-layer3','fixed layer3 execute required');await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});const download=resolve(env.RUNNER_TEMP,`moss-upstream-reference-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);await verifyLayer3Payload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await executeLayer3(root,resolve(root,'payload'),env);
}
export const SDPA_STAGED='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/sdpa-causal-contract-prep/payload';
export const SDPA_SHA='e6277e5b541095cf5637467d7f85d5899a068c39bf5a60b50a97b80907e5712b';
const SDPA_PHASES=['upload-sdpa-causal-contract','execute-sdpa-causal-contract'];
const SDPA_FILES=['capture_sdpa.py','frozen_runtime_bridge.py','sdpa-contract.json','source/capture-authentication.json','source/artifact-metadata.json','source/probe-authentication.json','source/probe-completion.json','source/native-trace.json','source/upstream-trace.json','source/comparison.json','input/q.f32le','input/k.f32le','input/v-projection.f32le','input/v.f32le','historical/hf-sdpa.f32le','historical/native-sdpa.f32le'];
const SDPA_CALLS=['cpu-f32-math-causal','mps-bf16-implicit-causal','mps-bf16-explicit-bool-causal'];
export function sdpaProfile(){return {staged:SDPA_STAGED,sha:SDPA_SHA,kind:'sc-26752-saved-sdpa-inputs/v1'};}
export function sdpaEnvelope(raw,ref,sha,backend){
 const value=JSON.parse(raw);check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&SDPA_PHASES.includes(value.phase),'closed saved SDPA envelope required');check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact saved SDPA source and mlx required');
 for(const key of ['artifact_id','artifact_run_id'])check(value.phase.startsWith('upload-')?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact saved SDPA artifact identity required');return value;
}
export async function verifySdpaPayload(root,profile=sdpaProfile()){
 await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(bytes.length<65536&&digest(bytes)===profile.sha,'saved SDPA manifest SHA mismatch');const manifest=JSON.parse(bytes);
 check(manifest.kind===profile.kind&&manifest.source?.runId==='40428788379'&&manifest.source.attempt==='1'&&manifest.source.toolingSha==='a6320337e843ea5234580bb0285f278470f63bfc'&&manifest.source.nativePin===BLOCK_TRACE_PIN&&manifest.source.artifactId==='12701291893'&&manifest.runtimeInventory?.path===resolve(PRIOR,'payload/runtime-files.json')&&manifest.runtimeInventory.sha256==='053fa7d3a714423a5ab82c9e7207768b4d3c53216ed9f175d22a907fde0e8cae','saved SDPA provenance changed');
 check(Array.isArray(manifest.files)&&manifest.files.map(x=>x.file).sort().join('\n')===SDPA_FILES.slice().sort().join('\n'),'exact saved SDPA payload closure required');const wanted=new Set(['payload-manifest.json']);
 for(const item of manifest.files){const path=safe(root,item.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.nlink===1&&Number.isSafeInteger(item.byteLength)&&item.byteLength<=4*1024*1024&&info.size===item.byteLength&&await fileHash(path)===item.sha256,'saved SDPA file mismatch');wanted.add(item.file);}
 async function walk(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){check(!entry.isSymbolicLink(),'saved SDPA payload link');const file=prefix+entry.name;if(entry.isDirectory())await walk(resolve(path,entry.name),file+'/');else check(wanted.has(file),'unlisted saved SDPA file');}}await walk(root);
 const contract=JSON.parse(await readFile(resolve(root,'sdpa-contract.json')));check(isDeepStrictEqual(contract.source,manifest.source)&&isDeepStrictEqual(contract.calls,SDPA_CALLS)&&contract.attentionCalls===3&&contract.encoderForwards===0&&contract.nativeBuilds===0&&contract.scale===0.0625&&contract.gqaFactor===4&&contract.hostUuid==='D299B954-4C63-5322-86A5-BD5667051634','fixed saved SDPA call/host contract changed');return {manifest,contract};
}
export function sdpaArtifact(artifact,source){check(String(artifact.id)===source.artifactId&&artifact.name===source.artifactName&&artifact.size_in_bytes===source.artifactBytes&&artifact.digest===`sha256:${source.artifactSha256}`&&artifact.expired===false&&String(artifact.workflow_run?.id)===source.runId&&artifact.workflow_run.head_sha===source.toolingSha,'saved SDPA original artifact changed');return artifact;}
export async function sdpaHost(execute=promisify(execFile)){
 const options={timeout:10000,maxBuffer:512*1024};let registry,graphics;try{registry=(await execute('/usr/sbin/ioreg',['-rd1','-c','IOPlatformExpertDevice'],options)).stdout;graphics=JSON.parse((await execute('/usr/sbin/system_profiler',['SPDisplaysDataType','-json'],options)).stdout);}catch{throw Error('saved SDPA bounded hardware metadata unavailable');}
 const uuid=/"IOPlatformUUID"\s*=\s*"([^"]+)"/.exec(registry)?.[1],gpu=graphics.SPDisplaysDataType?.find(x=>x._name==='Apple M5 Max'),metal=gpu?.spdisplays_mtlgpufamilysupport;
 check(uuid==='D299B954-4C63-5322-86A5-BD5667051634'&&metal==='spdisplays_metal4','saved SDPA requires its policy Mac UUID and Metal device');return {uuid,model:gpu._name,metal,originalArtifactUuid:'UNRECORDED; current policy guard is separate'};
}
export function sdpaCommands(root,payload){return ['-B',resolve(payload,'capture_sdpa.py'),'--payload',payload,'--manifest-sha',SDPA_SHA,'--output-dir',resolve(root,'upstream'),'--stat-node',process.execPath];}
export async function executeSdpa(root,payload,env=process.env){
 const producer=blockTraceRun(env),{manifest}=await verifySdpaPayload(payload);check(/^v24\./.test(process.version),'actual Node24 saved SDPA bridge required');const host=await sdpaHost(),artifact=sdpaArtifact(await actionsApi(env.MOSS_TRACE_TOKEN,'actions/artifacts/12701291893'),manifest.source);
 const bytes=await readFile(manifest.runtimeInventory.path);check(digest(bytes)===manifest.runtimeInventory.sha256,'saved SDPA runtime inventory changed');const runtime=JSON.parse(bytes);validateRuntime(runtime);check(await realpath(runtime.launcher)===runtime.executable,'saved SDPA frozen Python changed');const executable=runtime.files.find(x=>x.path===runtime.executable);check(executable&&await fileHash(executable.path)===executable.sha256,'saved SDPA runtime executable changed');
 const common={...env,MOSS_TRACE_TOKEN:'',PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'};
 check((await run('git',['rev-parse','HEAD'],{cwd:env.GITHUB_WORKSPACE,env:common,log:resolve(root,'tooling-source.log')})).trim()===producer.toolingSha,'saved SDPA actual tooling checkout changed');check(!(await run('git',['status','--porcelain'],{cwd:env.GITHUB_WORKSPACE,env:common,log:resolve(root,'tooling-clean.log')})).trim(),'saved SDPA tooling checkout must be clean');
 await writeFile(resolve(root,'probe-authentication.json'),JSON.stringify({kind:'sc-26752-saved-sdpa-controller-auth/v1',...producer,manifestSha256:SDPA_SHA,source:manifest.source,parentArtifact:artifact,host,scope:'3attention calls only; no native checkout/build, model weights or encoder'},null,2)+'\n',{flag:'wx'});
 const preflight="import importlib.util,pathlib,sys;p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('sealed_sdpa',p/'capture_sdpa.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.validate_inputs(p,m.module(p/'frozen_runtime_bridge.py').verify_payload(p,sys.argv[2]));m.producer(m.os.environ)";await run(runtime.launcher,['-B','-c',preflight,payload,SDPA_SHA],{cwd:payload,env:common,log:resolve(root,'input-preflight.log')});
 try{await run(runtime.launcher,sdpaCommands(root,payload),{cwd:payload,env:{...common,MOSS_SDPA_ADMITTED:'1'},log:resolve(root,'execution.log')});}finally{await verifySdpaPayload(payload);}
 const observation=JSON.parse(await readFile(resolve(root,'upstream/observation.json')));check(observation.actualToolingSha===producer.toolingSha&&observation.runId===producer.runId&&observation.attempt===producer.attempt&&observation.manifestSha256===SDPA_SHA&&observation.attentionCalls===3&&observation.encoderForwards===0&&observation.nativeBuilds===0&&observation.originalMpsRepeatByteExact===true&&observation.cpuFirstTokenExactlyV0===true&&observation.interpretation?.valid===true,'saved SDPA current capture/interpretation changed');
 const outputs={};for(const name of SDPA_CALLS){const path=resolve(root,'upstream',name+'.f32le'),record=observation.actualCalls[name];check(record?.status==='CAPTURED'&&record.output.file===name+'.f32le'&&(await lstat(path)).size===1163264&&await fileHash(path)===record.output.sha256,'saved SDPA actual output changed');outputs[name]=record.output.sha256;}
 await writeFile(resolve(root,'probe-completion.json'),JSON.stringify({kind:'sc-26752-saved-sdpa-completion/v1',...producer,manifestSha256:SDPA_SHA,source:manifest.source,observationSha256:await fileHash(resolve(root,'upstream/observation.json')),outputs,scope:'Supplemental contract diagnostic only; no quality/policy/reference replacement'},null,2)+'\n',{flag:'wx'});
}
async function sdpaMain(){
 const env=process.env;blockTraceRun(env);const value=sdpaEnvelope(env.MOSS_REFERENCE,env.SOURCE_REF,env.GITHUB_SHA,env.INPUT_BACKEND),root=resolve(BASE,`moss-sdpa-contract-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);check(process.argv[2]==='route','fixed saved SDPA route required');
 if(process.argv[3]==='validate'){if(value.phase.startsWith('upload-'))await verifySdpaPayload(SDPA_STAGED);await appendFile(env.GITHUB_OUTPUT,Object.entries({...value,root,payload:SDPA_STAGED}).map(([k,v])=>`${k}=${v??''}\n`).join(''));return;}
 check(process.argv[3]==='execute'&&value.phase==='execute-sdpa-causal-contract','fixed saved SDPA execute required');await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});const download=resolve(env.RUNNER_TEMP,`moss-upstream-reference-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`);await verifySdpaPayload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await executeSdpa(root,resolve(root,'payload'),env);
}
export function referenceFallback(phase){return ['upload-corrected-cases','execute-corrected-cases'].includes(phase)?'./moss-upstream-quality.mjs':'./moss-upstream-reference.mjs';}
async function encodingMain(){
 const phase=JSON.parse(process.env.MOSS_REFERENCE??'{}').phase;
 if(SDPA_PHASES.includes(phase))return sdpaMain();
 if(LAYER3_PHASES.includes(phase))return layer3Main();
 if(BLOCK_TRACE_UPSTREAM_PHASES.includes(phase))return blockTraceUpstreamMain();
 if(BLOCK_TRACE_PHASES.includes(phase))return blockTraceMain();
 if(!['upload-encoding','execute-encoding','upload-encoding-precision','execute-encoding-precision'].includes(phase)){
  check(process.argv[2]==='route'&&['validate','execute'].includes(process.argv[3]),'fixed reference route required');
  const code=await new Promise((done,fail)=>{const child=spawn(process.execPath,[fileURLToPath(new URL(referenceFallback(phase),import.meta.url)),process.argv[3]],{stdio:'inherit',env:process.env});child.on('error',fail);child.on('close',done);});check(code===0,'original reference route failed');return;
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
