// Owned CPU preparation through normal SceneWorks managed download jobs. Never generation.
import {execFileSync,spawnSync} from 'node:child_process';
import {readFile,writeFile,mkdir,lstat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sorted=v=>Array.isArray(v)?v.map(sorted):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,sorted(v[k])])):v;
const digest=v=>createHash('sha256').update(JSON.stringify(sorted(v))).digest('hex');
async function hashFile(p){const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');}
import {stripJsoncComments} from './lib/jsonc.mjs';
import {bootstrap,noSymlinkAncestors} from './h3-veda-owned-python.mjs';


const absolute=p=>typeof p==='string'&&path.isAbsolute(p),inside=(r,p)=>{const x=path.relative(r,p);return x&&!x.startsWith('..')&&!path.isAbsolute(x);},MLX_SYS_SOURCE='https://github.com/michaeltrefry/mlx-rs';
export function buildArguments(backend) {
  check(["mlx", "candle_cuda"].includes(backend), "explicit backend required");
  return ["build", "--release", "--locked", "--message-format=json-render-diagnostics", "-p", "sceneworks-rust-api", ...(backend === "candle_cuda" ? ["--features", "backend-candle"] : [])];
}
export function mlxBuildEnvironment(base) {
  const env = { ...base };
  delete env.PMETAL_METALLIB_PATH;
  delete env.PMETAL_MLX_PREBUILT_DIR;
  return env;
}
export function pinnedMlxSysPackageId(cargoLock) {
  const blocks = cargoLock.split(/(?=^\[\[package\]\]$)/m).filter((block) => /^name = "pmetal-mlx-sys"$/m.test(block));
  check(blocks.length === 1, "Cargo.lock must contain exactly one pmetal-mlx-sys package");
  const version = blocks[0].match(/^version = "([^"]+)"$/m)?.[1];
  const source = blocks[0].match(/^source = "git\+https:\/\/github\.com\/michaeltrefry\/mlx-rs\?rev=([a-f0-9]{40})#\1"$/m)?.[1];
  check(version && source, "Cargo.lock MLX sys source/version is not exact and pinned");
  return { revision: source, packageId: `git+${MLX_SYS_SOURCE}?rev=${source}#pmetal-mlx-sys@${version}` };
}
export function mlxMetallibBuildOutput(stdout, expectedPackageId, targetDir) {
  const messages = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (!line) continue;
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    if (message.reason === "build-script-executed" && message.package_id === expectedPackageId) messages.push(message);
  }
  check(messages.length === 1, `expected one exact pinned pmetal-mlx-sys build-script output, found ${messages.length}`);
  const outDir = messages[0].out_dir;
  check(absolute(outDir) && inside(path.resolve(targetDir), path.resolve(outDir)), "pinned MLX build-script OUT_DIR must be inside the durable Cargo target");
  return { packageId: expectedPackageId, outDir: path.resolve(outDir), path: path.join(path.resolve(outDir), "build", "lib", "mlx.metallib") };
}
function declaredCacheRoots(t,b){return [...new Set(t.cacheRootCandidates[b].map(v=>v.replace(/\$\{([A-Z_]+)\}/g,(_,n)=>process.env[n]??'UNBOUND')).filter(v=>!v.includes('UNBOUND')).map(v=>path.resolve(v)))];}

const check=(ok,m)=>{if(!ok)throw new Error(`H3 CPU cache: ${m}`);};
export const DOWNLOAD_PIN='75e03b5abc504d06688b75c9f7bc7a23f0873f5b';
export const ORDINARY_CATALOG_SHA='07e3079ecd9af63221743fbc6ec7125ffd2ea7f50303f83cd257e49265ab4989';
export const DOWNLOAD_SOURCE='16567882f0e07b5d6d95941adc42210d41dff7ab';
export const MISSING_SHA=new Set(['2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a','b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c','8d49fe5abad6195f321fc7d741cc024e006ff891a40c67844834866bcdcf9e59','0fac46be5b3498124f1bb596e9f9050338ae8028425650d5524ed40fce50f2f8']);
export const FETCH_TUPLES=[
 ['Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview','9a1fd3a41b4a754a7886e64e82edbddf599fd1bd','minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors',275415648,'2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a'],
 ['MiniMaxAI/MiniMax-H3','939557dc319dd91227e30195a763f272ba7f8765','text_encoder/model-00013-of-00014.safetensors',4875990584,'b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c'],
 ['prithivMLmods/MiniMax-H3-Rough-2D-Cartoon-Illustration','fc40d010a2b44fd4caa4b750b6f503eca5d61577','minimax-h3-rough-2d-cartoon-illustration-1200.safetensors',74570640,'8d49fe5abad6195f321fc7d741cc024e006ff891a40c67844834866bcdcf9e59'],
 ['akhaliq/MiniMax-H3-Claymation-Style-LoRA','991e08146a349e00b1837e77f81f415586b19063','checkpoints/full_claymation_h3_v1_claymation_h3_v1.safetensors',155109840,'0fac46be5b3498124f1bb596e9f9050338ae8028425650d5524ed40fce50f2f8']
];
const json=async p=>JSON.parse(await readFile(p,'utf8'));
const git=(root,...a)=>execFileSync('git',['-C',root,...a],{encoding:'utf8',timeout:15000,maxBuffer:8*1024*1024}).trim();
const bound=async p=>({path:p,bytes:(await lstat(p)).size,sha256:await hashFile(p)});
export function cacheEntries(t){
 const rows=[...t.artifacts.filter(a=>a.coverage.some(c=>c.backend==='mlx')),...Object.values(t.turboArtifacts),...t.styles.map(s=>({...s,...s.source}))];
 check(rows.length===194&&new Set(rows.map(a=>`${a.repo}@${a.revision}/${a.file}`)).size===194,'complete source-owned194 file set required');
 for(const a of rows){check(/^[\w.-]+\/[\w.-]+$/.test(a.repo)&&/^[a-f0-9]{40}$/.test(a.revision)&&/^[a-f0-9]{64}$/.test(a.sha256)&&a.bytes>0&&!path.isAbsolute(a.file)&&!a.file.split(/[\\/]/).includes('..'),'unbound/unsafe frozen artifact');if(MISSING_SHA.has(a.sha256))check(FETCH_TUPLES.some(t=>digest(t)===digest([a.repo,a.revision,a.file,a.bytes,a.sha256])),'approved download tuple differs despite valid SHA');}
 return rows.map(a=>({repo:a.repo,revision:a.revision,file:a.file,bytes:a.bytes,sha256:a.sha256,gitBlob:a.gitBlob??null,downloadType:t.styles.some(s=>s.source.repo===a.repo)?"lora_download":"model_download",loraId:t.styles.find(s=>s.source.repo===a.repo)?.id??null,allowManagedFetch:MISSING_SHA.has(a.sha256)}));
}
export function sameH3Catalog(a,b){for(const id of ['minimax_h3','minimax_h3_ref'])check(digest(a.models.find(m=>m.id===id))===digest(b.models.find(m=>m.id===id)),'ordinary download bootstrap H3 catalog differs');}
export function assertOwnedCpuRun({run,job,env=process.env,head}){
 check(env.GITHUB_ACTIONS==='true'&&env.RUNNER_NAME==='nax-macos-2'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&run.id===Number(env.GITHUB_RUN_ID)&&run.run_attempt===Number(env.GITHUB_RUN_ATTEMPT)&&run.event==='workflow_dispatch'&&run.head_sha===head&&run.status==='in_progress','exact live CPU Actions run/source required');
 check(job.name==='mac2_cache'&&job.runner_name==='nax-macos-2'&&job.runner_id===5281&&job.status==='in_progress','exact Mac2 CPU job/runner required');
 return {runId:run.id,runAttempt:run.run_attempt,jobId:job.id,runnerId:job.runner_id,runnerName:job.runner_name,sourceHead:head,resourceQueueVerified:false,physicalAdmission:false};
}
export function assertPrivateCpuRoot(runDir,cacheRoots){
 noSymlinkAncestors(runDir);check(path.isAbsolute(runDir)&&runDir!==ROOT&&!runDir.startsWith(ROOT+path.sep),'owned external CPU root required');
 for(const cache of cacheRoots)check(runDir!==cache&&!runDir.startsWith(cache+path.sep)&&!cache.startsWith(runDir+path.sep),'owned CPU tools/build root overlaps declared shared cache');
}
async function privateCpuRoot(runDir){const t=await json(path.join(ROOT,'config/terminal-evidence/epic-25140-host-template.json'));assertPrivateCpuRoot(runDir,declaredCacheRoots(t,'mlx'));}
export async function supervisedCpuBuild(mode,runDir,checkout,py){
 check(['toolchain','cargo'].includes(mode),'closed CPU preparation mode required');
 const helper=await bound(path.join(ROOT,'scripts/h3-veda-cpu-build.py')),python=await bound(py.interpreter.path);
 check(python.sha256===py.interpreter.sha256,'selected Python byte authority changed');
 const sourceBindings=await Promise.all(['Cargo.toml','Cargo.lock','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml','crates/sceneworks-worker/src/lib.rs'].map(f=>bound(path.join(checkout,f))));
 const plan={schemaVersion:1,kind:'epic25140_closed_CPU_build_plan',mode,runDir,sourceRoot:checkout,helper,python,sourceBindings};
 const planPath=path.join(runDir,`cpu-build-${mode}-plan.json`);await writeFile(planPath,JSON.stringify(plan,null,2)+'\n',{flag:'wx'});
 const child=spawnSync(python.path,[helper.path,planPath,String(process.pid),await hashFile(planPath)],{stdio:'inherit',timeout:mode==='cargo'?3640000:640000});
 check(child.status===0,'owned CPU preparation refused; inspect durable closure receipt');
 const receiptPath=path.join(runDir,'cpu-build',mode,'receipt.json'),receipt=await json(receiptPath),limit=mode==='cargo'?3600:600;
 check(receipt.prepared===true&&receipt.rc===0&&receipt.maxSeconds===limit&&receipt.closure?.reaped===true&&receipt.closure?.descendantsDrained===true&&receipt.helper.sha256===helper.sha256&&receipt.python.sha256===python.sha256,'actual CPU child closure/source/tool lifetime incomplete');
 for(const b of [receipt.stdout,receipt.stderr,...Object.values(receipt.tools)])check((await bound(b.path)).sha256===b.sha256,'actual CPU log/tool bytes changed');
 return {...receipt,receipt:await bound(receiptPath)};
}
export async function buildCacheProduct(runDir,{run=execFileSync,python}={}){
 await privateCpuRoot(runDir);await mkdir(runDir,{recursive:true});
 const checkout=path.join(runDir,'download-source');check(!await lstat(checkout).catch(e=>e.code==='ENOENT'?null:Promise.reject(e)),'source build exists; reuse only its completed attestation');
 await mkdir(checkout);const call=(...a)=>run('git',['-C',checkout,...a],{encoding:'utf8',timeout:120000});call('init');call('-c','credential.helper=','-c','credential.helper=!gh auth git-credential','fetch','--depth=1','https://github.com/SceneWorks/SceneWorks.git',DOWNLOAD_SOURCE);call('checkout','--detach',DOWNLOAD_SOURCE);
 check(await hashFile(path.join(checkout,'crates/sceneworks-worker/src/lib.rs'))==='e2cd68fce583d5e39e2051ea2cf6cc33919dd4b538726849d38c20ada459f353','ordinary source worker bytes differ before CPU-only overlay');
 const patch=path.join(ROOT,'config/terminal-evidence/epic-25140-cpu-worker-startup.patch');call('apply','--check',patch);call('apply',patch);
 check(git(checkout,'diff','--name-only')==='crates/sceneworks-worker/src/lib.rs','CPU-only setter overlay changed unrelated source');
 const old=JSON.parse(stripJsoncComments(await readFile(path.join(checkout,'config/manifests/builtin.models.jsonc'),'utf8'))),now=JSON.parse(stripJsoncComments(await readFile(path.join(ROOT,'config/terminal-evidence/epic-25140-cache-catalog.json'),'utf8')));check(await hashFile(path.join(checkout,'config/manifests/builtin.models.jsonc'))===ORDINARY_CATALOG_SHA,'ordinary source catalog bytes changed');sameH3Catalog(old,now);
 const pins=[...['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].flatMap(f=>[...git(checkout,'show','HEAD:'+f).matchAll(/git\s*=\s*"https:\/\/github.com\/SceneWorks\/inference"\s*,\s*rev\s*=\s*"([a-f0-9]{40})"/g)].map(m=>m[1]))];check(pins.length>0&&pins.every(p=>p===DOWNLOAD_PIN),'all ordinary source direct native pins must equal original75');const pin=DOWNLOAD_PIN;const lockPins=[...(await readFile(path.join(checkout,'Cargo.lock'),'utf8')).matchAll(/source = "git\+https:\/\/github.com\/SceneWorks\/inference\?rev=([a-f0-9]{40})#([a-f0-9]{40})"/g)];check(lockPins.length>0&&lockPins.every(m=>m[1]===pin&&m[2]===pin),'ordinary Cargo.lock native source authority differs');
 const py=python??await bootstrap(runDir),toolchainPreparation=await supervisedCpuBuild('toolchain',runDir,checkout,py),cargoPreparation=await supervisedCpuBuild('cargo',runDir,checkout,py);
 const target=path.join(runDir,'download-cargo-target'),args=buildArguments('mlx');check(cargoPreparation.stdout.bytes<=64*1024*1024,'CPU build log exceeds bounded parser');const output=await readFile(cargoPreparation.stdout.path,'utf8');
 check(git(checkout,'status','--porcelain','--untracked-files=all')==='M crates/sceneworks-worker/src/lib.rs','ordinary CPU source changed during preparation');for(const b of cargoPreparation.sourceBindings)check(await hashFile(b.path)===b.sha256,'actual CPU build source bytes changed');
 const binary=await bound(path.join(target,'release/sceneworks-rust-api')),sys=pinnedMlxSysPackageId(await readFile(path.join(checkout,'Cargo.lock'),'utf8')),resource=mlxMetallibBuildOutput(output,sys.packageId,target),metallib=await bound(resource.path);
 const value={schemaVersion:1,kind:'epic25140_cpu_download_source_build',sourceHead:DOWNLOAD_SOURCE,inferencePin:pin,overlay:await bound(patch),bootstrapCatalog:await bound(path.join(ROOT,'config/terminal-evidence/epic-25140-cache-catalog.json')),ordinaryLoras:await bound(path.join(checkout,'config/manifests/builtin.loras.jsonc')),ordinaryCatalog:await bound(path.join(checkout,'config/manifests/builtin.models.jsonc')),sourceWorker:await bound(path.join(checkout,'crates/sceneworks-worker/src/lib.rs')),cargoLock:await bound(path.join(checkout,'Cargo.lock')),binary,metallib,buildArgs:args,cpuPreparation:{toolchain:toolchainPreparation,cargo:cargoPreparation},rustc:cargoPreparation.tools.rustc,cargo:cargoPreparation.tools.cargo,runnerName:process.env.RUNNER_NAME,sourceRoot:checkout,cpuSourcePathVerified:true,actualStartupGpuContextVerified:false,actualManagedDownload:false,physicalAdmission:false,campaignReady:false};
 const outputPath=path.join(runDir,'cpu-download-build.json');await writeFile(outputPath,JSON.stringify(value,null,2)+'\n',{flag:'wx'});return value;
}
export async function planCache({runDir,sharedCache,attestationPath,owner,maxSeconds=480}){
 check(process.platform==='darwin'&&process.arch==='arm64'&&path.isAbsolute(runDir)&&path.isAbsolute(sharedCache)&&runDir!==sharedCache&&!runDir.startsWith(sharedCache+path.sep)&&!sharedCache.startsWith(runDir+path.sep)&&!runDir.startsWith(ROOT+path.sep),'owned same-volume Mac2 paths required');noSymlinkAncestors(runDir);noSymlinkAncestors(sharedCache);
 const head=git(ROOT,'rev-parse','HEAD');check(git(ROOT,'status','--porcelain')==='','exact clean source required');
 const t=await json(path.join(ROOT,'config/terminal-evidence/epic-25140-host-template.json')),entries=cacheEntries(t),a=await json(attestationPath);check(declaredCacheRoots(t,'mlx').includes(sharedCache),'shared cache is outside source-declared narrow roots');
 check(a.kind==='epic25140_cpu_download_source_build'&&a.sourceHead===DOWNLOAD_SOURCE&&a.cpuSourcePathVerified===true&&a.inferencePin===DOWNLOAD_PIN&&a.runnerName==='nax-macos-2'&&await hashFile(a.binary.path)===a.binary.sha256&&await hashFile(a.metallib.path)===a.metallib.sha256,'actual ordinary source-built download binary/resources required');
 for(const [mode,limit] of [['toolchain',600],['cargo',3600]]){const r=a.cpuPreparation?.[mode];check(r?.prepared===true&&r.rc===0&&r.maxSeconds===limit&&r.closure?.reaped===true&&r.closure?.descendantsDrained===true&&await hashFile(r.receipt.path)===r.receipt.sha256,'completed owned CPU preparation lifecycle required');check(Array.isArray(r.sourceBindings)&&r.sourceBindings.length===5,'CPU source binding census incomplete');for(const b of [r.helper,r.python,r.stdout,r.stderr,...r.sourceBindings,...Object.values(r.tools)])check(await hashFile(b.path)===b.sha256,'CPU preparation source/runtime/log/tool receipt changed');}
 check(a.overlay.sha256===await hashFile(path.join(ROOT,'config/terminal-evidence/epic-25140-cpu-worker-startup.patch'))&&await hashFile(a.sourceWorker.path)===a.sourceWorker.sha256,'actual CPU worker overlay/source changed');
 const catalogPath=path.join(ROOT,'config/terminal-evidence/epic-25140-cache-catalog.json');check(a.bootstrapCatalog.sha256===await hashFile(catalogPath)&&a.ordinaryCatalog.sha256===ORDINARY_CATALOG_SHA&&await hashFile(a.ordinaryCatalog.path)===ORDINARY_CATALOG_SHA,'source/build ordinary catalog identity changed');sameH3Catalog(JSON.parse(stripJsoncComments(await readFile(a.ordinaryCatalog.path,'utf8'))),await json(catalogPath));
 check(Number.isInteger(maxSeconds)&&maxSeconds>0&&maxSeconds<=480,'bounded CPU preparation deadline required');
 const p={schemaVersion:1,kind:'epic25140_owned_managed_cache_plan',backend:'mlx',sourceHead:head,template:await bound(path.join(ROOT,'config/terminal-evidence/epic-25140-host-template.json')),catalog:await bound(catalogPath),loraCatalog:a.ordinaryLoras,entries,styles:t.styles,owner,sharedCache,runDir,cacheRoot:path.join(runDir,'huggingface'),binary:a.binary,metallib:a.metallib,binaryAttestation:await bound(attestationPath),maxSeconds,physicalAdmission:false,campaignReady:false,sourceModules:{controller:await bound(path.join(ROOT,'scripts/h3-veda-cache-prepare.py')),guardian:await bound(path.join(ROOT,'scripts/h3-veda-guardian.py'))}};
 await mkdir(runDir,{recursive:true});const target=path.join(runDir,`cache-plan-${owner.runId}-${owner.runAttempt}.json`);await writeFile(target,JSON.stringify(p,null,2)+'\n',{flag:'wx'});return target;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 (async()=>{const [mode,runDir,sharedCache,attestationPath]=process.argv.slice(2);check(['build','prepare'].includes(mode),'build RUN_DIR | prepare RUN_DIR SHARED_CACHE ATTESTATION');const head=git(ROOT,'rev-parse','HEAD');check(head===process.env.GITHUB_SHA,'workflow exact checkout required');const api=r=>JSON.parse(execFileSync('gh',['api',`repos/SceneWorks/SceneWorks/${r}`],{encoding:'utf8',timeout:15000}));const run=api(`actions/runs/${process.env.GITHUB_RUN_ID}`),jobs=api(`actions/runs/${run.id}/jobs?per_page=100`),job=jobs.jobs?.filter(j=>j.name==='mac2_cache'&&j.status==='in_progress');check(job?.length===1,'owned live cache job absent/ambiguous');const owner=assertOwnedCpuRun({run,job:job[0],head});await privateCpuRoot(runDir);const py=await bootstrap(runDir);if(mode==='build'){console.log(JSON.stringify(await buildCacheProduct(runDir,{python:py})));return;}const plan=await planCache({runDir,sharedCache,attestationPath,owner});const child=spawnSync(py.interpreter.path,[path.join(ROOT,'scripts/h3-veda-cache-prepare.py'),plan,String(process.pid),await hashFile(plan)],{stdio:'inherit',env:{...process.env,S9_CACHE_PYTHON_SHA256:py.interpreter.sha256},timeout:550000});check(child.status===0,'CPU cache controller refused/incomplete closure');})().catch(e=>{console.error(e.message);process.exitCode=1;});
}
