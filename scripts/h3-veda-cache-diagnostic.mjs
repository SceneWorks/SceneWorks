// Closed metadata/CPU compiler diagnostic for one failed preparation; never API/generation.
import {execFileSync,spawnSync} from 'node:child_process';
import {readFile,writeFile,mkdir,lstat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {assertOwnedCpuRun,assertPrivateCpuRoot} from './h3-veda-cache-prepare.mjs';
import {bootstrap,noSymlinkAncestors} from './h3-veda-owned-python.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const FAILED_DIAGNOSTIC_ROOT='/Volumes/Models/epic-25140-cache-diagnostic-69fae2f10';
export const PRIOR={head:'299c99e8dbeb5ad37a9fa8ad3e7ba51e79f67328',runId:38534168074,jobId:115752914601,runDir:'/Volumes/Models/epic-25140-owned-cache-299c99e8d'};
const require=(v,m)=>{if(!v)throw Error('H3 diagnostic refusal: '+m);};
export function priorAuthority(run,job){
 require(run.id===PRIOR.runId&&run.head_sha===PRIOR.head&&run.run_attempt===1&&run.event==='workflow_dispatch'&&run.path==='.github/workflows/h3-veda-cache-prepare.yml'&&run.status==='completed'&&run.conclusion==='failure','exact completed prior failure required');
 require(job.id===PRIOR.jobId&&job.run_id===PRIOR.runId&&job.name==='mac2_cache'&&job.status==='completed'&&job.conclusion==='failure'&&job.runner_id===5281&&job.runner_name==='nax-macos-2','exact closed prior Mac2 job required');
 return {run,job};
}
export function diagnosticRoots(runDir,internal){
 assertPrivateCpuRoot(runDir,['/Volumes/Models/huggingface',PRIOR.runDir,FAILED_DIAGNOSTIC_ROOT]);
 require(path.isAbsolute(internal)&&!internal.startsWith('/Volumes/')&&internal!==ROOT&&!internal.startsWith(ROOT+path.sep),'actual internal RUNNER_TEMP root required');
 noSymlinkAncestors(internal);return {runDir,internal};
}
const hash=b=>createHash('sha256').update(b).digest('hex');
async function bound(p){noSymlinkAncestors(p);const s=await lstat(p);require(s.isFile(),'regular file required');return {path:p,bytes:s.size,sha256:hash(await readFile(p))};}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 (async()=>{
 const runDir=process.argv[2],head=execFileSync('git',['-C',ROOT,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
 require(head===process.env.GITHUB_SHA&&execFileSync('git',['-C',ROOT,'status','--porcelain'],{encoding:'utf8'}).trim()==='','clean current exact source required');
 const api=p=>JSON.parse(execFileSync('gh',['api',`repos/SceneWorks/SceneWorks/${p}`],{encoding:'utf8',timeout:15000}));
 const run=api(`actions/runs/${process.env.GITHUB_RUN_ID}`),jobs=api(`actions/runs/${run.id}/jobs?per_page=100`).jobs;
 const active=jobs.filter(j=>j.name==='mac2_cache'&&j.status==='in_progress');require(active.length===1,'unique current job required');
 const owner=assertOwnedCpuRun({run,job:active[0],head}),prior=priorAuthority(api(`actions/runs/${PRIOR.runId}`),api(`actions/jobs/${PRIOR.jobId}`));
 require(process.env.RUNNER_TEMP&&path.isAbsolute(process.env.RUNNER_TEMP),'actual runner temp missing');
 const internal=path.join(process.env.RUNNER_TEMP,`h3-cache-diagnostic-${owner.runId}-${owner.runAttempt}`);diagnosticRoots(runDir,internal);
 require(!await lstat(runDir).catch(e=>e.code==='ENOENT'?null:Promise.reject(e)),'fresh diagnostic root required');
 await mkdir(runDir,{recursive:true});const py=await bootstrap(runDir),helper=await bound(path.join(ROOT,'scripts/h3-veda-cache-diagnostic.py'));
 const plan={schemaVersion:1,kind:'epic25140_cache_failure_299_diagnostic',head,owner,prior,runDir,internal,helper,python:await bound(py.interpreter.path),fixtureDirectory:path.join(ROOT,'scripts/fixtures/h3-cache-failure-299'),supervisor:await bound(path.join(ROOT,'scripts/h3-veda-cpu-build.py'))};
 const bytes=Buffer.from(JSON.stringify(plan,null,2)+'\n'),planPath=path.join(runDir,'diagnostic-plan.json');await writeFile(planPath,bytes,{flag:'wx'});
 const child=spawnSync(plan.python.path,[helper.path,planPath,String(process.pid),hash(bytes)],{stdio:'inherit',timeout:1840000});require(child.status===0,'diagnostic refused/incomplete; inspect sealed receipt');
 })().catch(e=>{console.error(e.message);process.exitCode=1;});
}
