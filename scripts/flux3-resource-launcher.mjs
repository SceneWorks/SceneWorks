// One explicitly admitted adapter process, one stdin run request, NO outer reservation/retry.
// The native inner CLI verifies observed hardware and consumes immediately before generate.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { auditLedger, planHash, validateApprovedPlan } from './flux3-image-klein-terminal-budget.mjs';
import { flux3AnchorRequestDigest, resourceCaseDigest, verifyNativeResourceCapture } from './lib/flux3-resource-contract.mjs';
const loaderKeys = ['SCENEWORKS_FLUX3_KLEIN_ROOT','SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT','SCENEWORKS_FLUX3_KLEIN_REPOSITORY','SCENEWORKS_FLUX3_KLEIN_REVISION','SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT'];
async function validate(admission) {
  const { validateAdmission } = await import('./flux3-image-klein-terminal-session.mjs');
  const { verifyTerminalBuild } = await import('./flux3-image-klein-terminal-build.mjs');
  await validateAdmission(admission, admission.backend);
  await verifyTerminalBuild(admission, { role:'resource-adapter' });
}
async function guard(admission, pids) {
  const { guardHardware } = await import('./flux3-image-klein-terminal-session.mjs');
  return guardHardware(admission,pids);
}
function safeEnv(env) {
  return Object.fromEntries(Object.keys(env).filter(k => !k.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(k)
    && !['HF_ENDPOINT','HF_TOKEN_PATH','HF_XET_CACHE','TRANSFORMERS_CACHE','CARGO_TARGET_DIR'].includes(k)).map(k=>[k,env[k]]));
}
async function execute(binary, request, options, spawnChild = spawn) {
  const child = spawnChild(binary, [], { cwd:options.cwd, env:options.env, windowsHide:true, stdio:['pipe','pipe','pipe'] });
  assert(Number.isInteger(child.pid) && child.pid > 0, 'owned adapter PID missing');
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
  let stdout='', stderr='', failure;
  const stop = message => { failure ??= new Error(message); child.kill('SIGKILL'); }; // OWN child only.
  const timer = setTimeout(()=>stop('owned resource adapter deadline exceeded'), options.timeoutMs);
  const done = new Promise((resolve,reject)=>{
    child.once('error',reject);
    child.stdout.on('data',chunk=>{ if (stdout.length + chunk.length > 32*1024*1024) stop('adapter stdout limit exceeded'); else stdout+=chunk; });
    child.stderr.on('data',chunk=>{ if (stderr.length + chunk.length > 4*1024*1024) stop('adapter stderr limit exceeded'); else stderr+=chunk; });
    child.stdin.on('error',error=>{ failure ??= error; });
    child.once('close',(code,signal)=>resolve({code,signal}));
  });
  try {
    child.stdin.end(JSON.stringify(request)+'\n');
    const exit = await done;
    return { adapterPid:child.pid, stdout, stderr, ...exit, failure:failure?.message };
  } finally { clearTimeout(timer); }
}
export async function launchFlux3Resource({ admission, request, invocationId, resourceEnvironment = {} }, dependencies = {}) {
  // Overrides are explicit CPU fixtures ONLY; CLI input never supplies this object/module.
  const io = { validate, guard, execute, audit:auditLedger, readFile, writeFile, mkdir, now:Date.now, env:process.env, ...dependencies };
  assert.equal(admission.actualRun,true,'explicit actual resource admission required; inventory never executes');
  const deadline = admission.runDeadline ?? io.now() + admission.timeoutSeconds*1000;
  assert(deadline > io.now() && deadline <= io.now() + admission.timeoutSeconds*1000,'fixed resource run deadline required');
  const a = { ...admission, runDeadline:deadline };
  validateApprovedPlan(a.approvedPlan); assert.equal(planHash(a.approvedPlan),a.approvedPlanSha256);
  assert.equal(a.ledgerId,a.approvedPlan.ledgerId); assert.equal(a.ledgerRoot,a.approvedPlan.ledgerRoots[a.backend]);
  await io.validate(a); // fixed sealed resource-adapter producer role, source + MAIN ancestry.
  const invocation = a.approvedPlan.plan.generationInvocations.find(i=>i.invocationId===invocationId);
  assert(invocation && invocation.backend===a.backend && ['resource_anchor','resource_supplemental'].includes(invocation.kind),'one exact resource invocation required');
  assert.equal(request.action,'run','batch/default ladders are forbidden');
  for (const [key,expected] of Object.entries({backend:a.backend})) assert.equal(request.planned?.[key],expected);
  assert.equal(request.planned?.target?.provider,invocation.modelId); assert.equal(request.planned?.target?.tier,invocation.tier);
  const digest = invocation.kind==='resource_supplemental' ? resourceCaseDigest(request.resourceCase)
    : (assert.equal(request.resourceCase,undefined,'canonical anchor cannot carry supplemental inputs'),flux3AnchorRequestDigest(a.backend,invocation.modelId,invocation.tier));
  assert.equal(digest,invocation.requestSha256,'actual normalized request differs from approved permit');
  await io.audit({approval:a.approvedPlan,backend:a.backend,ledgerRoot:a.ledgerRoot}); // audit ONLY, no reserve/reset.
  const env = safeEnv(io.env);
  for (const key of Object.keys(resourceEnvironment)) { assert(loaderKeys.includes(key),'only fixed local loader environment permitted'); assert.equal(typeof resourceEnvironment[key],'string'); env[key]=resourceEnvironment[key]; }
  Object.assign(env,{HF_HOME:a.hfHome,HF_HUB_CACHE:path.join(a.hfHome,'hub'),HUGGINGFACE_HUB_CACHE:path.join(a.hfHome,'hub'),
    HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',SCENEWORKS_HUGGINGFACE_BASE_URL:'http://127.0.0.1:9',
    CUDA_VISIBLE_DEVICES:a.backend==='candle'?'1':'-1',CUDA_DEVICE_ORDER:'PCI_BUS_ID',SCENEWORKS_FLUX3_BUDGET_NODE:process.execPath});
  await io.mkdir(a.outputRoot,{recursive:true});
  const stem=path.join(a.outputRoot,`resource-${invocationId}-${randomBytes(8).toString('hex')}`);
  const approvalFile=`${stem}.approval.json`, admissionFile=`${stem}.admission.json`;
  await io.writeFile(approvalFile,JSON.stringify(a.approvedPlan)+'\n',{flag:'wx'});
  await io.writeFile(admissionFile,JSON.stringify(a)+'\n',{flag:'wx'});
  env.SCENEWORKS_FLUX3_INVOCATION=JSON.stringify({approvalFile,admissionFile,sourceSha:a.sourceSha,binarySha256:a.binarySha256,
    ledgerRoot:a.ledgerRoot,ledgerId:a.ledgerId,approvedPlanSha256:a.approvedPlanSha256,invocationId,backend:a.backend,
    caseId:invocation.caseId,kind:invocation.kind,requestSha256:digest});
  const actualRequest={...request,resourceEnvelope:invocation.kind==='resource_anchor'};
  await io.guard(a,[]); // Guard the shared device BEFORE any adapter spawn/weight load.
  assert(deadline>io.now(),'fixed resource deadline expired before owned spawn');
  const output=await io.execute(a.binary,actualRequest,{cwd:path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),env,timeoutMs:deadline-io.now()});
  await io.writeFile(`${stem}.stdout.json`,output.stdout,{flag:'wx'}); await io.writeFile(`${stem}.stderr.log`,output.stderr,{flag:'wx'});
  assert(!output.failure && output.code===0 && !output.signal,`owned resource adapter failed; permit may be spent, never retry: ${output.failure ?? output.code}`);
  const envelope=JSON.parse(output.stdout);
  const ledger=await io.audit({approval:a.approvedPlan,backend:a.backend,ledgerRoot:a.ledgerRoot});
  const ledgerEvent=ledger.events.find(e=>e.invocationId===invocationId); assert(ledgerEvent,'actual inner reservation absent');
  const verified=verifyNativeResourceCapture(envelope,{admission:a,invocation,adapterPid:output.adapterPid,ledgerEvent});
  const resultFile=`${stem}.${verified.anchorFragment?'anchor':'supplemental'}.json`;
  await io.writeFile(resultFile,JSON.stringify(verified.anchorFragment ?? verified.supplementalObservation)+'\n',{flag:'wx'});
  return {captureId:envelope.captureId,invocationId,adapterPid:output.adapterPid,resultFile,rawCaptureFile:`${stem}.stdout.json`,
    ledgerId:a.ledgerId,approvedPlanSha256:a.approvedPlanSha256,generationCount:ledger.state.generationCount};
}
if (process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  try { assert.equal(process.argv.length,3,'one explicit resource launch JSON file required');
    console.log(JSON.stringify(await launchFlux3Resource(JSON.parse(await readFile(process.argv[2],'utf8')))));
  } catch(error) { console.error(`BLOCKED: ${error.message}`); process.exitCode=1; }
}
