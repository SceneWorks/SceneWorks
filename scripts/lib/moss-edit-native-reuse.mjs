// Ordinary CUDA execution with unchanged native producers and a completed Mac handoff.
// No request/attempt recovery: the new CUDA root executes the complete allocated backend.
import {resolve,basename,dirname} from 'node:path';
import {readFile} from 'node:fs/promises';
import {VERSION,canonical,digest,json,requireThat,safePath,fileIdentity,treeIdentity,validateBinding,validateArtifactInventory} from './moss-edit-terminal.mjs';
import {allowedPrivateParent} from './moss-private-hf.mjs';
import {filesUnder} from './moss-edit-terminal-actions.mjs';

export function reuseProducers(env=process.env,{verify=false}={}) {
  const read=key=>{const value=env[key];requireThat(/^[1-9][0-9]*$/.test(value??''),`native reuse requires ${key}`);return value;};
  return {native:{runId:read('MOSS_NATIVE_RUN_ID'),attempt:read('MOSS_NATIVE_RUN_ATTEMPT')},mlx:{runId:read('MOSS_MLX_RUN_ID'),attempt:read('MOSS_MLX_RUN_ATTEMPT'),artifactId:read('MOSS_MLX_ARTIFACT_ID')},...(verify?{cuda:{runId:read('MOSS_EXECUTION_RUN_ID'),attempt:read('MOSS_EXECUTION_RUN_ATTEMPT')}}:{stopped:{runId:read('MOSS_CUDA_HF_RUN_ID'),attempt:read('MOSS_CUDA_HF_RUN_ATTEMPT'),artifactId:read('MOSS_CUDA_HF_ARTIFACT_ID')}})};
}
export async function executionTooling(root,head) {
  const source=treeIdentity(root);
  requireThat(/^[a-f0-9]{40}$/.test(head)&&source.head===head,'native reuse tooling differs from its clean exact checkout');
  const files=Object.fromEntries(Object.entries({harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs',qualityAmendment:'lib/moss-edit-quality-amendment.mjs',nativeReuse:'lib/moss-edit-native-reuse.mjs',controlledQuality:'lib/moss-controlled-cuda-quality.mjs'}).map(([name,path])=>[name,resolve(root,'scripts',path)]));
  return {sourceRoot:resolve(root),head,source,files,identities:Object.fromEntries(await Promise.all(Object.entries(files).map(async([name,path])=>[name,await fileIdentity(path)])))};
}
export async function validateOrdinaryMlx(prior,config,producer) {
  const root=resolve(prior,'evidence'),lock=await json(resolve(root,'protocol.lock.json')),plan=await json(resolve(root,'cases.json')),binding=validateBinding(await json(resolve(root,'bindings/mlx.json')),lock,'mlx');
  const marker=await json(resolve(prior,'execution.json')),handoff=await json(resolve(prior,'handoff/handoff.json'));
  requireThat(!lock.scope&&lock.candidateId===config.candidateId&&lock.sources.sceneworks.head===config.sourceSha&&lock.sources.inference.head===config.inferenceSha&&marker.version===VERSION&&marker.backend==='mlx'&&marker.execution==='COMPLETE'&&marker.lockSha256===digest(lock)&&marker.renders===25&&marker.refusals===29&&digest(plan)===lock.planSha256,'ordinary Mac completion or frozen source/plan changed');
  requireThat(handoff.version===VERSION&&handoff.projectId===binding.endpoint.projectId&&canonical(handoff.sources)===canonical(lock.sources)&&canonical(await filesUnder(resolve(prior,'handoff/data')))===canonical(handoff.files),'ordinary Mac handoff bytes changed');
  const capture=await json(resolve(root,'external/ordinary-execution-mlx.json'));
  requireThat(capture.runId===producer.runId&&capture.attempt===producer.attempt&&capture.backend==='mlx'&&capture.candidateId===lock.candidateId&&capture.lockSha256===digest(lock)&&capture.bindingSha256===digest(binding)&&canonical(capture.sources)===canonical(lock.sources)&&!capture.toolingSha&&capture.artifactName===`moss-execution-mlx-${producer.runId}-${producer.attempt}`,'ordinary Mac execution producer changed');
  const rows=plan.rows.filter(row=>row.backend==='mlx');requireThat(rows.length===25&&plan.requiredGuardCases.length===29,'ordinary Mac allocation changed');
  for(const row of rows) {
    const dir=resolve(root,'rows',row.id),receipt=await json(resolve(dir,'receipt.json')),request=await json(resolve(dir,'request.json')),job=await json(resolve(dir,'terminal-job.json'));
    requireThat(receipt.execution==='VERIFIED'&&receipt.rowId===row.id&&receipt.backend==='mlx'&&receipt.lockSha256===digest(lock)&&receipt.bindingSha256===digest(binding)&&receipt.requestSha256===digest(request)&&receipt.jobSha256===digest(job)&&receipt.jobId===job.id&&job.status===row.expected,'ordinary Mac row changed');
    for(const output of receipt.outputs)requireThat((await fileIdentity(safePath(dir,output.file))).sha256===output.sha256,'ordinary Mac output changed');
  }
  for(const name of plan.requiredGuardCases) {
    const dir=resolve(root,'guards',`mlx-${name}`),receipt=await json(resolve(dir,'receipt.json')),proof=await json(resolve(dir,'no-provider-load.json'));
    requireThat(receipt.backend==='mlx'&&receipt.case===name&&receipt.lockSha256===digest(lock)&&receipt.bindingSha256===digest(binding)&&proof.requestSha256===receipt.requestSha256&&proof.lockSha256===digest(lock)&&proof.backend==='mlx'&&proof.case===name&&proof.method==='source-bound-moss-boundaries/v1'&&proof.observation,'ordinary Mac guard changed');
  }
  return {lockSha256:digest(lock),bindingSha256:digest(binding),completion:await fileIdentity(resolve(prior,'execution.json')),handoff:await fileIdentity(resolve(prior,'handoff/handoff.json')),producer};
}
export async function readOnlyStoppedPackageSource(config,priorRoot,producers) {
  const prior=await json(resolve(priorRoot,'runtime.json')),transfer=await json(resolve(priorRoot,'hf-transfer.json'));
  const expectedRoot=`moss-execute-${producers.stopped.runId}-${producers.stopped.attempt}-cuda`,nativeRoot=`moss-build-${producers.native.runId}-${producers.native.attempt}-cuda`;
  requireThat(prior.backend==='cuda'&&prior.candidateId===config.candidateId&&prior.sourceSha===config.sourceSha&&prior.inferenceSha===config.inferenceSha&&basename(prior.root)===expectedRoot&&prior.hfHome===resolve(prior.root,'hf')&&allowedPrivateParent(dirname(prior.root),prior.sourceRoot),'stopped package source differs from exact owned producer');
  const inventory=validateArtifactInventory(await json(resolve(config.buildRoot,'cuda/inventory.json')),'cuda',{sceneworks:treeIdentity(config.sourceRoot),inference:treeIdentity(config.inferenceRoot)});
  requireThat(transfer.operation==='same-volume-private-HF-ownership-transfer'&&transfer.to===prior.hfHome&&basename(dirname(transfer.from))===nativeRoot&&transfer.from===resolve(dirname(transfer.from),'hf')&&transfer.candidateId===config.candidateId&&transfer.sourceSha===config.sourceSha&&transfer.inferenceSha===config.inferenceSha&&canonical(transfer.originalPackages)===canonical(inventory.files.packages)&&canonical(transfer.before)===canonical(transfer.after)&&canonical(transfer.after.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256}))),'stopped package transfer changed');
  for(const file of ['runtime.json','hf-transfer.json'])requireThat((await readFile(resolve(priorRoot,file))).equals(await readFile(resolve(prior.root,file))),'live stopped package owner differs from archived producer');
  requireThat(canonical(await fileIdentity(resolve(dirname(transfer.from),'hf-owner.json')))===canonical(transfer.originalOwner),'original native package owner changed');
  // isolatePackages checks each file hash while copying to a fresh private tree.
  return prior.hfHome;
}
