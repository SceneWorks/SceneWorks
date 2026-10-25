import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {envelope,PRIOR_PAYLOAD,CASES_STAGED,verifyCasePayload,authenticateModels} from './moss-upstream-quality.mjs';

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
test('sealed case data binds both actual Base/SFT model views without rehashing retained weights',async()=>{
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
