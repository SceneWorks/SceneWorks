// Self-verifying assertion mutations for the terminal harness repair, never model evidence.
// Every child runs only a focused CPU/in-process fake regression in an owned temporary tree.
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=new URL('..',import.meta.url);
const edit=(source,needle,replacement)=>{assert.equal(source.split(needle).length,2,`unique mutation anchor required: ${needle}`);return source.replace(needle,replacement);};
const C='h3-finetune-terminal-cases.mjs',R='h3-finetune-terminal-runtime.mjs',A='h3-finetune-terminal.mjs';
const casesTest='h3-finetune-terminal-cases.test.mjs',runtimeTest='h3-finetune-terminal-runtime.test.mjs',avTest='h3-finetune-terminal-av.test.mjs';
const mutations=[
  ['linked DTO enum',C,casesTest,'review repair linked DTO',s=>edit(s,'Object.assign(importBody, { linkedRootId:',"Object.assign(importBody, { ownershipMode:'linked',linkedRootId:")],
  ['linked runtime enum',R,runtimeTest,'review repair linked runtime',s=>{assert.ok(s.includes('body.ownershipMode===undefined'));return s.replaceAll('body.ownershipMode===undefined','true');}],
  ['actual LTX LoRA route',C,casesTest,'review repair LTX baseline',s=>edit(s,"id:`${prefix}_adapter_import`,method:'POST',path:'/api/v1/loras/import'","id:`${prefix}_adapter_import`,method:'POST',path:'/api/v1/models/import'")],
  ['portable value mapping',C,casesTest,'review repair portable mapping',s=>edit(s,"['sceneworks-core',null,'minimax_h3_codec::tests::actual_native_bytes_normalize_through_persisted_plan_and_shared_cache_in_all_precisions']","['sceneworks-worker',null,'video_jobs::minimax_h3_import_tests::native_and_sharded_plans_reach_the_real_h3_provider_load_seam']")],
  ['every successful closure',R,runtimeTest,'review repair all successful generalization',s=>edit(s,"assert.equal(matched.length,imported.length,'EVERY successful imported generation/replay needs its actual native load closure');",'/* unit_fake mutation: missing closure guard */')],
  ['empty readback closure',R,runtimeTest,'review repair readback rejects empty',s=>{
    const first="  for(const job of bundle.jobs.filter(j=>j.status==='completed'&&j.payload?.model?.startsWith('imported_')))assert.ok(byJob.has(job.id)&&bundle.media.some(m=>m.jobId===job.id),'completed imported generation/replay is missing an actual native closure/output binding');";
    const second="    if(bundle.jobs.find(j=>j.id===media.jobId)?.payload?.model?.startsWith('imported_'))assert.ok(byJob.has(media.jobId),'imported output cannot pass with zero load events');";
    return edit(edit(s,first,'/* unit_fake mutation: omitted closure inventory */'),second,'/* unit_fake mutation: omitted media closure */');
  }],
  ['authentic historical trace',R,runtimeTest,'review repair copied closure metadata',s=>edit(s,"assert.ok(observed.some(event=>JSON.stringify(event)===JSON.stringify(binding.event)),'native closure must occur in retained authentic worker/API trace, not copied/fabricated binding metadata');",'/* unit_fake mutation: omitted trace witness */')],
  ['ownership semantic assembly',R,runtimeTest,'review repair managed linked equivalence',s=>edit(s,"assert.deepEqual(semanticAssembly(binding.runtimeIdentity.quantization.observedAssembly),semanticAssembly(base.runtimeIdentity.quantization.observedAssembly),'ownership changed resolved semantic component/conditioning assembly');",'/* unit_fake mutation: omitted assembly equality */')],
  ['ownership baseline noise',R,runtimeTest,'review repair managed linked equivalence',s=>edit(s,"assert.ok(await difference(...pair)<=noiseLimit,'ownership repeated-baseline noise exceeds predeclared bound');",'void 0; /* unit_fake mutation: omitted repeated baseline bound */')],
  ['ownership decoded output',R,runtimeTest,'review repair managed linked equivalence',s=>edit(s,'differenceAcrossOwnership<=maximum','true')],
  ['actual metadata warmth',R,runtimeTest,'review repair cache coverage',s=>edit(s,"assert.equal(normalizeLoadEvent(warm.event).cacheState,'Warm','warm must actually hit generator metadata cache');",'/* unit_fake mutation: omitted warm observation */')],
  ['actual configured BTR',R,runtimeTest,'review repair cache coverage',s=>edit(s,"assert.equal(after.stream_transformer_blocks,true,'staged run must actually configure bounded transformer residency');",'/* unit_fake mutation: omitted actual BTR */')],
  ['source legal acknowledgment',C,casesTest,'review repair every import',s=>edit(s,'authority?.acknowledged===true','authority!==undefined')],
  ['inherited legal authority',C,casesTest,'review repair every import',s=>edit(s,'authority.subjects?.some(s=>s.role===role&&s.sha256===input.sha256)',"role==='component'||authority.subjects?.some(s=>s.role===role&&s.sha256===input.sha256)")],
  ['request AV binding is used',A,runtimeTest,'review repair request-bound AV validator',s=>edit(s,'export function validateAVAgainstRequest(decoded, request, resolvedContract) {','export function validateAVAgainstRequest(decoded, request, resolvedContract) { return {unit_fake_mutation:true};')],
  ['audio coverage bound',A,avTest,'short, long, wrong-rate',s=>edit(s,'Math.abs(audioDuration - expectedDuration) <= audioTolerance','true')],
  ['actual partial cold bytes',R,runtimeTest,'cold generation source-preparation cancellation fails loudly',s=>edit(s,"assert.ok(Number.isSafeInteger(partialBytes)&&partialBytes>0,'zero/tiny partial bytes are not a cold checkpoint');",'void 0; /* unit_fake mutation: omitted actual partial byte bound */')],
  ['memo is not cold content read',R,runtimeTest,'cold generation source-preparation cancellation fails loudly',s=>edit(s,"assert.equal(observed.acquisition,'ColdContentRead','only a cold content read is admissible; memo/other acquisitions are not cancellation evidence');",'void 0; /* unit_fake mutation: accepted memo as cold */')],
  ['joined preparation acknowledgment',R,runtimeTest,'cold generation source-preparation cancellation fails loudly',s=>edit(s,"assert.ok(typeof terminal.job.message==='string'&&terminal.job.message.includes(PREP_CANCELED_MARKER),'joined-teardown canceled root message marker required');",'void 0; /* unit_fake mutation: omitted actual joined marker */')],
  ['actual owned lease release witness',R,runtimeTest,'cold generation source-preparation cancellation fails loudly',s=>edit(s,"assert.equal(removal?.response?.removedManifestEntry,true,'refused model removal cannot prove source lease release');",'void 0; /* unit_fake mutation: omitted owned removal witness */')],
];

test('terminal repair source mutants are killed by executed focused assertions',async t=>{
  const dir=await mkdtemp(path.join(os.tmpdir(),'h3-repair-assertion-mutants-'));t.after(()=>rm(dir,{recursive:true,force:true}));
  const scripts=path.join(dir,'scripts');await mkdir(path.join(scripts,'ci'),{recursive:true});await mkdir(path.join(scripts,'lib'),{recursive:true});await mkdir(path.join(dir,'docs','testing'),{recursive:true});
  const files=[A,C,R,'h3-finetune-terminal-reduced.mjs','h3-finetune-terminal-stage.mjs','h3-finetune-terminal-api.mjs',casesTest,runtimeTest,avTest,'select-windows-cargo-cache.mjs','ci/h3-import-native-proof.mjs','lib/h3-campaign-build-profile.mjs'];
  const originals=new Map();for(const name of files){const source=await readFile(new URL(`scripts/${name}`,root),'utf8');originals.set(name,source);await writeFile(path.join(scripts,name),source);}
  await writeFile(path.join(dir,'docs','testing','sc-26691-h3-cases.json'),await readFile(new URL('docs/testing/sc-26691-h3-cases.json',root)));
  const env={...process.env,HF_HOME:path.join(dir,'hf'),HF_HUB_CACHE:path.join(dir,'hf','hub'),HUGGINGFACE_HUB_CACHE:path.join(dir,'hf','hub'),SCENEWORKS_DATA_DIR:path.join(dir,'data'),HF_HUB_OFFLINE:'1',HF_DATASETS_OFFLINE:'1'};delete env.NODE_TEST_CONTEXT;
  const run=(file,pattern)=>spawnSync(process.execPath,['--test',`--test-name-pattern=${pattern}`,path.join(scripts,file)],{encoding:'utf8',env,timeout:30000,maxBuffer:2*1024*1024});
  const baselineKeys=new Set();
  for(const [name,file,suite,pattern,mutate]of mutations){
    const key=`${suite}:${pattern}`;
    if(!baselineKeys.has(key)){const baseline=run(suite,pattern);assert.equal(baseline.status,0,`baseline must execute and pass before mutation ${name}\n${baseline.stdout}\n${baseline.stderr}`);assert.match(baseline.stdout,/pass 1\b/,'focused baseline must execute one test, not silently select zero');baselineKeys.add(key);}
    const source=originals.get(file),mutant=mutate(source);assert.notEqual(mutant,source);await writeFile(path.join(scripts,file),mutant);
    const syntax=spawnSync(process.execPath,['--check',path.join(scripts,file)],{encoding:'utf8',env});assert.equal(syntax.status,0,`non-compiling mutation is invalid: ${name}\n${syntax.stderr}`);
    const result=run(suite,pattern);await writeFile(path.join(scripts,file),source);
    assert.equal(result.status,1,`mutation survived ${name}\n${result.stdout}\n${result.stderr}`);
    assert.doesNotMatch(result.stdout+result.stderr,/ERR_MODULE_NOT_FOUND|SyntaxError|EADDRINUSE|listen EPERM/,'import/syntax/listener failure is not an assertion RED');
    assert.match(result.stdout+result.stderr,/AssertionError|Missing expected (exception|rejection)/,`must fail a meaningful assertion: ${name}`);
  }
});
