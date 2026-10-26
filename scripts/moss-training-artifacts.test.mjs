import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,readdirSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {stageLatestMossArtifacts} from './moss-training-artifacts.mjs';
const source={workflowSource:'a'.repeat(40),inferencePin:'b'.repeat(40),manifestSha256:'c'.repeat(64),cargoLockSha256:'d'.repeat(64),driverSha256:'e'.repeat(64),workflowSha256:'f'.repeat(64)};
function setup(t) {
 const root=mkdtempSync(path.join(os.tmpdir(),'moss-artifact-selection-'));
 t.after(()=>rmSync(root,{recursive:true,force:true}));
 const downloads=path.join(root,'downloads');mkdirSync(downloads);
 const add=(phase,backend,attempt,change={})=>{
  const directory=path.join(downloads,`${phase}-${backend}-attempt-${attempt}`);mkdirSync(directory);
  const receipt={phase,backend,ok:true,source,workflow:{runId:'123',attempt},...change};
  writeFileSync(path.join(directory,'driver.json'),JSON.stringify(receipt));writeFileSync(path.join(directory,'raw'),'SCHEMA TEST ONLY');
  return receipt;
 };
 const stage=()=>stageLatestMossArtifacts(downloads,path.join(root,'selected'),{phases:['capture'],source,runId:'123'});
 return {root,downloads,add,stage};
}
test('latest same-run attempts stage without changing originals or inventing attempts',t=>{
 const f=setup(t);f.add('capture','mlx',1);f.add('capture','cuda',1);f.add('capture','cuda',2);
 const before=readdirSync(f.downloads).map(name=>[name,readFileSync(path.join(f.downloads,name,'driver.json'))]);
 const result=f.stage();assert.deepEqual(result.bundles.map(x=>x.attempt),[1,2]);
 for(const [name,bytes] of before) assert.deepEqual(readFileSync(path.join(f.downloads,name,'driver.json')),bytes);
 assert.equal(JSON.parse(readFileSync(path.join(f.root,'selected','capture-cuda','driver.json'))).workflow.attempt,2);
 assert.throws(f.stage,/fresh/);
});
test('failed latest artifact is selected, never hidden behind older success',t=>{
 const f=setup(t);f.add('capture','mlx',1);f.add('capture','cuda',1);f.add('capture','cuda',2,{ok:false});
 assert.equal(f.stage().bundles[1].attempt,2);
 assert.equal(JSON.parse(readFileSync(path.join(f.root,'selected','capture-cuda','driver.json'))).ok,false);
});
test('wrong run, source pin, named attempt and ambiguous noncanonical names fail closed',t=>{
 for(const [key,value] of [['run',{workflow:{runId:'124',attempt:2}}],['source',{source:{...source,inferencePin:'0'.repeat(40)}}],['attempt',{workflow:{runId:'123',attempt:1}}]]) {
  const f=setup(t);f.add('capture','mlx',1);f.add('capture','cuda',2,value);assert.throws(f.stage,/another capture run|source mismatch|phase\/backend\/attempt/,key);
 }
 const f=setup(t);f.add('capture','mlx',1);mkdirSync(path.join(f.downloads,'capture-cuda-attempt-02'));
 assert.throws(f.stage,/missing attempt-qualified/);
});
