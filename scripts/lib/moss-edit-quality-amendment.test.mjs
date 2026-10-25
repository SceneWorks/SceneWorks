import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {VERSION,reviewVerdict,sha} from './moss-edit-terminal.mjs';
import {compareQualityRow,validateExecutionToolingCapture,AMENDMENT_SHA256} from './moss-edit-quality-amendment.mjs';

const hash='a'.repeat(64),reviewHash='b'.repeat(64),upstreamHash='c'.repeat(64);
const row={id:'quality-mlx-base-single',variant:'base',refs:['B'],anchors:['Bottle color','Bottle and orange placement']};
const outputs=[{sha256:hash}];
const review={version:VERSION,rowId:row.id,reviewer:'actual reviewer',reviewedAt:'2026-10-25T00:00:00Z',outputSha256:[hash],anchors:[
  {name:row.anchors[0],verdict:'PASS',observation:'The bottle is red.'},
  {name:row.anchors[1],verdict:'FAIL',observation:'The native orange is absent.'},
]};
const comparison=()=>({upstream:{'base-single':{image:{sha256:upstreamHash}}},rows:[{
  rowId:row.id,upstreamCell:'base-single',nativeReviewSha256:reviewHash,nativeOutputSha256:[hash],upstreamImageSha256:upstreamHash,
  reviewer:'actual comparison reviewer',reviewedAt:'2026-10-25T01:00:00Z',
  instructionFollowing:{native:'Native bottle is red.',upstream:'Upstream bottle is red.'},
  referencePreservation:{native:'Native misses an orange.',upstream:'Upstream misses an orange.'},
  anchors:[
    {name:row.anchors[0],nativeVerdict:'PASS',nativeObservation:'Red bottle observed.',upstreamVerdict:'PASS',upstreamObservation:'Red bottle observed.',comparativeObservation:'Both recolor bottle.',materialNativeOnlyRegression:false,classification:'both-pass'},
    {name:row.anchors[1],nativeVerdict:'FAIL',nativeObservation:'Orange missing.',upstreamVerdict:'FAIL',upstreamObservation:'Orange missing.',comparativeObservation:'Both omit the orange in this fixed case.',materialNativeOnlyRegression:false,classification:'shared-upstream-limitation'},
  ],sharedLimitations:[row.anchors[1]],verdict:'PASS',
}]});

test('approved amendment is pinned and a truthful shared limitation can pass while strict historical FAIL remains',()=>{
  assert.equal(AMENDMENT_SHA256,'596ad6038ae5e1ab211e646255b9a1f67be1cdd3a79ea07a741fce506d414fe9');
  assert.equal(reviewVerdict(row,review,outputs),'FAIL');
  assert.equal(compareQualityRow(row,review,reviewHash,outputs,comparison()),'PASS');
});
test('compound FAIL/FAIL still blocks when the reviewer observes a material native-only regression',()=>{
  const value=comparison(),anchor=value.rows[0].anchors[1];
  anchor.comparativeObservation='Both fail placement, but upstream retains the orange while native removes it.';
  anchor.materialNativeOnlyRegression=true;anchor.classification='native-only-regression';
  value.rows[0].sharedLimitations=[];value.rows[0].verdict='FAIL';
  assert.equal(compareQualityRow(row,review,reviewHash,outputs,value),'FAIL');
  value.rows[0].verdict='PASS';assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,value),/native-only regression/);
});
test('upstream-pass/native-fail and changed historical review or output cannot be waived',()=>{
  const value=comparison(),anchor=value.rows[0].anchors[1];
  anchor.upstreamVerdict='PASS';anchor.classification='native-only-regression';value.rows[0].sharedLimitations=[];
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,value),/cannot be waived/);
  assert.throws(()=>compareQualityRow(row,review,'d'.repeat(64),outputs,comparison()),/bound to actual reviewed outputs/);
  const changed=comparison();changed.rows[0].nativeOutputSha256=['d'.repeat(64)];
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,changed),/bound to actual reviewed outputs/);
});
test('actual paired visual narratives and every original anchor remain mandatory',()=>{
  const missing=comparison();missing.rows[0].anchors[1].comparativeObservation='';
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,missing),/paired feature observations/);
  const omitted=comparison();omitted.rows[0].anchors.pop();
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,omitted),/every original visual anchor/);
});
test('captured execution closure must match committed Git objects, independently of verifier HEAD',()=>{
  const root=fileURLToPath(new URL('../../',import.meta.url)),head=execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  const tree=execFileSync('git',['-C',root,'rev-parse',`${head}^{tree}`],{encoding:'utf8'}).trim();
  const original='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/SceneWorks/tooling';
  const paths={harness:'scripts/moss-edit-terminal.mjs',harnessLibrary:'scripts/lib/moss-edit-terminal.mjs',actionController:'scripts/moss-edit-terminal-actions.mjs',actionLibrary:'scripts/lib/moss-edit-terminal-actions.mjs',browserDriver:'scripts/moss-edit-terminal-browser.mjs',browserControls:'scripts/lib/moss-edit-terminal-controls.mjs'};
  const files=Object.fromEntries(Object.entries(paths).map(([name,path])=>[name,`${original}/${path}`]));
  const identities=Object.fromEntries(Object.entries(paths).map(([name,path])=>{const bytes=execFileSync('git',['-C',root,'show',`${head}:${path}`]);return[name,{byteLength:bytes.length,sha256:sha(bytes)}];}));
  const capture={repair:{retryOrdinal:3,tooling:{head,source:{head,tree},sourceRoot:original,files,identities}}};
  assert.equal(validateExecutionToolingCapture(capture,head,root),true);
  const changed=structuredClone(capture);changed.repair.tooling.identities.harness.sha256='d'.repeat(64);
  assert.throws(()=>validateExecutionToolingCapture(changed,head,root),/Git object/);
  const wrongPath=structuredClone(capture);wrongPath.repair.tooling.files.harness=resolve('/tmp','moss-edit-terminal.mjs');
  assert.throws(()=>validateExecutionToolingCapture(wrongPath,head,root),/path changed/);
});
