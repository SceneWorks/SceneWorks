// Source-paired CUDA candidate: preserve the old measured Mac producer while
// authenticating the narrow native fix after its merge onto inference main.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { canonical, requireThat, sha } from './moss-edit-terminal.mjs';

const SHA=/^[a-f0-9]{40}$/;
const CANDLE_CHANGES=[
  'crates/media/candle-gen/candle-gen-moss-image/src/lib.rs',
  'crates/media/candle-gen/candle-gen-moss-image/src/release.rs',
  'crates/media/candle-gen/candle-gen-moss-image/src/tests.rs',
];
const HOST_TEST='scripts/tests/test_media_campaign_supervisor.py';
const HOST_TEST_OBJECT='ffea01739ca003d136e6cf5b50e3742408efa98c';
const HOST_TEST_SHA256='3d9be3f2b98d086607a08e67faca59724ff0b81145d6afe2783e6ec6ade1ee83';
const HOST_TEST_REVIEWED='d73f4ce83a774b1c9b35992a0bd01a8946542f0e';
const FIXTURE_FILES=new Set([
  'crates/sceneworks-worker/src/mlx_fit_gate.rs',
  'crates/sceneworks-worker/src/mlx_phase_estimate.rs',
  'crates/sceneworks-worker/src/video_admission/tests.rs',
]);
const PRODUCT_PIN_FILES=new Set(['Cargo.toml','Cargo.lock','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml']);
const git=(root,...args)=>execFileSync('git',['-C',root,...args],{encoding:'utf8'}).trim();
const changed=(root,before,after)=>git(root,'diff','--name-only',before,after,'--').split('\n').filter(Boolean).sort();
const object=(root,rev,path)=>git(root,'rev-parse',`${rev}:${path}`);
export function cudaSourceContinuity({sceneworksRoot,inferenceRoot,oldSourceSha,oldInferenceSha,sourceSha,inferenceSha,auditPath,auditSha256}) {
  requireThat([oldSourceSha,oldInferenceSha,sourceSha,inferenceSha].every(value=>SHA.test(value))&&/^[a-f0-9]{64}$/.test(auditSha256??''),'source continuity needs exact old/new commits and reviewed audit bytes');
  const auditBytes=readFileSync(auditPath);requireThat(sha(auditBytes)===auditSha256,'reviewed Mac MOSS source carry-forward audit changed');
  const audit=JSON.parse(auditBytes);
  requireThat(audit.kind==='sc-26752-mac-moss-source-carry-forward-audit/v1'&&audit.measuredNative===oldInferenceSha&&audit.candleFixParent===oldInferenceSha&&audit.currentMainAudited==='350bd900ee559ad0e5f2227b3e551d81b74b9501'&&audit.candleFixAudited==='9bb5233840347b386eaa0fc52bfe228a9c21916f'&&audit.candleFixAndMainChangedPathsDisjoint===true&&audit.futureMergedNativeShaAudited===null,'reviewed source audit identity/limits changed');
  requireThat(canonical(changed(inferenceRoot,oldInferenceSha,audit.currentMainAudited))===canonical(audit.allMainChangedPaths.slice().sort())&&canonical(changed(inferenceRoot,oldInferenceSha,audit.candleFixAudited))===canonical(CANDLE_CHANGES.slice().sort())&&canonical(audit.candleFixChangedPaths.slice().sort())===canonical(CANDLE_CHANGES.slice().sort()),'pre-fix main or exact native cleanup source diff changed');
  git(inferenceRoot,'merge-base','--is-ancestor',audit.currentMainAudited,inferenceSha);
  git(inferenceRoot,'merge-base','--is-ancestor',audit.candleFixAudited,inferenceSha);
  git(inferenceRoot,'merge-base','--is-ancestor',HOST_TEST_REVIEWED,inferenceSha);
  requireThat(canonical(changed(inferenceRoot,audit.currentMainAudited,inferenceSha))===canonical([...CANDLE_CHANGES,HOST_TEST].sort()),'merged CUDA native source changed beyond reviewed Candle cleanup and host test repair');
  requireThat(object(inferenceRoot,inferenceSha,HOST_TEST)===HOST_TEST_OBJECT&&sha(execFileSync('git',['-C',inferenceRoot,'show',`${inferenceSha}:${HOST_TEST}`]))===HOST_TEST_SHA256,'reviewed host test repair differs from exact native CI fix');
  for(const entry of audit.byteIdenticalTreesAndBlobs)requireThat(entry.identical===true&&entry.oldObject===entry.newObject&&object(inferenceRoot,oldInferenceSha,entry.path)===entry.oldObject&&object(inferenceRoot,inferenceSha,entry.path)===entry.oldObject,`measured Mac closure changed: ${entry.path}`);
  for(const entry of audit.sharedContractOrMLXCoreChangedPaths)requireThat(object(inferenceRoot,audit.currentMainAudited,entry.path)===entry.newObject&&object(inferenceRoot,inferenceSha,entry.path)===entry.newObject,`audited unrelated main source changed in merge: ${entry.path}`);
  for(const entry of audit.candleFixBlobIdentities)requireThat(object(inferenceRoot,inferenceSha,entry.path)===entry.newObject,`corrected Candle source differs from native fix: ${entry.path}`);
  const product=changed(sceneworksRoot,oldSourceSha,sourceSha);
  requireThat(product.every(path=>PRODUCT_PIN_FILES.has(path)||FIXTURE_FILES.has(path)||path.startsWith('scripts/moss-edit-')||path.startsWith('scripts/lib/moss-edit-')||path.startsWith('scripts/ci/moss-upstream-')||path==='scripts/lib/moss-private-hf.mjs'||path==='scripts/qwen-image-2-1-app-lifecycle.test.mjs'||path==='.github/workflows/moss-edit-terminal.yml'||path==='.github/workflows/memory-catalog-campaign.yml'||path==='package.json'),'corrected CUDA SceneWorks source changed API, UI, model manifest or unrelated runtime code');
  const fixturePatch=git(sceneworksRoot,'diff','--unified=0',oldSourceSha,sourceSha,'--',...FIXTURE_FILES);
  const added=fixturePatch.split('\n').filter(line=>line.startsWith('+')&&!line.startsWith('+++'));
  const removed=fixturePatch.split('\n').filter(line=>line.startsWith('-')&&!line.startsWith('---'));
  requireThat(added.length===4&&added.every(line=>/^\+\s*native_quant:\s*None,\s*$/.test(line))&&removed.length===0,'MLX test fixtures changed beyond four required None fields');
  requireThat(product.some(path=>PRODUCT_PIN_FILES.has(path)),'new stationary product source has no inference pin change');
  return {oldSourceSha,oldInferenceSha,sourceSha,inferenceSha,auditSha256,preFixMainSha:audit.currentMainAudited,nativeFixSha:audit.candleFixAudited,hostTestRepairSha:HOST_TEST_REVIEWED,hostTestRepairObject:HOST_TEST_OBJECT,productChangedFiles:product,inferenceFixChangedFiles:CANDLE_CHANGES};
}

export function validateCudaRebaseRecord(record,old,current) {
  requireThat(record?.name==='sc-26757-corrected-cuda-candidate/v1'&&record.old?.lockSha256===old.lockSha256&&record.old?.candidateId===old.candidateId&&record.old?.sourceSha===old.sourceSha&&record.old?.inferenceSha===old.inferenceSha&&record.old?.verifiedRows===25&&record.old?.verifiedGuards===29&&record.current?.candidateId===current.candidateId&&record.current?.candidateId!==record.old?.candidateId&&record.current?.sourceSha===current.sourceSha&&record.current?.inferenceSha===current.inferenceSha&&record.continuity?.oldSourceSha===old.sourceSha&&record.continuity?.oldInferenceSha===old.inferenceSha&&record.continuity?.sourceSha===current.sourceSha&&record.continuity?.inferenceSha===current.inferenceSha,'CUDA source-paired handoff or old Mac completion identity changed');
  return record;
}
