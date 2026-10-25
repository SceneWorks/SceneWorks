// Source-paired CUDA candidate: preserve the old measured Mac producer while
// authenticating the narrow native fix after its merge onto inference main.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { canonical, requireThat, sha } from './moss-edit-terminal.mjs';
import { parse as parseCandidates, populationSha256, parseCrates, cratePopulationSha256 } from '../scan-inference-provenance.mjs';

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
const PRODUCT_PIN_FILES=new Set(['Cargo.toml','Cargo.lock','crates/sceneworks-core/Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml']);
const PIN_PROVENANCE_FILES=new Set(['crates/sceneworks-worker/src/catalog_semantic_jobs.rs','crates/sceneworks-memory-adapter/src/lib.rs']);
const PIN_DERIVED_FILES=new Set(['config/inference-provider-closures.json','config/anchor-loader-closures.json','config/inference-provenance-candidates.tsv','config/inference-third-party-source.json']);
// Git objects of the eleven official pin outputs from c65aea3c, with the audit
// comment's full-SHA prose shortened for H36 CI (its derived auditDigest is unchanged).
// Later tooling commits may not substitute generated facts for this fixed pair.
const PIN_OUTPUT_OBJECTS=new Map(Object.entries({
  'Cargo.lock':'e6feb2d0a6e419a79ff2e589e9144f552780d65a',
  'Cargo.toml':'6c0d98fc2105f997a3c3ae011a3950af5865d083',
  'config/anchor-loader-closures.json':'2fd9fb27f8e69f1fbd61d741fcba9bba24fe7e47',
  'config/inference-provenance-candidates.tsv':'76fbba15a6a29fce37a386a1394daea08816fb21',
  'config/inference-provider-closures.json':'6665dbb1160d81ff91c5a9f05fb2a244f1c73862',
  'config/inference-third-party-source.json':'1ca93910c8af60a9b75f74157a87df0fd294a4b5',
  'crates/sceneworks-core/Cargo.toml':'69e4bf702e5e2cf4139f222d9f5c0e3e464f3854',
  'crates/sceneworks-memory-adapter/Cargo.toml':'7c2e7712c5c186cd8544cc6be17a251e4bc4b02a',
  'crates/sceneworks-memory-adapter/src/lib.rs':'614b4133c67ef6aa5b0c7aa700f3fdf015603069',
  'crates/sceneworks-worker/Cargo.toml':'a54637106a75b4a27d307393107e70bf25031b38',
  'crates/sceneworks-worker/src/catalog_semantic_jobs.rs':'1310487d4e8c209ad78120e3aa77e772976c0b67',
}));
const git=(root,...args)=>execFileSync('git',['-C',root,...args],{encoding:'utf8',maxBuffer:64*1024*1024}).trim();
const changed=(root,before,after)=>git(root,'diff','--name-only',before,after,'--').split('\n').filter(Boolean).sort();
const object=(root,rev,path)=>git(root,'rev-parse',`${rev}:${path}`);
const blob=(root,rev,path)=>execFileSync('git',['-C',root,'show',`${rev}:${path}`],{encoding:'utf8',maxBuffer:64*1024*1024});
const equal=(left,right,message)=>requireThat(canonical(left)===canonical(right),message);
const inferencePins=text=>[...text.matchAll(/git\s*=\s*"https:\/\/github\.com\/SceneWorks\/inference"[^\n]*\brev\s*=\s*"([a-f0-9]{40})"/g)].map(match=>match[1]);
export const sourceAuditDigest=audit=>sha(JSON.stringify(Object.fromEntries(['artifacts','prospectiveDisclosures','provenanceScan','portedSourceAreas','crateCoverage','crateDispositions','includeSites'].map(key=>[key,audit[key]]))));
export function validateOfficialPinObjects(objectAt) {
  requireThat(PIN_OUTPUT_OBJECTS.size===PRODUCT_PIN_FILES.size+PIN_PROVENANCE_FILES.size+PIN_DERIVED_FILES.size,'official pin output inventory incomplete');
  for(const [path,expected] of PIN_OUTPUT_OBJECTS)requireThat(objectAt(path)===expected,`official pin output Git object changed: ${path}`);
}

export function validateOfficialPinTransition({oldSourceSha,oldInferenceSha,inferenceSha,readOld,readNew,treeObjects}) {
  for(const path of PRODUCT_PIN_FILES) {
    if(path==='Cargo.lock')continue;
    const before=readOld(path),after=readNew(path),pins=inferencePins(after);
    requireThat(pins.length>0&&pins.every(pin=>pin===inferenceSha)&&before.includes(oldInferenceSha)&&before.replaceAll(oldInferenceSha,inferenceSha)===after,`official inference manifest changed beyond its pin: ${path}`);
  }
  const lock=readNew('Cargo.lock');
  const lockSources=[...lock.matchAll(/source = "git\+https:\/\/github\.com\/SceneWorks\/inference[^\n]*"/g)].map(match=>match[0]);
  requireThat(lockSources.length>0&&lockSources.every(line=>line.includes(inferenceSha)&&!line.includes(oldInferenceSha)),'Cargo.lock retains a stale or mixed inference source');
  for(const path of PIN_PROVENANCE_FILES) {
    const before=readOld(path),after=readNew(path);
    requireThat(before.includes(oldInferenceSha)&&before.replaceAll(oldInferenceSha,inferenceSha)===after,`official pin provenance changed beyond its revision: ${path}`);
  }
  for(const [path,key,identity] of [
    ['config/inference-provider-closures.json','providers',entry=>({crate:entry.crate})],
    ['config/anchor-loader-closures.json','models',entry=>({engineId:entry.engineId??null,entryPoints:entry.entryPoints})],
  ]) {
    const before=JSON.parse(readOld(path)),after=JSON.parse(readNew(path));
    requireThat(before.inferenceRevision===oldInferenceSha&&after.inferenceRevision===inferenceSha&&before.digestVersion===after.digestVersion&&before._comment===after._comment,`official closure revision or schema changed: ${path}`);
    equal(Object.keys(before[key]),Object.keys(after[key]),`official closure declarations changed: ${path}`);
    for(const id of Object.keys(before[key]))equal(identity(before[key][id]),identity(after[key][id]),`official closure route changed: ${path}:${id}`);
  }
  const oldCandidates=parseCandidates(readOld('config/inference-provenance-candidates.tsv'));
  const candidates=parseCandidates(readNew('config/inference-provenance-candidates.tsv'));
  equal(oldCandidates.map(({path,markerSha256,area})=>({path,markerSha256,area})),candidates.map(({path,markerSha256,area})=>({path,markerSha256,area})),'official provenance scan changed candidate population or classifications');
  if(treeObjects)for(const candidate of candidates)requireThat(treeObjects.get(candidate.path)===candidate.blob,`official provenance scan does not match merged native Git object: ${candidate.path}`);
  const audit=JSON.parse(readNew('config/inference-third-party-source.json'));
  const previous=JSON.parse(readOld('config/inference-third-party-source.json'));
  const crates=parseCrates(readNew('config/inference-crate-prefixes.txt'));
  const expected=structuredClone(previous);
  expected._comment=audit._comment;
  expected.inferenceRevision=inferenceSha;
  expected.provenanceScan.revision=inferenceSha;
  expected.provenanceScan.matchedFiles=candidates.length;
  expected.provenanceScan.populationSha256=populationSha256(candidates);
  expected.crateCoverage.revision=inferenceSha;
  expected.crateCoverage.cratePrefixes=crates.length;
  expected.crateCoverage.cratePopulationSha256=cratePopulationSha256(crates);
  expected.auditDigest=sourceAuditDigest(expected);
  requireThat(audit._comment.includes(inferenceSha.slice(0,8))&&new RegExp(`\\b${candidates.length}\\b`).test(audit._comment)&&new RegExp(`\\b${crates.length}\\b`).test(audit._comment),'source-audit human summary does not identify the current pin and populations');
  equal(audit,expected,'official source audit changed beyond derived pin/population facts and reviewed summary');
  return {pinFiles:[...PRODUCT_PIN_FILES,...PIN_PROVENANCE_FILES,...PIN_DERIVED_FILES].sort(),providerCount:Object.keys(JSON.parse(readNew('config/inference-provider-closures.json')).providers).length,anchorCount:Object.keys(JSON.parse(readNew('config/anchor-loader-closures.json')).models).length,candidateCount:candidates.length,crateCount:crates.length};
}
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
  requireThat(product.every(path=>PRODUCT_PIN_FILES.has(path)||PIN_PROVENANCE_FILES.has(path)||PIN_DERIVED_FILES.has(path)||FIXTURE_FILES.has(path)||path.startsWith('scripts/moss-edit-')||path.startsWith('scripts/lib/moss-edit-')||path.startsWith('scripts/ci/moss-upstream-')||path==='scripts/ci/moss-first-step-probe.mjs'||path==='scripts/ci/moss-first-step-probe.test.mjs'||path==='scripts/lib/moss-private-hf.mjs'||path==='scripts/qwen-image-2-1-app-lifecycle.test.mjs'||path==='.github/workflows/moss-edit-terminal.yml'||path==='.github/workflows/memory-catalog-campaign.yml'||path==='package.json'),'corrected CUDA SceneWorks source changed API, UI, model manifest or unrelated runtime code');
  validateOfficialPinObjects(path=>object(sceneworksRoot,sourceSha,path));
  const nativeTree=new Map(git(inferenceRoot,'ls-tree','-r',inferenceSha).split('\n').filter(Boolean).map(line=>{const match=/^\d+ blob ([a-f0-9]{40})\t(.+)$/.exec(line);return match?[match[2],match[1]]:null}).filter(Boolean));
  const pin=validateOfficialPinTransition({oldSourceSha,oldInferenceSha,inferenceSha,readOld:path=>blob(sceneworksRoot,oldSourceSha,path),readNew:path=>blob(sceneworksRoot,sourceSha,path),treeObjects:nativeTree});
  const fixturePatch=git(sceneworksRoot,'diff','--unified=0',oldSourceSha,sourceSha,'--',...FIXTURE_FILES);
  const added=fixturePatch.split('\n').filter(line=>line.startsWith('+')&&!line.startsWith('+++'));
  const removed=fixturePatch.split('\n').filter(line=>line.startsWith('-')&&!line.startsWith('---'));
  requireThat(added.length===4&&added.every(line=>/^\+\s*native_quant:\s*None,\s*$/.test(line))&&removed.length===0,'MLX test fixtures changed beyond four required None fields');
  requireThat(product.some(path=>PRODUCT_PIN_FILES.has(path)),'new stationary product source has no inference pin change');
  return {oldSourceSha,oldInferenceSha,sourceSha,inferenceSha,auditSha256,preFixMainSha:audit.currentMainAudited,nativeFixSha:audit.candleFixAudited,hostTestRepairSha:HOST_TEST_REVIEWED,hostTestRepairObject:HOST_TEST_OBJECT,productChangedFiles:product,inferenceFixChangedFiles:CANDLE_CHANGES,pin};
}

export function validateCudaRebaseRecord(record,old,current) {
  requireThat(record?.name==='sc-26757-corrected-cuda-candidate/v1'&&record.old?.lockSha256===old.lockSha256&&record.old?.candidateId===old.candidateId&&record.old?.sourceSha===old.sourceSha&&record.old?.inferenceSha===old.inferenceSha&&record.old?.verifiedRows===25&&record.old?.verifiedGuards===29&&record.current?.candidateId===current.candidateId&&record.current?.candidateId!==record.old?.candidateId&&record.current?.sourceSha===current.sourceSha&&record.current?.inferenceSha===current.inferenceSha&&record.continuity?.oldSourceSha===old.sourceSha&&record.continuity?.oldInferenceSha===old.inferenceSha&&record.continuity?.sourceSha===current.sourceSha&&record.continuity?.inferenceSha===current.inferenceSha,'CUDA source-paired handoff or old Mac completion identity changed');
  return record;
}
