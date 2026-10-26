#!/usr/bin/env node
// Download a live moss-preview-<backend>-<cell>-<run>-<attempt> artifact from the
// upload step's artifact-url notice, visually inspect its three PNGs, then run:
// node scripts/moss-preview-inspection.mjs --preview /download/index.json \
//   --output /reviews/inspections.json --reviewer NAME \
//   --findings '{"before-untrained":["observed ..."],"after-trained":["observed ..."],"disable-control":["observed ..."]}'
// Repeat as cells publish. Pass the completed eight-cell JSON to the existing
// manual finalizer as moss_inspection_json. No command/model is executed here.
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MANIFEST_SHA256, CELLS, BOUNDS, fileDigest, decodePreviewPng } from './moss-training-acceptance.mjs';
const requireValue=(value,message)=>{if(!value)throw new Error(`MOSS inspection: ${message}`);};
export function parseInspectionArgs(argv) {
  const options={};
  for(let i=0;i<argv.length;i+=2) {
    const key=argv[i]?.slice(2);
    requireValue(argv[i]?.startsWith('--') && ['preview','output','reviewer','findings'].includes(key) && argv[i+1] && !Object.hasOwn(options,key),'expected unique --preview, --output, --reviewer, --findings only (no timestamp override)');
    options[key]=argv[i+1];
  }
  requireValue(['preview','output','reviewer','findings'].every(k=>options[k]),'all four options are required');
  return {...options,findings:JSON.parse(options.findings)};
}
export function recordInspection({preview,output,reviewer,findings}) {
  const index=JSON.parse(readFileSync(preview,'utf8'));
  requireValue(index.manifestSha256===MANIFEST_SHA256 && index.source?.manifestSha256===MANIFEST_SHA256,'manifest seal mismatch');
  requireValue(['mlx','cuda'].some(b=>CELLS.some(c=>index.caseId===`${b}/${c}`)),'unexpected case');
  requireValue(/^[1-9][0-9]*$/.test(index.workflow?.runId??'') && Number.isSafeInteger(index.workflow.attempt) && index.workflow.attempt>0,'capture run/attempt missing');
  requireValue(typeof reviewer==='string' && reviewer.trim(),'named reviewer required');
  const roles=['before-untrained','after-trained','disable-control'];
  requireValue(Array.isArray(index.images) && index.images.length===3 && roles.every((r,i)=>index.images[i].role===r),'three exact preview roles required');
  requireValue(findings && Object.keys(findings).length===3 && roles.every(r=>Array.isArray(findings[r]) && findings[r].length && findings[r].every(f=>typeof f==='string' && f.trim())),'actual findings for each image required');
  for(const image of index.images) {
    requireValue(image.file===`${image.role}.png`,'unsafe preview file');
    const file=path.join(path.dirname(preview),image.file);
    requireValue(fileDigest(file)===image.reviewedArtifact?.sha256,'reviewed PNG hash changed');
    const decoded=decodePreviewPng(readFileSync(file));
    requireValue(JSON.stringify(decoded)===JSON.stringify(image.decoded),'decoded PNG pixels changed');
    const [backend,...cell]=index.caseId.split('/');
    requireValue(typeof image.reviewedArtifact.path==='string' && image.reviewedArtifact.path.startsWith(`capture-${backend}/${cell.join('/')}/`) && !image.reviewedArtifact.path.includes('\\') && !image.reviewedArtifact.path.includes(':') && !image.reviewedArtifact.path.split('/').includes('..'),'unsafe final artifact reference');
  }
  const document=existsSync(output)?JSON.parse(readFileSync(output,'utf8')):{manifestSha256:MANIFEST_SHA256,captureRunId:index.workflow.runId,attempt:index.workflow.attempt,source:index.source,cases:[]};
  requireValue(document.manifestSha256===MANIFEST_SHA256 && document.captureRunId===index.workflow.runId && document.attempt===index.workflow.attempt && JSON.stringify(document.source)===JSON.stringify(index.source),'cannot append another source/run/attempt');
  requireValue(Array.isArray(document.cases) && !document.cases.some(c=>c.caseId===index.caseId),'case inspection already recorded');
  // Read the wall clock at recording time, after hashing/decoding. There is no
  // supplied timestamp, deadline override, extension, pause, or backdate path.
  const now=Date.now(),start=Date.parse(index.caseStartedAt),deadline=Date.parse(index.caseDeadlineAt);
  requireValue(Number.isFinite(start) && deadline===start+BOUNDS.case && now>=Date.parse(index.publishedAt) && now>=start,'invalid original compute deadline or inspection before publication');
  const inspections=index.images.map(image=>({reviewer:reviewer.trim(),reviewedArtifact:image.reviewedArtifact,findings:findings[image.role],completedAt:new Date(now).toISOString(),beforeFinalAcceptance:true}));
  document.cases.push({caseId:index.caseId,inspections});
  mkdirSync(path.dirname(path.resolve(output)),{recursive:true});
  writeFileSync(output,JSON.stringify(document,null,2)+'\n');
  return document;
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try {const document=recordInspection(parseInspectionArgs(process.argv.slice(2)));console.log(`Recorded actual-time inspection: ${document.cases.length}/8 cells (partial documents cannot finalize).`);}catch(error){console.error(error.message);process.exitCode=1;}
}
