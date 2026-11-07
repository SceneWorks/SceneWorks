// Same-run retry selection only. Preserve every downloaded attempt byte-for-byte;
// copy the newest named producer into fresh canonical paths for native consumers.
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
const requireValue=(value,message)=>{if(!value)throw new Error(`MOSS artifact selection: ${message}`);};
export function stageLatestMossArtifacts(downloadRoot,stagingRoot,{phases,source,runId}) {
  requireValue(/^[1-9][0-9]*$/.test(runId??''),'same capture run required');
  requireValue(!existsSync(stagingRoot),'canonical staging must be fresh');
  const candidates=[];
  const names=readdirSync(downloadRoot);
  requireValue(names.length<=1000,'download inventory exceeds GitHub artifact-page bound');
  for(const name of names) {
    const match=/^(capture|cross-load)-(mlx|cuda)-attempt-([1-9][0-9]*)$/.exec(name);
    if(!match || !phases.includes(match[1])) continue;
    const attempt=Number(match[3]);
    requireValue(Number.isSafeInteger(attempt),'artifact attempt exceeds integer range');
    candidates.push({name,phase:match[1],backend:match[2],attempt});
  }
  const selected=[];
  for(const phase of phases) for(const backend of ['mlx','cuda']) {
    const matches=candidates.filter(c=>c.phase===phase&&c.backend===backend);
    requireValue(matches.length,`missing attempt-qualified ${phase}/${backend}`);
    const candidate=matches.reduce((latest,item)=>item.attempt>latest.attempt?item:latest);
    const directory=path.join(downloadRoot,candidate.name);
    requireValue(lstatSync(directory).isDirectory()&&!lstatSync(directory).isSymbolicLink(),'artifact directory must be regular');
    const driver=readFileSync(path.join(directory,'driver.json'));
    const receipt=JSON.parse(driver);
    requireValue(receipt.phase===phase&&receipt.backend===backend&&receipt.workflow?.attempt===candidate.attempt,'artifact name differs from producing phase/backend/attempt');
    requireValue(receipt.workflow.runId===runId,'artifact belongs to another capture run');
    for(const key of ['workflowSource','inferencePin','manifestSha256','cargoLockSha256','driverSha256','workflowSha256'])
      requireValue(receipt.source?.[key]===source[key],`artifact source mismatch: ${key}`);
    // Never skip a failed latest producer in favor of an older successful bundle.
    // The existing complete-case/member verifier judges this selected receipt.
    selected.push({...candidate,bundle:`${phase}-${backend}`,driverSha256:createHash('sha256').update(driver).digest('hex')});
  }
  mkdirSync(stagingRoot);
  for(const item of selected) cpSync(path.join(downloadRoot,item.name),path.join(stagingRoot,item.bundle),{recursive:true,force:false,errorOnExist:true,verbatimSymlinks:true});
  return {runId,bundles:selected.map(({name,...item})=>({...item,artifactName:name}))};
}
