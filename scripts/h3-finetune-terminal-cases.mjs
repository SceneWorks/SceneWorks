// Concrete sc-26691 case preparation for the existing terminal/API harness.
// Pure data/scenario construction: no fetch, subprocess, app startup or model execution.
import assert from 'node:assert/strict';
import path from 'node:path';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { requiredCases, SOURCE, ARTIFACTS, hashFile, validatePlan, verifiedStagingRoot } from './h3-finetune-terminal.mjs';
import { operationalCases, operationalCaseById, chooseMatrixBackend } from './h3-finetune-terminal-reduced.mjs';
const caseCheck=c=>c.kind==='source-cancel'?'generation-source-prep-cancel':c.sourceVariant==='opaque-renamed-linked'?'renamed':c.sourceVariant==='independent-finetune-pulpcut-merge'?'independent-finetune':c.id.split('.').slice(2).join('.');
export const caseBook = JSON.parse(await readFile(new URL('../docs/testing/sc-26691-h3-cases.json', import.meta.url), 'utf8'));
const clone = structuredClone;
const cap = name => ({ $capture: name });
const route = (...parts) => ({ $concat: parts });
const eq = (pointer, equals) => ({ pointer, equals });
const hex = v => assert.match(v ?? '', /^[a-f0-9]{64}$/);
const absoluteSourcePath = v => typeof v === 'string' && (path.posix.isAbsolute(v) || path.win32.isAbsolute(v));
const knownModes = ['text_to_video', 'image_to_video', 'first_last_frame', 'reference_to_video'];
const lifecycleApi = new Set(['managed-linked-equivalence', 'restart-replay', 'removal', 'cancel', 'interruption-retry', 'bounded-loading', 'cold-warm-staged', 'generation-source-prep-cancel']);
const hasScenario = c => c.kind==='source-cancel' || c.kind==='full-av' || c.group !== 'numeric' && !c.id.endsWith('.generalization.existing-families') && (c.group !== 'lifecycle' || lifecycleApi.has(caseCheck(c)));
export const fixtureBindings = {
  generalization: {
    renamed: ['sceneworks-core', 'minimax_h3_import', 'an_h3_dit_is_identified_by_its_tensors_not_by_its_name'],
    'independent-finetune': ['sceneworks-core', 'minimax_h3_import', 'an_h3_dit_is_identified_by_its_tensors_not_by_its_name'],
    'native-single': ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::native_and_sharded_plans_reach_the_real_h3_provider_load_seam'],
    'sharded-diffusers': ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::linked_mixed_case_shards_reach_the_provider_as_lowercase_canonical_files'],
    'dense-f16': ['sceneworks-core', 'minimax_h3_import', 'the_import_gate_admits_only_a_complete_dense_h3_component_directory'],
    'dense-bf16': ['sceneworks-core', 'minimax_h3_import', 'the_import_gate_admits_only_a_complete_dense_h3_component_directory'],
    'dense-f32': ['sceneworks-core', 'minimax_h3_import', 'the_import_gate_admits_only_a_complete_dense_h3_component_directory'],
    'existing-families': ['sceneworks-core', 'checkpoint_ingest', null],
    'ltx-add-select-generate': ['sceneworks-core', 'checkpoint_ingest', null],
    'compatible-base': ['sceneworks-core', 'minimax_h3_import', 'compatible_base_downloads_never_replace_imported_task_partitions'],
  },
  lifecycle: {
    'managed-linked-equivalence': ['sceneworks-core', 'minimax_h3_import', 'managed_and_linked_imports_persist_a_plan_the_h3_assembly_binds'],
    'restart-replay': ['sceneworks-core', 'minimax_h3_import', 'weight_only_config_is_immutable_semantic_context_across_ownership_restart_and_normalization'],
    'source-drift': ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::source_mutation_before_or_during_normalization_cannot_publish'],
    'interruption-retry': ['sceneworks-core', 'minimax_h3_import', 'h3_interruption_retry_drift_relink_duplicates_and_leases_preserve_ownership'],
    removal: ['sceneworks-core', 'checkpoint_derivative', 'removing_a_root_drops_its_derivatives_and_leaves_the_library_intact'],
    cancel: ['sceneworks-core', 'checkpoint_ingest', 'a_cancelled_ingest_leaves_no_install_no_plan_and_no_record'],
    'relink-rescan': ['sceneworks-core', 'checkpoint_plan_store', 'relinking_to_a_different_library_refuses_to_serve_the_old_plan'],
    leases: ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::provider_resolution_holds_dense_and_native_source_leases_until_release'],
    duplicate: ['sceneworks-core', 'minimax_h3_import', 'h3_interruption_retry_drift_relink_duplicates_and_leases_preserve_ownership'],
    'bounded-loading': ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::imported_h3_admission_requires_a_known_sufficient_budget'],
    'cold-warm-staged': ['sceneworks-worker', null, 'video_jobs::minimax_h3_import_tests::bound_shared_tier_and_exact_snapshots_reach_native_loadspec_despite_distracting_main'],
    'provenance-license': ['sceneworks-rust-api', null, 'custom_h3_import_requires_the_selected_base_component_license_acknowledgment'],
  },
};
function cargo(root, backend, [pkg, target, selector], list = false) {
  const argv = ['cargo', 'test', '--manifest-path', path.join(root, 'Cargo.toml'), '-p', pkg];
  if (pkg === 'sceneworks-worker' && backend === 'cuda') argv.push('--features', 'backend-candle');
  if (target) argv.push('--test', target);
  else argv.push('--lib');
  argv.push('--');
  if (selector && !list) argv.push(selector);
  argv.push(...(list ? ['--list'] : ['--nocapture', '--test-threads=1']));
  return argv;
}
export function nativeCommands(root, backend, check) {
  assert.ok(['mlx', 'cuda'].includes(backend));
  const pkg = backend === 'mlx' ? 'mlx-gen-minimax-h3' : 'candle-gen-minimax-h3';
  const selectors = caseBook.numeric[check]?.selectors ?? [caseBook.numeric[check]?.selector];
  assert.ok(selectors.every(s => s && s.includes('::')), 'native arithmetic selector required');
  const base = ['cargo', 'test', '--manifest-path', path.join(root, 'Cargo.toml'), '-p', pkg, '--test', 'integration'];
  if (backend === 'cuda') base.push('--features', 'cuda');
  return {
    // Unfiltered listing of the ACTUAL binary, never a hand-trimmed fixture population.
    inventory: [...base, '--', '--list'],
    commands: selectors.map(s => [...base, '--', s, '--exact', '--ignored', '--nocapture', '--test-threads=1']),
    requirement: 'Each exact invocation must report >0 passed, zero failed/ignored and no SKIP; native device absence is failure. These are primitive fixtures, not real-model evidence.',
  };
}
export function referenceCommand(root, check, referenceDir, outputDir) {
  const n = caseBook.numeric[check]; assert.ok(n?.fixture);
  const cmd = ['python3', '-I', path.join(root, n.program)];
  if (check !== 'convrot') cmd.push('--reference-dir', referenceDir);
  if (check === 'w4a8') cmd.push('--convrot-fixture', path.join(root, 'tests/fixtures/h3_convrot_int8.json'));
  cmd.push('--output', path.join(outputDir, path.basename(n.fixture)));
  if (check === 'w4a8') cmd.push('--receipt', path.join(outputDir, 'w4a8-reference-receipt.json'));
  return cmd;
}
export async function prepareControlledLoKr(spec,directory) {
  assert.ok(path.isAbsolute(directory));
  assert.match(spec.target??'',/^transformer_blocks\.\d+\.attn\.to_(q|k|v|out)$/,'explicit supported body target required, not an AdaLN filter');
  hex(spec.sourceCheckpointSha256);hex(spec.sourceContractSha256);assert.ok(spec.sourceDeclaration,'actual source/shape declaration required');
  const [out,input]=spec.shape??[];assert.ok([out,input].every(v=>Number.isSafeInteger(v)&&v>0));
  assert.ok(Number.isSafeInteger(spec.seed)&&spec.seed>=0&&spec.seed<=0xffffffff);assert.ok(Number.isFinite(spec.amplitude)&&spec.amplitude>0);
  const factor=n=>{for(let k=Math.floor(Math.sqrt(n));k>0;k--)if(n%k===0)return k;};
  const a=factor(out),b=out/a,c=factor(input),d=input/c,shapes=[[a,c],[b,d]];
  const elements=a*c+b*d;assert.ok(elements*4<=2*1024*1024,'this CPU conditioning fixture must remain small; no dense whole-model delta');
  let state=spec.seed>>>0;const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return (state/4294967296*2-1)*spec.amplitude;};
  const tensors=shapes.map(shape=>{const bytes=Buffer.alloc(shape[0]*shape[1]*4);let nonzero=false;for(let i=0;i<bytes.length/4;i++){const value=Math.fround(next());assert.ok(Number.isFinite(value),'generated factor must be finite F32');nonzero ||= value!==0;bytes.writeFloatLE(value,i*4);}assert.ok(nonzero,'generated factors must have a nonzero compatibility effect');return bytes;});
  const header={__metadata__:{format:'pt',networkType:'lokr',rank:String(a),alpha:String(a),sc26691_provenance:'controlled-generated-body-only; NOT a public trained adapter',sourceCheckpointSha256:spec.sourceCheckpointSha256,sourceContractSha256:spec.sourceContractSha256}};let offset=0;
  for(const [i,suffix] of ['lokr_w1','lokr_w2'].entries()){header[`${spec.target}.${suffix}`]={dtype:'F32',shape:shapes[i],data_offsets:[offset,offset+tensors[i].length]};offset+=tensors[i].length;}
  const text=Buffer.from(JSON.stringify(header)),padding=(8-text.length%8)%8,padded=Buffer.concat([text,Buffer.alloc(padding,32)]),size=Buffer.alloc(8);size.writeBigUInt64LE(BigInt(padded.length));
  await mkdir(directory);const file=path.join(directory,'controlled-body-lokr.safetensors');await writeFile(file,Buffer.concat([size,padded,...tensors]),{flag:'wx'});
  const provenance={kind:'prepared-generated-input-not-campaign-proof',...(spec.proof_kind==='unit_fake'?{proof_kind:'unit_fake'}:{}),trained:false,license:'CC0-1.0 newly generated factor values; inherited base-model terms still apply to model execution',sourceCheckpointSha256:spec.sourceCheckpointSha256,sourceContractSha256:spec.sourceContractSha256,sourceDeclaration:spec.sourceDeclaration,target:spec.target,shape:spec.shape,factorShapes:shapes,seed:spec.seed,amplitude:spec.amplitude,tensorNames:Object.keys(header).filter(k=>k!=='__metadata__'),sha256:await hashFile(file),meaning:'Every generated factor/delta is retained. This does not trim, convert or make claims about a public trained adapter. Native target/shape compatibility and effect on the real checkpoint must be proved by the authorized case.'};
  await writeFile(path.join(directory,'provenance.json'),JSON.stringify(provenance,null,2)+'\n',{flag:'wx'});return {file,provenance};
}
export function concretePlan(swRoot, inferenceRoot, stageRoot, {inventory='reduced-v1',matrixDecision}={}) {
  if(inventory==='reduced-v1'){
    const cases=operationalCases().map(c=>{
      const scenario=path.join(stageRoot,`${c.id}.scenario.json`);
      return {...c,...(c.group==='artifact'?{artifact:ARTIFACTS[c.index-1]}:{}),...(c.kind!=='actual-artifact-probe'?{scenario}:{}),commands:[['node',path.join(swRoot,'scripts/h3-finetune-terminal-runtime.mjs'),c.kind==='actual-artifact-probe'?'actual-artifact-probe':'api',path.join(stageRoot,'campaign-plan.json'),c.id,swRoot,inferenceRoot]]};
    });
    return {kind:'preparation-not-campaign-evidence',inventory,source:SOURCE,matrixDecision,cases,regressionCatalog:requiredCases(),unresolvedInterfaces:caseBook.unresolvedInterfaces};
  }
  for (const r of [swRoot, inferenceRoot, stageRoot]) assert.ok(path.isAbsolute(r), 'absolute coordinator staging paths required');
  assert.deepEqual(caseBook.source, SOURCE);
  assert.deepEqual(caseBook.artifacts.map(a => a.artifact), ARTIFACTS);
  return {
    kind: 'preparation-not-campaign-evidence', source: SOURCE,
    inventoryCommand: ['node', path.join(swRoot, 'scripts/h3-finetune-terminal.mjs'), 'list'],
    finalPair: 'Coordinator parameters AFTER full implementation/reviews, native-main merge and permanent pin; never these inspection anchors.',
    unresolvedInterfaces: caseBook.unresolvedInterfaces,
    cases: requiredCases().map(c => {
      const check = caseCheck(c);
      const scenario = path.join(stageRoot, `${c.id}.scenario.json`);
      const transcript = path.join(stageRoot, `${c.id}.api.json`);
      const apiCommand = ['node', path.join(swRoot, 'scripts/h3-finetune-terminal-api.mjs'), scenario, transcript];
      if(check==='generation-source-prep-cancel')return {...c,scenario,transcript,commands:[[process.execPath,path.join(swRoot,'scripts/h3-finetune-terminal-runtime.mjs'),'generation-source-prep-cancel',path.join(stageRoot,'campaign-plan.json'),c.id,swRoot,inferenceRoot]],note:'Managed install -> fresh owned Idle restart -> generation; cancel only after an actual partial ColdContentRead checkpoint, then bind the joined-teardown root message, raw trace, and owned resume to the declared source. Never a forged receipt or a completed generation.'};
      if (c.group === 'numeric') {
        const mechanism=check==='tensor-mapping'?'convrot':check;
        const native=nativeCommands(inferenceRoot,c.backend,mechanism), authority=caseBook.numeric[mechanism];
        return {...c,...native,referenceRevision:authority.referenceRevision,
          numericProof:{kind:'native-pinned-fixture',mechanism,package:c.backend==='mlx'?'mlx-gen-minimax-h3':'candle-gen-minimax-h3',tests:authority.selectors??[authority.selector],...(check==='tensor-mapping'?{supportingCommands:[[...cargo(swRoot,c.backend,['sceneworks-core',null,'minimax_h3_codec::tests::actual_native_bytes_normalize_through_persisted_plan_and_shared_cache_in_all_precisions']),'--exact']]}:{})},
          referenceCommand:referenceCommand(inferenceRoot,mechanism,path.join(stageRoot,`reference-${mechanism}`),path.join(stageRoot,c.id)),reference:authority,
          adapterCommand:['node',path.join(swRoot,'scripts/h3-finetune-terminal-runtime.mjs'),'numeric',path.join(stageRoot,'campaign-plan.json'),c.id,swRoot,inferenceRoot],
          referenceMeaning:'Native GPU assertions against the existing precomputed independent hash-bound CPU reference fixture; not upstream CUDA execution. No extra flat-vector export requirement.',
        };
      }
      const binding = c.group === 'numeric'
        ? ['sceneworks-core', 'minimax_h3_import', caseBook.numeric['tensor-mapping'].selector]
        : fixtureBindings[c.group]?.[check];
      return {
        ...c, ...(hasScenario(c) ? { scenario, transcript } : {}),
        commands: hasScenario(c) ? [apiCommand] : [cargo(swRoot, c.backend, binding)],
        ...(binding ? { supportingFixture: cargo(swRoot, c.backend, binding), fixtureInventory: cargo(swRoot, c.backend, binding, true) } : {}),
        ...(c.artifact ? { frozenArtifact: caseBook.artifacts.find(a => a.artifact === c.artifact) } : {}),
        note: c.group === 'lifecycle' ? 'Lifecycle transition steps below plus source-bound fixture; restart/mutation/interrupt require task-owned external service/filesystem orchestration. Generation alone is NOT lifecycle acceptance.' : c.group === 'generalization' ? 'The actual staged real source must exercise this format/lineage. Supporting synthetic fixture alone is NOT real-model generalization.' : 'Actual app integration through existing API; requires staged scenario bindings and actual runtime receipt, not only HTTP success.',
      };
    }),
  };
}

export function assertLicenseAuthority(e, {allowUnitFake=false}={}) {
  const authority=e.licenseAuthority;
  assert.ok(authority?.acknowledged===true && authority.approvedBy && authority.reference && Number.isFinite(Date.parse(authority.approvedAt)), 'explicit source and inherited-component license acknowledgment authority required');
  assert.ok(authority.kind==='user-approved' || (allowUnitFake && authority.kind==='unit_fake'), 'synthetic consent is not real legal approval');
  const source=e.family==='ltx'?e.ltxAdapter:e.model;
  for(const [role,inputs] of [['source',[source]],['component',e.model.components??[]],['adapter',e.adapters??[]]])for(const input of inputs){hex(input?.sha256);if(role!=='source')assert.equal(input.licenseAcknowledged,true,'explicit inherited component/adapter acknowledgment required before any import');assert.ok(authority.subjects?.some(s=>s.role===role&&s.sha256===input.sha256),`missing explicit ${role} license authority for actual input`);}
  assert.ok(e.model.components?.length,'inherited component authority is required even before canceled/interrupted import');
  return authority;
}
function validateExecution(e, c, allowUnitFake=false) {
  assertLicenseAuthority(e,{allowUnitFake});
  assert.ok(e.label && /^[a-zA-Z0-9_-]+$/.test(e.label));
  if(e.family==='ltx') {
    assert.equal(caseCheck(c),'ltx-add-select-generate');
    assert.ok(e.model?.modelId?.startsWith('ltx_') && e.model.modelId===e.ltxAdapter?.baseModel,'existing compatible LTX base selection required, not a full-checkpoint import');
    assert.ok(e.ltxAdapter.id && absoluteSourcePath(e.ltxAdapter.sourcePath));hex(e.ltxAdapter.sha256);
  } else {
    assert.ok(e.model?.modelId?.startsWith('imported_'), 'explicit non-stock imported modelId required');
    assert.ok(absoluteSourcePath(e.model.sourcePath), 'staged absolute platform-native source path required'); hex(e.model.sha256);
  }
  if (e.model.sourceKind === 'directory' || e.model.sourceFormat === 'sharded-diffusers') {
    assert.equal(e.model.sourceKind, 'directory', 'sharded sources are directories, never a fictional aggregate-hashed file');
    assert.ok(e.model.files?.length > 0, 'directory source closure requires each actual file/hash');
    for (const file of e.model.files) { assert.ok(absoluteSourcePath(file.path)); hex(file.sha256); }
    assert.ok(e.model.files.some(f => f.sha256 === e.model.sha256), 'primary payload identity must be an actual source file, not a tree stamp');
  }
  if (c.artifact && e.identityControl?.kind!=='different-bytes-negative') {
    assert.equal(e.model.sha256, caseBook.artifacts.find(a => a.artifact === c.artifact).expectedSha256);
    assert.equal(e.model.partition, 'ref2va');
  }
  // Preload import cancellation/retry has a source identity, not a fictional native load.
  if (['cancel', 'interruption-retry'].includes(caseCheck(c))) return;
  assert.ok(knownModes.includes(e.mode));
  const check = caseCheck(c);
  const requiredMode = c.artifact ? 'reference_to_video' : c.group === 'capability' ? ({ text: 'text_to_video', image: 'image_to_video', 'last-frame-only': 'image_to_video', 'first-last-frame': 'first_last_frame', video: 'reference_to_video', 'ordered-multimodal': 'reference_to_video', 'audio-reference': 'reference_to_video' }[check]) : undefined;
  if (requiredMode) assert.equal(e.mode, requiredMode, 'case must exercise its actual API operation');
  if (check === 'ltx-add-select-generate') assert.equal(e.family, 'ltx', 'LTX parity must run LTX, not an H3 scenario with an LTX label');
  if (check === 'independent-finetune') {
    assert.ok(e.model.lineage?.source && e.model.lineage?.revision, 'independently published compatible fine-tune/merge provenance required; no new training is implied');
    assert.ok(!caseBook.artifacts.some(a => a.expectedSha256 === e.model.sha256), 'renamed Singularity is not independent generalization');
  }
  const format = { 'native-single': 'native-single', 'sharded-diffusers': 'sharded-diffusers', 'dense-f16': 'f16', 'dense-bf16': 'bf16', 'dense-f32': 'f32' }[check];
  if (format) assert.equal(e.model.sourceFormat, format, 'case must use its actual staged source format, not merely its label');
  if (check === 'renamed') {
    assert.ok(e.model.originalPath && e.model.originalPath !== e.model.sourcePath, 'supply original and actual renamed staged path');
    if(e.identityControl?.kind==='different-bytes-negative'){assert.ok(!e.reuseInstalled&&e.family!=='ltx'&&e.model.sourceKind!=='directory','different-bytes checksum challenge must import an actual owned single file, never reuse or generate');hex(e.identityControl.expectedSourceSha256);assert.equal(e.identityControl.expectedSourceSha256,e.model.originalSha256);assert.notEqual(e.model.sha256,e.model.originalSha256,'negative control must actually have different bytes');}
    else assert.equal(e.model.originalSha256, e.model.sha256, 'renaming must preserve actual contents');
  }
  const bases = { fl2va: 'minimax_h3', ref2va: 'minimax_h3_ref', hybrid: 'minimax_h3' };
  if (e.family !== 'ltx') {
    assert.equal(e.model.baseModel, bases[e.model.partition], 'explicit compatible base/partition pair required');
    assert.ok(['full', 'pruned'].includes(e.model.topology), 'topology and operation partition are separate: hybrid is not a pruning topology');
    if (e.model.partition === 'hybrid') {
      if (e.model.hybridBinding?.kind === 'single-checkpoint') assert.ok(e.model.hybridBinding.declarationSource, 'single hybrid checkpoint requires explicit source declaration, not shape inference');
      else { assert.ok(e.model.companion?.source, 'explicit hybrid checkpoint declaration or companion lineage required'); hex(e.model.companion?.sha256); }
    }
    assert.ok(e.model.supportedModes?.includes(e.mode), 'operation must come from the resolved imported contract');
    assert.ok(e.model.partition === 'hybrid' || (e.mode === 'reference_to_video' ? e.model.partition === 'ref2va' : e.model.partition === 'fl2va'), 'declared partition does not support this operation');
  }
  const s = e.sampling;
  assert.ok(s && s.contractSource && s.steps >= 2 && Number.isInteger(s.steps), 'declared trained NFE recipe required, not cheap 1-step');
  for (const k of ['width', 'height', 'fps', 'duration']) assert.ok(Number.isFinite(s[k]) && s[k] > 0, `declared ${k} required`);
  assert.ok(['q4', 'q8', 'bf16'].includes(e.model.componentTier));
  if (e.family !== 'ltx' && s.advanced?.mlxQuantize !== undefined) assert.equal(Number(s.advanced.mlxQuantize), { bf16: 0, q4: 4, q8: 8 }[e.model.componentTier], 'explicit shared precision must match bound component tier; never silently demote');
  assert.ok(e.model.components?.length > 0 && e.model.components.every(v => {
    hex(v.sha256); return v.source && v.revision && v.licenseAcknowledged === true;
  }), 'actual pinned licensed component closure required');
  for (const a of e.adapters ?? []) {
    hex(a.sha256); assert.ok(a.id && a.source && a.revision && a.compatibilitySource && a.licenseAcknowledged === true);
    assert.equal(a.topology, e.model.topology, 'adapter topology must match source, not just dimensions');
    assert.equal(a.partition, e.model.partition);
    const recipe=caseBook.sampling.turbo.find(r=>r.id===a.id);
    if(recipe){for(const key of ['steps','schedulerShift','audioSchedulerShift'])assert.equal(s[key],recipe[key],`selected accelerator must use its declared trained ${key}`);assert.equal(Math.min(s.width,s.height),recipe.trainingShortEdge);}
  }
}
function conditioning(check, assets) {
  const need = k => { assert.ok(typeof assets[k] === 'string' && assets[k].length > 0, `staged project asset required: ${k}`); return assets[k]; };
  switch (check) {
    case 'text': return [{}, { prompt: 'A blue cube slides left across a white table, synchronized soft impacts.' }];
    case 'image': return [{ sourceAssetId: need('imageA') }, { sourceAssetId: need('imageB') }];
    case 'last-frame-only': return [{ lastFrameAssetId: need('imageB') }, { lastFrameAssetId: need('imageA') }];
    case 'first-last-frame': return [{ sourceAssetId: need('imageA'), lastFrameAssetId: need('imageB') }, { sourceAssetId: need('imageB'), lastFrameAssetId: need('imageA') }];
    case 'video': return [{ sourceClipAssetIds: [need('clipA'), need('clipB')] }, { sourceClipAssetIds: [need('clipB'), need('clipA')] }, { sourceClipAssetIds: [need('clipA')] }];
    case 'audio-reference': return [{ referenceAssetIds: [need('imageA')], referenceAudioAssetIds: [need('voiceA')] }, { referenceAssetIds: [need('imageA')], referenceAudioAssetIds: [need('voiceB')] }, { referenceAssetIds: [need('imageA')], referenceAudioAssetIds: [] }];
    case 'ordered-multimodal': {
      const b = { referenceAssetIds: [need('imageA'), need('imageB')], sourceClipAssetIds: [need('clipA'), need('clipB')], referenceAudioAssetIds: [need('voiceA'), need('voiceB')] };
      return [b, ...Object.keys(b).flatMap(k => [{ ...b, [k]: [...b[k]].reverse() }, { ...b, [k]: [] }])];
    }
    default: return [{}];
  }
}
function assertCoverage(c, executions) {
  const check = caseCheck(c);
  if(check==='renamed')assert.ok(executions.length===2&&executions.filter(e=>e.identityControl?.kind==='different-bytes-negative').length===1,'renamed/content identity requires one actual copied source and one different-bytes refusal control');
  if (['turbo', 'lora', 'lokr'].includes(check)) {
    for (const cell of ['full', 'pruned', 'hybrid']) assert.ok(executions.some(e => (e.model.partition === 'hybrid' ? 'hybrid' : e.model.topology) === cell), `missing supported ${check}/${cell} cell`);
    if (check === 'turbo') for (const r of caseBook.sampling.turbo) {
      const selected = executions.filter(e => e.adapters?.some(a => a.id === r.id));
      assert.ok(selected.length > 0, `missing declared recipe ${r.id}`);
      for (const e of selected) {
        for (const k of ['steps', 'schedulerShift', 'audioSchedulerShift']) assert.equal(e.sampling[k], r[k], `trained ${r.id}/${k}`);
        assert.equal(Math.min(e.sampling.width, e.sampling.height), r.trainingShortEdge, 'trained turbo geometry required');
      }
    }
    for (const e of executions) assert.ok(e.adapters?.length > 0 && (check === 'turbo' || e.adapters.some(a => a.kind === check)), `missing selected ${check}`);
  }
  if (check === 'hybrid') {
    for (const mode of ['text_to_video', 'image_to_video', 'reference_to_video']) assert.ok(executions.some(e => e.mode === mode && e.model.partition === 'hybrid'), `hybrid must exercise ${mode}`);
    for (const e of executions) assert.ok((e.model.hybridBinding?.kind === 'single-checkpoint' && e.model.hybridBinding.declarationSource) || (e.model.companion?.source && e.model.companion.sha256), 'explicit single-checkpoint hybrid declaration or companion lineage required');
  }
}
export function reducedConditioning(c, assets) {
  const required=k=>{const value=assets[k];assert.ok(typeof value==='string'&&value.length||value&&typeof value.$capture==='string'&&value.$capture.length&&Object.keys(value).length===1,`staged project asset or exact owned API capture required: ${k}`);return clone(value);};
  let value;
  if(c.conditioning==='ordered-mixed')value={referenceAssetIds:[required('imageA'),required('imageB')],sourceClipAssetIds:[required('clipA'),required('clipB')],referenceAudioAssetIds:[required('voiceA'),required('voiceB')]};
  else if(c.conditioning==='reference')value={referenceAssetIds:[required('imageA')],referenceAudioAssetIds:[required('voiceA')]};
  else if(c.conditioning==='image')value={sourceAssetId:required('imageA')};
  else if(c.conditioning==='last-frame-only')value={lastFrameAssetId:required('imageB')};
  else if(c.conditioning==='first-last-frame')value={sourceAssetId:required('imageA'),lastFrameAssetId:required('imageB')};
  else value={};
  const changes={'real-drop-images':['referenceAssetIds','drop'],'real-drop-video':['sourceClipAssetIds','drop'],'real-drop-audio':['referenceAudioAssetIds','drop'],'real-reverse-images':['referenceAssetIds','reverse'],'real-reverse-video':['sourceClipAssetIds','reverse'],'real-reverse-audio':['referenceAudioAssetIds','reverse']};
  if(changes[c.slug]){const [key,operation]=changes[c.slug];value[key]=operation==='drop'?[]:[...value[key]].reverse();}
  if(c.slug==='real-hybrid-prompt-change')value.prompt='A blue cube slides left across a white table, synchronized soft impacts.';
  return value;
}
function assertReducedBindings(c,p,executions) {
  assert.ok(executions?.length,'actual staged execution binding required');
  const renamed=c.index===4;
  assert.equal(executions.length,renamed?2:1,'one AV submission per reduced case; only renamed identity adds an import-only negative');
  if(renamed)assertCoverage({...c,id:`${c.backend}.generalization.renamed`},executions);
  for(const e of executions){
    if(e.identityControl?.kind==='different-bytes-negative')continue;
    const artifact=/^artifact\.(\d)-/.exec(c.modelKey);if(artifact)assert.equal(e.model.sha256,caseBook.artifacts[Number(artifact[1])].expectedSha256,'reduced binding must use its frozen content identity');
    const expectedMode=({'text':'text_to_video','image':'image_to_video','last-frame-only':'image_to_video','first-last-frame':'first_last_frame','reference':'reference_to_video','ordered-mixed':'reference_to_video'})[c.conditioning];
    if(expectedMode)assert.equal(e.mode,expectedMode,'reduced surface must exercise its declared API operation');
    if(c.index>=15&&c.index<=20||c.index===22){assert.equal(e.model.sha256,caseBook.stagingAssets.hybrid[0].expectedSha256,'use the frozen actual b25 hybrid source');assert.equal(e.model.partition,'hybrid');assert.equal(e.model.hybridBinding?.kind,'single-checkpoint','actual one-checkpoint hybrid declaration required');}
    if(c.index===21){assert.equal(e.model.sha256,caseBook.stagingAssets.pulpcut[0].expectedSha256,'use the independently published frozen PulpCut merge');assert.equal(e.model.partition,'fl2va');assert.ok(e.model.lineage?.source&&e.model.lineage?.revision);}
    if(renamed){assert.equal(e.ownershipMode,'linked');assert.equal(e.model.sourceFormat,'native-single');}
    const adapters=e.adapters??[],accelerators=adapters.filter(a=>a.role==='accelerator'||caseBook.sampling.turbo.some(r=>r.id===a.id));
    if(['ref-turbo-baseline','lokr+ref-turbo-baseline','lokr-repeat','drop-lokr'].includes(c.adapterRecipe))assert.ok(accelerators.some(a=>a.id==='minimax_h3_ref2v_turbo_4step'),'declared RefTurbo accelerator must actually be selected');
    if(['lokr+ref-turbo-baseline','lokr-repeat'].includes(c.adapterRecipe))assert.ok(adapters.some(a=>a.kind==='lokr'),'actual LoKr must be selected');
    if(c.adapterRecipe==='drop-lokr')assert.ok(!adapters.some(a=>a.kind==='lokr'),'drop only the LoKr, retaining accelerator');
    if(c.adapterRecipe==='drop-lora')assert.equal(adapters.length,0,'LoRA drop must remove the selected adapter while retaining the trained sampling recipe');
    const recipe={'fl-turbo-4v01':'minimax_h3_turbo_4step_v01','fl-turbo-8':'minimax_h3_turbo_8step','fl-turbo-4v1.0-768':'minimax_h3_turbo_4step_768p'}[c.adapterRecipe];
    if(recipe){const declared=caseBook.sampling.turbo.find(r=>r.id===recipe);assert.ok(declared,`unknown frozen recipe ${recipe}`);assert.ok(adapters.some(a=>a.id===declared.id),'select the declared published adapter, not just its step count');}
  }
}
export function buildScenario(c, p, executions) {
  const reduced=operationalCaseById(c.id);
  assert.ok(reduced || requiredCases().some(expected => expected.id === c.id));
  if(reduced){for(const key of ['kind','index','slug','modelKey','conditioning','adapterRecipe'])assert.deepEqual(c[key],reduced[key],`reduced case reclassification: ${key}`);assertReducedBindings(c,p,executions);}
  assert.ok(executions?.length > 0, 'actual staged execution binding required');
  assert.ok(p.projectId && Number.isSafeInteger(p.seed) && p.seed>=0 && typeof p.requestedGpu === 'string' && p.requestedGpu.length > 0);
  assert.ok(p.timeoutSeconds > 0 && p.timeoutSeconds <= 7200);
  const u = new URL(p.baseUrl); assert.ok(['127.0.0.1', '[::1]', 'localhost'].includes(u.hostname));
  if(!reduced)assertCoverage(c, executions);
  if (c.group === 'capability') {
    assert.ok(p.sensitivity?.minimumControlDelta > 0 && Number.isFinite(p.sensitivity.minimumControlDelta), 'predeclared decoded-media intervention threshold required');
    assert.ok(Number.isFinite(p.sensitivity.maximumBaselineDelta) && p.sensitivity.maximumBaselineDelta >= 0, 'predeclared nondeterministic baseline bound required');
  }
  const check = caseCheck(c);
  assert.ok(hasScenario(c), 'this case uses its source-bound focused fixture, not a fake generation-only lifecycle scenario');
  if (check === 'managed-linked-equivalence') {
    assert.ok(executions.some(e => e.ownershipMode === 'linked') && executions.some(e => e.ownershipMode !== 'linked'), 'both managed and linked executions required');
    assert.equal(new Set(executions.map(e => e.model.sha256)).size, 1, 'equivalent ownership must use the same source bytes');
    assert.equal(executions.length,2,'one managed/linked controlled pair required');assert.equal(new Set(executions.map(e=>e.model.modelId)).size,2,'ownership comparison must install two distinct model selections');
    for(const key of ['maximumBaselineDelta','maximumVideoDelta','maximumAudioDelta'])assert.ok(Number.isFinite(p.equivalence?.[key])&&p.equivalence[key]>=0,`explicit ownership nondeterminism bound ${key} required`);
  }
  if (check === 'cold-warm-staged') {
    for (const label of ['cold', 'warm', 'staged']) assert.ok(executions.some(e => e.label === label), `missing ${label} execution; stage/cache state must also be observed, not inferred from the label`);
    const [cold,warm,staged]=['cold','warm','staged'].map(label=>executions.find(e=>e.label===label));
    assert.ok(!cold.reuseInstalled&&warm.reuseInstalled&&staged.reuseInstalled,'warm/staged reuse the actual first installed source');assert.equal(new Set([cold,warm,staged].map(e=>e.model.modelId)).size,1);
    assert.ok(staged.memoryCap && staged.memoryCap.kind===(c.backend==='cuda'?'cuda-vram-gb':'mlx-memory-gb') && Number.isFinite(staged.memoryCap.value) && staged.memoryCap.value>0,'staged run needs an explicit actual worker memory cap, not an invented advanced flag');
  }
  const steps = clone(p.setupSteps ?? []), comparisons = [], lifecycleActions = [];
  for (const e of executions) {
    validateExecution(e, c, p.proof_kind==='unit_fake');
    const prefix = e.label;
    if(e.family==='ltx') {
      const a=e.ltxAdapter;
      steps.push({id:`${prefix}_adapter_import`,method:'POST',path:'/api/v1/loras/import',body:{loraId:a.id,name:a.name??a.id,sourcePath:a.sourcePath,expectedSha256:a.sha256,family:'ltx-video',baseModel:a.baseModel,scope:'project',projectId:p.projectId,licenseAcknowledged:e.licenseAuthority.acknowledged},expectedStatus:201});
      steps.push({id:`${prefix}_adapter_installed`,waitJob:cap(`${prefix}_adapter_import/id`),assert:[eq('/status','completed')]});
      steps.push({id:`${prefix}_adapter_selected`,method:'GET',path:`/api/v1/loras?projectId=${encodeURIComponent(p.projectId)}`,expectedStatus:200,assert:[{pointer:'',contains:{id:a.id,baseModel:a.baseModel}}]});
    }
    const importBody = { modelId: e.model.modelId, name: `sc26691 ${prefix}`, sourcePath: e.model.sourcePath, expectedSha256: e.model.sha256, baseModel: e.model.baseModel, licenseAcknowledged: e.licenseAuthority.acknowledged };
    if(e.importSource){assert.equal(e.importSource.kind,'huggingFace');assert.notEqual(e.ownershipMode,'linked','choose the actual managed HF or approved linked source, not contradictory locators');assert.match(e.importSource.revision??'',/^[a-f0-9]{40}$/);importBody.source=clone(e.importSource);delete importBody.sourcePath;if(c.artifact){assert.equal(e.importSource.repo,SOURCE.repo);assert.equal(e.importSource.revision,SOURCE.revision);assert.deepEqual(e.importSource.files,[c.artifact]);}}
    if(e.identityControl?.kind==='different-bytes-negative'){assert.equal(check,'renamed');importBody.expectedSha256=e.identityControl.expectedSourceSha256;}
    if (e.model.sourceKind === 'directory') delete importBody.expectedSha256; // No invented directory digest; retain per-file closure in preparation plan.
    if (e.family !== 'ltx') importBody.minimaxH3Partition = e.model.partition;
    if (e.ownershipMode === 'linked') {
      assert.ok(e.linkedRootId && e.linkedRelativePath);
      delete importBody.sourcePath;
      Object.assign(importBody, { linkedRootId: e.linkedRootId, linkedRelativePath: e.linkedRelativePath });
    }
    if (check === 'restart-replay' && e.replayJobId) {
      assert.ok(e.reuseInstalled, 'explicit phased replay requires an actual previously owned persisted job');
      steps.push({ id: `${prefix}_original`, method: 'GET', path: `/api/v1/jobs/${encodeURIComponent(e.replayJobId)}`, expectedStatus: 200, assert: [eq('/status', 'completed'), eq('/payload/model', e.model.modelId)] });
      steps.push({ id: `${prefix}_replay`, method: 'POST', path: `/api/v1/jobs/${encodeURIComponent(e.replayJobId)}/duplicate`, body: {}, expectedStatus: 201, assert: [eq('/payload/model', e.model.modelId), eq('/payload/seed', p.seed)] });
      steps.push({ id: `${prefix}_replayed`, waitJob: cap(`${prefix}_replay/id`), assert: [eq('/status', 'completed'), eq('/payload/model', e.model.modelId)] });
      continue; // Actual process restart is an external precondition, never a fabricated HTTP call.
    }
    if (!e.reuseInstalled && e.family!=='ltx') {
      steps.push({ id: `${prefix}_import`, method: 'POST', path: '/api/v1/models/import', body: importBody, expectedStatus: 201, assert: [eq('/payload/modelId', e.model.modelId)] });
      if(e.identityControl?.kind==='different-bytes-negative'){
        steps.push({id:`${prefix}_refused`,waitJob:cap(`${prefix}_import/id`),expectedJobStatus:'failed',assert:[eq('/status','failed'),eq('/payload/expectedSha256',e.identityControl.expectedSourceSha256),{pointer:'/error',includesText:'integrity check'},{pointer:'/error',includesText:e.model.sha256},{pointer:'/error',includesText:e.identityControl.expectedSourceSha256}]});continue;
      }
      if (['cancel', 'interruption-retry'].includes(check)) {
        if(check==='interruption-retry')lifecycleActions.push({kind:'interrupt-import',beforeStep:`${prefix}_cancel`,jobCapture:`${prefix}_import/id`});
        steps.push({ id: `${prefix}_cancel`, method: 'POST', path: route('/api/v1/jobs/', cap(`${prefix}_import/id`), '/cancel'), expectedStatus: 200 });
        steps.push({ id: `${prefix}_cancelled`, waitJob: cap(`${prefix}_import/id`), expectedJobStatus: 'canceled', assert: [eq('/status', 'canceled')] });
        if (check === 'interruption-retry') {
          steps.push({ id: `${prefix}_retry`, method: 'POST', path: route('/api/v1/jobs/', cap(`${prefix}_import/id`), '/retry'), body: {}, expectedStatus: 201, assert: [eq('/payload/modelId', e.model.modelId)] });
          steps.push({ id: `${prefix}_retried`, waitJob: cap(`${prefix}_retry/id`), assert: [eq('/result/modelId', e.model.modelId)] });
        }
        continue; // No successful native load is asserted for preload cancellation/refusal.
      }
      steps.push({ id: `${prefix}_installed`, waitJob: cap(`${prefix}_import/id`), assert: [eq('/result/modelId', e.model.modelId)] });
    }
    const requestedCheck = c.group === 'capability' && !['turbo', 'lora', 'lokr', 'hybrid'].includes(check) ? check : e.conditioning ?? (e.mode === 'reference_to_video' ? 'audio-reference' : e.mode === 'first_last_frame' ? 'first-last-frame' : e.mode === 'image_to_video' ? 'image' : 'text');
    const selectedAdapters = (e.family==='ltx'?[e.ltxAdapter]:e.adapters ?? []).map(a => ({ id: a.id, weight: a.weight ?? 1 }));
    const variants = reduced?[reducedConditioning(c,p.assets??{})]:conditioning(requestedCheck, p.assets ?? {});
    if (['turbo', 'lora', 'lokr'].includes(check)) {
      // Hold checkpoint, seed, NFE, shifts and geometry fixed; remove ONLY adapters.
      const accelerator=a=>a.role==='accelerator'||caseBook.sampling.turbo.some(r=>r.id===a.id);
      let targets=(e.adapters??[]).filter(a=>check==='turbo'?accelerator(a):a.kind===check&&!accelerator(a));
      if(check==='lora'&&!targets.length)targets=(e.adapters??[]).filter(a=>a.kind==='lora');
      assert.ok(targets.length,'selected adapter mechanism must have a real controlled target');
      const removed=new Set(targets.map(a=>a.id));
      variants.splice(1, variants.length, { ...variants[0], loras: selectedAdapters.filter(a=>!removed.has(a.id)) });
      variants[0] = { ...variants[0], loras: selectedAdapters };
    }
    // Real repeated-baseline floor, then one mechanism intervention at a time.
    const variantsToRun = check==='managed-linked-equivalence'?[variants[0],variants[0]]:c.group === 'capability' ? [variants[0], variants[0], ...variants.slice(1)] : [variants[0]];
    for (const [i, variant] of variantsToRun.entries()) {
      const id = `${prefix}_av${i}`, s = e.sampling;
      if(check==='cold-warm-staged'&&e.label==='staged')lifecycleActions.push({kind:'restart',beforeStep:id,memoryCap:clone(e.memoryCap)});
      const body = {
        projectId: p.projectId, model: e.reuseInstalled || e.family==='ltx' ? e.model.modelId : cap(`${prefix}_installed/result/modelId`), mode: e.mode,
        prompt: e.prompt ?? (['audio-reference','ordered-multimodal'].includes(requestedCheck) ? 'A person faces the camera and says, "We are glad to welcome you today and share a clear account of the work we have finished." Natural continuous speech, steady camera, synchronized voice audio.' : 'A red cube slides right across a white table, synchronized soft impacts.'), seed: p.seed,
        requestedGpu: p.requestedGpu, duration: s.duration, fps: s.fps, width: s.width, height: s.height,
        advanced: { ...(s.advanced ?? {}), ...(e.family === 'ltx' ? {} : { mlxQuantize: { bf16: 0, q4: 4, q8: 8 }[e.model.componentTier] }), steps: s.steps, ...(s.schedulerShift === undefined ? {} : { schedulerShift: s.schedulerShift }), ...(s.audioSchedulerShift === undefined ? {} : { audioSchedulerShift: s.audioSchedulerShift }) },
        loras: selectedAdapters, ...variant,
      };
      steps.push({ id, method: 'POST', path: '/api/v1/video/jobs', body, expectedStatus: 201,
        assert: [eq('/payload/model', e.model.modelId), eq('/payload/seed', p.seed), eq('/payload/mode', e.mode), eq('/payload/advanced/steps',s.steps), eq('/payload/width',s.width), eq('/payload/height',s.height), eq('/payload/fps',s.fps), eq('/payload/duration',s.duration)] });
      if (check === 'generation-source-prep-cancel') {
        // Cold memo: a fresh owned Idle restart AFTER installation, BEFORE generation. The
        // generation is never waited to completion; the action cancels during preparation.
        assert.equal(variantsToRun.length, 1, 'cold source-preparation cancellation accepts exactly one actual generation');
        assert.ok(!e.reuseInstalled && e.family !== 'ltx' && e.ownershipMode!=='linked', 'a managed imported installation is required; linked/reused sources cannot prove cold preparation');
        lifecycleActions.push({ kind: 'restart', beforeStep: id });
        continue;
      }
      steps.push({ id: `${id}_done`, waitJob: cap(`${id}/id`), assert: [eq('/status', 'completed'), eq('/payload/model', e.model.modelId)] });
    }
    if (check === 'restart-replay') {
      assert.ok(!e.reuseInstalled, 'self-contained restart first creates its actual owned model/job');
      const id=`${prefix}_replay`;
      lifecycleActions.push({kind:'restart',beforeStep:id});
      steps.push({id,method:'POST',path:route('/api/v1/jobs/',cap(`${prefix}_av0/id`),'/duplicate'),body:{},expectedStatus:201,assert:[eq('/payload/model',e.model.modelId),eq('/payload/seed',p.seed)]});
      steps.push({id:`${id}_done`,waitJob:cap(`${id}/id`),assert:[eq('/status','completed'),eq('/payload/model',e.model.modelId)]});
    }
    if (check === 'removal') steps.push({ id: `${prefix}_removed`, method: 'DELETE', path: `/api/v1/models/${encodeURIComponent(e.model.modelId)}?permanent=true`, expectedStatus: 200, assert: [eq('/id', e.model.modelId), eq('/removedManifestEntry', true)] });
    if (c.group === 'capability') comparisons.push({ label: prefix, baseline: `${prefix}_av0_done`, repeatedBaseline: `${prefix}_av1_done`, interventions: variantsToRun.slice(2).map((v, i) => ({ job: `${prefix}_av${i + 2}_done`, allowedChanges: [...new Set([...Object.keys(variants[0]),...Object.keys(v)])].filter(k=>JSON.stringify(variants[0][k])!==JSON.stringify(v[k])), stream: check === 'audio-reference' || JSON.stringify(v.referenceAudioAssetIds) !== JSON.stringify(variants[0].referenceAudioAssetIds) ? 'audio' : 'video' })), stream: check === 'audio-reference' ? 'audio' : 'video', thresholds: clone(p.sensitivity), command: ['node', 'scripts/h3-finetune-terminal.mjs', 'compare-control', 'BASELINE_AV_FROM_ACTUAL_JOB', 'REPEATED_AV_FROM_ACTUAL_JOB', 'INTERVENTION_AV_FROM_ACTUAL_JOB', check === 'audio-reference' ? 'audio' : 'video', String(p.sensitivity.minimumControlDelta), String(p.sensitivity.maximumBaselineDelta)], policy: 'Bind checkpoint/source closure + seed + complete sampling/config; decode both AV streams. Measure repeated-baseline RMSE B. Predeclare intervention threshold T and require intervention RMSE > max(T,3*B). If noise floor is unstable, acceptance is unresolved, not any-hash-difference success.' });
  }
  if (check === 'generation-source-prep-cancel') {
    assert.equal(executions.length, 1, 'one cold source-preparation attempt');
    const prefix = executions[0].label, generation = `${prefix}_av0`;
    assert.ok(lifecycleActions.some(a => a.kind === 'restart' && a.beforeStep === generation), 'fresh owned Idle restart required before the cold generation');
    // The action itself performs the cooperative POST /cancel only after the actual partial
    // ColdContentRead checkpoint, then awaits the joined-teardown root marker and verification.
    lifecycleActions.push({ kind: 'source-cancel', beforeStep: `${prefix}_cancelled`, generationCapture: `${generation}/id`, modelCapture: `${prefix}_installed/result/modelId` });
    steps.push({ id: `${prefix}_cancelled`, waitJob: cap(`${generation}/id`), expectedJobStatus: 'canceled', assert: [eq('/status', 'canceled')] });
  }
  assert.equal(new Set(steps.map(s => s.id)).size, steps.length, 'execution labels must be unique');
  const positiveIdentity=check==='renamed'?executions.find(e=>e.identityControl?.kind!=='different-bytes-negative'):undefined,negativeIdentity=check==='renamed'?executions.find(e=>e.identityControl?.kind==='different-bytes-negative'):undefined;
  if(positiveIdentity)assert.ok(absoluteSourcePath(p.identificationControlReceipt),'actual staged identity-control receipt path required');
  const identificationControl=positiveIdentity?{positive:positiveIdentity.label,negative:negativeIdentity.label,receipt:p.identificationControlReceipt,originalPath:positiveIdentity.model.originalPath,expectedSha256:positiveIdentity.model.sha256}:undefined;
  const ownershipEquivalence=check==='managed-linked-equivalence'?{managed:executions.find(e=>e.ownershipMode!=='linked').label,linked:executions.find(e=>e.ownershipMode==='linked').label,thresholds:clone(p.equivalence)}:undefined;
  return { baseUrl: p.baseUrl, caseId: c.id, timeoutSeconds: p.timeoutSeconds, steps, comparisons, lifecycleActions,...(identificationControl?{identificationControl}:{}),...(check==='cold-warm-staged'?{cacheCoverage:{cold:'cold_av0_done',warm:'warm_av0_done',staged:'staged_av0_done'}}:{}),...(ownershipEquivalence?{ownershipEquivalence}:{}) };
}
// Deliberately not a run-case plan: no invented evidence.json or successful runtime metrics.
export async function prepareCases(parameters, destination) {
  assert.match(parameters.sceneworks ?? '', /^[a-f0-9]{40}$/);
  assert.match(parameters.inference ?? '', /^[a-f0-9]{40}$/);
  assert.ok(path.isAbsolute(destination));
  const inventory=parameters.inventory??'reduced-v1';
  const matrixDecision=parameters.matrixAnchors?chooseMatrixBackend({anchors:parameters.matrixAnchors}):parameters.matrixDecision;
  const plan = concretePlan(parameters.swRoot, parameters.inferenceRoot, destination,{inventory,matrixDecision});
  const scenarios = [];
  for (const c of plan.cases.filter(hasScenario)) {
    const executions = parameters.bindings?.[c.id];
    scenarios.push([c.id, buildScenario(c, { ...parameters, ...(parameters.backends?.[c.backend] ?? {}) }, executions)]);
  }
  // Exclusive directory: no overwrite/recovery of another campaign candidate.
  await mkdir(destination);
  for (const c of plan.cases.filter(c => c.group === 'numeric')) await mkdir(path.join(destination, c.id));
  const cases = plan.cases.map(c => ({ ...c, ...(parameters.bindings[c.id] ? { declaredSourceBindings: parameters.bindings[c.id].map(e => ({ label: e.label, model: e.model, family: e.family, licenseAuthority:e.licenseAuthority, ltxAdapter:e.ltxAdapter, identityControl:e.identityControl, historyBinding:e.historyBinding, importSource: e.importSource, mode: e.mode, sampling: e.sampling, adapters: e.adapters, ownershipMode: e.ownershipMode, linkedRootId: e.linkedRootId, linkedRelativePath: e.linkedRelativePath, reuseInstalled: e.reuseInstalled, replayJobId: e.replayJobId })) } : {}) }));
  await writeFile(path.join(destination, 'case-plan.json'), JSON.stringify({ ...plan, cases, responsibleTask: 'sc-26691', sceneworks: parameters.sceneworks, inference: parameters.inference }, null, 2) + '\n', { flag: 'wx' });
  for (const [id, scenario] of scenarios) await writeFile(path.join(destination, `${id}.scenario.json`), JSON.stringify(scenario, null, 2) + '\n', { flag: 'wx' });
  return { cases: plan.cases.length, scenarios: scenarios.length, status: 'prepared-not-executed', unresolvedInterfaces: plan.unresolvedInterfaces };
}
// Bind an already materialized casebook to ACTUAL staged input identities. No model bytes are read
// here; the authorized runner independently streams every declared input before/after execution.
// Inline route binder: content-addressed source paths and scenario bytes move together.
// License/attribution fields are copied unchanged; no acknowledgment is created here.
export async function bindStagedCampaignPlan(plan,{receiptPath,root,backend,identityControls,allowUnitFake=false}) {
  const bound=clone(plan),receipt=JSON.parse(await readFile(receiptPath,'utf8'));
  assert.equal(receipt.status,'completed');assert.ok(receipt.proof_kind==='frozen_stage_content'||allowUnitFake&&receipt.proof_kind==='unit_fake');
  bound.preparationState='bound';
  bound.assetStaging={root,backend,receipt:{path:receiptPath,sha256:await hashFile(receiptPath)}};
  if(!allowUnitFake)await verifiedStagingRoot(bound);else bound.proof_kind='unit_fake';
  const bySha=new Map(receipt.files.map(f=>{assert.equal(f.status,'completed');assert.equal(f.observedSha256,f.expectedSha256);assert.equal(f.actualBytes,f.expectedBytes);return [f.observedSha256,f.finalPath];}));
  const out=path.join(bound.output,'bound-terminal-inputs');await mkdir(out,{recursive:true});let boundInputs=0;
  for(const c of bound.cases.filter(c=>c.backend===backend)){
    const replacements=new Map();
    for(const input of c.inputs){const staged=bySha.get(input.sha256);if(staged){replacements.set(input.path,staged);boundInputs++;}}
    if(identityControls&&c.sourceVariant==='opaque-renamed-linked'){
      const positive=c.declaredSourceBindings.find(e=>e.identityControl?.kind!=='different-bytes-negative'),negative=c.declaredSourceBindings.find(e=>e.identityControl?.kind==='different-bytes-negative');
      assert.equal(positive.model.sha256,identityControls.renamed.actualSHA);assert.equal(negative.model.sha256,identityControls.differentBytes.actualSHA,'negative control requires its actual hash and explicit authority before it can bind');
      replacements.set(positive.model.sourcePath,identityControls.renamed.path);replacements.set(positive.model.originalPath,identityControls.sourcePath);replacements.set(negative.model.sourcePath,identityControls.differentBytes.path);
      const oldRoot=positive.model.sourcePath.slice(0,-positive.linkedRelativePath.length).replace(/[\\/]$/,'');assert.ok(oldRoot);replacements.set(oldRoot,root);replacements.set(positive.linkedRelativePath,path.relative(root,identityControls.renamed.path));
      const identityInput=c.inputs.find(i=>i.role==='identity-control-fixture');assert.ok(identityInput);replacements.set(identityInput.path,path.join(root,'fixtures','identity-controls.json'));
    }
    const rewrite=value=>typeof value==='string'?(replacements.get(value)??value):Array.isArray(value)?value.map(rewrite):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).map(([k,v])=>[k,rewrite(v)])):value;
    const declaredAuthority=JSON.stringify(c.declaredSourceBindings?.map(e=>e.licenseAuthority));
    const scenarioInput=c.inputs.find(i=>i.role==='scenario');
    if(scenarioInput){assert.equal(await hashFile(scenarioInput.path),scenarioInput.sha256,'unbound scenario changed');const scenario=rewrite(JSON.parse(await readFile(scenarioInput.path,'utf8'))),target=path.join(out,`${c.id}.scenario.json`);await writeFile(target,JSON.stringify(scenario,null,2)+'\n',{flag:'wx'});replacements.set(scenarioInput.path,target);scenarioInput.sha256=await hashFile(target);}
    const probeInput=c.inputs.find(i=>i.role==='actual-artifact-probe-spec');
    if(probeInput){assert.equal(await hashFile(probeInput.path),probeInput.sha256);const spec=rewrite(JSON.parse(await readFile(probeInput.path,'utf8')));const projection={schema:'sceneworks.h3.staging_receipt',schema_version:1,producer:'h3-finetune-stage-receipt projection',full_file_digest_verified:true,artifact_path:spec.artifact.path,artifact_sha256:spec.artifact.sha256,artifact_bytes:spec.artifact.bytes,parent_receipt_path:receiptPath,parent_receipt_sha256:bound.assetStaging.receipt.sha256};const file=receipt.files.find(f=>f.finalPath===spec.artifact.path&&f.observedSha256===spec.artifact.sha256);assert.ok(file&&file.actualBytes===spec.artifact.bytes,'probe artifact must match the actual verified staging record');const projectionPath=path.join(out,`${c.id}.staging-projection.json`);await writeFile(projectionPath,JSON.stringify(projection,null,2)+'\n',{flag:'wx'});spec.staging={receipt_path:projectionPath,receipt_sha256:await hashFile(projectionPath),verified_artifact_sha256:spec.artifact.sha256};c.inputs.push({path:projectionPath,sha256:spec.staging.receipt_sha256,role:'staging-projection'});const target=path.join(out,`${c.id}.probe-spec.json`);await writeFile(target,JSON.stringify(spec,null,2)+'\n',{flag:'wx'});replacements.set(probeInput.path,target);probeInput.sha256=await hashFile(target);}
    const rewritten=rewrite(c);Object.assign(c,rewritten);
    if(c.actualArtifactProbe)c.actualArtifactProbe.spec={path:c.inputs.find(i=>i.role==='actual-artifact-probe-spec').path,sha256:c.inputs.find(i=>i.role==='actual-artifact-probe-spec').sha256};
    for(const input of c.inputs.filter(i=>i.role==='identity-control-fixture'))input.sha256=await hashFile(input.path);
    c.inputs=c.inputs.filter(i=>i.role!=='staging-receipt');c.inputs.push({...bound.assetStaging.receipt,role:'staging-receipt'});
    assert.equal(JSON.stringify(c.declaredSourceBindings?.map(e=>e.licenseAuthority)),declaredAuthority,'binding cannot synthesize or change legal consent');
  }
  validatePlan(bound);return {plan:bound,boundInputs,blockedByAllowedRoots:0};
}
export async function finalizeCampaignPlan(parameters, preparedRoot) {
  const book=JSON.parse(await readFile(path.join(preparedRoot,'case-plan.json'),'utf8'));
  assert.equal(book.sceneworks,parameters.sceneworks);assert.equal(book.inference,parameters.inference);
  assert.ok(path.isAbsolute(parameters.output) && parameters.nextAuthorizedAction?.length>10);
  const filename=path.join(preparedRoot,'campaign-plan.json'),cases=[];
  for(const template of book.cases) {
    const c={...template,evidence:'evidence.json',timeoutSeconds:parameters.timeoutSeconds,inputs:clone(parameters.caseInputs?.[template.id]??[]),loadHostBudgetBytes:parameters.budgets?.[template.backend]?.hostBytes,loadDeviceBudgetBytes:parameters.budgets?.[template.backend]?.deviceBytes};
    if(c.kind==='full-av')c.sensitivity=clone(parameters.backends?.[c.backend]?.sensitivity??parameters.sensitivity);
    const cfg={...(parameters.runtime??{}),...(parameters.backends?.[c.backend]?.runtime??{}),...(parameters.caseRuntime?.[c.id]??{})};
    if(c.kind==='actual-artifact-probe'){
      c.actualArtifactProbe=clone(parameters.actualArtifactProbes?.[c.id]);assert.ok(c.actualArtifactProbe?.spec?.path,'actual selected-byte probe spec required');
      c.inputs.push({...c.actualArtifactProbe.spec,role:'actual-artifact-probe-spec'});c.command=c.commands[0];
    }
    else if(c.blockedPrerequisite){c.command=c.commands[0];}
    else if(c.numericProof){
      const mechanism=c.numericProof.mechanism, fixture=path.join(parameters.inferenceRoot,caseBook.numeric[mechanism].fixture);
      c.inputs.push({path:fixture,sha256:await hashFile(fixture),role:'numeric-fixture'});
      assert.ok(c.inputs.some(i=>i.role==='reference'),'stage actual pinned external reference blobs for native arithmetic');
      c.runtime={deviceId:cfg.deviceId};c.command=['node',path.join(parameters.swRoot,'scripts/h3-finetune-terminal-runtime.mjs'),'numeric',filename,c.id,parameters.swRoot,parameters.inferenceRoot];
    } else if(c.scenario || cfg.scenario){
      c.scenario=cfg.scenario??c.scenario;
      c.inputs.push({path:c.scenario,sha256:await hashFile(c.scenario),role:'scenario'});
      for(const source of c.declaredSourceBindings??[]){
        assertLicenseAuthority(source,{allowUnitFake:parameters.proof_kind==='unit_fake'});
        assert.ok(source.family==='ltx'?c.inputs.some(i=>i.role==='adapter'&&i.sha256===source.ltxAdapter.sha256):c.inputs.some(i=>i.role==='checkpoint'&&i.sha256===source.model.sha256),'stage each selected actual checkpoint/adapter closure');
        for(const component of [...(source.model.components??[]),...(source.adapters??[])])assert.ok(c.inputs.some(i=>i.sha256===component.sha256),'stage each actual bound component/adapter');
      }
      if(caseCheck(c)==='renamed'){const spec=JSON.parse(await readFile(c.scenario,'utf8')).identificationControl;assert.ok(spec?.receipt);c.inputs.push({path:spec.receipt,sha256:await hashFile(spec.receipt),role:'identity-control-fixture'});}
      c.runtime={...cfg,scenario:c.scenario};c.command=['node',path.join(parameters.swRoot,'scripts/h3-finetune-terminal-runtime.mjs'),'api',filename,c.id,parameters.swRoot,parameters.inferenceRoot];
      const sensitivity=parameters.backends?.[c.backend]?.sensitivity??parameters.sensitivity;
      if(c.group==='capability'){c.minimumControlDelta=sensitivity.minimumControlDelta;c.controlStream=c.id.endsWith('.audio-reference')?'audio':'video';}
    } else {
      c.command=c.commands[0];c.fixtureProof={kind:'source-bound-assertions',deviceId:cfg.deviceId};
      assert.ok(c.inputs.length,'bind the existing source fixture/program inputs, not a fictional model load');
    }
    cases.push(c);
  }
  const plan={version:1,preparationState:parameters.inlineAssetEnsure===true?'awaiting-inline-assets':'bound',inventory:book.inventory,nativeEvidence:parameters.nativeEvidence,matrixDecision:book.matrixDecision,matrixAnchors:parameters.matrixAnchors,proofIndex:parameters.proofIndex??[],coverageDisposition:'explicit-unproven-required',source:SOURCE,sceneworks:parameters.sceneworks,inference:parameters.inference,output:parameters.output,responsibleTask:'sc-26691',nextAuthorizedAction:parameters.nextAuthorizedAction,cases,...(parameters.proof_kind==='unit_fake'?{proof_kind:'unit_fake'}:{})};validatePlan(plan);
  await writeFile(filename,JSON.stringify(plan,null,2)+'\n',{flag:'wx'});return {plan:filename,cases:cases.length,status:'bound-not-executed'};
}
