import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm, mkdir, readFile } from "node:fs/promises";
import { stripJsoncComments } from "./jsonc.mjs";
import { tmpdir } from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { FLUX3_REQUIRED_CANONICAL_ANCHORS, FLUX3_ARTIFACT_PREREQUISITE_COORDINATES, flux3ConversionRecipe, validateFlux3AnchorPrerequisites, flux3AnchorRequestDigest, FLUX3_RESOURCE_OBLIGATIONS, missingFlux3ResourceObligations, normalizeResourceCase, resourceCaseDigest, bindResourceCaseInputs, flux3Checkpoint, flux3Conversion, flux3PackageRelativePath, assertSupplementalResourceObservation } from "./flux3-resource-contract.mjs";
const sha = value => createHash("sha256").update(value).digest("hex");
function fixture() {
  return { schemaVersion: 1, caseId: "candle.base.reference10.q4", scenario: "reference10", backend: "candle", modelId: "flux3_klein_base", tier: "q4", generation: { prompt: "move the red box", width: 1024, height: 1024 }, references: Array.from({length:10}, (_, index) => ({ instanceId: `attachment-${index}`, assetId: "same-asset", index, path: "ref.png", sha256: sha("reference") })) };
}
async function anchorPrerequisiteFixture() {
  const read = name => readFile(new URL(`../../${name}`, import.meta.url), "utf8");
  return {
    models: JSON.parse(stripJsoncComments(await read("config/manifests/builtin.models.jsonc"))).models,
    calibrationPlan: JSON.parse(await read("config/memory-calibration-plan.json")),
    artifactPrerequisites: JSON.parse(await read("docs/testing/sc-26243-terminal-artifact-prerequisites.json")),
  };
}
test("all sixteen REQUIRED canonical anchors are active published rows: 16 active / 0 blocked, measurement still unproven", async () => {
  const input = await anchorPrerequisiteFixture();
  const result = validateFlux3AnchorPrerequisites(input);
  assert.equal(result.requiredCanonicalAnchors, 16);
  assert.equal(result.activeCanonicalAnchors, 16);
  assert.deepEqual(result.blockedArtifactCoordinates, []);
  assert.deepEqual(result.activatedArtifactCoordinates.sort(), [...FLUX3_ARTIFACT_PREREQUISITE_COORDINATES].sort());
  assert.equal(result.terminalAcceptance, "NOT_PROVEN");
  assert.equal(result.measurementProven, false);
  assert.equal(result.executionAuthorized, false);
  assert.equal(new Set(FLUX3_REQUIRED_CANONICAL_ANCHORS).size, 16);
  assert.deepEqual(input.artifactPrerequisites.obligations, []);
  assert.equal(input.artifactPrerequisites.overallTerminalAcceptance, "NOT_PROVEN");
  assert.doesNotMatch(JSON.stringify(input.artifactPrerequisites), /HF403/, "the retired access blocker is not restated");
  for (const record of input.artifactPrerequisites.activated) {
    assert.ok(input.calibrationPlan.anchors[record.coordinate], `${record.coordinate} is an active canonical anchor`);
    assert.equal(record.measurement, "NOT_RUN");
  }
  const { loadProductTargets, terminalCatalog } = await import("../flux3-image-klein-terminal-plan.mjs");
  const catalog = terminalCatalog(await loadProductTargets());
  assert.equal(catalog.artifactPrerequisites.classification.terminalAcceptance, "NOT_PROVEN");
  assert.equal(catalog.artifactPrerequisites.obligations.length, 0);
  assert.ok(catalog.cases.filter(c => c.modelId === "flux3_klein_base" && ["q4", "q8"].includes(c.tier)).every(c => c.status === "NOT_RUN"),
    "published base q4/q8 cells are runnable-but-unrun, never PASS");
  const documentation = await readFile(new URL("../../docs/testing/sc-26243-terminal-case-plan.md", import.meta.url), "utf8");
  for (const record of input.artifactPrerequisites.activated) assert.ok(documentation.includes(record.coordinate));
  assert.ok(documentation.includes("Overall terminal acceptance remains NOT_PROVEN"));
  for (const flag of ["hosted_only=true", "run_memory_calibration=false", "run_five_rung_reference=false", "provision_qwen_snapshot=false", "provision_z_image_snapshot=false"]) assert.ok(documentation.includes(`-f ${flag}`));
  assert.ok(!documentation.includes('--ref "$BOOTSTRAP_REF" --json'), "do not recommend the proven-broken boolean-JSON CLI serialization");
});
test("each activated base row matches its genuine conversion receipt recipe, weights bytes and SHA", async () => {
  const input = await anchorPrerequisiteFixture();
  const base = input.models.find(m => m.id === "flux3_klein_base");
  for (const record of input.artifactPrerequisites.activated) {
    const platform = record.backend === "mlx" ? "macos" : "windows";
    const rows = base.downloads.filter(d => d.variant === record.tier && d.localDerivation && d.platforms.includes(platform));
    assert.equal(rows.length, 1, `${record.coordinate}: exactly one production row`);
    const [row] = rows;
    assert.equal(row.repo, record.repository); assert.equal(row.revision, record.resolvedRevision); assert.deepEqual(row.files, []);
    assert.equal(row.localDerivation.conversion, record.conversion);
    assert.equal(row.localDerivation.conversion, flux3Conversion(record.backend, record.tier));
    assert.deepEqual(record.receiptRecipe, flux3ConversionRecipe(record.backend, record.tier), "receipt recipe is the conversion id's own");
    assert.equal(row.localDerivation.weightsBytes, record.weightsBytes);
    assert.equal(row.localDerivation.weightsSha256, record.weightsSha256);
    assert.ok(row.footprint.diskSizeBytes > row.estimatedSizeBytes + record.weightsBytes, "disk = source + whole owned package");
    assert.deepEqual(row.platforms, record.backend === "mlx" ? ["macos"] : ["windows", "linux"]);
  }
});
test("a wrong SHA/size, recipe, missing record or unhonest status fails rather than activating", async () => {
  const input = await anchorPrerequisiteFixture();
  const row = (v, coordinate) => {
    const [modelId, tier, backend] = coordinate.split(":");
    return v.models.find(m => m.id === modelId).downloads.find(d => d.variant === tier && d.localDerivation &&
      d.platforms.includes(backend === "mlx" ? "macos" : "windows")).localDerivation;
  };
  for (const record of input.artifactPrerequisites.activated) {
    for (const [mutate, pattern] of [
      [v => { row(v, record.coordinate).weightsSha256 = "0".repeat(64); }, /differs from its genuine/],
      [v => { row(v, record.coordinate).weightsBytes += 1; }, /differs from its genuine/],
      [v => { v.artifactPrerequisites.activated.find(r => r.coordinate === record.coordinate).weightsSha256 = "f".repeat(64); }, /differs from its genuine/],
      [v => { v.artifactPrerequisites.activated.find(r => r.coordinate === record.coordinate).receiptRecipe.bits = 2; }, /recipe/],
      [v => { v.artifactPrerequisites.activated.find(r => r.coordinate === record.coordinate).measurement = "PASS"; }, /recipe\/evidence/],
      [v => { delete v.calibrationPlan.anchors[record.coordinate]; }, /BLOCKED/],
      [v => { v.artifactPrerequisites.activated = v.artifactPrerequisites.activated.filter(r => r.coordinate !== record.coordinate); }, /neither blocked nor activated/],
    ]) {
      const changed = structuredClone(input); mutate(changed);
      assert.throws(() => validateFlux3AnchorPrerequisites(changed), pattern, record.coordinate);
    }
  }
  for (const mutate of [
    v => { v.artifactPrerequisites.overallTerminalAcceptance = "PASS"; },
    v => { v.artifactPrerequisites.overallTerminalAcceptance = "BLOCKED"; },
    v => v.artifactPrerequisites.activated.push(v.artifactPrerequisites.activated[0]),
    v => { v.artifactPrerequisites.activated[0].coordinate = "flux3_klein:q4:candle"; },
  ]) {
    const changed = structuredClone(input); mutate(changed);
    assert.throws(() => validateFlux3AnchorPrerequisites(changed), /BLOCKED/);
  }
});
test("re-blocking an artifact (row, anchor and record withdrawn) restores an explicit BLOCKED obligation, never a silent gap", async () => {
  const input = await anchorPrerequisiteFixture();
  const record = input.artifactPrerequisites.activated.shift();
  const plannedAnchor = input.calibrationPlan.anchors[record.coordinate];
  delete input.calibrationPlan.anchors[record.coordinate];
  const base = input.models.find(m => m.id === record.modelId);
  base.downloads = base.downloads.filter(d => d.localDerivation?.conversion !== record.conversion);
  assert.throws(() => validateFlux3AnchorPrerequisites(input), /neither blocked nor activated/);
  input.artifactPrerequisites.obligations.push({ coordinate: record.coordinate, required: true, status: "BLOCKED",
    modelId: record.modelId, backend: record.backend, tier: record.tier, repository: record.repository,
    resolvedRevision: record.resolvedRevision, conversion: record.conversion,
    missingEvidence: ["realConvertedWeightsBytes", "realConvertedWeightsSha256", "productionLocalDerivationRow"],
    blocker: "DECLARED SYNTHETIC re-block for classification coverage", plannedAnchor });
  assert.throws(() => validateFlux3AnchorPrerequisites(input), /explicit diagnostic artifact prerequisite inventory/, "BLOCKED status required again");
  input.artifactPrerequisites.overallTerminalAcceptance = "BLOCKED";
  const result = validateFlux3AnchorPrerequisites(input);
  assert.equal(result.activeCanonicalAnchors, 15);
  assert.deepEqual(result.blockedArtifactCoordinates, [record.coordinate]);
  assert.equal(result.terminalAcceptance, "BLOCKED");
  // A blocked obligation still refuses a published row or an executable calibration row.
  for (const mutate of [
    v => { v.calibrationPlan.anchors[record.coordinate] = plannedAnchor; },
    v => { v.artifactPrerequisites.obligations[0].missingEvidence.pop(); },
    v => { v.artifactPrerequisites.obligations[0].status = "PASS"; },
  ]) {
    const changed = structuredClone(input); mutate(changed);
    assert.throws(() => validateFlux3AnchorPrerequisites(changed), /BLOCKED/);
  }
});
test("canonical native anchor hashes match the independent Rust witnesses for both checkpoints", () => {
  assert.equal(flux3AnchorRequestDigest("candle","flux3_klein","bf16"),"144040ed814a0836315c27e126a027fd750fe74a2400aceb2406346befbb455f");
  assert.equal(flux3AnchorRequestDigest("candle","flux3_klein_base","bf16"),"a5a90974bf6d058f30e6ca9899c3b58ca6053b006234fbd28b83237f19543760");
  assert.notEqual(flux3AnchorRequestDigest("mlx","flux3_klein_base","bf16"),flux3AnchorRequestDigest("candle","flux3_klein_base","bf16"));
});
test("supplemental identities bind checkpoint, backend format, actual options, reference instance/order and layout", () => {
  const value = fixture();
  assert.equal(new Set(FLUX3_RESOURCE_OBLIGATIONS).size, FLUX3_RESOURCE_OBLIGATIONS.length);
  assert.ok(FLUX3_RESOURCE_OBLIGATIONS.includes("mlx.base.pup_edit_boxes.bf16"));
  assert.ok(FLUX3_RESOURCE_OBLIGATIONS.every(id => id.endsWith(".bf16")), "supplemental axes do not expand across tiers");
  assert.deepEqual(missingFlux3ResourceObligations([]), FLUX3_RESOURCE_OBLIGATIONS);
  const obligation=structuredClone(value);obligation.caseId="candle.base.reference10.bf16";obligation.tier="bf16";
  assert.ok(!missingFlux3ResourceObligations([obligation]).includes(obligation.caseId));
  obligation.references.pop();assert.throws(()=>normalizeResourceCase(obligation),/ten ordered/);
  const normalized = normalizeResourceCase(value);
  assert.equal(normalized.generation.guidance, 3.5); assert.equal(normalized.generation.steps, 50);
  assert.equal(flux3Checkpoint(normalized.modelId).revision, "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4");
  assert.equal(flux3Conversion("candle", "q4"), "flux3-candle-ggml-q4_1-v1");
  assert.equal(flux3Conversion("mlx", "q8"), "flux3-mlx-affine-q8-g64-v1");
  assert.match(flux3PackageRelativePath(value.modelId, value.backend, value.tier).replaceAll("\\", "/"), /derived\/flux3_klein_base\/q4\/flux3-candle-ggml-q4_1-v1$/);
  const digest = resourceCaseDigest(value);
  assert.equal(digest, sha("sceneworks-flux3-supplemental-request-v1\0" + JSON.stringify(normalized)));
  assert.equal(digest, resourceCaseDigest(normalized));
  const reorderedKeys = Object.fromEntries(Object.entries(value).reverse());
  assert.equal(digest, resourceCaseDigest(reorderedKeys));
  for (const mutate of [v => v.generation.prompt += "!", v => v.generation.width = 1536, v => v.references[0].instanceId = "different", v => v.references.reverse().forEach((r,i) => r.index=i), v => v.generation.layout = { version:1, boxes:[{id:"a",reference_id:"attachment-0",bbox:[0,0,1,1]}] }]) {
    const changed = structuredClone(value); mutate(changed); assert.notEqual(digest, resourceCaseDigest(changed));
  }
  const distilled = structuredClone(value); distilled.modelId="flux3_klein"; distilled.caseId="candle.distilled.reference10.q4";
  assert.equal(normalizeResourceCase(distilled).generation.guidance, undefined);
  assert.equal(normalizeResourceCase(distilled).generation.steps, 16);
  distilled.generation.guidance=1; assert.throws(() => normalizeResourceCase(distilled), /guidance must be omitted/);
  for (const mutate of [v=>v.references[1].instanceId=v.references[0].instanceId, v=>v.references[1].index=0, v=>v.caseId="mlx.base.reference10.q4", v=>v.generation.layout={boxes:[{reference_id:"same-asset"}]}, v=>v.generation.pup=true]) {
    const changed=structuredClone(value);mutate(changed);assert.throws(()=>normalizeResourceCase(changed), /BLOCKED/);
  }
});
test("input binding hashes caller-provided local bytes, rejects missing/corrupt/escaping paths; duplicate assets stay separate instances", async t => {
  const root=await mkdtemp(path.join(tmpdir(),"flux3-resource-test-"));t.after(()=>rm(root,{recursive:true,force:true}));
  await writeFile(path.join(root,"ref.png"),"reference");
  const value=fixture();const bound=await bindResourceCaseInputs(value,{root});
  assert.equal(bound.inputInventory.length,10); assert.equal(bound.inputInventory[0].bytes,9);
  assert.notEqual(bound.inputInventory[0].instanceId,bound.inputInventory[1].instanceId);
  assert.equal(bound.requestSha256,resourceCaseDigest(value));
  await writeFile(path.join(root,"ref.png"),"changed");await assert.rejects(bindResourceCaseInputs(value,{root}), /checksum mismatch/);
  for(const candidate of ["missing.png","../outside.png","https://example.invalid/ref.png","//server/share/ref.png"]) {
    const changed=fixture();changed.references[0].path=candidate;await assert.rejects(bindResourceCaseInputs(changed,{root}));
  }
  await mkdir(path.join(root,"inner"));await writeFile(path.join(root,"outside.png"),"reference");
  const escaped=fixture();escaped.references.forEach(r=>r.path="../outside.png");
  await assert.rejects(bindResourceCaseInputs(escaped,{root:path.join(root,"inner")}), /escapes/);
});
test("supplemental receipts cannot pass the canonical anchor record validator", async () => {
  const resourceCase=normalizeResourceCase(fixture());
  const nativeOutput={schemaVersion:1,kind:"Images",format:"RGB8",representation:"interleaved_host_u8",identityDomain:"flux3_native_output_v1",width:1024,height:1024,bytes:1024*1024*3,sha256:sha("explicit fake carrier fixture")};
  const value={schemaVersion:1,evidenceKind:"flux3_supplemental_resource_observation_v1",anchorEligible:false,caseId:resourceCase.caseId,resourceCase,requestSha256:resourceCaseDigest(resourceCase),status:"OBSERVED",nativeOutput};
  assert.equal(assertSupplementalResourceObservation(value),value);
  for (const mutate of [v=>delete v.nativeOutput,v=>v.nativeOutput.width++,v=>v.nativeOutput.bytes--,v=>v.nativeOutput.format="RGBA8",v=>v.nativeOutput.sha256="short"]) {
    const changed=structuredClone(value);mutate(changed);assert.throws(()=>assertSupplementalResourceObservation(changed));
  }
  assert.equal(assertSupplementalResourceObservation(value),value);
  const { validateRecord }=await import("../memory-calibration-harness.mjs");
  assert.throws(()=>validateRecord({...value,id:value.caseId,logicalCaseId:value.caseId}), /invalid status/i);
  value.requestSha256="0".repeat(64);assert.throws(()=>assertSupplementalResourceObservation(value), /binding mismatch/);
});
test("optional PUP is separately frozen and full-inventoried; preparation never expands or admits cases", async t => {
  const root=await mkdtemp(path.join(tmpdir(),"flux3-resource-pup-"));t.after(()=>rm(root,{recursive:true,force:true}));
  await writeFile(path.join(root,"ref.png"),"reference");await mkdir(path.join(root,"pup"));
  const inventory=[];
  for(const name of ["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","chat_template.jinja","preprocessor_config.json"]) {
    const bytes=`fixture only ${name}`;await writeFile(path.join(root,"pup",name),bytes);inventory.push({path:name,bytes:Buffer.byteLength(bytes),sha256:sha(bytes)});
  }
  const value=fixture();value.generation.pup=true;
  value.components={prompt_upsampler:{root:"pup",repository:"black-forest-labs/flux-3-image-klein-pup",resolvedRevision:"8b069751eb5dcab23829d9c5eccb470eb8b8a4ed",inventory}};
  const bound=await bindResourceCaseInputs(value,{root});assert.equal(bound.inputInventory.filter(i=>i.role==="prompt_upsampler").length,6);
  const {prepareFlux3ResourceCases}=await import("../measure-memory-catalog.mjs");
  const prepared=await prepareFlux3ResourceCases([value],{root});assert.equal(prepared.length,1);assert.equal(prepared[0].status,"INPUTS_BOUND");assert.equal(prepared[0].resourceAdmission,"NOT_GRANTED");assert.equal(prepared[0].anchorEligible,false);
  await assert.rejects(prepareFlux3ResourceCases([value,value],{root}),/duplicate/);
  await writeFile(path.join(root,"pup","system_prompts.json"),"additional actually read input");
  await assert.rejects(bindResourceCaseInputs(value,{root}), /complete actual component/);
  const blocked=await prepareFlux3ResourceCases([value],{root});assert.equal(blocked[0].status,"BLOCKED");assert.equal(blocked.length,1);
  const wrong=structuredClone(value);wrong.components.prompt_upsampler.resolvedRevision="0".repeat(40);assert.throws(()=>normalizeResourceCase(wrong),/frozen PUP/);
});
