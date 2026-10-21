// FLUX3 terminal supplemental resources, NOT a second calibration grid or trusted pin registry.
// No downloads, conversions, native loads or measurements occur in this module.
import path from "node:path";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { realpath, stat, readdir } from "node:fs/promises";

export const FLUX3_CHECKPOINTS = Object.freeze({
  flux3_klein: Object.freeze({ repository: "black-forest-labs/flux-3-image-klein", revision: "8454017f9ebaada30c998fccd9f67a80c7580f68", checkpoint: "distilled", steps: 16 }),
  flux3_klein_base: Object.freeze({ repository: "black-forest-labs/flux-3-image-klein-base", revision: "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4", checkpoint: "base", steps: 50, guidance: 3.5 }),
});
// Deliberately explicit terminal resource obligations, not a tier/axis cross product. Native
// tiers each keep ONE canonical plain anchor; these BF16 scopes investigate additional consumers.
export const FLUX3_RESOURCE_OBLIGATIONS = Object.freeze([
  "candle.distilled.reference1.bf16", "candle.base.reference1.bf16",
  "mlx.distilled.reference1.bf16", "mlx.base.reference1.bf16",
  "candle.distilled.reference10.bf16", "candle.base.reference10.bf16",
  "mlx.distilled.reference10.bf16", "mlx.base.reference10.bf16",
  "candle.distilled.adapter.bf16", "candle.base.adapter.bf16",
  "mlx.distilled.adapter.bf16", "mlx.base.adapter.bf16",
  "candle.base.pup_text.bf16", "candle.base.pup_edit.bf16",
  "candle.base.pup_text_boxes.bf16", "candle.base.pup_edit_boxes.bf16",
  "mlx.base.pup_text.bf16", "mlx.base.pup_edit.bf16",
  "mlx.base.pup_text_boxes.bf16", "mlx.base.pup_edit_boxes.bf16",
]);
// Fixed REQUIRED canonical inventory, distinct from supplemental observations and the smaller
// representative run. Unpublished artifacts remain obligations, never executable calibration rows.
export const FLUX3_REQUIRED_CANONICAL_ANCHORS = Object.freeze([
  "flux3_klein:bf16:candle", "flux3_klein:q4:candle", "flux3_klein:q8:candle",
  "flux3_klein:fp8r:candle", "flux3_klein:nvfp4:candle",
  "flux3_klein:bf16:mlx", "flux3_klein:q4:mlx", "flux3_klein:q8:mlx",
  "flux3_klein_base:bf16:candle", "flux3_klein_base:q4:candle", "flux3_klein_base:q8:candle",
  "flux3_klein_base:fp8r:candle", "flux3_klein_base:nvfp4:candle",
  "flux3_klein_base:bf16:mlx", "flux3_klein_base:q4:mlx", "flux3_klein_base:q8:mlx",
]);
// Publication classification only: neither a trusted converted-artifact receipt nor permission
// to generate/capture. Actual package bytes and all execution admissions remain separately required.
export function validateFlux3AnchorPrerequisites({ models, calibrationPlan, artifactPrerequisites }) {
  if (artifactPrerequisites?.schemaVersion !== 1 ||
      artifactPrerequisites.kind !== "flux3-terminal-blocked-artifact-prerequisites" ||
      artifactPrerequisites.diagnosticOnly !== true || !Array.isArray(artifactPrerequisites.obligations) ||
      (artifactPrerequisites.obligations.length > 0 && artifactPrerequisites.overallTerminalAcceptance !== "BLOCKED")) {
    throw new Error("BLOCKED: explicit diagnostic artifact prerequisite inventory required");
  }
  const required = new Set(FLUX3_REQUIRED_CANONICAL_ANCHORS);
  const active = Object.keys(calibrationPlan.anchors).filter(key => key.startsWith("flux3_"));
  const blocked = new Map();
  const published = key => {
    const [modelId, tier, backend] = key.split(":");
    const checkpoint = flux3Checkpoint(modelId);
    const platform = backend === "mlx" ? "macos" : "windows";
    return models.find(model => model.id === modelId)?.downloads?.some(download => {
      if (download.coRequisite || download.variant !== tier ||
          download.repo !== checkpoint.repository || download.revision !== checkpoint.revision ||
          (download.platforms && !download.platforms.includes(platform))) return false;
      if (!["q4", "q8"].includes(tier)) return true;
      const conversion = download.localDerivation;
      return conversion?.fromVariant === "bf16" && conversion.conversion === flux3Conversion(backend, tier) &&
        conversion.weightsFile === "model.safetensors" && Number.isSafeInteger(conversion.weightsBytes) &&
        conversion.weightsBytes > 0 && /^[a-f0-9]{64}$/.test(conversion.weightsSha256 ?? "");
    }) ?? false;
  };
  for (const obligation of artifactPrerequisites.obligations) {
    const { coordinate, modelId, backend, tier } = obligation;
    if (!required.has(coordinate) || coordinate !== `${modelId}:${tier}:${backend}` || blocked.has(coordinate)) {
      throw new Error(`BLOCKED: unknown, duplicate or misbound artifact obligation ${coordinate}`);
    }
    const checkpoint = flux3Checkpoint(modelId);
    if (obligation.status !== "BLOCKED" || obligation.required !== true ||
        obligation.repository !== checkpoint.repository || obligation.resolvedRevision !== checkpoint.revision ||
        obligation.conversion !== flux3Conversion(backend, tier) || obligation.plannedAnchor?.provider !== modelId ||
        !obligation.blocker || JSON.stringify(obligation.missingEvidence) !== JSON.stringify([
          "realConvertedWeightsBytes", "realConvertedWeightsSha256", "productionLocalDerivationRow",
        ])) throw new Error(`BLOCKED: incomplete artifact prerequisite identity/evidence for ${coordinate}`);
    if (published(coordinate)) throw new Error(`BLOCKED: published ${coordinate} must activate its canonical calibration row`);
    blocked.set(coordinate, obligation);
  }
  for (const key of active) {
    if (!required.has(key) || blocked.has(key)) throw new Error(`BLOCKED: non-disjoint or unknown canonical anchor ${key}`);
    if (!published(key)) throw new Error(`BLOCKED: active canonical anchor ${key} lacks published pinned artifact bytes/SHA/row`);
    if (calibrationPlan.anchors[key]?.provider !== key.split(":")[0]) throw new Error(`BLOCKED: canonical provider mismatch ${key}`);
  }
  const classified = new Set([...active, ...blocked.keys()]);
  const missing = [...required].filter(key => !classified.has(key));
  if (missing.length || classified.size !== required.size) throw new Error(`BLOCKED: missing required canonical obligations: ${missing.join(", ")}`);
  return { requiredCanonicalAnchors: required.size, activeCanonicalAnchors: active.length,
    blockedArtifactCoordinates: [...blocked.keys()], terminalAcceptance: blocked.size ? "BLOCKED" : "NOT_PROVEN",
    measurementProven: false, executionAuthorized: false };
}
export function missingFlux3ResourceObligations(cases) {
  const ids = new Set(cases.map(value => normalizeResourceCase(value).caseId));
  return FLUX3_RESOURCE_OBLIGATIONS.filter(id => !ids.has(id));
}
export function flux3Checkpoint(modelId) {
  if (!Object.hasOwn(FLUX3_CHECKPOINTS, modelId)) throw new Error(`BLOCKED: unknown FLUX3 checkpoint ${modelId}`);
  return FLUX3_CHECKPOINTS[modelId];
}
export function flux3Conversion(backend, tier) {
  if (!["candle", "mlx"].includes(backend)) throw new Error("BLOCKED: unknown native backend");
  if (!["bf16", "q4", "q8", ...(backend === "candle" ? ["fp8r", "nvfp4"] : [])].includes(tier)) throw new Error(`BLOCKED: no native ${backend}:${tier} format; no substitution`);
  if (!["q4", "q8"].includes(tier)) return null;
  return backend === "candle" ? `flux3-candle-ggml-${tier === "q4" ? "q4_1" : "q8_0"}-v1` : `flux3-mlx-affine-${tier}-g64-v1`;
}
export function flux3PackageRelativePath(modelId, backend, tier) {
  flux3Checkpoint(modelId);
  const conversion = flux3Conversion(backend, tier);
  return conversion ? path.join("models", "derived", modelId, tier, conversion) : tier === "bf16" ? "." : path.join("variants", tier);
}
function exact(value, allowed, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`BLOCKED: ${name} must be an object`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`BLOCKED: unknown ${name}.${key}`);
}
function text(value, name) {
  if (typeof value !== "string" || !value.length) throw new Error(`BLOCKED: ${name} must be nonempty`);
  return value;
}
function hash(value) {
  if (!/^[a-f0-9]{64}$/.test(value ?? "")) throw new Error("BLOCKED: full lowercase SHA256 required");
  return value;
}
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
// Exact canonical native-anchor request identity also computed immediately before native generate.
// This is a planning hash, never a generation/capture authorization.
export function flux3AnchorRequestDigest(backend, modelId, tier) {
  const checkpoint = flux3Checkpoint(modelId);
  flux3Conversion(backend, tier);
  const actual = canonical({ backend, provider: modelId, tier,
    prompt: "a portrait of a person in a sunlit studio, editorial photograph",
    width: 1024, height: 1024, count: 1, seed: 26243,
    steps: checkpoint.steps, guidance: checkpoint.guidance ?? null, enhancePrompt: false });
  return createHash("sha256").update("sceneworks-flux3-anchor-request-v1\0").update(JSON.stringify(actual)).digest("hex");
}
export function normalizeResourceCase(value) {
  exact(value, ["schemaVersion", "caseId", "backend", "modelId", "tier", "scenario", "generation", "references", "adapters", "components"], "resourceCase");
  if (value.schemaVersion !== 1) throw new Error("BLOCKED: resourceCase schemaVersion must be 1");
  const checkpoint = flux3Checkpoint(value.modelId);
  flux3Conversion(value.backend, value.tier);
  const scenario = text(value.scenario, "scenario");
  if (!/^[a-z][a-z0-9_]*$/.test(scenario)) throw new Error("BLOCKED: invalid scenario");
  const caseId = `${value.backend}.${checkpoint.checkpoint}.${scenario}.${value.tier}`;
  if (value.caseId !== caseId) throw new Error(`BLOCKED: caseId must be ${caseId}`);
  exact(value.generation, ["prompt", "width", "height", "seed", "steps", "guidance", "pup", "layout"], "generation");
  const generation = { ...value.generation };
  text(generation.prompt, "generation.prompt");
  for (const axis of ["width", "height"]) if (!Number.isSafeInteger(generation[axis]) || generation[axis] < 64 || generation[axis] % 16) throw new Error(`BLOCKED: ${axis} must be a positive native multiple of 16`);
  generation.seed ??= 26243;
  generation.steps ??= checkpoint.steps;
  if (!Number.isSafeInteger(generation.seed) || generation.seed < 0 || !Number.isInteger(generation.steps) || generation.steps < 1) throw new Error("BLOCKED: invalid native sampling values");
  if (checkpoint.checkpoint === "distilled" && generation.guidance !== undefined) throw new Error("BLOCKED: distilled guidance must be omitted, not substituted with 1");
  if (checkpoint.checkpoint === "base") {
    generation.guidance ??= checkpoint.guidance;
    if (!Number.isFinite(generation.guidance) || generation.guidance < 0) throw new Error("BLOCKED: invalid base guidance");
  }
  generation.pup ??= false;
  if (typeof generation.pup !== "boolean") throw new Error("BLOCKED: explicit boolean PUP required");
  const references = value.references ?? [];
  if (!Array.isArray(references) || references.length > 10) throw new Error("BLOCKED: ordered 0-10 references required");
  const instances = new Set();
  for (const [index, reference] of references.entries()) {
    exact(reference, ["instanceId", "assetId", "index", "path", "sha256"], "reference");
    if (reference.index !== index) throw new Error("BLOCKED: reference index must preserve request order");
    text(reference.instanceId, "reference.instanceId"); text(reference.assetId, "reference.assetId"); text(reference.path, "reference.path"); hash(reference.sha256);
    if (instances.has(reference.instanceId)) throw new Error("BLOCKED: duplicate reference instanceId");
    instances.add(reference.instanceId);
  }
  if (generation.layout !== undefined) {
    // Native Layout::from_json remains the semantic authority. This layer only binds attachment
    // identities; it cannot rewrite source/target boxes or silently flatten a structured layout.
    const visit = node => {
      if (Array.isArray(node)) return node.forEach(visit);
      if (!node || typeof node !== "object") return;
      if (Object.hasOwn(node, "reference_id") && !instances.has(node.reference_id)) throw new Error("BLOCKED: layout reference_id must bind a request instanceId");
      Object.values(node).forEach(visit);
    };
    visit(generation.layout);
  }
  const adapters = value.adapters ?? [];
  if (!Array.isArray(adapters)) throw new Error("BLOCKED: adapters must be an array");
  for (const adapter of adapters) {
    exact(adapter, ["path", "sha256", "scale"], "adapter"); text(adapter.path, "adapter.path"); hash(adapter.sha256);
    if (!Number.isFinite(adapter.scale)) throw new Error("BLOCKED: finite adapter scale required");
  }
  const components = value.components ?? {};
  exact(components, ["prompt_upsampler"], "components");
  if (generation.pup !== Object.hasOwn(components, "prompt_upsampler")) throw new Error("BLOCKED: PUP case requires exactly its separately inventoried component, bypass must omit it");
  if (generation.pup) {
    const pup = components.prompt_upsampler;
    exact(pup, ["root", "repository", "resolvedRevision", "inventory"], "prompt_upsampler"); text(pup.root, "prompt_upsampler.root");
    if (pup.repository !== "black-forest-labs/flux-3-image-klein-pup" || pup.resolvedRevision !== "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed") throw new Error("BLOCKED: frozen PUP identity required");
    if (!Array.isArray(pup.inventory) || !pup.inventory.length) throw new Error("BLOCKED: complete PUP inventory required");
    const names = new Set();
    for (const file of pup.inventory) {
      exact(file, ["path", "bytes", "sha256"], "component file"); text(file.path, "component file.path"); hash(file.sha256);
      if (names.has(file.path) || !Number.isSafeInteger(file.bytes) || file.bytes <= 0) throw new Error("BLOCKED: duplicate/empty component inventory file");
      names.add(file.path);
    }
    for (const required of ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json"]) if (!names.has(required)) throw new Error(`BLOCKED: missing PUP inventory ${required}`);
  }
  if (FLUX3_RESOURCE_OBLIGATIONS.includes(caseId)) {
    if (scenario === "reference1" && (references.length !== 1 || generation.pup || adapters.length)) throw new Error("BLOCKED: reference1 obligation requires one reference (distinct 4MP preprocessing), PUP bypass and no adapter");
    if (scenario === "reference10" && (references.length !== 10 || generation.pup || adapters.length)) throw new Error("BLOCKED: reference10 obligation requires ten ordered references, PUP bypass and no adapter");
    if (scenario === "adapter" && (!adapters.length || generation.pup || references.length)) throw new Error("BLOCKED: adapter obligation requires a real adapter and otherwise plain conditioning");
    if (scenario.startsWith("pup_")) {
      const editing = scenario.includes("edit"); const boxes = scenario.endsWith("_boxes");
      if (!generation.pup || references.length !== (editing ? 10 : 0) || boxes !== (generation.layout !== undefined) || adapters.length) throw new Error("BLOCKED: PUP obligation mode requires its actual reference/layout request, no adapter substitution");
    }
  }
  return canonical({ schemaVersion: 1, caseId, backend: value.backend, modelId: value.modelId, tier: value.tier, scenario, generation, references, adapters, components });
}
export function resourceCaseDigest(value) {
  const normalized = normalizeResourceCase(value);
  return createHash("sha256").update("sceneworks-flux3-supplemental-request-v1\0").update(JSON.stringify(normalized)).digest("hex");
}
async function confined(root, candidate) {
  if (typeof candidate !== "string" || /^(?:[a-z]+:\/\/|\\\\|\/\/)/i.test(candidate)) throw new Error("BLOCKED: only caller-provided local paths allowed");
  const file = await realpath(path.resolve(root, candidate));
  const relative = path.relative(root, file);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error("BLOCKED: input escapes provided root");
  return file;
}
async function fileBinding(root, candidate, expected, bytes) {
  const file = await confined(root, candidate);
  const before = await stat(file);
  if (!before.isFile() || !before.size || (bytes !== undefined && before.size !== bytes)) throw new Error("BLOCKED: empty or mismatched local input");
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  const sha256 = digest.digest("hex");
  const after = await stat(file);
  if (sha256 !== expected || before.size !== after.size || before.mtimeMs !== after.mtimeMs) throw new Error("BLOCKED: local input checksum mismatch/change");
  return { path: file, bytes: after.size, sha256 };
}
export async function bindResourceCaseInputs(value, { root }) {
  const normalized = normalizeResourceCase(value);
  const canonicalRoot = await realpath(text(root, "input root"));
  const inputs = [];
  for (const reference of normalized.references) inputs.push({ role: "reference", instanceId: reference.instanceId, assetId: reference.assetId, index: reference.index, ...await fileBinding(canonicalRoot, reference.path, reference.sha256) });
  for (const [index, adapter] of normalized.adapters.entries()) inputs.push({ role: "adapter", index, scale: adapter.scale, ...await fileBinding(canonicalRoot, adapter.path, adapter.sha256) });
  const pup = normalized.components.prompt_upsampler;
  if (pup) {
    const componentRoot = await confined(canonicalRoot, pup.root);
    const actualNames = [];
    const walk = async (directory, prefix = "") => {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        const name = prefix + entry.name;
        if (entry.isDirectory()) await walk(path.join(directory, entry.name), `${name}/`);
        else {
          const file = await confined(componentRoot, name);
          if (!(await stat(file)).isFile()) throw new Error("BLOCKED: component inventory cannot follow directory links");
          actualNames.push(name);
        }
      }
    };
    await walk(componentRoot);
    if (JSON.stringify(actualNames.sort()) !== JSON.stringify(pup.inventory.map(file => file.path).sort())) throw new Error("BLOCKED: PUP inventory must cover the complete actual component directory");
    for (const file of pup.inventory) inputs.push({ role: "prompt_upsampler", ...await fileBinding(componentRoot, file.path, file.sha256, file.bytes) });
  }
  return { normalized, requestSha256: resourceCaseDigest(normalized), inputInventory: inputs };
}
export function assertSupplementalResourceObservation(value) {
  if (value?.evidenceKind !== "flux3_supplemental_resource_observation_v1" || value?.anchorEligible !== false) throw new Error("invalid supplemental resource observation");
  const expected = resourceCaseDigest(value.resourceCase);
  if (value.requestSha256 !== expected || value.caseId !== value.resourceCase.caseId) throw new Error("supplemental request binding mismatch");
  const image = value.nativeOutput;
  if (image?.schemaVersion !== 1 || image.kind !== "Images" || image.format !== "RGB8" || image.representation !== "interleaved_host_u8" || image.identityDomain !== "flux3_native_output_v1"
    || image.width !== value.resourceCase.generation.width || image.height !== value.resourceCase.generation.height || image.bytes !== image.width * image.height * 3) throw new Error("supplemental native output carrier mismatch");
  hash(image.sha256);
  return value;
}

export function flux3CaptureDigest(domain, value) {
  return createHash('sha256').update(domain).update(JSON.stringify(canonical(value))).digest('hex');
}
export function verifyNativeResourceCapture(value, { admission, invocation, adapterPid, ledgerEvent }) {
  const fail = message => { throw new Error(`BLOCKED: native resource capture ${message}`); };
  if (value?.anchorEligible !== false || value.captureId !== invocation.invocationId || value.caseId !== invocation.caseId
    || value.requestSha256 !== invocation.requestSha256) fail('case/request/capture identity mismatch');
  const { captureBindingSha256, ...bound } = value;
  if (captureBindingSha256 !== flux3CaptureDigest('flux3_native_capture_binding_v1\0', bound)) fail('envelope binding mismatch');
  const image = value.nativeOutput;
  const generation = invocation.kind === 'resource_supplemental' ? value.resourceCase?.generation : { width:1024, height:1024 };
  if (image?.schemaVersion !== 1 || image.kind !== 'Images' || image.format !== 'RGB8' || image.representation !== 'interleaved_host_u8'
    || image.identityDomain !== 'flux3_native_output_v1' || image.width !== generation?.width || image.height !== generation?.height
    || image.bytes !== image.width * image.height * 3 || !/^[a-f0-9]{64}$/.test(image.sha256 ?? '')) fail('native output mismatch');
  const actualRequest=image.actualGenerationRequest;
  const expectedRequest=invocation.kind==='resource_supplemental' ? normalizeResourceCase(value.resourceCase).generation
    : {prompt:'a portrait of a person in a sunlit studio, editorial photograph',width:1024,height:1024,seed:26243,
      steps:flux3Checkpoint(invocation.modelId).steps,guidance:flux3Checkpoint(invocation.modelId).guidance ?? null,pup:false};
  for(const [key,expected] of Object.entries({prompt:expectedRequest.prompt,width:expectedRequest.width,height:expectedRequest.height,
    seed:expectedRequest.seed,steps:expectedRequest.steps,guidance:expectedRequest.guidance ?? null,count:1,enhancePrompt:expectedRequest.pup,
    negativePrompt:null,trueCfg:null,sampler:null,scheduler:null,schedulerShift:null})) {
    if(actualRequest?.[key]!==expected) fail(`actual GenerationRequest ${key} mismatch`);
  }
  const references=invocation.kind==='resource_supplemental' ? value.resourceCase.references : [];
  if(actualRequest.conditioningCount!==references.length || actualRequest.referenceCarriers?.length!==references.length
    || JSON.stringify(canonical(actualRequest.layout))!==JSON.stringify(canonical(expectedRequest.layout ?? null))) fail('actual native references/layout mismatch');
  if(expectedRequest.layout === undefined && actualRequest.layoutBindings?.length!==0) fail('ordinary references cannot claim layout bindings');
  for(const [index,carrier] of actualRequest.referenceCarriers.entries()) {
    const source=value.inputInventory?.find(i=>i.role==='reference' && i.index===index && i.instanceId===references[index].instanceId);
    if(!source || JSON.stringify(canonical(source.decodedCarrier))!==JSON.stringify(canonical(carrier))) fail('actual decoded reference carrier differs from bound input');
  }
  const { hardwareObservation, ...reservation } = image.reservation ?? {};
  if (JSON.stringify(canonical(reservation)) !== JSON.stringify(canonical(ledgerEvent))) fail('not the actual immutable ledger event');
  for (const [key, expected] of Object.entries({ invocationId:invocation.invocationId, caseId:invocation.caseId, kind:invocation.kind,
    backend:invocation.backend, requestSha256:invocation.requestSha256, ledgerId:admission.ledgerId, approvedPlanSha256:admission.approvedPlanSha256 })) {
    if (reservation[key] !== expected) fail(`immutable reservation ${key} mismatch`);
  }
  for (const [key, expected] of Object.entries({ invocationId:invocation.invocationId, caseId:invocation.caseId, requestSha256:invocation.requestSha256,
    approvedPlanSha256:admission.approvedPlanSha256, sourceSha:admission.sourceSha, binarySha256:admission.binarySha256, adapterPid })) {
    if (hardwareObservation?.[key] !== expected) fail(`observed hardware ${key} binding mismatch`);
  }
  if (hardwareObservation.kind !== 'flux3-resource-producer-hardware-observation' || hardwareObservation.backend !== admission.backend
    || hardwareObservation.inferenceMainSha !== admission.inferenceMainSha || hardwareObservation.buildReceiptSha256 !== admission.binaryBuildReceipt.sha256
    || hardwareObservation.adapterProcess?.pid !== adapterPid || !hardwareObservation.adapterProcess.startedAt
    || !Number.isFinite(Date.parse(hardwareObservation.observedAt))) fail('actual producer/build observation missing');
  const proof = hardwareObservation.proof;
  if (admission.backend === 'candle') {
    if (proof?.physicalGpuId !== '1' || proof.gpuUuid?.toLowerCase() !== admission.gpuUuid?.toLowerCase()
      || proof.logicalCudaOrdinal !== 0 || !proof.ownComputePids?.includes(adapterPid)) fail('GPU1 UUID/PID proof missing');
  } else if (proof?.runner !== 'nax-macos-2' || proof.host?.split('.')[0] !== admission.hostName || proof.runtime?.platform !== 'darwin'
    || proof.runtime?.arch !== 'arm64' || !proof.coordinatorLeaseId || proof.exclusivity !== 'coordinator-attested-and-process-monitored') fail('secondary Mac runtime/host/lease proof missing');
  const spec = value.actualLoadSpec;
  const { sha256:specSha256, ...actualSpec } = spec ?? {};
  if (spec?.provenanceKind !== 'actual_borrowed_load_spec_before_catalog_load' || spec.backend !== invocation.backend
    || spec.provider !== invocation.modelId || spec.tier !== invocation.tier
    || specSha256 !== flux3CaptureDigest('flux3_actual_load_spec_v1\0',actualSpec)
    || value.loadedGenerator?.provider !== invocation.modelId || value.loadedGenerator?.backend !== invocation.backend) fail('actual loaded spec/provider mismatch');
  if (invocation.kind === 'resource_supplemental') {
    assertSupplementalResourceObservation(value);
    if (value.evidenceKind !== 'flux3_supplemental_resource_observation_v1' || value.anchorFragment !== undefined) fail('supplemental cannot extract an anchor');
    return { supplementalObservation:value, anchorFragment:null };
  }
  if (invocation.kind !== 'resource_anchor' || value.evidenceKind !== 'flux3_native_resource_capture_envelope_v1' || value.resourceCase !== undefined) fail('not a canonical native anchor envelope');
  if (value.anchorFragmentSha256 !== flux3CaptureDigest('flux3_canonical_anchor_fragment_v1\0',value.anchorFragment)) fail('extracted anchor fragment mismatch');
  return { anchorFragment:value.anchorFragment, supplementalObservation:null };
}
