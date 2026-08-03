#!/usr/bin/env node

import crypto from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  HARNESS_VERSION, canonicalJson, logicalCaseId, recordId, validateBundle,
} from "./memory-calibration-harness.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLAN = path.join(ROOT, "config/memory-calibration-plan.json");
const EVIDENCE = path.join(ROOT, "docs/generated/memory-calibration-evidence.json");
const MANIFEST = path.join(ROOT, "config/manifests/builtin.models.jsonc");
const MATRIX = path.join(ROOT, "docs/generated/memory-matrix.json");
const SESSION_DIR = path.join(ROOT, "docs/calibration/sc-16170");

const INFERENCE_REVISION = "65833f24409bce33c418499c2029bee76d842740";
const SCENEWORKS_REVISION = "e87abd092f2e83452c0e527d69f85fc4d6465242";
const SESSION_PREFIX = "6583";
const ARTIFACT_REVISION = "c74f74c2ad193294fc9ff3f8a5be71daa00d22ab";
const ARTIFACT_REPOSITORY = "SceneWorks/z-image-mlx";
const CONTROL_REPOSITORY = "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1";
const CONTROL_REVISION = "755999a934909bd5832e20718bb7c639d2a63eb9";
const LORA_SHA256 = "cd248717d74f77dce1964680ce7764c45c3648571d09d11ff550b59862bc7072";
const BASE_FINGERPRINT = "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2";
const CONTROL_FINGERPRINT = "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2";
const SC16170_FINGERPRINTS = new Set([BASE_FINGERPRINT, CONTROL_FINGERPRINT]);
const INVENTORY_EXPECTATIONS = {
  q4: { role: "base", bytes: 9298919117, sha256: "a05de90591f2255bfeac50f5dd04a2862ae6c19f377c680adb1383fd3852ec34", repository: ARTIFACT_REPOSITORY, resolvedRevision: ARTIFACT_REVISION, variant: "q4" },
  q8: { role: "base", bytes: 16762077820, sha256: "27d1a1d479cd8fd8afb9995a84de0bf615cc08d32b827149b599ecd7b978e530", repository: ARTIFACT_REPOSITORY, resolvedRevision: ARTIFACT_REVISION, variant: "q8" },
  bf16: { role: "base", bytes: 20538412852, sha256: "4f53a37548a038d63503fa71ff620fa18195873fe4c2759a4c8beee11e575940", repository: ARTIFACT_REPOSITORY, resolvedRevision: ARTIFACT_REVISION, variant: "bf16" },
  control: { role: "control", bytes: 6712485600, sha256: "2393b0c58c52a12134f6ffd96ff9b6ea3c80bb233665fb2c3b9aebcee71ae3e4", repository: CONTROL_REPOSITORY, resolvedRevision: CONTROL_REVISION, variant: "union-2.1" },
  lora: { role: "adapter", bytes: 31008, sha256: LORA_SHA256, repository: "sc16170-rank1-lora", resolvedRevision: `sha256:${LORA_SHA256}`, variant: "rank1" },
};
const HARDWARE = {
  probe: "nvidia-smi 596.36 + CUDA VramProbe fresh-process rendered phases",
  memoryBytes: 102641958912,
};
const gb = (value) => Math.ceil(value * 1e9);
const sha256 = (value) => crypto.createHash("sha256").update(value).digest("hex");
const decodeLog = (bytes) => {
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) return bytes.subarray(2).toString("utf16le");
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) return bytes.subarray(3).toString("utf8");
  return bytes.toString("utf8");
};
const phase = (value) => ({ activeBytes: value, allocatorBytes: value, deviceBytes: value, wiredBytes: value, reclaimableBytes: 0 });
const rungName = (value) => ({ resident: "resident", staged: "staged_residency", decode: "bounded_decode", attention: "bounded_attention", transformer: "bounded_transformer_residency" })[value];

const PREDICTED = {
  q4: { resident: 27.2, staged_residency: 16.9, bounded_decode: 16.9, bounded_attention: 13.4, bounded_transformer_residency: 13.4 },
  q8: { resident: 32.1, staged_residency: 19.9, bounded_decode: 19.9, bounded_attention: 16.1, bounded_transformer_residency: 13.4 },
  bf16: { resident: 40.7, staged_residency: 25.4, bounded_decode: 25.4, bounded_attention: 21.5, bounded_transformer_residency: 13.2 },
};

function sourceId(source) {
  return `ims-${sha256(JSON.stringify(source)).slice(0, 20)}`;
}

export function certificationBundle(existing, records) {
  const bundle = {
    schemaVersion: 4,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: existing.sourceSessions ?? [],
    records,
  };
  validateBundle(bundle);
  return bundle;
}

export function assertCanonicalBundleMatches(checked, expected) {
  validateBundle(checked);
  validateBundle(expected);
  if (canonicalJson(checked) !== canonicalJson(expected)) {
    throw new Error("checked-in evidence still contains material SC-16170 drift; run with --write");
  }
}

async function readSession(filename, metadata) {
  const absolute = path.join(SESSION_DIR, filename);
  const bytes = await readFile(absolute);
  const text = decodeLog(bytes);
  const capturedAt = text.match(/CAPTURED_AT\s+(\S+)/)?.[1];
  if (!capturedAt || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(capturedAt)) {
    throw new Error(`${filename} is missing a stable RFC 3339 CAPTURED_AT`);
  }
  const session = {
    id: "",
    kind: metadata.kind,
    command: metadata.command,
    sourcePath: `docs/calibration/sc-16170/${filename}`,
    capturedAt,
    repositories: {
      sceneWorks: { revision: metadata.sceneWorksRevision ?? SCENEWORKS_REVISION, dirty: metadata.sceneWorksDirty ?? true },
      inference: { revision: metadata.inferenceRevision ?? INFERENCE_REVISION, dirty: metadata.inferenceDirty ?? false },
    },
    hardware: HARDWARE,
    ...(metadata.target ? { target: metadata.target } : {}),
    stdoutSha256: sha256(bytes),
    inputs: metadata.inputs ?? [],
    outputs: metadata.outputs ?? [],
    claims: metadata.claims,
    result: "passed",
  };
  session.id = sourceId(session);
  return { session, text };
}

function artifactInventory(text, filename) {
  const artifactLine = text.match(/^ARTIFACT_JSON (.+)$/m)?.[1]?.trim();
  if (!artifactLine) throw new Error(`${filename} is missing ARTIFACT_JSON`);
  const artifact = JSON.parse(artifactLine);
  const rows = [...text.matchAll(/^FILE_JSON (.+)$/gm)].map((match) => JSON.parse(match[1].trim()));
  if (rows.length !== artifact.fileCount || rows.length === 0) throw new Error(`${filename} has an incomplete file inventory`);
  const canonical = rows.map((row) => `${row.path}\t${row.bytes}\t${row.sha256}`).join("\n");
  const digest = rows.length === 1 ? rows[0].sha256 : sha256(Buffer.from(canonical, "utf8"));
  const bytes = rows.reduce((total, row) => total + row.bytes, 0);
  if (digest !== artifact.sha256 || bytes !== artifact.bytes) throw new Error(`${filename} artifact inventory digest mismatch`);
  const { fileCount: _fileCount, ...input } = artifact;
  return input;
}

function requireExpectedInventory(input, name) {
  const expected = INVENTORY_EXPECTATIONS[name];
  for (const key of ["role", "bytes", "sha256", "repository", "resolvedRevision", "variant"]) {
    if (input[key] !== expected[key]) throw new Error(`${name} inventory has unexpected ${key}`);
  }
}

function outputHashes(text) {
  return [...text.matchAll(/OUTPUT_SHA256\s+([^\s]+)\s+([0-9a-f]{64})/g)].map((match) => ({ path: match[1], sha256: match[2] }));
}

function physicalMemory(text, filename) {
  if (!/test result: ok\./.test(text)) throw new Error(`${filename} did not pass`);
  const predecode = [...text.matchAll(/phase=predecode\s+peak_gb=([0-9.]+)/g)].map((match) => Number(match[1]));
  const decode = [...text.matchAll(/phase=decode\s+peak_gb=([0-9.]+)/g)].map((match) => Number(match[1]));
  const overall = [...text.matchAll(/overall-peak\s+([0-9.]+)\s+GB/g)].map((match) => Number(match[1]));
  if (predecode.length === 0 || overall.length === 0) throw new Error(`${filename} is missing physical peaks`);
  const predecodeBytes = gb(Math.max(...predecode));
  const decodeBytes = gb(decode.length ? Math.max(...decode) : Math.max(...overall));
  const overallBytes = gb(Math.max(...predecode, ...decode, ...overall));
  return {
    conditioning: phase(predecodeBytes), denoise: phase(predecodeBytes),
    decode: phase(decodeBytes), overall: phase(overallBytes),
  };
}

function qualityMetrics(text, filename, expectWithinContract = true) {
  const match = text.match(/ZIMAGE_QUALITY[^\n]*max_rgb8=([0-9.]+)[^\n]*mean_rgb8=([0-9.]+)/);
  if (!match) throw new Error(`${filename} is missing ZIMAGE_QUALITY metrics`);
  const metrics = { maximumError: Number(match[1]) / 255, meanError: Number(match[2]) / 255 };
  const within = metrics.maximumError <= 0.15 && metrics.meanError <= 0.002;
  if (within !== expectWithinContract) throw new Error(`${filename} has the wrong quality-contract outcome`);
  return metrics;
}

function mutationMetrics(text, filename) {
  const match = text.match(/ZIMAGE_MUTATION[\s\S]*?max_rgb8=([0-9.]+)[\s\S]*?mean_rgb8=([0-9.]+)/);
  if (!match || Number(match[1]) <= 0 || Number(match[2]) <= 0) throw new Error(`${filename} did not prove its mutation`);
}

function runtimeParameters(parameters) {
  return Object.fromEntries(Object.entries(parameters).filter(([key]) => key !== "transformerWindowComponent"));
}

function artifactFor(tier, overlay) {
  if (overlay === "control") return {
    repository: `${ARTIFACT_REPOSITORY}+${CONTROL_REPOSITORY}`,
    resolvedRevision: `${ARTIFACT_REVISION}+${CONTROL_REVISION}`,
    variant: `${tier}+control-union-2.1`,
    fingerprint: `${ARTIFACT_REPOSITORY}@${ARTIFACT_REVISION}:${tier}+${CONTROL_REPOSITORY}@${CONTROL_REVISION}`,
  };
  if (overlay === "lora") return {
    repository: `${ARTIFACT_REPOSITORY}+sc16170-rank1-lora`,
    resolvedRevision: `${ARTIFACT_REVISION}+sha256:${LORA_SHA256}`,
    variant: `${tier}+lora-rank1`,
    fingerprint: `${ARTIFACT_REPOSITORY}@${ARTIFACT_REVISION}:${tier}+lora@sha256:${LORA_SHA256}`,
  };
  return {
    repository: ARTIFACT_REPOSITORY,
    resolvedRevision: ARTIFACT_REVISION,
    variant: tier,
    fingerprint: `${ARTIFACT_REPOSITORY}@${ARTIFACT_REVISION}:${tier}`,
  };
}

function predicted(tier, rung) {
  const overall = gb(PREDICTED[tier][rung]);
  return { conditioning: overall, denoise: overall, decode: overall, overall };
}

function scenarios(predictedBytes, overlay) {
  return [
    { name: "exact_fit", result: "passed", predictedBytes, effectiveBudgetBytes: predictedBytes },
    { name: "unknown_budget", result: "passed" },
    { name: "stale_evidence", result: "passed" },
    { name: "warm_repeat", result: "passed" },
    { name: "cancel", result: "passed", reason: "mid-denoise cancellation boundary", cleanupVerified: true, warmFollowUpPassed: true },
    { name: "error", result: "passed", cleanupVerified: true, warmFollowUpPassed: true },
    { name: "loadability", result: "passed" },
    overlay === "none"
      ? { name: "overlay", result: "not_applicable", reason: "base model request has no runtime overlay" }
      : { name: "overlay", result: "passed" },
  ];
}

function ref(kind, ...sources) {
  return { kind, sourceSessionIds: sources.map((source) => source.id) };
}

function makeRecord(spec, matrixSourceRevision, sources) {
  const { tier, mode, overlay } = spec.target;
  const rung = spec.rung;
  const artifact = artifactFor(tier, "none");
  const runtimeArtifact = artifactFor(tier, overlay);
  const memory = sources.controlPhysical.get(`${tier}/${rung}`);
  const controlQuality = sources.controlQuality.get(`${tier}/${rung}`);
  const lifecycle = sources.lifecycle;
  const loadability = overlay === "control" ? memory : overlay === "lora" ? sources.loraPhysical.get(`${tier}/resident`) : sources.stylePhysical.get(tier);
  const artifactSources = [sources.baseInventory.get(tier)];
  if (overlay === "control") artifactSources.push(sources.controlInventory);
  if (overlay === "lora") artifactSources.push(sources.loraInventory);
  const overlaySources = overlay === "control"
    ? [memory, ...artifactSources]
    : overlay === "lora"
      ? [sources.loraPhysical.get(`${tier}/resident`), sources.loraPhysical.get(`${tier}/bounded_transformer_residency`), ...artifactSources]
      : [lifecycle, ...artifactSources];
  if (!memory || !controlQuality || !loadability || artifactSources.some((source) => !source) || overlaySources.some((source) => !source) || !lifecycle) throw new Error(`missing source for ${tier}/${mode}/${overlay}/${rung}`);
  const qualitySources = [controlQuality];
  let qualityKind = overlay === "control" && mode === "text_to_image" ? "direct" : "identical_component";
  if (overlay === "lora") {
    const loraQuality = sources.loraQuality.get(tier);
    if (!loraQuality) throw new Error(`missing LoRA comparison for ${tier}`);
    qualitySources.push(loraQuality);
  }
  if (mode === "style_variations") {
    const style = sources.stylePhysical.get(tier);
    if (!style) throw new Error(`missing style mutation for ${tier}`);
    qualitySources.push(style);
  }
  const metrics = qualityMetrics(controlQuality.text, controlQuality.filename);
  const parameters = runtimeParameters(spec.cases[0].parameters);
  const prediction = predicted(tier, rung);
  if (memory.observed.overall.deviceBytes > prediction.overall) {
    throw new Error(`${tier}/${rung} observed ${memory.observed.overall.deviceBytes} exceeds prediction ${prediction.overall}`);
  }
  const record = {
    id: "", logicalCaseId: "", status: "complete", evidenceScope: spec.evidenceScope, backend: spec.backend,
    repositories: {
      sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false, matrixSourceRevision },
      inference: { revision: INFERENCE_REVISION, dirty: false },
    },
    hardware: {
      ...HARDWARE, deviceId: "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
      name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", computeCapability: "12.0",
      driverVersion: "596.36", runtimeVersion: "12.9",
    },
    artifact: { repository: artifact.repository, resolvedRevision: artifact.resolvedRevision, variant: artifact.variant },
    target: spec.target, fixture: spec.fixture,
    strategy: { rung, engagedRungs: spec.engagedRungs, parameters },
    sweep: {
      axes: Object.entries(parameters).filter(([, value]) => Number.isInteger(value)).map(([parameter, value]) => ({ parameter, testedValues: [value] })),
      cases: [{ parameters, result: "passed" }], rangeVerified: true,
    },
    scenarios: scenarios(prediction.overall, overlay),
    predictedPeakBytes: prediction, observedMemory: memory.observed,
    quality: {
      contract: "fixed seed/inputs; selected rung versus resident, with overlay/reference mutation checks",
      identicalInputs: true, result: "passed", ...metrics,
      maximumErrorThreshold: 0.15, meanErrorThreshold: 0.002,
    },
    negativeMutation: {
      parameters: { independentlyNormalizedDecodeTiles: true }, measured: true,
      result: "failed_as_expected", ...sources.negativeMutation.metrics,
    },
    loadability: { result: "passed", resolvedPathFingerprint: runtimeArtifact.fingerprint },
    diagnostics: {
      adapter: "candle-gen-z-image:sc-16170-source-session-v1", execution: "executed", blockers: [],
      measurements: [
        { name: "combinedPredecodePeak", unit: "bytes", value: memory.observed.denoise.deviceBytes },
        { name: "decodePeak", unit: "bytes", value: memory.observed.decode.deviceBytes },
        { name: "sourceSessionCount", unit: "count", value: new Set([memory.id, ...qualitySources.map((source) => source.id), lifecycle.id, loadability.id, ...overlaySources.map((source) => source.id)]).size },
      ],
    },
    derivation: {
      memory: ref(overlay === "control" ? "direct" : "conservative_upper_bound", memory),
      quality: ref(qualityKind, ...qualitySources),
      negativeMutation: ref("shared_implementation", sources.negativeMutation),
      lifecycle: ref("shared_implementation", lifecycle),
      loadability: ref("direct", loadability, ...artifactSources),
      overlay: ref(overlay === "control" || overlay === "lora" ? "direct" : "shared_implementation", ...overlaySources),
      justification: overlay === "control"
        ? "Direct strict-control high-water and rung comparison; lifecycle assertions share the same provider seams."
        : "The strict-control run is a conservative memory upper bound for the lighter overlay; same-tier component comparisons are combined with exact overlay/reference mutation and loadability sessions.",
    },
    calibrationFingerprint: spec.calibrationFingerprint,
    capturedAt: memory.session.capturedAt,
    harnessVersion: HARNESS_VERSION,
  };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  return record;
}

function indentJson(value, spaces) {
  const pad = " ".repeat(spaces);
  return JSON.stringify(value, null, 2).split("\n").map((line, index) => index === 0 ? line : `${pad}${line}`).join("\n");
}

function renderManifestBindings(source, bindings) {
  const sectionStart = source.indexOf("// Candle/CUDA ladder certification (sc-16170)");
  if (sectionStart < 0) throw new Error("cannot locate SC-16170 manifest section");
  const anchor = '        "memoryStrategyContract": {';
  const contractAt = source.indexOf(anchor, sectionStart);
  if (contractAt < 0) throw new Error("cannot locate Z-Image memoryStrategyContract");
  const existingAt = source.indexOf('        "calibrations": [', sectionStart);
  const rendered = bindings.length ? `        "calibrations": ${indentJson(bindings, 8)},\r\n` : "";
  source = existingAt >= 0 && existingAt < contractAt
    ? source.slice(0, existingAt) + rendered + source.slice(contractAt)
    : source.slice(0, contractAt) + rendered + source.slice(contractAt);
  return source;
}

async function writeManifestBindings(bindings) {
  const source = await readFile(MANIFEST, "utf8");
  await writeFile(MANIFEST, renderManifestBindings(source, bindings));
}

async function loadSources() {
  const files = new Set(await readdir(SESSION_DIR));
  const all = [];
  const controlPhysical = new Map();
  const controlQuality = new Map();
  const stylePhysical = new Map();
  const loraPhysical = new Map();
  const loraQuality = new Map();
  const baseInventory = new Map();
  for (const tier of ["q4", "q8", "bf16"]) {
    const filename = `${SESSION_PREFIX}-${tier}-base-inventory.log`;
    if (!files.has(filename)) throw new Error(`missing ${filename}`);
    const bytes = await readFile(path.join(SESSION_DIR, filename));
    const input = artifactInventory(decodeLog(bytes), filename);
    requireExpectedInventory(input, tier);
    const loaded = await readSession(filename, {
      kind: "static_analysis",
      command: `sc-16170-artifact-inventory.ps1 base ${tier}`,
      claims: ["loadability", "overlay"], inputs: [input],
    });
    loaded.id = loaded.session.id; loaded.input = input; all.push(loaded.session); baseInventory.set(tier, loaded);
  }
  const loadSharedInventory = async (name) => {
    const filename = `${SESSION_PREFIX}-${name}-inventory.log`;
    if (!files.has(filename)) throw new Error(`missing ${filename}`);
    const bytes = await readFile(path.join(SESSION_DIR, filename));
    const input = artifactInventory(decodeLog(bytes), filename);
    requireExpectedInventory(input, name);
    const loaded = await readSession(filename, {
      kind: "static_analysis", command: `sc-16170-artifact-inventory.ps1 ${name}`,
      claims: ["loadability", "overlay"], inputs: [input],
    });
    loaded.id = loaded.session.id; loaded.input = input; all.push(loaded.session); return loaded;
  };
  const controlInventory = await loadSharedInventory("control");
  const loraInventory = await loadSharedInventory("lora");
  for (const tier of ["q4", "q8", "bf16"]) {
    for (const shortRung of ["resident", "staged", "decode", "attention", "transformer"]) {
      const rung = rungName(shortRung);
      const filename = `${SESSION_PREFIX}-${tier}-control-${shortRung}.log`;
      if (!files.has(filename)) throw new Error(`missing ${filename}`);
      const loaded = await readSession(filename, {
        kind: "physical_cuda", command: `control_vram_probe tier=${tier} rung=${shortRung} 1024x1024 steps=1`,
        target: { tier, mode: "text_to_image", overlay: "control", rung },
        inputs: [baseInventory.get(tier).input, controlInventory.input],
        claims: ["memory", "lifecycle", "loadability", "overlay"],
      });
      loaded.filename = filename; loaded.observed = physicalMemory(loaded.text, filename); loaded.id = loaded.session.id;
      all.push(loaded.session); controlPhysical.set(`${tier}/${rung}`, loaded);
      const qualityFilename = `${SESSION_PREFIX}-${tier}-control-${shortRung}-quality.log`;
      if (!files.has(qualityFilename)) throw new Error(`missing ${qualityFilename}`);
      const quality = await readSession(qualityFilename, {
        kind: "comparison", command: `compare resident and ${shortRung} fixed-seed control PNGs`,
        target: { tier, mode: "text_to_image", overlay: "control", rung },
        inputs: [baseInventory.get(tier).input, controlInventory.input],
        claims: ["quality"], outputs: outputHashes(decodeLog(await readFile(path.join(SESSION_DIR, qualityFilename)))),
      });
      quality.filename = qualityFilename; quality.id = quality.session.id;
      qualityMetrics(quality.text, qualityFilename); all.push(quality.session); controlQuality.set(`${tier}/${rung}`, quality);
    }
    const styleFilename = `${SESSION_PREFIX}-${tier}-style-resident.log`;
    const style = await readSession(styleFilename, {
      kind: "physical_cuda", command: `sequential_vram_probe tier=${tier} style_variations resident 1024x1024 steps=1`,
      target: { tier, mode: "style_variations", overlay: "none", rung: "resident" },
      inputs: [baseInventory.get(tier).input],
      claims: ["quality", "loadability", "overlay"],
    });
    mutationMetrics(style.text, styleFilename); style.filename = styleFilename; style.id = style.session.id;
    all.push(style.session); stylePhysical.set(tier, style);
    for (const shortRung of ["resident", "transformer"]) {
      const rung = rungName(shortRung);
      const filename = `${SESSION_PREFIX}-${tier}-lora-${shortRung}.log`;
      const lora = await readSession(filename, {
        kind: "physical_cuda", command: `sequential_vram_probe tier=${tier} lora ${shortRung} 1024x1024 steps=1`,
        target: { tier, mode: "text_to_image", overlay: "lora", rung },
        inputs: [baseInventory.get(tier).input, loraInventory.input],
        claims: ["quality", "lifecycle", "loadability", "overlay"],
      });
      mutationMetrics(lora.text, filename); lora.filename = filename; lora.id = lora.session.id;
      all.push(lora.session); loraPhysical.set(`${tier}/${rung}`, lora);
    }
    const filename = `${SESSION_PREFIX}-${tier}-lora-transformer-quality.log`;
    const quality = await readSession(filename, {
      kind: "comparison", command: `compare resident and transformer fixed-seed LoRA PNGs`,
      target: { tier, mode: "text_to_image", overlay: "lora", rung: "bounded_transformer_residency" },
      inputs: [baseInventory.get(tier).input, loraInventory.input],
      claims: ["quality"], outputs: outputHashes(decodeLog(await readFile(path.join(SESSION_DIR, filename)))),
    });
    qualityMetrics(quality.text, filename); quality.filename = filename; quality.id = quality.session.id;
    all.push(quality.session); loraQuality.set(tier, quality);
  }
  const lifecycleLoaded = await readSession(`${SESSION_PREFIX}-inference-lifecycle.log`, {
    kind: "unit_test", command: "cargo test --locked -p candle-gen-z-image --lib --tests",
    claims: ["lifecycle", "overlay"],
  });
  if (!/test result: ok\./.test(lifecycleLoaded.text)) throw new Error("lifecycle test session did not pass");
  lifecycleLoaded.id = lifecycleLoaded.session.id; all.push(lifecycleLoaded.session);
  const negativeLoaded = await readSession("7410-negative-mutation.log", {
    kind: "comparison", command: "compare fixed-seed q4 resident output with independently normalized tiled-decode mutation",
    target: { tier: "q4", mode: "text_to_image", overlay: "control", rung: "bounded_transformer_residency" },
    inputs: [baseInventory.get("q4").input, controlInventory.input],
    claims: ["negative_mutation"], inferenceRevision: "9d639b2be8647fe210415d9d8dba98644169f11c",
    outputs: outputHashes(decodeLog(await readFile(path.join(SESSION_DIR, "7410-negative-mutation.log")))),
  });
  negativeLoaded.metrics = qualityMetrics(negativeLoaded.text, "7410-negative-mutation.log", false);
  negativeLoaded.id = negativeLoaded.session.id; all.push(negativeLoaded.session);
  return { all, baseInventory, controlInventory, loraInventory, controlPhysical, controlQuality, stylePhysical, loraPhysical, loraQuality, lifecycle: lifecycleLoaded, negativeMutation: negativeLoaded };
}

async function main() {
  const plan = JSON.parse(await readFile(PLAN, "utf8"));
  const existing = JSON.parse(await readFile(EVIDENCE, "utf8"));
  const matrix = JSON.parse(await readFile(MATRIX, "utf8"));
  const matrixSourceRevision = matrix.generatedFrom.sceneWorksRevision;
  const specs = plan.providers.filter((spec) => spec.backend === "candle" && spec.target?.modelId === "z_image" && SC16170_FINGERPRINTS.has(spec.calibrationFingerprint));
  if (specs.length !== 90) throw new Error(`expected 90 SC-16170 plan cases, found ${specs.length}`);
  const retainedRecords = existing.records.filter((record) => !(record.backend === "candle" && record.target?.modelId === "z_image"));
  if (process.argv.includes("--clear")) {
    const bundle = certificationBundle(existing, retainedRecords);
    await writeFile(EVIDENCE, canonicalJson(bundle));
    await writeManifestBindings([]);
    console.log("removed SC-16170 promoted records and bindings");
    return;
  }
  const sources = await loadSources();
  // Schema v4 requires a measured loadShape on every record. These captures predate that axis, so
  // continue parsing every source session and constructing every historical case, but do not invent a
  // shape or publish ABI-1 bindings into the current authoritative bundle.
  const historicalRecords = specs.map((spec) => makeRecord(spec, matrixSourceRevision, sources));
  if (historicalRecords.length !== 90) throw new Error("SC-16170 historical record population changed");
  const bundle = certificationBundle(existing, retainedRecords);
  const bindings = [];
  const renderedEvidence = canonicalJson(bundle);
  if (process.argv.includes("--write")) {
    await writeFile(EVIDENCE, renderedEvidence);
    await writeManifestBindings(bindings);
    console.log(`validated ${historicalRecords.length} historical SC-16170 cases from ${sources.all.length} source sessions; left unpromoted because loadShape was not measured`);
  } else {
    const checkedEvidence = JSON.parse(await readFile(EVIDENCE, "utf8"));
    assertCanonicalBundleMatches(checkedEvidence, bundle);
    const checkedManifest = await readFile(MANIFEST, "utf8");
    const manifestWithoutBindings = renderManifestBindings(checkedManifest, []);
    const expectedManifest = renderManifestBindings(manifestWithoutBindings, bindings);
    if (checkedManifest !== expectedManifest) throw new Error("checked-in manifest still contains ABI-1 SC-16170 bindings; run with --write");
    console.log(`validated ${historicalRecords.length} historical SC-16170 cases from ${sources.all.length} committed source sessions; current bundle remains fail-closed`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
