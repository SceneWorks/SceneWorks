import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmod, mkdir, mkdtemp, readFile, readdir, stat, symlink, writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";

import {
  CARGO_METADATA_TIMEOUT_MS,
  CAMPAIGN_ENTRY_FIXTURE,
  CAMPAIGN_ENTRY_IDENTITY,
  CAMPAIGN_ENTRY_LOGICAL_CASE_ID,
  CAMPAIGN_ENTRY_PROVIDER,
  CAMPAIGN_FAILURE_RECEIPT_TYPE,
  CHILD_ATTESTATION_TIMEOUT_SECONDS,
  CanaryInterrupted,
  MAX_FOOTPRINT_BYTES,
  MAX_RUNTIME_SECONDS,
  MIN_PREFLIGHT_FREE_BYTES,
  MIN_RUNTIME_FREE_BYTES,
  PREPARATION_SCHEMA_VERSION,
  PROVIDER_PHASES,
  PRODUCT_ENVELOPE_CANARY_PROFILE,
  acquireCampaignEntryOutcome,
  acquirePreparationLock,
  campaignEntryAdapterRequest,
  campaignEntryCanonicalFragment,
  campaignEntryFailurePath,
  campaignEntryFailureReceipt,
  campaignEntryOutcomeReservationPath,
  campaignEntryPlan,
  canaryWatchdogEnvironment,
  canaryRequest,
  cargoSourceStatusIsClean,
  cleanupCanaryScratch,
  cloneArtifactTree,
  exactToolchain,
  foreignHeavyProcesses,
  inventoryAtRoot,
  locateBuiltMetallib,
  parseMemoryFreePercent,
  parseSwapFreeBytes,
  preparationCacheKey,
  preparationIdentity,
  preparationLockPath,
  prepareCanaryCache,
  privateArtifactRoots,
  preservePrimaryFailure,
  preserveFailureReceiptSuppression,
  publishCampaignEntryCanonicalOutcome,
  publishCampaignEntryFailureReceipt,
  releaseUnpublishedCampaignEntryOutcome,
  preflightFreeFloor,
  repositoryToolchain,
  runtimeFreeFloor,
  runOwnedCommand,
  sealedArtifactIdentity,
  telemetryResolutionBytes,
  validateCampaignEntryAdapterResponse,
  validateCampaignEntryBundle,
  validateCampaignEntryFailureEvents,
  validateCampaignEntryFailureReceipt,
  validateCampaignEntryHarnessRequest,
  validateCampaignEntryWatchdogEvents,
  validateCanaryResponse,
  validatePreparedCache,
  watchdogFailureSummary,
} from "./run-ltx-safety-canary.mjs";

import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
import { canonicalJson, runProviderPlan, validateBundle } from "./memory-calibration-harness.mjs";

const TEXT_ENCODER_INVENTORY = {
  root: "/models/gemma",
  files: 17,
  bytes: 26_427_894_918,
  sha256: "abde2d155aa8991747cc2999d40688d29a50261c080c0d51fac20357653928d7",
};

async function campaignEntryFixture() {
  const config = JSON.parse(await readFile(new URL(
    "../docs/calibration/sc-18946/ltx-mlx-single-pass-sweep.json",
    import.meta.url,
  ), "utf8"));
  const { provider, planned } = campaignEntryPlan(config);
  const request = {
    action: "run",
    planned,
    repositories: {
      sceneWorks: {
        revision: "1".repeat(40), dirty: false, matrixSourceRevision: "1".repeat(40),
      },
      inference: { revision: "2".repeat(40), dirty: false, closureDigest: "3".repeat(64) },
    },
    repositoryPaths: { sceneWorks: "/scene", inference: "/inference" },
    hardware: { probe: "mlx", memoryBytes: 128 * 1024 ** 3 },
  };
  return { config, provider, planned, request };
}

function campaignEntryResponse(hostMemoryBytes = 128 * 1024 ** 3) {
  const diagnostics = [
    ["renderedFrames", 121], ["outputFps", 30], ["audioTrackDecoded", 1],
    ["decodeTilingEngaged", 0], ["decodeTileSpatialPx", 0],
    ["decodeTileOverlapPx", 0], ["latentTemporalDepth", 16], ["latentTokens", 6_144],
  ].map(([name, value]) => ({ name, unit: "count", value }));
  return {
    status: "runtime_complete",
    loadShape: "eager_materialization",
    artifact: {
      repository: "SceneWorks/ltx-2.3-mlx",
      resolvedRevision: "01df27d308466533aa09d251e3aebdcc627d07eb",
      variant: "q4",
    },
    strategy: {
      rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
    },
    output: {
      frames: 121, fps: 30,
      audio: { present: true, samples: 48_000, sampleRate: 48_000, channels: 2 },
      firstFrameNondegenerate: true,
    },
    diagnostics: {
      adapter: "memory-mlx-adapter:ltx-2-3-provider-contract-video",
      execution: "executed",
      blockers: [],
      measurements: diagnostics,
    },
    _campaignEntry: {
      identity: CAMPAIGN_ENTRY_IDENTITY,
      inferenceRevision: "2".repeat(40),
      watchdog: {
        required: true,
        protocol: "sceneworks-memory-watchdog-v1",
        maxFootprintBytes: MAX_FOOTPRINT_BYTES,
        maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
        hostMemoryBytes,
        minInitialMemoryFreeBytes: preflightFreeFloor(hostMemoryBytes),
        minMemoryFreeBytes: runtimeFreeFloor(hostMemoryBytes),
      },
      mlxLimits: { memoryLimitBytes: MAX_FOOTPRINT_BYTES, wiredLimitBytes: MAX_FOOTPRINT_BYTES },
      cleanup: {
        preProviderActiveBytes: 0,
        preProviderCacheBytes: 0,
        expectedPersistentActive: {
          identity: "mlx-gen-ltx-transformer-ones-cache-av-bfloat16-v1",
          videoDimension: 4_096,
          audioDimension: 2_048,
          dtype: "bfloat16",
          bytesPerElement: 2,
          bytes: 12_288,
        },
        postCleanupActiveBytes: 12_288,
        postCleanupCacheBytes: 0,
      },
    },
  };
}

function campaignEntryRuntimeResponse(hostMemoryBytes = 128 * 1024 ** 3) {
  const response = campaignEntryResponse(hostMemoryBytes);
  const phase = (value) => ({
    activeBytes: value, allocatorBytes: value + 10, reclaimableBytes: 10,
  });
  return {
    ...response,
    sweep: { axes: [], cases: [{ parameters: {}, result: "passed" }], rangeVerified: true },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: 200, effectiveBudgetBytes: 200 },
      { name: "unknown_budget", result: "passed", reason: "unknown budget rejected" },
      { name: "stale_evidence", result: "passed", reason: "stale evidence rejected" },
      { name: "warm_repeat", result: "passed", reason: "warm repeat stayed deterministic" },
      {
        name: "cancel", result: "passed", reason: "cancel cleaned and recovered",
        cleanupVerified: true, warmFollowUpPassed: true,
      },
      {
        name: "error", result: "passed", reason: "error cleaned and recovered",
        cleanupVerified: true, warmFollowUpPassed: true,
      },
      { name: "loadability", result: "passed" },
      { name: "overlay", result: "not_applicable", reason: "base-only runtime record" },
    ],
    predictedPeakBytes: { conditioning: 100, denoise: 200, decode: 150, overall: 200 },
    observedMemory: {
      conditioning: phase(100), denoise: phase(200), decode: phase(150), overall: phase(200),
    },
    quality: {
      contract: "exact campaign-entry repeat determinism",
      identicalInputs: true,
      result: "passed",
      maximumError: 0,
      meanError: 0,
      rootMeanSquareError: 0,
      maximumErrorThreshold: 0.08,
      meanErrorThreshold: 0.01,
      rootMeanSquareErrorThreshold: 0.02,
    },
    negativeMutation: null,
    loadability: { result: "passed", resolvedPathFingerprint: "exact@campaign-entry:q4" },
    capturedAt: "2026-08-17T12:00:00Z",
  };
}

const PROCESS_IDENTITIES = [
  { pid: 1234, pgid: 1234, started: "Sun Aug 17 12:00:00 2026" },
  { pid: 1235, pgid: 1234, started: "Sun Aug 17 12:00:01 2026" },
];

function stableCompactJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableCompactJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableCompactJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function chainWatchdogEvents(events) {
  let previousEventHash = "0".repeat(64);
  for (const [index, event] of events.entries()) {
    delete event.eventSequence;
    delete event.previousEventHash;
    delete event.eventHash;
    Object.assign(event, { eventSequence: index + 1, previousEventHash });
    event.eventHash = createHash("sha256").update(stableCompactJson(event)).digest("hex");
    previousEventHash = event.eventHash;
  }
  return events;
}

function campaignFailureEvents(hostMemoryBytes = 128 * 1024 ** 3, phaseCount = 1) {
  const runtimeFloor = runtimeFreeFloor(hostMemoryBytes);
  let at = 1;
  const events = [
    {
      at: at++, event: "started", pid: 1234, pgid: 1234,
      providerPhase: null, processIdentities: PROCESS_IDENTITIES,
    },
    {
      at: at++, event: "sample", phase: "before_child_release", providerPhase: null,
      physicalFootprintBytes: 1, memoryFreeBytes: runtimeFloor, swapFreeBytes: 0,
    },
    {
      at: at++, event: "sample", phase: "child_attested_before_allocation", providerPhase: null,
      physicalFootprintBytes: 2, memoryFreeBytes: runtimeFloor, swapFreeBytes: 0,
    },
    { at: at++, event: "child_attested", providerPhase: null },
  ];
  let providerPhase = null;
  for (let index = 0; index < phaseCount; index += 1) {
    providerPhase = { sequence: index + 1, name: PROVIDER_PHASES[index] };
    events.push({
      at: at++, event: "provider_phase", providerPhase, authenticated: true,
    });
  }
  const physicalFootprintBytes = MAX_FOOTPRINT_BYTES + 1;
  const reason = `physical_footprint_at_or_above_${MAX_FOOTPRINT_BYTES}:observed_${physicalFootprintBytes}`;
  events.push({
    at: at++, event: "sample", phase: "runtime", providerPhase,
    physicalFootprintBytes, memoryFreeBytes: runtimeFloor, swapFreeBytes: 0,
  });
  events.push({
    at: at++, event: "hard_stop", reason, providerPhase,
    processIdentities: PROCESS_IDENTITIES,
  });
  events.push({
    at: at++, event: "terminated", reason, providerPhase,
    processIdentities: PROCESS_IDENTITIES,
  });
  return chainWatchdogEvents(events);
}

function campaignSuccessEvents(hostMemoryBytes = 128 * 1024 ** 3) {
  const events = campaignFailureEvents(hostMemoryBytes, PROVIDER_PHASES.length);
  const sample = events.at(-3);
  sample.physicalFootprintBytes = 3;
  events.splice(-2, 2, {
    at: events.at(-2).at, event: "child_completed",
    providerPhase: { sequence: PROVIDER_PHASES.length, name: PROVIDER_PHASES.at(-1) },
  });
  return chainWatchdogEvents(events);
}

function campaignFailurePreparation(
  root = "/private/prepared", sceneWorksRevision = "1".repeat(40),
  inferenceRevision = "3".repeat(40),
) {
  const identity = preparationIdentity("a".repeat(40), "b".repeat(40), "1.97.1");
  const key = preparationCacheKey(identity);
  const preparationRoot = path.join(root, key);
  const fileIdentity = (name, mode, digest) => ({
    path: path.join(preparationRoot, "adapter", name), sha256: digest.repeat(64),
    device: "1", inode: name === "mlx.metallib" ? "2" : "3", size: 100,
    mtimeNs: "4", ctimeNs: "5", mode,
  });
  const prepared = {
    adapter: fileIdentity("memory-mlx-adapter", 0o500, "d"),
    metallib: fileIdentity("mlx.metallib", 0o400, "c"),
  };
  const manifestIdentity = (file) => ({
    sha256: file.sha256, size: file.size,
    seal: {
      device: file.device, inode: file.inode, mtimeNs: file.mtimeNs,
      ctimeNs: file.ctimeNs, mode: file.mode,
    },
  });
  prepared.manifest = {
    schemaVersion: PREPARATION_SCHEMA_VERSION, key, identity,
    preparedFrom: { sceneWorksRevision, inferenceRevision },
    artifacts: {
      numericTier: { content: identity.artifact.numericTier, seal: {} },
      textEncoder: { content: identity.artifact.textEncoder, seal: {} },
    },
    adapter: manifestIdentity(prepared.adapter),
    metallib: manifestIdentity(prepared.metallib),
  };
  return { identity, key, preparationRoot, prepared };
}

async function cleanCampaignHarnessRepo(t) {
  const root = await mkdtemp(path.join(tmpdir(), "sc20191-harness-repo-"));
  t.after(() => cleanupCanaryScratch(root));
  await mkdir(path.join(root, "docs/generated"), { recursive: true });
  await writeFile(path.join(root, "docs/generated/memory-matrix.json"), JSON.stringify({
    generatedFrom: { sceneWorksRevision: `source-tree:${"a".repeat(64)}` },
  }));
  for (const args of [
    ["init", root],
    ["-C", root, "config", "user.email", "fixture@example.invalid"],
    ["-C", root, "config", "user.name", "Fixture"],
    ["-C", root, "add", "docs/generated/memory-matrix.json"],
    ["-C", root, "commit", "-m", "fixture"],
  ]) {
    const result = spawnSync("git", args, { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
  }
  return root;
}

async function cacheFixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "sc19741-cache-test-"));
  t.after(() => cleanupCanaryScratch(root));
  const sources = {
    numericTier: path.join(root, "source-q4"),
    textEncoder: path.join(root, "source-gemma"),
  };
  await mkdir(sources.numericTier);
  await mkdir(sources.textEncoder);
  await writeFile(path.join(sources.numericTier, "weights.safetensors"), "numeric-tier\n");
  await writeFile(path.join(sources.textEncoder, "encoder.safetensors"), "text-encoder\n");
  const [numericTier, textEncoder] = await Promise.all([
    hashArtifactInventory(sources.numericTier),
    hashArtifactInventory(sources.textEncoder),
  ]);
  const identity = preparationIdentity("1".repeat(40), "2".repeat(40), "1.97.1");
  identity.artifact.numericTier = {
    files: numericTier.files, bytes: numericTier.bytes, sha256: numericTier.sha256,
  };
  identity.artifact.textEncoder = {
    files: textEncoder.files, bytes: textEncoder.bytes, sha256: textEncoder.sha256,
  };
  const key = preparationCacheKey(identity);
  const preparationRoot = path.join(root, "prepared", key);
  let builds = 0;
  const build = async (stage, signal, hooks = {}) => {
    builds += 1;
    const device = (await stat(stage, { bigint: true })).dev;
    const roots = privateArtifactRoots(stage);
    await mkdir(path.dirname(roots.numericTier), { recursive: true, mode: 0o700 });
    await cloneArtifactTree(sources.numericTier, roots.numericTier, device, signal);
    const preparedNumeric = await hashArtifactInventory(roots.numericTier, { signal });
    await hooks.afterNumericClone?.(stage, signal);
    await cloneArtifactTree(sources.textEncoder, roots.textEncoder, device, signal);
    const preparedText = await hashArtifactInventory(roots.textEncoder, { signal });
    await hooks.afterArtifactClone?.(stage, signal);
    const adapterDirectory = path.join(stage, "adapter");
    await mkdir(adapterDirectory, { mode: 0o700 });
    await writeFile(path.join(adapterDirectory, "memory-mlx-adapter"), "adapter\n", { mode: 0o500 });
    await writeFile(path.join(adapterDirectory, "mlx.metallib"), "metallib\n", { mode: 0o400 });
    await chmod(path.join(adapterDirectory, "memory-mlx-adapter"), 0o500);
    await chmod(path.join(adapterDirectory, "mlx.metallib"), 0o400);
    await chmod(adapterDirectory, 0o500);
    return {
      artifacts: { numericTier: preparedNumeric, textEncoder: preparedText },
    };
  };
  const options = {
    preparationRoot,
    key,
    identity,
    preparedFrom: {
      sceneWorksRevision: "3".repeat(40),
      inferenceRevision: "4".repeat(40),
    },
    build,
  };
  return { root, sources, identity, key, preparationRoot, options, build, builds: () => builds };
}

test("the runner freezes the exact tiny bounded-decode canary", () => {
  const request = canaryRequest(128 * 1024 ** 3, TEXT_ENCODER_INVENTORY);
  assert.equal(request.action, "canary");
  assert.equal(request.planned._diagnosticOnly, true);
  assert.equal(request.planned.evidenceScope, "fixture");
  assert.deepEqual(request.planned.target.geometry, {
    width: 256, height: 256, batch: 1, frames: 9,
  });
  assert.deepEqual(request.planned.strategy.parameters, {
    decodeTileEdge: 192, decodeOverlap: 64,
  });
  assert.deepEqual(request.planned._canary, {
    identity: "sc-19741-safety", videoMode: "no_audio", fps: 24, seed: 1234,
  });
  assert.equal(request.planned._watchdog.maxFootprintBytes, 53_347_146_863);
  assert.deepEqual(request.planned._artifact.numericTierInventory, {
    files: 11,
    bytes: 20_467_690_460,
    sha256: "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a",
  });
  assert.throws(() => canaryRequest(128 * 1024 ** 3));
});

test("SC-20191 transforms only the exact canonical harness row without changing its identity", async () => {
  const { config, provider, planned, request } = await campaignEntryFixture();
  assert.equal(planned.logicalCaseId, CAMPAIGN_ENTRY_LOGICAL_CASE_ID);
  assert.equal(planned.fixture, CAMPAIGN_ENTRY_FIXTURE);
  assert.equal(provider.name, CAMPAIGN_ENTRY_PROVIDER);
  assert.deepEqual(planned.target.geometry, {
    width: 768, height: 512, batch: 1, frames: 121,
  });
  assert.deepEqual(planned.strategy, {
    rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
  });
  const original = structuredClone(request);
  const expectedSource = {
    sceneWorksRevision: "1".repeat(40),
    inferenceRevision: "2".repeat(40),
    sceneWorksRepo: "/scene",
    inferenceRepo: "/inference",
  };
  validateCampaignEntryHarnessRequest(request, planned, expectedSource);
  const adapter = campaignEntryAdapterRequest(request, planned, expectedSource);
  assert.deepEqual(request, original, "the canonical harness request must remain untouched");
  assert.equal(adapter.action, "campaign_entry");
  assert.equal(adapter.planned.logicalCaseId, CAMPAIGN_ENTRY_LOGICAL_CASE_ID);
  assert.equal(adapter.planned._campaignEntry.identity, CAMPAIGN_ENTRY_IDENTITY);
  assert.equal(adapter.planned._watchdog.maxFootprintBytes, MAX_FOOTPRINT_BYTES);
  const mutatedConfig = structuredClone(config);
  mutatedConfig.providers.find((entry) => entry.name === CAMPAIGN_ENTRY_PROVIDER)
    .target.geometry.frames = 97;
  assert.throws(() => campaignEntryPlan(mutatedConfig), /frozen campaign-entry provider changed/);

  for (const mutate of [
    (value) => { value.action = "run_batch"; },
    (value) => { value.planned.logicalCaseId = "implan-mutated"; },
    (value) => { value.planned.target.geometry.frames = 97; },
    (value) => { value.planned.target.tier = "q8"; },
    (value) => { value.planned.fixture = CAMPAIGN_ENTRY_FIXTURE.replace("seed18946", "seed1"); },
    (value) => { value.planned.strategy.parameters = { decodeTileEdge: 192 }; },
    (value) => { value.planned.evidenceScope = "fixture"; },
    (value) => { value.planned.modelLoadPolicy = "batch_rungs"; },
    (value) => { value.repositories.sceneWorks.dirty = true; },
    (value) => { value.repositories.inference.revision = "4".repeat(40); },
    (value) => { value.repositoryPaths.sceneWorks = "/foreign"; },
    (value) => { value.hardware.memoryBytes = 0; },
    (value) => { value.planned._watchdog = {}; },
  ]) {
    const mutated = structuredClone(request);
    mutate(mutated);
    assert.throws(
      () => campaignEntryAdapterRequest(mutated, planned, expectedSource),
      /SC-20191|canonical SC-18946/,
    );
  }
});

test("SC-20191 rejects carrier, cleanup and watchdog mutations", () => {
  const hostMemoryBytes = 128 * 1024 ** 3;
  const response = campaignEntryResponse(hostMemoryBytes);
  validateCampaignEntryAdapterResponse(response, {
    inferenceRevision: "2".repeat(40), hostMemoryBytes,
  });
  for (const mutate of [
    (value) => { value.output.audio.present = false; },
    (value) => { value.output.audio.samples = 0; },
    (value) => { value.output.frames = 120; },
    (value) => { value.strategy.parameters = { decodeTileEdge: 192 }; },
    (value) => { value.diagnostics.measurements.find((entry) =>
      entry.name === "decodeTilingEngaged").value = 1; },
    (value) => { value.diagnostics.measurements.find((entry) =>
      entry.name === "latentTokens").value = 6_143; },
    (value) => { value._campaignEntry.watchdog.maxFootprintBytes -= 1; },
    (value) => { value._campaignEntry.watchdog.minSwapFreeBytes = 1; },
    (value) => { value._campaignEntry.mlxLimits.memoryLimitBytes -= 1; },
    (value) => { value._campaignEntry.cleanup.expectedPersistentActive.bytes += 1; },
    (value) => { value._campaignEntry.cleanup.postCleanupActiveBytes += 1; },
    (value) => { value._campaignEntry.cleanup.postCleanupCacheBytes = 1; },
  ]) {
    const mutated = structuredClone(response);
    mutate(mutated);
    assert.throws(
      () => validateCampaignEntryAdapterResponse(mutated, {
        inferenceRevision: "2".repeat(40), hostMemoryBytes,
      }),
      /SC-20191/,
    );
  }

  const runtimeFloor = runtimeFreeFloor(hostMemoryBytes);
  const validEvents = campaignSuccessEvents(hostMemoryBytes);
  assert.equal(validateCampaignEntryWatchdogEvents(validEvents, hostMemoryBytes), 3);
  for (const events of [
    validEvents.filter((event) => event.event !== "child_completed"),
    validEvents.filter((event) => event.phase !== "runtime"),
    validEvents.filter((event) => event.providerPhase?.name !== "lifecycle_cancel"),
    [...validEvents.slice(0, -1), structuredClone(validEvents.find((event) =>
      event.providerPhase?.name === "cleanup" && event.event === "provider_phase")), validEvents.at(-1)],
    [validEvents[0], ...validEvents.slice(1).reverse()],
    [...validEvents, { event: "hard_stop" }],
    validEvents.map((event) => event.phase === "before_child_release"
      ? { ...event, physicalFootprintBytes: MAX_FOOTPRINT_BYTES } : event),
    validEvents.map((event) => event.phase === "before_child_release"
      ? { ...event, memoryFreeBytes: runtimeFloor - 1 } : event),
  ]) assert.throws(() => validateCampaignEntryWatchdogEvents(events, hostMemoryBytes), /SC-20(?:191|216)/);
});

test("SC-20216 failure receipt is phase-authenticated, non-ingestible and mutation-sensitive", () => {
  const hostMemoryBytes = 128 * 1024 ** 3;
  const { identity, key, preparationRoot, prepared } = campaignFailurePreparation();
  const events = campaignFailureEvents(hostMemoryBytes);
  const canonicalOutput = "/private/results/canonical.json";
  const outcome = {
    canonicalOutput,
    failureOutput: campaignEntryFailurePath(canonicalOutput),
    reservation: campaignEntryOutcomeReservationPath(canonicalOutput),
    choice: `${campaignEntryOutcomeReservationPath(canonicalOutput)}.choice`,
    outcomeChoice: "failure",
    canonicalBundleAbsentAtPublication: true,
    outcomeReservationHeldAtPublication: true,
  };
  const receipt = campaignEntryFailureReceipt({
    sceneWorksRevision: "1".repeat(40), sceneWorksTree: "a".repeat(40),
    inferenceRevision: "3".repeat(40), inferenceTree: "b".repeat(40),
    identity, preparationKey: key, preparationRoot,
    prepared, hostMemoryBytes, events, outcome,
  });
  assert.equal(receipt.recordType, CAMPAIGN_FAILURE_RECEIPT_TYPE);
  assert.equal(receipt.ingestible, false);
  assert.equal(receipt.canonicalBundlePublished, false);
  assert.equal(receipt.watchdog.failure.terminalProviderPhase.name, "common_load");
  assert.equal(receipt.watchdog.failure.firstViolatingSample.physicalFootprintBytes,
    MAX_FOOTPRINT_BYTES + 1);
  assert.throws(() => validateBundle(receipt), /schema|object|property|records|required/i);
  validateCampaignEntryFailureReceipt(receipt);

  const mutations = [
    (value) => { value.ingestible = true; },
    (value) => { value.outcome.canonicalBundleAbsentAtPublication = false; },
    (value) => { value.outcome.outcomeChoice = "canonical"; },
    (value) => { value.outcome.choice += ".mutated"; },
    (value) => { value.cleanup.runRootEmpty = false; },
    (value) => { value.cleanup = {}; },
    (value) => { value.campaignCase.target.geometry.frames = 120; },
    (value) => { value.artifacts.metallib.sha256 = "0".repeat(64); },
    (value) => { value.artifacts.preparation.manifest.preparedFrom.sceneWorksRevision
      = "9".repeat(40); },
    (value) => { value.artifacts.preparation.manifest.artifacts.numericTier.content.bytes += 1; },
    (value) => { value.artifacts.preparation.root = value.artifacts.preparation.key; },
    (value) => { delete value.watchdog.events[0].providerPhase; },
    (value) => { value.watchdog.events.at(-1).processIdentities
      = [value.watchdog.events.at(-1).processIdentities[0]]; },
    (value) => { value.watchdog.events.find((event) => event.event === "provider_phase")
      .providerPhase.sequence = 2; },
    (value) => { value.watchdog.events.find((event) => event.event === "provider_phase")
      .providerPhase.name = "primary_decode"; },
    (value) => { value.watchdog.events.find((event) => event.event === "provider_phase")
      .authenticated = false; },
    (value) => { value.watchdog.events.find((event) => event.event === "hard_stop")
      .providerPhase = null; },
    (value) => { value.watchdog.events.find((event) => event.event === "hard_stop")
      .reason = "physical_footprint_at_or_above_mutated"; },
    (value) => { value.watchdog.events.at(-1).at = 0; },
    (value) => { value.watchdog.eventChain.head = "0".repeat(64); },
    (value) => { value.watchdog.events.splice(1, 1); },
    (value) => { [value.watchdog.events[1], value.watchdog.events[2]]
      = [value.watchdog.events[2], value.watchdog.events[1]]; },
    (value) => { value.watchdog.events.splice(2, 0, structuredClone(value.watchdog.events[1])); },
    (value) => { value.watchdog.failure.firstViolatingEventIndex += 1; },
  ];
  for (const mutate of mutations) {
    const changed = structuredClone(receipt);
    mutate(changed);
    assert.throws(() => validateCampaignEntryFailureReceipt(changed), /SC-20216/);
  }

  const interrupted = campaignFailureEvents(hostMemoryBytes);
  interrupted.at(-3).physicalFootprintBytes = 3;
  interrupted.at(-2).reason = "monitor_signal_SIGTERM";
  interrupted.at(-1).reason = "monitor_signal_SIGTERM";
  chainWatchdogEvents(interrupted);
  const interruption = validateCampaignEntryFailureEvents(interrupted, hostMemoryBytes);
  assert.equal(interruption.firstViolatingSample, null);
  assert.equal(interruption.reason, "monitor_signal_SIGTERM");

  const thresholdWithoutSample = campaignFailureEvents(hostMemoryBytes);
  thresholdWithoutSample.at(-3).physicalFootprintBytes = 3;
  chainWatchdogEvents(thresholdWithoutSample);
  assert.throws(
    () => validateCampaignEntryFailureEvents(thresholdWithoutSample, hostMemoryBytes),
    /threshold hard stop omitted its exact first violating sample/,
  );

  const unequalTerminalIdentities = campaignFailureEvents(hostMemoryBytes);
  unequalTerminalIdentities.at(-2).processIdentities = [
    ...unequalTerminalIdentities.at(-2).processIdentities,
    { pid: 1236, pgid: 1234, started: "Sun Aug 17 12:00:02 2026" },
  ];
  chainWatchdogEvents(unequalTerminalIdentities);
  assert.throws(
    () => validateCampaignEntryFailureEvents(unequalTerminalIdentities, hostMemoryBytes),
    /exact process termination identity/,
  );
});

test("SC-20216 publishes only after validation and never overwrites a failure receipt", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "sc20216-failure-publish-"));
  t.after(() => cleanupCanaryScratch(root));
  const canonical = path.join(root, "canonical.json");
  const outcome = await acquireCampaignEntryOutcome(canonical);
  const output = outcome.failureOutput;
  const hostMemoryBytes = 128 * 1024 ** 3;
  const { identity, key, preparationRoot, prepared } = campaignFailurePreparation(root);
  const order = [];
  const build = (publication) => {
    order.push("build");
    return campaignEntryFailureReceipt({
      sceneWorksRevision: "1".repeat(40), sceneWorksTree: "a".repeat(40),
      inferenceRevision: "3".repeat(40), inferenceTree: "b".repeat(40),
      identity, preparationKey: key, preparationRoot,
      prepared, hostMemoryBytes, events: campaignFailureEvents(hostMemoryBytes),
      outcome: publication,
    });
  };
  await assert.rejects(() => publishCampaignEntryFailureReceipt(outcome, {
    verify: async () => { throw new Error("residue remained"); }, build,
  }), /residue remained/);
  assert.equal(await readFile(output, "utf8").catch(() => null), null);
  assert.deepEqual(order, []);
  await publishCampaignEntryFailureReceipt(outcome, {
    verify: async () => { order.push("cleanup-and-identity-verified"); }, build,
  });
  order.push("published");
  assert.deepEqual(order, ["cleanup-and-identity-verified", "build", "published"]);
  assert.equal(await readFile(canonical, "utf8").catch(() => null), null);
  validateCampaignEntryFailureReceipt(JSON.parse(await readFile(output, "utf8")));
  await assert.rejects(() => publishCampaignEntryFailureReceipt(outcome, {
    verify: async () => {}, build,
  }), /EEXIST/);
  await assert.rejects(() => acquireCampaignEntryOutcome(canonical), /EEXIST/);
});

test("SC-20216 jointly reserves canonical and failure outcomes across races and stale state", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "sc20216-outcome-race-"));
  t.after(() => cleanupCanaryScratch(root));
  const canonical = path.join(root, "canonical.json");
  const raced = await Promise.allSettled([
    acquireCampaignEntryOutcome(canonical), acquireCampaignEntryOutcome(canonical),
  ]);
  assert.equal(raced.filter((result) => result.status === "fulfilled").length, 1);
  assert.equal(raced.filter((result) => result.status === "rejected").length, 1);
  const outcome = raced.find((result) => result.status === "fulfilled").value;
  const hostMemoryBytes = 128 * 1024 ** 3;
  const { identity, key, preparationRoot, prepared } = campaignFailurePreparation(root);
  const publications = await Promise.allSettled([
    publishCampaignEntryCanonicalOutcome(outcome, { canonical: true }),
    publishCampaignEntryFailureReceipt(outcome, {
      verify: async () => {},
      build: (publication) => campaignEntryFailureReceipt({
        sceneWorksRevision: "1".repeat(40), sceneWorksTree: "a".repeat(40),
        inferenceRevision: "3".repeat(40), inferenceTree: "b".repeat(40),
        identity, preparationKey: key, preparationRoot, prepared, hostMemoryBytes,
        events: campaignFailureEvents(hostMemoryBytes), outcome: publication,
      }),
    }),
  ]);
  assert.equal(publications.filter((result) => result.status === "fulfilled").length, 1);
  assert.equal(publications.filter((result) => result.status === "rejected").length, 1);
  const [canonicalBytes, failureBytes] = await Promise.all([
    readFile(canonical, "utf8").catch(() => null),
    readFile(outcome.failureOutput, "utf8").catch(() => null),
  ]);
  assert.notEqual(canonicalBytes === null, failureBytes === null,
    "exactly one jointly reserved outcome must publish");

  const retryDirectory = path.join(root, "retry");
  await mkdir(retryDirectory);
  const retry = path.join(retryDirectory, "retry.json");
  const unpublished = await acquireCampaignEntryOutcome(retry);
  await releaseUnpublishedCampaignEntryOutcome(unpublished);
  const reacquired = await acquireCampaignEntryOutcome(retry);
  await releaseUnpublishedCampaignEntryOutcome(reacquired);
  await writeFile(campaignEntryOutcomeReservationPath(retry), "stale-owner\n", { flag: "wx" });
  await assert.rejects(() => acquireCampaignEntryOutcome(retry), /EEXIST/);

  const partialDirectory = path.join(root, "partial");
  await mkdir(partialDirectory);
  const partialOutput = path.join(partialDirectory, "canonical.json");
  const partial = await acquireCampaignEntryOutcome(partialOutput);
  await assert.rejects(() => publishCampaignEntryCanonicalOutcome(partial, { invalid: 1n }),
    /BigInt/);
  await releaseUnpublishedCampaignEntryOutcome(partial);
  await assert.rejects(() => acquireCampaignEntryOutcome(partialOutput), /EEXIST/,
    "a claimed but interrupted outcome must remain fail-closed");
});

test("SC-20191 publishes one schema-valid canonical runtime record after stripping private evidence", async (t) => {
  const { provider } = await campaignEntryFixture();
  const repo = await cleanCampaignHarnessRepo(t);
  let runRequests = 0;
  const result = await runProviderPlan({
    config: { providers: [provider] },
    providerCommand: ["synthetic-contained-provider"],
    sceneWorksRepo: repo,
    inferenceRepo: repo,
    backend: "mlx",
    providerName: CAMPAIGN_ENTRY_PROVIDER,
    closureDigestFor: async () => "b".repeat(64),
    executeProvider: async (_command, _args, input) => {
      const request = JSON.parse(input);
      if (request.action === "probe") {
        return canonicalJson({
          hardware: {
            probe: "synthetic contained MLX provider",
            memoryBytes: 128 * 1024 ** 3,
            model: "MacFixture",
            chip: "Apple Fixture",
            osVersion: "macOS fixture",
            metalDevice: "Apple Fixture",
            mlxMemoryLimitBytes: 96 * 1024 ** 3,
            wiredLimitBytes: 64 * 1024 ** 3,
          },
        });
      }
      assert.equal(request.action, "run");
      runRequests += 1;
      const response = campaignEntryRuntimeResponse(request.hardware.memoryBytes);
      response._campaignEntry.inferenceRevision = request.repositories.inference.revision;
      validateCampaignEntryAdapterResponse(response, {
        inferenceRevision: request.repositories.inference.revision,
        hostMemoryBytes: request.hardware.memoryBytes,
      });
      return canonicalJson(campaignEntryCanonicalFragment(response, 1_234));
    },
  });
  assert.equal(runRequests, 1);
  validateCampaignEntryBundle(result);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].status, "runtime_complete");
  assert.equal(Object.hasOwn(result.records[0], "output"), false);
  assert.equal(Object.hasOwn(result.records[0], "containment"), false);
  assert.equal(Object.hasOwn(result.records[0], "_campaignEntry"), false);

  for (const mutate of [
    (record) => { record.scenarios.find((entry) => entry.name === "cancel").cleanupVerified = false; },
    (record) => { record.scenarios.find((entry) => entry.name === "error").result = "not_run"; },
    (record) => { record.diagnostics.measurements.find((entry) =>
      entry.name === "campaignOwnedProcessGroupResidue").value = 1; },
    (record) => { record.diagnostics.measurements.find((entry) =>
      entry.name === "campaignWatchdogMaxObservedFootprintBytes").value = MAX_FOOTPRINT_BYTES; },
  ]) {
    const changed = structuredClone(result);
    mutate(changed.records[0]);
    assert.throws(() => validateCampaignEntryBundle(changed), /runtime lifecycle|schema|SC-20191/);
  }
});

test("the runner freezes the distinct full-A/V product-envelope canary", () => {
  const request = canaryRequest(
    128 * 1024 ** 3, TEXT_ENCODER_INVENTORY, PRODUCT_ENVELOPE_CANARY_PROFILE,
  );
  assert.equal(request.action, "product_envelope_canary");
  assert.equal(request.planned.fixture,
    "ltx-2-3-mlx-q4-768x512-f97-fps24-seed1234-product-envelope-canary");
  assert.deepEqual(request.planned.target.geometry, {
    width: 768, height: 512, batch: 1, frames: 97,
  });
  assert.deepEqual(request.planned._canary, {
    identity: "sc-20169-product-envelope", videoMode: "default_av", fps: 24, seed: 1234,
  });
  assert.deepEqual(request.planned.strategy.parameters, {
    decodeTileEdge: 192, decodeOverlap: 64,
  });
  assert.throws(() => canaryRequest(
    128 * 1024 ** 3, TEXT_ENCODER_INVENTORY, "campaign",
  ), /unsupported canary profile/);
});

test("private artifact clones preserve the canonical immutable snapshot roots", async () => {
  const scratch = "/private/canary-scratch";
  const roots = privateArtifactRoots(scratch);
  const snapshotRoot = path.join(
    scratch,
    "artifacts/models--SceneWorks--ltx-2.3-mlx/snapshots",
    "01df27d308466533aa09d251e3aebdcc627d07eb",
  );
  assert.deepEqual(roots, {
    numericTier: path.join(snapshotRoot, "q4"),
    textEncoder: path.join(snapshotRoot, "gemma"),
  });
  assert.notEqual(roots.numericTier, path.join(scratch, "artifacts/q4"));
  assert.notEqual(roots.textEncoder, path.join(scratch, "artifacts/gemma"));
  assert.equal(path.relative(scratch, roots.numericTier).startsWith(".."), false);
  assert.equal(path.relative(scratch, roots.textEncoder).startsWith(".."), false);
  const runnerSource = await readFile(new URL("./run-ltx-safety-canary.mjs", import.meta.url), "utf8");
  assert.match(runnerSource, /privateArtifactRoots\(stage\)/);
  assert.match(runnerSource, /rename\(stage, preparationRoot\)/);
});

test("the runner refuses promotable or identity-drifted adapter output", () => {
  const response = {
    status: "diagnostic_canary_complete",
    canaryIdentity: "sc-19741-safety",
    inferenceRevision: "6f3a84ef4ad4f858c6fe199e14925a01a7943f97",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    calibrationFingerprint: "sc-19109-ltx-2-3-mlx-memory-ladder-v1",
    target: {
      provider: "ltx_2_3",
      tier: "q4",
      geometry: { width: 256, height: 256, frames: 9, fps: 24 },
      videoMode: "no_audio",
      audio: false,
    },
    artifact: {
      repository: "SceneWorks/ltx-2.3-mlx",
      resolvedRevision: "01df27d308466533aa09d251e3aebdcc627d07eb",
      numericTierInventory: {
        files: 11,
        bytes: 20_467_690_460,
        sha256: "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a",
      },
      // serde_json serializes map keys in a different order from this JavaScript request. Typed
      // inventory comparison must accept the same fields independent of their insertion order.
      textEncoderInventory: {
        bytes: TEXT_ENCODER_INVENTORY.bytes,
        files: TEXT_ENCODER_INVENTORY.files,
        root: TEXT_ENCODER_INVENTORY.root,
        sha256: TEXT_ENCODER_INVENTORY.sha256,
      },
    },
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "staged_residency", "bounded_decode"],
      parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      spatialDecodeTiles: 4,
    },
    watchdog: {
      required: true,
      protocol: "sceneworks-memory-watchdog-v1",
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes: 128 * 1024 ** 3,
      minInitialMemoryFreeBytes: preflightFreeFloor(128 * 1024 ** 3),
      minMemoryFreeBytes: runtimeFreeFloor(128 * 1024 ** 3),
    },
    mlxLimits: {
      memoryLimitBytes: MAX_FOOTPRINT_BYTES,
      wiredLimitBytes: MAX_FOOTPRINT_BYTES,
    },
    observedMemory: {
      preProviderActiveBytes: 0,
      preProviderCacheBytes: 0,
      conditioning: { activeBytes: 10, reclaimableBytes: 2, allocatorBytes: 12 },
      denoise: { activeBytes: 20, reclaimableBytes: 3, allocatorBytes: 23 },
      decode: { activeBytes: 30, reclaimableBytes: 4, allocatorBytes: 34 },
      peakActiveBytes: 30,
      expectedPersistentActive: {
        identity: "mlx-gen-ltx-transformer-ones-cache-av-bfloat16-v1",
        videoDimension: 4_096,
        audioDimension: 2_048,
        dtype: "bfloat16",
        bytesPerElement: 2,
        bytes: (4_096 + 2_048) * 2,
      },
      postCleanupActiveBytes: (4_096 + 2_048) * 2,
      postCleanupCacheBytes: 0,
    },
    output: {
      frames: 9,
      fps: 24,
      audio: { present: false, samples: 0, sampleRate: 0, channels: 0 },
      frameTimelineSeconds: 1 / 3,
      firstFrameNondegenerate: true,
    },
  };
  assert.deepEqual(
    validateCanaryResponse(
      structuredClone(response), response.inferenceRevision, TEXT_ENCODER_INVENTORY,
      response.watchdog.hostMemoryBytes,
    ),
    response,
  );
  const mutations = [
    (value) => { value.promotable = true; },
    (value) => { value.target.audio = true; },
    (value) => { value.target.geometry.frames = 97; },
    (value) => { value.strategy.parameters.decodeTileEdge = 384; },
    (value) => { value.strategy.engagedRungs = ["resident", "bounded_decode"]; },
    (value) => { value.watchdog.required = false; },
    (value) => { value.watchdog.protocol = "self-asserted"; },
    (value) => { value.watchdog.maxFootprintBytes += 1; },
    (value) => { value.watchdog.maxRuntimeSeconds += 1; },
    (value) => { value.watchdog.minInitialMemoryFreeBytes -= 1; },
    (value) => { value.watchdog.minInitialMemoryFreePercent = 70; },
    (value) => { value.watchdog.minMemoryFreeBytes -= 1; },
    (value) => { value.watchdog.minSwapFreeBytes = 1024 ** 3; },
    (value) => { value.mlxLimits.wiredLimitBytes = MAX_FOOTPRINT_BYTES + 1; },
    (value) => { value.observedMemory.decode.allocatorBytes += 1; },
    (value) => { value.observedMemory.peakActiveBytes -= 1; },
    (value) => { value.output.firstFrameNondegenerate = false; },
    (value) => { value.inferenceRevision = "0".repeat(40); },
    (value) => { value.artifact.resolvedRevision = "0".repeat(40); },
    (value) => { value.artifact.numericTierInventory.sha256 = "0".repeat(64); },
    (value) => { value.artifact.textEncoderInventory.sha256 = "0".repeat(64); },
  ];
  for (const mutate of mutations) {
    const changed = structuredClone(response);
    mutate(changed);
    assert.throws(() => validateCanaryResponse(
      changed, response.inferenceRevision, TEXT_ENCODER_INVENTORY,
      response.watchdog.hostMemoryBytes,
    ));
  }
  for (const [mutate, expected] of [
    [
      (value) => { delete value.observedMemory.preProviderActiveBytes; },
      /preProviderActiveBytes must be a non-negative safe integer/,
    ],
    [
      (value) => { value.observedMemory.preProviderActiveBytes = 1; },
      /postCleanupActiveBytes 12288 did not equal pre-provider active 1 plus intentional persistent active 12288 = 12289/,
    ],
    [
      (value) => {
        value.observedMemory.preProviderActiveBytes = Number.MAX_SAFE_INTEGER;
        value.observedMemory.postCleanupActiveBytes = Number.MAX_SAFE_INTEGER;
      },
      /pre-provider plus persistent active-byte arithmetic overflowed/,
    ],
    [
      (value) => { value.observedMemory.postCleanupActiveBytes -= 1; },
      /postCleanupActiveBytes 12287 did not equal pre-provider active 0 plus intentional persistent active 12288 = 12288/,
    ],
    [
      (value) => { value.observedMemory.postCleanupActiveBytes += 1; },
      /postCleanupActiveBytes 12289 did not equal pre-provider active 0 plus intentional persistent active 12288 = 12288/,
    ],
    [
      (value) => { value.observedMemory.preProviderCacheBytes = 1; },
      /preProviderCacheBytes 1 did not attest the cleared pre-provider cache 0/,
    ],
    [
      (value) => { value.observedMemory.postCleanupCacheBytes = 1; },
      /postCleanupCacheBytes 1 did not return to observedMemory\.preProviderCacheBytes 0/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.bytes += 1; },
      /expectedPersistentActive\.bytes 12289 did not equal 12288/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.videoDimension -= 1; },
      /expectedPersistentActive\.videoDimension 4095 did not equal 4096/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.audioDimension += 1; },
      /expectedPersistentActive\.audioDimension 2049 did not equal 2048/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.dtype = "float32"; },
      /expectedPersistentActive\.dtype "float32" did not equal "bfloat16"/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.bytesPerElement = 4; },
      /expectedPersistentActive\.bytesPerElement 4 did not equal 2/,
    ],
    [
      (value) => { value.observedMemory.expectedPersistentActive.identity = "generic-cache"; },
      /expectedPersistentActive\.identity "generic-cache" did not equal/,
    ],
  ]) {
    const changed = structuredClone(response);
    mutate(changed);
    assert.throws(() => validateCanaryResponse(
      changed, response.inferenceRevision, TEXT_ENCODER_INVENTORY,
      response.watchdog.hostMemoryBytes,
    ), expected);
  }
});

test("the runner requires exact non-ingestible full-A/V product-envelope evidence", () => {
  const hostMemoryBytes = 128 * 1024 ** 3;
  const response = {
    status: "diagnostic_product_envelope_canary_complete",
    canaryIdentity: "sc-20169-product-envelope",
    inferenceRevision: "6f3a84ef4ad4f858c6fe199e14925a01a7943f97",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    calibrationFingerprint: "sc-19109-ltx-2-3-mlx-memory-ladder-v1",
    target: {
      provider: "ltx_2_3",
      tier: "q4",
      geometry: { width: 768, height: 512, frames: 97, fps: 24 },
      videoMode: "default_av",
      audio: true,
    },
    artifact: {
      repository: "SceneWorks/ltx-2.3-mlx",
      resolvedRevision: "01df27d308466533aa09d251e3aebdcc627d07eb",
      numericTierInventory: {
        files: 11,
        bytes: 20_467_690_460,
        sha256: "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a",
      },
      textEncoderInventory: { ...TEXT_ENCODER_INVENTORY },
    },
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "staged_residency", "bounded_decode"],
      parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      spatialDecodeTiles: 24,
    },
    watchdog: {
      required: true,
      protocol: "sceneworks-memory-watchdog-v1",
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes,
      minInitialMemoryFreeBytes: preflightFreeFloor(hostMemoryBytes),
      minMemoryFreeBytes: runtimeFreeFloor(hostMemoryBytes),
    },
    mlxLimits: {
      memoryLimitBytes: MAX_FOOTPRINT_BYTES,
      wiredLimitBytes: MAX_FOOTPRINT_BYTES,
    },
    observedMemory: {
      preProviderActiveBytes: 0,
      preProviderCacheBytes: 0,
      conditioning: { activeBytes: 10, reclaimableBytes: 2, allocatorBytes: 12 },
      denoise: { activeBytes: 20, reclaimableBytes: 3, allocatorBytes: 23 },
      decode: { activeBytes: 30, reclaimableBytes: 4, allocatorBytes: 34 },
      peakActiveBytes: 30,
      expectedPersistentActive: {
        identity: "mlx-gen-ltx-transformer-ones-cache-av-bfloat16-v1",
        videoDimension: 4_096,
        audioDimension: 2_048,
        dtype: "bfloat16",
        bytesPerElement: 2,
        bytes: 12_288,
      },
      postCleanupActiveBytes: 12_288,
      postCleanupCacheBytes: 0,
    },
    output: {
      frames: 97,
      fps: 24,
      audio: { present: true, samples: 192_000, sampleRate: 48_000, channels: 2 },
      frameTimelineSeconds: 4,
      firstFrameNondegenerate: true,
    },
  };
  assert.deepEqual(validateCanaryResponse(
    structuredClone(response), response.inferenceRevision, TEXT_ENCODER_INVENTORY,
    hostMemoryBytes, PRODUCT_ENVELOPE_CANARY_PROFILE,
  ), response);

  const mutations = [
    (value) => { value.status = "diagnostic_canary_complete"; },
    (value) => { value.canaryIdentity = "sc-19741-safety"; },
    (value) => { value.promotable = true; },
    (value) => { value.ingestible = true; },
    (value) => { value.target.geometry.width = 256; },
    (value) => { value.target.geometry.height = 768; },
    (value) => { value.target.geometry.frames = 89; },
    (value) => { value.target.geometry.fps = 30; },
    (value) => { value.target.videoMode = "no_audio"; },
    (value) => { value.target.audio = false; },
    (value) => { value.output.audio.present = false; },
    (value) => { value.output.audio.samples = 0; },
    (value) => { value.output.audio.sampleRate = 0; },
    (value) => { value.output.audio.channels = 0; },
    (value) => { value.output.frameTimelineSeconds = 97 / 24; },
    (value) => { value.strategy.parameters.decodeTileEdge = 384; },
    (value) => { value.strategy.spatialDecodeTiles = 1; },
    (value) => { value.strategy.spatialDecodeTiles = 23; },
    (value) => { value.watchdog.maxFootprintBytes -= 1; },
    (value) => { value.mlxLimits.memoryLimitBytes -= 1; },
    (value) => { value.observedMemory.peakActiveBytes -= 1; },
    (value) => { value.observedMemory.postCleanupActiveBytes += 1; },
    (value) => { value.artifact.numericTierInventory.sha256 = "0".repeat(64); },
  ];
  for (const mutate of mutations) {
    const changed = structuredClone(response);
    mutate(changed);
    assert.throws(() => validateCanaryResponse(
      changed, response.inferenceRevision, TEXT_ENCODER_INVENTORY,
      hostMemoryBytes, PRODUCT_ENVELOPE_CANARY_PROFILE,
    ));
  }
});

test("host preflight parses telemetry and excludes only actual GPU or shared-lane owners", () => {
  assert.equal(parseMemoryFreePercent("System-wide memory free percentage: 92%\n"), 92);
  assert.throws(() => parseMemoryFreePercent("no percentage"));
  assert.equal(
    parseSwapFreeBytes("vm.swapusage: total = 4096.00M used = 2880.00M free = 1216.00M"),
    1216 * 1024 ** 2,
  );
  assert.throws(() => parseSwapFreeBytes("vm.swapusage: unavailable"));
  const table = [
    "10 1 /usr/bin/cargo test --locked",
    "11 1 /usr/bin/rustc --crate-name x",
    "12 1 /tmp/sequential_residency_real_weights-deadbeef",
    "13 1 /usr/bin/python3 /tmp/MiniMax/real_weight.py",
    "14 1 /bin/zsh -lc ps | rg cargo",
    "15 1 /Applications/Xcode.app/toolchains/metal -c kernel.metal",
    "16 1 /opt/actions-runner/bin/Runner.Worker spawnclient 42 43",
  ].join("\n");
  assert.deepEqual(
    foreignHeavyProcesses(table).map((line) => Number(line.split(/\s+/, 1)[0])),
    [12, 13, 16],
  );
});

test("Cargo source cleanliness allows only Cargo's own sentinel", () => {
  assert.equal(cargoSourceStatusIsClean(""), true);
  assert.equal(cargoSourceStatusIsClean("?? .cargo-ok\n"), true);
  for (const mutation of [
    " M crates/media/mlx-gen/mlx-gen-ltx/src/vae.rs",
    "?? injected.rs",
    "?? .cargo-ok\n?? injected.rs",
    "?? nested/.cargo-ok",
  ]) {
    assert.equal(cargoSourceStatusIsClean(mutation), false, mutation);
  }
});

test("controller interruption preserves shell signal status after cleanup", () => {
  assert.equal(new CanaryInterrupted("SIGINT").exitCode, 130);
  assert.equal(new CanaryInterrupted("SIGTERM").exitCode, 143);
});

test("watchdog failures retain the exact hard-stop reason before scratch cleanup", () => {
  const status = { code: 97, signal: null };
  assert.equal(
    watchdogFailureSummary(status, [
      JSON.stringify({ event: "started" }),
      JSON.stringify({ event: "hard_stop", reason: "child_attestation_failed:TimeoutError" }),
      JSON.stringify({ event: "terminated" }),
    ].join("\n")),
    "watchdog failed closed: code=97 signal=null reason=child_attestation_failed:TimeoutError",
  );
  assert.match(watchdogFailureSummary(status, ""), /reason=event_stream_unavailable$/);
  assert.match(watchdogFailureSummary(status, "not-json\n"), /reason=event_stream_malformed$/);
  assert.match(
    watchdogFailureSummary(status, JSON.stringify({ event: "hard_stop", reason: "line1\nline2" })),
    /reason=line1 line2$/,
  );
  assert.match(
    watchdogFailureSummary(status, [
      JSON.stringify({ event: "hard_stop", reason: "physical_footprint_at_or_above_limit" }),
      '{"event":"terminated"',
    ].join("\n")),
    /reason=physical_footprint_at_or_above_limit;event_stream_malformed$/,
  );
  assert.match(
    watchdogFailureSummary(status, `${JSON.stringify({ event: "started" })}\n`),
    /reason=hard_stop_event_absent$/,
  );
});

test("scratch cleanup cannot mask the primary watchdog or signal failure", () => {
  const primary = new CanaryInterrupted("SIGTERM");
  const cleanup = new Error("permission denied\nwhile removing scratch");
  const combined = preservePrimaryFailure(primary, cleanup);
  assert.strictEqual(combined, primary);
  assert.equal(combined.exitCode, 143);
  assert.match(combined.message, /scratch cleanup failed: permission denied while removing scratch$/);
  assert.strictEqual(combined.cause, cleanup);
  assert.strictEqual(preservePrimaryFailure(primary, null), primary);
  assert.strictEqual(preservePrimaryFailure(null, cleanup), cleanup);
});

test("a stale failure receipt cannot mask the primary watchdog failure", () => {
  const primary = new Error("watchdog hard stop");
  const collision = Object.assign(new Error("destination exists"), { code: "EEXIST" });
  const combined = preserveFailureReceiptSuppression(primary, collision);
  assert.equal(combined, primary);
  assert.match(combined.message, /watchdog hard stop/);
  assert.match(combined.message, /failure receipt suppressed: destination exists/);
  assert.equal(combined.cause, collision);
});

test("the production runner can only launch through the identity-checked watchdog", async () => {
  const source = await readFile(new URL("./run-ltx-safety-canary.mjs", import.meta.url), "utf8");
  assert.match(source, /scripts\/memory-calibration-watchdog\.py/);
  assert.match(source, /"--max-footprint-bytes", String\(MAX_FOOTPRINT_BYTES\)/);
  assert.match(source, /"--max-runtime-seconds", String\(MAX_RUNTIME_SECONDS\)/);
  assert.equal(MAX_RUNTIME_SECONDS, 1_800);
  assert.equal(MIN_PREFLIGHT_FREE_BYTES, MAX_FOOTPRINT_BYTES * 2);
  assert.equal(MIN_RUNTIME_FREE_BYTES, MAX_FOOTPRINT_BYTES);
  assert.equal(telemetryResolutionBytes(128 * 1024 ** 3), Math.ceil(128 * 1024 ** 3 / 100));
  assert.throws(() => telemetryResolutionBytes(0));
  assert.equal(
    preflightFreeFloor(128 * 1024 ** 3),
    MIN_PREFLIGHT_FREE_BYTES + telemetryResolutionBytes(128 * 1024 ** 3),
  );
  assert.equal(
    runtimeFreeFloor(128 * 1024 ** 3),
    MIN_RUNTIME_FREE_BYTES + telemetryResolutionBytes(128 * 1024 ** 3),
  );
  assert.match(source, /"--min-memory-free-bytes", String\(runtimeMemoryFreeFloor\)/);
  assert.doesNotMatch(source, /--min-swap-free-bytes/);
  assert.match(source, /status\.code !== 0 \|\| status\.signal/);
  assert.match(source, /event\.event === "hard_stop" \|\| event\.event === "terminated"/);
  assert.doesNotMatch(source, /memory-mlx-adapter[^\n]*spawn\(/);
  assert.doesNotMatch(source, /--child-adapter|async function child\(/);
  assert.match(source, /buildExactAdapter\(/);
  assert.match(source, /locateBuiltMetallib\(target\)/);
  assert.match(source, /inferenceCargoSource\(/);
  assert.equal(CARGO_METADATA_TIMEOUT_MS, 15 * 60 * 1000);
  assert.match(source, /timeout: CARGO_METADATA_TIMEOUT_MS/);
  assert.match(source, /await assertRuntimeAssetIdentities\(adapter, metallib, signal\)/);
  assert.match(source, /validatePreparedCache\(preparationRoot, preparationKey, identity, signal\)/);
  assert.match(source, /sealedArtifactIdentity\(root\)/);
  const reuseValidation = source.slice(
    source.indexOf("export async function validatePreparedCache("),
    source.indexOf("async function sealPreparationDirectories("),
  );
  assert.doesNotMatch(reuseValidation, /hashArtifactInventory\(/,
    "cache reuse must not repeat the 46.9 GB content hash");
  assert.match(source, /const resolutionEnv = Object\.fromEntries/);
  assert.match(source, /const resolvedRustupHome = path\.resolve/);
  assert.match(source, /HOME: privateHome/);
  assert.match(source, /RUSTUP_HOME = resolvedRustupHome/);
  assert.match(source, /RUSTUP_TOOLCHAIN: channel/);
  assert.match(source, /CARGO_HOME: privateCargoHome/);
  assert.match(source, /CARGO_TARGET_DIR: privateTarget/);
  assert.match(source, /RUSTC: rustc/);
  assert.match(source, /TMPDIR: privateTemp/);
  assert.doesNotMatch(source, /TMPDIR: process\.env\.TMPDIR/);
  assert.match(source, /Cargo inference source tree differs from the verified checkout/);
  assert.match(source, /response\?\.inferenceRevision !== expectedInferenceRevision/);
  assert.match(source, /--require-child-attestation/);
  assert.match(source, /--require-provider-phases/);
  assert.equal(CHILD_ATTESTATION_TIMEOUT_SECONDS, 30);
  assert.match(source, /--child-attestation-timeout", String\(CHILD_ATTESTATION_TIMEOUT_SECONDS\)/);
  assert.match(source, /event\.event === "child_attested"/);
  assert.match(source, /await chmod\(adapterDirectory, 0o500\)/);
  assert.match(source, /await chmod\(metallibPath, 0o400\)/);
  assert.match(source, /PMETAL_METALLIB_PATH: metallibPath/);
  assert.match(source, /const runtimeHome = path\.join\(runScratch, "home"\)/);
  assert.match(source, /await exactEntries\(runtimeHome, \[\]\)/);
  assert.doesNotMatch(source, /\.cache\/pmetal\/lib\/mlx\.metallib/,
    "the canary must never name the mutable global metallib cache");
  assert.match(source, /runOwnedCommand\("\/bin\/cp", \["-c", cloneSource, output\]/);
  assert.match(source, /await cleanupCanaryScratch\(runScratch\)/);
  const failureRead = source.indexOf("const eventBytes = await readFile(eventsPath");
  const cleanup = source.indexOf("await cleanupCanaryScratch(runScratch)");
  assert.ok(failureRead >= 0 && cleanup > failureRead,
    "watchdog failure reason must be read before scratch cleanup");
  const failurePublisher = source.slice(
    source.indexOf("export async function publishCampaignEntryFailureReceipt("),
    source.indexOf("async function assertPreparationHasNoTransientResidue("),
  );
  assert.ok(failurePublisher.indexOf("await verify()")
    < failurePublisher.indexOf('return publishCampaignEntryOutcome(outcome, "failure"'),
  "failure validation and cleanup must precede atomic publication");
  const failureHandling = source.slice(
    source.indexOf("if (status.code !== 0 || status.signal)"),
    source.indexOf("signal?.throwIfAborted()", source.indexOf("if (status.code !== 0 || status.signal)")),
  );
  assert.ok(failureHandling.indexOf("const watchdogError = new Error(watchdogFailureSummary")
    < failureHandling.indexOf("validateCampaignEntryFailureEvents"),
  "the original watchdog failure must exist before receipt or postcondition validation");
  assert.match(failureHandling, /preserveFailureReceiptSuppression\(watchdogError, error\)/);
  assert.match(source, /acquireCampaignEntryOutcome\(output\)/);
  assert.match(source, /canonicalBundleAbsentAtPublication: true/);
  assert.equal(source.match(/await assertHostPreflight\(/g)?.length, 2,
    "each contained execution profile has one immediate model-release check");
  assert.doesNotMatch(source, /300_000|5 \* 60 \* 1_000/);
  assert.match(
    source,
    /const preLaunchHost = await assertHostPreflight\(memoryBytes, signal\);\n\s*const status = await new Promise\(\(resolve, reject\) => \{\n\s*const childProcess = spawn/,
    "the fresh host gate must be the final operation before watchdog launch",
  );
  assert.match(
    source,
    /await assertHostPreflight\(hostMemoryBytes, signal\);\n\s*const status = await new Promise\(\(resolve, reject\) => \{\n\s*const child = spawn/,
    "the campaign entry host gate must also be the final operation before watchdog launch",
  );
  assert.match(source, /sourceAfterRun: \{ sceneWorks: sceneWorksAfterRun, inference: inferenceAfterRun \}/);
  const campaignController = source.slice(
    source.indexOf("async function runCampaignEntryController("),
    source.indexOf("async function controller("),
  );
  assert.doesNotMatch(campaignController, /onProviderCheckpoint/,
    "the one-row campaign entry must never publish a partial harness checkpoint");
  for (const operation of [
    "await cleanupCanaryScratch(runRoot)",
    "await assertCampaignSourceState(providerOptions, signal)",
    "await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal)",
    "await assertPreparationHasNoTransientResidue(preparationRoot)",
    "validateCampaignEntryBundle(bundle)",
  ]) {
    assert.ok(campaignController.indexOf(operation) >= 0
      && campaignController.indexOf(operation)
        < campaignController.indexOf("await publishCampaignEntryCanonicalOutcome("),
    `${operation} must precede atomic canonical publication`);
  }
  assert.match(source, /cancellation\.abort\(new CanaryInterrupted\(signalName\)\)/);
  assert.match(source, /process\.exitCode = error instanceof CanaryInterrupted \? error\.exitCode : 1/);
});

test("the runner binds the pinned toolchain and private same-volume artifact clones", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "sc19741-runner-test-"));
  try {
    const source = path.join(root, "source");
    const destination = path.join(root, "destination");
    await mkdir(source);
    await writeFile(path.join(source, "weights.safetensors"), "immutable\n");
    const device = (await stat(root, { bigint: true })).dev;
    await cloneArtifactTree(source, destination, device);
    assert.equal(await readFile(path.join(destination, "weights.safetensors"), "utf8"), "immutable\n");
    assert.equal((await stat(destination)).mode & 0o777, 0o500);
    assert.equal((await stat(path.join(destination, "weights.safetensors"))).mode & 0o777, 0o400);
    const original = { root: source, files: 1, bytes: 10, sha256: "abc" };
    assert.deepEqual(inventoryAtRoot(original, destination), {
      root: destination, files: 1, bytes: 10, sha256: "abc",
    });

    const linked = path.join(root, "linked");
    await mkdir(linked);
    await symlink(path.join(source, "weights.safetensors"), path.join(linked, "weights.safetensors"));
    const resolved = path.join(root, "resolved");
    await cloneArtifactTree(linked, resolved, device);
    assert.equal(await readFile(path.join(resolved, "weights.safetensors"), "utf8"), "immutable\n");
    assert.equal((await stat(path.join(resolved, "weights.safetensors"))).isFile(), true);

    const privateRoots = privateArtifactRoots(root);
    await mkdir(path.dirname(privateRoots.numericTier), { recursive: true, mode: 0o700 });
    await cloneArtifactTree(source, privateRoots.numericTier, device);
    await cloneArtifactTree(source, privateRoots.textEncoder, device);
    assert.equal(
      await readFile(path.join(privateRoots.numericTier, "weights.safetensors"), "utf8"),
      "immutable\n",
    );
    assert.equal(
      await readFile(path.join(privateRoots.textEncoder, "weights.safetensors"), "utf8"),
      "immutable\n",
    );

    assert.equal(await repositoryToolchain(), "1.97.1");
    if (process.platform === "darwin") {
      const scratch = path.join(root, "toolchain");
      await mkdir(scratch);
      const toolchain = await exactToolchain(scratch);
      assert.equal(toolchain.channel, "1.97.1");
      assert.equal(toolchain.env.RUSTUP_TOOLCHAIN, toolchain.channel);
      assert.equal(toolchain.env.RUSTUP_HOME,
        path.resolve(process.env.RUSTUP_HOME ?? path.join(process.env.HOME, ".rustup")));
      assert.equal(toolchain.env.HOME, path.join(scratch, "home"));
      assert.notEqual(toolchain.env.HOME, process.env.HOME);
      assert.equal(toolchain.env.PATH,
        ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"].join(":"));
      assert.equal(toolchain.env.CARGO_HOME, path.join(scratch, "cargo-home"));
      assert.equal(toolchain.env.CARGO_TARGET_DIR, path.join(scratch, "target"));
      assert.equal(toolchain.env.TMPDIR, path.join(scratch, "tmp"));
      assert.ok(path.isAbsolute(toolchain.env.RUSTC));
      assert.equal(path.dirname(toolchain.env.RUSTC), path.dirname(toolchain.cargo));
      for (const directory of [
        toolchain.env.HOME, toolchain.env.CARGO_HOME,
        toolchain.env.CARGO_TARGET_DIR, toolchain.env.TMPDIR,
      ]) assert.equal((await stat(directory)).mode & 0o777, 0o700);
      assert.deepEqual(await readdir(toolchain.env.HOME), []);
      assert.ok(path.isAbsolute(toolchain.cargo));
      const probe = path.join(root, "cargo-rustc-probe");
      await mkdir(path.join(probe, "src"), { recursive: true });
      await writeFile(path.join(probe, "Cargo.toml"), [
        "[package]", 'name = "sc19741-cargo-rustc-probe"', 'version = "0.0.0"',
        'edition = "2024"', "",
      ].join("\n"));
      await writeFile(path.join(probe, "src/main.rs"), "fn main() {}\n");
      const probeArgs = [
        "check", "--offline", "--manifest-path", path.join(probe, "Cargo.toml"),
      ];
      await assert.rejects(() => runOwnedCommand(toolchain.cargo, probeArgs, {
        cwd: probe,
        env: { ...toolchain.env, RUSTC: path.join(root, "missing-rustc") },
        timeout: 30_000,
      }));
      await runOwnedCommand(toolchain.cargo, probeArgs, {
        cwd: probe, env: toolchain.env, timeout: 30_000,
      });
    }

    const disposable = path.join(root, "disposable");
    await mkdir(disposable);
    await chmod(disposable, 0o700);
    await writeFile(path.join(disposable, "large-build-output"), "x");
    await cleanupCanaryScratch(disposable);
    await assert.rejects(() => stat(disposable), { code: "ENOENT" });
  } finally {
    await cleanupCanaryScratch(root);
  }
});

test("the exact Cargo build metallib is uniquely selected and overrides every global cache", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "sc19741-metallib-test-"));
  t.after(() => cleanupCanaryScratch(root));
  const target = path.join(root, "target");
  const first = path.join(
    target, "release", "build", "pmetal-mlx-sys-first", "out", "build", "lib", "mlx.metallib",
  );
  await mkdir(path.dirname(first), { recursive: true });
  await writeFile(first, "exact build metallib\n");
  assert.equal(await locateBuiltMetallib(target), first);

  const prepared = path.join(root, "prepared", "adapter", "mlx.metallib");
  const runtimeHome = path.join(root, "runtime-home");
  await mkdir(runtimeHome, { mode: 0o700 });
  const roots = { numericTier: "/prepared/q4", textEncoder: "/prepared/gemma" };
  const environment = canaryWatchdogEnvironment({
    HOME: "/Users/test",
    PMETAL_METALLIB_PATH: "/Users/test/.cache/pmetal/lib/mlx.metallib",
    PMETAL_CACHE_DIR: "/Users/test/.cache/pmetal",
    XDG_CACHE_HOME: "/Users/test/.cache",
  }, roots, prepared, runtimeHome);
  assert.equal((await stat(runtimeHome)).mode & 0o777, 0o700);
  assert.deepEqual(await readdir(runtimeHome), []);
  assert.equal(environment.HOME, runtimeHome);
  assert.notEqual(environment.HOME, "/Users/test");
  assert.equal(environment.PMETAL_METALLIB_PATH, prepared);
  assert.equal(Object.hasOwn(environment, "PMETAL_CACHE_DIR"), false);
  assert.equal(Object.hasOwn(environment, "XDG_CACHE_HOME"), false);
  assert.equal(environment.SCENEWORKS_LTX_ROOT, roots.numericTier);
  assert.equal(environment.SCENEWORKS_LTX_TEXT_ENCODER_ROOT, roots.textEncoder);
  assert.notEqual(environment.PMETAL_METALLIB_PATH,
    "/Users/test/.cache/pmetal/lib/mlx.metallib");
  assert.throws(() => canaryWatchdogEnvironment(
    {}, roots, "relative/mlx.metallib", runtimeHome,
  ));
  assert.throws(() => canaryWatchdogEnvironment({}, roots, prepared, "relative/home"));

  const second = path.join(
    target, "release", "build", "pmetal-mlx-sys-second", "out", "build", "lib", "mlx.metallib",
  );
  await mkdir(path.dirname(second), { recursive: true });
  await writeFile(second, "ambiguous metallib\n");
  await assert.rejects(() => locateBuiltMetallib(target), /exactly one.*found 2/);
});

test("preparation identity and cache keys change for every canonical input", () => {
  assert.equal(PREPARATION_SCHEMA_VERSION, 3);
  const identity = preparationIdentity("1".repeat(40), "2".repeat(40), "1.97.1");
  const key = preparationCacheKey(identity);
  for (const mutate of [
    (value) => { value.sceneWorksTree = "3".repeat(40); },
    (value) => { value.inferenceTree = "4".repeat(40); },
    (value) => { value.toolchainChannel = "1.97.2"; },
    (value) => { value.artifact.revision = "5".repeat(40); },
    (value) => { value.artifact.numericTier.sha256 = "6".repeat(64); },
    (value) => { value.artifact.textEncoder.bytes += 1; },
  ]) {
    const changed = structuredClone(identity);
    mutate(changed);
    assert.notEqual(preparationCacheKey(changed), key);
  }
  assert.throws(() => preparationIdentity("short", "2".repeat(40), "1.97.1"));
  assert.throws(() => preparationIdentity("1".repeat(40), "2".repeat(40), "stable"));
});

test("a completed cache is atomically reused without rebuilding or rehashing file contents", async (t) => {
  const fixture = await cacheFixture(t);
  const first = await prepareCanaryCache(fixture.options);
  assert.equal(first.reused, false);
  assert.equal(fixture.builds(), 1);
  const second = await prepareCanaryCache({
    ...fixture.options,
    build: async () => { throw new Error("completed cache must not rebuild"); },
  });
  assert.equal(second.reused, true);
  assert.equal(fixture.builds(), 1);
  assert.deepEqual(second.manifest.artifacts.numericTier.content, fixture.identity.artifact.numericTier);
  assert.deepEqual(second.manifest.artifacts.textEncoder.content, fixture.identity.artifact.textEncoder);
  assert.match(second.manifest.metallib.sha256, /^[0-9a-f]{64}$/);
  assert.equal(second.manifest.metallib.seal.mode, 0o400);
  assert.equal(second.metallib.path,
    path.join(fixture.preparationRoot, "adapter", "mlx.metallib"));
  assert.equal((await stat(fixture.preparationRoot)).mode & 0o777, 0o500);
  assert.equal(
    (await stat(path.join(fixture.preparationRoot, "prepared.json"))).mode & 0o777,
    0o400,
  );
  assert.deepEqual((await readdir(fixture.preparationRoot)).sort(), [
    "adapter", "artifacts", "prepared.json", "prepared.sha256",
  ]);
  assert.deepEqual((await readdir(path.join(fixture.preparationRoot, "adapter"))).sort(), [
    "memory-mlx-adapter", "mlx.metallib",
  ]);
});

test("cache validation rejects canonical identity, completion, permission and metadata mutations", async (t) => {
  const canonical = await cacheFixture(t);
  await prepareCanaryCache(canonical.options);
  const changedIdentity = structuredClone(canonical.identity);
  changedIdentity.artifact.numericTier.sha256 = "0".repeat(64);
  await assert.rejects(
    () => validatePreparedCache(
      canonical.preparationRoot, canonical.key, changedIdentity,
    ),
    /identity|canonical/,
  );

  const manifestFixture = await cacheFixture(t);
  await prepareCanaryCache(manifestFixture.options);
  const manifestPath = path.join(manifestFixture.preparationRoot, "prepared.json");
  const completionPath = path.join(manifestFixture.preparationRoot, "prepared.sha256");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.artifacts.numericTier.content.sha256 = "0".repeat(64);
  const bytes = `${JSON.stringify(manifest, null, 2)}\n`;
  await chmod(manifestPath, 0o600);
  await chmod(completionPath, 0o600);
  await writeFile(manifestPath, bytes);
  await writeFile(completionPath, `${createHash("sha256").update(bytes).digest("hex")}\n`);
  await chmod(manifestPath, 0o400);
  await chmod(completionPath, 0o400);
  await assert.rejects(
    () => validatePreparedCache(
      manifestFixture.preparationRoot, manifestFixture.key, manifestFixture.identity,
    ),
    /canonical artifact identity/,
  );

  const permissions = await cacheFixture(t);
  await prepareCanaryCache(permissions.options);
  await chmod(path.join(permissions.preparationRoot, "prepared.json"), 0o600);
  await assert.rejects(
    () => validatePreparedCache(permissions.preparationRoot, permissions.key, permissions.identity),
    /file mode changed/,
  );

  const adapter = await cacheFixture(t);
  await prepareCanaryCache(adapter.options);
  const adapterPath = path.join(adapter.preparationRoot, "adapter", "memory-mlx-adapter");
  await chmod(adapterPath, 0o700);
  await assert.rejects(
    () => validatePreparedCache(adapter.preparationRoot, adapter.key, adapter.identity),
    /adapter executable.*mode 500/,
  );

  const metallib = await cacheFixture(t);
  await prepareCanaryCache(metallib.options);
  const metallibPath = path.join(metallib.preparationRoot, "adapter", "mlx.metallib");
  await chmod(metallibPath, 0o600);
  await writeFile(metallibPath, "drifted global-cache candidate\n");
  await chmod(metallibPath, 0o400);
  await assert.rejects(
    () => validatePreparedCache(metallib.preparationRoot, metallib.key, metallib.identity),
    /metallib changed/,
  );

  const metadata = await cacheFixture(t);
  await prepareCanaryCache(metadata.options);
  const artifactPath = path.join(
    privateArtifactRoots(metadata.preparationRoot).numericTier,
    "weights.safetensors",
  );
  await chmod(artifactPath, 0o600);
  await writeFile(artifactPath, "changed-tier\n");
  await chmod(artifactPath, 0o400);
  await assert.rejects(
    () => validatePreparedCache(metadata.preparationRoot, metadata.key, metadata.identity),
    /seal changed/,
  );
});

test("partial publication, orphaned ownership and concurrent preparation recover safely", async (t) => {
  const partial = await cacheFixture(t);
  await mkdir(path.dirname(partial.preparationRoot), { recursive: true, mode: 0o700 });
  await mkdir(partial.preparationRoot, { mode: 0o700 });
  await writeFile(path.join(partial.preparationRoot, "partial"), "interrupted\n");
  const lock = preparationLockPath(partial.preparationRoot);
  await mkdir(lock, { mode: 0o700 });
  await writeFile(path.join(lock, "owner.json"), JSON.stringify({
    schemaVersion: 2,
    pid: 999_999_999,
    processIdentity: { startIdentity: "old process birth", executable: "/usr/bin/node" },
    token: "orphan",
    startedAt: 0,
  }), { mode: 0o400 });
  const recovered = await prepareCanaryCache(partial.options);
  assert.equal(recovered.reused, false);
  assert.equal(partial.builds(), 1);
  await assert.rejects(() => stat(lock), { code: "ENOENT" });

  const concurrent = await cacheFixture(t);
  const slowBuild = async (...args) => {
    await delay(50);
    return concurrent.build(...args);
  };
  const options = { ...concurrent.options, build: slowBuild };
  const [left, right] = await Promise.all([
    prepareCanaryCache(options),
    prepareCanaryCache(options),
  ]);
  assert.equal(concurrent.builds(), 1);
  assert.deepEqual([left.reused, right.reused].sort(), [false, true]);
  assert.equal((await readdir(path.dirname(concurrent.preparationRoot)))
    .some((entry) => entry.includes(".stage-") || entry.endsWith(".lock")), false);
});

test("a reused live PID with a different process start identity is recovered as stale", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "sc19741-pid-reuse-test-"));
  t.after(() => cleanupCanaryScratch(root));
  const preparationRoot = path.join(root, "prepared", "a".repeat(64));
  await mkdir(path.dirname(preparationRoot), { recursive: true, mode: 0o700 });
  const lock = preparationLockPath(preparationRoot);
  await mkdir(lock, { mode: 0o700 });
  const reusedPid = 4242;
  const oldIdentity = {
    startIdentity: "Sun Aug 16 01:02:03 2026",
    executable: "/usr/local/bin/node",
  };
  await writeFile(path.join(lock, "owner.json"), `${JSON.stringify({
    schemaVersion: 2,
    pid: reusedPid,
    processIdentity: oldIdentity,
    token: "old-owner",
    startedAt: 0,
  })}\n`, { mode: 0o400 });

  const blocked = new AbortController();
  const blockedTimer = setTimeout(
    () => blocked.abort(new Error("verified live owner wait bounded by test")), 100,
  );
  await assert.rejects(
    acquirePreparationLock(
      preparationRoot,
      blocked.signal,
      async (pid) => pid === reusedPid ? oldIdentity : null,
    ),
    /verified live owner wait bounded by test/,
  );
  clearTimeout(blockedTimer);
  assert.equal((await stat(lock)).isDirectory(), true,
    "a matching process-birth identity must keep exclusive ownership");

  const currentIdentity = {
    startIdentity: "Sun Aug 16 04:05:06 2026",
    executable: "/usr/local/bin/node",
  };
  const release = await acquirePreparationLock(
    preparationRoot,
    undefined,
    async (pid) => pid === reusedPid
      ? { ...oldIdentity, startIdentity: "Sun Aug 16 03:04:05 2026" }
      : currentIdentity,
  );
  const newOwner = JSON.parse(await readFile(path.join(lock, "owner.json"), "utf8"));
  assert.equal(newOwner.pid, process.pid);
  assert.deepEqual(newOwner.processIdentity, currentIdentity);
  await release();
  await assert.rejects(() => stat(lock), { code: "ENOENT" });
});

test("cancellation during clone, build and publication removes owned staging and locks", async (t) => {
  for (const phase of ["clone", "build", "publish"]) {
    const fixture = await cacheFixture(t);
    const cancellation = new AbortController();
    let reached;
    const reachedPhase = new Promise((resolve) => { reached = resolve; });
    let build = fixture.build;
    const hooks = {};
    if (phase === "clone") {
      hooks.afterNumericClone = async () => {
        reached();
        await delay(60_000, undefined, { signal: cancellation.signal });
      };
    } else if (phase === "build") {
      build = async (stage, signal) => {
        reached();
        await delay(60_000, undefined, { signal });
        return fixture.build(stage, signal);
      };
    } else {
      hooks.beforePublish = async () => {
        reached();
        await delay(60_000, undefined, { signal: cancellation.signal });
      };
    }
    const operation = prepareCanaryCache({
      ...fixture.options, build, hooks, signal: cancellation.signal,
    });
    await reachedPhase;
    cancellation.abort(new CanaryInterrupted("SIGTERM"));
    await assert.rejects(
      operation,
      (error) => error instanceof CanaryInterrupted || error.cause instanceof CanaryInterrupted,
      phase,
    );
    const parent = path.dirname(fixture.preparationRoot);
    const entries = await readdir(parent);
    assert.equal(entries.some((entry) => entry.includes(".stage-") || entry.endsWith(".lock")), false);
    await assert.rejects(() => stat(fixture.preparationRoot), { code: "ENOENT" });
  }
});

test("owned command cancellation terminates descendants before rejecting", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "sc19741-owned-command-"));
  t.after(() => cleanupCanaryScratch(root));
  const pidFile = path.join(root, "child.pid");
  const cancellation = new AbortController();
  const operation = runOwnedCommand("/bin/sh", [
    "-c", `sleep 60 & child=$!; echo $child > ${JSON.stringify(pidFile)}; wait`,
  ], { signal: cancellation.signal });
  for (let attempt = 0; attempt < 100 && !(await readFile(pidFile, "utf8").catch(() => "")); attempt += 1) {
    await delay(10);
  }
  const childPid = Number((await readFile(pidFile, "utf8")).trim());
  cancellation.abort(new CanaryInterrupted("SIGTERM"));
  await assert.rejects(operation, CanaryInterrupted);
  let childExists = true;
  try {
    process.kill(childPid, 0);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
    childExists = false;
  }
  if (childExists) {
    const childStatus = spawnSync("/bin/ps", ["-o", "stat=", "-p", String(childPid)], {
      encoding: "utf8",
    });
    assert.equal(childStatus.error, undefined);
    assert.equal(childStatus.signal, null);
    assert.equal(childStatus.stderr, "");
    const childStates = childStatus.stdout.split("\n").map((state) => state.trim()).filter(Boolean);
    if (childStatus.status === 1 && childStates.length === 0) {
      assert.throws(
        () => process.kill(childPid, 0),
        (error) => error.code === "ESRCH",
        "owned descendant survived cancellation after ps reported no process",
      );
    } else {
      assert.equal(childStatus.status, 0);
      assert.equal(childStates.length, 1);
      assert.match(childStates[0], /^Z/, `owned descendant survived cancellation: state=${childStates[0]}`);
    }
  }
});
