// sc-23001: the YuE2 memory-profile harness — plan, identity keying, stage attribution, currency.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { appendFile, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  BACKENDS,
  boundaryAcknowledgment,
  CUDA_RESERVE_BYTES,
  REQUEST_KEYS,
  ROOT,
  admitToCorpus,
  buildRecord,
  captureEnv,
  cargoTestArgv,
  caseFile,
  caseIdentity,
  corpusFileName,
  corpusStatus,
  coverage,
  expandCases,
  externalCaseItem,
  gradeRecord,
  isWithinRepository,
  parseArgs,
  parseStageMarks,
  parseWatchdogSamples,
  planCapture,
  readDurableWatchdogEvents,
  readSources,
  startBoundaryWatcher,
  stagePeaks,
  validatePlan,
  validateRecord,
} from "./yue2-memory-profile.mjs";

const sources = await readSources();
const clone = (value) => JSON.parse(JSON.stringify(value));
const GIB = 1024 ** 3;

async function waitUntil(check) {
  const deadline = Date.now() + 2000;
  while (!check()) {
    if (Date.now() >= deadline) assert.fail("timed out waiting for boundary test state");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

// What a current (v2) capture's outcome.json states about its run (`outcome_json`).
const RUN_FIELDS = Object.freeze({
  truncated: { abc: false, semantic: false },
  stageSeconds: { load: 2, plan: 5, semantic: 9, acoustic: 4, decode: 1 },
  engineTiming: { nar_seconds: 3.5, vae_seconds: 1.0 },
  runIdentity: "run-identity",
  planIdentity: "plan-identity",
  decoder: { release: "standard" },
  latent: { sha256: "d".repeat(64) },
});

function record(item, overrides = {}) {
  const declared = sources.closures.providers[sources.plan.lane];
  const marks = [
    { stage: "load", at: 10 },
    { stage: "plan", at: 20 },
    { stage: "semantic", at: 30 },
    { stage: "acoustic", at: 40 },
    { stage: "decode", at: 50 },
    { stage: "done", at: 60 },
  ];
  const samples = [
    { at: 5, bytes: 1 * GIB },
    { at: 15, bytes: 8 * GIB },
    { at: 25, bytes: 9 * GIB },
    { at: 35, bytes: 10 * GIB },
    { at: 45, bytes: 11 * GIB },
    { at: 55, bytes: 12 * GIB },
    { at: 65, bytes: 99 * GIB },
  ];
  const estimate = (total) => ({ deviceBytes: total, hostBytes: 0, totalBytes: total });
  return buildRecord({
    item,
    identity: caseIdentity(item, sources.manifest),
    engine: {
      lane: sources.plan.lane,
      inferenceRevision: "a".repeat(40),
      closureDigest: declared.digest,
      digestVersion: sources.closures.digestVersion,
    },
    sceneworks: { revision: "b".repeat(40), dirty: false },
    hardware: { memoryBytes: 64 * GIB },
    admission: {
      outcome: "admitted",
      estimate: {
        weights: { deviceBytes: 4 * GIB, hostBytes: item.arMode === "experimentalFp8" ? 3 * GIB : 0 },
        stages: {
          load: estimate(9 * GIB),
          plan: estimate(10 * GIB),
          semantic: estimate(10 * GIB),
          acoustic: estimate(12 * GIB),
          decode: estimate(11 * GIB),
        },
      },
    },
    outcome: {
      status: "completed", audioSeconds: 30, rms: 0.1, ...RUN_FIELDS,
      ...(item.arMode === "experimentalFp8" ? { engineQuantization: "fp8" } : {}),
    },
    sampler: "synthetic",
    samples,
    marks,
    capturedAt: "2026-09-26T00:00:00Z",
    ...overrides,
  });
}

test("the checked-in plan validates against the catalog and closure table and expands to its cases", () => {
  validatePlan(sources.plan, sources);
  const cases = expandCases(sources.plan);
  // Every backend measures every catalog tier at the product default request.
  for (const backend of BACKENDS) {
    const tiers = cases
      .filter((item) => item.backend === backend && item.requestName === "default")
      .map((item) => item.tier)
      .sort();
    assert.deepEqual(tiers, ["bf16", "q4", "q8"], backend);
  }
  for (const item of cases) {
    assert.match(item.id, /^yue2:(bf16|q8|q4):(metal|cuda):[a-z][a-z0-9-]*$/);
    const identity = caseIdentity(item, sources.manifest);
    const entry = sources.manifest.models.find((model) => model.id === "yue2");
    const row = entry.downloads.find((download) => !download.coRequisite && download.variant === item.tier);
    assert.equal(identity.model.repo, row.repo);
    assert.equal(identity.model.revision, row.revision);
    // A derived tier is identified by its pinned weights; the released bf16 by repo@revision.
    assert.equal(identity.model.weightsSha256 === null, item.tier === "bf16", item.id);
    assert.equal(identity.decoder.componentId, item.decoder === "legacy" ? "vae_legacy" : "vae");
  }
  const fp8 = cases.filter((item) => item.arMode === "experimentalFp8");
  assert.deepEqual(fp8.map((item) => item.id), ["yue2:bf16:cuda:experimental-fp8-ar"]);
  assert.equal(caseFile(fp8[0]).arMode, "experimentalFp8");
});

test("an off-plan explicit precision case uses the guarded capture path without replacing corpus cases", async () => {
  const body = {
    id: "yue2:bf16:metal:strict-bf16-standard", tier: "bf16", decoder: "standard",
    computePolicy: "bf16", request: clone(sources.plan.requests.default.request),
  };
  const item = externalCaseItem(body, sources.plan);
  assert.deepEqual(caseFile(item), body);
  assert.equal(item.backend, "metal");
  assert.throws(() => externalCaseItem({ ...body, id: "yue2:bf16:metal:default" }, sources.plan), /cannot be overridden/);
  assert.throws(() => externalCaseItem({ ...body, computePolicy: undefined }, sources.plan), /explicit computePolicy/);
  const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-precision-case-"));
  try {
    const file = path.join(dir, "case.json");
    await writeFile(file, JSON.stringify(body));
    const planned = await planCapture({ externalCaseFile: file, outDir: dir, sources });
    assert.equal(planned.item.computePolicy, "bf16");
    assert.equal(planned.guarded.eventFile, path.join(planned.outDir, "watchdog.jsonl"));
    await assert.rejects(planCapture({ externalCaseFile: path.join(dir, "repo", "..case.json"),
      outDir: dir, sources, root: path.join(dir, "repo") }), /--case-file must be outside/);
    await assert.rejects(planCapture({ externalCaseFile: file, outDir: path.join(dir, "repo", "..cache"),
      sources, root: path.join(dir, "repo") }), /--out must be outside/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("an explicit precision record must preserve its effective model and VAE dtypes", () => {
  const item = externalCaseItem({
    id: "yue2:bf16:metal:strict-bf16-standard", tier: "bf16", decoder: "standard",
    computePolicy: "bf16", request: clone(sources.plan.requests.default.request),
  }, sources.plan);
  const captured = record(item);
  captured.outcome.engineComputePolicy = "bf16";
  captured.outcome.engineModelDtype = "bfloat16";
  captured.outcome.engineVaeDtype = "bfloat16";
  validateRecord(captured);
  captured.outcome.engineVaeDtype = "float32";
  assert.throws(() => validateRecord(captured), /model\/VAE dtypes/);
});

test("the plan refuses an undeclared lane, a non-catalog tier and a field the entrypoint cannot read", () => {
  const undeclared = clone(sources.plan);
  undeclared.lane = "candle:not_a_lane";
  assert.throws(() => validatePlan(undeclared, sources), /not declared/);
  const tier = clone(sources.plan);
  tier.requests.default.tiers.push("nvfp4");
  assert.throws(() => validatePlan(tier, sources), /not a catalog tier/);
  const field = clone(sources.plan);
  field.requests.default.request.temperature = 0.7;
  assert.throws(() => validatePlan(field, sources), /unknown request field temperature/);
  const unsupported = clone(sources.plan);
  unsupported.requests["experimental-fp8-ar"].backends = ["metal"];
  assert.throws(() => validatePlan(unsupported, sources), /bf16 on CUDA only/);
  const inverted = clone(sources.plan);
  inverted.requests["long-context"].request.semanticSampling = { minTokens: 10, maxTokens: 5 };
  assert.throws(() => validatePlan(inverted, sources), /minTokens exceeds maxTokens/);
});

test("the case file's request fields are exactly the native entrypoint's (deny_unknown_fields)", async () => {
  const source = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/yue2_memory_profile.rs"), "utf8");
  const struct = source.match(/pub\(crate\) struct CaseRequest \{([\s\S]*?)\n\}/)[1];
  const fields = [...struct.matchAll(/^\s+([a-z_]+): /gm)]
    .map(([, name]) => name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()))
    .sort();
  assert.deepEqual(fields, [...REQUEST_KEYS].sort());
  // The entrypoint the controller invokes exists under the name it invokes.
  assert.match(source, /\nfn capture_case\(\)/);
  const lib = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/lib.rs"), "utf8");
  assert.match(lib, /\nmod yue2_memory_profile;/);
  for (const item of expandCases(sources.plan)) {
    assert.deepEqual(Object.keys(caseFile(item)), item.arMode
      ? ["id", "tier", "decoder", "arMode", "request"]
      : ["id", "tier", "decoder", "request"]);
  }
});

test("an FP8 profile record is mode-bound", () => {
  const item = expandCases(sources.plan).find((row) => row.arMode === "experimentalFp8");
  const current = record(item);
  assert.equal(current.request.arMode, "experimentalFp8");
  assert.equal(current.outcome.engineQuantization, "fp8");
  assert.equal(gradeRecord(current, sources).status, "current");
  const altered = clone(current);
  altered.request.arMode = "native";
  assert.match(gradeRecord(altered, sources).reasons.join(), /arMode native is now experimentalFp8/);
  const unpriced = clone(current);
  unpriced.admission.estimate.weights.hostBytes = 0;
  assert.throws(() => validateRecord(unpriced), /retained BF16 AR originals/);
  const nativeEngine = clone(current);
  nativeEngine.outcome.engineQuantization = "none";
  assert.throws(() => validateRecord(nativeEngine), /effective FP8 AR mode/);
  const unprovenEngine = clone(current);
  delete unprovenEngine.outcome.engineQuantization;
  assert.throws(() => validateRecord(unprovenEngine), /effective FP8 AR mode/);
});

test("samples are attributed to the stage in force; start-up and teardown belong to none", () => {
  const item = expandCases(sources.plan)[0];
  const built = record(item);
  assert.deepEqual(built.measured.stages, {
    load: { peakBytes: 8 * GIB, samples: 1 },
    plan: { peakBytes: 9 * GIB, samples: 1 },
    semantic: { peakBytes: 10 * GIB, samples: 1 },
    acoustic: { peakBytes: 11 * GIB, samples: 1 },
    decode: { peakBytes: 12 * GIB, samples: 1 },
  });
  assert.equal(built.measured.peakBytes, 12 * GIB, "the 99 GiB teardown sample is no stage's");
  assert.throws(() => parseStageMarks('{"stage":"plan","at":1}\n'), /start at the load/);
  assert.throws(() => parseStageMarks('{"stage":"load","at":2}\n{"stage":"plan","at":1}\n'), /back in time/);
  assert.throws(() => parseStageMarks('{"stage":"load","at":1}\n{"stage":"nar","at":2}\n'), /unknown stage/);
  const samples = parseWatchdogSamples(
    '{"event":"started"}\n{"event":"sample","at":3.5,"physicalFootprintBytes":42}\n',
  );
  assert.deepEqual(samples, [{ at: 3.5, bytes: 42 }]);
  assert.deepEqual(stagePeaks(samples, parseStageMarks('{"stage":"load","at":3}\n')), {
    load: { peakBytes: 42, samples: 1 },
  });
});

test("Metal boundary needs two new serial watchdog samples after the request is observed", () => {
  const state = {};
  const request = { sequence: 1, stage: "load", requestedAt: 100 };
  const event = (eventSequence, at) => ({ event: "sample", eventSequence, at, physicalFootprintBytes: 42 });
  assert.equal(boundaryAcknowledgment(request, state, [event(5, 99)], "metal"), null);
  assert.equal(boundaryAcknowledgment(request, state, [event(5, 99), event(6, 101)], "metal"), null,
    "the first event might have started before the request");
  assert.deepEqual(boundaryAcknowledgment(request, state, [event(5, 99), event(6, 101), event(7, 103)], "metal"), {
    sequence: 1, stage: "load", sampleAt: 103, sampleBytes: 42, sampleSerial: 7, sampler: "metal-watchdog",
  });
  assert.equal(boundaryAcknowledgment(request, state, [event(5, 99), event(6, 101), event(7, 103)], "metal"), null);
  assert.throws(() => boundaryAcknowledgment({ ...request, sequence: 3 }, state, [], "metal"), /skipped, regressed/);
  assert.throws(() => boundaryAcknowledgment({ ...request, stage: "semantic" }, state, [], "metal"), /changed at the same sequence/);
  assert.throws(() => boundaryAcknowledgment({ ...request, requestedAt: 101 }, state, [], "metal"), /changed at the same sequence/);
  assert.throws(() => boundaryAcknowledgment({ ...request, sequence: 2 }, {}, [], "metal"), /not one/);
});

test("Metal watcher does not ack flushed watchdog events before journal fsync, and fails closed on sync error", async () => {
  const event = (eventSequence, at) => `${JSON.stringify({
    event: "sample", eventSequence, at, physicalFootprintBytes: 42,
  })}\n`;
  for (const failSync of [false, true]) {
    const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-watchdog-durability-"));
    const journal = path.join(dir, "watchdog.jsonl");
    await writeFile(path.join(dir, "request.json"), JSON.stringify({ sequence: 1, stage: "load", requestedAt: 100 }));
    await writeFile(journal, event(1, 99));
    let releaseSync;
    let blocked = false;
    const gate = new Promise((resolve) => { releaseSync = resolve; });
    const reader = (file) => readDurableWatchdogEvents(file, async (handle, body) => {
      if (body.includes('"eventSequence":3')) {
        blocked = true;
        await gate;
        if (failSync) throw new Error("journal fsync failed");
      }
      await handle.sync();
    });
    const watcher = startBoundaryWatcher(dir, "metal", journal, reader);
    try {
      await waitUntil(() => watcher.sequence === 1);
      await appendFile(journal, event(2, 101) + event(3, 102));
      await waitUntil(() => blocked || watcher.failure !== null);
      assert.equal(watcher.failure, null);
      await assert.rejects(readFile(path.join(dir, "ack.json")), { code: "ENOENT" });
      releaseSync();
      if (failSync) {
        await waitUntil(() => watcher.failure !== null);
        assert.match((await watcher.stop()).message, /journal fsync failed/);
        await assert.rejects(readFile(path.join(dir, "ack.json")), { code: "ENOENT" });
      } else {
        await waitUntil(() => existsSync(path.join(dir, "ack.json")) || watcher.failure !== null);
        assert.equal(watcher.failure, null);
        assert.equal(JSON.parse(await readFile(path.join(dir, "ack.json"), "utf8")).sampleSerial, 3);
      }
    } finally {
      releaseSync();
      await watcher.stop();
      await rm(dir, { recursive: true, force: true });
    }
  }
});

test("CUDA boundary accepts only a fresh successful nvidia-smi call after observation", () => {
  const state = {};
  const request = { sequence: 1, stage: "decode", requestedAt: 100 };
  const before = { startedAt: 99, at: 101, bytes: 42 };
  assert.equal(boundaryAcknowledgment(request, state, [before], "cuda"), null);
  assert.equal(boundaryAcknowledgment(request, state, [before], "cuda"), null,
    "a dropped tick leaves the stage blocked");
  assert.equal(boundaryAcknowledgment(request, state, [before, { startedAt: 101, at: 102, bytes: 0 }], "cuda"), null);
  assert.deepEqual(boundaryAcknowledgment(request, state, [before, { startedAt: 101, at: 102, bytes: 0 },
    { startedAt: 102, at: 103, bytes: 43 }], "cuda"), {
    sequence: 1, stage: "decode", sampleAt: 103, sampleBytes: 43, sampleSerial: 3, sampler: "cuda-nvidia-smi",
  });
});

test("completed records require every actual stage, including a short load and no-plan run", () => {
  const item = expandCases(sources.plan)[0];
  const noPlan = record(item, {
    marks: [
      { stage: "load", at: 10 }, { stage: "semantic", at: 11.389314 },
      { stage: "acoustic", at: 20 }, { stage: "decode", at: 30 }, { stage: "done", at: 40 },
    ],
    samples: [
      { at: 9.44723, bytes: 1 }, { at: 11.847188, bytes: 2 },
      { at: 15, bytes: 3 }, { at: 25, bytes: 4 }, { at: 35, bytes: 5 },
    ],
    outcome: { status: "completed", audioSeconds: 30, rms: 0.1, ...RUN_FIELDS,
      stageSeconds: { load: 1.389314, semantic: 8, acoustic: 10, decode: 10 } },
  });
  assert.throws(() => validateRecord(noPlan), /completed record measured no load/);
  const covered = record(item, {
    marks: [
      { stage: "load", at: 10 }, { stage: "semantic", at: 12 },
      { stage: "acoustic", at: 20 }, { stage: "decode", at: 30 }, { stage: "done", at: 40 },
    ],
    samples: [{ at: 11, bytes: 2 }, { at: 15, bytes: 3 }, { at: 25, bytes: 4 }, { at: 35, bytes: 5 }],
    outcome: { status: "completed", audioSeconds: 30, rms: 0.1, ...RUN_FIELDS,
      stageSeconds: { load: 2, semantic: 8, acoustic: 10, decode: 10 } },
  });
  validateRecord(covered);
  assert.equal(covered.measured.stages.plan, undefined);
  const missingAcoustic = clone(covered);
  delete missingAcoustic.measured.stages.acoustic;
  assert.throws(() => validateRecord(missingAcoustic), /completed acoustic stage has no external sample/);
});

test("a record is current only under the declared closure, the catalog identity and a clean tree", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.tier === "q8");
  const current = record(item);
  assert.equal(gradeRecord(current, sources).status, "current");

  const staleClosure = clone(current);
  staleClosure.identity.engine.closureDigest = "0".repeat(64);
  const graded = gradeRecord(staleClosure, sources);
  assert.equal(graded.status, "stale");
  assert.match(graded.reasons.join(), /stale closure/);

  const oldModel = clone(current);
  oldModel.identity.model.revision = "c".repeat(40);
  assert.match(gradeRecord(oldModel, sources).reasons.join(), /model.revision/);
  const oldWeights = clone(current);
  oldWeights.identity.model.weightsSha256 = "d".repeat(64);
  assert.match(gradeRecord(oldWeights, sources).reasons.join(), /model.weightsSha256/);
  const oldDecoder = clone(current);
  oldDecoder.identity.decoder.revision = "e".repeat(40);
  assert.match(gradeRecord(oldDecoder, sources).reasons.join(), /decoder.revision/);
  const dirty = clone(current);
  dirty.identity.sceneworks.dirty = true;
  assert.match(gradeRecord(dirty, sources).reasons.join(), /dirty/);
  const retired = clone(current);
  retired.caseId = "yue2:q8:metal:retired";
  assert.match(gradeRecord(retired, sources).reasons.join(), /not in the plan/);
  const version = clone(current);
  version.identity.engine.digestVersion = "inference-closure-digest v3";
  assert.match(gradeRecord(version, sources).reasons.join(), /digest version/);
});

test("ingest refuses stale and failed records and admits current ones", () => {
  const item = expandCases(sources.plan)[0];
  const current = record(item);
  assert.equal(admitToCorpus(current, sources), current);
  const stale = clone(current);
  stale.identity.engine.closureDigest = "1".repeat(64);
  assert.throws(() => admitToCorpus(stale, sources), /rejected .*stale closure/);
  const failed = clone(current);
  failed.outcome = { status: "failed", exit: 101 };
  assert.throws(() => admitToCorpus(failed, sources), /capture failed/);
  assert.equal(corpusFileName(current), `${item.id.replaceAll(":", "__")}.json`);
  const refused = clone(current);
  refused.outcome = { status: "refused", message: "yue2: … short" };
  assert.equal(admitToCorpus(refused, sources), refused, "a refusal on real hardware is evidence");
});

test("coverage compares the footprint with the stage total, and nvidia-smi with device + reserve", () => {
  const cases = expandCases(sources.plan);
  const metal = record(cases.find((item) => item.backend === "metal"));
  const rows = Object.fromEntries(coverage(metal).map((row) => [row.stage, row]));
  assert.equal(rows.load.covered, true);
  assert.equal(rows.decode.covered, false, "12 GiB measured over an 11 GiB estimate is under-priced");
  const cuda = record(cases.find((item) => item.backend === "cuda"));
  const cudaRows = Object.fromEntries(coverage(cuda).map((row) => [row.stage, row]));
  assert.equal(cudaRows.decode.estimatedBytes, 11 * GIB + CUDA_RESERVE_BYTES);
  assert.equal(cudaRows.decode.covered, true);
  const refused = clone(metal);
  refused.outcome = { status: "refused" };
  assert.deepEqual(coverage(refused), []);
});

test("shared CUDA records keep global capacity peaks without claiming an owned admission comparison", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.backend === "cuda");
  const shared = record(item, { cudaSharedDevice: true, sampler: "nvidia-smi memory.used" });
  validateRecord(shared);
  const decode = coverage(shared).find((row) => row.stage === "decode");
  assert.deepEqual(decode, { stage: "decode", measuredBytes: null, globalDevicePeakBytes: 12 * GIB,
    estimatedBytes: 11 * GIB + CUDA_RESERVE_BYTES, ratio: null, covered: null,
    reason: "owned CUDA peak unavailable on shared device" });
  assert.equal(shared.measured.peakBytes, 12 * GIB, "the raw global peak is retained");
  const changingForeignUse = clone(shared);
  changingForeignUse.measured.stages.decode.peakBytes = 50 * GIB;
  const changed = coverage(changingForeignUse).find((row) => row.stage === "decode");
  assert.equal(changed.globalDevicePeakBytes, 50 * GIB);
  assert.equal(changed.covered, null, "a changing foreign allocation cannot prove an own-budget pass or failure");
  const isolated = record(item);
  isolated.measured.stages.decode.peakBytes = 50 * GIB;
  assert.equal(coverage(isolated).find((row) => row.stage === "decode").covered, false,
    "the old isolated-device over-budget check still fails");
  const invalidMetal = record(expandCases(sources.plan).find((candidate) => candidate.backend === "metal"),
    { cudaSharedDevice: true, sampler: "nvidia-smi memory.used" });
  assert.throws(() => validateRecord(invalidMetal), /invalid shared CUDA measurement scope/);
  const invalidSampler = clone(shared);
  invalidSampler.measured.sampler = "synthetic";
  assert.throws(() => validateRecord(invalidSampler), /invalid shared CUDA measurement scope/);
});

test("shared CUDA coverage uses verified owned stage peaks, refuses over-budget and missing-stage claims", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.backend === "cuda");
  const proof = { sha256: "a".repeat(64), physicalIndex: 1, cudaOrdinal: 0,
    uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", pci: "00000000:C1:00.0",
    luid: "luid_0x00000000_0x0001f78f" };
  const stages = Object.fromEntries(["load", "plan", "semantic", "acoustic", "decode"]
    .map((stage) => [stage, { peakBytes: 10 * GIB, samples: 3 }]));
  const owned = { sampler: "windows-gpu-process-memory dedicated", selectedLuid: proof.luid,
    proofSha256: proof.sha256, process: { pid: 123, parentPid: 456,
      createdUtc: "2026-10-04T20:00:02Z", executableSha256: "b".repeat(64) },
    journalSha256: "c".repeat(64), stages, peakBytes: 10 * GIB, faults: [], complete: true };
  const shared = record(item, { cudaSharedDevice: true, sampler: "nvidia-smi memory.used",
    deviceProof: proof, ownedMeasurement: owned });
  shared.measured.stages.decode.peakBytes = 50 * GIB;
  validateRecord(shared);
  assert.equal(coverage(shared).find((row) => row.stage === "decode").covered, true,
    "foreign use changes global peak without changing owned coverage");
  shared.measured.stages.decode.peakBytes = 55 * GIB;
  assert.equal(coverage(shared).find((row) => row.stage === "decode").covered, true);
  shared.measured.owned.stages.decode.peakBytes = 14 * GIB;
  shared.measured.owned.peakBytes = 14 * GIB;
  assert.equal(coverage(shared).find((row) => row.stage === "decode").covered, false,
    "an owned peak above device+reserve remains under-priced");
  delete shared.measured.owned.stages.decode;
  shared.measured.owned.peakBytes = 10 * GIB;
  assert.throws(() => validateRecord(shared), /complete owned CUDA measurement has gaps/);
  shared.measured.owned.complete = false;
  assert.equal(coverage(shared).find((row) => row.stage === "decode").covered, null);
});

test("the corpus status resumes a campaign: current cases are not captured again", () => {
  const cases = expandCases(sources.plan);
  const done = record(cases[0]);
  const stale = clone(record(cases[1]));
  stale.identity.engine.closureDigest = "2".repeat(64);
  const status = Object.fromEntries(
    corpusStatus(sources.plan, [done, stale], sources).map((row) => [row.id, row.status]),
  );
  assert.equal(status[cases[0].id], "current");
  assert.equal(status[cases[1].id], "stale");
  assert.equal(status[cases[2].id], "missing");
});

test("a record must carry its identity", () => {
  const item = expandCases(sources.plan)[0];
  const missing = clone(record(item));
  delete missing.identity.engine.closureDigest;
  assert.throws(() => validateRecord(missing), /closure digest/);
  const unidentified = clone(record(item));
  unidentified.identity.decoder.revision = "";
  assert.throws(() => validateRecord(unidentified), /repo@revision/);
});

test("capture planning: CUDA builds the candle feature, Metal runs under the footprint guard, --out stays outside the repo", async () => {
  assert.ok(cargoTestArgv("cuda").includes("backend-candle"));
  assert.ok(!cargoTestArgv("metal").includes("backend-candle"));
  assert.deepEqual(cargoTestArgv("metal").slice(-6), ["--", "--ignored", "--exact", "--nocapture", "--test-threads", "1"]);
  assert.ok(cargoTestArgv("metal", { noRun: true }).includes("--no-run"));
  const out = path.join(os.tmpdir(), "yue2-profile-test");
  const metal = await planCapture({ caseId: "yue2:q4:metal:default", outDir: out, sources });
  assert.equal(metal.guarded.eventFile, path.join(out, "yue2__q4__metal__default", "watchdog.jsonl"));
  assert.equal(metal.env.SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE, "1");
  assert.equal(metal.env.SCENEWORKS_YUE2_PROFILE_SAMPLER, "metal-watchdog");
  const cuda = await planCapture({ caseId: "yue2:q4:cuda:default", outDir: out, gpuId: 1, sources });
  assert.equal(cuda.guarded, null);
  assert.equal(cuda.env.SCENEWORKS_GPU_ID, "1");
  assert.equal(cuda.env.SCENEWORKS_YUE2_PROFILE_SAMPLER, "cuda-nvidia-smi");
  await assert.rejects(
    planCapture({ caseId: "yue2:q4:metal:default", outDir: path.join(ROOT, ".tmp"), sources }),
    /outside the repository/,
  );
  await assert.rejects(planCapture({ caseId: "yue2:q2:metal:default", outDir: out, sources }), /unknown case/);
  assert.deepEqual(parseArgs(["capture", "--case", "x", "--gpu-id", "0", "--dry-run"]), {
    command: "capture", files: [], caseId: "x", gpuId: 0, dryRun: true,
  });
  assert.throws(() => parseArgs(["run", "--backend", "rocm"]), /unknown backend/);
  assert.equal(parseArgs(["capture", "--cuda-shared-device"]).cudaSharedDevice, true);
  const cudaItem = expandCases(sources.plan).find((item) => item.backend === "cuda");
  const metalItem = expandCases(sources.plan).find((item) => item.backend === "metal");
  const external = path.join(os.tmpdir(), "yue2-shared-profile-plan");
  await assert.rejects(planCapture({ caseId: cudaItem.id, outDir: external, cudaSharedDevice: true, sources }), /requires --gpu-id/);
  await assert.rejects(planCapture({ caseId: metalItem.id, outDir: external, gpuId: 1, cudaSharedDevice: true, sources }), /requires a CUDA case/);
  await assert.rejects(planCapture({ caseId: cudaItem.id, outDir: external, gpuId: 1, cudaSharedDevice: true, sources }), /requires an absolute --cuda-shared-device-proof/);
  const proofDir = await mkdtemp(path.join(os.tmpdir(), "yue2-shared-proof-"));
  try {
    const proofFile = path.join(proofDir, "proof.json");
    const files = Object.fromEntries(Array.from({ length: 29 }, (_, i) => [`raw-${i}.json`, "0".repeat(64)]));
    await writeFile(proofFile, JSON.stringify({ backend: "cuda", label: "before-strict-bf16-standard", admitted: true,
      census: JSON.stringify({ physicalMode: "shared-gpu1", admission: true, commandExit: 0,
        validatedDevice: { physicalMode: "shared-gpu1", physicalIndex: 1, cudaOrdinal: 0,
          uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", pci: "00000000:C1:00.0",
          luid: "luid_0x00000000_0x0001f78f" }, diagnosticFiles: files, diagnosticFileBytesB64: files }) }));
    const planned = await planCapture({ caseId: cudaItem.id, outDir: external, gpuId: 1,
      cudaSharedDevice: true, cudaSharedDeviceProof: proofFile, sources });
    assert.equal(planned.env.CUDA_VISIBLE_DEVICES, "1");
    assert.equal(planned.env.SCENEWORKS_GPU_ID, "1");
    assert.equal(planned.deviceProof.physicalIndex, 1);
    assert.match(planned.deviceProof.sha256, /^[0-9a-f]{64}$/);
  } finally { await rm(proofDir, { recursive: true, force: true }); }
  const envOptions = { caseFilePath: path.join(external, "case.json"), outDir: external, gpuId: 1, base: { CUDA_VISIBLE_DEVICES: "0" } };
  assert.equal(captureEnv({ ...envOptions, backend: "cuda" }).CUDA_VISIBLE_DEVICES, "1");
  assert.equal(captureEnv({ ...envOptions, backend: "metal" }).CUDA_VISIBLE_DEVICES, "0");
});

test("capture paths stay outside the repo across Windows volumes and exact directory boundaries", () => {
  const cases = [
    [path.win32, "D:\\actions\\app", "E:\\sceneworks-terminal\\cases\\case.json", false],
    [path.win32, "D:\\actions\\app", "D:\\actions\\cases\\case.json", false],
    [path.win32, "D:\\actions\\app", "D:\\actions\\app-next\\case.json", false],
    [path.win32, "D:\\actions\\app", "D:\\actions\\app", true],
    [path.win32, "D:\\actions\\app", "D:\\actions\\app\\cases\\case.json", true],
    [path.win32, "D:\\actions\\app", "D:\\actions\\app\\..cache\\case.json", true],
    [path.posix, "/repo/app", "/cases/case.json", false],
    [path.posix, "/repo/app", "/repo/cases/case.json", false],
    [path.posix, "/repo/app", "/repo/app-next/case.json", false],
    [path.posix, "/repo/app", "/repo/app", true],
    [path.posix, "/repo/app", "/repo/app/cases/case.json", true],
    [path.posix, "/repo/app", "/repo/app/..cache/case.json", true],
  ];
  for (const [pathApi, root, target, inside] of cases) {
    assert.equal(isWithinRepository(root, target, pathApi), inside, `${root} -> ${target}`);
  }
});

test("a record states the run's truncation, stage times and identities, or none of them (sc-23002)", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.backend === "metal");
  const run = {
    truncated: { abc: false, semantic: true },
    stageSeconds: { load: 2.5, plan: 7.5, semantic: 10, acoustic: 4, decode: 1.5 },
    engineTiming: { nar_seconds: 3.5, vae_seconds: 1.25 },
    runIdentity: "run-identity",
    planIdentity: "plan-identity",
    decoder: { release: "standard" },
    latent: { sha256: "c".repeat(64) },
  };
  const withRun = (outcome) => record(item, { outcome: { status: "completed", audioSeconds: 30, rms: 0.1, ...outcome } });
  const bare = record(item, { outcome: { status: "completed", audioSeconds: 30, rms: 0.1 } });
  // A legacy (v1) record captured before these fields existed stays valid and ingestible…
  const legacy = { ...bare, schema: "sceneworks-yue2-memory-profile-record-v1" };
  validateRecord(legacy);
  admitToCorpus(legacy, sources);
  // …but a current record without them is refused outright, at validation and at ingest.
  assert.throws(() => validateRecord(bare), /none of the run fields but not truncated/);
  assert.throws(() => admitToCorpus(bare, sources), /none of the run fields/);
  assert.throws(() => validateRecord({ ...legacy, outcome: { ...legacy.outcome, latent: { sha256: "c".repeat(64) } } }), /but not truncated/);
  // A record carrying them is valid and ingestible with them intact.
  const current = withRun(run);
  validateRecord(current);
  assert.deepEqual(admitToCorpus(current, sources).outcome.latent, run.latent);
  // A partial set is a broken capture, never a weaker record.
  assert.throws(() => validateRecord(withRun({ ...run, latent: undefined })), /but not latent/);
  assert.throws(() => validateRecord(withRun({ ...run, truncated: { abc: false, semantic: "yes" } })), /truncated\.semantic/);
  assert.throws(() => validateRecord(withRun({ ...run, stageSeconds: { ...run.stageSeconds, warmup: 1 } })), /unknown stage warmup/);
  assert.throws(() => validateRecord(withRun({ ...run, latent: { sha256: "short" } })), /latent has no sha256/);
  assert.throws(() => admitToCorpus(withRun({ ...run, runIdentity: null }), sources), /runIdentity/);
});
