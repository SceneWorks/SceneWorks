// sc-23002: the YuE2 terminal acceptance driver's pure parts — case planning, dependency skips,
// stage timing, WAV evidence, memory parsing, record validation and summary rendering. The driver's
// service half (API + worker + real renders) runs only on the terminal hosts.
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CASES,
  COVERED_ELSEWHERE,
  RECORD_SCHEMA,
  SONG_RUN_FILES,
  buildSummary,
  caseRunnerIds,
  coldBefore,
  exitCodeFor,
  hubModeOf,
  isLatentCorruptionRefusal,
  isMissingSourceRefusal,
  recordedServiceEnv,
  serviceDeviations,
  stopWorkerSafely,
  tierAssertions,
  workerBusy,
  dependencySkip,
  deviceMatches,
  footprintBytes,
  insideRepository,
  nvidiaUsedBytes,
  parseArgs,
  parseWav,
  peakOf,
  planCases,
  renderAssertions,
  renderMarkdown,
  serviceEnv,
  stageOf,
  stageTimeline,
  validateRecord,
  verifyRunArtifacts,
} from "./yue2-acceptance.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

test("every case has a runner and a unique id; dependencies name earlier cases", () => {
  const ids = CASES.map((item) => item.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual([...caseRunnerIds()].sort(), [...ids].sort());
  for (const [index, item] of CASES.entries()) {
    for (const need of [...item.needs, ...(item.dryNeeds ?? [])]) {
      assert.ok(ids.indexOf(need) >= 0 && ids.indexOf(need) < index, `${item.id} needs ${need}, which must run first`);
    }
  }
});

test("the plan runs every case on a real run and skips only with a stated reason", () => {
  const cuda = planCases({ platform: "cuda" });
  assert.ok(cuda.every((entry) => entry.action === "run"), JSON.stringify(cuda.filter((entry) => entry.action !== "run")));
  // A dry run starts no worker: only the worker-free cases run, every other one says why it did not.
  const dry = planCases({ platform: "metal", dryRun: true });
  assert.deepEqual(
    dry.filter((entry) => entry.action === "run").map((entry) => entry.id),
    ["catalog-preflight", "isolation-v1-v2", "transcription-blocked", "licence-gate"],
  );
  for (const entry of dry.filter((item) => item.action === "skip")) assert.match(entry.reason, /dry run/);
  // The operator's --skip is recorded as such.
  const skipped = planCases({ platform: "cuda", skip: ["batch-serial"] }).find((entry) => entry.id === "batch-serial");
  assert.deepEqual(skipped, { id: "batch-serial", action: "skip", reason: "skipped by the operator (--skip)" });
  assert.throws(() => planCases({ platform: "cuda", skip: ["no-such-case"] }), /unknown case/);
  assert.throws(() => planCases({ platform: "rocm" }), /metal or cuda/);
});

test("a Metal worker kill needs the owner's explicit opt-in; CUDA always runs it", () => {
  const metal = planCases({ platform: "metal" }).find((entry) => entry.id === "worker-kill-resume");
  assert.equal(metal.action, "skip");
  assert.match(metal.reason, /--allow-metal-worker-kill/);
  assert.equal(planCases({ platform: "metal", allowMetalWorkerKill: true }).find((entry) => entry.id === "worker-kill-resume").action, "run");
  assert.equal(planCases({ platform: "cuda" }).find((entry) => entry.id === "worker-kill-resume").action, "run");
});

test("a case whose dependency did not pass is skipped naming the dependency and its reason", () => {
  const decode = CASES.find((item) => item.id === "decode-standard-legacy");
  const records = new Map([
    ["install-cold", { status: "passed" }],
    ["create-cot-full", { status: "failed", reason: "job completed: status failed" }],
  ]);
  assert.equal(dependencySkip(decode, records), "dependency create-cot-full failed: job completed: status failed");
  assert.equal(dependencySkip(decode, new Map([["install-cold", { status: "passed" }]])), "dependency create-cot-full has not run");
  records.set("create-cot-full", { status: "passed" });
  assert.equal(dependencySkip(decode, records), null);
  // The licence gate needs installed weights to execute its job — except in a dry run, which only
  // exercises the refusal and the submission.
  const licence = CASES.find((item) => item.id === "licence-gate");
  const notInstalled = new Map([["install-cold", { status: "skipped", reason: "dry run" }]]);
  assert.match(dependencySkip(licence, notInstalled), /install-cold skipped/);
  assert.equal(dependencySkip(licence, notInstalled, { dryRun: true }), null);
});

test("arguments: platform and out are required; --skip repeats; unknown input is refused", () => {
  const options = parseArgs(["--platform", "cuda", "--out", "E:/evidence", "--skip", "batch-serial", "--skip", "admission", "--hf-home", "E:/run/hf-home", "--dry-run"]);
  assert.equal(options.platform, "cuda");
  assert.deepEqual(options.skip, ["batch-serial", "admission"]);
  assert.equal(options.hfHome, "E:/run/hf-home");
  assert.throws(() => parseArgs(["--platform", "cuda", "--out", "x", "--hf-hub", "E:/huggingface/hub"]), /unknown argument --hf-hub/);
  assert.equal(options.dryRun, true);
  assert.throws(() => parseArgs(["--out", "x"]), /--platform/);
  assert.throws(() => parseArgs(["--platform", "metal"]), /--out/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--skip", "nope"]), /unknown case/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--bogus"]), /unknown argument/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--port", "70000"]), /TCP port/);
});

test("the stage comes from the worker's progress message; timings run from one stage to the next", () => {
  const at = (stage, message, status = "running") => ({ status, message, stage });
  assert.equal(stageOf(at(null, "Checking YuE2 eligibility.", "preparing")), "eligibility");
  assert.equal(stageOf(at(null, "Loading YuE2.")), "load");
  assert.equal(stageOf(at(null, "Planning the score: token 4 of 4096.")), "plan");
  assert.equal(stageOf(at(null, "Generating semantic tokens: 9 of 1125.")), "semantic");
  assert.equal(stageOf(at(null, "Acoustic synthesis: step 2 of 32.")), "acoustic");
  assert.equal(stageOf(at(null, "Decoding audio.")), "decode");
  assert.equal(stageOf(at(null, "Saving the YuE2 run.", "saving")), "saving");
  assert.equal(stageOf({ status: "queued" }), "queued");
  assert.equal(stageOf({ status: "canceled", message: "YuE2 job canceled by user." }), "canceled");
  assert.equal(stageOf(at(null, "Cancellation requested. Waiting for worker acknowledgement.")), "cancel-requested");
  const timeline = stageTimeline([
    { at: 0, stage: "queued" },
    { at: 1000, stage: "load" },
    { at: 5000, stage: "semantic" },
    { at: 6000, stage: "semantic" },
    { at: 15000, stage: "acoustic" },
    { at: 18000, stage: "decode" },
    { at: 19500, stage: "completed" },
  ]);
  assert.deepEqual(timeline.order, ["queued", "load", "semantic", "acoustic", "decode", "completed"]);
  assert.deepEqual(timeline.seconds, { queued: 1, load: 4, semantic: 10, acoustic: 3, decode: 1.5, completed: 0 });
});

function wav({ tag, bits, channels, rate, samples, extensible = false }) {
  const bytes = bits / 8;
  const data = Buffer.alloc(samples.length * bytes);
  samples.forEach((value, index) => {
    if (tag === 1) data.writeInt16LE(Math.round(value * 32767), index * bytes);
    else data.writeFloatLE(value, index * bytes);
  });
  const fmt = Buffer.alloc(extensible ? 40 : 16);
  fmt.writeUInt16LE(extensible ? 0xfffe : tag, 0);
  fmt.writeUInt16LE(channels, 2);
  fmt.writeUInt32LE(rate, 4);
  fmt.writeUInt32LE(rate * channels * bytes, 8);
  fmt.writeUInt16LE(channels * bytes, 12);
  fmt.writeUInt16LE(bits, 14);
  if (extensible) {
    fmt.writeUInt16LE(22, 16);
    fmt.writeUInt16LE(tag, 24);
  }
  // An INFO list before the data, as the worker's licence-tagged asset carries.
  const info = Buffer.concat([Buffer.from("LIST"), Buffer.from([4, 0, 0, 0]), Buffer.from("INFO")]);
  const chunk = (id, body) => Buffer.concat([Buffer.from(id), Buffer.from(Uint32Array.of(body.length).buffer), body]);
  const body = Buffer.concat([Buffer.from("WAVE"), chunk("fmt ", fmt), info, chunk("data", data)]);
  return Buffer.concat([Buffer.from("RIFF"), Buffer.from(Uint32Array.of(body.length).buffer), body]);
}

test("WAV evidence: duration and RMS of the run's float WAV and the PCM16 asset", () => {
  const samples = Array.from({ length: 48000 * 2 }, (_, index) => (index % 2 ? 0.5 : -0.5));
  const float = parseWav(wav({ tag: 3, bits: 32, channels: 2, rate: 48000, samples }));
  assert.equal(float.encoding, "float32");
  assert.equal(float.sampleRate, 48000);
  assert.equal(float.channels, 2);
  assert.equal(float.durationSeconds, 1);
  assert.ok(Math.abs(float.rms - 0.5) < 1e-6, float.rms);
  const pcm = parseWav(wav({ tag: 1, bits: 16, channels: 2, rate: 48000, samples, extensible: true }));
  assert.equal(pcm.encoding, "pcm16");
  assert.equal(pcm.durationSeconds, 1);
  assert.ok(Math.abs(pcm.rms - 0.5) < 1e-3, pcm.rms);
  assert.equal(parseWav(wav({ tag: 3, bits: 32, channels: 2, rate: 48000, samples: [0, 0, 0, 0] })).rms, 0);
  assert.throws(() => parseWav(Buffer.from("not a wav file at all")), /RIFF/);
  assert.throws(() => parseWav(wav({ tag: 1, bits: 16, channels: 1, rate: 8000, samples: [0] }).subarray(0, 36)), /fmt or data/);
});

test("memory samples: footprint sums the service pids, nvidia-smi is MiB, the peak keeps its gaps", () => {
  const payload = { processes: [
    { pid: 10, auxiliary: { phys_footprint: 1_000 } },
    { pid: 11, auxiliary: { phys_footprint: 2_000 } },
    { pid: 99, auxiliary: { phys_footprint: 9_999_999 } },
  ] };
  assert.equal(footprintBytes(payload, [10, 11]), 3_000);
  assert.throws(() => footprintBytes({ processes: [{ pid: 10, auxiliary: {} }] }, [10]), /phys_footprint/);
  assert.throws(() => footprintBytes({ processes: [] }, [10]), /none of the service pids/);
  assert.equal(nvidiaUsedBytes("2048\n"), 2048 * 1024 * 1024);
  assert.throws(() => nvidiaUsedBytes("[N/A]"), /memory\.used/);
  assert.deepEqual(peakOf([{ bytes: 5 }, { bytes: 9 }, { bytes: 7 }], ["timeout"], "s"), {
    sampler: "s", samples: 3, faults: 1, firstFault: "timeout", peakBytes: 9, baselineBytes: 5,
  });
  assert.equal(peakOf([], ["timeout"], "s").peakBytes, null);
});

function renderEvidence(overrides = {}) {
  const identity = "run-1";
  return {
    platform: "metal",
    job: { status: "completed", result: { yue2: { run: { identity }, truncated: { abc: false, semantic: true } } } },
    runResult: { identity, truncated: { abc: false, semantic: true } },
    runConfig: { device: "metal", model_dtype: "bfloat16" },
    artifactCheck: { ok: true, detail: "10 artifacts re-hashed" },
    runAudio: { sampleRate: 48000, channels: 2, rms: 0.12, durationSeconds: 31.5 },
    assetAudio: { durationSeconds: 31.5 },
    ...overrides,
  };
}

test("render evidence: production device, the engine's own identity and truncation, verified artifacts, audible 48 kHz stereo", () => {
  const ok = renderAssertions(renderEvidence());
  assert.ok(ok.every((assertion) => assertion.ok), JSON.stringify(ok.filter((assertion) => !assertion.ok)));
  const failing = (overrides) => renderAssertions(renderEvidence(overrides)).filter((assertion) => !assertion.ok).map((assertion) => assertion.name);
  // A CPU fallback is not production-device evidence, whatever else passed.
  assert.deepEqual(failing({ runConfig: { device: "cpu" } }), ["production device"]);
  assert.deepEqual(failing({ platform: "cuda" }), ["production device"]);
  assert.deepEqual(failing({ runResult: { identity: "other", truncated: { abc: false, semantic: true } } }), ["run identity"]);
  // The app must report the engine's truncation, not a default.
  assert.deepEqual(failing({ job: { status: "completed", result: { yue2: { run: { identity: "run-1" }, truncated: { abc: false, semantic: false } } } } }), ["truncation reported"]);
  assert.deepEqual(failing({ runResult: { identity: "run-1" } }), ["truncation reported"]);
  assert.deepEqual(failing({ artifactCheck: { ok: false, detail: "latent.npy hashes to x" } }), ["artifacts verify"]);
  assert.deepEqual(failing({ runAudio: { sampleRate: 44100, channels: 2, rms: 0.1, durationSeconds: 31.5 } }), ["48 kHz stereo"]);
  assert.deepEqual(failing({ runAudio: { sampleRate: 48000, channels: 2, rms: 0, durationSeconds: 31.5 } }), ["audible"]);
  // A cached decode's top-level device is the SOURCE run's; its own device is cached_decode.device.
  const decode = { runResult: { kind: "cached_decode", identity: "run-1", truncated: { abc: false, semantic: true } } };
  assert.deepEqual(failing({ ...decode, runConfig: { device: "metal", cached_decode: { device: "metal" } } }), []);
  assert.deepEqual(failing({ ...decode, runConfig: { device: "metal", cached_decode: { device: "cpu" } } }), ["decode ran on the production device"]);
  assert.deepEqual(failing({ ...decode, runConfig: { device: "metal" } }), ["decode ran on the production device"]);
  // The library asset must exist, parse, and last as long as the run's audio (within one frame).
  assert.deepEqual(failing({ assetAudio: null }), ["library asset is the run's audio"]);
  assert.deepEqual(failing({ assetAudio: { durationSeconds: 31.5 + 1 / 48000 } }), []);
  assert.deepEqual(failing({ assetAudio: { durationSeconds: 31.5 + 2 / 48000 } }), ["library asset is the run's audio"]);
  assert.equal(deviceMatches("cuda", "cuda"), true);
  assert.equal(deviceMatches("metal", "cuda"), false);
});

const PINS = { q8: "8".repeat(64), q4: "4".repeat(64) };
const MANIFEST_ENTRY = { downloads: [
  { variant: "bf16", default: true },
  { variant: "q8", localDerivation: { weightsSha256: PINS.q8 } },
  { variant: "q4", localDerivation: { weightsSha256: PINS.q4 } },
  { componentId: "vae", coRequisite: true },
] };
const tierRun = (tier, sha, overrides = {}) => ({
  tier,
  runConfig: { weight_tier: tier, quantization: "none", ...overrides.config },
  runResult: { weights: { weight_tier: tier, quantization: "none", mot_native_weights_sha256: sha, ...overrides.weights } },
  manifestEntry: MANIFEST_ENTRY,
});
const tierFailing = (input) => tierAssertions(input).filter((assertion) => !assertion.ok).map((assertion) => assertion.name);

test("tier evidence comes from the engine: tier, no FP8, and the pinned derived weights", () => {
  assert.deepEqual(tierFailing(tierRun("bf16", "b".repeat(64))), []);
  assert.deepEqual(tierFailing(tierRun("q8", PINS.q8)), []);
  assert.deepEqual(tierFailing(tierRun("q4", PINS.q4)), []);
  // The worker's echo of the request is not evidence: the engine's own records must say the tier.
  assert.deepEqual(tierFailing(tierRun("q8", PINS.q8, { config: { weight_tier: "bf16" } })), ["engine ran the q8 tier"]);
  assert.deepEqual(tierFailing(tierRun("q8", PINS.q8, { weights: { weight_tier: "bf16" } })), ["engine ran the q8 tier"]);
  assert.deepEqual(tierFailing(tierRun("q4", PINS.q4, { config: { quantization: "fp8" } })), ["AR ran without FP8"]);
  // A derived tier must have loaded exactly its pinned weights; bf16 none of them.
  assert.deepEqual(tierFailing(tierRun("q4", PINS.q8)), ["loaded the pinned derived q4 weights"]);
  assert.deepEqual(tierFailing(tierRun("q8", "b".repeat(64))), ["loaded the pinned derived q8 weights"]);
  assert.deepEqual(tierFailing(tierRun("bf16", PINS.q4)), ["loaded no derived tier's weights"]);
});

test("refusal matchers name the specific failure, not any error mentioning a record", () => {
  const corrupt = `yue2: the source run does not verify against its recorded identity: /p/yue2/runs/yue2run_1/latent.npy: SHA-256 ${"a".repeat(64)}, recorded ${"b".repeat(64)}`;
  assert.equal(isLatentCorruptionRefusal(corrupt), true);
  assert.equal(isLatentCorruptionRefusal(corrupt.replace("latent.npy", "semantic.npy")), false);
  assert.equal(isLatentCorruptionRefusal("yue2: the run was recorded as failed"), false);
  assert.equal(isLatentCorruptionRefusal("YuE2 is not eligible to run now: the acknowledgment recorded is stale"), false);
  assert.equal(isLatentCorruptionRefusal(null), false);
  assert.equal(isMissingSourceRefusal("yue2: the source run /p/yue2/runs/yue2run_1 is missing"), true);
  assert.equal(isMissingSourceRefusal("yue2: no decoder is installed; missing weights"), false);
});

test("a cold install starts from nothing installed on a fresh hub", async () => {
  const none = { installState: "missing", variants: [{ variant: "bf16", installed: false, installState: "missing" }, { variant: "q8", installed: false, installState: "derivationPending" }] };
  assert.equal(coldBefore(none), true);
  assert.equal(coldBefore({ ...none, installState: "installed" }), false);
  assert.equal(coldBefore({ ...none, variants: [...none.variants, { variant: "q4", installed: true, installState: "installed" }] }), false);
  assert.equal(coldBefore({ installState: "missing", variants: [] }), false);
  const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-acceptance-hub-"));
  try {
    assert.equal(await hubModeOf(path.join(dir, "absent")), "fresh");
    assert.equal(await hubModeOf(dir), "fresh");
    await mkdir(path.join(dir, "models--m-a-p--YuE2-3B"));
    assert.equal(await hubModeOf(dir), "preseeded");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("the artifact check re-hashes every recorded file and names the first that does not verify", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-acceptance-run-"));
  try {
    const artifacts = {};
    for (const name of [...SONG_RUN_FILES, "stages/plan.json"]) {
      const body = Buffer.from(`bytes of ${name}`);
      await mkdir(path.dirname(path.join(dir, name)), { recursive: true });
      await writeFile(path.join(dir, name), body);
      artifacts[name] = { sha256: sha(body), bytes: body.length };
    }
    const result = { kind: "song", artifacts };
    assert.deepEqual(await verifyRunArtifacts(dir, result), { ok: true, detail: `${SONG_RUN_FILES.length + 1} artifacts re-hashed`, files: SONG_RUN_FILES.length + 1 });
    await writeFile(path.join(dir, "latent.npy"), "bytes of latent.npX");
    assert.match((await verifyRunArtifacts(dir, result)).detail, /latent\.npy hashes to/);
    await rm(path.join(dir, "audio.wav"));
    assert.match((await verifyRunArtifacts(dir, result)).detail, /audio\.wav is missing|latent\.npy/);
    const { "audio.wav": _dropped, ...partial } = artifacts;
    assert.match((await verifyRunArtifacts(dir, { kind: "song", artifacts: partial })).detail, /omits audio\.wav/);
    assert.equal((await verifyRunArtifacts(dir, { kind: "song" })).ok, false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

function passedRecord(caseId = "create-cot-full", overrides = {}) {
  return {
    schema: RECORD_SCHEMA,
    caseId,
    acceptance: ["AT1"],
    platform: "metal",
    memorySampled: true,
    status: "passed",
    reason: null,
    assertions: [{ name: "job completed", ok: true }],
    requests: [],
    jobs: [{
      jobId: "job_1", kind: "create", status: "completed", rendered: true, tier: "bf16", runIdentity: "run-1",
      output: { runAudioSha256: "a".repeat(64), durationSeconds: 31.5, rms: 0.1 },
      truncated: { abc: false, semantic: false }, device: "metal", dtype: { model: "bfloat16", vae: "float32" },
      stageTimings: { observed: { order: [], seconds: {} } }, durationSeconds: 120,
    }],
    peakMemory: { peakBytes: 12 * 1024 ** 3 },
    ...overrides,
  };
}

test("a record stands as evidence only when it is complete; a non-pass must say why", () => {
  validateRecord(passedRecord());
  const job = passedRecord().jobs[0];
  const broken = (overrides) => () => validateRecord(passedRecord("create-cot-full", overrides));
  assert.throws(broken({ assertions: [{ name: "audible", ok: false }] }), /failing assertions: audible/);
  assert.throws(broken({ assertions: [] }), /asserted nothing/);
  assert.throws(broken({ status: "skipped" }), /must say why/);
  assert.throws(broken({ status: "failed", reason: " " }), /must say why/);
  assert.throws(broken({ status: "blocked", reason: "x" }), /only an expected refusal/);
  const blockers = [{ componentId: "yue2_sheetsage2", reason: "owner licensing decision", unblock: "record a licence basis" }];
  validateRecord(passedRecord("transcription-blocked", { status: "blocked", reason: "component_blocked", jobs: [], blockers }));
  assert.throws(() => validateRecord(passedRecord("transcription-blocked", { status: "blocked", reason: "component_blocked", jobs: [] })), /reason and unblock/);
  assert.throws(() => validateRecord(passedRecord("transcription-blocked", { status: "blocked", reason: "x", jobs: [], blockers: [{ reason: "r" }] })), /reason and unblock/);
  assert.throws(broken({ jobs: [{ ...job, output: { ...job.output, runAudioSha256: undefined } }] }), /no output sha256/);
  assert.throws(broken({ jobs: [{ ...job, truncated: { abc: false } }] }), /no truncation flags/);
  assert.throws(broken({ jobs: [{ ...job, device: null }] }), /device\/dtype/);
  assert.throws(broken({ jobs: [{ ...job, stageTimings: undefined }] }), /stage timings/);
  assert.throws(broken({ peakMemory: { peakBytes: null, firstFault: "footprint exited 1" } }), /no peak memory sample \(footprint exited 1\)/);
  validateRecord(passedRecord("catalog-preflight", { memorySampled: false, peakMemory: null, jobs: [] }));
  assert.throws(broken({ caseId: "long-context" }), /unknown case/);
});

function summaryOf(statuses, meta = {}) {
  const records = CASES.map((item) => ({
    ...passedRecord(item.id, { jobs: [] }),
    status: statuses[item.id] ?? (item.expectBlocked ? "blocked" : "passed"),
    reason: statuses[item.id] && statuses[item.id] !== "passed" ? `why ${item.id}` : item.expectBlocked ? "blocked as designed" : null,
    blockers: (statuses[item.id] ?? (item.expectBlocked ? "blocked" : "passed")) === "blocked"
      ? [{ componentId: "yue2_sheetsage2", reason: "owner licensing decision", unblock: "record a licence basis" }]
      : undefined,
  }));
  return buildSummary(records, { platform: "cuda", ...meta });
}

test("the verdict passes only a complete run: any failure fails it, any skip or dry run leaves it incomplete", () => {
  // The designed transcription refusal is an open owner decision: never a plain pass, never exit 0.
  const blocked = summaryOf({});
  assert.equal(blocked.verdict, "pass-with-blockers");
  assert.equal(blocked.counts.blocked, 1);
  assert.deepEqual(blocked.blockers, [{ caseId: "transcription-blocked", acceptance: ["AT1"], componentId: "yue2_sheetsage2", reason: "owner licensing decision", unblock: "record a licence basis" }]);
  assert.equal(exitCodeFor(blocked), 2);
  const clean = summaryOf({ "transcription-blocked": "passed" });
  assert.equal(clean.verdict, "pass");
  assert.equal(exitCodeFor(clean), 0);
  assert.equal(exitCodeFor(summaryOf({ "batch-serial": "failed" })), 1);
  assert.equal(exitCodeFor(summaryOf({ "batch-serial": "skipped" })), 1);
  assert.equal(exitCodeFor(summaryOf({}, { dryRun: true })), 0);
  assert.equal(exitCodeFor(summaryOf({ "batch-serial": "failed" }, { dryRun: true })), 1);
  assert.equal(summaryOf({ "batch-serial": "failed" }).verdict, "fail");
  assert.equal(summaryOf({ "batch-serial": "skipped" }).verdict, "incomplete");
  assert.equal(summaryOf({ "batch-serial": "skipped", admission: "failed" }).verdict, "fail");
  assert.equal(summaryOf({}, { dryRun: true }).verdict, "incomplete");
  // A run that stopped early fails, even when nothing it recorded failed.
  const stopped = buildSummary([passedRecord("catalog-preflight", { jobs: [] })], { platform: "cuda", fatal: "the worker did not register idle in 180 s" });
  assert.equal(stopped.verdict, "fail");
  assert.equal(stopped.fatal, "the worker did not register idle in 180 s");
  const partial = buildSummary([passedRecord("catalog-preflight", { jobs: [] })], { platform: "cuda" });
  assert.equal(partial.verdict, "incomplete");
  assert.equal(partial.missing.length, CASES.length - 1);
  assert.deepEqual(summaryOf({}).coveredElsewhere, COVERED_ELSEWHERE);
});

test("the markdown table has one row per case, carries hashes not audio, and escapes table syntax", () => {
  const records = [passedRecord(), { ...passedRecord("batch-serial", { jobs: [] }), status: "failed", reason: "take 2 | overlapped\ntake 1" }];
  const markdown = renderMarkdown(buildSummary(records, { platform: "metal", identity: { sceneworksRevision: "b".repeat(40), inferencePin: "c".repeat(40), apiBinarySha256: "d".repeat(64) } }));
  assert.match(markdown, /Verdict: \*\*fail\*\*/);
  assert.match(markdown, /\| create-cot-full \| AT1 \| passed \| create·bf16·metal\/bfloat16 completed \| aaaaaaaaaaaaaaaa \| 31\.5 \| 0\.1000 \| false\/false \| 12\.00 GiB \|/);
  assert.match(markdown, /take 2 \\\| overlapped take 1/);
  assert.match(markdown, /long-context/);
  assert.match(markdown, /CC BY-NC 4\.0/);
  assert.doesNotMatch(markdown, /\.wav/);
  // A blocker is listed with its reason and unblock condition, as an owner decision.
  const withBlocker = renderMarkdown(summaryOf({}));
  assert.match(withBlocker, /Verdict: \*\*pass-with-blockers\*\*/);
  assert.match(withBlocker, /## Blockers — owner decision required \(not a pass\)/);
  assert.match(withBlocker, /\*\*transcription-blocked\*\* \(AT1\) — yue2_sheetsage2: owner licensing decision\. Unblock: record a licence basis/);
  assert.doesNotMatch(renderMarkdown(summaryOf({ "transcription-blocked": "passed" })), /Blockers/);
});

const SHELL = {
  PATH: "/bin", HOME: "/home/u",
  HF_HOME: "/home/hf", HF_HUB_CACHE: "/shared/hub", HUGGINGFACE_HUB_CACHE: "/shared/hub", HF_HUB_OFFLINE: "1", HF_TOKEN: "hf_secret",
  SCENEWORKS_ACCESS_TOKEN: "t", SCENEWORKS_MLX_MEMORY_CAP_GB: "8", SCENEWORKS_HEARTBEAT_SECONDS: "5", SCENEWORKS_JOBS_DB_PATH: "/elsewhere.db",
  SCENEWORKS_RESOLVED_CACHE_ENABLED: "true", CUDA_VISIBLE_DEVICES: "3", TRANSFORMERS_CACHE: "/t",
};
const common = { base: SHELL, url: "http://127.0.0.1:5000", port: 5000, dataDir: "D", configDir: "C", hfHome: "H", workerId: "w", driverPid: 42 };
const driverKeys = (env) => Object.keys(env).filter((key) => /^(SCENEWORKS_|HF_|HUGGINGFACE_|TRANSFORMERS_)/.test(key) || key === "CUDA_VISIBLE_DEVICES").sort();

test("the services get the desktop's shipped environment and nothing inherited from the shell", () => {
  const resolved = { SCENEWORKS_RESOLVED_CACHE_ENABLED: "false", SCENEWORKS_RESOLVED_CACHE_MAX_BYTES: "68719476736", SCENEWORKS_RESOLVED_CACHE_INACTIVITY_SECONDS: "1209600" };
  const shared = { SCENEWORKS_DATA_DIR: "D", SCENEWORKS_CONFIG_DIR: "C", HF_HOME: "H", ...resolved };
  // apps/desktop/src/setup.rs spawn_api, per platform.
  const apiCommon = { ...shared, SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: "5000", SCENEWORKS_TRUST_LOOPBACK: "true", SCENEWORKS_RUN_UTILITY_INPROCESS: "true", SCENEWORKS_PARENT_PID: "42" };
  const pick = (env) => Object.fromEntries(driverKeys(env).map((key) => [key, env[key]]));
  assert.deepEqual(pick(serviceEnv({ ...common, platform: "metal", role: "api" })), { ...apiCommon, SCENEWORKS_MLX_REQUIRED: "1" });
  assert.deepEqual(pick(serviceEnv({ ...common, platform: "cuda", role: "api" })), { ...apiCommon, SCENEWORKS_CANDLE_REQUIRED: "1", SCENEWORKS_CANDLE_UNSUPPORTED_MODE: "enforce" });
  // supervise_mlx_worker (no parent-death watch on Metal: see serviceDeviations).
  assert.deepEqual(pick(serviceEnv({ ...common, platform: "metal", role: "worker" })), {
    ...shared, SCENEWORKS_WORKER_ONLY: "1", SCENEWORKS_GPU_ID: "mlx", SCENEWORKS_WORKER_ID: "w", SCENEWORKS_API_URL: "http://127.0.0.1:5000",
  });
  // supervise_candle_worker's env + supervisor.rs child_environment for GPU 0.
  assert.deepEqual(pick(serviceEnv({ ...common, platform: "cuda", role: "worker", gpuId: "0" })), {
    ...shared, SCENEWORKS_WORKER_ONLY: "1", SCENEWORKS_BACKEND_CANDLE_ENABLED: "true", SCENEWORKS_WORKER_ID: "w", SCENEWORKS_API_URL: "http://127.0.0.1:5000",
    SCENEWORKS_PARENT_PID: "42", SCENEWORKS_WORKER_CHILD: "1", SCENEWORKS_GPU_ID: "0", CUDA_VISIBLE_DEVICES: "0", SCENEWORKS_UTILITY_JOBS: "0",
  });
  // Non-app variables pass through.
  assert.equal(serviceEnv({ ...common, platform: "metal", role: "api" }).PATH, "/bin");
  assert.throws(() => serviceEnv({ ...common, platform: "metal", role: "both" }), /unknown service role/);
  assert.equal(serviceDeviations("metal").length, 2);
  assert.match(serviceDeviations("cuda").join(" "), /per-GPU child/);
});

test("offline takes the hub away for real; an admission cap is the only extra; secrets are never recorded", () => {
  const offline = serviceEnv({ ...common, platform: "metal", role: "worker", offline: true, extra: { SCENEWORKS_MLX_MEMORY_CAP_GB: "24" } });
  assert.equal(offline.HF_HUB_OFFLINE, "1");
  assert.equal(offline.TRANSFORMERS_OFFLINE, "1");
  assert.equal(offline.SCENEWORKS_HUGGINGFACE_BASE_URL, "http://127.0.0.1:9");
  assert.equal(offline.SCENEWORKS_MLX_MEMORY_CAP_GB, "24");
  const recorded = recordedServiceEnv({ ...offline, SCENEWORKS_ACCESS_TOKEN: "t", HF_TOKEN: "x" });
  assert.equal(recorded.SCENEWORKS_ACCESS_TOKEN, undefined);
  assert.equal(recorded.HF_TOKEN, undefined);
  assert.equal(recorded.PATH, undefined);
  assert.equal(recorded.HF_HOME, "H");
  assert.deepEqual(Object.keys(recorded), [...Object.keys(recorded)].sort());
});

function fakeWorker(rows) {
  const calls = { quiesce: 0, signals: [] };
  let index = 0;
  return {
    calls,
    alive: () => true,
    quiesce: async () => { calls.quiesce += 1; return { idle: false, reason: "still busy" }; },
    readRow: async () => {
      const row = rows[Math.min(index, rows.length - 1)];
      index += 1;
      if (row instanceof Error) throw row;
      return row;
    },
    signalStop: async (options) => { calls.signals.push(options); },
  };
}

test("a busy (or unreadable) Metal worker is never signalled unless the owner allowed it", async () => {
  const busy = { id: "w", status: "busy", currentJobId: "job_7" };
  for (const row of [busy, { id: "w", status: "idle", currentJobId: "job_7" }, new Error("ECONNREFUSED")]) {
    const fake = fakeWorker([row]);
    const outcome = await stopWorkerSafely({ platform: "metal", allowMetalWorkerKill: false, ...fake });
    assert.equal(outcome.stopped, false);
    assert.deepEqual(fake.calls.signals, [], "no signal reaches a Metal worker that may be mid-render");
    assert.equal(fake.calls.quiesce, 1, "the job is cancelled through the API first");
    assert.match(outcome.reason, /left running/);
  }
  // An idle Metal worker is stopped with SIGTERM only (no escalation).
  const idle = fakeWorker([{ id: "w", status: "idle", currentJobId: null }]);
  assert.deepEqual(await stopWorkerSafely({ platform: "metal", allowMetalWorkerKill: false, ...idle }), { stopped: true, signaled: true, reason: null });
  assert.deepEqual(idle.calls.signals, [{ force: false }]);
  // With the owner's say-so, or on CUDA, a busy worker is stopped (and may be escalated).
  const allowed = fakeWorker([busy]);
  assert.equal((await stopWorkerSafely({ platform: "metal", allowMetalWorkerKill: true, ...allowed })).stopped, true);
  assert.deepEqual(allowed.calls.signals, [{ force: true }]);
  const cuda = fakeWorker([busy]);
  assert.equal((await stopWorkerSafely({ platform: "cuda", allowMetalWorkerKill: false, ...cuda })).stopped, true);
  assert.deepEqual(cuda.calls.signals, [{ force: true }]);
  // A worker already gone needs nothing.
  const gone = { ...fakeWorker([busy]), alive: () => false };
  assert.equal((await stopWorkerSafely({ platform: "metal", allowMetalWorkerKill: false, ...gone })).signaled, false);
  assert.equal(workerBusy(null), false);
  assert.equal(workerBusy(undefined), true);
});

test("evidence and app state must live outside the repository", () => {
  const root = path.resolve("/repo/SceneWorks");
  assert.equal(insideRepository(root, root), true);
  assert.equal(insideRepository(path.join(root, "target", "evidence"), root), true);
  assert.equal(insideRepository(path.resolve("/tmp/yue2"), root), false);
  assert.equal(insideRepository(path.resolve("/repo/SceneWorks-evidence"), root), false);
});
