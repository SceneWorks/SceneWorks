import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CARGO_METADATA_TIMEOUT_MS,
  CHILD_ATTESTATION_TIMEOUT_SECONDS,
  CanaryInterrupted,
  MAX_FOOTPRINT_BYTES,
  MAX_RUNTIME_SECONDS,
  MIN_PREFLIGHT_FREE_BYTES,
  MIN_RUNTIME_FREE_BYTES,
  MIN_SWAP_FREE_BYTES,
  canaryRequest,
  cargoSourceStatusIsClean,
  cleanupCanaryScratch,
  cloneArtifactTree,
  exactToolchain,
  foreignHeavyProcesses,
  inventoryAtRoot,
  parseMemoryFreePercent,
  parseSwapFreeBytes,
  privateArtifactRoots,
  preservePrimaryFailure,
  preflightFreeFloor,
  repositoryToolchain,
  runtimeFreeFloor,
  telemetryResolutionBytes,
  validateCanaryResponse,
  watchdogFailureSummary,
} from "./run-ltx-safety-canary.mjs";

const TEXT_ENCODER_INVENTORY = {
  root: "/models/gemma",
  files: 17,
  bytes: 26_427_894_918,
  sha256: "abde2d155aa8991747cc2999d40688d29a50261c080c0d51fac20357653928d7",
};

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
    videoMode: "default", fps: 24, seed: 1234,
  });
  assert.equal(request.planned._watchdog.maxFootprintBytes, 53_347_146_863);
  assert.deepEqual(request.planned._artifact.numericTierInventory, {
    files: 11,
    bytes: 20_467_690_460,
    sha256: "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a",
  });
  assert.throws(() => canaryRequest(128 * 1024 ** 3));
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
  assert.match(runnerSource,
    /mkdir\(path\.dirname\(privateNumericTierRoot\), \{ recursive: true, mode: 0o700 \}\)/);
});

test("the runner refuses promotable or identity-drifted adapter output", () => {
  const response = {
    status: "diagnostic_canary_complete",
    inferenceRevision: "6f3a84ef4ad4f858c6fe199e14925a01a7943f97",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    calibrationFingerprint: "sc-19109-ltx-2-3-mlx-memory-ladder-v1",
    target: {
      provider: "ltx_2_3",
      tier: "q4",
      geometry: { width: 256, height: 256, frames: 9, fps: 24 },
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
    },
    watchdog: {
      required: true,
      protocol: "sceneworks-memory-watchdog-v1",
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes: 128 * 1024 ** 3,
      minInitialMemoryFreeBytes: preflightFreeFloor(128 * 1024 ** 3),
      minInitialMemoryFreePercent: 70,
      minMemoryFreeBytes: runtimeFreeFloor(128 * 1024 ** 3),
      minSwapFreeBytes: MIN_SWAP_FREE_BYTES,
    },
    mlxLimits: {
      memoryLimitBytes: MAX_FOOTPRINT_BYTES,
      wiredLimitBytes: MAX_FOOTPRINT_BYTES,
    },
    observedMemory: { postCleanupActiveBytes: 0, postCleanupCacheBytes: 0 },
    output: { firstFrameNondegenerate: true },
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
    (value) => { value.target.audio = false; },
    (value) => { value.target.geometry.frames = 97; },
    (value) => { value.strategy.parameters.decodeTileEdge = 384; },
    (value) => { value.strategy.engagedRungs = ["resident", "bounded_decode"]; },
    (value) => { value.watchdog.required = false; },
    (value) => { value.watchdog.protocol = "self-asserted"; },
    (value) => { value.watchdog.maxFootprintBytes += 1; },
    (value) => { value.watchdog.maxRuntimeSeconds += 1; },
    (value) => { value.watchdog.minInitialMemoryFreeBytes -= 1; },
    (value) => { value.watchdog.minInitialMemoryFreePercent -= 1; },
    (value) => { value.watchdog.minMemoryFreeBytes -= 1; },
    (value) => { value.mlxLimits.wiredLimitBytes = MAX_FOOTPRINT_BYTES + 1; },
    (value) => { value.observedMemory.postCleanupActiveBytes = 1; },
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
});

test("host preflight parsers reject pressure, swap and executable-identity mutations", () => {
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
  ].join("\n");
  assert.deepEqual(
    foreignHeavyProcesses(table).map((line) => Number(line.split(/\s+/, 1)[0])),
    [10, 11, 12, 13, 15],
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
  assert.match(source, /"--min-swap-free-bytes", String\(MIN_SWAP_FREE_BYTES\)/);
  assert.match(source, /status\.code !== 0 \|\| status\.signal/);
  assert.match(source, /event\.event === "hard_stop" \|\| event\.event === "terminated"/);
  assert.doesNotMatch(source, /memory-mlx-adapter[^\n]*spawn\(/);
  assert.doesNotMatch(source, /--child-adapter|async function child\(/);
  assert.match(source, /buildExactAdapter\(/);
  assert.match(source, /inferenceCargoSource\(/);
  assert.equal(CARGO_METADATA_TIMEOUT_MS, 15 * 60 * 1000);
  assert.match(source, /timeout: CARGO_METADATA_TIMEOUT_MS/);
  assert.match(source, /await assertAdapterIdentity\(adapter, signal\)/);
  assert.match(source, /pre-launch q4/);
  assert.match(source, /post-run q4/);
  assert.match(source, /pre-launch text-encoder/);
  assert.match(source, /post-run text-encoder/);
  assert.match(source, /const resolutionEnv = Object\.fromEntries/);
  assert.match(source, /RUSTUP_TOOLCHAIN: channel/);
  assert.match(source, /CARGO_HOME: path\.join\(scratch, "cargo-home"\)/);
  assert.match(source, /CARGO_TARGET_DIR: path\.join\(scratch, "target"\)/);
  assert.match(source, /TMPDIR: privateTemp/);
  assert.doesNotMatch(source, /TMPDIR: process\.env\.TMPDIR/);
  assert.match(source, /Cargo inference source tree differs from the verified checkout/);
  assert.match(source, /response\?\.inferenceRevision !== expectedInferenceRevision/);
  assert.match(source, /--require-child-attestation/);
  assert.equal(CHILD_ATTESTATION_TIMEOUT_SECONDS, 30);
  assert.match(source, /--child-attestation-timeout", String\(CHILD_ATTESTATION_TIMEOUT_SECONDS\)/);
  assert.match(source, /event\.event === "child_attested"/);
  assert.match(source, /await chmod\(adapterDirectory, 0o500\)/);
  assert.match(source, /execFileAsync\("\/bin\/cp", \["-c", cloneSource, output\]/);
  assert.match(source, /await cleanupCanaryScratch\(scratch\)/);
  const failureRead = source.indexOf("const eventBytes = await readFile(eventsPath");
  const cleanup = source.indexOf("await cleanupCanaryScratch(scratch)");
  assert.ok(failureRead >= 0 && cleanup > failureRead,
    "watchdog failure reason must be read before scratch cleanup");
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
      assert.equal(toolchain.env.CARGO_HOME, path.join(scratch, "cargo-home"));
      assert.equal(toolchain.env.TMPDIR, path.join(scratch, "tmp"));
      assert.equal((await stat(toolchain.env.TMPDIR)).mode & 0o777, 0o700);
      assert.ok(path.isAbsolute(toolchain.cargo));
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
