import assert from "node:assert/strict";
import { mkdtemp, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { ownedSamplerFault, parseOwnedCounter, readOwnedReading, sharedDeviceProof } from "./yue2-windows-owned-gpu.mjs";

const luid = "luid_0x00000000_0x0001f78f";
const now = Date.parse("2026-10-04T20:00:10Z");
const source = {
  pid: 1234, parentPid: 4321, createdUtc: "2026-10-04T20:00:02Z",
  executablePath: "D:\\target\\release\\deps\\sceneworks_worker-deadbeef.exe",
  commandLine: "sceneworks_worker-deadbeef.exe yue2_memory_profile::capture_case --ignored --exact",
  counter: String.raw`\GPU Process Memory(*)\Dedicated Usage`,
  rows: [
    { instance: `pid_1234_${luid}_phys_0`, status: "0", cookedValue: 2_000_000_000 },
    { instance: `pid_1234_${luid}_phys_0`, status: "0", cookedValue: 3_000_000_000 },
  ],
};
const opts = { pid: 1234, cargoPid: 4321, cargoStartedAt: Date.parse("2026-10-04T20:00:00Z"), luid, now };
const files = Object.fromEntries(Array.from({ length: 29 }, (_, i) => [`raw-${i}.json`, "0".repeat(64)]));
const proof = { backend: "cuda", label: "before-strict-bf16-standard", admitted: true,
  census: JSON.stringify({ physicalMode: "shared-gpu1", admission: true, commandExit: 0,
    validatedDevice: { physicalMode: "shared-gpu1", physicalIndex: 1, cudaOrdinal: 0,
      uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", pci: "00000000:C1:00.0", luid },
    diagnosticFiles: files, diagnosticFileBytesB64: files }) };

test("owned counter sums only a live child on the selected LUID with valid status", () => {
  const parsed = parseOwnedCounter(source, opts);
  assert.equal(parsed.bytes, 5_000_000_000);
  assert.equal(parsed.rows, 2);
  assert.equal(parsed.identity.parentPid, 4321);
  assert.equal(parsed.identity.createdUtc, new Date(source.createdUtc).toISOString());
  assert.deepEqual(parseOwnedCounter(source, { ...opts, expectedIdentity: parsed.identity }), parsed);
});

test("owned counter refuses wrong PID, parent, LUID, status, creation, executable and missing rows", () => {
  for (const changed of [
    { ...source, pid: 999 },
    { ...source, parentPid: 999 },
    { ...source, createdUtc: "2026-10-04T19:00:00Z" },
    { ...source, executablePath: "D:\\target\\other.exe" },
    { ...source, commandLine: "unrelated-test.exe" },
    { ...source, rows: [{ ...source.rows[0], instance: "pid_1234_luid_0x00000000_0x00020b8c_phys_0" }] },
    { ...source, rows: [{ ...source.rows[0], status: "1" }] },
    { ...source, rows: [] },
  ]) assert.throws(() => parseOwnedCounter(changed, opts), /owned CUDA/);
  const parsed = parseOwnedCounter(source, opts);
  assert.throws(() => parseOwnedCounter(source, { ...opts, expectedIdentity: { ...parsed.identity, createdUtc: "2026-10-04T20:00:03.000Z" } }), /identity changed/);
});

test("shared device proof binds the selected physical GPU, logical ordinal and retained raw evidence", () => {
  const parsed = sharedDeviceProof(Buffer.from(JSON.stringify(proof)), 1);
  assert.equal(parsed.luid, luid);
  assert.equal(parsed.physicalIndex, 1);
  assert.match(parsed.sha256, /^[0-9a-f]{64}$/);
  assert.throws(() => sharedDeviceProof(Buffer.from(JSON.stringify(proof)), 0), /does not bind/);
  const wrong = structuredClone(proof);
  wrong.census = JSON.stringify({ ...JSON.parse(proof.census), diagnosticFileBytesB64: {} });
  assert.throws(() => sharedDeviceProof(Buffer.from(JSON.stringify(wrong)), 1), /does not bind/);
});

test("owned reading binds the native process file and stage request before attributing a counter", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-owned-counter-"));
  try {
    const shared = { marksFile: path.join(dir, "stages.jsonl"), processFile: path.join(dir, "profile-process.json"),
      boundaryDir: dir, proof: { luid } };
    await writeFile(shared.marksFile, `${JSON.stringify({ stage: "load", at: Date.now() / 1000 })}\n`);
    await writeFile(shared.processFile, JSON.stringify({ processId: 1234 }));
    const cargoStartedAt = Date.now() - 5000;
    const live = { ...source, createdUtc: new Date(Date.now() - 2000).toISOString() };
    const query = () => JSON.stringify(live);
    const reading = readOwnedReading(shared, { cargoPid: 4321, cargoStartedAt }, query);
    assert.equal(reading.pid, 1234);
    assert.equal(reading.bytes, 5_000_000_000);
    await writeFile(path.join(dir, "request.json"), JSON.stringify({ processId: 999 }));
    assert.throws(() => readOwnedReading(shared, { cargoPid: 4321, cargoStartedAt }, query), /stage request changed owned test PID/);
    await writeFile(path.join(dir, "request.json"), JSON.stringify({ processId: 1234 }));
    await unlink(shared.processFile);
    let missing;
    try {
      readOwnedReading(shared, { cargoPid: 4321, cargoStartedAt, expectedIdentity: reading.identity }, query);
    } catch (cause) { missing = cause; }
    assert.equal(missing?.code, "ENOENT", "the published native PID file disappeared mid-capture");
    assert.match(ownedSamplerFault(missing, true), /ENOENT/,
      "a missing file after verified identity must enter the durable fault journal");
    assert.equal(ownedSamplerFault(missing, false), null, "a process may not have published its first file yet");
    await writeFile(shared.marksFile, `${JSON.stringify({ stage: "done", at: Date.now() / 1000 })}\n`);
    assert.equal(readOwnedReading(shared, { cargoPid: 4321, cargoStartedAt }, () => { throw new Error("counter called after done"); }), null);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
