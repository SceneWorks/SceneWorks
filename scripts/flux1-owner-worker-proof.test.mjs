import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { sha256 } from "./lib/file-sha256.mjs";
import { assertDiskBudget, assertOwnerTrace, assertSerializedTrajectory, imageBody, parseOwnerTrace, verifyArtifact } from "./flux1-owner-worker-proof.mjs";

const sha = "a".repeat(40);
const worker = "owned-mlx-worker";
const entry = (seed, kind, name) => ({ seed, job_id: `${kind}-${seed}`, worker_id: worker, backend: "mlx", asset_id: `${kind}-${seed}-asset`, generation_set_id: `${kind}-${seed}-set`, file: name, sha256: sha256(Buffer.from(name)), completed_at_ms: seed === 33111 ? 100 : 300 });

test("owner proof compares each queued native output with its isolated seed and rejects overlap", () => {
  const isolated = [entry(33111, "alone", "a.png"), entry(33122, "alone", "b.png")];
  const queued = [entry(33111, "queued", "a.png"), entry(33122, "queued", "b.png")];
  const samples = [{ "queued-33111": "running", "queued-33122": "queued" }];
  assert.doesNotThrow(() => assertSerializedTrajectory(isolated, queued, samples));
  assert.throws(() => assertSerializedTrajectory(isolated, [{ ...queued[0], sha256: "0".repeat(64) }, queued[1]], samples), /changed/);
  assert.throws(() => assertSerializedTrajectory(isolated, queued, []), /remain queued/);
});

test("image request keeps declared model, global seed, geometry and sampler", () => {
  const body = imageBody("project", { id: "flux_dev", guidance: 3.5 }, 33111);
  assert.equal(body.model, "flux_dev"); assert.equal(body.seed, 33111);
  assert.deepEqual(body.advanced, { steps: 4, sampler: "euler", guidanceScale: 3.5 });
  assert.equal(body.count, 1);
});

test("disk preflight accounts for remaining pinned sources and build state", () => {
  const gib = 1024 ** 3;
  assert.doesNotThrow(() => assertDiskBudget({ sameVolume: true, homeFree: 70 * gib, outputFree: 70 * gib, remainingWeights: 20 * gib }));
  assert.throws(() => assertDiskBudget({ sameVolume: true, homeFree: 60 * gib, outputFree: 60 * gib, remainingWeights: 20 * gib }), /insufficient/);
  assert.throws(() => assertDiskBudget({ sameVolume: false, homeFree: 20 * gib, outputFree: 40 * gib, remainingWeights: 20 * gib }), /output volume/);
});

test("retained artifact verifier detects changed image bytes and missing provider", async () => {
  const output = await mkdtemp(path.join(os.tmpdir(), "flux1-owner-artifact-"));
  const evidence = path.join(output, "evidence"); await mkdir(evidence);
  const models = [];
  for (const id of ["flux_dev", "flux_schnell"]) {
    const isolated = [], queued = [];
    for (const seed of [33111, 33122]) for (const kind of ["isolated", "queued"]) {
      const file = `${id}-${kind}-${seed}.png`, bytes = `${id}-${seed}`;
      await writeFile(path.join(evidence, file), bytes);
      (kind === "isolated" ? isolated : queued).push({ ...entry(seed, kind, file), job_id: `${id}-${kind}-${seed}`, sha256: sha256(Buffer.from(bytes)) });
    }
    const activeFile = `${id}-active-33133.png`, activeBytes = `${id}-33133`;
    await writeFile(path.join(evidence, activeFile), activeBytes);
    const active = { ...entry(33133, "active", activeFile), job_id: `${id}-active`, sha256: sha256(Buffer.from(activeBytes)) };
    const recoveryFile = `${id}-recovery-33166.png`, recoveryBytes = `${id}-33166`;
    await writeFile(path.join(evidence, recoveryFile), recoveryBytes);
    const recovery = { ...entry(33166, "recovery", recoveryFile), job_id: `${id}-recovery`, sha256: sha256(Buffer.from(recoveryBytes)) };
    models.push({ id, repo: id === "flux_dev" ? "SceneWorks/flux1-dev-mlx" : "SceneWorks/flux1-schnell-mlx", revision: id === "flux_dev" ? "323fd12d79f78ad444e882e8d8e871914584f2b9" : "bba3ae01dfd94089f173c05edd4e1a4c551f2599", variant: "q4", isolated, queued, observations: [{ [`${id}-queued-33111`]: "running", [`${id}-queued-33122`]: "queued" }], active, cancel: { status: "canceled", job_id: `${id}-cancel` }, cancel_active: { status: "canceled", job_id: `${id}-cancel-active` }, recovery });
  }
  const ownerTrace = []; let sequence = 0;
  for (const model of models) {
    for (const entry of [...model.isolated, ...model.queued, model.active]) for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_end"]) ownerTrace.push({ job_id: entry.job_id, engine_id: model.id === "flux_dev" ? "flux1_dev" : "flux1_schnell", phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
    for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_failed"]) ownerTrace.push({ job_id: model.cancel_active.job_id, engine_id: model.id === "flux_dev" ? "flux1_dev" : "flux1_schnell", phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
    for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_end"]) ownerTrace.push({ job_id: model.recovery.job_id, engine_id: model.id === "flux_dev" ? "flux1_dev" : "flux1_schnell", phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
  }
  const receipt = { schema_version: 1, kind: "flux1_native_owner_worker_proof", status: "passed", identity: { source_sha: sha, inference_sha: sha, dispatch_sha: sha, binary_sha256: "b".repeat(64), flux_dev_model_index_sha256: "c".repeat(64), flux_schnell_model_index_sha256: "d".repeat(64) }, worker: { id: worker, gpuId: "mlx", capabilities: ["gpu"] }, models, owner_trace: ownerTrace, cleanup: [{ exited: true }, { exited: true }] };
  const file = path.join(evidence, "receipt.json"); await writeFile(file, JSON.stringify(receipt));
  await verifyArtifact(output);
  await writeFile(path.join(evidence, models[0].queued[0].file), "changed");
  await assert.rejects(() => verifyArtifact(output), /digest mismatch/);
  await writeFile(path.join(evidence, models[0].queued[0].file), `${models[0].id}-${models[0].queued[0].seed}`);
  const changed = JSON.parse(await readFile(file, "utf8")); changed.models.pop(); await writeFile(file, JSON.stringify(changed));
  await assert.rejects(() => verifyArtifact(output), /both FLUX/);
});

test("native owner trace parser and verifier reject cross-thread or interleaved work", () => {
  const model = { id: "flux_dev", isolated: [{ job_id: "a" }, { job_id: "b" }], queued: [{ job_id: "c" }, { job_id: "d" }], active: { job_id: "e" }, cancel: { job_id: "f" }, cancel_active: { job_id: "m" }, recovery: { job_id: "n" } };
  const other = { ...model, id: "flux_schnell", isolated: [{ job_id: "g" }, { job_id: "h" }], queued: [{ job_id: "i" }, { job_id: "j" }], active: { job_id: "k" }, cancel: { job_id: "l" }, cancel_active: { job_id: "o" }, recovery: { job_id: "p" } };
  const trace = []; let sequence = 0;
  for (const [row, engine] of [[model, "flux1_dev"], [other, "flux1_schnell"]]) {
    for (const entry of [...row.isolated, ...row.queued, row.active]) for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_end"]) trace.push({ job_id: entry.job_id, engine_id: engine, phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
    for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_failed"]) trace.push({ job_id: row.cancel_active.job_id, engine_id: engine, phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
    for (const phase of ["prepare_begin", "prepare_end", "drive_begin", "drive_end"]) trace.push({ job_id: row.recovery.job_id, engine_id: engine, phase, owner_thread: "ThreadId(7)", sequence: ++sequence });
  }
  const log = trace.map((fields) => JSON.stringify({ level: "INFO", fields: { event: "flux1_owner_turn", ...fields } })).join("\n");
  assert.deepEqual(parseOwnerTrace(log), trace);
  assert.doesNotThrow(() => assertOwnerTrace([model, other], trace));
  assert.throws(() => assertOwnerTrace([model, other], trace.map((event, index) => index ? event : { ...event, owner_thread: "ThreadId(8)" })), /single owning thread/);
  const interleaved = trace.map((event) => event.job_id === "d" ? { ...event, sequence: event.sequence - 8 } : event);
  assert.throws(() => assertOwnerTrace([model, other], interleaved), /sequence|interleaved|order/);
});

test("proof workflow is manual and retains evidence only", async () => {
  const workflow = await readFile(new URL("../.github/workflows/flux1-owner-worker-proof.yml", import.meta.url), "utf8");
  const dispatcher = await readFile(new URL("../.github/workflows/memory-catalog-campaign.yml", import.meta.url), "utf8");
  assert.match(workflow, /workflow_call:/);
  assert.doesNotMatch(workflow, /^\s+(push|pull_request|workflow_dispatch):/m);
  assert.match(dispatcher, /workflow_dispatch:/);
  assert.match(dispatcher, /inputs\.profile == 'flux1-owner-worker-proof'/);
  assert.match(dispatcher, /uses: \.\/\.github\/workflows\/flux1-owner-worker-proof\.yml/);
  assert.match(workflow, /--verify-artifact/);
  assert.match(workflow, /\/evidence\//);
});
