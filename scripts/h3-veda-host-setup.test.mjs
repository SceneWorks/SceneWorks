import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm, symlink, cp, stat, realpath } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ROOT, hashFile, digest, validateShape, catalogs, matrix } from "./h3-veda-acceptance.mjs";
import { assertSetupSource, physicalMetadata, declaredCacheRoots, cacheMetadata, inventory, stage } from "./h3-veda-host-setup.mjs";
const H = "a".repeat(40), I = "c".repeat(40);
const templateFile = "config/terminal-evidence/epic-25140-host-template.json";
const planFile = "config/terminal-evidence/epic-25140-corpus.plan.json";
const bundleFile = "config/terminal-evidence/epic-25140-carriers.json";
const json = async (p) => JSON.parse(await readFile(p, "utf8"));
async function temp(t) { const p = await realpath(await mkdtemp(path.join(os.tmpdir(), "s9-setup-"))); t.after(() => rm(p, { recursive: true, force: true })); return p; }
const physical = { physicalGpuId: "A299B954-4C63-5322-86A5-BD5667051634", metalDeviceName: "Apple Test CPU Fixture", admission: false };
async function observation(t, root = ROOT) {
  const scratch = await temp(t), inferenceRoot = path.join(scratch, "inference");
  for (const [file, bytes] of [["scripts/release/veda_cuda_guard.py", "# CPU source fixture"], ["scripts/media_campaign_windows.py", "# CPU source fixture"], ["crates/media/mlx-gen/mlx-gen-minimax-h3/src/veda/mod.rs", 'pub const KERNEL: &str = "fixture-mlx";'], ["crates/kernels/candle-quant-kernels/src/veda_int8.rs", 'name: "fixture-cuda",']]) { const p = path.join(inferenceRoot, file); await mkdir(path.dirname(p), { recursive: true }); await writeFile(p, bytes); }
  return { schemaVersion: 1, kind: "epic25140_cpu_host_inventory", observedAt: Date.now(), backend: "mlx", runnerName: "nax-macos-2", source: { sceneworks: { root, head: H }, inference: { root: inferenceRoot, head: I } }, physical, cache: [], ownedStageRoot: path.join(scratch, "owned"), templateSha256: await hashFile(path.join(root, templateFile)), planSha256: await hashFile(path.join(root, planFile)), campaignReady: false, metadataOnly: true };
}
const sourceCheck = () => {};

test("inventory accepts exact current feature source without inventing permanent-main status", () => {
  const calls = [], exec = (a) => { calls.push(a); return a.includes("status") ? "" : a.includes("/native") ? I : H; };
  const value = assertSetupSource({ backend: "mlx", runnerName: "nax-macos-2", expectedRunner: "nax-macos-2", sceneWorksSha: H, inferenceSha: I, root: "/sw", inferenceRoot: "/native", exec });
  assert.equal(value.inference.head, I); assert.equal(calls.some((c) => c.includes("refs/remotes/origin/main")), false);
  assert.throws(() => assertSetupSource({ backend: "mlx", runnerName: "nax-macos", expectedRunner: "nax-macos-2", sceneWorksSha: H, inferenceSha: I, exec }), /named physical runner/);
  assert.throws(() => assertSetupSource({ backend: "mlx", runnerName: "nax-macos-2", expectedRunner: "nax-macos-2", sceneWorksSha: H, inferenceSha: I, exec: () => "dirty" }), /dirty\/stale/);
});
test("Mac inventory extracts UUID and Metal model, rejects primary Mac and ambiguous devices", () => {
  const probe = (uuid, devices) => physicalMetadata("mlx", { platform: "darwin", exec: (a) => JSON.stringify(a.includes("SPHardwareDataType") ? { SPHardwareDataType: [{ platform_UUID: uuid, serial_number: "DO_NOT_EXPORT" }] } : { SPDisplaysDataType: devices }) });
  const devices = [{ sppci_model: "Apple M Test", spdisplays_mtlgpufamilysupport: "Metal 3" }];
  assert.equal(probe(physical.physicalGpuId, devices).metalDeviceName, "Apple M Test"); assert.doesNotMatch(JSON.stringify(probe(physical.physicalGpuId, devices)), /DO_NOT_EXPORT/);
  assert.throws(() => probe("D299B954-4C63-5322-86A5-BD5667051634", devices), /differ/); assert.throws(() => probe(physical.physicalGpuId, [...devices, ...devices]), /ambiguous/);
});
test("Windows inventory reports every actual GPU and never selects an invented UUID", () => {
  const value = physicalMetadata("candle_cuda", { platform: "win32", exec: () => "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, Test GPU, 999.1, 24576\n1, GPU-bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee, Other GPU, 999.1, 24576" });
  assert.equal(value.devices.length, 2); assert.equal(value.admission, false);
  assert.throws(() => physicalMetadata("candle_cuda", { platform: "win32", exec: () => "0, invented, GPU, 1, 10" }), /malformed/);
});
test("cache inventory is finite metadata-only and rejects a snapshot symlink escaping the declared root", async (t) => {
  const root = await temp(t), cache = path.join(root, "cache"), a = { id: "weight", repo: "author/model", revision: H, file: "weight.bin", bytes: 4, sha256: digest("data"), coverage: [{ backend: "mlx" }] };
  const snapshot = path.join(cache, "hub/models--author--model/snapshots", H); await mkdir(snapshot, { recursive: true }); const outside = path.join(root, "private.bin"); await writeFile(outside, "data"); await symlink(outside, path.join(snapshot, "weight.bin"));
  const v = await cacheMetadata({ artifacts: [a], turboArtifacts: {}, styles: [] }, "mlx", [cache]); assert.equal(v[0].files[0].confined, false); assert.equal(v[0].files[0].observedBytes, null); assert.equal(v[0].contentVerified, false);
  assert.deepEqual(declaredCacheRoots({ cacheRootCandidates: { mlx: ["${HOME}/only-approved", "${MISSING}/unknown"] } }, "mlx", { HOME: root }), [path.join(root, "only-approved")]);
});
test("inventory runs without a manifest, cache, corpus, binary or future permanent pin", async (t) => {
  const scratch = await temp(t);
  const v = await inventory({ backend: "mlx", runnerName: "nax-macos-2", expectedRunner: "nax-macos-2", sceneWorksSha: H, inferenceSha: I, env: { HOME: scratch }, sourceCheck: () => ({ sceneworks: { root: ROOT, head: H }, inference: { root: path.join(scratch, "native"), head: I } }), authority: () => ({ readable: true, runnerName: "nax-macos-2" }), probe: () => physical });
  assert.equal(v.metadataOnly, true); assert.equal(v.campaignReady, false); assert.equal(v.physicalAdmission, false); assert.equal(v.cache.some((c) => c.contentVerified), false);
  await assert.rejects(inventory({ backend: "mlx", runnerName: "nax-macos-2", env: { HOME: scratch }, sourceCheck: () => ({}), authority: () => ({ readable: false }), probe: () => physical }), /ownership/);
});
test("source template retains all axes, both styles, all required artifact slots and explicit refusals", async () => {
  const t = await json(path.join(ROOT, templateFile)), c = await catalogs(ROOT), rows = matrix(c, t.styles);
  assert.equal(rows.filter((r) => !r.style).length, 5400); assert.equal(rows.length, 11016); assert.equal(rows.filter((r) => r.kind === "expected_refusal").length, 1512); assert.equal(t.styles.length, 2);
  for (const backend of ["mlx", "candle_cuda"]) for (const tier of ["q4", "q8", "bf16"]) for (const model of ["minimax_h3", "minimax_h3_ref"]) for (const component of ["dit", "text_encoder", "video_vae", "audio_vae", "predictor"]) assert.ok(t.artifacts.some((a) => a.coverage.some((v) => v.backend === backend && v.tier === tier && v.model === model && v.component === component)), `${backend}/${tier}/${model}/${component}`);
});
test("stage copies every exact source byte and keeps missing counterpart evidence unready", async (t) => {
  const o = await observation(t), r = await stage({ observation: o, sourceCheck });
  assert.equal(r.stagedCarriers, 33); assert.equal(r.stagedBytes, 901110); assert.equal(r.styleInclusiveCells, 11016); assert.equal(r.campaignReady, false); assert.equal(r.productImported, false);
  const m = await json(r.manifestPath), plan = await json(path.join(ROOT, planFile));
  for (const [key, a] of Object.entries(plan.assets)) { assert.equal(await hashFile(m.assets[key].path), a.sha256); assert.equal((await stat(m.assets[key].path)).size, a.bytes); assert.equal(m.assets[key].assetId, "UNBOUND"); }
  assert.equal(m.runners.mlx.physicalGpuId, physical.physicalGpuId); assert.equal(m.runners.candle_cuda.physicalGpuId, null); assert.equal(m.runners.candle_cuda.workerBinary, null); assert.equal(m.sourceFreeze.stage, "preparation-unbound");
  assert.equal(m.preparation.missingBindings.productImports, true, "only actual owned API imports can clear this binding");
  assert.throws(() => validateShape(m, { models: [] }), /permanent inference/);
});
test("stage rejects stale inventory, changed source and overwrite before modifying the target", async (t) => {
  const o = await observation(t);
  await assert.rejects(stage({ observation: { ...o, observedAt: Date.now() - 60001 }, sourceCheck }), /fresh/);
  await assert.rejects(stage({ observation: o, sourceCheck: () => { throw new Error("source dirty/stale"); } }), /dirty\/stale/);
  await mkdir(o.ownedStageRoot); await writeFile(path.join(o.ownedStageRoot, "foreign"), "keep"); await assert.rejects(stage({ observation: o, sourceCheck }), /exists/); assert.equal(await readFile(path.join(o.ownedStageRoot, "foreign"), "utf8"), "keep");
});
test("stage refuses unobserved caches and source-tree output without writes", async (t) => {
  const o = await observation(t); await assert.rejects(stage({ observation: o, cacheRoot: "/unobserved", sourceCheck }), /allowlist/); await assert.rejects(stage({ observation: o, runDir: path.join(ROOT, "not-allowed"), sourceCheck }), /outside/);
});
test("corrupt source bundle cannot materialize a single carrier", async (t) => {
  const scratch = await temp(t), root = path.join(scratch, "sw"); await mkdir(path.join(root, "config/terminal-evidence"), { recursive: true }); await mkdir(path.join(root, "config/manifests"));
  for (const f of [templateFile, planFile, bundleFile, "config/manifests/builtin.models.jsonc"]) await cp(path.join(ROOT, f), path.join(root, f));
  const o = await observation(t, root), bundle = await json(path.join(root, bundleFile)); bundle.assets[Object.keys(bundle.assets)[0]] = Buffer.from("wrong").toString("base64"); await writeFile(path.join(root, bundleFile), JSON.stringify(bundle));
  const template = await json(path.join(root, templateFile)); template.carrierBundleSha256 = await hashFile(path.join(root, bundleFile)); await writeFile(path.join(root, templateFile), JSON.stringify(template)); o.templateSha256 = await hashFile(path.join(root, templateFile));
  await assert.rejects(stage({ observation: o, root, sourceCheck }), /source carrier differs/); await assert.rejects(stat(o.ownedStageRoot), /ENOENT/);
});
test("inventory and stage workflow branches cannot reach builds, tokens, imports or GPU campaign", async () => {
  const w = await readFile(path.join(ROOT, ".github/workflows/h3-veda-terminal.yml"), "utf8"); assert.match(w, /options: \[inventory, stage, build, prepare, resume, review\]/); assert.equal((w.match(/CPU-only host inventory or synthetic corpus staging/g) ?? []).length, 2);
  for (const [step, condition] of [
    ["Generate masked job-owned private API token", "inputs.mode == 'prepare' || inputs.mode == 'resume'"],
    ["Build durable normal product API", "inputs.mode == 'build'"],
    ["Build durable normal Candle/CUDA", "inputs.mode == 'build'"],
    ["Prepare or verify owned private corpus", "inputs.mode == 'prepare' || inputs.mode == 'resume'"],
    ["Run bounded guardian-owned native campaign", "inputs.mode == 'resume'"],
  ]) {
    const blocks = w.split("      - name: ").filter((b) => b.startsWith(step)); assert.ok(blocks.length > 0); for (const b of blocks) assert.equal(b.split("\n")[1].trim(), `if: ${condition}`);
  }
  for (const field of ["manifest_path", "corpus_root", "offline_cache_root", "run_dir"]) assert.match(w.slice(w.indexOf(`      ${field}:`), w.indexOf(`      ${field}:`) + 330), /required: false/);
});
