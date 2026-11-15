// Source-only checks for the SW-7103 app lifecycle: no model, API, worker or device is run.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { deflateSync } from "node:zlib";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { assertDiskBudget, chooseCudaCard, CUDA_LEASE_KEY, parseCudaCards, assertPlan, assertRendered, assertSelection, assertTrained, BUILD_HEADROOM_BYTES, CAPTIONS, decodePng, FAMILY, imageBody, LICENSE_URL, meanAbsoluteDifference, MODEL_ID, STATE_HEADROOM_BYTES, syntheticPng, TARGETS, tierRows, trainingBody, TRAINING } from "./minimax-h3-image-train-app-lifecycle.mjs";

const read = (name) => readFile(new URL(name, import.meta.url), "utf8");
const SHA = "a".repeat(40), PIN = "b".repeat(40);
const catalog = async () => JSON.parse(stripJsoncComments(await read("../config/manifests/builtin.models.jsonc")));
const core = async () => read("../crates/sceneworks-core/src/training.rs");

test("selection admits only the named backend on its own host and exact revisions", () => {
  const env = { H3I_APP_BACKEND: "mlx", H3I_APP_TIER: "q4", H3I_APP_SOURCE_SHA: SHA, H3I_APP_INFERENCE_SHA: PIN, GITHUB_SHA: SHA, RUNNER_NAME: "nax-macos-2" };
  assert.deepEqual(assertSelection(env, "darwin"), { backend: "mlx", tier: "q4" });
  assert.deepEqual(assertSelection({ ...env, H3I_APP_BACKEND: "cuda", RUNNER_NAME: "cuda-windows-2" }, "win32"), { backend: "cuda", tier: "q4" });
  for (const [patch, platform, message] of [
    [{ RUNNER_NAME: "nax-macos" }, "darwin", /not an admitted mlx host/],
    [{}, "win32", /mlx needs macOS/],
    [{ H3I_APP_BACKEND: "cuda", RUNNER_NAME: "cuda-windows-3" }, "win32", /not an admitted cuda host/],
    [{ H3I_APP_TIER: "fp8" }, "darwin", /bf16, q8 or q4/],
    [{ H3I_APP_SOURCE_SHA: "main" }, "darwin", /40-hex/],
    [{ GITHUB_SHA: PIN }, "darwin", /dispatched commit/],
  ]) assert.throws(() => assertSelection({ ...env, ...patch }, platform), message);
});

test("the shipped training targets admit the lifecycle's settings", async () => {
  const source = await core();
  for (const id of Object.values(TARGETS)) assert.match(source, new RegExp(`"${id}"`));
  assert.match(source, /"steps": \[200, 6000\]/, "200 steps is the targets' lower step limit");
  assert.match(source, /"resolutions": \[1024, 1536, MINIMAX_H3_IMAGE_TRAINING_RESOLUTION\]/);
  assert.match(source, /"baseTiers": \["auto", "bf16", "q8", "q4"\]/);
  assert.match(source, /MINIMAX_H3_IMAGE_LICENSE_URL: &str =\s+"https:\/\/huggingface.co\/MiniMaxAI\/MiniMax-H3-Image\/blob\/492fd8c4eca666f1085f57e30738c1e0d86b3439\/LICENSE"/);
  assert.equal(TRAINING.steps, 200); assert.equal(TRAINING.resolution, 1024);
});

const target = (mode) => ({ id: TARGETS[mode], family: FAMILY, baseModel: MODEL_ID, limits: { baseTiers: ["auto", "bf16", "q8", "q4"] }, ui: { requiresLicenseAcknowledgment: true, licenseUrl: LICENSE_URL }, defaults: { steps: 2000, resolution: 2048, rank: 16, alpha: 16, advanced: { networkType: "lora", baseTier: "auto", sampleEvery: 500 } } });

test("training bodies keep the target defaults and pin steps, canvas, rank and tier", () => {
  const body = trainingBody(target("edit"), "ds", "edit", "q4");
  assert.equal(body.targetId, TARGETS.edit); assert.equal(body.licenseAcknowledged, true); assert.equal(body.dryRun, false);
  assert.equal(body.config.steps, 200); assert.equal(body.config.resolution, 1024); assert.equal(body.config.rank, 8);
  assert.equal(body.config.advanced.baseTier, "q4"); assert.equal(body.config.advanced.sampleEvery, 0); assert.equal(body.config.advanced.gradientCheckpointing, true);
  assert.throws(() => trainingBody(target("t2i"), "ds", "edit", "q4"), /exact minimax_h3_image_edit_lora/);
  assert.throws(() => trainingBody({ ...target("t2i"), ui: { requiresLicenseAcknowledgment: false } }, "ds", "t2i", "q4"), /licence gate/);
  assert.throws(() => trainingBody({ ...target("t2i"), limits: { baseTiers: ["bf16"] } }, "ds", "t2i", "q4"), /q4 base tier/);
  const plan = { target: { targetId: body.targetId, kernel: body.targetId, baseModel: MODEL_ID }, config: { ...body.config }, dataset: { datasetId: "ds", items: [{ referenceImagePaths: ["a.png"] }] } };
  assert.equal(assertPlan({ type: "lora_train", payload: { dryRun: false, plan } }, body, "edit"), plan);
  assert.throws(() => assertPlan({ type: "lora_train", payload: { dryRun: false, plan: { ...plan, dataset: { datasetId: "ds", items: [{}] } } } }, body, "edit"), /ordered reference/);
});

test("a trained adapter passes only when registered in the library under the H3-Image family", () => {
  const plan = { target: { targetId: TARGETS.t2i }, output: { outputDir: "/o", fileName: "a.safetensors" } };
  const job = { id: "j", status: "completed", workerId: "w", error: null, payload: { plan }, result: { backend: "mlx", targetId: TARGETS.t2i, networkType: "lora", stepsCompleted: 200, finalLoss: 0.1, loraId: "l", fileName: "a.safetensors", outputPath: "/o/a.safetensors" } };
  const library = { id: "l", scope: "project", installState: "installed", source: { provider: "training" }, files: ["a.safetensors"], installedPath: "/o", family: FAMILY, baseModel: MODEL_ID, networkType: "lora", licenseUrl: LICENSE_URL, provenance: { trainingJobId: "j", targetId: TARGETS.t2i } };
  const metadata = { family: FAMILY, rank: "8" };
  assert.equal(assertTrained(job, library, metadata, "t2i", "mlx", "w").lora_id, "l");
  assert.throws(() => assertTrained(job, { ...library, family: "minimax-h3" }, metadata, "t2i", "mlx", "w"), /H3-Image family/);
  assert.throws(() => assertTrained(job, library, metadata, "t2i", "cuda", "w"), /real cuda trainer/);
  assert.throws(() => assertTrained(job, undefined, metadata, "t2i", "mlx", "w"), /registered in the project LoRA library/);
  assert.throws(() => assertTrained(job, library, metadata, "edit", "mlx", "w"), /training mode/);
});

test("image bodies follow the Image Studio request shape and the saved recipe must name the LoRA", () => {
  const withLora = imageBody("p", "edit", "q4", "l", "ref");
  assert.deepEqual(withLora, { projectId: "p", mode: "edit_image", model: MODEL_ID, prompt: CAPTIONS.edit, negativePrompt: "", seed: 7103, width: 1024, height: 1024, count: 1, advanced: { mlxQuantize: 4 }, loras: [{ id: "l", weight: 1 }], referenceAssetIds: ["ref"] });
  assert.equal(imageBody("p", "t2i", "bf16", null).advanced.mlxQuantize, 0);
  assert.equal(imageBody("p", "t2i", "q8", null).referenceAssetIds, undefined);
  const job = { id: "j", status: "completed", workerId: "w", error: null, result: { model: MODEL_ID, assetIds: ["a"] } };
  const asset = { id: "a", type: "image", file: { path: "x.png" }, recipe: { model: MODEL_ID, mode: "edit_image", seed: 7103, prompt: CAPTIONS.edit, loras: [{ id: "l", weight: 1 }] } };
  assert.deepEqual(assertRendered(job, asset, withLora, "mlx", "w").loras, ["l"]);
  assert.throws(() => assertRendered(job, { ...asset, recipe: { ...asset.recipe, loras: [] } }, withLora, "mlx", "w"), /exactly the applied LoRAs/);
});

test("PNG decoding recovers every filter type and the difference metric", () => {
  const png = syntheticPng(false, 64), same = syntheticPng(false, 64), other = syntheticPng(true, 64);
  const decoded = decodePng(png);
  assert.equal(decoded.width, 64); assert.deepEqual([...decoded.pixels.subarray(0, 3)], [245, 235, 180]);
  assert.equal(meanAbsoluteDifference(png, same), 0);
  assert.ok(meanAbsoluteDifference(png, other) > 0);
  // Re-encode the same rows with Sub/Up/Average/Paeth filters: decoding must give identical pixels.
  const { pixels } = decoded, stride = 64 * 3;
  for (const filter of [1, 2, 3, 4]) {
    const rows = Buffer.alloc((stride + 1) * 64);
    for (let y = 0; y < 64; y++) {
      rows[y * (stride + 1)] = filter;
      for (let x = 0; x < stride; x++) {
        const a = x >= 3 ? pixels[y * stride + x - 3] : 0, b = y ? pixels[(y - 1) * stride + x] : 0, c = x >= 3 && y ? pixels[(y - 1) * stride + x - 3] : 0;
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        const predictor = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
        rows[y * (stride + 1) + 1 + x] = (pixels[y * stride + x] - predictor) & 0xff;
      }
    }
    const idat = deflateSync(rows), length = Buffer.alloc(4); length.writeUInt32BE(idat.length);
    const start = png.indexOf("IDAT") - 4, end = start + 12 + png.readUInt32BE(start);
    const reencoded = Buffer.concat([png.subarray(0, start), length, Buffer.from("IDAT"), idat, Buffer.alloc(4), png.subarray(end)]);
    assert.deepEqual(decodePng(reencoded).pixels, pixels, `filter ${filter}`);
  }
});

test("the disk budget counts the tier's missing rows plus build and state headroom", async () => {
  const entry = (await catalog()).models.find((model) => model.id === MODEL_ID);
  const rows = tierRows(entry, "q4", "darwin");
  assert.ok(rows.some((row) => row.variant === "q4" && !row.coRequisite && row.repo === "SceneWorks/minimax-h3-image-mlx"));
  assert.ok(rows.some((row) => row.componentId === "text_encoder" && row.variant === "q4"));
  assert.ok(rows.some((row) => row.componentId === "image_vae"));
  assert.ok(!rows.some((row) => row.variant && row.variant !== "q4"), "no other tier's rows");
  const windows = tierRows(entry, "bf16", "win32").filter((row) => row.componentId === "text_encoder");
  assert.deepEqual(windows.map((row) => row.files), [["text_encoder/*"]], "Windows takes the whole bf16 text encoder row");
  const need = BUILD_HEADROOM_BYTES + STATE_HEADROOM_BYTES + 5;
  assert.equal(assertDiskBudget({ freeBytes: need, missingBytes: 5, includeBuild: true }).required_bytes, need);
  assert.throws(() => assertDiskBudget({ freeBytes: need - 1, missingBytes: 5, includeBuild: true }), /insufficient free disk/);
});

test("the workflow is reachable only from the manual dispatcher and pins each backend's host", async () => {
  const workflow = await read("../.github/workflows/minimax-h3-image-train-app-lifecycle.yml");
  const triggers = workflow.slice(workflow.indexOf("\non:"), workflow.indexOf("\npermissions:"));
  assert.match(triggers, /workflow_call:/);
  assert.doesNotMatch(triggers, /\n  (?:push|pull_request|schedule|merge_group|workflow_dispatch):/);
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-mage\]/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.doesNotMatch(workflow, /rw-starvector|\bnax\]/, "never Michael's dev Mac");
  assert.equal((workflow.match(/node scripts\/minimax-h3-image-train-app-lifecycle\.mjs --preflight-only/g) ?? []).length, 2, "both backends refuse a wrong selection before build");
  const dispatcher = await read("../.github/workflows/memory-catalog-campaign.yml");
  assert.match(dispatcher, /\n {10}- minimax-h3-image-train-app-lifecycle\n/);
  assert.match(dispatcher, /minimax-h3-image-train-app-lifecycle:\n {4}if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'minimax-h3-image-train-app-lifecycle' \}\}/);
  assert.match(dispatcher, /uses: \.\/\.github\/workflows\/minimax-h3-image-train-app-lifecycle\.yml\n {4}with:\n {6}source_sha: \$\{\{ inputs\.ref \}\}/);
  assert.match(dispatcher, /backend: \$\{\{ inputs\.backend == 'candle' && 'cuda' \|\| 'mlx' \}\}/);
});

test("the CUDA lifecycle runs only as the child of the exact-UUID lease on the card chosen before the build", async () => {
  const workflow = await read("../.github/workflows/minimax-h3-image-train-app-lifecycle.yml");
  const cuda = workflow.slice(workflow.indexOf("\n  cuda:"));
  const select = cuda.indexOf("--select-cuda-card"), build = cuda.indexOf("--build-only"), lease = cuda.indexOf("windows_cuda_execution.py");
  assert.ok(select > 0 && select < build && build < lease, "choose the card, build without a lease, then run under it");
  assert.match(cuda, /python -I scripts\/ci\/windows_cuda_execution\.py --actor minimax-h3-image-train-app --budget-minutes [0-9]+ --physical-indices %H3I_APP_CUDA_INDEX% -- node scripts\\minimax-h3-image-train-app-lifecycle\.mjs\n/);
  assert.doesNotMatch(cuda, /functional-coexistence/, "strict-idle admission only");
  assert.equal((cuda.match(/node scripts[\\/]minimax-h3-image-train-app-lifecycle\.mjs(?:\n|$)/g) ?? []).length, 1, "the lifecycle itself is launched once, under the lease");
  const mlx = workflow.slice(workflow.indexOf("\n  mlx:"), workflow.indexOf("\n  cuda:"));
  assert.ok(mlx.indexOf("--build-only") > 0 && mlx.indexOf("--build-only") < mlx.lastIndexOf("node scripts/minimax-h3-image-train-app-lifecycle.mjs\n"));
  const script = await read("./minimax-h3-image-train-app-lifecycle.mjs");
  assert.equal(CUDA_LEASE_KEY, "SW_CI_CUDA_EXECUTION_ACTIVE");
  assert.match(await read("./ci/windows_cuda_execution.py"), /ACTIVE_KEY = "SW_CI_CUDA_EXECUTION_ACTIVE"/);
  assert.match(script, /if \(phase === "run"\) fact\(env\[CUDA_LEASE_KEY\]/);
  assert.match(script, /assertTrained\(completed, library, metadata, mode, receipt\.worker\.gpuId, workerId\)/, "results carry backend_label(gpu_id), not the lane name");
});

test("the leased card is the largest by total memory, lowest index on a tie, never by free memory", () => {
  const cards = parseCudaCards("0, GPU-aaaa-1, RTX A, 49140\n1, GPU-bbbb-2, RTX B, 97887\n2, GPU-cccc-3, RTX C, 97887\n");
  assert.equal(chooseCudaCard(cards).index, "1");
  assert.throws(() => parseCudaCards(""), /no usable CUDA card/);
  assert.throws(() => parseCudaCards("0, not-a-uuid, X, 100"), /no usable CUDA card/);
});
