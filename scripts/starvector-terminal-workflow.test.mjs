import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { validateCorpusAssets, validateTerminalServiceClosure } from "./starvector-terminal-readiness.mjs";
import { productServiceBackendEnv, productServiceBuildArgs, productServiceStateRoot } from "./starvector-terminal-product-service.mjs";

const workflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");
const readiness = await readFile(".github/workflows/starvector-terminal-readiness.yml", "utf8");
const hash = (value) => createHash("sha256").update(value).digest("hex");
const pin = "c6eb6d8e9545193eac844f6fea2db79e4d14bf2a";
const execFile = promisify(execFileCallback);

test("terminal workflow is dispatch-only, serial, and seals raw evidence", () => {
  assert.match(workflow, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(workflow, /^\s+(push|pull_request|schedule):/m);
  for (const edge of ["needs: mlx-1b", "needs: mlx-8b", "needs: cuda-1b"]) assert.match(workflow, new RegExp(edge));
  assert.match(workflow, /needs: \[mlx-1b, mlx-8b, cuda-1b, cuda-8b\]/);
  assert.match(workflow, /starvector-terminal-producer\.mjs run/g);
  assert.match(workflow, /starvector-terminal-producer\.mjs seal/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: \/Users\/Shared\/SceneWorks\/terminal-leases/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: C:\\\\ProgramData\\\\SceneWorks\\\\terminal-leases/);
  assert.match(workflow, /scripts\/starvector-terminal-route\.mjs/);
  assert.equal((workflow.match(/starvector-terminal-product-service\.mjs start/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-product-service\.mjs stop/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-case-bundle\.mjs/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-assets\.mjs/g) ?? []).length, 4);
  assert.match(workflow, /starvector_terminal_lease/);
  assert.equal((workflow.match(/cargo build --release --locked -p sceneworks-worker --bin starvector_terminal_lease/g) ?? []).length, 4);
  assert.doesNotMatch(workflow, /RUNNER_TEMP[^\n]*\.lease/);
  assert.match(workflow, /Upload combined evidence even on failure/);
  assert.equal((workflow.match(/timeout-minutes: 720/g) ?? []).length, 4);
});

test("terminal workflow has no install or model download step", () => {
  assert.doesNotMatch(workflow, /(?:pip|npm|cargo)\s+install|huggingface-cli|curl .*models|wget .*models/i);
  assert.match(workflow, /STARVECTOR_TERMINAL_WEIGHTS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_METRICS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_METRICS_PYTHON/);
  assert.match(workflow, /STARVECTOR_TERMINAL_CASE_BUNDLE/);
  assert.match(workflow, /STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS: "1"/);
  assert.match(workflow, /cross-repository lease/);
  assert.match(workflow, /starvector-terminal-metrics-environment-v1\.json/);
});

test("source-built product service enables the native backend for each campaign host", () => {
  assert.deepEqual(productServiceBuildArgs("darwin"), ["build", "--locked", "-p", "sceneworks-rust-api"]);
  assert.deepEqual(productServiceBackendEnv("darwin"), {});
  assert.deepEqual(productServiceBuildArgs("win32"), ["build", "--locked", "-p", "sceneworks-rust-api", "--features", "backend-candle"]);
  assert.deepEqual(productServiceBackendEnv("win32"), { SCENEWORKS_BACKEND_CANDLE_ENABLED: "true" });
  assert.equal(productServiceStateRoot(path.join("tmp", "tuple")), path.join("tmp", "tuple-product-service-state"));
  assert.notEqual(productServiceStateRoot(path.join("tmp", "tuple")), path.join("tmp", "tuple", "product-service-state"));
});

test("readiness workflow is an identity-only dispatch on both campaign hosts", () => {
  assert.match(readiness, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(readiness, /^\s+(push|pull_request|schedule):/m);
  assert.match(readiness, /runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);
  assert.equal((workflow.match(/runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/g) ?? []).length, 3);
  assert.match(readiness, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.equal((readiness.match(/starvector-terminal-readiness\.mjs/g) ?? []).length, 2);
  assert.equal((readiness.match(/if: \$\{\{ always\(\) \}\}/g) ?? []).length, 2);
  assert.match(readiness, /STARVECTOR_TERMINAL_INFERENCE_ROOT: \/Users\/Shared\/SceneWorks\/starvector-terminal\/inference/);
  assert.match(workflow, /STARVECTOR_TERMINAL_INFERENCE_ROOT: \/Users\/Shared\/SceneWorks\/starvector-terminal\/inference/);
  assert.doesNotMatch(`${workflow}\n${readiness}`, /\/opt\/sceneworks-terminal/);
  assert.match(readiness, /STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT: D:\\\\sceneworks-terminal\\\\inference-preflight\\\\starvector-terminal-preflight\.json/);
  assert.doesNotMatch(readiness, /campaign_run_id|permanent_pin|concurrency:/);
});

test("readiness workflow cannot start services, claim leases, or execute models", () => {
  assert.doesNotMatch(readiness, /starvector-terminal-product-service|starvector-terminal-producer\.mjs|starvector_terminal_lease|STARVECTOR_TERMINAL_LEASE|vector_generate|cargo\s+(?:build|run)|(?:pip|npm|cargo)\s+install|huggingface-cli|\bcurl\b|\bwget\b/i);
  assert.match(readiness, /Upload macOS readiness report even on failure/);
  assert.match(readiness, /Upload Windows readiness report even on failure/);
});

test("readiness CLI writes a structured failure report before returning nonzero", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-report-")), output = path.join(root, "nested", "report.json");
  await assert.rejects(() => execFile(process.execPath, ["scripts/starvector-terminal-readiness.mjs", root, path.join(root, "missing-plan.json"), root, root, root, process.execPath, root, output]));
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.schema_version, 1); assert.equal(report.kind, "starvector_terminal_readiness"); assert.equal(report.status, "failed"); assert.match(report.error, /ENOENT/);
});

test("readiness validates the complete service tree closure without materializing it", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-weights-"));
  await mkdir(path.join(root, "app")); await mkdir(path.join(root, "hf"));
  await writeFile(path.join(root, "app", "receipts.json"), "receipts"); await writeFile(path.join(root, "hf", "weights.bin"), "weights");
  const tree = async (name) => {
    const file = path.join(root, name, name === "app" ? "receipts.json" : "weights.bin"), bytes = await readFile(file);
    return hash(JSON.stringify([[path.basename(file), bytes.length, hash(bytes)]]));
  };
  const weights = { models: { "starvector-1b": {}, "starvector-8b": {} }, terminal_service_closure: { app_data_relative_path: "app", app_data_sha256: await tree("app"), hf_home_relative_path: "hf", hf_home_sha256: await tree("hf") } };
  const result = await validateTerminalServiceClosure(root, weights);
  assert.equal(result.app_data.file_count, 1); assert.equal(result.hf_home.file_count, 1);
  await writeFile(path.join(root, "hf", "weights.bin"), "drift");
  await assert.rejects(() => validateTerminalServiceClosure(root, weights), /service closure tree hash mismatch/);
});

test("readiness binds all 120 source assets and every suite identity to the pinned corpus", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-corpus-")), inference = path.join(root, "inference"), assets = path.join(root, "assets");
  await mkdir(path.join(inference, "release"), { recursive: true }); await mkdir(path.join(inference, "scripts", "release"), { recursive: true }); await mkdir(assets);
  for (const [name, bytes] of [["source.svg", "svg"], ["input.png", "input"], ["reference.png", "reference"]]) await writeFile(path.join(assets, name), bytes);
  const sources = Array.from({ length: 4 }, (_, index) => ({ dataset: `dataset-${index}`, revision: String(index + 1).repeat(40), row_identity_sha256: "" }));
  const rows = Array.from({ length: 120 }, (_, case_index) => ({ case_index, dataset: sources[Math.floor(case_index / 30)].dataset, revision: sources[Math.floor(case_index / 30)].revision, row_index: case_index % 30, filename: `${case_index}.svg`, svg_path: "source.svg", svg_sha256: hash("svg"), input_png_path: "input.png", png_sha256: hash("input"), reference_png: "reference.png", reference_png_sha256: hash("reference") }));
  const record = (row) => JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 });
  sources.forEach((source, index) => { source.row_identity_sha256 = hash(`${rows.slice(index * 30, index * 30 + 30).map(record).join("\n")}\n`); });
  const rowIdentity = hash(`${rows.map(record).join("\n")}\n`);
  const parityIdentity = hash(`${sources.flatMap((_, index) => rows.slice(index * 30, index * 30 + 5)).map(record).join("\n")}\n`);
  const prompts = Array.from({ length: 60 }, (_, case_index) => { const prompt = `prompt-${case_index}`; return { case_index, case_id: `prompt-v1-${case_index}`, prompt, prompt_sha256: hash(prompt), raster_model: "raster", vector_model: "starvector_8b", expected_raster_revision: "raster-revision", expected_vector_revision: "vector-revision" }; });
  const corpus = { upstream_image_quality_cases: { row_identity_sha256: rowIdentity, sources }, deterministic_parity_cases: { row_identity_sha256: parityIdentity }, sceneworks_owned_suites: { prompt_composition: { content_identity_sha256: hash(prompts.map((entry) => entry.prompt_sha256).join("\n")) } } };
  await writeFile(path.join(inference, "release", "corpus.json"), JSON.stringify(corpus));
  await writeFile(path.join(inference, "scripts", "release", "starvector_terminal_evidence.mjs"), `import { createHash } from "node:crypto"; export function validatePlan(value) { return createHash("sha256").update(JSON.stringify(value)).digest("hex"); }\n`);
  const lifecycle = ["load", "unload", "reload", "memory_reported"], limits = ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"];
  const index = { inference_revision: pin, row_identity_sha256: rowIdentity, rows, lifecycle_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => [tuple, lifecycle.map((operation) => ({ case_id: `${tuple}-${operation}`, operation }))])), limit_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => [tuple, limits.map((finish_reason) => ({ case_id: `${tuple}-${finish_reason}`, finish_reason }))])), prompt_composition: prompts };
  await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index));
  const result = await validateCorpusAssets(inference, "release/corpus.json", assets, pin);
  assert.equal(result.asset_file_references, 360); assert.equal(result.prompt_sha256, corpus.sceneworks_owned_suites.prompt_composition.content_identity_sha256);
  await writeFile(path.join(assets, "input.png"), "drift");
  await assert.rejects(() => validateCorpusAssets(inference, "release/corpus.json", assets, pin), /input PNG hash mismatch/);
});
