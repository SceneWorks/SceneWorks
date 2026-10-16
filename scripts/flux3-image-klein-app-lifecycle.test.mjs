import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { assertCheckoutIdentity, assertModel, assertDownload, imageBody, assertAccepted, assertSavedAsset, assertServedPng, redactSensitiveText, runLifecycle } from "./flux3-image-klein-app-lifecycle.mjs";

const sha = "a".repeat(40), inference = "b".repeat(40);
const revision = "8454017f9ebaada30c998fccd9f67a80c7580f68";
const repo = "black-forest-labs/flux-3-image-klein";
const files = ["config.json", "manifest.json", "model.safetensors", "ae.safetensors", "LICENSE.md", "NOTICE.md", "text_encoder/config.json", "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/model.safetensors.index.json", "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors"];
const model = () => ({ id: "flux3_klein", family: "flux3", gated: true, credentialHost: "huggingface.co", licenseUrl: `https://huggingface.co/${repo}/blob/main/LICENSE.md`, capabilities: ["text_to_image"], downloads: [{ provider: "huggingface", repo, revision, files: [] }] });
const context = { projectId: "project_test", workerId: "worker_test", backend: "mlx", adapter: "mlx_flux3" };
function fixture() {
  const body = imageBody(context.projectId);
  // Production persistence returns this raw sidecar in job.result.assets. Library GET hydrates
  // three additional fields; cloning one identical envelope for all surfaces hides contract bugs.
  const sidecar = { schemaVersion: 1, id: "asset_test", projectId: context.projectId, generationSetId: "genset_test", type: "image", displayName: "teapot", createdAt: "2026-10-16T12:00:00Z", origin: "image", status: { favorite: false, rating: 0, rejected: false, trashed: false }, file: { path: "assets/images/genset_test/output.png", mimeType: "image/png", width: 1024, height: 1024 }, lineage: { jobId: "job_test" }, recipe: { model: body.model, mode: body.mode, adapter: context.adapter, prompt: body.prompt, seed: body.seed, normalizedSettings: { width: 1024, height: 1024, count: 1, family: "flux3" }, rawAdapterSettings: { steps: 16, numInferenceSteps: 16, realModelInference: true, repo } } };
  const generationSet = { schemaVersion: 1, id: sidecar.generationSetId, projectId: context.projectId, jobId: "job_test", mode: body.mode, model: body.model, prompt: body.prompt, negativePrompt: body.negativePrompt, count: 1, createdAt: "2026-10-16T11:59:59Z", recipe: structuredClone(sidecar.recipe) };
  const asset = { ...structuredClone(sidecar), url: `/api/v1/projects/${context.projectId}/files/${sidecar.file.path}`, sidecarPath: "assets/images/genset_test/output.sceneworks.json", generationSet };
  const job = { id: "job_test", projectId: context.projectId, type: "image_generate", payload: body, status: "completed", stage: "completed", backend: "mlx", workerId: context.workerId, error: null, result: { model: "flux3_klein", adapter: "mlx_flux3", generationSetId: sidecar.generationSetId, assetIds: [sidecar.id], assets: [structuredClone(sidecar)] } };
  return { body, job, asset, sidecar };
}

test("registered dispatcher isolates lifecycle from catalog and calls exact-commit Mac workflow", async () => {
  const dispatcher = await readFile(new URL("../.github/workflows/memory-catalog-campaign.yml", import.meta.url), "utf8");
  assert.match(dispatcher, /- flux3-image-klein-app-lifecycle/);
  assert.match(dispatcher, /flux3-app-lifecycle:\s+if:.*inputs.profile == 'flux3-image-klein-app-lifecycle'/);
  assert.match(dispatcher, /uses: \.\/.github\/workflows\/flux3-image-klein-app-lifecycle.yml/);
  assert.match(dispatcher, /source_sha: \$\{\{ inputs.ref \}\}/);
  assert.equal((dispatcher.match(/\(inputs.profile == '' \|\| inputs.profile == 'catalog'\)/g) ?? []).length, 2);
  const workflow = await readFile(new URL("../.github/workflows/flux3-image-klein-app-lifecycle.yml", import.meta.url), "utf8");
  assert.match(workflow, /workflow_call:/); assert.doesNotMatch(workflow, /workflow_dispatch:|push:|pull_request:/);
  assert.match(workflow, /rw-mage/); assert.match(workflow, /rw-starvector/);
  assert.match(workflow, /timeout-minutes: 180/);
  assert.match(workflow, /ref: \$\{\{ inputs.source_sha \}\}/);
  assert.match(workflow, /--preflight-only/);
  assert.doesNotMatch(workflow, /secrets\.HF_TOKEN|measure-memory|bump-inference|memory.*adapter/);
  const jobEnv = workflow.slice(workflow.indexOf("    env:\n"), workflow.indexOf("    steps:\n"));
  assert.doesNotMatch(jobEnv, /runner\.temp/, "runner context is unavailable in job-level env");
  const script = await readFile(new URL("./flux3-image-klein-app-lifecycle.mjs", import.meta.url), "utf8");
  assert.match(script, /must\("POST", "\/api\/v1\/models\/flux3_klein\/download"/);
  assert.match(script, /must\("POST", "\/api\/v1\/image\/jobs"/);
  assert.doesNotMatch(script, /\/progress["`]|\/claim["`]|\/assets["`],|\/import["`]/, "controller must not impersonate the worker or import a provider image");
});

test("exact source, dispatch identity and clean checkout refuse mismatch", () => {
  const input = { source: sha, inference, dispatch: sha, head: sha, dirty: "" };
  assert.doesNotThrow(() => assertCheckoutIdentity(input));
  for (const mutation of [{ source: "main" }, { inference: "HEAD" }, { dispatch: inference }, { head: inference }, { dirty: " M Cargo.toml" }]) assert.throws(() => assertCheckoutIdentity({ ...input, ...mutation }));
});

test("controller retains failed exact-source receipt before build, auth or process launch", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "flux3-app-refusal-")), output = path.join(directory, "attempt");
  try {
    await assert.rejects(runLifecycle({ ...process.env, RUNNER_TEMP: directory, FLUX3_APP_OUTPUT: output, FLUX3_APP_SOURCE_SHA: sha, FLUX3_APP_INFERENCE_SHA: inference, GITHUB_SHA: inference, FLUX3_APP_RUNNER_TARGET: "secondary" }, { preflightOnly: true }), /dispatch SHA/);
    const receipt = JSON.parse(await readFile(path.join(output, "evidence", "receipt.json"), "utf8"));
    assert.equal(receipt.status, "failed"); assert.equal(receipt.phase, "preflight");
    assert.deepEqual(receipt.cleanup, []); assert.equal(receipt.auth, undefined); assert.equal(receipt.saved_asset, undefined);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("pinned bundled components, license and gated auth are mandatory", () => {
  assert.doesNotThrow(() => assertModel(model()));
  for (const mutate of [(m) => m.downloads[0].revision = sha, (m) => m.downloads[0].repo = "unapproved/rehost", (m) => m.gated = false, (m) => m.credentialHost = "elsewhere", (m) => m.licenseUrl = "https://example.com", (m) => m.downloads[0].files = ["model.safetensors"], (m) => m.downloads[0].files = files]) { const m = model(); mutate(m); assert.throws(() => assertModel(m)); }
  const job = { type: "model_download", payload: { modelId: "flux3_klein", repo, revision } };
  assert.doesNotThrow(() => assertDownload(job));
  assert.throws(() => assertDownload({ ...job, payload: { ...job.payload, revision: sha } }));
});

test("ordinary typed image request and saved library asset bind job, runtime and provenance", () => {
  const { body, job, asset, sidecar } = fixture();
  assert.doesNotThrow(() => assertAccepted(job, body));
  assert.deepEqual(job.result.assets[0], sidecar);
  assert.notDeepEqual(job.result.assets[0], asset, "the production library response is hydrated, not the raw job envelope");
  assert.notEqual(asset.generationSet.createdAt, sidecar.createdAt, "the generation set is created before the image sidecar");
  assert.equal(assertSavedAsset(job, asset, sidecar, context).asset_id, asset.id);
  for (const mutate of [(f) => f.job.status = "running", (f) => f.job.type = "other", (f) => f.job.backend = "candle", (f) => f.job.workerId = "other", (f) => f.job.result.assetWrites = [{}], (f) => f.job.result.assetIds = ["other"], (f) => f.asset.lineage.jobId = "other", (f) => f.asset.recipe.adapter = "procedural_stub", (f) => f.asset.recipe.model = "flux2_klein", (f) => f.asset.recipe.prompt = "different", (f) => f.asset.recipe.seed++, (f) => f.asset.recipe.rawAdapterSettings.realModelInference = false, (f) => f.asset.file.path = "../../provider.png", (f) => f.sidecar.recipe.seed++, (f) => f.job.payload.model = "flux2_klein"]) { const f = fixture(); mutate(f); assert.throws(() => assertSavedAsset(f.job, f.asset, f.sidecar, context)); }
  assert.throws(() => assertAccepted({ ...job, payload: { ...body, advanced: { ...body.advanced, steps: 1 } } }, body));
});

test("raw job and library persisted projections reject recipe, file and provenance divergence", () => {
  for (const [name, mutate] of [
    ["job recipe", (f) => f.job.result.assets[0].recipe.seed++],
    ["library recipe", (f) => f.asset.recipe.seed++],
    ["sidecar recipe", (f) => f.sidecar.recipe.seed++],
    ["job file", (f) => f.job.result.assets[0].file.path = "assets/images/genset_test/other.png"],
    ["library file", (f) => f.asset.file.path = "assets/images/genset_test/other.png"],
    ["sidecar file", (f) => f.sidecar.file.path = "assets/images/genset_test/other.png"],
    ["job provenance", (f) => f.job.result.assets[0].lineage.jobId = "another_job"],
    ["library provenance", (f) => f.asset.lineage.jobId = "another_job"],
    ["sidecar provenance", (f) => f.sidecar.lineage.jobId = "another_job"],
    ["unrecognized persisted field", (f) => f.asset.unexpectedPersistedField = true],
  ]) {
    const f = fixture(); mutate(f);
    assert.throws(() => assertSavedAsset(f.job, f.asset, f.sidecar, context), /must agree/, name);
  }
});

test("hydrated fields bind exact project file, sidecar and embedded generation provenance", () => {
  for (const [name, mutate] of [
    ["URL project", (a) => a.url = a.url.replace(context.projectId, "another_project")],
    ["URL file", (a) => a.url = a.url.replace("output.png", "other.png")],
    ["relative sidecar path", (a) => a.sidecarPath = "assets/images/genset_test/other.sceneworks.json"],
    ["absolute sidecar path", (a) => a.sidecarPath = `/project/${a.sidecarPath}`],
    ["generation set missing", (a) => delete a.generationSet],
    ["generation set schema", (a) => a.generationSet.schemaVersion = 2],
    ["generation set id", (a) => a.generationSet.id = "another_set"],
    ["generation set project", (a) => a.generationSet.projectId = "another_project"],
    ["generation set job", (a) => a.generationSet.jobId = "another_job"],
    ["generation set mode", (a) => a.generationSet.mode = "image_edit"],
    ["generation set model", (a) => a.generationSet.model = "flux2_klein"],
    ["generation set prompt", (a) => a.generationSet.prompt = "another_prompt"],
    ["generation set negative prompt", (a) => a.generationSet.negativePrompt = "not_requested"],
    ["generation set count", (a) => a.generationSet.count = 2],
    ["generation creation provenance", (a) => a.generationSet.createdAt = ""],
    ["generation recipe", (a) => a.generationSet.recipe.seed++],
  ]) {
    const f = fixture(); mutate(f.asset);
    assert.throws(() => assertSavedAsset(f.job, f.asset, f.sidecar, context), /hydrated/, name);
  }
});

test("negative controls kill removed dispatch and worker provenance guards", () => {
  const dispatchMutant = Function("fact", `return (${assertCheckoutIdentity.toString().replace("source === dispatch &&", "")});`)((v, m) => assert.ok(v, m));
  const input = { source: sha, inference, dispatch: inference, head: sha, dirty: "" };
  assert.throws(() => assertCheckoutIdentity(input));
  assert.doesNotThrow(() => dispatchMutant(input), "removing dispatch guard admits a wrong workflow source; negative test kills it");
  const compile = (source) => Function("assert", "fact", "imageBody", "assertAccepted", "MODEL_REPO", `return (${source});`)(assert, (v, m) => assert.ok(v, m), imageBody, assertAccepted, repo);
  const workerMutant = compile(assertSavedAsset.toString().replace("job.workerId === workerId &&", ""));
  const f = fixture(); f.job.workerId = "unowned_worker";
  assert.throws(() => assertSavedAsset(f.job, f.asset, f.sidecar, context));
  assert.doesNotThrow(() => workerMutant(f.job, f.asset, f.sidecar, context), "removing worker ownership guard admits unrelated completion; negative test kills it");
  const publicationMutant = compile(assertSavedAsset.toString().replace('!Object.hasOwn(result, "assetWrites") &&', ""));
  const p = fixture(); p.job.result.assetWrites = [{}];
  assert.throws(() => assertSavedAsset(p.job, p.asset, p.sidecar, context));
  assert.doesNotThrow(() => publicationMutant(p.job, p.asset, p.sidecar, context), "removing publication guard admits unfinished persistence; negative test kills it");
});

test("backend-neutral asset validator permits a real Candle identity without claiming Mac evidence", () => {
  const f = fixture(); f.job.backend = "candle"; f.job.result.adapter = "candle_flux3";
  for (const a of [f.asset, f.sidecar, f.job.result.assets[0], f.asset.generationSet]) a.recipe.adapter = "candle_flux3";
  assert.doesNotThrow(() => assertSavedAsset(f.job, f.asset, f.sidecar, { ...context, backend: "candle", adapter: "candle_flux3" }));
});

test("artifact text redacts raw and URL-encoded runner credentials", () => {
  const value = ["fixture", "account", "only"].join("/");
  const raw = `Authorization: Bearer ${value}; url=${encodeURIComponent(value)}; ${value}`;
  const redacted = redactSensitiveText(raw, new Set([value]));
  assert.equal(redacted, "Authorization: Bearer [REDACTED]; url=[REDACTED]; [REDACTED]");
  assert.equal(redactSensitiveText("public job identity", new Set([value])), "public job identity");
});

test("served PNG must exactly match persisted bytes and requested dimensions", () => {
  const bytes = Buffer.alloc(40); Buffer.from("89504e470d0a1a0a", "hex").copy(bytes); bytes.write("IHDR", 12); bytes.writeUInt32BE(1024, 16); bytes.writeUInt32BE(1024, 20);
  assert.doesNotThrow(() => assertServedPng(bytes, bytes, 1024, 1024));
  const other = Buffer.from(bytes); other[39] = 1;
  assert.throws(() => assertServedPng(bytes, other, 1024, 1024));
  assert.throws(() => assertServedPng(bytes, bytes, 512, 512));
  assert.throws(() => assertServedPng(Buffer.from("provider PNG path"), bytes, 1024, 1024));
});
