import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { assemblePreflight, assembleWeights, downloadExact, installCheckout } from "./starvector-terminal-provision.mjs";

const execFile = promisify(execFileCallback);
const digest = (value) => createHash("sha256").update(value).digest("hex");
const workflow = await readFile(".github/workflows/starvector-terminal-provision.yml", "utf8");

test("provision workflow is dispatch-only and never runs a model, service, campaign, or lease", () => {
  assert.match(workflow, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(workflow, /^\s+(push|pull_request|schedule):/m);
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.match(workflow, /inference_revision:[\s\S]*required: true/);
  assert.match(workflow, /inference_preflight_run_id:[\s\S]*required: true/);
  assert.match(workflow, /\/Users\/Shared\/SceneWorks\/starvector-terminal/);
  assert.ok(workflow.includes("D:\\sceneworks-terminal"));
  assert.equal((workflow.match(/path: sceneworks/g) ?? []).length, 2);
  assert.equal((workflow.match(/path: inference-source/g) ?? []).length, 2);
  assert.equal((workflow.match(/working-directory: .*sceneworks/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /starvector-terminal-product-service|starvector-terminal-producer\.mjs\s+(?:run|seal)|starvector_terminal_lease|campaign_run_id|vector_generate/i);
  assert.equal((workflow.match(/starvector-terminal-readiness\.mjs/g) ?? []).length, 2);
  assert.equal((workflow.match(/Upload .* provisioning readiness even on failure/g) ?? []).length, 2);
  assert.match(workflow, /STARVECTOR_PROMPT_PROVIDER: candle_flux/);
  assert.doesNotMatch(workflow, /STARVECTOR_PROMPT_PROVIDER: flux_diffusers/);
});

test("preflight assembly requires and copies exactly two inventories plus four hooks", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-preflight-")), source = path.join(root, "source"), destination = path.join(root, "destination");
  await mkdir(path.join(source, "inventory"), { recursive: true }); await mkdir(path.join(source, "hooks"), { recursive: true });
  const inventory_artifacts = [], hook_logs = [];
  for (const tier of ["1b", "8b"]) { const relative = `inventory/${tier}.json`, bytes = `inventory-${tier}`; await writeFile(path.join(source, relative), bytes); inventory_artifacts.push({ tier, path: relative, sha256: digest(bytes) }); }
  for (const backend of ["mlx", "candle-cuda"]) for (const tier of ["1b", "8b"]) { const relative = `hooks/${backend}-${tier}.log`, bytes = `${backend}-${tier}`; await writeFile(path.join(source, relative), bytes); hook_logs.push({ backend, tier, path: relative, sha256: digest(bytes) }); }
  const revision = "a".repeat(40), index = { workflow_run_id: "123", workflow_run_attempt: 1, head_sha: revision, inventory_artifacts, hook_logs };
  await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(index));
  await assemblePreflight(source, destination, revision); await assemblePreflight(source, destination, revision);
  assert.deepEqual(JSON.parse(await readFile(path.join(destination, "starvector-terminal-preflight.json"))), index);
  const incomplete = structuredClone(index); incomplete.hook_logs.pop(); await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(incomplete));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "incomplete"), revision), /cardinality/);
  const duplicate = structuredClone(index); duplicate.hook_logs[3].backend = "mlx"; await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(duplicate));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "duplicate"), revision), /identities are incomplete/);
  const provenance = structuredClone(index); provenance.workflow_run_attempt = 0; await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(provenance));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "provenance"), revision), /identity\/cardinality/);
});

test("weights assembly inventories fixed model roots and copies only source-produced service state", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-weights-")), hostRoot = path.join(root, "host"), sources = path.join(root, "sources");
  for (const [relative, file] of [["host/weights/models/starvector-1b", "one.bin"], ["host/weights/models/starvector-8b", "eight.bin"], ["sources/prompt", "prompt.bin"], ["sources/hf", "blob.bin"]]) { await mkdir(path.join(root, relative), { recursive: true }); await writeFile(path.join(root, relative, file), `${relative}-${file}`); }
  await mkdir(path.join(sources, "app", "models", "receipt"), { recursive: true });
  const receipt = (repo, snapshotRevision) => ({ schemaVersion: 2, repo, snapshotRevision, resolvedFiles: ["weights.bin"] });
  await writeFile(path.join(sources, "app", "models", "receipt", "weights.bin"), "weights");
  const validReceipts = { ...receipt("SceneWorks/flux1-schnell-mlx", "revision"), receipts: [receipt("starvector/starvector-1b-im2svg", "380ab95d25a8e9ab1dc825debe238b4953ae13b9"), receipt("starvector/starvector-8b-im2svg", "518beea8dcb5f7a37c5911e92d1d62a76beee7f9"), receipt("SceneWorks/flux1-schnell-mlx", "revision")] };
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(validReceipts));
  const manifest = await assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptRaster: path.join(sources, "prompt"), promptProvider: "provider", promptModel: "model", promptRevision: "revision" });
  assert.deepEqual(Object.keys(manifest.models).sort(), ["starvector-1b", "starvector-8b"]);
  assert.equal(manifest.prompt_raster.provider_id, "provider");
  assert.match(manifest.terminal_service_closure.app_data_sha256, /^[a-f0-9]{64}$/);
  await assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptRaster: path.join(sources, "prompt"), promptProvider: "provider", promptModel: "model", promptRevision: "revision" });
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify({ ...receipt("SceneWorks/flux1-schnell-mlx", "revision"), receipts: [receipt("starvector/starvector-1b-im2svg", "380ab95d25a8e9ab1dc825debe238b4953ae13b9"), receipt("starvector/starvector-8b-im2svg", "518beea8dcb5f7a37c5911e92d1d62a76beee7f9"), { ...receipt("SceneWorks/flux1-schnell-mlx", "revision"), resolvedFiles: ["missing.bin"] }] }));
  await assert.rejects(() => assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptRaster: path.join(sources, "prompt"), promptProvider: "provider", promptModel: "model", promptRevision: "revision" }), /lacks resolved file/);
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(validReceipts));
  await symlink(path.join(sources, "hf", "blob.bin"), path.join(sources, "hf", "linked.bin"));
  await assert.rejects(() => assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptRaster: path.join(sources, "prompt"), promptProvider: "provider", promptModel: "model", promptRevision: "revision" }), /symlink/);
});

test("exact downloader is idempotent and refuses existing drift without network access", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-download-")), target = path.join(root, "artifact");
  await writeFile(target, "fixed"); await downloadExact("https://example.invalid/artifact", target, digest("fixed"));
  await assert.rejects(() => downloadExact("https://example.invalid/artifact", target, digest("other")), /existing download differs/);
});

test("inference checkout publication binds an exact clean detached revision", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-checkout-")), source = path.join(root, "source"), destination = path.join(root, "destination");
  await mkdir(path.join(source, "release"), { recursive: true }); await mkdir(path.join(source, "scripts", "release"), { recursive: true });
  for (const relative of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) await writeFile(path.join(source, relative), relative);
  await execFile("git", ["init", source]); await execFile("git", ["-C", source, "add", "."]); await execFile("git", ["-C", source, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "fixture"]);
  const revision = (await execFile("git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
  assert.equal(await installCheckout(source, destination, revision), revision); assert.equal(await installCheckout(source, destination, revision), revision);
  await writeFile(path.join(destination, "drift"), "drift"); await assert.rejects(() => installCheckout(source, destination, revision), /not exact and clean/);
});
