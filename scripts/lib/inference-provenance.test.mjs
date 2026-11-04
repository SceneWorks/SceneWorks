import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { inferenceRevision, INFERENCE_PROVENANCE_MANIFEST } from "./inference-provenance.mjs";

test("new evidence provenance reads the neutral dependency and ignores neighbouring revision identities", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "sw-inference-provenance-"));
  try {
    const file = path.join(root, INFERENCE_PROVENANCE_MANIFEST);
    await mkdir(path.dirname(file), { recursive: true });
    const pin = "a".repeat(40);
    const producer = "b".repeat(40);
    const manifest = `[dependencies]\nother = { rev = "${producer}" }\nsceneworks-gen-core = { git = "https://github.com/SceneWorks/inference", rev = "${pin}" }\n`;
    await writeFile(file, manifest);
    assert.equal(await inferenceRevision(root), pin);
    assert.equal(await readFile(file, "utf8"), manifest);
    await writeFile(file, manifest.replaceAll("\n", "\r\n"));
    assert.equal(await inferenceRevision(root), pin, "Windows CRLF checkouts share the same provenance");
    await writeFile(file, manifest.replace(pin, "main"));
    await assert.rejects(inferenceRevision(root), /no exact inference contract pin/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("the compiled Rust provenance, tooling and both consumers share the neutral owner", async () => {
  assert.match(await inferenceRevision(), /^[0-9a-f]{40}$/);
  const root = fileURLToPath(new URL("../..", import.meta.url));
  const [build, provenance, adapter, catalog] = await Promise.all([
    "crates/sceneworks-core/build.rs", "crates/sceneworks-core/src/inference_provenance.rs",
    "crates/sceneworks-memory-adapter/src/lib.rs", "crates/sceneworks-worker/src/catalog_semantic_jobs.rs",
  ].map((file) => readFile(path.join(root, file), "utf8")));
  assert.match(build, /sceneworks-gen-core = /);
  assert.match(provenance, /env!\("SCENEWORKS_INFERENCE_REVISION"\)/);
  assert.match(adapter, /inference_provenance::INFERENCE_REVISION as INFERENCE_PIN/);
  assert.match(catalog, /inference_provenance::INFERENCE_REVISION as INFERENCE_RUNTIME_REVISION/);
  assert.doesNotMatch(adapter, /pub const INFERENCE_PIN: &str = "[0-9a-f]{40}"/);
});
