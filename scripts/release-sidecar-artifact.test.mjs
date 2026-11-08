import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { artifactBinary, cudaReleaseLine, selectCandidateRun, verifyReceiptFields } from "./release-sidecar-artifact.mjs";

const source = async (name) => readFile(new URL(`../${name}`, import.meta.url), "utf8");
const job = (workflow, name) => {
  const start = workflow.indexOf(`\n  ${name}:`);
  assert.ok(start >= 0, `${name} exists`);
  const next = workflow.slice(start + 1).search(/\n  [a-z][a-z0-9-]*:/);
  return workflow.slice(start, next < 0 ? undefined : start + 1 + next);
};

test("release sidecar receipt rejects source, target, compiler, size and content drift", () => {
  const expected = {
    schema: 1, platform: "linux", target: "x86_64-unknown-linux-gnu",
    features: "embed-web,backend-candle", compute_cap: "80",
    source_sha: "a".repeat(40), cargo_lock_sha256: "b".repeat(64), rustc: "rustc 1.98.0",
  };
  const compilerOutput = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.9, V12.9.86\nBuild cuda_12.9.r12.9/compiler.36037853_0";
  const receipt = {
    ...expected, nvcc: cudaReleaseLine(compilerOutput), host_compiler: "gcc 11.4",
    binary_bytes: 100, binary_sha256: "c".repeat(64),
  };
  assert.doesNotThrow(() => verifyReceiptFields(receipt, expected, 100, "c".repeat(64)));
  assert.throws(() => cudaReleaseLine(compilerOutput.replace("release 12.9", "release 12.8")), /CUDA 12.9/);
  for (const key of Object.keys(expected)) {
    assert.throws(() => verifyReceiptFields({ ...receipt, [key]: "wrong" }, expected, 100, "c".repeat(64)), /mismatch/);
  }
  assert.throws(() => verifyReceiptFields({ ...receipt, nvcc: "release 12.8" }, expected, 100, "c".repeat(64)), /compiler provenance/);
  assert.throws(() => verifyReceiptFields({ ...receipt, host_compiler: "" }, expected, 100, "c".repeat(64)), /host compiler provenance/);
  assert.throws(() => verifyReceiptFields(receipt, expected, 101, "c".repeat(64)), /size or SHA-256/);
  assert.throws(() => verifyReceiptFields(receipt, expected, 100, "d".repeat(64)), /size or SHA-256/);
  assert.equal(artifactBinary("/input", "linux"), "/input/sceneworks-api");
  assert.equal(artifactBinary("/input", "windows"), "/input/sceneworks-api.exe");
  assert.throws(() => artifactBinary("/input", "macos"), /unsupported/);
});

test("tag lookup accepts only a successful preparation run from the exact release commit", () => {
  const sha = "a".repeat(40);
  const valid = { id: 5, head_sha: sha, head_branch: "release/next", path: ".github/workflows/release-sidecars.yml", event: "push", status: "completed", conclusion: "success" };
  for (const changed of [
    { head_sha: "b".repeat(40) }, { head_branch: "main" }, { path: ".github/workflows/release.yml" },
    { event: "workflow_dispatch" },
    { status: "in_progress" }, { conclusion: "failure" },
  ]) assert.equal(selectCandidateRun([{ ...valid, ...changed }], sha, "linux"), null);
  assert.equal(selectCandidateRun([{ ...valid, conclusion: "failure" }, valid], sha, "windows")?.id, 5);
});

test("pre-tag sidecars are unsigned and source-bound; tag jobs verify before signing", async () => {
  const prep = await source(".github/workflows/release-sidecars.yml");
  const release = await source(".github/workflows/release.yml");
  const builder = await source("apps/desktop/scripts/build-sidecar.mjs");
  assert.match(prep, /branches: \[release\/next\]/);
  assert.match(prep, /github\.ref == 'refs\/heads\/release\/next'/);
  assert.doesNotMatch(prep, /TAURI_SIGNING_PRIVATE_KEY|AZURE_CLIENT_SECRET|gh release upload/);
  const windowsPrep = job(prep, "windows"), linuxPrep = job(prep, "linux");
  assert.match(windowsPrep, /runs-on: \[self-hosted, Windows, X64, cuda\]/);
  assert.match(windowsPrep, /build-sidecar\.mjs/);
  assert.match(windowsPrep, /seal windows/);
  assert.match(linuxPrep, /runs-on: ubuntu-22\.04/);
  assert.ok(linuxPrep.indexOf("select-linux-flash-cache.mjs") < linuxPrep.indexOf("cache-directories:"));
  assert.ok(linuxPrep.indexOf("cache-directories:") < linuxPrep.indexOf("build-sidecar.mjs"));
  assert.match(linuxPrep, /seal linux/);
  for (const platform of ["windows", "linux"]) {
    const packaging = job(release, platform);
    assert.ok(packaging.indexOf(`find ${platform}`) < packaging.indexOf(`release-sidecar-${platform}`));
    assert.ok(packaging.indexOf(`verify ${platform}`) < packaging.indexOf("TAURI_SIGNING_PRIVATE_KEY:"));
    assert.match(packaging, /SCENEWORKS_RELEASE_SIDECAR_ROOT:/);
    assert.match(packaging, /SCENEWORKS_RELEASE_SIDECAR_RUN_ID: \$\{\{ steps\.sidecar\.outputs\.run-id \}\}/);
    assert.doesNotMatch(packaging, /cargo build -p sceneworks-rust-api/);
  }
  assert.match(builder, /await verifyReleaseSidecar\(/);
  assert.match(builder, /const src = prebuiltBinary \?\?/);
  assert.match(job(release, "linux"), /--max-entry-bytes 1610612736/);
});
