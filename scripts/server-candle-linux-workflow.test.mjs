// Build-cache/workflow controls only; no Cargo, CUDA compiler or model execution.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, writeFile, readdir, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { linuxFlashCacheIdentity, prepareLinuxFlashCache, configureLinuxFlashCache, selectCudaforgeNvcc } from "./select-linux-flash-cache.mjs";
import { discoverRootTests } from "./run-root-tests.mjs";

const pin = "073e34f21382dc25ef711e0620ebf16bc7e74b6d";
const lock = (sha = pin) => `source = "git+https://github.com/SceneWorks/inference?rev=${sha}#${sha}"\n`;
const inputs = () => ({ lock: lock(), nvcc: "nvcc12.9.86 binary-sha256", hostCompiler: "gcc/g++11.4 binary-sha256", arch: "80", target: "x86_64-unknown-linux-gnu" });
function buildJob(workflow) {
  const start = workflow.indexOf("\n  build-candle:");
  assert.ok(start >= 0);
  const next = workflow.slice(start + 1).search(/\n  [a-z][a-z0-9-]*:/);
  return workflow.slice(start, next < 0 ? undefined : start + 1 + next);
}
test("standard Linux build persistently shares unchanged CUDA compilation across profiles", async () => {
  const job = buildJob(await readFile(new URL("../.github/workflows/server-candle-linux.yml", import.meta.url), "utf8"));
  for (const pattern of [
    /timeout-minutes: 240/, /runs-on: ubuntu-22\.04/, /CUDA_COMPUTE_CAP: "80"/,
    /toolchain: "1\.98\.0"/, /cuda: "12\.9\.1"/,
    /cache-directories: \$\{\{ env.CANDLE_FLASH_ATTN_BUILD_DIR \}\}/,
    /env-vars: CANDLE_FLASH_ATTN_BUILD_DIR NVCC NVCC_CCBIN/, /cache-on-failure: true/,
    /Swatinem\/rust-cache@f0d9c3887740aee45f6153b24b3a6b815192ec16/,
  ]) assert.match(job, pattern);
  const setup = job.indexOf("run: node scripts/select-linux-flash-cache.mjs");
  assert.ok(setup > job.indexOf('cuda: "12.9.1"'), "inspect actual installed toolkit");
  assert.ok(setup < job.indexOf("uses: Swatinem/rust-cache@"), "choose the exact extra directory before restore");
  for (const command of [
    "cargo fetch --locked", "bash scripts/check-gen-core-skew.sh --self-test",
    "bash scripts/check-gen-core-skew.sh sceneworks-worker --features backend-candle",
    "bash scripts/check-gen-core-skew.sh sceneworks-rust-api --features embed-web,backend-candle",
    "cargo clippy -p sceneworks-worker --features backend-candle -- -D warnings",
    "cargo check -p sceneworks-memory-adapter --features candle --bin memory-candle-adapter",
    "npm ci --prefix apps/web", "npm run build --prefix apps/web",
    "cargo build -p sceneworks-rust-api --release --features embed-web,backend-candle",
  ]) assert.ok(job.includes(command), command);
  assert.doesNotMatch(job, /github\.run_(?:id|attempt)|rm -rf|CANDLE_FLASH_ATTN_BUILD_DIR:/);
  assert.ok(discoverRootTests().includes("scripts/server-candle-linux-workflow.test.mjs"));
});
test("persistent Flash identity follows source/compiler/arch inputs, never run identity", async () => {
  const original = inputs(), identity = linuxFlashCacheIdentity(original);
  const actualLock = await readFile(new URL("../Cargo.lock", import.meta.url), "utf8");
  assert.match(linuxFlashCacheIdentity({ ...original, lock: actualLock }), /^[0-9a-f]{64}$/);
  assert.equal(identity, linuxFlashCacheIdentity({ ...original }));
  assert.equal(identity, linuxFlashCacheIdentity({ ...original, runId: "unrelated", runAttempt: 99 }));
  for (const changed of [
    { lock: lock("a".repeat(40)) }, { lock: original.lock + "# dependency change\n" },
    { nvcc: "nvcc other-binary-sha256" }, { hostCompiler: "gcc other-binary-sha256" },
    { arch: "90" }, { target: "aarch64-unknown-linux-gnu" },
    { nvccPrependFlags: "--ptxas-options=-v" }, { nvccAppendFlags: "--disable-warnings" },
  ]) assert.notEqual(identity, linuxFlashCacheIdentity({ ...original, ...changed }));
  for (const key of Object.keys(original)) {
    for (const missing of [undefined, ""]) assert.throws(() => linuxFlashCacheIdentity({ ...original, [key]: missing }));
  }
  for (const bad of [lock("main"), lock().replace(pin, "a".repeat(40)), lock() + lock("b".repeat(40))]) {
    assert.throws(() => linuxFlashCacheIdentity({ ...original, lock: bad }), /one exact inference/);
  }
});
test("persistent Flash setup creates a canonical directory and retains identical-input objects", async () => {
  const box = await mkdtemp(path.join(tmpdir(), "candle-linux-cache-"));
  try {
    const githubEnv = path.join(box, "github-env");
    const directory = await configureLinuxFlashCache(githubEnv, box, inputs());
    assert.equal(await readFile(githubEnv, "utf8"), `CANDLE_FLASH_ATTN_BUILD_DIR=${directory}\n`, "actual setup publishes a newline-terminated canonical path");
    assert.equal(directory, await realpath(directory), "build.rs canonicalize has an existing directory");
    await writeFile(path.join(directory, "libflashattention.a"), "previous successful identical-input build");
    assert.equal(await prepareLinuxFlashCache(box, inputs()), directory);
    assert.deepEqual(await readdir(directory), ["libflashattention.a"], "no destructive startup pruning");
    const different = await prepareLinuxFlashCache(box, { ...inputs(), nvcc: "different actual nvcc" });
    assert.notEqual(different, directory);
    assert.deepEqual(await readdir(different), []);
    await assert.rejects(prepareLinuxFlashCache("relative", inputs()), /absolute/);
    await assert.rejects(prepareLinuxFlashCache(box, { ...inputs(), lock: lock("main") }), /one exact inference/);
  } finally { await rm(box, { recursive: true, force: true }); }
});
test("NVCC selection matches pinned cudaforge 0.1.5 precedence and missing-path fallback", () => {
  const env = { NVCC: "/explicit/nvcc", CUDA_HOME: "/home/cuda", CUDA_PATH: "/path/cuda" };
  const present = new Set([env.NVCC, "/home/cuda/bin/nvcc", "/path/cuda/bin/nvcc.exe", "/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]);
  const select = (pathNvcc) => selectCudaforgeNvcc({ env, pathNvcc, exists: (candidate) => present.has(candidate) });
  assert.equal(select("/PATH/nvcc"), env.NVCC);
  present.delete(env.NVCC);
  assert.equal(select("/PATH/nvcc"), "/PATH/nvcc", "nonexistent NVCC does not mask PATH");
  assert.equal(select(), "/home/cuda/bin/nvcc");
  present.delete("/home/cuda/bin/nvcc");
  assert.equal(select(), "/path/cuda/bin/nvcc.exe", "CUDA_PATH follows HOME and uses the exact .5 suffix");
  present.delete("/path/cuda/bin/nvcc.exe");
  assert.equal(select(), "/usr/local/cuda/bin/nvcc");
  present.delete("/usr/local/cuda/bin/nvcc");
  assert.equal(select(), "/opt/cuda/bin/nvcc");
  present.clear();
  assert.throws(() => select(), /No nvcc/);
});
