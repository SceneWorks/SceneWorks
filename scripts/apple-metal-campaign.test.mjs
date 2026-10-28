// CPU source/metadata fixtures only. No weights, Cargo build, GPU probe or production lock.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync, execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, mkdir, readFile, writeFile, rm, realpath, access, copyFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { cleanupSourceWorktree } from "./ci/memory-catalog/source-worktree.mjs";
import { bindMetallib, readBoundAppMetallib } from "./ci/memory-catalog/bind-metallib.mjs";
const run = promisify(execFile);
const root = fileURLToPath(new URL("../", import.meta.url));
const helper = path.join(root, "scripts/ci/memory-catalog/source-worktree.mjs");
const git = (directory, ...args) => execFileSync("git", ["-C", directory, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
async function temporary(t) {
  const directory = await realpath(await mkdtemp(path.join(tmpdir(), "sw-campaign-cpu-")));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
test("two concurrent exact-pin source preparations preserve cache and both capture HEADs", async (t) => {
  const directory = await temporary(t), cache = path.join(directory, "objects");
  await mkdir(cache); git(cache, "init", "--quiet");
  git(cache, "config", "user.name", "CPU fixture"); git(cache, "config", "user.email", "fixture@example.invalid");
  const pins = [];
  for (const value of ["first", "second"]) {
    await writeFile(path.join(cache, "metadata.txt"), value);
    git(cache, "add", "metadata.txt"); git(cache, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", value);
    pins.push(git(cache, "rev-parse", "HEAD"));
  }
  const cachedHead = git(cache, "rev-parse", "HEAD"), cachedTree = git(cache, "rev-parse", "HEAD^{tree}");
  const captures = await Promise.all(pins.map(async (pin) => (await run(process.execPath, [helper, "prepare", cache, pin, directory])).stdout.trim()));
  assert.notEqual(...captures);
  for (const [index, source] of captures.entries()) {
    assert.equal(git(source, "rev-parse", "HEAD"), pins[index]);
    assert.equal(await readFile(path.join(source, "metadata.txt"), "utf8"), index === 0 ? "first" : "second");
    assert.equal(git(cache, "rev-parse", `refs/sceneworks/source-pins/${pins[index]}`), pins[index]);
  }
  assert.equal(git(cache, "rev-parse", "HEAD"), cachedHead);
  assert.equal(git(cache, "rev-parse", "HEAD^{tree}"), cachedTree);
  await writeFile(path.join(captures[0], "foreign.txt"), "must not delete");
  await assert.rejects(cleanupSourceWorktree(captures[0]), /refusing cleanup/);
  await access(path.join(captures[0], "foreign.txt"));
  await rm(path.join(captures[0], "foreign.txt"));
  await cleanupSourceWorktree(captures[0]);
  assert.equal(git(captures[1], "rev-parse", "HEAD"), pins[1]);
  await cleanupSourceWorktree(captures[1]);
  assert.equal(git(cache, "rev-parse", "HEAD"), cachedHead);
  await assert.rejects(cleanupSourceWorktree(cache));
});
async function nativeFixture(t, prebuilt = false, profile = "release", consumer) {
  const artifactName = consumer === "app" ? "sceneworks-rust-api" : ["iris", "supra2"].includes(consumer) ? "sceneworks_worker" : profile === "release" ? "memory-mlx-adapter" : "dump-engine-capabilities";
  const directory = await temporary(t), target = path.join(directory, "target"), output = path.join(target, profile, "build", "pmetal-mlx-sys-fixture", "out");
  await mkdir(output, { recursive: true });
  const library = prebuilt ? path.join(directory, "pinned-prebuilt", "Release-metal") : path.join(output, "lib");
  await mkdir(library, { recursive: true });
  const pin = "f".repeat(40), source = path.join(library, "mlx.metallib"), lockFile = path.join(directory, "Cargo.lock"), messages = path.join(directory, "build.jsonl");
  await writeFile(source, "opaque CPU metadata fixture, not an executable metallib");
  for (const name of ["libmlx.a", "libmlxc.a"]) await writeFile(path.join(library, name), "opaque archive fixture; never linked");
  await writeFile(path.join(library, "pmetal-mlx-prebuilt.txt"), `schema=1\nbuild_type=${profile === "release" ? "Release" : "Debug"}\ntarget=aarch64-apple-darwin\nfeatures=accelerate,metal\ndeployment_target=15.0\nfingerprint=${"a".repeat(64)}\n`);
  await writeFile(path.join(path.dirname(output), "output"), `cargo:metallib=${source}\n`);
  await writeFile(lockFile, `name = "pmetal-mlx-sys"\nversion = "0.2.4"\nsource = "git+https://github.com/michaeltrefry/mlx-rs?rev=${pin}#${pin}"\n`);
  const adapter = ["iris", "supra2"].includes(consumer) ? path.join(target, profile, "deps", `${artifactName}-aabbccdd`) : path.join(target, profile, artifactName);
  await mkdir(path.dirname(adapter), { recursive: true }); await writeFile(adapter, "CPU text fixture, never executed");
  const records = [
    { reason: "build-script-executed", package_id: `git+https://github.com/michaeltrefry/mlx-rs?rev=${pin}#pmetal-mlx-sys@0.2.4`, out_dir: output, linked_paths: [`native=${library}`] },
    { reason: "compiler-artifact", target: { name: artifactName }, executable: adapter, profile: { debug_assertions: profile === "debug", test: ["iris", "supra2"].includes(consumer) } },
    { reason: "build-finished", success: true },
  ];
  await writeFile(messages, records.map(JSON.stringify).join("\n"));
  return { messages, lockFile, targetRoot: target, temporaryRoot: directory, records, source, profile, consumer };
}
for (const prebuilt of [false, true]) test(`${prebuilt ? "prebuilt" : "source-built"} selected native artifact binds a private metallib without global fallback`, async (t) => {
  const fixture = await nativeFixture(t, prebuilt);
  const binding = await bindMetallib({ ...fixture, explicitOverride: prebuilt ? fixture.source : "" });
  assert.notEqual(binding.path, fixture.source);
  assert.equal(await readFile(binding.path, "utf8"), await readFile(fixture.source, "utf8"));
  const receipt = JSON.parse(await readFile(path.join(path.dirname(binding.path), "binding.json"), "utf8"));
  assert.equal(receipt.pin, "f".repeat(40)); assert.equal(receipt.profile, "release");
  assert.equal(receipt.flavor.build_type, "Release"); assert.match(receipt.executableSha256, /^[0-9a-f]{64}$/);
  await writeFile(fixture.source, "later cache replacement cannot change job-bound bytes");
  assert.notEqual(await readFile(binding.path, "utf8"), await readFile(fixture.source, "utf8"));
});
for (const prebuilt of [false, true]) test(`Debug ${prebuilt ? "prebuilt" : "source"} native reuse keeps the genuine selected manifest and archives`, async (t) => {
  const fixture = await nativeFixture(t, prebuilt, "debug");
  const directory = path.dirname(fixture.source), manifest = path.join(directory, "pmetal-mlx-prebuilt.txt");
  const original = await readFile(manifest, "utf8");
  const binding = await bindMetallib(fixture);
  assert.equal(binding.nativeDirectory, directory);
  const receipt = JSON.parse(await readFile(binding.receipt, "utf8"));
  assert.equal(receipt.profile, "debug"); assert.equal(receipt.flavor.build_type, "Debug");
  assert.match(receipt.nativeArchives["libmlx.a"], /^[0-9a-f]{64}$/);
  assert.equal(await readFile(manifest, "utf8"), original, "never manufacture or rewrite a passing prebuilt key");
  const environment = path.join(fixture.temporaryRoot, "github-env"); await writeFile(environment, "");
  await run(process.execPath, [path.join(root, "scripts/ci/memory-catalog/bind-metallib.mjs"), fixture.messages, "--debug-worker"], {
    cwd: fixture.temporaryRoot,
    env: { ...process.env, CARGO_TARGET_DIR: fixture.targetRoot, RUNNER_TEMP: fixture.temporaryRoot,
      GITHUB_ENV: environment, WORK_DIR_NATIVE: fixture.temporaryRoot, PMETAL_METALLIB_PATH: "", PMETAL_MLX_PREBUILT_DIR: "" },
  });
  assert.match(await readFile(environment, "utf8"), /PMETAL_METALLIB_PATH=.*memory-catalog-metallib-/);
  assert.ok((await readFile(environment, "utf8")).includes(`PMETAL_MLX_PREBUILT_DIR=${directory}\n`));
  await writeFile(manifest, original.replace("build_type=Debug", "build_type=Release"));
  await assert.rejects(bindMetallib(fixture), /debug Metal flavor/);
  await writeFile(manifest, original);
  await writeFile(fixture.messages, fixture.records.map((entry) => JSON.stringify(entry.reason === "build-script-executed" ? { ...entry, linked_paths: [] } : entry)).join("\n"));
  await assert.rejects(bindMetallib(fixture), /refusing shared fallback/);
});

test("stale other-ref, ambiguous outputs, wrong profile and mismatching explicit overrides refuse before capture", async (t) => {
  const fixture = await nativeFixture(t);
  const save = async (records) => writeFile(fixture.messages, records.map(JSON.stringify).join("\n"));
  await save(fixture.records.map((entry) => entry.reason === "build-script-executed" ? { ...entry, package_id: entry.package_id.replace("f".repeat(40), "e".repeat(40)) } : entry));
  await assert.rejects(bindMetallib(fixture), /exactly one pinned/);
  await save([...fixture.records, { ...fixture.records[0], out_dir: `${fixture.records[0].out_dir}-other` }]);
  await assert.rejects(bindMetallib(fixture), /exactly one pinned/);
  await save(fixture.records.map((entry) => entry.reason === "compiler-artifact" ? { ...entry, profile: { debug_assertions: true, test: false } } : entry));
  await assert.rejects(bindMetallib(fixture), /release build/);
  await save(fixture.records);
  const originalLock = await readFile(fixture.lockFile, "utf8");
  await writeFile(fixture.lockFile, originalLock.replace(`#${"f".repeat(40)}`, `#${"e".repeat(40)}`));
  await assert.rejects(bindMetallib(fixture), /requested\/native Git revision/);
  await writeFile(fixture.lockFile, originalLock);
  const manifest = path.join(path.dirname(fixture.source), "pmetal-mlx-prebuilt.txt");
  const originalManifest = await readFile(manifest, "utf8");
  await writeFile(manifest, originalManifest.replace("build_type=Release", "build_type=Debug"));
  await assert.rejects(bindMetallib(fixture), /release Metal flavor/);
  await writeFile(manifest, originalManifest);
  const other = path.join(fixture.temporaryRoot, "wrong-flavor.metallib"); await writeFile(other, "wrong native bytes");
  await assert.rejects(bindMetallib({ ...fixture, explicitOverride: other }), /override mismatches/);
  await rm(path.join(path.dirname(fixture.source), "pmetal-mlx-prebuilt.txt"));
  await assert.rejects(bindMetallib(fixture), /ENOENT/);
});
for (const consumer of ["iris", "supra2"]) test(`selected ${consumer} no-run executable and shader stage privately; other-SHA cache survives and substitutions refuse`, async (t) => {
  const fixture = await nativeFixture(t, false, "release", consumer);
  const old = path.join(fixture.temporaryRoot, "other-source-sha", "active-binary");
  await mkdir(path.dirname(old)); await writeFile(old, "another active source cache");
  const native = await bindMetallib(fixture);
  assert.notEqual(native.executable, fixture.records[1].executable);
  assert.equal(await readFile(native.executable, "utf8"), await readFile(fixture.records[1].executable, "utf8"));
  assert.equal(await readFile(old, "utf8"), "another active source cache");
  await assert.rejects(bindMetallib({ ...fixture, copy: async (from, to, flags) => {
    await copyFile(from, to, flags);
    if (from === fixture.records[1].executable) await writeFile(from, "concurrent replacement");
  } }), /executable changed during copy/);
  assert.equal(await readFile(old, "utf8"), "another active source cache");
  await assert.rejects(bindMetallib({ ...fixture, copy: async (from, to, flags) => {
    await copyFile(from, to, flags);
    if (from === fixture.source) await writeFile(from, "concurrent shader replacement");
  } }), /metallib changed during binding/);
  const iris = await readFile(path.join(root, `scripts/ci/${consumer}-mlx-real-weights/run.sh`), "utf8");
  assert.doesNotMatch(iris, /-mindepth 1 -maxdepth 1 -type d ! -name/);
  assert.ok(iris.includes(`--${consumer}-test --print-path`));
  assert.match(iris, /bin="\$\(dirname "\$PMETAL_METALLIB_PATH"\)\/sceneworks_worker"/);
});
test("app start/reload uses the same binary-bound private shader without changing service-state schema", async (t) => {
  const fixture = await nativeFixture(t, false, "release", "app");
  const native = await bindMetallib(fixture), record = JSON.parse(await readFile(native.receipt, "utf8"));
  const expected = { receipt: native.receipt, lockFile: fixture.lockFile, executable: record.executable, executableSha256: record.executableSha256 };
  assert.equal(await readBoundAppMetallib(expected), native.path);
  await rm(fixture.source);
  assert.equal(await readBoundAppMetallib(expected), native.path, "reload never borrows a later shared cache file");
  const wrong = path.join(fixture.temporaryRoot, "wrong-explicit.metallib"); await writeFile(wrong, "different shader");
  await assert.rejects(readBoundAppMetallib({ ...expected, explicitOverride: wrong }), /override mismatches/);
  await writeFile(native.path, "changed owned shader");
  await assert.rejects(readBoundAppMetallib(expected), /metallib bytes changed/);
});

function assertCampaignSources(workflow, inference, launcher) {
  assert.match(workflow, /CARGO_TARGET_DIR: \$\{\{ github.workspace \}\}\/\.\.\/cargo-target/);
  assert.match(workflow, /ADAPTER=%GITHUB_WORKSPACE%\\target\\release\\memory-candle-adapter\.exe/);
  assert.match(workflow, /--message-format=json-render-diagnostics[^\n]*[\s\S]*bind-metallib\.mjs/);
  assert.doesNotMatch(inference, /checkout --force|reset --hard|clean -ffdx|rev-parse FETCH_HEAD|rm -rf/);
  assert.match(inference, /source-worktree\.mjs prepare/);
  assert.match(inference, /INFERENCE_REPO=\$\(native_path "\$WORKTREE_U"\)/);
  assert.ok(launcher.indexOf('wait "$WALK_PID"') < launcher.indexOf('source-worktree.mjs cleanup'));
}
test("per-run ready keys are backed by exact source isolation and scheduled-job-private native artifacts", async () => {
  const workflow = await readFile(path.join(root, ".github/workflows/memory-catalog-campaign.yml"), "utf8");
  const inference = await readFile(path.join(root, "scripts/ci/memory-catalog/inference.sh"), "utf8");
  const launcher = await readFile(path.join(root, "scripts/ci/memory-catalog/run.sh"), "utf8");
  assertCampaignSources(workflow, inference, launcher);
  assert.throws(() => assertCampaignSources(workflow, `${inference}\ngit reset --hard PIN`, launcher));
  assert.throws(() => assertCampaignSources(workflow.replace("bind-metallib.mjs", "unbound.mjs"), inference, launcher));
});
