#!/usr/bin/env node
// Metadata/file copy only. Select this successful Cargo invocation's actual mlx-sys output,
// never a global fallback or newest directory. Does not run Cargo, a model, or a device probe.
import { readFile, writeFile, realpath, mkdtemp, copyFile, stat, appendFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const inside = (root, file) => { const rel = path.relative(root, file); return rel !== "" && rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel); };
async function nonemptyFile(file) {
  const info = await stat(file);
  if (!info.isFile() || info.size <= 0) throw new Error(`selected native artifact is not a nonempty file: ${file}`);
}
export async function bindMetallib({ messages, lockFile, targetRoot, temporaryRoot, explicitOverride = "", explicitNativeDirectory = "", profile = "release", consumer = profile === "debug" ? "worker" : "adapter", copy = copyFile }) {
  const consumers = { adapter: ["release", "memory-mlx-adapter", false], worker: ["debug", "dump-engine-capabilities", false], app: ["release", "sceneworks-rust-api", false], iris: ["release", "sceneworks_worker", true] };
  if (!consumers[consumer] || consumers[consumer][0] !== profile) throw new Error("matching explicit consumer/profile required");
  const [, artifactName, testArtifact] = consumers[consumer];
  const buildType = profile === "release" ? "Release" : "Debug";
  const lock = await readFile(lockFile, "utf8");
  const identity = lock.match(/name = "pmetal-mlx-sys"\nversion = "([^"]+)"\nsource = "git\+[^\n]+\?rev=([0-9a-f]{40})#([0-9a-f]{40})"/u);
  const [, version, pin, nativeGitRevision] = identity ?? [];
  if (!pin || pin !== nativeGitRevision) throw new Error("Cargo.lock has no matching exact pmetal-mlx-sys requested/native Git revision");
  const selected = new Map(), artifacts = new Set();
  let completed = false;
  for (const line of (await readFile(messages, "utf8")).split(/\r?\n/)) {
    if (!line.startsWith("{")) continue;
    const message = JSON.parse(line);
    if (message.reason === "build-script-executed" && message.package_id?.endsWith(`?rev=${pin}#pmetal-mlx-sys@${version}`)) selected.set(message.out_dir, message);
    if (message.reason === "compiler-artifact" && message.target?.name === artifactName && message.executable
        && message.profile?.debug_assertions === (profile === "debug") && message.profile?.test === testArtifact) artifacts.add(message.executable);
    if (message.reason === "build-finished") completed = message.success === true;
  }
  if (!completed || selected.size !== 1 || artifacts.size !== 1) throw new Error(`successful ${profile} build did not identify exactly one pinned mlx-sys output and executable`);
  const [selectedOutput, selectedMessage] = [...selected][0];
  const target = await realpath(targetRoot), output = await realpath(selectedOutput);
  const executable = await realpath([...artifacts][0]);
  const expectedExecutable = testArtifact
    ? path.dirname(executable) === await realpath(path.join(target, profile, "deps")) && /^sceneworks_worker-[0-9a-f]+$/.test(path.basename(executable))
    : executable === await realpath(path.join(target, profile, artifactName));
  if (!inside(target, output) || path.basename(path.dirname(path.dirname(path.dirname(output)))) !== profile
      || !expectedExecutable) throw new Error(`mlx-sys output/executable is not in the selected ${profile} Cargo target`);
  const scriptOutput = await readFile(path.join(path.dirname(output), "output"), "utf8");
  const candidates = [...new Set(scriptOutput.split(/\r?\n/).filter((line) => line.startsWith("cargo:metallib=")).map((line) => line.slice("cargo:metallib=".length)))];
  if (candidates.length !== 1) throw new Error("selected mlx-sys build did not publish one exact metallib; refusing global-cache fallback");
  const source = await realpath(candidates[0]), nativeDirectory = path.dirname(source);
  await nonemptyFile(source);
  const linkedDirectories = await Promise.all((selectedMessage.linked_paths ?? []).filter((entry) => entry.startsWith("native=")).map((entry) => realpath(entry.slice(7))));
  if (!linkedDirectories.includes(nativeDirectory)) throw new Error("metallib is not in this invocation's selected native link directory; refusing shared fallback");
  const manifestPath = path.join(nativeDirectory, "pmetal-mlx-prebuilt.txt");
  const manifestBytes = await readFile(manifestPath);
  const flavor = Object.fromEntries(manifestBytes.toString().split(/\r?\n/).filter((line) => line.includes("=")).map((line) => [line.slice(0, line.indexOf("=")), line.slice(line.indexOf("=") + 1)]));
  if (flavor.schema !== "1" || flavor.build_type !== buildType || !flavor.target?.endsWith("-apple-darwin")
      || !flavor.features?.split(",").includes("metal") || !/^[0-9a-f]{64}$/.test(flavor.fingerprint ?? "")) throw new Error(`selected native manifest does not match the ${profile} Metal flavor`);
  const archives = {};
  for (const name of ["libmlx.a", "libmlxc.a"]) {
    const file = path.join(nativeDirectory, name); await nonemptyFile(file);
    if (profile === "debug") archives[name] = digest(await readFile(file));
  }
  if (explicitNativeDirectory && await realpath(explicitNativeDirectory) !== nativeDirectory) throw new Error("explicit native directory mismatches the actual selected build");
  const before = digest(await readFile(source));
  if (explicitOverride && digest(await readFile(await realpath(explicitOverride))) !== before) throw new Error("explicit metallib override mismatches this pin/flavor build; refusing substitution");
  const owned = await mkdtemp(path.join(await realpath(temporaryRoot), "memory-catalog-metallib-"));
  const bound = path.join(owned, "mlx.metallib");
  await copy(source, bound);
  const hash = digest(await readFile(bound));
  if (hash !== before || digest(await readFile(source)) !== before) throw new Error("selected metallib changed during binding; no capture may start");
  const executableHash = digest(await readFile(executable));
  let executionPath = executable;
  if (testArtifact) {
    executionPath = path.join(owned, artifactName);
    await copy(executable, executionPath, constants.COPYFILE_FICLONE);
    if (digest(await readFile(executionPath)) !== executableHash || digest(await readFile(executable)) !== executableHash) throw new Error("selected test executable changed during copy; no model may start");
  }
  const receipt = path.join(owned, "binding.json");
  await writeFile(receipt, JSON.stringify({
    schema: 1, pin, nativeGitRevision, packageId: selectedMessage.package_id, profile, consumer,
    source, explicitOverride: explicitOverride || null, nativeDirectory, nativeArchives: archives,
    buildOutput: output, nativeManifest: manifestPath, nativeManifestSha256: digest(manifestBytes), flavor,
    path: bound, sha256: hash, executable, executionPath, executableSha256: executableHash,
    nativeReuse: profile === "debug" ? "existing genuine archives/manifest; the next test build must verify its own full native key through PMETAL_MLX_PREBUILT_DIR; no build/test OUT_DIR equivalence claim" : null,
    schedulingScope: "runner-service/workspace-private target; one scheduled job per runner service, not universal exclusion from user-local builds",
    evidence: "source/build metadata and file identity only; no model or raw-kernel execution",
  }, null, 2) + "\n");
  return { path: bound, pin, sha256: hash, receipt, nativeDirectory, profile, executable: executionPath };
}
export async function readBoundAppMetallib({ receipt, lockFile, executable, executableSha256, explicitOverride = "" }) {
  const record = JSON.parse(await readFile(receipt, "utf8")), lock = await readFile(lockFile, "utf8");
  const match = lock.match(/name = "pmetal-mlx-sys"\nversion = "([^"]+)"\nsource = "git\+[^\n]+\?rev=([0-9a-f]{40})#([0-9a-f]{40})"/u);
  if (!match || match[2] !== match[3] || record.pin !== match[2] || record.nativeGitRevision !== match[3]
      || record.consumer !== "app" || record.profile !== "release" || record.flavor?.build_type !== "Release"
      || record.executable !== await realpath(executable) || record.executableSha256 !== executableSha256
      || digest(await readFile(executable)) !== executableSha256) throw new Error("owned app native binding differs from binary/source provenance");
  const bound = await realpath(record.path);
  if (!path.basename(path.dirname(bound)).startsWith("memory-catalog-metallib-") || digest(await readFile(bound)) !== record.sha256) throw new Error("owned app metallib bytes changed");
  if (explicitOverride && digest(await readFile(explicitOverride)) !== record.sha256) throw new Error("explicit app metallib override mismatches the owned binding");
  return bound;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [messages, ...flags] = process.argv.slice(2);
  if (flags.some((flag) => !["--print-path", "--debug-worker", "--iris-test"].includes(flag)) || (flags.includes("--debug-worker") && flags.includes("--iris-test"))) throw new Error("expected one supported native consumer and optional --print-path");
  const profile = flags.includes("--debug-worker") ? "debug" : "release";
  const consumer = flags.includes("--iris-test") ? "iris" : profile === "debug" ? "worker" : "adapter";
  bindMetallib({ messages, lockFile: "Cargo.lock", targetRoot: process.env.CARGO_TARGET_DIR, temporaryRoot: process.env.RUNNER_TEMP,
    explicitOverride: process.env.PMETAL_METALLIB_PATH ?? "", explicitNativeDirectory: process.env.PMETAL_MLX_PREBUILT_DIR ?? "", profile, consumer })
    .then(async (binding) => {
      if ((!process.env.GITHUB_ENV && !flags.includes("--print-path")) || !process.env.WORK_DIR_NATIVE || /[\r\n]/.test(binding.path + binding.nativeDirectory)) throw new Error("workflow environment or explicit path handoff, plus evidence directory, required");
      await copyFile(binding.receipt, path.join(process.env.WORK_DIR_NATIVE, "native-metallib-binding.json"));
      if (process.env.GITHUB_ENV) await appendFile(process.env.GITHUB_ENV, `PMETAL_METALLIB_PATH=${binding.path}\n`);
      // The supported native verifier compares schema/fingerprint/target/deployment/features/build
      // type before linking. Keep the real manifest untouched; never invent a passing key.
      if (profile === "debug" && process.env.GITHUB_ENV) await appendFile(process.env.GITHUB_ENV, `PMETAL_MLX_PREBUILT_DIR=${binding.nativeDirectory}\n`);
      const message = `job-scoped ${profile} metallib ${binding.sha256} at exact mlx-rs ${binding.pin}`;
      if (flags.includes("--print-path")) { console.error(message); console.log(binding.path); }
      else console.log(message);
    }).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
