import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, rename, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";
import { promisify } from "node:util";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";
import { assemblePreflight, assembleWeights, downloadExact, installCheckout, tree } from "./starvector-terminal-provision.mjs";
import { validateTerminalServiceClosure } from "./starvector-terminal-readiness.mjs";

const execFile = promisify(execFileCallback);
const digest = (value) => createHash("sha256").update(value).digest("hex");
const workflow = await readFile(".github/workflows/starvector-terminal-provision.yml", "utf8");
const windowsWorkflow = await readFile(".github/workflows/desktop-windows.yml", "utf8");
const windowsPythonProbe = await readFile("scripts/select-starvector-windows-python.ps1", "utf8");
const windowsPythonProbeTest = await readFile("scripts/select-starvector-windows-python.test.ps1", "utf8");

test("file and tree identities stream exact bytes with bounded reads and portable ordering", async () => {
  const chunks = [Buffer.from("large-file-"), Buffer.from("identity")];
  let opened;
  const streamed = await fileSha256("virtual-4995740600-byte-shard", {
    openReadStream(file, options) {
      opened = { file, highWaterMark: options.highWaterMark };
      return Readable.from(chunks);
    },
  });
  assert.equal(streamed, digest(Buffer.concat(chunks)));
  assert.deepEqual(opened, { file: "virtual-4995740600-byte-shard", highWaterMark: 1024 * 1024 });

  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-tree-"));
  await mkdir(path.join(root, "nested"));
  await writeFile(path.join(root, "z.bin"), "z");
  await writeFile(path.join(root, "nested", "a.bin"), "a");
  const calls = [];
  const identity = await tree(root, root, async (file) => {
    calls.push(file);
    return digest(await readFile(file));
  });
  assert.deepEqual(identity.entries.map((entry) => entry.path), ["nested/a.bin", "z.bin"]);
  assert.deepEqual(identity.entries.map((entry) => entry.sha256), [digest("a"), digest("z")]);
  assert.equal(calls.length, 2);
  assert.equal(identity.aggregate_sha256, digest(JSON.stringify(identity.entries)));
});

test("streaming file identity propagates read failures", async () => {
  await assert.rejects(() => fileSha256("virtual-shard", {
    openReadStream() {
      return Readable.from((async function* () {
        yield Buffer.from("partial");
        throw new Error("stream read failed");
      })());
    },
  }), /stream read failed/);
});

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

function assertWindowsMetricsPythonContract(step) {
  assert.doesNotMatch(step, /\bpy\s+-3\.11\b/);
  assert.match(step, /select-starvector-windows-python\.ps1/);
  assert.match(step, /\$bootstrap = Select-StarVectorBootstrapPython/);
  assert.match(step, /\$bootstrapPython = \$bootstrap\.Executable/);
  assert.match(step, /\$bootstrapIdentity = \$bootstrap\.Identity/);
  assert.doesNotMatch(step, /& \$candidate/);
  assert.match(step, /& \$bootstrapPython -m venv \$metricsRoot/);
  assert.match(step, /if \(\$LASTEXITCODE -ne 0\) \{ throw 'failed to create the terminal metrics venv/);
  assert.match(step, /Test-Path \$metricsPython -PathType Leaf/);
  assert.match(step, /Invoke-StarVectorPythonIdentityProbe -Executable \$metricsPython -IncludeBaseExecutable/);
  assert.match(step, /if \(\$venvProbe\.ExitCode -ne 0\) \{ throw 'terminal metrics venv Python identity probe failed/);
  assert.match(step, /\$venvProbe\.StdOut \| ConvertFrom-Json -ErrorAction Stop/);
  assert.match(step, /\$observedBasePython = Resolve-StarVectorWindowsExecutable \$venvIdentity\.base_executable/);
  assert.match(step, /OrdinalIgnoreCase\.Equals\(\$bootstrapPython, \$observedBasePython\)/);
  assert.match(step, /\[int\]\$venvIdentity\.version\[2\] -ne \[int\]\$bootstrapIdentity\.version\[2\]/);
  assert.match(step, /& \$metricsPython -m pip install/);
  assert.match(step, /starvector-terminal-provision\.mjs metrics[^\n]*\$metricsRoot \$metricsPython/);
}

function assertWindowsPythonProbeContract(source) {
  assert.match(source, /Get-Command python\.exe -All -CommandType Application/);
  assert.match(source, /\$text -match '\[\\r\\n"\]'/);
  assert.match(source, /\$text -notmatch '\^\[A-Za-z\]:\\\\'/);
  assert.match(source, /\[IO\.Path\]::GetFullPath\(\$text\)/);
  assert.match(source, /Test-Path -LiteralPath \$full -PathType Leaf/);
  assert.match(source, /New-Object System\.Diagnostics\.ProcessStartInfo/);
  assert.match(source, /\$startInfo\.FileName = \$Executable/);
  assert.match(source, /\$startInfo\.UseShellExecute = \$false/);
  assert.match(source, /\$startInfo\.RedirectStandardOutput = \$true/);
  assert.match(source, /\$startInfo\.RedirectStandardError = \$true/);
  assert.match(source, /\$startInfo\.CreateNoWindow = \$true/);
  assert.match(source, /\$stdoutTask = \$process\.StandardOutput\.ReadToEndAsync\(\)/);
  assert.match(source, /\$stderrTask = \$process\.StandardError\.ReadToEndAsync\(\)/);
  assert.match(source, /\$process\.WaitForExit\(\)/);
  assert.match(source, /ExitCode = \$process\.ExitCode/);
  assert.match(source, /StdOut = \$stdoutTask\.Result/);
  assert.match(source, /StdErr = \$stderrTask\.Result/);
  assert.match(source, /base_executable'':getattr\(sys,''_base_executable'',None\)/);
  assert.match(source, /if \(\$probe\.ExitCode -ne 0\) \{ continue \}/);
  assert.match(source, /\$probe\.StdOut \| ConvertFrom-Json -ErrorAction Stop/);
  assert.match(source, /\[int\]\$identity\.version\[1\] -ge 12/);
  assert.doesNotMatch(source, /& \$candidate|Invoke-Expression|Start-Process/);
}

test("Windows metrics provisioning selects, uses, and verifies one explicit Python 3.12+ executable", () => {
  const start = workflow.indexOf("- name: Provision pinned metric runtime and official checkpoints", workflow.indexOf("  provision-windows:"));
  const end = workflow.indexOf("- name: Materialize the exact pinned 120-row corpus", start);
  const step = workflow.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assertWindowsMetricsPythonContract(step);
  const isSafeDriveAbsolute = (value) => /^[A-Za-z]:\\/.test(value) && !/[\r\n"]/.test(value);
  assert.equal(isSafeDriveAbsolute("C:\\Python312\\python.exe"), true);
  for (const rejected of ["python.exe", "C:python.exe", "\\python.exe", "\\\\server\\share\\python.exe", "C:/Python312/python.exe", "C:\\Python312\\py\nthon.exe", 'C:\\Python312\\py"thon.exe']) {
    assert.equal(isSafeDriveAbsolute(rejected), false, rejected);
  }
});

test("Windows Python probes capture native stderr without invoking through PowerShell", () => {
  assertWindowsPythonProbeContract(windowsPythonProbe);
  assert.match(windowsPythonProbeTest, /\$ErrorActionPreference = 'Stop'/);
  assert.match(windowsPythonProbeTest, /01-bad python\.exe/);
  assert.match(windowsPythonProbeTest, /02-malformed python\.exe/);
  assert.match(windowsPythonProbeTest, /03-valid python\.exe/);
  assert.match(windowsPythonProbeTest, /Traceback from the first fake Python candidate/);
  assert.match(windowsPythonProbeTest, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$bad, \$malformed,/);
  assert.match(windowsPythonProbeTest, /all invalid candidates must fail closed/);
  assert.match(windowsWorkflow, /"scripts\/select-starvector-windows-python\.ps1"/);
  assert.match(windowsWorkflow, /"scripts\/select-starvector-windows-python\.test\.ps1"/);
  assert.match(windowsWorkflow, /shell: powershell\s+run: \.\\scripts\\select-starvector-windows-python\.test\.ps1/);
});

test("Windows Python probe contract rejects native-process safety mutations", () => {
  for (const [label, mutation] of [
    ["shell execution", (value) => value.replace("$startInfo.UseShellExecute = $false", "$startInfo.UseShellExecute = $true")],
    ["stdout capture", (value) => value.replace("$startInfo.RedirectStandardOutput = $true", "$startInfo.RedirectStandardOutput = $false")],
    ["stderr capture", (value) => value.replace("$startInfo.RedirectStandardError = $true", "$startInfo.RedirectStandardError = $false")],
    ["candidate path binding", (value) => value.replace("$startInfo.FileName = $Executable", "$startInfo.FileName = 'python.exe'")],
    ["exit status", (value) => value.replace("ExitCode = $process.ExitCode", "ExitCode = 0")],
    ["bad-candidate continuation", (value) => value.replace("if ($probe.ExitCode -ne 0) { continue }", "if ($probe.ExitCode -ne 0) { break }")],
    ["PowerShell invocation", (value) => value.replace("$probe = Invoke-StarVectorPythonIdentityProbe -Executable $canonicalCandidate", "$probe = & $candidate -c $code")],
  ]) {
    const changed = mutation(windowsPythonProbe);
    assert.notEqual(changed, windowsPythonProbe, `${label} mutation must alter the helper fixture`);
    assert.throws(() => assertWindowsPythonProbeContract(changed), { name: "AssertionError" }, label);
  }
});

test("Windows metrics Python contract rejects path, base-interpreter, and patch-version guard mutations", () => {
  const start = workflow.indexOf("- name: Provision pinned metric runtime and official checkpoints", workflow.indexOf("  provision-windows:"));
  const end = workflow.indexOf("- name: Materialize the exact pinned 120-row corpus", start);
  const step = workflow.slice(start, end);
  for (const [label, mutation] of [
    ["selection helper", (value) => value.replace("$bootstrap = Select-StarVectorBootstrapPython", "$bootstrap = Get-Command python.exe")],
    ["venv captured probe", (value) => value.replace("$venvProbe = Invoke-StarVectorPythonIdentityProbe -Executable $metricsPython -IncludeBaseExecutable", "$venvProbe = & $metricsPython -c $code")],
    ["base path equality", (value) => value.replace("OrdinalIgnoreCase.Equals($bootstrapPython, $observedBasePython)", "OrdinalIgnoreCase.Equals($bootstrapPython, $bootstrapPython)")],
    ["patch version equality", (value) => value.replace("[int]$venvIdentity.version[2] -ne [int]$bootstrapIdentity.version[2]", "[int]$venvIdentity.version[2] -ne [int]$venvIdentity.version[2]")],
  ]) {
    const changed = mutation(step);
    assert.notEqual(changed, step, `${label} mutation must alter the workflow fixture`);
    assert.throws(() => assertWindowsMetricsPythonContract(changed), { name: "AssertionError" }, label);
  }
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
  const recovered = path.join(root, "recovered"), staleStaging = `${recovered}.staging-${process.pid}`;
  await mkdir(staleStaging, { recursive: true }); await writeFile(path.join(staleStaging, "partial"), "partial");
  await assemblePreflight(source, recovered, revision);
  assert.deepEqual(JSON.parse(await readFile(path.join(recovered, "starvector-terminal-preflight.json"))), index);
  assert.equal(await lstat(staleStaging).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  const incomplete = structuredClone(index); incomplete.hook_logs.pop(); await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(incomplete));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "incomplete"), revision), /cardinality/);
  const duplicate = structuredClone(index); duplicate.hook_logs[3].backend = "mlx"; await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(duplicate));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "duplicate"), revision), /identities are incomplete/);
  const provenance = structuredClone(index); provenance.workflow_run_attempt = 0; await writeFile(path.join(source, "starvector-terminal-preflight.json"), JSON.stringify(provenance));
  await assert.rejects(() => assemblePreflight(source, path.join(root, "provenance"), revision), /identity\/cardinality/);
});

test("weights assembly inventories fixed model roots and copies only source-produced service state", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-weights-")), hostRoot = path.join(root, "host"), sources = path.join(root, "sources");
  for (const [relative, file] of [["host/weights/models/starvector-1b", "one.bin"], ["host/weights/models/starvector-8b", "eight.bin"]]) { await mkdir(path.join(root, relative), { recursive: true }); await writeFile(path.join(root, relative, file), `${relative}-${file}`); }
  await mkdir(path.join(sources, "app", "models", "receipt"), { recursive: true });
  const revisions = { one: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", eight: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9", flux: "a".repeat(40) };
  const snapshot = (repo, revision) => path.join(sources, "hf", "hub", `models--${repo.replace("/", "--")}`, "snapshots", revision);
  for (const [repo, revision] of [["starvector/starvector-1b-im2svg", revisions.one], ["starvector/starvector-8b-im2svg", revisions.eight]]) { const dir = snapshot(repo, revision); await mkdir(dir, { recursive: true }); await writeFile(path.join(dir, "weights.bin"), `${repo}-weights`); }
  const fluxSnapshot = snapshot("SceneWorks/flux1-schnell-mlx", revisions.flux), fluxBlob = path.join(sources, "hf", "hub", "models--SceneWorks--flux1-schnell-mlx", "blobs", "flux");
  await mkdir(path.join(fluxSnapshot, "q4"), { recursive: true }); await mkdir(path.dirname(fluxBlob), { recursive: true }); await writeFile(fluxBlob, "flux-q4"); await symlink(path.relative(path.join(fluxSnapshot, "q4"), fluxBlob), path.join(fluxSnapshot, "q4", "weights.bin"));
  const receipt = (repo, modelId, snapshotRevision, extra = {}) => ({ schemaVersion: 2, repo, modelId, snapshotRevision, resolvedFiles: ["weights.bin"], ...extra });
  const fluxReceipt = receipt("SceneWorks/flux1-schnell-mlx", "flux_schnell", revisions.flux, { variant: "q4", resolvedTier: "q4", resolvedFiles: ["q4/weights.bin"] });
  const validReceipts = { ...fluxReceipt, receipts: [receipt("starvector/starvector-1b-im2svg", "starvector_1b", revisions.one), receipt("starvector/starvector-8b-im2svg", "starvector_8b", revisions.eight), fluxReceipt] };
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(validReceipts));
  const manifest = await assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptProvider: "candle_flux", promptModel: "flux_schnell", promptRevision: revisions.flux });
  assert.deepEqual(Object.keys(manifest.models).sort(), ["starvector-1b", "starvector-8b"]);
  assert.equal(manifest.prompt_raster.provider_id, "candle_flux");
  assert.match(manifest.terminal_service_closure.app_data_sha256, /^[a-f0-9]{64}$/);
  const weightsRoot = path.join(hostRoot, "weights");
  const readiness = await validateTerminalServiceClosure(weightsRoot, manifest);
  assert.equal(readiness.app_data.sha256, manifest.terminal_service_closure.app_data_sha256);
  assert.equal(readiness.hf_home.sha256, manifest.terminal_service_closure.hf_home_sha256);
  const assembledHf = await tree(path.join(weightsRoot, "service-closure", "hf-home"));
  assert.equal(assembledHf.aggregate_sha256, manifest.terminal_service_closure.hf_home_sha256);
  const first = assembledHf.entries[0];
  for (const mutation of [
    { ...first, path: `changed/${first.path}` },
    { ...first, byte_size: first.byte_size + 1 },
    { ...first, sha256: digest("changed") },
  ]) {
    const entries = assembledHf.entries.map((entry, index) => index === 0 ? mutation : entry);
    assert.notEqual(terminalTreeSha256(entries), assembledHf.aggregate_sha256);
  }
  assert.throws(() => terminalTreeSha256([...assembledHf.entries].reverse()), /uniquely sorted/);
  assert.throws(() => terminalTreeSha256(assembledHf.entries.map((entry) => [entry.path, entry.byte_size, entry.sha256])), /exactly \{path, byte_size, sha256\}/);
  assert.deepEqual(terminalTreeEntry(first.path, first.byte_size, first.sha256), first);
  const assembledFile = path.join(weightsRoot, "service-closure", "hf-home", ...first.path.split("/"));
  const originalBytes = await readFile(assembledFile), renamedFile = `${assembledFile}.renamed`;
  await rename(assembledFile, renamedFile);
  await assert.rejects(() => validateTerminalServiceClosure(weightsRoot, manifest), /service closure tree hash mismatch/);
  await rename(renamedFile, assembledFile);
  await writeFile(assembledFile, Buffer.concat([originalBytes, Buffer.from("size-drift")]));
  await assert.rejects(() => validateTerminalServiceClosure(weightsRoot, manifest), /service closure tree hash mismatch/);
  const hashDrift = Buffer.from(originalBytes); hashDrift[0] ^= 0xff;
  await writeFile(assembledFile, hashDrift);
  await assert.rejects(() => validateTerminalServiceClosure(weightsRoot, manifest), /service closure tree hash mismatch/);
  await writeFile(assembledFile, originalBytes);
  await validateTerminalServiceClosure(weightsRoot, manifest);
  await assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptProvider: "candle_flux", promptModel: "flux_schnell", promptRevision: revisions.flux });
  const missing = structuredClone(validReceipts); missing.receipts[2].resolvedFiles = ["q4/missing.bin"]; missing.resolvedFiles = ["q4/missing.bin"];
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(missing));
  await assert.rejects(() => assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptProvider: "candle_flux", promptModel: "flux_schnell", promptRevision: revisions.flux }), /HF closure lacks resolved file/);
  const wrongVariant = structuredClone(validReceipts); wrongVariant.variant = "q8"; wrongVariant.receipts[2].variant = "q8";
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(wrongVariant));
  await assert.rejects(() => assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptProvider: "candle_flux", promptModel: "flux_schnell", promptRevision: revisions.flux }), /lacks a source-produced receipt/);
  await writeFile(path.join(sources, "app", "models", "receipt", ".sceneworks-download-complete.json"), JSON.stringify(validReceipts));
  const outside = path.join(root, "outside.bin"); await writeFile(outside, "outside"); await symlink(outside, path.join(sources, "hf", "escape.bin"));
  await assert.rejects(() => assembleWeights({ hostRoot, serviceAppData: path.join(sources, "app"), serviceHfHome: path.join(sources, "hf"), promptProvider: "candle_flux", promptModel: "flux_schnell", promptRevision: revisions.flux }), /symlink escapes/);
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
