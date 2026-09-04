import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { chmod, lstat, mkdir, mkdtemp, readFile, realpath, rename, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";
import { promisify } from "node:util";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";
import { removeStarVectorMacMetricsTree, selectStarVectorMacPython, validateStarVectorMacVenv } from "./select-starvector-macos-python.mjs";
import { assemblePreflight, assembleWeights, downloadExact, installCheckout, tree } from "./starvector-terminal-provision.mjs";
import { validateTerminalServiceClosure } from "./starvector-terminal-readiness.mjs";

const execFile = promisify(execFileCallback);
const digest = (value) => createHash("sha256").update(value).digest("hex");
const workflow = await readFile(".github/workflows/starvector-terminal-provision.yml", "utf8");
const windowsWorkflow = await readFile(".github/workflows/desktop-windows.yml", "utf8");
const windowsWheelWorkflow = await readFile(".github/workflows/starvector-metrics-windows.yml", "utf8");
const macosPythonSelector = await readFile("scripts/select-starvector-macos-python.mjs", "utf8");
const windowsPythonProbe = await readFile("scripts/select-starvector-windows-python.ps1", "utf8");
const windowsPythonProbeTest = await readFile("scripts/select-starvector-windows-python.test.ps1", "utf8");
const windowsWheelVerification = await readFile("scripts/verify-starvector-windows-metric-wheels.ps1", "utf8");

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
  assert.match(step, /\$bootstrap = Select-StarVectorBootstrapPython -CandidatePaths @\(\$env:STARVECTOR_METRICS_BOOTSTRAP_PYTHON\)/);
  assert.match(step, /\$bootstrapPython = \$bootstrap\.Executable/);
  assert.match(step, /\$bootstrapIdentity = \$bootstrap\.Identity/);
  assert.doesNotMatch(step, /Get-Command python\.exe/);
  assert.doesNotMatch(step, /& \$candidate/);
  assert.match(step, /\$rebuildMetricsVenv = \$existingProbe\.ExitCode -ne 0/);
  assert.match(step, /OrdinalIgnoreCase\.Equals\(\$bootstrapPython, \$existingBasePython\)/);
  assert.match(step, /\[int\]\$existingIdentity\.version\[0\] -ne \[int\]\$bootstrapIdentity\.version\[0\]/);
  assert.match(step, /\[int\]\$existingIdentity\.version\[1\] -ne \[int\]\$bootstrapIdentity\.version\[1\]/);
  assert.match(step, /\[int\]\$existingIdentity\.version\[2\] -ne \[int\]\$bootstrapIdentity\.version\[2\]/);
  assert.match(step, /if \(\$rebuildMetricsVenv\) \{\s+Remove-StarVectorWindowsDirectoryTree -TargetRoot \$metricsRoot -AllowedRoot 'D:\\sceneworks-terminal\\metrics'/);
  assert.doesNotMatch(step, /Remove-Item[^\n]*-Recurse/);
  assert.match(step, /& \$bootstrapPython -m venv \$metricsRoot/);
  assert.match(step, /if \(\$LASTEXITCODE -ne 0\) \{ throw 'failed to create the terminal metrics venv/);
  assert.match(step, /Test-Path \$metricsPython -PathType Leaf/);
  assert.match(step, /\$venvProbe = Invoke-StarVectorPythonIdentityProbe -Executable \$metricsPython -IncludeBaseExecutable/);
  assert.match(step, /if \(\$venvProbe\.ExitCode -ne 0\) \{ throw 'terminal metrics venv Python identity probe failed/);
  assert.match(step, /\$venvProbe\.StdOut \| ConvertFrom-Json -ErrorAction Stop/);
  assert.match(step, /\$observedBasePython = Resolve-StarVectorWindowsExecutable \$venvIdentity\.base_executable/);
  assert.match(step, /OrdinalIgnoreCase\.Equals\(\$bootstrapPython, \$observedBasePython\)/);
  assert.match(step, /\[int\]\$venvIdentity\.version\[2\] -ne \[int\]\$bootstrapIdentity\.version\[2\]/);
  assert.match(step, /\[string\]\$venvIdentity\.implementation -cne 'CPython'/);
  assert.match(step, /\[string\]\$venvIdentity\.architecture -cne 'AMD64'/);
  assert.match(step, /\[int\]\$venvIdentity\.pointer_bits -ne 64/);
  assert.match(step, /& \$metricsPython -m pip install --disable-pip-version-check --only-binary=:all: --retries 5 --timeout 60/);
  assert.match(step, /starvector-terminal-provision\.mjs metrics[^\n]*\$metricsRoot \$metricsPython/);
}

function assertWindowsPythonProbeContract(source) {
  assert.doesNotMatch(source, /Get-Command python\.exe/);
  assert.match(source, /\[Parameter\(Mandatory = \$true\)\]/);
  assert.ok(source.includes("[regex]::Replace($Name.Trim().ToLowerInvariant(), '[-_.]+', '-')"));
  assert.match(source, /function Remove-StarVectorWindowsDirectoryTree/);
  assert.match(source, /OrdinalIgnoreCase\.Equals\(\$canonicalTarget, \$canonicalAllowed\)/);
  assert.match(source, /New-Object System\.Collections\.Stack/);
  assert.equal((source.match(/Attributes -band \[IO\.FileAttributes\]::ReparsePoint/g) ?? []).length, 2);
  assert.match(source, /Get-ChildItem -LiteralPath \$item\.FullName -Force -ErrorAction Stop/);
  assert.match(source, /Remove-Item -LiteralPath \$item\.FullName -Force -ErrorAction Stop/);
  assert.doesNotMatch(source, /Remove-Item[^\n]*-Recurse/);
  assert.match(source, /refusing to remove a metrics venv outside the exact workflow-owned terminal root/);
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
  assert.match(source, /platform\.python_implementation\(\)/);
  assert.match(source, /platform\.machine\(\)/);
  assert.match(source, /struct\.calcsize\(''P''\)\*8/);
  assert.match(source, /if \(\$probe\.ExitCode -ne 0\) \{ continue \}/);
  assert.match(source, /\$probe\.StdOut \| ConvertFrom-Json -ErrorAction Stop/);
  assert.match(source, /\[int\]\$identity\.version\[1\] -eq 12/);
  assert.doesNotMatch(source, /\[int\]\$identity\.version\[1\] -ge 12/);
  assert.match(source, /\[string\]\$identity\.implementation -ceq 'CPython'/);
  assert.match(source, /\[string\]\$identity\.architecture -ceq 'AMD64'/);
  assert.match(source, /\[int\]\$identity\.pointer_bits -eq 64/);
  assert.match(source, /OrdinalIgnoreCase\.Equals\(\$canonicalCandidate, \$canonicalExecutable\)/);
  assert.doesNotMatch(source, /& \$candidate|Invoke-Expression|Start-Process/);
}

test("terminal metrics provisioning fixes CPython 3.12 and wheel-only installation on both hosts", () => {
  assert.equal((workflow.match(/uses: actions\/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97/g) ?? []).length, 1);
  assert.equal((workflow.match(/python-version: "3\.12"/g) ?? []).length, 1);
  assert.equal((workflow.match(/update-environment: false/g) ?? []).length, 1);
  assert.match(workflow, /Select supported existing CPython 3\.12 arm64 for terminal metrics/);
  assert.match(workflow, /select-starvector-macos-python\.mjs select \/opt\/homebrew\/bin\/python3\.12 \/usr\/local\/bin\/python3\.12/);
  assert.match(workflow, /select-starvector-macos-python\.mjs verify-venv "\$STARVECTOR_METRICS_BOOTSTRAP_PYTHON" "\$STARVECTOR_TERMINAL_ROOT\/metrics\/bin\/python"/);
  assert.match(workflow, /remove-metrics-tree "\$STARVECTOR_TERMINAL_ROOT\/metrics" \/Users\/Shared\/SceneWorks\/starvector-terminal\/metrics/);
  assert.doesNotMatch(workflow.slice(workflow.indexOf("  provision-macos:"), workflow.indexOf("  provision-windows:")), /actions\/setup-python|RUNNER_TOOL_CACHE|AGENT_TOOLSDIRECTORY|\/Users\/runner\/hostedtoolcache/);
  assert.match(workflow, /Set up supported CPython 3\.12 x64 for terminal metrics[\s\S]*?architecture: x64/);
  assert.equal((workflow.match(/pip install --disable-pip-version-check --only-binary=:all: --retries 5 --timeout 60/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /pip install --disable-pip-version-check (?!.*--only-binary=:all:)/);
  assert.match(macosPythonSelector, /identity\.version\[1\] === 12/);
  assert.match(macosPythonSelector, /identity\.implementation === "CPython"/);
  assert.match(macosPythonSelector, /\["arm64", "aarch64"\]\.includes\(identity\.architecture\.toLowerCase\(\)\)/);
  assert.match(macosPythonSelector, /identity\.pointer_bits === 64/);

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

test("macOS Python selection ignores unwritable action caches and verifies the exact venv base and micro", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-macos-python-"));
  const writeFakePython = async (name, identity, exitCode = 0) => {
    const executable = path.join(root, name);
    const payload = JSON.stringify({ executable, base_executable: identity.base_executable ?? executable, ...identity });
    await writeFile(executable, `#!/bin/sh\nprintf '%s\\n' '${payload}'\nexit ${exitCode}\n`);
    await chmod(executable, 0o755);
    return executable;
  };
  const invalid = await writeFakePython("python-invalid", { version: [3, 14, 1], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  const bootstrap = await writeFakePython("python-valid", { version: [3, 12, 13], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  const venv = await writeFakePython("venv-python", { base_executable: bootstrap, version: [3, 12, 13], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  const previousRunnerCache = process.env.RUNNER_TOOL_CACHE;
  const previousAgentTools = process.env.AGENT_TOOLSDIRECTORY;
  process.env.RUNNER_TOOL_CACHE = "/root/unwritable-hostedtoolcache";
  process.env.AGENT_TOOLSDIRECTORY = "/root/unwritable-agent-tools";
  try {
    assert.equal(selectStarVectorMacPython([invalid, bootstrap]).executable, await realpath(bootstrap));
    assert.equal(validateStarVectorMacVenv(bootstrap, venv).venv.identity.version[2], 13);
  } finally {
    if (previousRunnerCache === undefined) delete process.env.RUNNER_TOOL_CACHE; else process.env.RUNNER_TOOL_CACHE = previousRunnerCache;
    if (previousAgentTools === undefined) delete process.env.AGENT_TOOLSDIRECTORY; else process.env.AGENT_TOOLSDIRECTORY = previousAgentTools;
  }
  const wrongMicro = await writeFakePython("venv-wrong-micro", { base_executable: bootstrap, version: [3, 12, 12], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  assert.throws(() => validateStarVectorMacVenv(bootstrap, wrongMicro), /selected exact CPython/);
  await chmod(bootstrap, 0o644);
  assert.throws(() => selectStarVectorMacPython([bootstrap]), /existing absolute CPython/);
});

test("macOS stale metrics cleanup is exact-root bounded and refuses symlink escape", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-macos-metrics-"));
  const metrics = path.join(root, "metrics");
  const outside = path.join(root, "outside");
  await mkdir(path.join(metrics, "nested"), { recursive: true });
  await mkdir(outside, { recursive: true });
  await writeFile(path.join(metrics, "nested", "inside.txt"), "inside");
  await writeFile(path.join(outside, "sentinel.txt"), "outside");
  assert.throws(() => removeStarVectorMacMetricsTree(metrics, path.join(root, "wrong")), /exact workflow-owned terminal root/);
  await symlink(outside, path.join(metrics, "escape"));
  assert.throws(() => removeStarVectorMacMetricsTree(metrics, metrics), /containing a symlink/);
  assert.equal(await readFile(path.join(outside, "sentinel.txt"), "utf8"), "outside");
  await rename(path.join(metrics, "escape"), path.join(root, "detached-symlink"));
  removeStarVectorMacMetricsTree(metrics, metrics);
  assert.equal(await lstat(metrics).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  assert.equal(await readFile(path.join(outside, "sentinel.txt"), "utf8"), "outside");
});

test("Windows Python probes capture native stderr without invoking through PowerShell", () => {
  assertWindowsPythonProbeContract(windowsPythonProbe);
  assert.match(windowsPythonProbeTest, /\$ErrorActionPreference = 'Stop'/);
  assert.match(windowsPythonProbeTest, /01-bad python\.exe/);
  assert.match(windowsPythonProbeTest, /02-malformed python\.exe/);
  assert.match(windowsPythonProbeTest, /03-python311\.exe/);
  assert.match(windowsPythonProbeTest, /04-python314\.exe/);
  assert.match(windowsPythonProbeTest, /05-python312-arm64\.exe/);
  assert.match(windowsPythonProbeTest, /06-python312-amd64\.exe/);
  assert.match(windowsPythonProbeTest, /Traceback from the first fake Python candidate/);
  assert.match(windowsPythonProbeTest, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$bad, \$malformed, \$belowMinimum, \$newerUnsupported, \$wrongArchitecture,/);
  assert.match(windowsPythonProbeTest, /all invalid candidates must fail closed/);
  assert.match(windowsPythonProbeTest, /ConvertTo-StarVectorPythonDistributionName 'Open\.CLIP_Torch'/);
  assert.match(windowsPythonProbeTest, /New-Item -ItemType Junction -Path \$junction -Target \$outside/);
  assert.match(windowsPythonProbeTest, /junction refusal must preserve the outside sentinel/);
  assert.match(windowsPythonProbeTest, /directory deletion must refuse a root junction/);
  assert.match(windowsPythonProbeTest, /a normal metrics tree must be removed after validation/);
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
    ["permissive Python minor", (value) => value.replace("[int]$identity.version[1] -eq 12", "[int]$identity.version[1] -ge 12")],
    ["implementation identity", (value) => value.replace("[string]$identity.implementation -ceq 'CPython'", "$true")],
    ["architecture identity", (value) => value.replace("[string]$identity.architecture -ceq 'AMD64'", "$true")],
    ["pointer width", (value) => value.replace("[int]$identity.pointer_bits -eq 64", "$true")],
    ["reported executable equality", (value) => value.replace("[StringComparer]::OrdinalIgnoreCase.Equals($canonicalCandidate, $canonicalExecutable)", "$true")],
    ["distribution name separator folding", (value) => value.replace("'[-_.]+'", "'[-_]+'")],
    ["exact deletion root", (value) => value.replace("[StringComparer]::OrdinalIgnoreCase.Equals($canonicalTarget, $canonicalAllowed)", "$true")],
    ["reparse-point refusal", (value) => value.replace("($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0", "$false")],
    ["non-recursive removal", (value) => value.replace("Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop", "Remove-Item -LiteralPath $item.FullName -Recurse -Force -ErrorAction Stop")],
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
    ["selection helper", (value) => value.replace("$bootstrap = Select-StarVectorBootstrapPython -CandidatePaths @($env:STARVECTOR_METRICS_BOOTSTRAP_PYTHON)", "$bootstrap = Get-Command python.exe")],
    ["venv captured probe", (value) => value.replace("$venvProbe = Invoke-StarVectorPythonIdentityProbe -Executable $metricsPython -IncludeBaseExecutable", "$venvProbe = & $metricsPython -c $code")],
    ["base path equality", (value) => value.replace("OrdinalIgnoreCase.Equals($bootstrapPython, $observedBasePython)", "OrdinalIgnoreCase.Equals($bootstrapPython, $bootstrapPython)")],
    ["patch version equality", (value) => value.replace("[int]$venvIdentity.version[2] -ne [int]$bootstrapIdentity.version[2]", "[int]$venvIdentity.version[2] -ne [int]$venvIdentity.version[2]")],
    ["stale rebuild gate", (value) => value.replace("if ($rebuildMetricsVenv)", "if ($false)")],
    ["stale exact root", (value) => value.replace("-AllowedRoot 'D:\\sceneworks-terminal\\metrics'", "-AllowedRoot $metricsRoot")],
    ["stale base mismatch", (value) => value.replace("OrdinalIgnoreCase.Equals($bootstrapPython, $existingBasePython)", "OrdinalIgnoreCase.Equals($bootstrapPython, $bootstrapPython)")],
    ["stale major mismatch", (value) => value.replace("[int]$existingIdentity.version[0] -ne [int]$bootstrapIdentity.version[0]", "$false")],
    ["stale minor mismatch", (value) => value.replace("[int]$existingIdentity.version[1] -ne [int]$bootstrapIdentity.version[1]", "$false")],
    ["stale micro mismatch", (value) => value.replace("[int]$existingIdentity.version[2] -ne [int]$bootstrapIdentity.version[2]", "$false")],
    ["wheel-only install", (value) => value.replace("--only-binary=:all:", "--prefer-binary")],
    ["stale venv replacement", (value) => value.replace("Remove-StarVectorWindowsDirectoryTree -TargetRoot $metricsRoot", "Write-Host $metricsRoot")],
  ]) {
    const changed = mutation(step);
    assert.notEqual(changed, step, `${label} mutation must alter the workflow fixture`);
    assert.throws(() => assertWindowsMetricsPythonContract(changed), { name: "AssertionError" }, label);
  }
});

function assertHostedWindowsWheelContract(hostedWorkflow, verification) {
  assert.match(hostedWorkflow, /runs-on: windows-2022/);
  assert.match(hostedWorkflow, /timeout-minutes: 25/);
  assert.match(hostedWorkflow, /uses: actions\/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97/);
  assert.match(hostedWorkflow, /python-version: "3\.12"[\s\S]*?architecture: x64[\s\S]*?update-environment: false/);
  assert.equal((hostedWorkflow.match(/shell: powershell/g) ?? []).length, 2);
  assert.match(hostedWorkflow, /select-starvector-windows-python\.test\.ps1/);
  assert.match(hostedWorkflow, /"scripts\/starvector-terminal-metrics\.py"/);
  assert.match(hostedWorkflow, /STARVECTOR_METRICS_BOOTSTRAP_PYTHON: \$\{\{ steps\.metrics-python\.outputs\.python-path \}\}/);
  assert.match(hostedWorkflow, /verify-starvector-windows-metric-wheels\.ps1/);
  assert.match(verification, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$BootstrapPython\)/);
  assert.match(verification, /'pip', 'download'[\s\S]*?'--only-binary=:all:'[\s\S]*?'--retries', '5'/);
  assert.match(verification, /Where-Object \{ \$_\.Extension -cne '\.whl' \}/);
  assert.match(verification, /'pip', 'install'[\s\S]*?'--no-index'[\s\S]*?'--find-links'[\s\S]*?'--only-binary=:all:'/);
  assert.match(verification, /\$seen\.ContainsKey\(\$canonicalName\)/);
  assert.match(verification, /importlib\.metadata\.distribution\(sys\.argv\[1\]\);print\(d\.name\);print\(d\.version\)/);
  assert.doesNotMatch(verification, /json\.dumps/);
  assert.match(verification, /\$observedIdentity\.Count -ne 2/);
  assert.match(verification, /\$observedCanonicalName = ConvertTo-StarVectorPythonDistributionName/);
  assert.match(verification, /\$observedCanonicalName -cne \[string\]\$package\.canonical_name/);
  assert.match(verification, /\[string\]\$observedIdentity\[1\] -cne \[string\]\$package\.version/);
  assert.match(verification, /import PIL,lpips,numpy,open_clip,skimage,torch,torchvision/);
}

test("hosted Windows resolves and installs the metric lock exclusively from wheels under PowerShell 5.1", () => {
  assertHostedWindowsWheelContract(windowsWheelWorkflow, windowsWheelVerification);
});

test("hosted Windows wheel contract rejects source-distribution and ambient-interpreter mutations", () => {
  for (const [label, workflowMutation, scriptMutation] of [
    ["ambient interpreter", (value) => value.replace("${{ steps.metrics-python.outputs.python-path }}", "python.exe"), (value) => value],
    ["floating minor", (value) => value.replace('python-version: "3.12"', 'python-version: ">=3.12"'), (value) => value],
    ["import surface trigger", (value) => value.replace('      - "scripts/starvector-terminal-metrics.py"\n', ''), (value) => value],
    ["source download", (value) => value, (value) => value.replace("'--only-binary=:all:', '--retries'", "'--prefer-binary', '--retries'")],
    ["online install", (value) => value, (value) => value.replace("'--no-index', '--find-links'", "'--index-url', 'https://pypi.org/simple', '--find-links'")],
    ["source archive acceptance", (value) => value, (value) => value.replace("$_.Extension -cne '.whl'", "$false")],
    ["canonical duplicate detection", (value) => value, (value) => value.replace("$seen.ContainsKey($canonicalName)", "$seen.ContainsKey($name)")],
    ["JSON native-argument quoting", (value) => value, (value) => value.replace("print(d.name);print(d.version)", 'print(json.dumps({"name":d.name,"version":d.version}))')],
    ["metadata line count", (value) => value, (value) => value.replace("$observedIdentity.Count -ne 2", "$false")],
    ["observed canonical identity", (value) => value, (value) => value.replace("$observedCanonicalName -cne [string]$package.canonical_name", "$false")],
    ["lpips import", (value) => value, (value) => value.replace(",lpips", "")],
    ["open_clip import", (value) => value, (value) => value.replace(",open_clip", "")],
  ]) {
    const changedWorkflow = workflowMutation(windowsWheelWorkflow);
    const changedScript = scriptMutation(windowsWheelVerification);
    assert.ok(changedWorkflow !== windowsWheelWorkflow || changedScript !== windowsWheelVerification, `${label} mutation must alter a fixture`);
    assert.throws(() => assertHostedWindowsWheelContract(changedWorkflow, changedScript), { name: "AssertionError" }, label);
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
