import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { chmod, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rename, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";
import { promisify } from "node:util";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { terminalPinPaths } from "./lib/starvector-terminal-pin-paths.mjs";
import { terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";
import { removeStarVectorMacMetricsTree, selectStarVectorMacPython, validateStarVectorMacVenv } from "./select-starvector-macos-python.mjs";
import { assemblePreflight, assembleWeights, downloadExact, installCheckout, installPinnedCheckout, installUpstreamPackages, prepareUpstreamSource, runUpstreamPip, upstreamPipProgress, pinnedCheckoutLockPath, tree, validatePreflightMetadata, validatePreflightTransport, validateSealedPreflightIndex } from "./starvector-terminal-provision.mjs";
import { validateTerminalServiceClosure } from "./starvector-terminal-readiness.mjs";

const execFile = promisify(execFileCallback);
const digest = (value) => createHash("sha256").update(value).digest("hex");
const workflow = await readFile(".github/workflows/starvector-terminal-provision.yml", "utf8");
const readinessWorkflow = await readFile(".github/workflows/starvector-terminal-readiness.yml", "utf8");
const campaignWorkflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");
const windowsWorkflow = await readFile(".github/workflows/desktop-windows.yml", "utf8");
const windowsWheelWorkflow = await readFile(".github/workflows/starvector-metrics-windows.yml", "utf8");
const macosPythonSelector = await readFile("scripts/select-starvector-macos-python.mjs", "utf8");
const windowsPythonProbe = await readFile("scripts/select-starvector-windows-python.ps1", "utf8");
const windowsPythonProbeTest = await readFile("scripts/select-starvector-windows-python.test.ps1", "utf8");
const windowsPythonProvision = await readFile("scripts/provision-starvector-windows-python.ps1", "utf8");
const windowsPythonProvisionTest = await readFile("scripts/provision-starvector-windows-python.test.ps1", "utf8");
const windowsWheelVerification = await readFile("scripts/verify-starvector-windows-metric-wheels.ps1", "utf8");
const instantIdWorker = await readFile("crates/sceneworks-worker/src/image_jobs/instantid.rs", "utf8");
const instantIdSmoke = await readFile("crates/sceneworks-worker/src/instantid_pid_gpu_smoke.rs", "utf8");
const terminalPlan = JSON.parse(await readFile("release/starvector-terminal-campaign-v1.json", "utf8"));

function workflowRunBodies(source) {
  const lines = source.split("\n");
  const bodies = [];
  for (let index = 0; index < lines.length; index += 1) {
    const block = lines[index].match(/^(\s*)run:\s*[|>]\s*$/);
    const inline = lines[index].match(/^(\s*)run:\s+(.+)$/);
    if (inline && !block) {
      bodies.push(inline[2]);
      continue;
    }
    if (!block) continue;
    const indent = block[1].length;
    const body = [];
    for (index += 1; index < lines.length; index += 1) {
      const line = lines[index];
      if (line.trim() && line.match(/^\s*/)[0].length <= indent) {
        index -= 1;
        break;
      }
      body.push(line);
    }
    bodies.push(body.join("\n"));
  }
  return bodies;
}

function assertNoDirectInputInterpolationInShell(source) {
  for (const body of workflowRunBodies(source)) assert.doesNotMatch(body, /\$\{\{\s*inputs\./);
}

function assertC6dCandleInstantIdPaths(workerSource, smokeSource) {
  assert.match(workerSource, /let paths = InstantIdPaths \{[\s\S]*?openpose: openpose\.clone\(\),[\s\S]*?face_dir: Some\([\s\S]*?scrfd_path[\s\S]*?\.parent\(\)[\s\S]*?\.to_path_buf\(\),[\s\S]*?\);/);
  assert.match(workerSource, /match &openpose \{[\s\S]*?model\.with_openpose\(source\)/);
  assert.match(workerSource, /model\.with_face\(face_dir\)/);
  assert.match(smokeSource, /InstantId::load\(&InstantIdPaths \{[\s\S]*?openpose: openpose[\s\S]*?WeightsSource::Dir\(dir\.clone\(\)\)[\s\S]*?face_dir: Some\(face_dir\.clone\(\)\),[\s\S]*?\}\)/);
  assert.match(smokeSource, /\.with_openpose\(&WeightsSource::Dir\(dir\.clone\(\)\)\)/);
  assert.match(smokeSource, /\.with_face\(&face_dir\)/);
}

test("c6d Candle InstantID initializers price resolved OpenPose and face paths before preserving builder loads", () => {
  assertC6dCandleInstantIdPaths(instantIdWorker, instantIdSmoke);
  for (const [label, changedWorker, changedSmoke] of [
    ["production openpose", instantIdWorker.replace("                openpose: openpose.clone(),\n", ""), instantIdSmoke],
    ["production face", instantIdWorker.replace(/                face_dir: Some\([\s\S]*?                \),\n/, ""), instantIdSmoke],
    ["smoke openpose", instantIdWorker, instantIdSmoke.replace("        openpose: openpose.as_ref().map(|dir| WeightsSource::Dir(dir.clone())),\n", "")],
    ["smoke face", instantIdWorker, instantIdSmoke.replace("        face_dir: Some(face_dir.clone()),\n", "")],
  ]) {
    assert.ok(changedWorker !== instantIdWorker || changedSmoke !== instantIdSmoke, `${label} mutation must alter a fixture`);
    assert.throws(() => assertC6dCandleInstantIdPaths(changedWorker, changedSmoke), { name: "AssertionError" }, label);
  }
});

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
  assert.match(workflow, /default: "33928871038"/);
  assert.match(workflow, /default: starvector-terminal-preflight-42ab6f2b8b9815205bc215c6d19c2b7714c908fe-33928871038-1/);
  assert.equal((workflow.match(/starvector-terminal-pin-paths\.mjs/g) ?? []).length, 2);
  assert.equal((workflow.match(/preflight-transport release[\\/]starvector-terminal-campaign-v1\.json/g) ?? []).length, 2);
  assert.equal((workflow.match(/preflight-metadata release[\\/]starvector-terminal-campaign-v1\.json/g) ?? []).length, 2);
  assert.equal((workflow.match(/gh api "repos\//g) ?? []).length, 4);
  assert.equal((workflow.match(/artifact-ids: \$\{\{ steps\.preflight-transport\.outputs\.artifact-id \}\}/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /STARVECTOR_TERMINAL_ROOT[\\/]inference|STARVECTOR_TERMINAL_ROOT[\\/]inference-preflight|STARVECTOR_TERMINAL_ROOT[\\/]corpora/);
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

test("workflow shell blocks consume untrusted dispatch inputs only through quoted environment variables", () => {
  for (const source of [workflow, readinessWorkflow, campaignWorkflow]) assertNoDirectInputInterpolationInShell(source);
  assert.match(workflow, /preflight-transport[^\n]*"\$STARVECTOR_INFERENCE_REVISION" "\$STARVECTOR_PREFLIGHT_RUN_ID" "\$STARVECTOR_PREFLIGHT_ARTIFACT_NAME"/);
  assert.match(workflow, /preflight-transport[^\n]*"\$env:STARVECTOR_INFERENCE_REVISION" "\$env:STARVECTOR_PREFLIGHT_RUN_ID" "\$env:STARVECTOR_PREFLIGHT_ARTIFACT_NAME"/);
  assert.match(campaignWorkflow, /validate-dispatch[^\n]*"\$STARVECTOR_TERMINAL_PERMANENT_PIN" "\$STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID"/);
  assert.match(campaignWorkflow, /validate-dispatch[^\n]*"\$env:STARVECTOR_TERMINAL_PERMANENT_PIN" "\$env:STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID"/);
  const expression = (name) => '"' + "$" + "{{ inputs." + name + " }}\"";
  for (const [label, source, mutation] of [
    ["Bash direct permanent-pin expression", campaignWorkflow, (value) => value.replace('"$STARVECTOR_TERMINAL_PERMANENT_PIN"', expression("permanent_pin"))],
    ["PowerShell direct campaign expression", campaignWorkflow, (value) => value.replace('"$env:STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID"', expression("campaign_run_id"))],
    ["Bash direct preflight run expression", workflow, (value) => value.replace('"$STARVECTOR_PREFLIGHT_RUN_ID"', expression("inference_preflight_run_id"))],
    ["PowerShell direct artifact expression", workflow, (value) => value.replace('"$env:STARVECTOR_PREFLIGHT_ARTIFACT_NAME"', expression("inference_preflight_artifact"))],
  ]) {
    const changed = mutation(source);
    assert.notEqual(changed, source, label);
    assert.throws(() => assertNoDirectInputInterpolationInShell(changed), { name: "AssertionError" }, label);
  }
});

test("provision transport accepts only the sealed current native-preflight run and artifact", async () => {
  const accepted = await validatePreflightTransport("release/starvector-terminal-campaign-v1.json", {
    revision: "42ab6f2b8b9815205bc215c6d19c2b7714c908fe",
    workflowRunId: "33928871038",
    artifactName: "starvector-terminal-preflight-42ab6f2b8b9815205bc215c6d19c2b7714c908fe-33928871038-1",
  });
  assert.deepEqual(accepted, {
    revision: "42ab6f2b8b9815205bc215c6d19c2b7714c908fe",
    workflow_run_id: "33928871038",
    artifact_name: "starvector-terminal-preflight-42ab6f2b8b9815205bc215c6d19c2b7714c908fe-33928871038-1",
    workflow_run_attempt: 1,
    artifact_id: 9957850431,
    artifact_digest: "sha256:609a2850118c206a4a38698e1680e81b78666d6b72be07d66ba677b0f50a9831",
  });
  for (const [label, mutation] of [
    ["revision", { revision: "0".repeat(40) }],
    ["run", { workflowRunId: "33851645748" }],
    ["artifact name", { artifactName: "starvector-terminal-preflight" }],
  ]) {
    await assert.rejects(() => validatePreflightTransport("release/starvector-terminal-campaign-v1.json", {
      revision: accepted.revision,
      workflowRunId: accepted.workflow_run_id,
      artifactName: accepted.artifact_name,
      ...mutation,
    }), /sealed terminal plan/, label);
  }
});

test("live preflight artifact and run metadata must match every sealed transport identity", async () => {
  const run = {
    id: 33928871038,
    run_attempt: 1,
    head_sha: "42ab6f2b8b9815205bc215c6d19c2b7714c908fe",
    workflow_id: 312370029,
    name: "Real-weight validation",
    path: ".github/workflows/real-weights.yml",
    event: "workflow_dispatch",
    status: "completed",
    conclusion: "success",
    repository: { id: 1299380446, full_name: "SceneWorks/inference" },
    head_repository: { id: 1299380446, full_name: "SceneWorks/inference" },
  };
  const artifact = {
    id: 9957850431,
    name: "starvector-terminal-preflight-42ab6f2b8b9815205bc215c6d19c2b7714c908fe-33928871038-1",
    size_in_bytes: 6456,
    digest: "sha256:609a2850118c206a4a38698e1680e81b78666d6b72be07d66ba677b0f50a9831",
    expired: false,
    workflow_run: {
      id: 33928871038,
      head_sha: "42ab6f2b8b9815205bc215c6d19c2b7714c908fe",
      repository_id: 1299380446,
      head_repository_id: 1299380446,
    },
  };
  await validatePreflightMetadata("release/starvector-terminal-campaign-v1.json", artifact, run);
  for (const [label, target, mutate] of [
    ["artifact id", "artifact", (value) => { value.id += 1; }],
    ["artifact name", "artifact", (value) => { value.name += "-other"; }],
    ["artifact size", "artifact", (value) => { value.size_in_bytes += 1; }],
    ["artifact digest", "artifact", (value) => { value.digest = `sha256:${"0".repeat(64)}`; }],
    ["artifact expiry", "artifact", (value) => { value.expired = true; }],
    ["artifact run", "artifact", (value) => { value.workflow_run.id += 1; }],
    ["artifact head", "artifact", (value) => { value.workflow_run.head_sha = "0".repeat(40); }],
    ["artifact repository id", "artifact", (value) => { value.workflow_run.repository_id += 1; }],
    ["artifact head repository id", "artifact", (value) => { value.workflow_run.head_repository_id += 1; }],
    ["run id", "run", (value) => { value.id += 1; }],
    ["run attempt", "run", (value) => { value.run_attempt += 1; }],
    ["run head", "run", (value) => { value.head_sha = "0".repeat(40); }],
    ["workflow id", "run", (value) => { value.workflow_id += 1; }],
    ["workflow name", "run", (value) => { value.name += " other"; }],
    ["workflow path", "run", (value) => { value.path = ".github/workflows/other.yml"; }],
    ["workflow event", "run", (value) => { value.event = "push"; }],
    ["run status", "run", (value) => { value.status = "in_progress"; }],
    ["run conclusion", "run", (value) => { value.conclusion = "failure"; }],
    ["repository", "run", (value) => { value.repository.full_name = "fork/inference"; }],
    ["head repository", "run", (value) => { value.head_repository.full_name = "fork/inference"; }],
  ]) {
    const changedArtifact = structuredClone(artifact);
    const changedRun = structuredClone(run);
    mutate(target === "artifact" ? changedArtifact : changedRun);
    await assert.rejects(
      () => validatePreflightMetadata("release/starvector-terminal-campaign-v1.json", changedArtifact, changedRun),
      /live preflight artifact and workflow metadata/,
      label,
    );
  }
});

test("downloaded preflight index must exactly match the checked-in native-preflight identity", () => {
  const expected = terminalPlan.inference_preflight;
  const observed = {
    workflow_run_id: expected.workflow_run_id,
    workflow_run_attempt: expected.workflow_run_attempt,
    head_sha: expected.head_sha,
    inventory_artifacts: structuredClone(expected.inventory_artifacts),
    hook_logs: structuredClone(expected.hook_logs),
  };
  assert.deepEqual(validateSealedPreflightIndex(structuredClone(observed), expected), observed);
  for (const [label, mutation] of [
    ["run", (value) => { value.workflow_run_id = "33851645748"; }],
    ["attempt", (value) => { value.workflow_run_attempt = 2; }],
    ["head", (value) => { value.head_sha = "0".repeat(40); }],
    ["inventory identity", (value) => { value.inventory_artifacts[0].sha256 = "0".repeat(64); }],
    ["hook identity", (value) => { value.hook_logs[0].sha256 = "0".repeat(64); }],
  ]) {
    const changed = structuredClone(observed);
    mutation(changed);
    assert.throws(() => validateSealedPreflightIndex(changed, expected), /sealed terminal plan provenance/, label);
  }
});

test("provision, readiness, and campaign share one pin-root formula with no legacy fallback", () => {
  for (const [label, source, expectedCalls] of [
    ["provision", workflow, 2],
    ["readiness", readinessWorkflow, 2],
    ["campaign", campaignWorkflow, 6],
  ]) {
    assert.equal((source.match(/scripts[\\/]lib[\\/]starvector-terminal-pin-paths\.mjs/g) ?? []).length, expectedCalls, `${label} helper calls`);
    assert.doesNotMatch(source, /starvector-terminal[\\/]inference(?:[\\/]|\s|$)/, `${label} legacy inference fallback`);
    assert.doesNotMatch(source, /starvector-terminal[\\/]inference-preflight(?:[\\/]|\s|$)/, `${label} legacy preflight fallback`);
    assert.doesNotMatch(source, /starvector-terminal[\\/]corpora[\\/]starvector-terminal-v1/, `${label} legacy corpus fallback`);
  }
  for (const shared of ["weights", "metrics"]) assert.match(workflow + readinessWorkflow + campaignWorkflow, new RegExp(`starvector-terminal[\\\\/]${shared}`));
  assert.match(workflow, /Join-Path \$env:STARVECTOR_TERMINAL_ROOT 'python\\cpython-3\.12\.10-x64-nuget'/);
  assert.match(workflow, /Join-Path \$env:STARVECTOR_TERMINAL_ROOT '\.locks\\cpython-3\.12\.10-x64-nuget\.lock'/);
  assert.match(campaignWorkflow, /SceneWorks[\\\\/]terminal-leases|ProgramData[\\\\/]SceneWorks[\\\\/]terminal-leases/);
});

function assertWindowsPinResolutionFailsBeforeEnvironmentPublication(source, expectedSteps) {
  const guarded = /\$pinEnvironment = node scripts\\lib\\starvector-terminal-pin-paths\.mjs[^\n]*\n\s+if \(\$LASTEXITCODE -ne 0\) \{ throw 'immutable pin-keyed terminal path resolution failed' \}\n\s+\$pinEnvironment \| Out-File -FilePath \$env:GITHUB_ENV -Encoding utf8 -Append/g;
  assert.equal((source.match(guarded) ?? []).length, expectedSteps);
}

test("Windows pin and transport validation fail before publishing outputs", () => {
  assertWindowsPinResolutionFailsBeforeEnvironmentPublication(workflow, 1);
  assertWindowsPinResolutionFailsBeforeEnvironmentPublication(readinessWorkflow, 1);
  assertWindowsPinResolutionFailsBeforeEnvironmentPublication(campaignWorkflow, 2);
  const windowsJob = workflow.slice(workflow.indexOf("  provision-windows:"));
  const transport = windowsJob.indexOf("preflight-transport");
  const artifactApi = windowsJob.indexOf("actions/artifacts/$artifactId", transport);
  const runApi = windowsJob.indexOf("actions/runs/$env:STARVECTOR_PREFLIGHT_RUN_ID", artifactApi);
  const metadata = windowsJob.indexOf("preflight-metadata", runApi);
  const output = windowsJob.indexOf('"artifact-id=$artifactId"', metadata);
  const download = windowsJob.indexOf("- name: Download the complete inference preflight artifact", output);
  assert.ok(transport >= 0 && artifactApi > transport && runApi > artifactApi && metadata > runApi && output > metadata && download > output);
});

function assertWindowsCorpusCompilesWithoutInheritedRustcWrapper(workflowSource) {
  const jobStart = workflowSource.indexOf("  provision-windows:");
  const stepStart = workflowSource.indexOf("- name: Materialize the exact pinned 120-row corpus", jobStart);
  const stepEnd = workflowSource.indexOf("- name: Read-only readiness validation", stepStart);
  assert.ok(jobStart >= 0 && stepStart > jobStart && stepEnd > stepStart);
  const step = workflowSource.slice(stepStart, stepEnd);
  const clearWrapper = step.indexOf("Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue");
  const cargo = step.indexOf("cargo run --release --locked -p sceneworks-worker --bin starvector_terminal_corpus");
  assert.equal((step.match(/Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue/g) ?? []).length, 1);
  assert.ok(clearWrapper >= 0 && clearWrapper < cargo);
}

test("Windows corpus compilation cannot inherit the persistent runner's sccache wrapper", () => {
  assertWindowsCorpusCompilesWithoutInheritedRustcWrapper(workflow);
  for (const [label, mutation] of [
    ["missing wrapper clear", (value) => value.replace("Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue", "Write-Host $env:RUSTC_WRAPPER")],
    ["wrapper cleared after Cargo", (value) => value.replace(/(\s+)(Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue)(\s+)(cargo run --release --locked -p sceneworks-worker --bin starvector_terminal_corpus[^\n]+)/, "$1$4$3$2")],
  ]) {
    const changed = mutation(workflow);
    assert.notEqual(changed, workflow, `${label} mutation must alter the workflow fixture`);
    assert.throws(() => assertWindowsCorpusCompilesWithoutInheritedRustcWrapper(changed), { name: "AssertionError" }, label);
  }
});

function assertWindowsMetricsPythonContract(step) {
  assert.doesNotMatch(step, /\bpy\s+-3\.11\b/);
  assert.match(step, /provision-starvector-windows-python\.ps1/);
  assert.match(step, /\$bootstrap = Select-StarVectorBootstrapPython -CandidatePaths @\(\$bootstrapPython\) -ExpectedVersion @\(3, 12, 10\)/);
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
  assert.match(step, /& \$bootstrapPython -m venv --copies \$metricsRoot/);
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
  assert.ok((source.match(/Attributes -band \[IO\.FileAttributes\]::ReparsePoint/g) ?? []).length >= 4);
  assert.match(source, /Get-ChildItem -LiteralPath \$item\.FullName -Force -ErrorAction Stop/);
  assert.match(source, /Remove-Item -LiteralPath \$item\.FullName -Force -ErrorAction Stop/);
  assert.doesNotMatch(source, /Remove-Item[^\n]*-Recurse/);
  assert.match(source, /refusing to remove a metrics venv outside the exact workflow-owned terminal root/);
  assert.match(source, /\$text -match '\[\\r\\n"\]'/);
  assert.match(source, /\$text -notmatch '\^\[A-Za-z\]:\\\\'/);
  assert.match(source, /\[IO\.Path\]::GetFullPath\(\$text\)/);
  assert.match(source, /Test-Path -LiteralPath \$full -PathType Leaf/);
  assert.match(source, /Assert-StarVectorWindowsPathComponents -Path \$full -LeafType File/);
  assert.match(source, /New-Object System\.Diagnostics\.ProcessStartInfo/);
  assert.match(source, /\$startInfo\.FileName = \$Executable/);
  assert.match(source, /\$startInfo\.UseShellExecute = \$false/);
  assert.match(source, /\$startInfo\.RedirectStandardOutput = \$true/);
  assert.match(source, /\$startInfo\.RedirectStandardError = \$true/);
  assert.match(source, /\$startInfo\.CreateNoWindow = \$true/);
  assert.match(source, /\$stdoutTask = \$process\.StandardOutput\.ReadToEndAsync\(\)/);
  assert.match(source, /\$stderrTask = \$process\.StandardError\.ReadToEndAsync\(\)/);
  assert.match(source, /\$process\.WaitForExit\(\$TimeoutMilliseconds\)/);
  assert.match(source, /\$process\.Kill\(\)/);
  assert.match(source, /ExitCode = \$process\.ExitCode/);
  assert.match(source, /StdOut = \$stdoutTask\.Result/);
  assert.match(source, /StdErr = \$stderrTask\.Result/);
  assert.match(source, /import ensurepip,json,pip,platform,struct,sys,venv/);
  assert.match(source, /base_executable'':getattr\(sys,''_base_executable'',None\)/);
  assert.match(source, /platform\.python_implementation\(\)/);
  assert.match(source, /platform\.machine\(\)/);
  assert.match(source, /struct\.calcsize\(''P''\)\*8/);
  assert.match(source, /if \(\$probe\.ExitCode -ne 0\) \{ continue \}/);
  assert.match(source, /\$probe\.StdOut \| ConvertFrom-Json -ErrorAction Stop/);
  assert.match(source, /\[int\]\$identity\.version\[1\] -eq \$ExpectedVersion\[1\]/);
  assert.match(source, /\$ExpectedVersion\.Count -eq 2 -or \[int\]\$identity\.version\[2\] -eq \$ExpectedVersion\[2\]/);
  assert.doesNotMatch(source, /\[int\]\$identity\.version\[1\] -ge \$ExpectedVersion\[1\]/);
  assert.match(source, /\[string\]\$identity\.implementation -ceq 'CPython'/);
  assert.match(source, /\[string\]\$identity\.architecture -ceq 'AMD64'/);
  assert.match(source, /\[int\]\$identity\.pointer_bits -eq 64/);
  assert.match(source, /OrdinalIgnoreCase\.Equals\(\$canonicalCandidate, \$canonicalExecutable\)/);
  assert.doesNotMatch(source, /& \$candidate|Invoke-Expression|Start-Process/);
}

function assertWindowsPortablePythonContract(step, source, testSource) {
  assert.match(source, /https:\/\/api\.nuget\.org\/v3-flatcontainer\/python\/3\.12\.10\/python\.3\.12\.10\.nupkg/);
  assert.match(source, /0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8/);
  assert.match(source, /bbda4dcf688a94211b62d50968a91b38f305d0b8d1ecd90269f74a86f8a0a4fcebb7ca162a0753a47691eb3df0c964009bd3d8194c6fd19afae8d5fd01e1cc0f/);
  assert.match(source, /StarVectorWindowsPythonPackageBytes = 14515433/);
  assert.match(source, /4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a/);
  assert.match(source, /9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120/);
  assert.match(source, /StarVectorWindowsPythonMaximumPackageBytes = 20MB/);
  assert.match(source, /StarVectorWindowsPythonMaximumExpandedBytes = 80MB/);
  assert.match(source, /StarVectorWindowsPythonMaximumArchiveEntries = 1400/);
  assert.match(source, /function Invoke-StarVectorWindowsPythonLock/);
  assert.match(source, /\$stream = \[IO\.File\]::Open\(\$canonicalLock, \[IO\.FileMode\]::OpenOrCreate, \[IO\.FileAccess\]::ReadWrite, \[IO\.FileShare\]::None\)/);
  assert.match(source, /timed out waiting for the exclusive portable Python lock/);
  assert.match(source, /function Save-StarVectorWindowsPythonPackage/);
  assert.match(source, /\$handler\.AllowAutoRedirect = \$false/);
  assert.match(source, /\$client\.Timeout = \[TimeSpan\]::FromSeconds\(\$TimeoutSeconds\)/);
  assert.match(source, /ContentLength -ne \$ExpectedBytes/);
  assert.match(source, /\$bytes\.LongLength -ne \$ExpectedBytes/);
  assert.match(source, /Get-StarVectorSha256 \$destinationPath/);
  assert.match(source, /Get-StarVectorSha512 \$destinationPath/);
  assert.match(source, /\$entryCount -gt \$MaximumEntries/);
  assert.match(source, /\$expandedBytes -gt \$MaximumExpandedBytes/);
  assert.match(source, /path escapes its staging root/);
  assert.match(source, /Add-Type -AssemblyName System\.IO\.Compression\.FileSystem/);
  assert.match(source, /\[IO\.Compression\.ZipFile\]::OpenRead\(\$ArchivePath\)/);
  assert.match(source, /Python Software Foundation/);
  assert.match(source, /https:\/\/github\.com\/Python\/CPython\.git/);
  assert.match(source, /0cc8128/);
  assert.match(source, /\.signature\.p7s/);
  assert.match(source, /Get-StarVectorSha256 \$python\) -cne \$ExpectedPythonSha256/);
  assert.match(source, /Get-StarVectorSha256 \$pythonDll\) -cne \$ExpectedDllSha256/);
  assert.match(source, /Assert-StarVectorWindowsPathComponents -Path \$DestinationRoot -LeafType Directory -AllowMissingLeaf/);
  assert.match(source, /\[IO\.Directory\]::Move\(\$stagingRoot, \$DestinationRoot\)/);
  assert.doesNotMatch(source, /msiexec|python-3\.12\.10-amd64\.exe|HKLM:|HKCR:|choco install/i);

  assert.match(step, /timeout-minutes: 70/);
  assert.match(step, /\$pythonRoot = Join-Path \$env:STARVECTOR_TERMINAL_ROOT 'python\\cpython-3\.12\.10-x64-nuget'/);
  assert.match(step, /\$lockPath = Join-Path \$env:STARVECTOR_TERMINAL_ROOT '\.locks\\cpython-3\.12\.10-x64-nuget\.lock'/);
  assert.doesNotMatch(step, /\$lockPath[^\n]*python\\/);
  assert.equal((step.match(/Invoke-StarVectorWindowsPythonLock -LockPath \$lockPath -ScriptBlock \{/g) ?? []).length, 1);
  const lockedBody = step.match(/Invoke-StarVectorWindowsPythonLock -LockPath \$lockPath -ScriptBlock \{\n([\s\S]*?)\n          \}\s*$/)?.[1];
  assert.ok(lockedBody, "portable Python and metric setup must remain inside the exclusive lock body");
  const install = lockedBody.indexOf("Install-StarVectorWindowsPythonPackage");
  const venv = lockedBody.indexOf("& $bootstrapPython -m venv --copies $metricsRoot", install);
  const metrics = lockedBody.indexOf("starvector-terminal-provision.mjs metrics", venv);
  assert.ok(install >= 0 && venv > install && metrics > venv);
  assert.match(step, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$bootstrapPython\) -ExpectedVersion @\(3, 12, 10\)/);
  assert.doesNotMatch(step, /actions\/setup-python|RUNNER_TOOL_CACHE|AGENT_TOOLSDIRECTORY|GITHUB_PATH|\$env:Path\s*=|GITHUB_OUTPUT/);

  assert.match(testSource, /partial setup-python installer root/);
  assert.match(testSource, /RUNNER_TOOL_CACHE = 'D:\\actions-runner\\_work\\_tool'/);
  assert.match(testSource, /AGENT_TOOLSDIRECTORY = 'E:\\different-runner\\_work\\_tool'/);
  assert.match(testSource, /exclusive lock did not reject a concurrent shared-root provisioner/);
  assert.match(testSource, /portable Python did not reject a \$junctionKind-root junction/);
  assert.doesNotMatch(testSource, /CreateFromDirectory/);
  assert.match(testSource, /Replace\(\[IO\.Path\]::DirectorySeparatorChar, \[IO\.Path\]::AltDirectorySeparatorChar\)/);
  assert.match(testSource, /New-SingleEntryZip -DestinationPath \$backslashArchive -EntryName 'tools\\python\.exe'/);
  assert.match(testSource, /archive entry with a backslash name was not rejected/);
  assert.match(testSource, /archive entry-count limit was not enforced/);
  assert.match(testSource, /archive expansion-size limit was not enforced/);
  assert.match(testSource, /Install-StarVectorWindowsPythonPackage -DestinationRoot \$officialRoot/);
  assert.match(testSource, /official portable python\.exe hash changed/);
  assert.match(testSource, /official portable python312\.dll hash changed/);
  assert.match(testSource, /& \$official -m venv --copies \$venvRoot/);
  assertWindowsPortablePythonFixtureArchiveContract(testSource);
}

function assertWindowsPortablePythonFixtureArchiveContract(testSource) {
  const compressionLoad = testSource.indexOf("Add-Type -AssemblyName System.IO.Compression\n");
  const fileSystemLoad = testSource.indexOf("Add-Type -AssemblyName System.IO.Compression.FileSystem\n");
  const firstArchiveConstruction = testSource.indexOf("[IO.Compression.ZipArchive]::new");
  assert.ok(compressionLoad >= 0 && fileSystemLoad > compressionLoad && firstArchiveConstruction > fileSystemLoad);

  for (const functionName of ["New-CanonicalZipFromDirectory", "New-SingleEntryZip"]) {
    const start = testSource.indexOf(`function ${functionName} {`);
    const nextFunction = testSource.indexOf("\nfunction ", start + 1);
    const scriptBody = testSource.indexOf("\n$root =", start + 1);
    const end = nextFunction >= 0 ? nextFunction : scriptBody;
    const body = testSource.slice(start, end);
    const streamOpen = body.indexOf("$stream = [IO.File]::Open(");
    const guardedConstruction = body.indexOf("try {\n    $archive = [IO.Compression.ZipArchive]::new", streamOpen);
    const archiveDispose = body.indexOf("$archive.Dispose()", guardedConstruction);
    const streamDispose = body.indexOf("$stream.Dispose()", archiveDispose);
    assert.ok(start >= 0 && streamOpen >= 0 && guardedConstruction > streamOpen && archiveDispose > guardedConstruction && streamDispose > archiveDispose, functionName);
    assert.match(body, /ZipArchive\]::new\(\$stream, \[IO\.Compression\.ZipArchiveMode\]::Create, \$true\)/);
  }
}

test("PowerShell 5.1 ZIP fixtures load their defining assembly and guard every acquired stream", () => {
  assertWindowsPortablePythonFixtureArchiveContract(windowsPythonProvisionTest);
  for (const [label, mutation] of [
    ["compression assembly", (value) => value.replace("Add-Type -AssemblyName System.IO.Compression\n", "")],
    ["guard before archive construction", (value) => value.replace("  try {\n    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)", "  $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)\n  try {")],
    ["archive disposal", (value) => value.replace("      $archive.Dispose()", "      Write-Host $archive")],
    ["stream disposal", (value) => value.replace("    $stream.Dispose()", "    Write-Host $stream")],
    ["outer stream ownership", (value) => value.replace("[IO.Compression.ZipArchiveMode]::Create, $true", "[IO.Compression.ZipArchiveMode]::Create, $false")],
  ]) {
    const changed = mutation(windowsPythonProvisionTest);
    assert.notEqual(changed, windowsPythonProvisionTest, `${label} mutation must alter the fixture`);
    assert.throws(() => assertWindowsPortablePythonFixtureArchiveContract(changed), { name: "AssertionError" }, label);
  }
});

test("terminal metrics provisioning fixes CPython 3.12 and wheel-only installation on both hosts", () => {
  assert.equal((workflow.match(/uses: actions\/setup-python@/g) ?? []).length, 0);
  assert.equal((workflow.match(/python-version:/g) ?? []).length, 0);
  assert.equal((workflow.match(/update-environment:/g) ?? []).length, 0);
  assert.match(workflow, /Select supported existing CPython 3\.12 arm64 for terminal metrics/);
  assert.match(workflow, /select-starvector-macos-python\.mjs select \/opt\/homebrew\/bin\/python3\.12 \/usr\/local\/bin\/python3\.12/);
  assert.match(workflow, /select-starvector-macos-python\.mjs verify-venv "\$STARVECTOR_METRICS_BOOTSTRAP_PYTHON" "\$STARVECTOR_TERMINAL_ROOT\/metrics\/bin\/python"/);
  assert.match(workflow, /remove-metrics-tree "\$STARVECTOR_TERMINAL_ROOT\/metrics" \/Users\/Shared\/SceneWorks\/starvector-terminal\/metrics/);
  assert.doesNotMatch(workflow.slice(workflow.indexOf("  provision-macos:"), workflow.indexOf("  provision-windows:")), /actions\/setup-python|RUNNER_TOOL_CACHE|AGENT_TOOLSDIRECTORY|\/Users\/runner\/hostedtoolcache/);
  const windowsJob = workflow.slice(workflow.indexOf("  provision-windows:"));
  assert.doesNotMatch(windowsJob, /actions\/setup-python|RUNNER_TOOL_CACHE|AGENT_TOOLSDIRECTORY|D:\\actions-runner|cuda-windows-2/);
  assert.match(windowsJob, /Provision exact portable CPython 3\.12\.10 x64 and terminal metrics/);
  assert.equal((workflow.match(/pip install --disable-pip-version-check --only-binary=:all: --retries 5 --timeout 60/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /pip install --disable-pip-version-check (?!.*--only-binary=:all:)/);
  assert.match(macosPythonSelector, /identity\.version\[1\] === 12/);
  assert.match(macosPythonSelector, /identity\.implementation === "CPython"/);
  assert.match(macosPythonSelector, /\["arm64", "aarch64"\]\.includes\(identity\.architecture\.toLowerCase\(\)\)/);
  assert.match(macosPythonSelector, /identity\.pointer_bits === 64/);
  assert.match(macosPythonSelector, /"prefix":sys\.prefix,"base_prefix":sys\.base_prefix/);
  assert.match(macosPythonSelector, /spawnSync\(path\.resolve\(value\)/);
  assert.doesNotMatch(macosPythonSelector, /spawnSync\(executable,/);
  assert.match(macosPythonSelector, /canonicalObservedPrefix !== canonicalExpectedPrefix \|\| canonicalObservedPrefix === canonicalBasePrefix/);
  assert.match(macosPythonSelector, /if \(item\.isSymbolicLink\(\)\) \{\s+unlinkSync\(frame\.file\);/);

  const start = workflow.indexOf("- name: Provision exact portable CPython 3.12.10 x64 and terminal metrics", workflow.indexOf("  provision-windows:"));
  const end = workflow.indexOf("- name: Materialize the exact pinned 120-row corpus", start);
  const step = workflow.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assertWindowsMetricsPythonContract(step);
  assertWindowsPortablePythonContract(step, windowsPythonProvision, windowsPythonProvisionTest);
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
    await mkdir(path.dirname(executable), { recursive: true });
    const payload = JSON.stringify({
      executable,
      base_executable: identity.base_executable ?? executable,
      prefix: identity.prefix ?? path.dirname(path.dirname(executable)),
      base_prefix: identity.base_prefix ?? root,
      ...identity,
    });
    await writeFile(executable, `#!/bin/sh\nprintf '%s\\n' '${payload}'\nexit ${exitCode}\n`);
    await chmod(executable, 0o755);
    return executable;
  };
  const invalid = await writeFakePython("python-invalid", { version: [3, 14, 1], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  const bootstrap = await writeFakePython("python-valid", { prefix: root, base_prefix: root, version: [3, 12, 13], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  const venv = await writeFakePython("metrics/bin/python", { base_executable: bootstrap, version: [3, 12, 13], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
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
  assert.throws(() => validateStarVectorMacVenv(bootstrap, bootstrap), /not an isolated venv/);
  const wrongMicro = await writeFakePython("wrong-metrics/bin/python", { base_executable: bootstrap, version: [3, 12, 12], implementation: "CPython", architecture: "arm64", pointer_bits: 64 });
  assert.throws(() => validateStarVectorMacVenv(bootstrap, wrongMicro), /selected exact CPython/);
  await chmod(bootstrap, 0o644);
  assert.throws(() => selectStarVectorMacPython([bootstrap]), /existing absolute CPython/);
});

function selectLocalMacPythonOrSkip(context) {
  try {
    return selectStarVectorMacPython(["/opt/homebrew/bin/python3.12", "/usr/local/bin/python3.12"]);
  } catch {
    context.skip("requires a local CPython 3.12 arm64 interpreter");
    return null;
  }
}

test("macOS venv validation executes the supplied venv path and rejects the bootstrap itself", async (context) => {
  const bootstrap = selectLocalMacPythonOrSkip(context);
  if (!bootstrap) return;
  const root = await mkdtemp(path.join(tmpdir(), "starvector-real-macos-venv-"));
  const metrics = path.join(root, "metrics");
  await execFile(bootstrap.executable, ["-m", "venv", metrics]);
  const validated = validateStarVectorMacVenv(bootstrap.executable, path.join(metrics, "bin", "python"));
  assert.equal(await realpath(validated.venv.identity.prefix), await realpath(metrics));
  assert.notEqual(await realpath(validated.venv.identity.prefix), await realpath(validated.venv.identity.base_prefix));
  assert.throws(() => validateStarVectorMacVenv(bootstrap.executable, bootstrap.executable), /not an isolated venv/);
  removeStarVectorMacMetricsTree(metrics, metrics);
});

test("macOS stale metrics cleanup removes real venv symlinks without traversing external targets", async (context) => {
  const bootstrap = selectLocalMacPythonOrSkip(context);
  if (!bootstrap) return;
  const root = await mkdtemp(path.join(tmpdir(), "starvector-macos-metrics-"));
  const metrics = path.join(root, "metrics");
  const outside = path.join(root, "outside");
  await execFile(bootstrap.executable, ["-m", "venv", metrics]);
  await mkdir(outside, { recursive: true });
  await writeFile(path.join(outside, "sentinel.txt"), "outside");
  assert.throws(() => removeStarVectorMacMetricsTree(metrics, path.join(root, "wrong")), /exact workflow-owned terminal root/);
  await symlink(outside, path.join(metrics, "external-link"));
  removeStarVectorMacMetricsTree(metrics, metrics);
  assert.equal(await lstat(metrics).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  assert.equal(await readFile(path.join(outside, "sentinel.txt"), "utf8"), "outside");

  const rootSymlink = path.join(root, "metrics-root-link");
  await symlink(outside, rootSymlink);
  assert.throws(() => removeStarVectorMacMetricsTree(rootSymlink, rootSymlink), /root is a symlink|containing a symlink/);
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
  assert.match(windowsPythonProbeTest, /07-hanging-python\.exe/);
  assert.match(windowsPythonProbeTest, /Traceback from the first fake Python candidate/);
  assert.match(windowsPythonProbeTest, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$bad, \$malformed, \$belowMinimum, \$newerUnsupported, \$wrongArchitecture,/);
  assert.match(windowsPythonProbeTest, /all invalid candidates must fail closed/);
  assert.match(windowsPythonProbeTest, /exact-micro selection accepted a different CPython 3\.12 patch release/);
  assert.match(windowsPythonProbeTest, /executable selection traversed a junction parent/);
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
    ["missing probe timeout", (value) => value.replace("$process.WaitForExit($TimeoutMilliseconds)", "$process.WaitForExit()")],
    ["permissive Python minor", (value) => value.replace("[int]$identity.version[1] -eq $ExpectedVersion[1]", "[int]$identity.version[1] -ge $ExpectedVersion[1]")],
    ["ignored Python micro", (value) => value.replace("$ExpectedVersion.Count -eq 2 -or [int]$identity.version[2] -eq $ExpectedVersion[2]", "$true")],
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

test("portable Windows Python contract rejects provenance, bounds, lock, and path-safety mutations", () => {
  const start = workflow.indexOf("- name: Provision exact portable CPython 3.12.10 x64 and terminal metrics", workflow.indexOf("  provision-windows:"));
  const end = workflow.indexOf("- name: Materialize the exact pinned 120-row corpus", start);
  const step = workflow.slice(start, end);
  for (const [label, stepMutation, sourceMutation] of [
    ["floating package", (value) => value, (value) => value.replace("python/3.12.10/python.3.12.10.nupkg", "python/3.12.10/python.nupkg")],
    ["package sha256", (value) => value, (value) => value.replace("0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8", "0".repeat(64))],
    ["package size", (value) => value, (value) => value.replace("StarVectorWindowsPythonPackageBytes = 14515433", "StarVectorWindowsPythonPackageBytes = 0")],
    ["python exe hash", (value) => value, (value) => value.replace("4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a", "0".repeat(64))],
    ["runtime dll hash", (value) => value, (value) => value.replace("9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120", "0".repeat(64))],
    ["archive entry cap", (value) => value, (value) => value.replace("StarVectorWindowsPythonMaximumArchiveEntries = 1400", "StarVectorWindowsPythonMaximumArchiveEntries = 10000")],
    ["expanded byte cap", (value) => value, (value) => value.replace("StarVectorWindowsPythonMaximumExpandedBytes = 80MB", "StarVectorWindowsPythonMaximumExpandedBytes = 256MB")],
    ["redirect refusal", (value) => value, (value) => value.replace("$handler.AllowAutoRedirect = $false", "$handler.AllowAutoRedirect = $true")],
    ["exclusive file lock", (value) => value, (value) => value.replace("[IO.FileShare]::None", "[IO.FileShare]::ReadWrite")],
    ["destination component validation", (value) => value, (value) => value.replaceAll("Assert-StarVectorWindowsPathComponents -Path $DestinationRoot -LeafType Directory -AllowMissingLeaf", "Write-Host $DestinationRoot")],
    ["missing workflow lock", (value) => value.replace("Invoke-StarVectorWindowsPythonLock -LockPath $lockPath -ScriptBlock {", "& {"), (value) => value],
    ["lock inside mutable Python root", (value) => value.replace("'.locks\\cpython-3.12.10-x64-nuget.lock'", "'python\\cpython-3.12.10-x64-nuget.lock'"), (value) => value],
    ["ambient venv links", (value) => value.replace("-m venv --copies $metricsRoot", "-m venv $metricsRoot"), (value) => value],
    ["path mutation", (value) => value.replace("$bootstrapPython = $bootstrap.Executable", "$env:Path = \"$bootstrapPython;$env:Path\""), (value) => value],
  ]) {
    const changedStep = stepMutation(step);
    const changedSource = sourceMutation(windowsPythonProvision);
    assert.ok(changedStep !== step || changedSource !== windowsPythonProvision, `${label} mutation must alter a fixture`);
    assert.throws(() => assertWindowsPortablePythonContract(changedStep, changedSource, windowsPythonProvisionTest), { name: "AssertionError" }, label);
  }
});

test("Windows metrics Python contract rejects path, base-interpreter, and patch-version guard mutations", () => {
  const start = workflow.indexOf("- name: Provision exact portable CPython 3.12.10 x64 and terminal metrics", workflow.indexOf("  provision-windows:"));
  const end = workflow.indexOf("- name: Materialize the exact pinned 120-row corpus", start);
  const step = workflow.slice(start, end);
  for (const [label, mutation] of [
    ["selection helper", (value) => value.replace("$bootstrap = Select-StarVectorBootstrapPython -CandidatePaths @($bootstrapPython) -ExpectedVersion @(3, 12, 10)", "$bootstrap = Get-Command python.exe")],
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
  assert.equal((hostedWorkflow.match(/shell: powershell/g) ?? []).length, 3);
  assert.match(hostedWorkflow, /select-starvector-windows-python\.test\.ps1/);
  assert.match(hostedWorkflow, /"scripts\/provision-starvector-windows-python\.ps1"/);
  assert.match(hostedWorkflow, /"scripts\/provision-starvector-windows-python\.test\.ps1"/);
  assert.match(hostedWorkflow, /timeout-minutes: 10\s+run: \.\\scripts\\provision-starvector-windows-python\.test\.ps1/);
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
    ["portable provision trigger", (value) => value.replace('      - "scripts/provision-starvector-windows-python.ps1"\n', ''), (value) => value],
    ["portable provision test", (value) => value.replace("run: .\\scripts\\provision-starvector-windows-python.test.ps1", "run: Write-Host skipped"), (value) => value],
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

test("pinned inference checkout publication is same-pin idempotent and derives the immutable destination", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-pinned-checkout-")), source = path.join(root, "source"), hostRoot = path.join(root, "host");
  await mkdir(hostRoot);
  await mkdir(path.join(source, "release"), { recursive: true }); await mkdir(path.join(source, "scripts", "release"), { recursive: true });
  for (const relative of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) await writeFile(path.join(source, relative), relative);
  await execFile("git", ["init", source]); await execFile("git", ["-C", source, "add", "."]); await execFile("git", ["-C", source, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "fixture"]);
  const revision = (await execFile("git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
  const expected = terminalPinPaths(hostRoot, revision).inferenceRoot;
  // Recover the exact old implementation's interrupted direct-clone shape.
  await mkdir(expected, { recursive: true });
  await writeFile(path.join(expected, "partial-clone"), "incomplete");
  const foreignStaging = `${expected}.staging-foreign-owner`;
  await mkdir(foreignStaging);
  await writeFile(path.join(foreignStaging, "owner"), "other provision");
  assert.deepEqual(
    await Promise.all([
      installPinnedCheckout(source, hostRoot, revision),
      installPinnedCheckout(source, hostRoot, revision),
      installPinnedCheckout(source, hostRoot, revision),
    ]),
    [revision, revision, revision],
  );
  assert.equal(await readFile(path.join(foreignStaging, "owner"), "utf8"), "other provision");
  assert.equal(await installPinnedCheckout(source, hostRoot, revision), revision);
  assert.equal((await execFile("git", ["-C", expected, "rev-parse", "HEAD"])).stdout.trim(), revision);
  assert.equal(await lstat(path.join(hostRoot, "inference")).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  assert.deepEqual((await readdir(path.dirname(expected))).filter((name) => name.includes(`staging-${process.pid}-`)), []);

  await rm(expected, { recursive: true });
  const outside = path.join(root, "outside-checkout");
  await mkdir(outside);
  await writeFile(path.join(outside, "sentinel"), "outside");
  await symlink(outside, expected, process.platform === "win32" ? "junction" : "dir");
  await assert.rejects(() => installPinnedCheckout(source, hostRoot, revision), /symlink, junction, reparse|redirected component/);
  assert.equal(await readFile(path.join(outside, "sentinel"), "utf8"), "outside");
});

test("pinned checkout retries when the owner releases between contention and inspection", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-checkout-lock-release-"));
  const source = path.join(root, "source"), hostRoot = path.join(root, "host");
  try {
    await mkdir(hostRoot);
    for (const relative of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) {
      await mkdir(path.dirname(path.join(source, relative)), { recursive: true });
      await writeFile(path.join(source, relative), relative);
    }
    await execFile("git", ["init", source]);
    await execFile("git", ["-C", source, "add", "."]);
    await execFile("git", ["-C", source, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "fixture"]);
    const revision = (await execFile("git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
    // Isolate the filesystem interleaving in a child; no global mocks affect other tests.
    const child = `
      import assert from "node:assert/strict";
      import fs from "node:fs/promises";
      import { syncBuiltinESMExports } from "node:module";
      const [moduleUrl, source, hostRoot, revision, lockRoot] = process.argv.slice(1);
      await fs.mkdir(lockRoot, { recursive: true });
      const mkdir = fs.mkdir;
      let released = false;
      fs.mkdir = async (target, options) => {
        try { return await mkdir(target, options); }
        catch (error) {
          if (target === lockRoot && error.code === "EEXIST" && !released) {
            released = true;
            // The previous owner releases after our failed mkdir, before our lstat.
            await fs.rm(lockRoot, { recursive: true });
          }
          throw error;
        }
      };
      syncBuiltinESMExports();
      const { installPinnedCheckout } = await import(moduleUrl);
      assert.equal(await installPinnedCheckout(source, hostRoot, revision), revision);
      assert.equal(released, true);
      await assert.rejects(fs.lstat(lockRoot), { code: "ENOENT" });
    `;
    await execFile(process.execPath, ["--input-type=module", "-e", child,
      new URL("./starvector-terminal-provision.mjs", import.meta.url).href,
      source, hostRoot, revision, pinnedCheckoutLockPath(hostRoot, revision)]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("checkout concurrency guards are exact-pin scoped", async () => {
  const root = path.resolve(path.join(tmpdir(), "starvector-lock-scope"));
  const first = pinnedCheckoutLockPath(root, "1".repeat(40));
  const second = pinnedCheckoutLockPath(root, "2".repeat(40));
  assert.notEqual(first, second);
  assert.equal(path.dirname(first), path.join(root, ".locks"));
  assert.match(path.basename(first), /^inference-checkout-1{40}\.lock$/);
});

test("failed atomic checkout removes only its unique staging directory", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-provision-checkout-failure-"));
  const destination = path.join(root, "destination");
  const foreignStaging = `${destination}.staging-foreign-owner`;
  await mkdir(foreignStaging);
  await writeFile(path.join(foreignStaging, "owner"), "other provision");
  await assert.rejects(
    () => installCheckout(path.join(root, "missing-source"), destination, "a".repeat(40)),
    /Command failed|repository .* does not exist|fatal:/,
  );
  assert.equal(await readFile(path.join(foreignStaging, "owner"), "utf8"), "other provision");
  assert.deepEqual((await readdir(root)).filter((name) => name.includes(`staging-${process.pid}-`)), []);
});


test("upstream source restores canonical bytes from an exact clean CRLF checkout", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-upstream-eol-"));
  try {
    const source = path.join(root, "source.git"), destination = path.join(root, "upstream-source");
    const bytes = "def upstream():\n    return 1\n";
    await mkdir(path.join(source, "starvector"), { recursive: true });
    await execFile("git", ["init", source]);
    await execFile("git", ["-C", source, "config", "core.autocrlf", "false"]);
    await writeFile(path.join(source, "starvector/source.py"), bytes);
    await execFile("git", ["-C", source, "add", "."]);
    await execFile("git", ["-C", source, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "fixture"]);
    const revision = (await execFile("git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
    await execFile("git", ["clone", "--config", "core.autocrlf=true", source, destination]);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), bytes.replaceAll("\n", "\r\n"));
    assert.equal((await execFile("git", ["-C", destination, "status", "--porcelain"])).stdout, "");
    // This host attribute overrides core.autocrlf=false/core.eol=lf, including
    // checkout-index. Exercise actual Git conversion, not a mocked file read.
    const attributes = path.join(root, "host-attributes");
    await writeFile(attributes, "*.py text eol=crlf\n");
    await execFile("git", ["-C", destination, "config", "core.attributesFile", attributes]);
    const lock = { implementation_repository: source.slice(0, -4), implementation_revision: revision,
      python_source_sha256: digest(JSON.stringify([{ path: "starvector/source.py", sha256: digest(bytes) }])) };
    await prepareUpstreamSource(destination, lock);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), bytes);
    assert.equal((await execFile("git", ["-C", destination, "status", "--porcelain"])).stdout, "");
    assert.equal((await execFile("git", ["-C", destination, "rev-parse", "HEAD"])).stdout.trim(), revision);
    await prepareUpstreamSource(destination, lock);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), bytes);
    const fresh = path.join(root, "fresh-source");
    await prepareUpstreamSource(fresh, lock);
    assert.equal(await readFile(path.join(fresh, "starvector/source.py"), "utf8"), bytes);

    const attributesBefore = await readFile(path.join(destination, ".git/info/attributes"));
    await assert.rejects(() => prepareUpstreamSource(destination, { ...lock, python_source_sha256: "0".repeat(64) }), /Git blob identity mismatch/);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), bytes);
    assert.deepEqual(await readFile(path.join(destination, ".git/info/attributes")), attributesBefore);

    const dirty = bytes + "# unauthorized edit\n";
    await writeFile(path.join(destination, "starvector/source.py"), dirty);
    await assert.rejects(() => prepareUpstreamSource(destination, lock), /exact and clean/);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), dirty);
    await writeFile(path.join(destination, "starvector/source.py"), bytes);
    await assert.rejects(() => prepareUpstreamSource(destination, { ...lock, implementation_revision: "a".repeat(40) }), /exact and clean/);
    assert.equal(await readFile(path.join(destination, "starvector/source.py"), "utf8"), bytes);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});


test("upstream source rejects a clean Git symlink before modifying ordinary files", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-upstream-symlink-"));
  try {
    await mkdir(path.join(root, "starvector"));
    await execFile("git", ["init", root]);
    await execFile("git", ["-C", root, "config", "core.autocrlf", "false"]);
    // core.symlinks=false is the normal Windows representation and makes this
    // real symlink-mode Git fixture portable without elevated OS privileges.
    await execFile("git", ["-C", root, "config", "core.symlinks", "false"]);
    const target = "../../outside.py";
    await writeFile(path.join(root, "starvector/danger.py"), target);
    await writeFile(path.join(root, "starvector/source.py"), "audited = True\n");
    await execFile("git", ["-C", root, "add", "."]);
    const oid = (await execFile("git", ["-C", root, "hash-object", "-w", "starvector/danger.py"])).stdout.trim();
    await execFile("git", ["-C", root, "update-index", "--cacheinfo", `120000,${oid},starvector/danger.py`]);
    await execFile("git", ["-C", root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "symlink fixture"]);
    assert.equal((await execFile("git", ["-C", root, "status", "--porcelain"])).stdout, "");
    const revision = (await execFile("git", ["-C", root, "rev-parse", "HEAD"])).stdout.trim();
    await assert.rejects(() => prepareUpstreamSource(root, { implementation_revision: revision }), /ordinary contained Git blob/);
    assert.equal(await readFile(path.join(root, "starvector/danger.py"), "utf8"), target);
    assert.equal(await readFile(path.join(root, "starvector/source.py"), "utf8"), "audited = True\n");
    assert.equal(await lstat(path.join(root, ".git/info/attributes")).catch(() => null), null);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("upstream pip downloads retain exact CUDA builds and stop within finite retry bounds", async () => {
  const lock = JSON.parse(await readFile("release/starvector-terminal-upstream-lock-v1.json", "utf8"));
  const calls = [];
  await installUpstreamPackages("fixture-python", lock, async (...args) => { calls.push(args); });
  assert.equal(calls.length, 2);
  for (const [python, args, bounds] of calls) {
    assert.equal(python, "fixture-python");
    assert.equal(args[args.indexOf("--progress-bar") + 1], "raw");
    assert.equal(args[args.indexOf("--timeout") + 1], "120");
    assert.equal(args[args.indexOf("--retries") + 1], "3");
    assert.equal(bounds.timeout, 60 * 60 * 1000);
    assert.ok(args.includes("torch==2.7.1+cu128"));
    assert.ok(args.includes("torchvision==0.22.1+cu128"));
    assert.ok(!args.includes("--no-cache-dir"));
  }
  assert.equal(calls[0][1][calls[0][1].indexOf("--index-url") + 1], "https://download.pytorch.org/whl/cu128");
  assert.deepEqual(calls[1][1].filter(arg => arg.includes("==")), Object.entries(lock.required_packages).map(([name, version]) => `${name}==${version}`));
  for (const failAt of [1, 2]) {
    let attempts = 0;
    const failure = new Error("fixture socket timeout");
    await assert.rejects(() => installUpstreamPackages("fixture-python", lock, async () => {
      if (++attempts === failAt) throw failure;
    }), error => error === failure);
    assert.equal(attempts, failAt, "failed install must not restart or advance to the next stage");
  }
  let attempts = 0;
  await assert.rejects(() => installUpstreamPackages("fixture-python", { ...lock, torch_index_url: "https://wrong.invalid" }, async () => { attempts++; }), /locked CUDA/);
  assert.equal(attempts, 0);
});


test("upstream pip diagnostics reflect only numeric progress and public locked wheel identity", () => {
  assert.deepEqual(upstreamPipProgress("Progress 123 of 456"), { event: "download", bytes: 123, total_bytes: 456 });
  assert.equal(upstreamPipProgress("Progress 457 of 456"), null);
  assert.equal(upstreamPipProgress("Progress 9007199254740993 of 0"), null);
  assert.deepEqual(upstreamPipProgress("Downloading https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl?token=FIXTURE_PRIVATE"), { event: "download_started", package: "torch" });
  for (const line of ["Authorization: Bearer FIXTURE_PRIVATE", "PIP_INDEX_URL=FIXTURE_PRIVATE", "config.json {secret: FIXTURE_PRIVATE}", "Downloading https://user:FIXTURE_PRIVATE@download.pytorch.org/a.whl", "Downloading https://private.invalid/FIXTURE_PRIVATE.whl", "ERROR: FIXTURE_PRIVATE"]) assert.equal(upstreamPipProgress(line), null);
});

test("upstream pip child reports byte progress while suppressing secret-like stdout and stderr", async () => {
  const events = [];
  const script = `process.stdout.write('Authorization: Bearer FIXTURE_PRIVATE\\nProgress 1');
    process.stderr.write('config.json FIXTURE_PRIVATE\\n');
    setTimeout(() => process.stdout.write(' of 10\\n'), 30);
    setTimeout(() => process.stdout.write('Progress 10 of 10\\nInstalling collected packages: fixture\\n'), 80);
    setTimeout(() => process.exit(0), 140);`;
  await runUpstreamPip(process.execPath, ["-e", script], { timeout: 3000, maxBuffer: 4096 }, { emit: line => events.push(JSON.parse(line)), heartbeatMs: 20, noProgressMs: 1000 });
  assert.ok(events.some(event => event.last_progress?.bytes === 1));
  assert.equal(events.at(-1).event, "completed");
  assert.equal(events.at(-1).last_transfer.bytes, 10);
  assert.doesNotMatch(JSON.stringify(events), /FIXTURE_PRIVATE|Authorization|config.json/);
});

test("upstream pip watchdog closes its owned child on repeated or decreasing transfer counts", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-pip-watchdog-"));
  try {
    for (const [name, script, reason] of [
      ["stalled", "let count=0; setInterval(() => console.log('Progress '+(++count%2)+' of 10'), 20);", "download_stalled"],
      ["unsupported", "setInterval(() => console.log('Unrecognized progress FIXTURE_PRIVATE'), 20);", "no_observable_download_progress"],
    ]) {
      const pidPath = path.join(root, name), events = [];
      const code = `require('node:fs').writeFileSync(${JSON.stringify(pidPath)}, String(process.pid)); ${script}`;
      await assert.rejects(() => runUpstreamPip(process.execPath, ["-e", code], { timeout: 5000, maxBuffer: 16384 }, { emit: line => events.push(JSON.parse(line)), heartbeatMs: 30, noProgressMs: 600 }), new RegExp(reason));
      assert.equal(events.at(-1).failure, reason);
      assert.equal(events.at(-1).killed, true);
      const pid = Number(await readFile(pidPath, "utf8"));
      assert.throws(() => process.kill(pid, 0), /ESRCH/);
      assert.doesNotMatch(JSON.stringify(events), /FIXTURE_PRIVATE/);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("upstream pip process deadline remains explicit after installation starts", async () => {
  const events = [];
  const script = "console.log('Installing collected packages: fixture'); setInterval(() => {}, 100);";
  await assert.rejects(() => runUpstreamPip(process.execPath, ["-e", script], { timeout: 1000, maxBuffer: 4096 }, { emit: line => events.push(JSON.parse(line)), heartbeatMs: 20, noProgressMs: 300 }), /process_deadline/);
  assert.equal(events.at(-1).failure, "process_deadline");
  assert.equal(events.at(-1).killed, true);
});
