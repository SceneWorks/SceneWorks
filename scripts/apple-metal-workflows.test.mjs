// Source-only launch/lifetime contracts; no workflows, models or device probes are executed.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { execFileSync, spawnSync } from "node:child_process";
const source = (name) => readFile(new URL(name, import.meta.url), "utf8");
function job(workflow, name) {
  const start = workflow.indexOf(`\n  ${name}:`);
  assert.ok(start >= 0, `${name} entrypoint exists`);
  const next = workflow.slice(start + 1).search(/\n  [a-z][a-z0-9-]*:/);
  return workflow.slice(start, next < 0 ? undefined : start + 1 + next);
}
function assertTupleLifetimes(workflow) {
  assert.match(workflow, /group: starvector-terminal-\$\{\{ github.run_id \}\}-\$\{\{ github.run_attempt \}\}/);
  for (const backend of ["mlx", "cuda"]) for (const tier of ["1b", "8b"]) {
    const item = job(workflow, `${backend}-${tier}`), budget = item.match(/\n    timeout-minutes: ([0-9]+)/)?.[1];
    const line = item.split("\n").find((line) => line.includes("scripts/starvector-terminal-owned-tuple.mjs"));
    assert.ok(line, "one closed start/assets/bundle/produce/stop transaction must own the actual entrypoint");
    assert.equal(item.match(/scripts\/starvector-terminal-owned-tuple\.mjs/g)?.length, 1);
    assert.doesNotMatch(item, /product-service\.mjs (?:start|stop)/, "no separate start-only guard or duplicate stop");
    assert.ok(line.endsWith(` ${budget}`), "preserve each job's original budget, not a new model limit");
    assert.ok(line.includes(backend === "mlx" ? `mlx:${tier}` : `candle-cuda:${tier}`), "preserve actual tuple/model selection");
    if (backend === "cuda") {
      assert.match(line, /windows_cuda_execution\.py --actor starvector-cuda-(?:1b|8b) --budget-minutes [0-9]+ --service-url "\$env:STARVECTOR_TERMINAL_API_URL" --physical-indices 0 -- node scripts\/starvector-terminal-owned-tuple\.mjs/,
        "the actual service port and source-forced NVML0 UUID must guard the WHOLE Windows child lifetime");
    }
  }
  assert.match(job(workflow, "upstream-reference"), /windows_cuda_execution\.py --actor starvector-upstream --budget-minutes [0-9]+ --physical-indices 0 -- node scripts\/starvector-terminal-upstream\.mjs run/);
}
test("all terminal GPU tuples hold the actual GPU and API-port resource through the one old stop operation", async () => {
  const workflow = await source("../.github/workflows/starvector-terminal.yml");
  assertTupleLifetimes(workflow);
  assert.throws(() => assertTupleLifetimes(workflow.replace(/--service-url "\$env:STARVECTOR_TERMINAL_API_URL" /, "")), /actual service port/);
  assert.throws(() => assertTupleLifetimes(workflow.replace("--physical-indices 0 -- node scripts/starvector-terminal-owned-tuple.mjs", "--cuda-ordinals 1 -- node scripts/starvector-terminal-owned-tuple.mjs")), /actual service port/);
  assert.throws(() => assertTupleLifetimes(workflow.replace("scripts/starvector-terminal-owned-tuple.mjs", "scripts/starvector-terminal-product-service.mjs start")), /closed start/);
  const launcher = await source("./starvector-terminal-owned-tuple.mjs");
  assert.match(launcher, /await withMacMetalExecutionLease\(budget \* 60, async \(\) => \{/);
  assert.ok(launcher.indexOf("await withMacMetalExecutionLease") < launcher.indexOf('process.on("SIGINT", interrupt)'), "queued cancellation must not later launch a model");
});
test("NAX kernels execute only in the dedicated approved manual workflow", async () => {
  const workflow = await source("../.github/workflows/macos-mlx.yml");
  const manual = await source("../.github/workflows/nax-kernel-validation.yml");
  const ordinary = job(workflow, "nax-worker");
  assert.match(ordinary, /run: cargo test -p sceneworks-worker --test nax_guard --no-run/);
  assert.doesNotMatch(workflow, /unit-hf-sandbox\.mjs -- cargo test -p sceneworks-worker --test nax_guard/);
  const dispatch = job(workflow, "nax-kernel-validation");
  assert.match(dispatch, /github\.event_name == 'workflow_dispatch'/);
  assert.match(dispatch, /inputs\.verification_profile == 'nax-kernel-guard'/);
  assert.match(dispatch, /uses: \.\/\.github\/workflows\/nax-kernel-validation\.yml/);
  assert.match(dispatch, /source_sha: \$\{\{ inputs\.source_sha \}\}/);
  assert.match(dispatch, /mac_go: \$\{\{ inputs\.mac_go \}\}/);
  const triggers = manual.slice(manual.indexOf("\non:"), manual.indexOf("\npermissions:"));
  assert.match(triggers, /workflow_dispatch:/);
  assert.match(triggers, /workflow_call:/);
  assert.doesNotMatch(triggers, /\n  (?:push|pull_request|schedule|merge_group):/);
  assert.match(manual, /github\.event_name == 'workflow_dispatch'/);
  assert.match(manual, /runs-on: \[self-hosted, macOS, ARM64, nax\]/);
  assert.match(manual, /CARGO_TARGET_DIR: \$\{\{ github\.workspace \}\}\/\.\.\/cargo-target/);
  assert.match(manual, /fetch-prebuilt-mlx\.sh --github-env/);
  assert.doesNotMatch(manual, /MACOSX_DEPLOYMENT_TARGET:|MACOSX_DEPLOYMENT_TARGET=/);
  assert.match(manual, /bind-metallib\.mjs .* --debug-worker/);
  assert.match(manual, /metal_execution_lock\.py --wait-seconds 2700 --/);
  assert.match(manual, /unit-hf-sandbox\.mjs -- cargo test -p sceneworks-worker --test nax_guard/);
  assert.match(manual, /tee "\$RUNNER_TEMP\/nax-guard\.log"/);
  assert.match(manual, /cancel-in-progress: false/);
  assert.match(workflow, /workflow_dispatch' && format\('manual-\{0\}-\{1\}', github.run_id, github.run_attempt\)/);
  assert.match(workflow, /github.event_name == 'pull_request' && github.ref \|\| github.sha/);
  assert.match(ordinary, /bind-metallib\.mjs .* --debug-worker/);
  assert.equal((ordinary.match(/manual-native-release-build\.jsonl" --print-path/g) ?? []).length, 2);
});

test("NAX manual startup rejects missing approval and mismatched source before native setup", async () => {
  const manual = await source("../.github/workflows/nax-kernel-validation.yml");
  const step = manual.match(/- name: Require exact selected source and explicit GPU approval before native setup[\s\S]*?run: \|\n([\s\S]*?)(?=\n      - )/);
  assert.ok(step);
  const body = step[1].replace(/^          /gm, "");
  assert.ok(manual.indexOf(step[0]) < manual.indexOf("dtolnay/rust-toolchain@"));
  const cwd = new URL("..", import.meta.url);
  const sha = execFileSync("git", ["rev-parse", "HEAD"], { cwd, encoding: "utf8" }).trim();
  const valid = { ...process.env, SELECTED_SOURCE: sha, DISPATCH_SOURCE: sha, MAC_GO: "true" };
  const run = patch => spawnSync("bash", ["-euc", body], { cwd, env: { ...valid, ...patch }, encoding: "utf8" });
  assert.equal(run({}).status, 0);
  for (const patch of [{ MAC_GO: "false" }, { MAC_GO: "" }, { SELECTED_SOURCE: "main" }, { DISPATCH_SOURCE: "0".repeat(40) }, { SELECTED_SOURCE: "0".repeat(40), DISPATCH_SOURCE: "0".repeat(40) }]) {
    assert.notEqual(run(patch).status, 0, JSON.stringify(patch));
  }
});

test("Iris and app lifecycle retain every ready request and never upload staged executables/shaders", async () => {
  for (const name of ["iris-mlx-real-weights", "qwen-image-2-1-app-lifecycle", "supra2-mlx-real-weights"]) {
    const workflow = await source(`../.github/workflows/${name}.yml`);
    assert.ok(workflow.includes(`group: ${name}-\${{ github.run_id }}-\${{ github.run_attempt }}`));
    assert.doesNotMatch(workflow, /path:.*memory-catalog-metallib-/);
  }
  const iris = await source("./ci/iris-mlx-real-weights/run.sh");
  assert.match(iris, /WORK_DIR_NATIVE="\$out" node "\$native_binding_script"/);
  assert.doesNotMatch(iris, /-mindepth 1 -maxdepth 1 -type d ! -name/);
  const app = await source("./qwen-image-2-1-app-lifecycle.mjs");
  assert.match(app, /childEnv\.PMETAL_METALLIB_PATH = native.path/);
  assert.ok(app.indexOf("await bindMetallib(") < app.indexOf('"metal-preflight"'));
  const service = await source("./starvector-terminal-product-service.mjs");
  assert.match(service, /serviceEnv\.PMETAL_METALLIB_PATH = nativeMetallib/);
  assert.match(service, /serviceEnv\.PMETAL_METALLIB_PATH = await readBoundAppMetallib/);
});

test("merged Supra2 model invocations lease Metal before memory admission and keep their selected native artifact", async () => {
  const script = await source('./ci/supra2-mlx-real-weights/run.sh');
  assert.match(script, /--supra2-test --print-path/);
  assert.match(script, /bin="\$\(dirname "\$PMETAL_METALLIB_PATH"\)\/sceneworks_worker"/);
  assert.match(script, /export SCENEWORKS_METAL_ENTRY_SCRIPT out mem_gib bin guard_py hub bundle/);
  assert.match(script, /metal_execution_lock\.py/);
  const held = script.indexOf('metal_execution_entry "$limit_s" true');
  assert.ok(held >= 0 && held < script.indexOf('local log="$out/$name.log"'));
  assert.doesNotMatch(script, /-mindepth 1 -maxdepth 1 -type d ! -name/);
});

test("the newly wrapped campaign walk cannot invoke the broad-kill preflight", async () => {
  for (const file of ["../.github/workflows/memory-catalog-campaign.yml", "./ci/memory-catalog/run.sh", "./ci/memory-catalog/inference.sh"]) {
    assert.doesNotMatch(await source(file), /gpu-preflight\.sh/);
  }
});
