// Source-only launch/lifetime contracts; no workflows, models or device probes are executed.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
test("self-hosted Metal kernel step is leased; hosted isolated macOS job remains unchanged", async () => {
  const workflow = await source("../.github/workflows/macos-mlx.yml");
  const original = "cargo test -p sceneworks-worker --test nax_guard";
  const guard = new RegExp("python3\\.12 -I scripts/ci/metal_execution_lock\\.py --wait-seconds [0-9]+ --\\n\\s+" + original);
  assert.match(workflow, guard);
  assert.match(workflow, /workflow_dispatch' && format\('manual-\{0\}-\{1\}', github.run_id, github.run_attempt\)/);
  assert.match(workflow, /github.event_name == 'pull_request' && github.ref \|\| github.sha/);
  assert.match(job(workflow, "nax-worker"), /bind-metallib\.mjs .* --debug-worker/);
  assert.equal((job(workflow, "nax-worker").match(/manual-native-release-build\.jsonl" --print-path/g) ?? []).length, 2,
    "both existing manual captures bind their actual selected release artifact before execution");
  assert.ok(workflow.includes("macos-26"));
  const calls = workflow.split("\n").filter((line) => line.includes("metal_execution_lock.py"));
  assert.equal(calls.length, 1, "only the shared self-hosted raw-kernel step gets this lease");
  assert.throws(() => assert.match(workflow.replace("metal_execution_lock.py", "unguarded.py"), guard));
});
test("Iris and app lifecycle retain every ready request and never upload staged executables/shaders", async () => {
  for (const name of ["iris-mlx-real-weights", "qwen-image-2-1-app-lifecycle"]) {
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

test("the newly wrapped campaign walk cannot invoke the broad-kill preflight", async () => {
  for (const file of ["../.github/workflows/memory-catalog-campaign.yml", "./ci/memory-catalog/run.sh", "./ci/memory-catalog/inference.sh"]) {
    assert.doesNotMatch(await source(file), /gpu-preflight\.sh/);
  }
});
