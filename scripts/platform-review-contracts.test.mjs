import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function source(path) {
  return readFile(new URL(`../${path}`, import.meta.url), "utf8");
}

test("Windows workflows watch the local Rust runner action", async () => {
  for (const workflow of [
    ".github/workflows/windows-candle.yml",
    ".github/workflows/desktop-windows.yml",
  ]) {
    assert.match(
      await source(workflow),
      /^\s+- "\.github\/actions\/prepare-rust-runner\/\*\*"/m,
      workflow,
    );
  }
});

test("Windows runner prep falls back when its optional rustc wrapper is missing", async () => {
  const action = await source(".github/actions/prepare-rust-runner/action.yml");
  assert.match(action, /\$rustcWrapper = \$env:RUSTC_WRAPPER/);
  assert.match(action, /Test-Path -LiteralPath \$rustcWrapper -PathType Leaf/);
  assert.match(action, /Get-Command \$rustcWrapper -CommandType Application/);
  assert.match(
    action,
    /Add-Content -Path \$env:GITHUB_ENV -Value 'RUSTC_WRAPPER='/,
  );
});

test("Docker relevance gate paginates and checks for truncated file lists", async () => {
  const workflow = await source(".github/workflows/check.yml");
  assert.match(workflow, /gh api --paginate/);
  assert.match(workflow, /docker-smoke-relevance\.mjs --expected-count/);
  assert.doesNotMatch(workflow, /gh pr view .*--json files/);
});

test("every release job is confined to refs/tags/v", async () => {
  const workflow = await source(".github/workflows/release.yml");
  const jobConditions = [...workflow.matchAll(/^\s{4}if:\s*(.+)$/gm)].map((match) => match[1]);
  assert.equal(jobConditions.length, 3);
  for (const condition of jobConditions) {
    assert.match(condition, /startsWith\(github\.ref, 'refs\/tags\/v'\)/);
  }
  assert.ok(
    workflow.includes('if [[ "${TAG#v}" == *-* ]]; then'),
    "prerelease classification must use the validated v-tag",
  );
});

test("Lens smoke only terminates processes it started", async () => {
  const script = await source("scripts/smoke-lens.ps1");
  assert.doesNotMatch(script, /Get-Process/);
  assert.match(script, /taskkill \/F \/T \/PID \$\(\$p\.Id\)/);
});

test("health check defaults to the compose API port", async () => {
  assert.match(
    await source("scripts/check-health.mjs"),
    /http:\/\/localhost:8010/,
  );
});

test("Docker cleanup relies on the configured host uid instead of a root container", async () => {
  const script = await source("scripts/check-docker-api-runtime.mjs");
  assert.doesNotMatch(script, /--entrypoint", "rm"/);
  assert.match(script, /SCENEWORKS_UID/);
});

test("Rust Docker dependency layers include every image-memory adapter target", async () => {
  const dockerfile = await source("docker/rust.Dockerfile");
  assert.equal(
    (
      dockerfile.match(
        /COPY crates\/sceneworks-image-memory-adapter\/Cargo\.toml/g,
      ) ?? []
    ).length,
    2,
  );
  for (const target of ["src/lib.rs", "src/bin/candle.rs", "src/bin/mlx.rs"]) {
    assert.equal(
      (
        dockerfile.match(
          new RegExp(
            `crates/sceneworks-image-memory-adapter/${target.replace(".", "\\.")}`,
            "g",
          ),
        ) ?? []
      ).length,
      2,
      target,
    );
  }
});

test("all three manifest scripts import the shared JSONC parser", async () => {
  for (const scriptPath of [
    "scripts/check-scaffold.mjs",
    "scripts/check-download-patterns.mjs",
    "scripts/check-no-nc-weights.mjs",
  ]) {
    const script = await source(scriptPath);
    assert.match(script, /import \{ stripJsoncComments \} from "\.\/lib\/jsonc\.mjs";/);
    assert.doesNotMatch(script, /function stripJsoncComments/);
  }
});

test("macOS image-memory calibration dispatch is opt-in and secret-scoped", async () => {
  const workflow = await source(".github/workflows/macos-mlx.yml");
  assert.match(workflow, /run_image_memory_calibration:/);
  assert.match(
    workflow,
    /provision_qwen_snapshot:\s+description:[^\n]+\s+required: false\s+type: boolean\s+default: false/,
  );
  assert.match(
    workflow,
    /timeout-minutes: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.run_image_memory_calibration && inputs\.provision_qwen_snapshot && 240 \|\| 45 \}\}/,
  );
  assert.match(
    workflow,
    /provision_qwen_snapshot requires run_image_memory_calibration=true/,
  );
  assert.match(
    workflow,
    /QWEN_ROOT_OVERRIDE: \$\{\{ secrets\.SCENEWORKS_QWEN_IMAGE_ROOT \}\}/,
  );
  assert.doesNotMatch(workflow, /^\s+qwen_root:/m);
  assert.match(
    workflow,
    /models--SceneWorks--qwen-image-mlx\/snapshots\/\$QWEN_REVISION\/bf16/,
  );
  const huggingFaceRoot =
    "$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/$QWEN_REVISION/bf16";
  const sceneWorksRoot =
    "$HOME/Library/Application Support/SceneWorks/data/cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/$QWEN_REVISION/bf16";
  assert.equal(workflow.split(huggingFaceRoot).length - 1, 1);
  assert.equal(workflow.split(sceneWorksRoot).length - 1, 1);
  assert.ok(workflow.indexOf(huggingFaceRoot) < workflow.indexOf(sceneWorksRoot));
  assert.match(
    workflow,
    /if \[\[ -n "\$QWEN_ROOT_OVERRIDE" \]\]; then\s+QWEN_ROOT="\$QWEN_ROOT_OVERRIDE"\s+else/,
  );
  assert.match(workflow, /if \[\[ -d "\$QWEN_HF_ROOT" \]\]; then/);
  assert.match(workflow, /elif \[\[ -d "\$QWEN_APP_ROOT" \]\]; then/);
  assert.doesNotMatch(workflow, /\bfind\b.*qwen|\bls\b.*qwen/i);
  assert.match(workflow, /command -v python3\.12 >\/dev\/null/);
  assert.match(workflow, /python3\.12 --version \| grep -E '\^Python 3\\\.12\\\.'/);
  assert.match(
    workflow,
    /python3\.12 -m venv "\$RUNNER_TEMP\/qwen-provision-venv"/,
  );
  assert.match(
    workflow,
    /"\$RUNNER_TEMP\/qwen-provision-venv\/bin\/python" -m pip install/,
  );
  assert.match(workflow, /huggingface_hub==0\.36\.0/);
  assert.match(workflow, /from huggingface_hub import snapshot_download/);
  assert.match(workflow, /repo_id="SceneWorks\/qwen-image-mlx"/);
  assert.match(workflow, /revision=os\.environ\["QWEN_REVISION"\]/);
  assert.match(workflow, /allow_patterns=\["bf16\/\*\*"\]/);
  assert.match(workflow, /token=False/);
  assert.match(workflow, /HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"/);
  assert.match(workflow, /HF_HUB_DISABLE_PROGRESS_BARS: "1"/);
  assert.match(workflow, /repository_cache = os\.path\.join\(/);
  assert.match(
    workflow,
    /huggingface_cache\s+if os\.path\.isdir\(repository_cache\)\s+else application_cache/,
  );
  assert.doesNotMatch(workflow, /raise SystemExit\(0\)/);
  assert.match(
    workflow,
    /"Library",\s+"Application Support",\s+"SceneWorks",\s+"data",\s+"cache",\s+"huggingface",\s+"hub"/,
  );
  const provisioning = workflow.slice(
    workflow.indexOf("- name: Set up runner Python 3.12 for Qwen provisioning"),
    workflow.indexOf("- name: Resolve exact Qwen calibration snapshot"),
  );
  assert.doesNotMatch(
    provisioning,
    /secrets\.|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|QWEN_REPOSITORY|local_dir/,
  );
  assert.doesNotMatch(provisioning, /actions\/setup-python/);
  assert.equal(
    provisioning.split('repo_id="SceneWorks/qwen-image-mlx"').length - 1,
    1,
  );
  assert.equal(provisioning.split('allow_patterns=["bf16/**"]').length - 1, 1);
  assert.match(
    workflow,
    /QWEN_REPOSITORY" != "SceneWorks\/qwen-image-mlx"/,
  );
  assert.match(workflow, /QWEN_ROOT="\$\(cd "\$QWEN_ROOT" && pwd -P\)"/);
  assert.match(
    workflow,
    /EXPECTED_SUFFIX="\/models--SceneWorks--qwen-image-mlx\/snapshots\/\$QWEN_REVISION\/bf16"/,
  );
  assert.match(workflow, /QWEN_ROOT" != \*"\$EXPECTED_SUFFIX"/);
  assert.match(
    workflow,
    /cargo build --release --locked -p sceneworks-image-memory-adapter/,
  );
  const inferenceCheckout = workflow.slice(
    workflow.indexOf("- name: Check out the exact inference calibration source"),
    workflow.indexOf("- name: Build and run the authoritative MLX calibration adapter"),
  );
  assert.match(inferenceCheckout, /repository: SceneWorks\/inference/);
  assert.match(inferenceCheckout, /ref: \$\{\{ inputs\.inference_revision \}\}/);
  assert.match(
    inferenceCheckout,
    /token: \$\{\{ secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| github\.token \}\}/,
  );
  assert.match(
    workflow,
    /x-access-token:\$\{\{ secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| github\.token \}\}@github\.com\/SceneWorks\/inference\.insteadOf/,
  );
  assert.match(workflow, /--backend mlx/);
  assert.match(workflow, /--provider mlx-qwen-vae-decode/);
  assert.match(
    workflow,
    /image-memory-calibration-harness\.mjs check/,
  );
  assert.match(
    workflow,
    /actions\/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a/,
  );
  assert.match(
    workflow,
    /if: \$\{\{ success\(\) && github\.event_name == 'workflow_dispatch' && inputs\.run_image_memory_calibration \}\}/,
  );
});

test("MLX calibration probe derives the production wired ceiling without guessing", async () => {
  const adapter = await source(
    "crates/sceneworks-image-memory-adapter/src/bin/mlx.rs",
  );
  assert.match(adapter, /sysctl\("iogpu\.wired_limit_mb"\)/);
  assert.match(adapter, /sysctl\("kern\.memorystatus_wired_mem_limit"\)/);
  assert.match(adapter, /\.checked_mul\(1024 \* 1024\)/);
  assert.match(
    adapter,
    /u64::try_from\(mlx_default_memory_limit\)[\s\S]*?\/ 3[\s\S]*?\* 2/,
  );
  assert.match(adapter, /source: "mlx_default_memory_limit\/1\.5"/);
  assert.match(adapter, /"wiredLimitBytes": wired_limit\.bytes/);
});
