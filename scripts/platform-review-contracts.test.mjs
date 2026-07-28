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
});
