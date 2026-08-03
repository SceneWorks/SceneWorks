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

test("Windows CUDA isolates Cargo dependency checkouts after toolchain discovery", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const prepare = workflow.indexOf("uses: ./.github/actions/prepare-rust-runner");
  const isolate = workflow.indexOf("name: Isolate Cargo dependency checkout");
  const fetch = workflow.indexOf("name: Fetch the private inference release");
  assert.ok(prepare >= 0 && prepare < isolate && isolate < fetch);
  assert.match(workflow, /Join-Path \$env:RUNNER_TEMP 'cargo-home'/);
  assert.match(
    workflow,
    /Add-Content -Path \$env:GITHUB_ENV -Value "CARGO_HOME=\$jobCargoHome"/,
  );
});

test("Windows Krea provisioning accepts supported newer Python 3 runtimes", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  assert.match(workflow, /Python 3\.12 or newer/);
  assert.match(workflow, /\[int\]\$Matches\.minor -lt 12/);
  assert.doesNotMatch(workflow, /\^Python 3\\\.12\\\./);
  assert.match(
    workflow,
    /token: \$\{\{ secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| github\.token \}\}/,
  );
});

test("Windows CUDA runs the Candle adapter's platform-only unit tests", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  assert.match(
    workflow,
    /cargo test -p sceneworks-memory-adapter --features candle --bin memory-candle-adapter/,
  );
  assert.match(workflow, /console\.log\(JSON\.stringify\(a,null,2\)\)/);
  assert.match(workflow, /'amortizable','unable_to_amortize'/);
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

test("Rust Docker dependency layers include every memory-strategy adapter target", async () => {
  const dockerfile = await source("docker/rust.Dockerfile");
  assert.equal(
    (
      dockerfile.match(
        /COPY crates\/sceneworks-memory-adapter\/Cargo\.toml/g,
      ) ?? []
    ).length,
    2,
  );
  for (const target of ["src/lib.rs", "src/bin/candle.rs", "src/bin/mlx.rs"]) {
    assert.equal(
      (
        dockerfile.match(
          new RegExp(
            `crates/sceneworks-memory-adapter/${target.replace(".", "\\.")}`,
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

test("macOS memory-strategy calibration dispatch is opt-in and secret-scoped", async () => {
  const workflow = await source(".github/workflows/macos-mlx.yml");
  assert.match(workflow, /run_memory_calibration:/);
  assert.match(
    workflow,
    /provision_qwen_snapshot:\s+description:[^\n]+\s+required: false\s+type: boolean\s+default: false/,
  );
  assert.match(
    workflow,
    /qwen_tier:\s+description:[^\n]+\s+required: false\s+type: choice\s+options:\s+- bf16\s+- q4\s+- q8\s+default: bf16/,
  );
  assert.match(
    workflow,
    /timeout-minutes: \$\{\{ github\.event_name == 'workflow_dispatch' && \(\(inputs\.run_memory_calibration && inputs\.provision_qwen_snapshot\) \|\| \(inputs\.run_five_rung_reference && inputs\.provision_z_image_snapshot\)\) && 240 \|\| github\.event_name == 'workflow_dispatch' && inputs\.run_five_rung_reference && 120 \|\| 45 \}\}/,
  );
  assert.match(
    workflow,
    /provision_qwen_snapshot requires run_memory_calibration=true/,
  );
  assert.match(
    workflow,
    /QWEN_ROOT_OVERRIDE: \$\{\{ secrets\.SCENEWORKS_QWEN_IMAGE_ROOT \}\}/,
  );
  assert.doesNotMatch(workflow, /^\s+qwen_root:/m);
  assert.match(
    workflow,
    /models--SceneWorks--qwen-image-mlx\/snapshots\/\$QWEN_REVISION\/\$QWEN_TIER/,
  );
  const huggingFaceRoot =
    "$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/$QWEN_REVISION/$QWEN_TIER";
  const sceneWorksRoot =
    "$HOME/Library/Application Support/SceneWorks/data/cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/$QWEN_REVISION/$QWEN_TIER";
  assert.equal(workflow.split(huggingFaceRoot).length - 1, 1);
  assert.equal(workflow.split(sceneWorksRoot).length - 1, 1);
  assert.ok(workflow.indexOf(huggingFaceRoot) < workflow.indexOf(sceneWorksRoot));
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
  assert.match(workflow, /allow_patterns=\[f"\{os\.environ\['QWEN_TIER'\]\}\/\*\*"\]/);
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
  assert.equal(
    provisioning.split('allow_patterns=[f"{os.environ[\'QWEN_TIER\']}/**"]').length - 1,
    1,
  );
  assert.match(
    workflow,
    /QWEN_REPOSITORY" != "SceneWorks\/qwen-image-mlx"/,
  );
  assert.match(workflow, /QWEN_ROOT="\$\(cd "\$QWEN_ROOT" && pwd -P\)"/);
  assert.match(
    workflow,
    /QWEN_OVERRIDE_ROOT="\$\(cd "\$QWEN_ROOT_OVERRIDE" && pwd -P\)"[\s\S]*if \[\[ "\$QWEN_OVERRIDE_ROOT" == \*"\$EXPECTED_SUFFIX" \]\]; then[\s\S]*QWEN_ROOT="\$QWEN_OVERRIDE_ROOT"/,
  );
  assert.match(workflow, /if \[\[ -z "\$QWEN_ROOT" \]\]; then[\s\S]*QWEN_HF_ROOT=/);
  assert.match(
    workflow,
    /EXPECTED_SUFFIX="\/models--SceneWorks--qwen-image-mlx\/snapshots\/\$QWEN_REVISION\/\$QWEN_TIER"/,
  );
  assert.match(workflow, /QWEN_ROOT" != \*"\$EXPECTED_SUFFIX"/);
  assert.match(
    workflow,
    /cargo build --release --locked -p sceneworks-memory-adapter/,
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
  assert.match(workflow, /QWEN_SEED=15511/);
  assert.match(workflow, /QWEN_SEED=16353/);
  assert.match(workflow, /--fixture "qwen-image-\$\{QWEN_TIER\}-seed\$\{QWEN_SEED\}-step2"/);
  assert.match(workflow, /--fresh-per-case/);
  assert.match(
    workflow,
    /memory-calibration-harness\.mjs check/,
  );
  assert.match(
    workflow,
    /actions\/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a/,
  );
  assert.match(
    workflow,
    /if: \$\{\{ success\(\) && github\.event_name == 'workflow_dispatch' && inputs\.run_memory_calibration \}\}/,
  );
});

test("MLX calibration probe derives the production wired ceiling without guessing", async () => {
  const adapter = await source(
    "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  );
  assert.match(adapter, /sysctl\("iogpu\.wired_limit_mb"\)/);
  assert.match(adapter, /sysctl\("kern\.memorystatus_wired_mem_limit"\)/);
  assert.match(adapter, /\.checked_mul\(1024 \* 1024\)/);
  assert.match(
    adapter,
    /u64::try_from\(mlx_default_memory_limit\)[\s\S]*?\/ 3[\s\S]*?\* 2/,
  );
  assert.match(adapter, /source: "mlx_default_memory_limit\/1\.5"/);
  const probe = adapter.slice(
    adapter.indexOf("fn probe()"),
    adapter.indexOf("#[cfg(test)]"),
  );
  assert.equal(probe.match(/get_memory_limit\(\)/g)?.length, 1);
  assert.match(
    probe,
    /let mlx_default_memory_limit = get_memory_limit\(\);/,
  );
  assert.match(
    probe,
    /"mlxMemoryLimitBytes": mlx_default_memory_limit/,
  );
  assert.match(
    adapter,
    /SCENEWORKS_MLX_WIRED_LIMIT_BYTES must be greater than zero/,
  );
  assert.match(adapter, /"wiredLimitBytes": wired_limit\.bytes/);
});

test("memory adapters bind every emitted overlay verdict to the requested target", async () => {
  const mlx = await source("crates/sceneworks-memory-adapter/src/bin/mlx.rs");
  const candle = await source("crates/sceneworks-memory-adapter/src/bin/candle.rs");
  const krea = mlx.slice(mlx.indexOf("fn run_krea_control("), mlx.indexOf("fn run_qwen("));
  const qwen = mlx.slice(mlx.indexOf("fn run_qwen_vae_probe("), mlx.indexOf("fn run(request:"));
  const candleReference = candle.slice(
    candle.indexOf("fn run_five_rung_reference("),
    candle.indexOf("fn run(request:"),
  );

  assert.match(
    krea,
    /validate_exact_overlay_target\(request, "control:1", KREA_CONTROL_EXECUTION_PATH\)\?/,
  );
  assert.ok(
    krea.indexOf("validate_exact_overlay_target") < krea.indexOf("let parameters"),
    "Krea must reject a mismatched target before provider work",
  );
  assert.equal(qwen.match(/protocol::plain_gated_fragment\(/g)?.length, 2);
  assert.doesNotMatch(qwen, /protocol::gated_fragment\(/);
  assert.match(
    qwen,
    /validate_plain_overlay_target\(request, QWEN_PLAIN_EXECUTION_PATH\)\?/,
  );
  assert.match(
    qwen,
    /validate_plain_overlay_target\(request, QWEN_PROVIDER_EXECUTION_PATH\)\?/,
  );
  assert.match(
    candleReference,
    /let execution_path = plain_execution_path\(request\)\?;[\s\S]*validate_plain_overlay_target\(request, execution_path\)\?/,
  );
  assert.ok(
    candleReference.indexOf("validate_plain_overlay_target") < candleReference.indexOf("load_five_rung_generator"),
    "the Candle five-rung reference must reject a mismatched overlay before provider work",
  );
  assert.ok(
    candleReference.indexOf("for item in planned") <
      candleReference.lastIndexOf("load_five_rung_generator(&first_request)?"),
    "the Candle batch must validate every target before its one model load",
  );
  assert.equal(candle.match(/protocol::plain_gated_fragment\(/g)?.length, 3);
  assert.doesNotMatch(candle, /protocol::gated_fragment\(/);
});

test("MLX parity control preserves the real comparison and applies only a planned output bias", async () => {
  const adapter = await source(
    "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  );
  assert.match(
    adapter,
    /let \(actual_maximum, actual_mean\) = decoded_max_mean_abs\(&baseline, &tiled, None\)\?/,
  );
  assert.match(
    adapter,
    /decoded_max_mean_abs\(&baseline, &tiled, comparison_output_bias\)\?/,
  );
  assert.match(adapter, /"result": "passed",\s*"maximumError": actual_maximum,\s*"meanError": actual_mean/);
  assert.match(adapter, /"maximumError": mutated_maximum,\s*"meanError": mutated_mean/);
  assert.doesNotMatch(adapter, /MAX_THRESHOLD\s*=\s*[^;]*5e-2/);
  assert.doesNotMatch(adapter, /MEAN_THRESHOLD\s*=\s*[^;]*5e-2/);
  const comparison = adapter.slice(
    adapter.indexOf("fn decoded_max_mean_abs("),
    adapter.indexOf("fn sweep("),
  );
  const shapeGuard = comparison.indexOf(
    "protocol::validate_comparison_shapes(left.shape(), right.shape())?",
  );
  const flatten = comparison.indexOf(".reshape(&[-1])");
  assert.ok(shapeGuard >= 0, "MLX comparison must guard exact output shapes");
  assert.ok(flatten > shapeGuard, "shape equality must be checked before either output is flattened");
});

test("Krea lifecycle cleanup uses the established warm follow-up peak contract", async () => {
  const adapter = await source(
    "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  );
  assert.match(
    adapter,
    /lifecycle_control_peak\.saturating_add\(lifecycle_control_peak \/ 50\)/,
  );
  assert.match(adapter, /recovery_peak > lifecycle_recovery_limit/);
  assert.match(
    adapter,
    /"lifecycleWarmControlPeak", "bytes", lifecycle_control_peak/,
  );
  assert.match(
    adapter,
    /"lifecycleMaximumRecoveryPeak", "bytes", lifecycle_max_recovery_peak/,
  );
});

test("Z-Image cleanup attestation bounds retained bytes and recovery peaks against a clean warm control", async () => {
  const adapter = await source(
    "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  );
  assert.match(
    adapter,
    /LifecycleMemoryBounds::from_clean_warm\(\s*lifecycle_clean_warm_peak,\s*lifecycle_clean_post_cleanup/,
  );
  assert.match(adapter, /lifecycle_bounds\.allows_retained\(cancel_post_cleanup\)/);
  assert.match(adapter, /lifecycle_bounds\.allows_retained\(error_post_cleanup\)/);
  assert.match(adapter, /lifecycle_bounds\.allows_warm_peak\(cancel_recovery_peak\)/);
  assert.match(adapter, /lifecycle_bounds\.allows_warm_peak\(error_recovery_peak\)/);
  for (const measurement of [
    "lifecycleCleanWarmPeak",
    "lifecycleCleanPostCleanupActive",
    "lifecycleCleanPostCleanupCache",
    "lifecycleMaximumFaultPostCleanupActive",
    "lifecycleMaximumFaultPostCleanupCache",
    "lifecycleMaximumRecoveryPeak",
  ]) {
    assert.match(adapter, new RegExp(`"${measurement}", "bytes"`), measurement);
  }
});

test("the Rust gate verifies the generated docs derived from Rust sources", async () => {
  // sc-16268: `check:memory-matrix`, `check:calibration-cost-model` and `check:tier-integrity` all
  // read Rust sources, but lived only in `npm run check` — so a Rust-only change passed the gate
  // contributors are told to run and failed `parity` in CI. The fix is one string in `rust:check`,
  // which is exactly the kind of wiring a later edit silently undoes; this pins it.
  const scripts = JSON.parse(await source("package.json")).scripts;
  for (const sub of [
    "check:memory-matrix",
    "check:calibration-cost-model",
    "check:tier-integrity",
  ]) {
    assert.match(scripts["check:rust-derived-docs"], new RegExp(`\\b${sub}\\b`), sub);
  }
  assert.match(scripts["rust:check"], /\bcheck:rust-derived-docs\b/);
  assert.match(scripts.check, /\bcheck:rust-derived-docs\b/);
  // The pre-push hook runs it too, on the same trigger as the neither/candle builds.
  assert.match(await source("scripts/git-hooks/pre-push"), /npm run --silent check:rust-derived-docs/);
});

test("macOS lanes lint every crate they ship, in the configuration they ship it", async () => {
  // sc-17026. `d26d818b` added a Candle-only helper with no `cfg`; on macOS every call
  // site compiled out and `-D warnings` promoted the dead code to a hard error, so
  // `origin/main` would not build on Apple Silicon.
  //
  // The Linux `parity` lane CANNOT catch that class of bug: image_jobs.rs gates
  // `include!("image_jobs/base.rs")` on `any(macos, all(not(macos), backend-candle))`,
  // so on Linux without candle the file is not compiled at all. Verified by reverting
  // the sc-17007 `cfg` and running each lane's own command — macOS exits 101, the
  // Linux target exits 0. macOS+default is the ONLY configuration in the fleet that
  // compiles it bare, which makes these lanes the sole coverage for it.
  //
  // macos-mlx.yml did catch it. But it was scoped to `-p sceneworks-worker`, so the
  // other crates a Mac ships were dark. Both halves are pinned here.
  const mlx = await source(".github/workflows/macos-mlx.yml");
  assert.match(mlx, /^\s+run: cargo clippy --all-targets -- -D warnings$/m);
  // `npm run rust:check` runs a bare `cargo test` too, so the ordinary lane must not
  // narrow either half back to the single crate. A manual exact-pin Z-Image recapture
  // may skip only its circular evidence-currentness assertion until the emitted records
  // are ingested; the following ordinary PR run exercises that assertion again.
  assert.match(mlx, /RECAPTURE_Z_IMAGE: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.run_five_rung_reference \}\}/);
  assert.match(
    mlx,
    /cargo test -- --skip shipped_z_image_manifest_admits_all_five_current_exact_mlx_ladder_rungs/,
  );
  assert.match(mlx, /else\n\s+cargo test\n\s+fi/);
  assert.doesNotMatch(mlx, /cargo test -p sceneworks-worker/);
  // Running the command is only half of it — the lane must actually TRIGGER for the
  // crates it now lints. apps/rust-api carries the largest macOS-conditional surface
  // outside the worker and is NOT under `crates/**`, so without this path entry the
  // widened lint is declared but unreachable for rust-api-only PRs. That is the same
  // declaration-without-reachability trap as the bug this story came from.
  // Lint must come BEFORE test. A failed step aborts the job, so with the tests first
  // any unrelated red test skips the lint entirely — observed on PR #2078's first run,
  // where an inherited mlx_fit_gate failure (sc-17037) meant the widened clippy never
  // executed. Lint coverage gated behind a fully green test suite is not coverage.
  const clippyAt = mlx.indexOf("run: cargo clippy --all-targets -- -D warnings");
  const testAt = mlx.indexOf("            cargo test\n");
  assert.ok(clippyAt > 0, "macos-mlx.yml must lint every default member");
  assert.ok(testAt > 0, "macos-mlx.yml must run the workspace tests");
  assert.ok(clippyAt < testAt, "macos-mlx.yml must run clippy BEFORE cargo test");

  const mlxPaths = mlx.slice(mlx.indexOf("paths: &mlx_paths"), mlx.indexOf("pull_request:"));
  assert.ok(mlxPaths.length > 0, "macos-mlx.yml must declare an &mlx_paths anchor");
  for (const watched of ['- "crates/**"', '- "apps/rust-api/**"', '- "apps/rust-worker/**"']) {
    assert.ok(mlxPaths.includes(watched), `&mlx_paths must watch ${watched}`);
  }
  // The narrow spelling this replaced. `--all-targets` on one crate is not coverage of
  // the default-member set, and re-scoping it would silently re-dark the other crates.
  assert.doesNotMatch(mlx, /cargo clippy -p sceneworks-worker/);

  // apps/desktop is excluded from default-members, so the step above cannot reach it.
  const desktop = await source(".github/workflows/desktop-macos-check.yml");
  assert.match(desktop, /^\s+run: npm run desktop:check$/m);
  assert.match(desktop, /^\s+run: node apps\/desktop\/scripts\/stage-test-sidecars\.mjs$/m);
  assert.match(desktop, /^\s+runs-on: macos-/m);
  // A `pull_request:` trigger, or this lane is decoration. Match the key wherever it
  // sits under `on:` — pinning it to the FIRST key would fail on a harmless reorder.
  assert.match(desktop, /^ {2}pull_request:$/m);
});

test("the macOS desktop typecheck lane overrides the MLX deployment-target pin", async () => {
  // sc-17026. /.cargo/config.toml pins MACOSX_DEPLOYMENT_TARGET=26.2 for MLX's NAX
  // Metal kernels. apps/desktop links neither, and a hosted macOS image tops out below
  // 26.2 — without the documented 15.0 override this lane cannot build at all. Cargo
  // only applies its `[env]` value when the variable is unset, so the override must be
  // an actual export, and the lane must watch the file carrying the pin.
  const desktop = await source(".github/workflows/desktop-macos-check.yml");
  assert.match(desktop, /^\s+MACOSX_DEPLOYMENT_TARGET: "15\.0"$/m);
  assert.match(desktop, /^\s+- "\.cargo\/config\.toml"$/m);
  assert.match(await source(".cargo/config.toml"), /MACOSX_DEPLOYMENT_TARGET = "26\.2"/);
});

test("the MLX memory adapter is guarded on a PR lane, like its Candle twin", async () => {
  // sc-17026. sceneworks-memory-adapter is excluded from `default-members`, so the
  // workspace clippy step cannot reach it. Its Candle bin is compile-guarded on a real
  // PR lane (windows-candle.yml); the MLX bin was built ONLY inside macos-mlx.yml's
  // `workflow_dispatch` calibration path, so a break in the authoritative MLX
  // calibration adapter was invisible until someone dispatched a run.
  //
  // `clippy ... -D warnings`, not `cargo check`: the defect class here is a WARNING
  // (a symbol whose cfg'd call sites compiled out), which `check` cannot fail on. This
  // is the only place -D warnings reaches this crate on any platform/feature set, so
  // a downgrade back to `check` would silently reopen the hole.
  const mlx = await source(".github/workflows/macos-mlx.yml");
  assert.match(
    mlx,
    /run: cargo clippy -p sceneworks-memory-adapter --features mlx --all-targets -- -D warnings/,
  );
  // windows-candle.yml is the Candle twin's PR guard. server-candle-linux.yml also
  // checks that bin but is `workflow_dispatch`-only, so it is deliberately NOT cited
  // as PR coverage — asserting that keeps a future reader from repeating the mistake.
  assert.match(
    await source(".github/workflows/windows-candle.yml"),
    /cargo check -p sceneworks-memory-adapter --features candle --bin memory-candle-adapter/,
  );
  assert.match(await source(".github/workflows/server-candle-linux.yml"), /^on:\n {2}workflow_dispatch:/m);
  // The guard must sit on the unconditional PR path, not behind the dispatch-only
  // calibration steps — that placement is exactly the hole this closes.
  const guard = mlx.indexOf("cargo clippy -p sceneworks-memory-adapter");
  const firstDispatchOnly = mlx.indexOf("if: ${{ github.event_name == 'workflow_dispatch'");
  assert.ok(guard > 0, "macos-mlx.yml must clippy the MLX memory adapter");
  assert.ok(firstDispatchOnly > 0, "macos-mlx.yml must still have dispatch-only calibration steps");
  assert.ok(guard < firstDispatchOnly, "MLX adapter guard must precede the dispatch-only steps");
});
