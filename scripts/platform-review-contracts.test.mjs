import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function source(path) {
  return readFile(new URL(`../${path}`, import.meta.url), "utf8");
}

// GitHub filter-pattern syntax: `*` matches any run of characters except `/`, `**` matches any
// run including `/`. `**/` is treated as "zero or more directories", matching the convention
// GitHub's own `**/README.md` example implies. That reading is also the SAFE one for every guard
// below: it makes `config/engine-capabilities/**/*` count as matching a file directly in that
// directory, so an ambiguous pattern is rejected rather than quietly permitted. Nobody needs to
// spell it that way.
function matches(glob, target) {
  let pattern = "";
  for (let i = 0; i < glob.length; i += 1) {
    const char = glob[i];
    if (char === "*") {
      if (glob[i + 1] === "*") {
        if (glob[i + 2] === "/") {
          pattern += "(?:.*/)?";
          i += 2;
        } else {
          pattern += ".*";
          i += 1;
        }
      } else {
        pattern += "[^/]*";
      }
    } else {
      pattern += "\\^$+?.()|[]{}".includes(char) ? `\\${char}` : char;
    }
  }
  return new RegExp(`^${pattern}$`).test(target);
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

test("Windows runner prep resolves rustup outside the job's CARGO_HOME", async () => {
  const action = await source(".github/actions/prepare-rust-runner/action.yml");
  // rustup lives in the profile that installed it. The runner services pin CARGO_HOME
  // to a per-runner dependency cache (D:\cargo-home-N) that has no bin\ at all, and
  // windows-candle.yml repoints it again at $RUNNER_TEMP\cargo-home — so deriving
  // rustup.exe from CARGO_HOME alone finds nothing and skips the authoritative
  // `rustup which cargo` resolution on every run (sc-13166).
  assert.match(action, /\$userCargoBin = Join-Path \(Join-Path \$env:USERPROFILE '\.cargo'\) 'bin'/);
  assert.match(action, /Get-Command rustup\.exe -CommandType Application/);
  assert.match(action, /\$cargoPath = \(& \$rustupExe which cargo \| Out-String\)\.Trim\(\)/);
  assert.doesNotMatch(action, /\$rustupExe = Join-Path \$cargoHome/);
});

test("Windows runner prep only warns about a genuinely dangling .cargo\\bin", async () => {
  const action = await source(".github/actions/prepare-rust-runner/action.yml");
  // A 0-byte reparse point is rustup's NORMAL Windows proxy shape and dispatches fine;
  // the only state worth reporting is rustup.exe itself being gone. The old check
  // warned on both the healthy shape and on a bin-less CARGO_HOME ("rustup was
  // deleted"), so it fired on 100% of runs and became invisible (sc-13166).
  // Scoped to the emitted ::warning lines — the header comment still narrates the old
  // wording as history, and that prose must not trip this guard.
  assert.doesNotMatch(action, /::warning[^\n]*rustup was deleted/);
  assert.doesNotMatch(action, /::warning[^\n]*byte reparse point/);
  assert.match(action, /\$profileRustup = Join-Path \$userCargoBin 'rustup\.exe'/);
  assert.match(action, /if \(-not \$rustupIntact\)/);
  // Still non-fatal: the lane is immune via the PATH prepend.
  assert.match(action, /::warning title=Dangling \.cargo\\bin proxies on candle runner::/);
});

test("Windows runner prep emits only ASCII inside its PowerShell strings", async () => {
  const action = await source(".github/actions/prepare-rust-runner/action.yml");
  // The step body is handed to powershell.exe as a file. If it is ever decoded as
  // CP1252 rather than UTF-8, an em dash (E2 80 94) becomes "â€" + U+201D, and
  // PowerShell honors U+201D as a STRING DELIMITER — the whole step dies on a parse
  // error before it can resolve a toolchain. Comments tolerate the mangling (the rest
  // of the line is ignored); quoted strings do not. So: non-ASCII stays in comments.
  const offenders = [];
  for (const [i, line] of action.split("\n").entries()) {
    if (/^\s*#/.test(line)) continue; // YAML/PowerShell comment lines are safe
    if (!/^[\x09\x20-\x7e]*$/.test(line)) offenders.push(`${i + 1}: ${line.trim()}`);
  }
  assert.deepEqual(offenders, []);
});

test("Windows CUDA isolates Cargo dependency checkouts after toolchain discovery", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const prepare = workflow.indexOf("uses: ./.github/actions/prepare-rust-runner");
  const disableWrapper = workflow.indexOf(
    "name: Disable unstable sccache wrapper for the heavy Candle lane",
  );
  const isolate = workflow.indexOf("name: Isolate Cargo dependency checkout");
  const fetch = workflow.indexOf("name: Fetch the private inference release");
  assert.ok(
    prepare >= 0 && prepare < disableWrapper && disableWrapper < isolate && isolate < fetch,
  );
  assert.match(
    workflow,
    /Add-Content -Path \$env:GITHUB_ENV -Value 'RUSTC_WRAPPER='/,
  );
  assert.match(
    workflow,
    /Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue/,
  );
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
  // The lane is now split by HARDWARE, not by cost: `macos-checks` (hosted macos-26)
  // carries the build, both clippy steps and the workspace suite, while `nax-worker`
  // (self-hosted, M5) carries only the matrix-unit guard. Pin BOTH halves against each
  // other — a skip on one side is only safe while the other side actually runs the
  // skipped test, and nothing else in the fleet would notice if it stopped.
  //
  // `npm run rust:check` runs a bare `cargo test` (the whole default-member set), so the
  // hosted half must stay workspace-wide. Narrowing it back to `-p sceneworks-worker`
  // would re-dark macOS-conditional code in sceneworks-core / rust-api / image-quality,
  // which is precisely what sc-17026 fixed.
  const NAX_TEST = "nax_16bit_sdpa_is_correct";
  assert.match(mlx, new RegExp(`^\\s+run: cargo test -- --skip ${NAX_TEST}$`, "m"));
  // The one exemption is the targeted guard invocation below; a bare narrowing of the
  // suite to the single crate is still forbidden.
  assert.doesNotMatch(mlx, /run: cargo test -p sceneworks-worker\s*$/m);
  // ...and the skipped test must demonstrably run somewhere. Without this, deleting the
  // self-hosted half would drop the entire NAX verdict while every lane stayed green —
  // the same declared-but-unreachable trap as sc-17026, one job over.
  assert.match(mlx, /^\s+run: cargo test -p sceneworks-worker --test nax_guard$/m);
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
  const testAt = mlx.indexOf(`run: cargo test -- --skip ${NAX_TEST}`);
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

test("both stage-1 lanes verify their own capability dump, LAST and reachably", async () => {
  // sc-17119 (mlx) + sc-17592 (candle). config/engine-capabilities/capabilities.<backend>.json is
  // read as a SOURCE by every other guard: bump-inference.mjs checks only its existence, declared
  // backend and `inferenceRevision`, and the vitest drift guard re-derives the catalog from its
  // contents. All of that is satisfied by a RESTAMP — rewriting the revision line over a stale
  // engine list — which is how both files were actually produced through two consecutive pin bumps.
  // Only a lane that LINKS the engine can tell the difference, and there is exactly one such PR lane
  // per backend.
  const lanes = [
    [".github/workflows/macos-mlx.yml", "capabilities.mlx.json"],
    [".github/workflows/windows-candle.yml", "capabilities.candle.json"],
  ];
  for (const [path, file] of lanes) {
    const lane = await source(path);
    const verifyAt = lane.indexOf(`- name: Verify ${file} is a real dump, not a restamp`);
    assert.ok(verifyAt > 0, `${path} must verify ${file} against a fresh dump`);
    // Re-dump to a SCRATCH dir and compare. Dumping over the checked-in file would make the
    // comparison vacuous and mutate the tree on a red run.
    assert.match(lane, /bin dump-engine-capabilities/, path);

    // LAST on the PR path. A step failure aborts the job, and this one goes red on exactly the
    // routine pin-bump PRs where nobody re-dumped — so placed earlier it would cancel the coverage
    // each lane uniquely carries (macOS: `nax_guard`; Windows: the only PR run of
    // `cargo test -p sceneworks-worker --features backend-candle`). A missing dump must not suppress
    // unrelated verdicts.
    //
    // "Last" means last among steps that RUN on a pull request, not last in the file: macos-mlx.yml
    // keeps a long `workflow_dispatch`-only calibration tail after it, which is skipped on every PR
    // and so cannot be cancelled by this step. Asserting the ordering rather than mere presence is
    // the point — nothing else would notice an unconditional step being appended later.
    for (const block of lane.slice(verifyAt).split(/\n {6}- (?=name: |uses: )/).slice(1)) {
      assert.match(
        block,
        /if: \$\{\{[^\n]*github\.event_name == 'workflow_dispatch'/,
        `${path}: "${block.split("\n")[0]}" runs after the dump-verification step on the PR path. ` +
          "That step must stay last for everything a PR executes, so its failure cannot cancel " +
          "coverage this lane is the only place to have. Move it above the verification step.",
      );
    }

    // Reachability. A restamp touches ONLY the facts file, so without this path entry the lane does
    // not run at all on the one PR the step exists to catch — declared but unreachable, the same
    // trap sc-17026 was about.
    //
    // Pinned to THIS lane's own file, not `config/engine-capabilities/**` (sc-17665). The directory
    // glob satisfied reachability too, but it also woke each lane for the OTHER backend's dump — a
    // whole self-hosted job, on the fleet's most constrained resource, for a file the woken lane
    // never opens.
    //
    // Asserted by MATCHING the declared globs against both filenames, not by forbidding particular
    // spellings. A spelling blocklist only catches a straight revert: keeping the narrow entry and
    // *adding* `config/engine-capabilities/*.json`, `config/engine-capabilities/**/*` or `config/**`
    // restores the cross-wake in full, and every one of those passes a `**`/`*` blocklist. Matching
    // is spelling-independent and closes all of them at once.
    const anchorAt = lane.indexOf("paths: &");
    assert.ok(anchorAt > 0, `${path} must declare a paths anchor`);
    const declared = [
      ...lane
        .slice(anchorAt, lane.indexOf("pull_request:"))
        .matchAll(/^ {6}- "([^"]+)"$/gm),
    ].map((match) => match[1]);
    // Every file this lane's verify step actually DIFFS must be watched. Read out of the step body
    // rather than listed here, because the failure this closes was a step growing a new file while
    // the filter stayed as it was — a hardcoded list would have been updated by the same edit that
    // grew the step, or not at all, so it could not have caught it.
    //
    // sc-17593 added an audio diff to BOTH lanes' steps; sc-17665 had just narrowed both filters
    // from `config/engine-capabilities/**` to a single filename. Each is right on its own. Together
    // they left the audio verification declared but unreachable, which `97a7655a9` — an audio-only
    // re-dump — demonstrated by waking neither stage-1 lane.
    const diffed = new Set(
      [
        ...lane
          .slice(verifyAt)
          .split("\n      - name:")[0]
          .matchAll(/config[/\\]engine-capabilities[/\\][\w./\\-]*\.json/g),
      ].map((match) => match[0].replaceAll("\\", "/")),
    );
    assert.ok(
      diffed.size >= 2,
      `${path}: expected the verify step to diff at least ${file} and the audio dump, found ` +
        `${JSON.stringify([...diffed])}`,
    );
    for (const target of diffed) {
      assert.ok(
        declared.some((glob) => matches(glob, target)),
        `${path}'s verify step diffs ${target}, but no declared path matches it — so an edit to ` +
          "that file alone never triggers the lane, and the check is declared but unreachable. " +
          "Add it to the paths anchor.",
      );
    }

    const own = `config/engine-capabilities/${file}`;
    assert.ok(
      declared.some((glob) => matches(glob, own)),
      `${path} must watch ${own}, or a restamp of it — which touches nothing else — never ` +
        "triggers the lane and the verification step is declared but unreachable.",
    );
    for (const [, otherFile] of lanes) {
      if (otherFile === file) continue;
      const foreign = `config/engine-capabilities/${otherFile}`;
      const culprits = declared.filter((glob) => matches(glob, foreign));
      assert.deepEqual(
        culprits,
        [],
        `${path} triggers on ${foreign} via ${JSON.stringify(culprits)}, but has no step that ` +
          "reads it. That wakes an entire self-hosted job — the fleet's most constrained " +
          "resource — for a file this lane cannot check. Narrow the pattern to this lane's own dump.",
      );
    }
  }
});

test("every workspace path a self-hosted lane watches maps to a package that lane builds", async () => {
  // sc-17703, generalising sc-17665's lesson to the whole trigger surface. `apps/rust-worker/**`
  // sat in windows-candle.yml's paths while no cargo invocation in that job built the package
  // living there (`sceneworks-rust-worker` — a 4-line binary wrapper nothing depends on), so a
  // wrapper edit woke a ~24m run on the `cuda` pool for zero coverage. Pin the PROPERTY, not the
  // spelling (epic 17702 rule 6): every `crates/`- or `apps/`-shaped `paths:` entry on a
  // self-hosted lane may only wake the lane for workspace members that lane's own cargo
  // invocations build, directly or through the local dependency graph; every other entry must be
  // declared in the reasons-required allow-list below.
  //
  // Derived from the artifacts, not restated: the built set is parsed from each workflow's
  // `cargo ... -p <pkg>` lines (a package-less `cargo test`/`clippy`/`check`/`build` acts on the
  // whole default-member set, which is what macos-mlx.yml's hosted macos-checks job runs), and
  // the closure comes from the `path = "..."` edges in the members' Cargo.toml files. So
  // re-adding the rust-worker entry to the candle lane, watching a member no invocation reaches,
  // or narrowing a lane's build out from under an entry it still watches all go red with no test
  // edit.
  const rootManifest = await source("Cargo.toml");
  const listOf = (key) => {
    // Anchored to line start: a bare indexOf("members = [") would land inside
    // "default-members = [" if the root manifest's blocks were ever reordered.
    const start = rootManifest.search(new RegExp(`^${key} = \\[`, "m"));
    assert.ok(start >= 0, `root Cargo.toml must declare ${key}`);
    const block = rootManifest.slice(start, rootManifest.indexOf("]", start));
    return [...block.matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  };
  const memberDirs = listOf("members");
  const defaultMemberDirs = listOf("default-members");

  const normalize = (base, rel) => {
    const parts = base.split("/");
    for (const seg of rel.split("/")) {
      if (seg === "..") parts.pop();
      else if (seg !== "." && seg !== "") parts.push(seg);
    }
    return parts.join("/");
  };
  const dirOf = new Map();
  const manifests = new Map();
  for (const dir of memberDirs) {
    const manifest = await source(`${dir}/Cargo.toml`);
    const name = manifest.match(/^name = "([^"]+)"$/m);
    assert.ok(name, `${dir}/Cargo.toml must name its package`);
    dirOf.set(name[1], dir);
    manifests.set(dir, manifest);
  }
  const dependsOn = new Map();
  for (const [dir, manifest] of manifests) {
    // `path = "..."` also appears under [[bin]] targets (e.g. sceneworks-memory-adapter's
    // src/bin/mlx.rs); resolving against the member set filters those out — only an edge that
    // lands on another workspace member is a dependency.
    const edges = [...manifest.matchAll(/path\s*=\s*"([^"]+)"/g)]
      .map((edge) => normalize(dir, edge[1]))
      .filter((target) => manifests.has(target));
    dependsOn.set(dir, edges);
  }
  const closure = (seedDirs) => {
    const reached = new Set();
    const queue = [...seedDirs];
    while (queue.length > 0) {
      const dir = queue.pop();
      if (reached.has(dir)) continue;
      reached.add(dir);
      queue.push(...dependsOn.get(dir));
    }
    return reached;
  };

  const lanes = [
    {
      path: ".github/workflows/windows-candle.yml",
      // Every non-cargo entry needs its reason recorded HERE, or the test fails. Watching a path
      // no step reads costs a whole self-hosted job (epic 17702 rule 1), and "symmetry with the
      // sibling lane" is never the reason (rule 2; sc-17592 is the scar).
      allowed: [
        "config/manifests/**", // include_str!'d into the worker; the manifest drift guard reads it
        "config/engine-capabilities/capabilities.candle.json", // the restamp-verify step diffs it
        // The audio dump the SAME step also diffs (sc-17593). On BOTH lanes, unlike the media
        // files: AUDIO_BACKEND is candle everywhere, so either box produces this one file and
        // both verify steps open it. That is the test sc-17703 applies — a step here reads it —
        // and not symmetry for its own sake.
        "config/engine-capabilities/audio/capabilities.candle.json",
        "Cargo.toml", // workspace graph + lints: changes what every invocation here resolves
        "Cargo.lock", // dependency pins, incl. the inference revision the whole lane compiles
        "rust-toolchain.toml", // no toolchain action on this lane; cargo auto-selects this pin
        ".cargo/config.toml", // git-fetch-with-cli, without which the token-injected inference fetch breaks
        ".github/actions/prepare-rust-runner/**", // the job's own first step
        ".github/workflows/windows-candle.yml", // the lane itself
      ],
    },
    {
      path: ".github/workflows/macos-mlx.yml",
      allowed: [
        "config/manifests/**", // include_str!'d into the worker; the manifest drift guard reads it
        "config/engine-capabilities/capabilities.mlx.json", // the restamp-verify step diffs it
        // The audio dump the SAME step also diffs (sc-17593). On BOTH lanes, unlike the media
        // files: AUDIO_BACKEND is candle everywhere, so either box produces this one file and
        // both verify steps open it. That is the test sc-17703 applies — a step here reads it —
        // and not symmetry for its own sake.
        "config/engine-capabilities/audio/capabilities.candle.json",
        "Cargo.toml", // workspace graph + lints: changes what every invocation here resolves
        "Cargo.lock", // dependency pins, incl. the MLX revision the whole lane compiles
        "rust-toolchain.toml", // governs the toolchain cargo resolves under the dtolnay install
        ".cargo/config.toml", // the MACOSX_DEPLOYMENT_TARGET=26.2 pin every build here compiles under
        ".github/workflows/macos-mlx.yml", // the lane itself
      ],
    },
  ];
  for (const { path, allowed } of lanes) {
    const lane = await source(path);
    // Only steps a pull_request actually executes may justify a PR path entry. A
    // dispatch-only calibration build that reaches a member is not PR coverage of it, so
    // drop every step block gated on workflow_dispatch before scanning (same step-splitting
    // idiom as the capability-dump ordering check above).
    const prSteps = lane
      .split(/\n {6}- (?=name: |uses: )/)
      .filter(
        (block) => !/if: \$\{\{[^\n]*github\.event_name == 'workflow_dispatch'/.test(block),
      )
      .join("\n");
    // Strip comment lines, then re-join backslash line continuations so a `-p <pkg>` split
    // across lines cannot degrade into a "package-less" invocation (which the rule below
    // would over-widen into the whole default-member set).
    const executable = prSteps
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n")
      .replace(/\\\n\s*/g, " ");
    // LINE-INITIAL commands only. An `echo`ed fix-it message that merely QUOTES a
    // package-less `cargo test` must not count as a workspace-wide build — counting it
    // would quietly widen `covered` to every default member and defuse this whole guard.
    const invocations = [
      ...executable.matchAll(/^\s*(?:run: )?cargo +(?:test|check|clippy|build|run)\b[^\n]*/gm),
    ];
    assert.ok(invocations.length > 0, `${path} must contain cargo invocations to audit against`);
    const built = new Set();
    for (const [invocation] of invocations) {
      const packages = [...invocation.matchAll(/(?:-p|--package)[ =]([A-Za-z0-9_-]+)/g)];
      if (packages.length === 0) {
        // A package-less build/test/lint acts on the whole default-member set — that is
        // exactly macos-mlx.yml's hosted workspace-wide `cargo test` / `cargo clippy`.
        for (const dir of defaultMemberDirs) built.add(dir);
      } else {
        for (const [, pkg] of packages) {
          const dir = dirOf.get(pkg);
          assert.ok(dir, `${path} invokes cargo on ${pkg}, which is not a workspace member`);
          built.add(dir);
        }
      }
    }
    const covered = closure(built);

    const anchorAt = lane.indexOf("paths: &");
    assert.ok(anchorAt > 0, `${path} must declare a paths anchor`);
    const anchorBlock = lane.slice(anchorAt, lane.indexOf("pull_request:"));
    const declared = [...anchorBlock.matchAll(/^ {6}- "([^"]+)"$/gm)].map((entry) => entry[1]);
    assert.ok(declared.length > 0, `${path} must declare path filters`);
    // Completeness: an unquoted entry (`- apps/x/**`) or a trailing inline comment would
    // fail the parse above and slip past this audit entirely. Every list item must be a
    // bare quoted string, or the "EVERY paths entry" contract is fiction.
    const entryLines = anchorBlock.split("\n").filter((line) => /^ {6}- /.test(line));
    assert.equal(
      declared.length,
      entryLines.length,
      `${path}: every paths entry must be a bare double-quoted string so this audit can ` +
        "parse it; rewrite the unmatched entries.",
    );

    for (const glob of declared) {
      if (!/^(?:crates|apps)\//.test(glob)) {
        assert.ok(
          allowed.includes(glob),
          `${path} watches ${JSON.stringify(glob)}, which is neither a workspace path this job ` +
            "builds nor a declared non-cargo entry. If a step really reads it, add it to this " +
            "test's allow-list WITH the reason; symmetry with the sibling lane is not one (sc-17592).",
        );
        continue;
      }
      // Judge the entry by every member it can wake the lane for: a glob wakes a member if
      // it can match the member's own manifest (`<dir>/Cargo.toml` — every member has one)
      // or if it names a path INSIDE the member (a legitimate narrowing like
      // "crates/x/tests/**", which cannot match the manifest but still wakes only that member).
      const woken = memberDirs.filter(
        (dir) => matches(glob, `${dir}/Cargo.toml`) || glob.startsWith(`${dir}/`),
      );
      assert.ok(
        woken.length > 0,
        `${path} watches ${JSON.stringify(glob)}, which matches no workspace member — dead ` +
          "weight or a typo.",
      );
      for (const dir of woken) {
        assert.ok(
          covered.has(dir),
          `${path} triggers on ${dir} via ${JSON.stringify(glob)}, but no cargo invocation in ` +
            "that workflow builds it, directly or through a path dependency. That wakes a whole " +
            "self-hosted job for an edit it cannot cover — exactly how apps/rust-worker/** sat " +
            "on the candle lane (sc-17703). Narrow the entry, or make the job build the member.",
        );
      }
    }
  }
});

test("the Rust toolchain is pinned to one concrete version, everywhere", async () => {
  // sc-17717. `channel = "stable"` let the effective toolchain move with NO file changing:
  // hosted lanes' dtolnay steps installed whatever stable was current at run time while the
  // self-hosted boxes used whatever was on disk — so lanes could compile the same commit
  // with DIFFERENT rustcs, and a new stable's clippy lints could red `-D warnings` lanes on
  // unrelated PRs. Three properties pinned here:
  //
  //   1. rust-toolchain.toml's channel is a concrete x.y.z, never a floating channel.
  //   2. Every workflow's dtolnay `toolchain:` input equals that exact version — a
  //      straggler left at `stable` double-installs on every hosted job and re-floats the
  //      persistent Macs. Deriving the expectation from rust-toolchain.toml means a bump
  //      only has to edit files, never this test.
  //   3. Every dtolnay step HAS a `toolchain:` input. Omitting it makes the action fall
  //      back to its tag (the pinned SHA tracks dtolnay's `stable` branch), which
  //      reintroduces the float without the word "stable" appearing anywhere.
  //
  // The rustfmt/clippy components ride along: the self-hosted boxes pre-install the pin
  // with both (see the bump recipe in rust-toolchain.toml), and clippy lanes assume them.
  const toolchainFile = await source("rust-toolchain.toml");
  const channelMatch = toolchainFile.match(/^channel = "([^"]+)"$/m);
  assert.ok(channelMatch, "rust-toolchain.toml must declare a channel");
  const pin = channelMatch[1];
  assert.match(
    pin,
    /^\d+\.\d+\.\d+$/,
    "rust-toolchain.toml must pin a concrete x.y.z version. 'stable' floats: the effective " +
      "toolchain then changes with no file changing, which no CI trigger can catch (sc-17717).",
  );
  assert.match(
    toolchainFile,
    /^components = \["rustfmt", "clippy"\]$/m,
    "the pin must carry rustfmt and clippy — the fmt gate and every -D warnings lane assume them",
  );

  const workflowFiles = (await readdir(new URL("../.github/workflows/", import.meta.url)))
    .filter((file) => file.endsWith(".yml") || file.endsWith(".yaml"))
    .sort();
  assert.ok(workflowFiles.length > 0, "no workflows found — the audit would be vacuous");
  let inputsSeen = 0;
  for (const file of workflowFiles) {
    const workflow = await source(`.github/workflows/${file}`);
    const dtolnaySteps = [...workflow.matchAll(/uses: dtolnay\/rust-toolchain@/g)].length;
    const inputs = [...workflow.matchAll(/^\s+toolchain: (.+)$/gm)].map((entry) => entry[1]);
    assert.equal(
      inputs.length,
      dtolnaySteps,
      `${file}: every dtolnay/rust-toolchain step needs an explicit toolchain: input — ` +
        "omitting it falls back to the action tag's floating stable.",
    );
    for (const value of inputs) {
      inputsSeen += 1;
      assert.equal(
        value,
        `"${pin}"`,
        `${file}: toolchain input ${value} must be "${pin}" (quoted), the version ` +
          "rust-toolchain.toml pins — one version everywhere, bumped together.",
      );
    }
  }
  assert.ok(inputsSeen > 0, "no toolchain inputs found anywhere — the lockstep audit is vacuous");
});

// ---------------------------------------------------------------------------------------------
// REQUIRED CHECKS AND THE MERGE QUEUE (sc-17014)
//
// A check can only be REQUIRED if it reports on every gated event. GitHub distinguishes the two
// ways a check can go missing, and only one of them is safe:
//   * job skipped by `if:`            -> Success, satisfies a required check;
//   * workflow skipped by `paths:`    -> Pending forever, blocks the PR.
// The merge queue adds a second, louder failure mode: it evaluates the same required set against
// `gh-readonly-queue/main/**`, so a lane with no `merge_group:` trigger strands the group until
// `check_response_timeout_minutes` evicts its entries — silently re-ordering the queue.
//
// So every required lane must (a) trigger on merge_group, (b) NOT path-filter its pull_request
// trigger, and (c) gate its expensive jobs on the shared `changes` relevance job instead.
// ---------------------------------------------------------------------------------------------

/** Lanes whose jobs are (or are intended to be) required status checks. */
const REQUIRED_LANES = [
  { path: ".github/workflows/macos-mlx.yml", anchor: "mlx_paths" },
  { path: ".github/workflows/desktop-windows.yml", anchor: "desktop_paths" },
  { path: ".github/workflows/desktop-linux-check.yml", anchor: "desktop_linux_check_paths" },
  { path: ".github/workflows/desktop-macos-check.yml", anchor: "desktop_macos_check_paths" },
];

test("every required lane reports on the merge queue and drops its PR path filter", async () => {
  for (const { path } of REQUIRED_LANES) {
    const lane = await source(path);
    assert.match(
      lane,
      /^ {2}merge_group:$/m,
      `${path}: a required check with no merge_group: trigger strands every queued entry until ` +
        "check_response_timeout_minutes evicts it.",
    );
    // The pull_request trigger must carry no `paths:` — anchor definition or alias. A filtered
    // workflow's check sits Pending forever, which is the deadlock this whole restructure exists
    // to avoid. Scoped to the pull_request block so `push:` keeping its filter stays legal.
    const prAt = lane.indexOf("\n  pull_request:");
    assert.ok(prAt > 0, `${path} must declare a pull_request trigger`);
    const prBlock = lane.slice(prAt + 1).split(/\n {2}(?=[a-z_]+:)/)[0];
    assert.doesNotMatch(
      prBlock,
      /^\s+paths:/m,
      `${path}: the pull_request trigger must NOT be path-filtered — its check would stay ` +
        "Pending instead of passing. Filter in the `changes` job instead.",
    );
  }
});

test("every required lane delegates its path filter to the shared gate, with a real anchor", async () => {
  const { parseWorkflowPaths } = await import("./merge-group-relevance.mjs");
  for (const { path, anchor } of REQUIRED_LANES) {
    const lane = await source(path);
    assert.match(
      lane,
      /uses: \.\/\.github\/workflows\/changed-paths\.yml/,
      `${path} must call the shared changed-paths gate rather than re-implementing it`,
    );
    // The gate reads the lane's own anchor at runtime, so a typo in either input silently turns
    // the filter into a permanent "run everything" — safe, but dead. Pin both against the file.
    const declared = /lane: (\S+)\s+anchor: (\S+)/.exec(lane);
    assert.ok(declared, `${path} must pass both lane: and anchor: to the gate`);
    assert.equal(declared[1], path, `${path}: the gate must be pointed at its own lane file`);
    assert.equal(declared[2], anchor, `${path}: gate anchor input must be ${anchor}`);
    assert.ok(
      parseWorkflowPaths(lane, declared[2]).length > 0,
      `${path} has no \`paths: &${declared[2]}\` anchor for the gate to read`,
    );
    // A gate nothing consumes is decoration.
    assert.match(
      lane,
      /needs\.changes\.outputs\.relevant == 'true'/,
      `${path}: at least one job must be conditioned on the gate's verdict`,
    );
  }
});

test("dropping the PR path filter did not expose the self-hosted pools", async () => {
  // This is the hazard the restructure creates. The pull_request `paths:` filter used to be what
  // kept docs-only PRs off the two-Mac `nax` pool and the `cuda` pool; with it gone, ONLY the
  // `changes` gate does. A self-hosted job that forgot `needs: changes` would run on every PR.
  const mlx = await source(".github/workflows/macos-mlx.yml");
  assert.match(
    mlx,
    /if: \$\{\{ github\.event_name != 'merge_group' && needs\.changes\.outputs\.relevant == 'true'/,
    "nax-worker must be gated on `changes` AND excluded from merge groups — without the gate, " +
      "every docs-only PR now wakes the two-Mac nax pool.",
  );

  const desktop = await source(".github/workflows/desktop-windows.yml");
  // package-windows is main + dispatch only. Before merge_group existed, `!= 'pull_request'`
  // expressed that; it no longer does, and this job is the heavy candle/CUDA package on the
  // `cuda` pool. It must exclude merge_group explicitly.
  assert.match(
    desktop,
    /if: \$\{\{ github\.event_name != 'pull_request' && github\.event_name != 'merge_group' \}\}/,
    "package-windows must exclude merge_group explicitly, or every queued entry wakes the " +
      "candle/CUDA packaging job on the self-hosted cuda pool.",
  );
  // …and its cheap sibling must NOT skip on merge groups, or the required check passes vacuously.
  assert.match(
    desktop,
    /github\.event_name == 'pull_request' \|\| github\.event_name == 'merge_group'/,
    "build-windows is the required check; it must actually run on the speculative merge rather " +
      "than skip into a free Success.",
  );
});

test("windows-candle stays out of the queue and out of the required set", async () => {
  // ~24m median, p90 32m (measured 2026-08-05 over 85 runs) against a 60m check-response timeout,
  // on the self-hosted `cuda` pool. Its merge-time stand-in is check.yml's hosted `candle`
  // typecheck. Making candle-worker required would force a merge_group: trigger here, and p90
  // queue wait (18m) + p90 run already reaches ~50m of the 60m budget.
  assert.doesNotMatch(
    await source(".github/workflows/windows-candle.yml"),
    /^ {2}merge_group:$/m,
    "windows-candle.yml must stay out of the merge queue; check.yml's `candle` job is its " +
      "merge-time stand-in. See sc-17014 for the (A)/(B)/(C) decision if this changes.",
  );
});

test("the always-on lanes stay unfiltered so they can be required", async () => {
  const check = await source(".github/workflows/check.yml");
  assert.match(check, /^ {2}merge_group:$/m, "check.yml carries web + parity + candle");
  // check.yml has never had path filters, and must not grow one: all three of its jobs are
  // required, and a filter would strand them Pending.
  assert.doesNotMatch(
    check.slice(check.indexOf("on:"), check.indexOf("jobs:")),
    /paths:/,
    "check.yml must stay unfiltered — web, parity and candle are all required checks",
  );
});
