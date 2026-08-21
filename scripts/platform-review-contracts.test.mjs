import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

import { SOURCE_PATHS } from "./generate-memory-matrix.mjs";
import {
  DECISIONS_PATH as CANDLE_ADMISSION_DECISIONS_PATH,
  SOURCE_PATHS as CANDLE_ADMISSION_SOURCE_PATHS,
} from "./generate-candle-admission-inventory.mjs";
import { buildPlans as buildLtxPlans } from "./generate-ltx-sc18946-plan.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

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
  const fetch = workflow.indexOf("name: Fetch the pinned inference release");
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

// sc-18677 generalized windows-candle.yml's provisioning from "the Krea q4 snapshot" to "the
// snapshot named by the provision_* inputs", so epic 17137 can land MiniMax-H3 weights on the
// CUDA box for sc-17153/sc-17156's per-tier VRAM measurement. The story's acceptance criterion
// is that a Krea dispatch "still behaves identically -- proven, not assumed". These tests are
// that proof, and they are structural rather than prose-matching: they pin the input DEFAULTS
// and the path-construction EXPRESSIONS, which together determine the resolved snapshot path.
//
// The path the old hardcoded step produced, verbatim:
//   %USERPROFILE%\.cache\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\q4
// Every fragment below is a factor of that string. Change any one and a Krea five-rung capture
// silently reads another directory -- or, if it is lucky, throws.
// Assertions about a step must be scoped TO that step. A bare `assert.match(workflow, ...)` is
// satisfied by an identical string anywhere in the file, which is not hypothetical: dropping
// `inputs.provision_snapshot` from the Provision step's `if:` left the suite green because the
// neighbouring "Validate runner Python" step carries the byte-identical condition. Slice first.
function stepBody(workflow, name) {
  const at = workflow.indexOf(`      - name: ${name}\n`);
  assert.ok(at >= 0, `windows-candle.yml must keep a step named ${name}`);
  const next = workflow.indexOf("\n      - ", at + 1);
  return workflow.slice(at, next === -1 ? undefined : next);
}

function dispatchInputs(workflow) {
  const start = workflow.indexOf("  workflow_dispatch:\n    inputs:\n");
  assert.ok(start >= 0, "windows-candle.yml must keep a workflow_dispatch inputs block");
  const end = workflow.indexOf("\nconcurrency:", start);
  assert.ok(end > start, "could not find the end of the workflow_dispatch block");
  const names = [];
  const defaults = {};
  let current = null;
  for (const line of workflow.slice(start, end).split("\n")) {
    // Deliberately permissive: GitHub allows digits, case and hyphens in an input name, and this
    // helper backs the "at most 25 inputs" cap check. A narrower pattern would silently skip an
    // input and let a workflow GitHub rejects sail through as 25-or-fewer.
    const header = line.match(/^ {6}([A-Za-z0-9_-]+):$/);
    if (header) {
      current = header[1];
      names.push(current);
      defaults[current] = undefined;
      continue;
    }
    const def = current && line.match(/^ {8}default: (.*)$/);
    if (def) defaults[current] = def[1].trim().replace(/^"|"$/g, "");
  }
  return { names, defaults };
}

function assertDispatchInputLimit(workflow) {
  const { names } = dispatchInputs(workflow);
  assert.ok(names.length <= 25, `workflow_dispatch allows at most 25 inputs, found ${names.length}`);
}

test("windows-candle enforces GitHub's current 25-input workflow_dispatch ceiling", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  assertDispatchInputLimit(workflow);
  const extra = Array.from(
    { length: 25 },
    (_, index) => `      mutation_input_${index}:\n        type: boolean\n        default: false\n`,
  ).join("");
  const overLimit = workflow.replace("      run_sc19057_wan_capture:\n", `${extra}      run_sc19057_wan_capture:\n`);
  assert.throws(() => assertDispatchInputLimit(overLimit), /at most 25 inputs, found 35/);
});

test("windows-candle provisioning is model-parameterized, not Krea-hardcoded", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const { names, defaults } = dispatchInputs(workflow);

  // GitHub's current ceiling is 25. Keep the generalized provision family rather than cloning it
  // per model; the dedicated limit test above proves this workflow remains dispatchable.
  assertDispatchInputLimit(workflow);

  for (const gone of ["provision_krea_snapshot", "krea_repository", "krea_revision"]) {
    assert.ok(!names.includes(gone), `${gone} was renamed; two provisioning paths must not coexist`);
  }
  for (const required of [
    "provision_snapshot",
    "provision_repository",
    "provision_revision",
    "provision_patterns",
    "provision_subdir",
    "provision_cache_dir",
  ]) {
    assert.ok(names.includes(required), `missing generalized input ${required}`);
  }

  // The defaults ARE the Krea dispatch. With these values and no other input set, the
  // generalized steps must reconstruct the old hardcoded path exactly.
  assert.equal(defaults.provision_repository, "SceneWorks/krea-2-turbo-mlx");
  assert.equal(defaults.provision_patterns, "q4/**");
  assert.equal(defaults.provision_subdir, "q4");
  assert.equal(defaults.provision_cache_dir, undefined, "an empty cache dir must mean the historical location");
});

test("windows-candle rebuilds the exact Krea snapshot path from the generalized inputs", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");

  // cache dir: %USERPROFILE%\.cache\huggingface\hub when provision_cache_dir is empty. This box
  // sets HF_HOME=E:\huggingface, and honoring it here would relocate Krea's cache -- so the
  // default deliberately ignores HF_HOME. A caller that wants another cache passes it in.
  assert.match(workflow, /\$cache = Join-Path \$env:USERPROFILE '\.cache\\huggingface\\hub'/);
  assert.doesNotMatch(
    workflow,
    /^\s*\$cache = .*HF_HOME/m,
    "the default cache dir must not be derived from HF_HOME",
  );

  // models--<owner>--<name>\snapshots\<revision>[\<subdir>]
  assert.match(workflow, /\$folder = 'models--' \+ \$env:PROVISION_REPOSITORY\.Replace\('\/', '--'\)/);
  assert.match(workflow, /\$subdirTail = '\\' \+ \$env:PROVISION_SUBDIR\.Replace\('\/', '\\'\)/);
  assert.match(
    workflow,
    /\$suffix = '\\' \+ \$folder \+ '\\snapshots\\' \+ \$env:PROVISION_REVISION \+ \$subdirTail/,
  );

  // The resolve step still asserts existence AND that the canonical path ends with that exact
  // suffix, so a stale cache entry or a lookalike repo cannot satisfy it.
  assert.match(workflow, /Test-Path -LiteralPath \$root -PathType Container/);
  assert.match(
    workflow,
    /\$root\.EndsWith\(\$env:PROVISION_SNAPSHOT_SUFFIX, \[StringComparison\]::OrdinalIgnoreCase\)/,
  );

  // The PYTHON half of the cache binding, not just the PowerShell half. Centralizing the cache
  // path exists because it was previously spelled twice in two languages with nothing tying
  // them; pinning only the PowerShell side leaves a re-hardcoded `os.path.join(USERPROFILE...)`
  // green, which downloads the whole component set to C: before the resolve step throws.
  assert.match(workflow, /cache_dir=os\.environ\["PROVISION_CACHE_DIR"\],/);
  assert.doesNotMatch(
    workflow,
    /cache_dir=os\.path\.join\(os\.environ\["USERPROFILE"\]/,
    "the provisioning cache dir must come from the shared resolved value, not a second hardcoding",
  );

  // The memory-adapter binaries read these env names via required_env; renaming the dispatch
  // inputs must not rename the runtime contract (bin/candle.rs, bin/mlx.rs).
  assert.match(workflow, /"SCENEWORKS_KREA_ROOT=\$root" \| Out-File/);
  assert.match(workflow, /SCENEWORKS_KREA_REPOSITORY: \$\{\{ inputs\.provision_repository \}\}/);
  assert.match(workflow, /SCENEWORKS_KREA_REVISION: \$\{\{ inputs\.provision_revision \}\}/);
  // ...and SCENEWORKS_KREA_ROOT stays scoped to Krea, so an H3 dispatch cannot hand the
  // five-rung adapter a MiniMax root under a Krea-shaped name.
  assert.match(workflow, /\$isKrea = \$env:PROVISION_REPOSITORY -eq 'SceneWorks\/krea-2-turbo-mlx'/);
  assert.match(workflow, /if \(\$isKrea\) \{\n\s*"SCENEWORKS_KREA_ROOT=\$root"/);
  // The CONSUMPTION side too, not just the export side. `secrets.SCENEWORKS_KREA_ROOT` is a
  // Krea-specific override; dropping the `$isKrea -and` would let it redirect an H3 resolve.
  assert.match(workflow, /if \(\$isKrea -and \$env:KREA_ROOT_OVERRIDE\) \{/);
});

// sc-18677: the provisioning branch's timeout is the whole point of the change (a 144 GB fetch
// under the ordinary 45m cap dies mid-download), and the doc claims Krea's two dispatch shapes
// keep their exact previous budgets. Pin the whole expression the way the macOS twin above pins
// its lane's -- otherwise a revert to a flat `timeout-minutes: 45` passes every other test here.
test("windows-candle keeps the provisioning and five-rung timeout budgets", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  assert.match(
    workflow,
    // The LTX Eros acceptance arm (SC-18902, from main) rides ahead of the provisioning budget;
    // the provisioning / five-rung / default budgets keep their exact prior values behind it.
    /timeout-minutes: \$\{\{ github\.event_name == 'workflow_dispatch' && \(inputs\.run_ltx_eros_acceptance \|\| inputs\.run_sc19057_wan_capture\) && 360 \|\| github\.event_name == 'workflow_dispatch' && inputs\.provision_snapshot && 240 \|\| github\.event_name == 'workflow_dispatch' && inputs\.run_five_rung_reference && 120 \|\| 45 \}\}/,
  );
});

test("windows-candle keeps the five-rung guards while decoupling provisioning", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");

  // Provisioning is now a first-class outcome: sc-17153/sc-17156 need H3 weights resident and
  // there is no H3 five-rung fixture. The old coupling throw must be gone...
  assert.doesNotMatch(workflow, /requires run_five_rung_reference=true/);
  // Scoped to the Provision step itself: the identical `if:` string also appears on the
  // "Validate runner Python" step, so a file-wide match would stay green if this step's gate were
  // dropped -- and an ungated Provision step re-downloads the snapshot on every five-rung run.
  assert.match(
    stepBody(workflow, "Provision exact snapshot"),
    /if: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.provision_snapshot \}\}/,
  );
  // ...but every five-rung guard it used to sit beside must survive, keyed on the new names.
  assert.match(workflow, /throw 'inference_revision must be an exact lowercase 40-hex commit'/);
  assert.match(
    workflow,
    /\$env:PROVISION_REVISION -notmatch '\^\[0-9a-f\]\{40\}\$'/,
  );
  assert.match(
    workflow,
    /\$env:PROVISION_REPOSITORY -ne 'SceneWorks\/krea-2-turbo-mlx'/,
    "a five-rung capture is still only valid against the fixed Krea reference artifact",
  );
  assert.match(workflow, /does not match the adapter's compiled INFERENCE_PIN/);

  // The resolve step must still run for a five-rung dispatch that does NOT provision, and the
  // params step that FEEDS it (PROVISION_CACHE_DIR / _SNAPSHOT_SUFFIX / _SUBDIR_TAIL) must run on
  // the same wider condition. Scoped per step for the reason at the top of this block: the two
  // `if:` strings are byte-identical, so one file-wide match is satisfied by EITHER step and three
  // mutations stayed green -- narrowing the resolve step to five-rung-only (a provision-only
  // dispatch then exports no snapshot root at all, skipping AC1's whole proof step), narrowing it
  // to provision-only (a five-rung-without-provisioning dispatch loses it), and narrowing the
  // params step (the resolve step then joins two empty env vars and throws, or worse, matches).
  const resolveGate =
    /if: \$\{\{ github\.event_name == 'workflow_dispatch' && \(inputs\.run_five_rung_reference \|\| inputs\.provision_snapshot\) \}\}/;
  assert.match(stepBody(workflow, "Resolve exact snapshot"), resolveGate);
  assert.match(stepBody(workflow, "Resolve snapshot provisioning parameters"), resolveGate);
});

test("windows-candle routes weights dispatches to a real-weights runner, like the MLX lane", async () => {
  const candle = await source(".github/workflows/windows-candle.yml");
  const mlx = await source(".github/workflows/macos-mlx.yml");

  // The MLX lane is the template: base labels for ordinary runs, plus a weights label for a
  // dispatch that needs real weights on disk. Assert the template still looks like that, so this
  // guard cannot outlive the convention it mirrors.
  assert.match(mlx, /runs-on: \$\{\{ \(github\.event_name == 'workflow_dispatch' && \(inputs\.run_memory_calibration \|\| inputs\.run_five_rung_reference\)\) && fromJSON\('\["self-hosted","macOS","ARM64","nax","weights"\]'\)/);

  // The `cuda` pool spans two registration levels and only the org-level half carries
  // `real-weights`; a provisioning job on the other half finds no snapshot.
  assert.match(
    candle,
    /runs-on: \$\{\{ \(github\.event_name == 'workflow_dispatch' && \(inputs\.provision_snapshot \|\| inputs\.run_five_rung_reference \|\| inputs\.run_sc19057_wan_capture\)\) && fromJSON\('\["self-hosted","Windows","X64","cuda","real-weights"\]'\) \|\| fromJSON\('\["self-hosted","Windows","X64","cuda"\]'\) \}\}/,
  );
  // Ordinary PR/push runs must NOT be narrowed to the real-weights half: that would cut the
  // available pool for this ~24m lane in half for no coverage.
  assert.doesNotMatch(
    candle,
    /^\s*runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/m,
  );
});

test("windows-candle provisioning can never degrade into a whole-repo fetch", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  // MiniMaxAI/MiniMax-H3 is ~498 GB because FL2VA/ and Ref2VA/ re-package the same components;
  // the set sc-18677 provisions is 144.051 GB. snapshot_download treats allow_patterns=[] as
  // "everything", so an empty list is a 354.424 GB accident on a box that shares its disk with
  // CI -- FL2VA/ and Ref2VA/ alone are 288.102 GB of it. Both the
  // validation step and the Python body must refuse it.
  assert.match(
    workflow,
    /throw 'provision_patterns must name at least one allow-pattern; an empty list would fetch the whole repository'/,
  );
  assert.match(workflow, /raise SystemExit\("provision_patterns resolved to an empty allow-list"\)/);
  assert.match(workflow, /allow_patterns=patterns,/);

  // A non-zero pip/python exit must fail the step: `@'...'@ | python -` does not propagate.
  assert.match(workflow, /if \(\$LASTEXITCODE -ne 0\) \{ throw "snapshot provisioning failed with exit code \$LASTEXITCODE" \}/);

  assert.match(
    workflow,
    /if \(\$LASTEXITCODE -ne 0\) \{ throw "installing huggingface_hub failed with exit code \$LASTEXITCODE" \}/,
  );

  // With provision_subdir empty the snapshot directory exists as soon as ANY file lands, so
  // existence alone is not proof. Every declared component's literal prefix must be present.
  //
  // Pin the LOOP BODY, not just the throw string: replacing the `if (-not $head) { continue }`
  // guard with an unconditional `continue` makes the assertion vacuous while leaving the error
  // message -- and every other assertion in this file -- untouched.
  const resolve = stepBody(workflow, "Resolve exact snapshot");
  // Pin the THROW, not the message. Every other assertion here survives `throw` ->
  // `Write-Warning`: the head computation, the guard, the Join-Path, the Test-Path and the string
  // all still match, while "fails loudly if absent" quietly becomes a log line.
  //
  // All THREE of the resolve step's assertions need that treatment, not just the component one.
  // The existence check and the canonical-suffix check are pinned elsewhere in this file as
  // EXPRESSIONS (`Test-Path -LiteralPath $root -PathType Container`, `$root.EndsWith(...)`), which
  // a `throw` -> `Write-Warning` downgrade leaves untouched. The suffix one is the dangerous case:
  // downgraded, a snapshot whose canonical path does not match the requested repo+revision is
  // accepted and exported as SCENEWORKS_PROVISIONED_ROOT / SCENEWORKS_KREA_ROOT, so a five-rung
  // capture or a per-tier VRAM measurement silently runs against the WRONG weights.
  assert.match(resolve, /throw "the exact snapshot is not available on this runner/);
  assert.match(resolve, /throw "the resolved root does not match the requested repository/);
  assert.match(resolve, /throw "provisioned snapshot is missing declared components under/);
  assert.match(resolve, /\$head = \(\$pattern -split '\[\\\*\\\?\\\[\]'\)\[0\]/);
  assert.match(resolve, /if \(-not \$head\) \{ continue \}/);
  assert.match(resolve, /foreach \(\$pattern in \(\$env:PROVISION_PATTERNS -split/);
  assert.match(resolve, /\$component = Join-Path \$snapshotRoot \$head\.Replace\('\/', '\\'\)/);
  assert.match(resolve, /if \(-not \(Test-Path -LiteralPath \$component\)\) \{ \$missing \+= \$head \}/);
  // Without Resolve-Path the EndsWith below compares the raw Join-Path output against the suffix
  // it was just built from -- a tautology -- and nothing normalizes a traversal before it.
  assert.match(resolve, /\$root = \(Resolve-Path -LiteralPath \$root\)\.Path/);
});

// sc-18677: provisioning must stay anonymous. This box sets HF_HOME=E:\huggingface, which can hold
// a credential; with an implicit token a gated repo turns a "not entitled" failure into a silent
// success on whoever's token the runner happens to carry. macos-mlx.yml pins both of these for its
// own provisioning block and this lane pinned neither.
test("windows-candle provisioning stays anonymous", async () => {
  const provision = stepBody(await source(".github/workflows/windows-candle.yml"), "Provision exact snapshot");
  assert.match(provision, /HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"/);
  assert.match(provision, /^\s+token=False,$/m);
});

// sc-18677 section 8.1, generalized. GitHub substitutes an input's `default` for any empty dispatch
// value, so a non-empty default makes "the absence of this thing" INEXPRESSIBLE unless the step
// body understands a sentinel. That cost run 31509409586. Any future provision_* input with a
// non-empty default has to make the same decision consciously.
test("every OPTIONAL provision_* input with a non-empty default has a sentinel", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const { names, defaults } = dispatchInputs(workflow);
  const validate = stepBody(workflow, "Validate dispatch inputs");
  const params = stepBody(workflow, "Resolve snapshot provisioning parameters");

  let checked = 0;
  for (const name of names.filter((n) => n.startsWith("provision_"))) {
    const def = defaults[name];
    if (def === undefined || def === "" || def === "false") continue;
    // "Optional" is derived, not listed: the validation step wraps an optional input's checks in
    // `if ($env:NAME)`. A required input (provision_repository, provision_patterns) is checked
    // unconditionally, and "unset" is not a state it can meaningfully have.
    const env = name.toUpperCase();
    const optional = new RegExp(`if \\(\\$env:${env}(_INPUT)?\\) \\{`).test(validate);
    if (!optional) continue;
    checked += 1;
    assert.match(
      params,
      new RegExp(`\\$env:${env}(_INPUT)? -ne '\\.'`),
      `${name} is optional and defaults to ${JSON.stringify(def)}, so an empty dispatch value ` +
        "cannot unset it; the params step must honor a '.' sentinel or the default must be empty",
    );
  }
  assert.ok(checked > 0, "this guard must actually examine an input, or it is vacuous");
});

// sc-18677: the containment checks around the two operator inputs that reach the filesystem.
// `provision_subdir` is joined onto the cache path; each `provision_patterns` entry's literal
// head is joined onto the snapshot root by the component-presence assertion above. Both are
// validated on the SAME condition as the steps that consume them, not on provision_snapshot
// alone -- a five-rung-only dispatch consumes both just the same.
test("windows-candle validates every provisioning input that reaches a path", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  assert.match(
    workflow,
    /if \(\$env:PROVISION_SNAPSHOT -eq 'true' -or \$env:RUN_FIVE_RUNG_REFERENCE -eq 'true' -or \$env:RUN_SC19057_WAN_CAPTURE -eq 'true'\) \{/,
    "provisioning-input validation must cover every capture path that also consumes them",
  );
  assert.match(workflow, /throw 'provision_subdir must not traverse out of the snapshot'/);
  // The root sentinel. GitHub substitutes an input's `default` for any empty dispatch value, so
  // `-f provision_subdir=` resolves to `q4`, not to "no subdir" (proved by run 31509409586).
  // The default must stay `q4` to keep a Krea dispatch identical, so without '.' a
  // root-resolved model like MiniMax-H3 cannot be expressed at all.
  assert.match(
    workflow,
    /if \(\$env:PROVISION_SUBDIR -and \$env:PROVISION_SUBDIR -ne '\.'\) \{/,
    "'.' must mean the snapshot root; an empty input cannot override a non-empty default",
  );
  assert.match(
    workflow,
    /description: "Tier\/subdirectory under the snapshot the resolve step must prove\. Use '\.' for a model whose components sit at the snapshot ROOT/,
    "the sentinel must be documented on the input the operator actually reads",
  );
  assert.match(
    workflow,
    /throw "provision_patterns entries must not traverse out of the snapshot: \$pattern"/,
  );
  assert.match(
    workflow,
    /throw "provision_patterns entries must be relative to the snapshot root: \$pattern"/,
  );
  // Containment is decided by CANONICALIZING, not by matching path segments against '..'.
  // Segment-equality was bypassable two ways: `..*` yields head `..` while the pattern contains
  // no `..` segment, and `.. ` survives -contains yet Win32 strips the trailing space. The
  // GetFullPath probe kills the first class; the TrimEnd(' ', '.') segment check kills the
  // second, which GetFullPath does NOT normalize.
  const validate = stepBody(workflow, "Validate dispatch inputs");
  assert.match(validate, /\$head = \(\(\$pattern -split '\[\\\*\\\?\\\[\]'\)\[0\]\)\.TrimEnd\('\/'\)/);
  assert.match(validate, /if \(-not \$segment\.TrimEnd\(' ', '\.'\)\) \{/);
  // The validate step has its own loop and its own head guard, distinct from the resolve step's.
  // Pin BOTH, scoped: `foreach ($pattern in @($patterns[0]))` checks only the first of thirteen
  // H3 patterns, and `if ($true) { continue }` skips every one, while every other assertion in
  // this file keeps matching because the resolve step still spells them correctly.
  assert.match(validate, /foreach \(\$pattern in \$patterns\) \{/);
  assert.match(validate, /if \(-not \$head\) \{ continue \}/);
  // Both containment guards must stay FATAL. There are two throws with this message -- the
  // segment check and the canonical probe -- and turning either into a Write-Host leaves the
  // string present, so presence alone is not the property worth asserting.
  assert.equal(
    (validate.match(/throw "provision_patterns entries must not traverse out of the snapshot: \$pattern"/g) || []).length,
    2,
    "both the segment check and the canonical-containment probe must throw",
  );
  assert.match(
    validate,
    /\$full\.StartsWith\(\$probe \+ '\\', \[StringComparison\]::OrdinalIgnoreCase\)/,
  );
  // ...and the resolve step canonicalizes independently, so the proof does not rest on
  // validation having run.
  assert.match(
    stepBody(workflow, "Resolve exact snapshot"),
    /throw "declared component escapes the snapshot root: \$pattern"/,
  );
  // provision_cache_dir is written verbatim into $GITHUB_ENV.
  assert.match(workflow, /throw 'provision_cache_dir must be a single line'/);
  assert.match(workflow, /throw 'provision_cache_dir must be an absolute path with a drive letter'/);
  // IsPathRooted accepts `\foo` and `C:foo` -- rooted, but not absolute -- so the check would not
  // mean what its message says. PowerShell 5.1's .NET has no IsPathFullyQualified.
  assert.match(workflow, /\$env:PROVISION_CACHE_DIR_INPUT -notmatch '\^\[A-Za-z\]:\\\\'/);
});

// sc-18691: provisioning must be INDEPENDENT of the compile chain. It used to sit BELOW
// `cargo test -p sceneworks-worker --features backend-candle` with no guard, so an unrelated build
// break made landing weights on the CUDA box impossible rather than merely slow -- the fetch was
// never reached. Epic 17137 hits that concretely at sc-17149, which must land `transformer_ref`
// (+66.28 GB) onto a box whose resident set is already 144.051 GB.
//
// TWO properties, pinned by two separate tests because either alone is insufficient. ORDER without
// SKIP still drags a weights-only run red on an unrelated break and still burns the lane's ~24m of
// box time; SKIP without ORDER leaves a five-rung dispatch's provisioning downstream of the compile
// chain. Both are DERIVED from the workflow's own cargo invocations rather than from a hand-written
// step order, so a NEW compile step added above provisioning, or added unguarded, goes red.

const PROVISIONING_STEPS = [
  "Validate dispatch inputs",
  "Resolve snapshot provisioning parameters",
  "Validate runner Python for snapshot provisioning",
  "Provision exact snapshot",
  "Resolve exact snapshot",
];

const COMPILE_CHAIN_STEPS = [
  "Fetch the pinned inference release",
  "Test the candle GPU worker (backend-candle)",
  "Check the candle sidecar builds (rust-api, backend-candle)",
  "Check and test the candle memory adapter (lib + memory-candle-adapter)",
  "Clippy (candle worker)",
  "Verify capabilities.candle.json content against a fresh dump",
];

// The ORDERED view of the job's steps. `stepBody()` above finds one step by name; this keeps
// position, and keeps the `uses:`-only steps (checkout, prepare-rust-runner) that have no name at
// all and so are invisible to `stepBody`.
function jobSteps(workflow) {
  const start = workflow.indexOf("\n    steps:\n");
  assert.ok(start >= 0, "windows-candle.yml must keep a steps: block");
  const body = workflow.slice(start);
  const steps = [];
  const marker = "\n      - ";
  for (let at = body.indexOf(marker); at !== -1; ) {
    const next = body.indexOf(marker, at + 1);
    const chunk = body.slice(at, next === -1 ? undefined : next);
    const named = chunk.match(/^\n      - name: (.*)$/m);
    const used = chunk.match(/^\n      - uses: (.*)$/m);
    steps.push({
      name: named ? named[1] : `uses:${used ? used[1].trim() : "?"}`,
      body: chunk,
      // A cargo COMMAND, line-initial, so an `echo`ed fix-it message that merely QUOTES
      // `cargo run -p sceneworks-worker` (the restamp step has one) counts as the prose it is.
      cargo: chunk.split("\n").some((line) => /^\s+(run: )?cargo\s/.test(line)),
    });
    at = next;
  }
  return steps;
}

// A step's body with comment lines removed. Both YAML and PowerShell comment with a leading `#`,
// and the counts below must not be movable by editing prose -- in either direction. This file's own
// header comments narrate `throw` and `continue-on-error` as history.
function stepCode(workflow, name) {
  return stepBody(workflow, name)
    .split("\n")
    .filter((line) => !/^\s*#/.test(line))
    .join("\n");
}

test("windows-candle provisions before it compiles anything", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const steps = jobSteps(workflow);
  const indexOf = (name) => {
    const at = steps.findIndex((step) => step.name === name);
    assert.ok(at >= 0, `windows-candle.yml must keep a step named ${name}`);
    return at;
  };

  const compiling = steps.map((step, at) => (step.cargo ? at : -1)).filter((at) => at >= 0);
  // Anti-vacuity. If the derivation stops recognising cargo steps, "provisioning comes first" is
  // trivially true and this whole test means nothing -- which is precisely how the sibling audit at
  // the bottom of this file silently emptied itself when sc-18691 changed a guard's polarity.
  assert.ok(
    compiling.length >= COMPILE_CHAIN_STEPS.length,
    `expected at least ${COMPILE_CHAIN_STEPS.length} cargo steps, derived ${compiling.length}`,
  );
  for (const name of COMPILE_CHAIN_STEPS) {
    assert.ok(steps[indexOf(name)].cargo, `${name} must still be a cargo invocation`);
  }

  const firstCompile = Math.min(...compiling);
  for (const name of PROVISIONING_STEPS) {
    assert.ok(
      indexOf(name) < firstCompile,
      `${name} must run before the first cargo step (${steps[firstCompile].name}); a build ` +
        "break must not be able to starve a weights dispatch",
    );
  }
  // Stronger than "ahead of the compile chain": ahead of the Rust setup too. prepare-rust-runner
  // fails loudly on a broken rustup, and a weights fetch has no business depending on toolchain
  // discovery -- it needs the checkout and the box's Python, nothing else.
  for (const name of PROVISIONING_STEPS) {
    assert.ok(
      indexOf(name) < indexOf("uses:./.github/actions/prepare-rust-runner"),
      `${name} must not depend on toolchain discovery either`,
    );
  }
  // ...but still after the checkout, which every one of them reads (INFERENCE_PIN lives in the
  // worktree, and an unchecked-out repo has no workflow to run).
  assert.ok(
    steps[0].name.startsWith("uses:actions/checkout@"),
    "the checkout must remain the job's first step",
  );
});

test("a weights-only dispatch skips the entire compile chain, pinned per step", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");

  // The WHOLE expression. A weights-only dispatch is `provision_snapshot` true AND
  // neither real execution mode selected; every other shape must still compile. Half of this is
  // not a weaker version of it -- dropping either negative arm would strip the compile chain off
  // that real capture, and dropping `github.event_name == 'workflow_dispatch'` would strip it off
  // every PR and push.
  const skip =
    /^        if: \$\{\{ !\(github\.event_name == 'workflow_dispatch' && inputs\.provision_snapshot && !inputs\.run_five_rung_reference && !inputs\.run_sc19057_wan_capture\) \}\}$/m;

  // SCOPED PER STEP. The expression is byte-identical on all six steps, so one file-wide
  // `assert.match` is satisfied by any single survivor and a narrowing mutation on the other five
  // stays green. That is not hypothetical -- it is trap 1 from sc-18677's third review round, which
  // found three such mutations green against this same file.
  for (const name of COMPILE_CHAIN_STEPS) {
    assert.match(stepBody(workflow, name), skip, `${name} must carry the weights-only skip guard`);
  }

  // Derived backstop, so a NEW compile step cannot appear without a decision: every cargo step is
  // either skipped for a weights-only dispatch, or gated on the five-rung capture or the LTX Eros
  // acceptance capture (SC-18902) -- neither of which is a weights-only dispatch, so both
  // legitimately compile what they run.
  for (const step of jobSteps(workflow).filter((candidate) => candidate.cargo)) {
    assert.ok(
      skip.test(step.body) ||
        /if: \$\{\{[^\n]*inputs\.(run_five_rung_reference|run_ltx_eros_acceptance|run_sc19057_wan_capture)/.test(step.body),
      `${step.name} would compile on a weights-only dispatch; guard it or gate it`,
    );
  }
});

// sc-18691 AC2. Decoupling must not turn a genuine provisioning failure into a silent skip. With
// the compile chain skipped, these five steps ARE the entire verdict of a weights-only dispatch, so
// every failure mode they carry has to stay fatal.
//
// PIN THE THROW, NOT THE MESSAGE -- trap 2 from sc-18677's third review round, and it is live in
// this file right now: "Windows Krea provisioning accepts supported newer Python 3 runtimes" above
// asserts the string `Python 3.12 or newer`, which survives a `throw` -> `Write-Warning` downgrade
// completely intact. Counting `throw`s is what makes a downgrade red, and the count is taken PER
// STEP because a file-wide count is satisfied by adding a throw anywhere else.
test("every failure mode a weights-only dispatch can hit is still fatal", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");

  const throwCounts = {
    // repo-id shape, revision shape, empty allow-list, rooted pattern, the two containment guards
    // (segment shape + canonical probe), subdir shape, subdir traversal, cache-dir newline,
    // cache-dir absoluteness; the three five-rung guards (inference_revision shape, the fixed Krea
    // repository, INFERENCE_PIN agreement); mutual exclusion; and six exact SC-19057 guards.
    "Validate dispatch inputs": 20,
    // Pure derivation: it writes GITHUB_ENV and throws nothing.
    "Resolve snapshot provisioning parameters": 0,
    // A non-zero `python --version`, and a minor version below 12.
    "Validate runner Python for snapshot provisioning": 2,
    // pip install, and the heredoc'd snapshot_download -- `@'...'@ | python -` does not propagate
    // its exit code, so the explicit $LASTEXITCODE check is the only thing that fails the step.
    "Provision exact snapshot": 2,
    // Absent snapshot, canonical-suffix mismatch, escaping component, missing components. The
    // suffix one is the dangerous downgrade: warned rather than thrown, a snapshot that does NOT
    // match the requested repo+revision is exported as SCENEWORKS_PROVISIONED_ROOT and a per-tier
    // VRAM measurement silently runs against the wrong weights.
    "Resolve exact snapshot": 4,
  };
  for (const [name, expected] of Object.entries(throwCounts)) {
    assert.equal(
      (stepCode(workflow, name).match(/\bthrow /g) || []).length,
      expected,
      `${name} must keep exactly ${expected} fatal throw(s); a throw -> Write-Warning downgrade ` +
        "leaves every message assertion in this file green",
    );
  }
  // The Python body guards itself with a `raise`, which the count above cannot see.
  assert.match(
    stepCode(workflow, "Provision exact snapshot"),
    /raise SystemExit\("provision_patterns resolved to an empty allow-list"\)/,
  );

  // Decoupling by SKIPPING is safe; decoupling by SWALLOWING is not. `continue-on-error` on any of
  // these would let a weights-only dispatch report success with no weights on the box -- strictly
  // worse than the coupling this story removed, because the coupling at least failed visibly.
  for (const name of [...PROVISIONING_STEPS, ...COMPILE_CHAIN_STEPS]) {
    assert.doesNotMatch(
      stepCode(workflow, name),
      /continue-on-error|always\(\)/,
      `${name} must not degrade a failure into a warning`,
    );
  }
});

test("Windows CUDA runs the Candle adapter's platform-only unit tests", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  // NO `--bin` selector (sc-18808 review): it excludes the crate's LIB test target, and the crate
  // is outside `default-members`, so a plain `cargo test` never reaches it either. Under the old
  // selector the shared protocol guards in src/lib.rs executed in ZERO lanes on either platform.
  assert.match(workflow, /^ +cargo test -p sceneworks-memory-adapter --features candle$/m);
  assert.doesNotMatch(
    workflow,
    /cargo test -p sceneworks-memory-adapter --features candle --bin/,
    "a --bin selector on the TEST step drops the lib target, where the shared protocol guards live",
  );
  assert.match(workflow, /console\.log\(JSON\.stringify\(a,null,2\)\)/);
  assert.match(workflow, /'amortizable','unable_to_amortize'/);
});

// The MLX twin of the pin above (sc-18250). The adapter crate sits outside the workspace
// default-members, so the macOS lane's bare `cargo test` never compiles it; without this exact
// invocation the memory-mlx-adapter unit tests — including the sc-18104 dispatch-refusal guards —
// run in NO CI lane, and reverting those fixes merges green.
test("macOS MLX runs the MLX adapter's platform-only unit tests", async () => {
  const workflow = await source(".github/workflows/macos-mlx.yml");
  const hostedStart = workflow.indexOf("  macos-checks:");
  const hostedEnd = workflow.indexOf("  nax-worker:");
  assert.ok(hostedStart >= 0, "macos-checks job not found");
  assert.ok(hostedEnd > hostedStart, "nax-worker must follow macos-checks");
  const hosted = workflow.slice(hostedStart, hostedEnd);
  // Same no-`--bin` rule as the Candle twin above (sc-18808 review). `--bin memory-mlx-adapter`
  // silently drops the crate's lib test target; `memory-candle-adapter` still stays out on its own
  // `required-features = ["candle"]`, so omitting the selector widens coverage without widening the
  // lane's platform surface.
  assert.match(
    hosted,
    /^ {6}- name: Test the MLX memory adapter \(lib \+ memory-mlx-adapter\)\n {8}run: cargo test -p sceneworks-memory-adapter --features mlx$/m,
  );
  assert.doesNotMatch(
    hosted,
    /cargo test -p sceneworks-memory-adapter --features mlx --bin/,
    "a --bin selector drops the lib target, where the shared protocol guards live",
  );
});

// The third `--bin` guard (sc-18808 review), and the one that actually runs on a feature-targeted
// PR. The two above pin CI workflows that do NOT: windows-candle.yml has no `pull_request` trigger
// for these branches, so `check.yml`'s hosted `candle` job — which is just
// `scripts/check-candle-build.mjs` — is the ONLY candle-configured compiler a PR here reaches.
// `cargo check --bin` does not compile `#[cfg(test)]`, so under the narrow selector the crate's
// candle test module was not typechecked in any PR-reachable lane at all: a test that failed to
// compile merged green and first broke on the self-hosted `cuda` pool ~24m later.
test("the PR-reachable candle lane typechecks the memory adapter's TESTS, not just its bin", async () => {
  const script = await source("scripts/check-candle-build.mjs");
  assert.match(
    script,
    /\["check", "-p", "sceneworks-memory-adapter", "--features", "candle", "--all-targets"\]/,
  );
  assert.doesNotMatch(
    script,
    /"--bin",\s*\n?\s*"memory-candle-adapter"/,
    "`cargo check --bin` skips #[cfg(test)] — the candle test module would go untypechecked on PRs",
  );
  // And this script is genuinely the lane that runs: check.yml must still call it.
  assert.match(await source(".github/workflows/check.yml"), /run: npm run rust:check:candle$/m);
  assert.match(await source("package.json"), /"rust:check:candle": "node scripts\/check-candle-build\.mjs"/);
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

test("Rust Docker builders copy every production generated embed from sceneworks-core", async () => {
  const coreSources = await Promise.all(
    ["memory_calibration.rs", "video_memory_curves.rs"].map((file) =>
      source(`crates/sceneworks-core/src/${file}`),
    ),
  );
  const generatedEmbeds = new Set(
    [
      ...coreSources
        .join("\n")
        .matchAll(/include_str!\("\.\.\/\.\.\/\.\.\/(docs\/generated\/[^"\n]+)"\)/g),
    ].map((match) => match[1]),
  );
  assert(generatedEmbeds.has("docs/generated/memory-calibration-evidence.json"));
  assert(generatedEmbeds.has("docs/generated/video-memory-curves.json"));
  assert(
    [...generatedEmbeds].some((path) => path.startsWith("docs/generated/ltx-mlx-")),
    "the promoted video-memory curve must compile at least one immutable LTX evidence source",
  );

  const dockerfile = await source("docker/rust.Dockerfile");
  for (const path of generatedEmbeds) {
    const copy = path.startsWith("docs/generated/ltx-mlx-")
      ? "COPY docs/generated/ltx-mlx-*.json ./docs/generated/"
      : `COPY ${path} ./docs/generated/`;
    assert.equal(
      dockerfile.split(copy).length - 1,
      2,
      `${path} must be present in both the ordinary and Candle Rust builder contexts`,
    );
  }
});

test("both Rust Docker builders carry the mechanically digested web capability sources", async () => {
  const dockerfile = await source("docker/rust.Dockerfile");
  const matrix = await source(
    "crates/sceneworks-core/src/jobs_store/routing/matrix.rs",
  );
  const webEmbeds = [
    ...matrix.matchAll(
      /include_str!\(\s*"\.\.\/\.\.\/\.\.\/\.\.\/\.\.\/apps\/web\/src\/([^"\n]+)"\s*\)/g,
    ),
  ].map((match) => match[1]);
  assert.ok(webEmbeds.length > 0, "matrix must retain its production web source inputs");
  const stage = (name) => {
    const heading = new RegExp(`^FROM [^\\r\\n]+ AS ${name}\\r?$`, "m").exec(
      dockerfile,
    );
    assert.ok(heading, `${name} Docker stage must exist`);
    const start = heading.index + heading[0].length;
    const end = dockerfile.indexOf("\nFROM ", start);
    return dockerfile.slice(start, end === -1 ? undefined : end);
  };
  for (const name of ["builder", "candle-builder"]) {
    assert.match(
      stage(name),
      /^COPY apps\/web\/src \.\/apps\/web\/src$/m,
      `${name} must copy the whole owning source root`,
    );
  }
  for (const relative of webEmbeds) {
    assert.doesNotMatch(relative, /(?:^|\/)\.\.(?:\/|$)/, relative);
  }
});

test("both Rust Docker builders carry the compiled memory-calibration evidence", async () => {
  const dockerfile = await source("docker/rust.Dockerfile");
  assert.equal(
    (
      dockerfile.match(
        /^COPY docs\/generated\/memory-calibration-evidence\.json \.\/docs\/generated\/$/gm,
      ) ?? []
    ).length,
    2,
  );
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

// sc-18854. The download-pattern gate is split into a networked RECORDER (`--write`, writes
// config/download-pattern-evidence.json) and a hermetic GATE (`--check`, grades the committed
// listings offline). The offline gate remains available by name after the sc-19758 gate teardown;
// wiring the live/recording mode into any workflow would put huggingface.co on a required context,
// which is exactly what the split exists to prevent. Two GitHub-runner TLS flakes on outbound
// downloads were observed the day this landed.
//
// The negative assertion is the load-bearing half: nothing stops a future change from
// "simplifying" the offline gate back into the live one.
test("the offline download-pattern gate remains callable and its networked modes stay out of every workflow", async () => {
  const pkg = JSON.parse(await source("package.json"));
  assert.match(pkg.scripts["check:download-patterns:offline"], /check-download-patterns\.mjs --self-test/);
  assert.match(pkg.scripts["check:download-patterns:offline"], /check-download-patterns\.mjs --check/);
  // The gate is only as good as its evidence, so the recorder must stay reachable by name.
  assert.match(pkg.scripts["record:download-patterns"], /check-download-patterns\.mjs --write/);

  const dir = new URL("../.github/workflows/", import.meta.url);
  const workflows = (await readdir(dir)).filter((name) => /\.ya?ml$/.test(name));
  assert.ok(workflows.length >= 10, `expected the workflow dir to be populated, got ${workflows.length}`);
  for (const name of workflows) {
    const text = await source(`.github/workflows/${name}`);
    // A bare script invocation not immediately followed by --check/--self-test would be the
    // live (networked) mode.
    assert.doesNotMatch(
      text,
      /check-download-patterns\.mjs(?!\s+--(?:check|self-test))/,
      `${name} must not invoke the live download-pattern check`,
    );
    // ...and the same via the npm aliases. `check:download-patterns:offline` is permitted.
    assert.doesNotMatch(
      text,
      /check:download-patterns(?!:offline)/,
      `${name} must not run the live check:download-patterns alias`,
    );
    assert.doesNotMatch(
      text,
      /record:download-patterns/,
      `${name} must not run the download-pattern recorder`,
    );
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
  // The Cargo fetch itself carries NO credential (sc-17879). `SceneWorks/inference` is public, so
  // the `url.…insteadOf` rewrite this lane used to inject bought nothing, and on a fork -- where
  // `secrets.*` expands to empty -- it emitted `https://x-access-token:@github.com/...` and broke
  // the fetch outright. The dispatch-only calibration checkout above is a separate mechanism and
  // keeps its `|| github.token` fallback.
  assert.doesNotMatch(workflow, /x-access-token:/);
  assert.doesNotMatch(workflow, /GIT_CONFIG_(?:COUNT|KEY_0|VALUE_0)/);
  assert.match(workflow, /--backend mlx/);
  assert.match(workflow, /QWEN_SEED=15511/);
  assert.match(workflow, /QWEN_SEED=16353/);
  assert.match(workflow, /--fixture "qwen-image-\$\{QWEN_TIER\}-seed\$\{QWEN_SEED\}-step2"/);
  assert.match(workflow, /--fresh-per-case/);
  assert.match(workflow, /hash-artifact-inventory\.mjs/);
  assert.match(workflow, /--raw-log-dir "\$SCENEWORKS_MEMORY_CAPTURE_DIR"/);
  assert.match(workflow, /--source-path-prefix "\$SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX"/);
  assert.match(workflow, /--source-root "\$SCENEWORKS_MEMORY_CAPTURE_DIR"/);
  assert.match(workflow, /\$\{\{ runner\.temp \}\}\/memory-mlx-raw/);
  assert.match(
    workflow,
    /qwen_source_path_prefix:\s+description:[^\n]+\s+required: false\s+type: string\s+default: "docs\/calibration\/sc-18353"/,
  );
  assert.match(
    workflow,
    /memory-calibration-harness\.mjs check/,
  );
  // Pinned to the SHAPE, not to a review-time SHA. scripts/lib/action-pins.mjs is the authority on
  // this and says so outright: the control is that the reference resolves to an immutable 40-hex
  // commit, and Dependabot rewrites the SHA in place, so freezing the value here only means every
  // automated bump reddens a test that has nothing to do with what it is checking.
  assert.match(workflow, /actions\/upload-artifact@[0-9a-f]{40}\b/);
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
  // sc-16268: `check:memory-matrix` and `check:tier-integrity` both read Rust sources, but lived
  // only in `npm run check` — so a Rust-only change passed the gate contributors are told to run and
  // failed `parity` in CI. The fix is one string in `rust:check`, which is exactly the kind of
  // wiring a later edit silently undoes; this pins it. (sc-18100 removed the third member,
  // `check:calibration-cost-model`, along with its generator and artifacts.)
  const scripts = JSON.parse(await source("package.json")).scripts;
  for (const sub of [
    "check:memory-matrix",
    "check:tier-integrity",
    // sc-19049: the candle admission inventory + decision baseline. Same shape as its two
    // neighbours — it reads Rust sources, so a Rust-only change can stale it, so it belongs on the
    // Rust gate and not only in `npm run check`.
    "check:candle-admission",
  ]) {
    assert.match(scripts["check:rust-derived-docs"], new RegExp(`\\b${sub}\\b`), sub);
  }
  assert.match(scripts["rust:check"], /\bcheck:rust-derived-docs\b/);
  // sc-19758 removed the `npm run check` arm of this. That chain was 18 steps of pin-keyed gates
  // and is now the unit tests alone; the derived-docs check keeps its two other entry points, the
  // `rust:check` gate above and the pre-push hook below, both of which still run it.
  // The pre-push hook runs it too, on the same trigger as the neither/candle builds.
  assert.match(await source("scripts/git-hooks/pre-push"), /npm run --silent check:rust-derived-docs/);
});

test("the pre-push derived-docs trigger covers every non-Rust source the matrix is hashed from", async () => {
  // sc-18098. `check:rust-derived-docs` catches a stale matrix in under a second; the pre-push hook
  // only runs it when the pushed diff LOOKS like it touched an input. Its Rust/Cargo arm covers the
  // `.rs` and `Cargo.toml` entries of `generate-memory-matrix.mjs#SOURCE_PATHS`; this arm has to
  // cover the rest, and two of them — the closure table and the rung-4 survey — were missing, so a
  // pin bump that re-derived one lane's closure digest pushed clean and heard about the stale matrix
  // from `parity` fifteen minutes later.
  //
  // Derived from SOURCE_PATHS rather than restated, so a NEW hashed source is covered or this reds
  // (the epic-18093 slices are actively adding and removing them).
  const hook = await source("scripts/git-hooks/pre-push");
  const pattern = hook.match(/'(\^\(config\/manifests[^']+)'/)?.[1];
  assert.ok(pattern, "the derived-docs trigger pattern is still a single-quoted ERE in the hook");
  const trigger = new RegExp(pattern);
  const rustArm = /(^|\/)([^/]+\.rs|Cargo\.(toml|lock)|rustfmt\.toml)$/;
  // Imported reconciliation logic also changes the generated summary/gate even though it is code,
  // not a hashed data source, so a module-only edit must run the same stale-artifact check.
  const derivedInputs = [
    ...Object.values(SOURCE_PATHS),
    "scripts/lib/memory-contract-reconciliation.mjs",
    // sc-19049: the candle admission inventory hangs off the same gate and hashes its own source
    // set, which overlaps the matrix's but is not a subset of it (the packaged video curves and the
    // candle request-scope module are hashed here and nowhere else). Derived, not restated, for the
    // same reason the matrix's list is.
    ...Object.values(CANDLE_ADMISSION_SOURCE_PATHS),
    "scripts/generate-candle-admission-inventory.mjs",
    "scripts/lib/manifest-memory-declarations.mjs",
    // sc-19049: the Rust-emitted decisions the baseline is BUILT FROM. Deliberately outside
    // CANDLE_ADMISSION_SOURCE_PATHS (fingerprinting an artifact derived from the inventory would
    // close a regenerate/hash loop), so it has to be listed here or an edit to the decisions would
    // push a stale baseline and hear about it from `parity`.
    CANDLE_ADMISSION_DECISIONS_PATH,
  ];
  for (const relative of derivedInputs) {
    if (rustArm.test(relative)) continue;
    assert.ok(trigger.test(relative), `${relative} must trigger the pre-push derived-docs check`);
  }
  // The pattern is anchored, not a substring sweep: a same-named file elsewhere must not fire it.
  assert.equal(trigger.test("vendor/config/inference-provider-closures.json"), false);
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

test("both stage-1 lanes verify native capability content last among coverage, reachably, and publish only its evidence", async () => {
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
    const verifyAt = lane.indexOf(`- name: Verify ${file} content against a fresh dump`);
    assert.ok(verifyAt > 0, `${path} must verify ${file} against a fresh dump`);
    if (path.endsWith("macos-mlx.yml")) {
      const hostedAt = lane.indexOf("\n  macos-checks:");
      const naxAt = lane.indexOf("\n  nax-worker:");
      assert.ok(
        hostedAt < verifyAt && verifyAt < naxAt,
        "the weights-free MLX facts producer belongs on the hosted Mac job, not the M5/NAX pool",
      );
      assert.doesNotMatch(
        lane.slice(naxAt),
        /Verify capabilities\.mlx\.json|Upload fresh MLX capability facts/,
        "the NAX-only job must not duplicate the hosted MLX facts producer",
      );
    }
    // Re-dump to a SCRATCH dir and compare all capability content while deliberately allowing the
    // two valid inference revision labels to differ. Pin-only label drift is not capability drift
    // (e14171984); every other field remains part of the native comparison.
    assert.match(lane, /bin dump-engine-capabilities/, path);
    assert.match(lane, /node scripts[\\/]compare-engine-capability-facts\.mjs/, path);

    // LAST on the PR path. A step failure aborts the job, so a real native capability mismatch placed
    // earlier would cancel the coverage each lane uniquely carries (macOS: the hosted full workspace
    // suite; Windows: the only PR run of `cargo test -p sceneworks-worker --features backend-candle`).
    // A missing dump must not suppress unrelated verdicts.
    //
    // "Last" means last among steps that RUN in the same job on a pull request. The Mac workflow has
    // a later, separate NAX job; bounding this scan at the next job key keeps its M5-only steps out.
    // Asserting the ordering rather than mere presence is
    // the point — nothing else would notice an unconditional step being appended later.
    const afterVerify = lane.slice(verifyAt);
    const nextJobAt = afterVerify.search(/\n {2}[A-Za-z0-9_-]+:\n/);
    const verifyJobTail = nextJobAt < 0 ? afterVerify : afterVerify.slice(0, nextJobAt);
    for (const block of verifyJobTail.split(/\n {6}- (?=name: |uses: )/).slice(1)) {
      // The Candle lane publishes its fresh dump ONLY when verification failed, so the upload is
      // diagnostic evidence for a red run rather than a per-PR measurement publication.
      const candleFailureArtifact =
        path === ".github/workflows/windows-candle.yml" &&
        block.startsWith(
          "name: Upload fresh Candle capability facts after a verification failure",
        ) &&
        /if: \$\{\{ always\(\) && steps\.verify_candle_capabilities\.outcome == 'failure' \}\}/.test(
          block,
        );
      if (candleFailureArtifact) continue;
      if (/^name: Upload fresh MLX capability facts/m.test(block)) {
        assert.match(block, /if: \$\{\{ always\(\) \}\}/);
        assert.match(block, /uses: actions\/upload-artifact@[0-9a-f]{40}/);
        assert.match(block, /path: \$\{\{ runner\.temp \}\}\/engine-capability-facts-verify/);
        continue;
      }
      assert.match(
        block,
        /if: \$\{\{[^\n]*github\.event_name == 'workflow_dispatch'/,
        `${path}: "${block.split("\n")[0]}" runs after the dump-verification step on the PR path. ` +
          "That step must stay last for everything a PR executes, so its failure cannot cancel " +
          "coverage this lane is the only place to have. Move it above the verification step.",
      );
    }

    // The rich runtime descriptor artifact is part of the same native evidence contract. It must
    // be generated by the one matching-platform producer and content-compared beside the legacy
    // preview projection; hashing a narrow supportsPreview file is not descriptor drift protection.
    assert.match(lane, /runtime[\\/]capabilities\.(?:mlx|candle)\.json/);
    assert.match(lane, /backend-capability-facts-(?:mlx|candle)/);

    // Reachability. A facts edit can touch ONLY the facts file, so without this path entry the lane
    // does not run at all on the PR the step exists to catch — declared but unreachable, the same
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
      `${path} must watch ${own}, or a content edit to it — which touches nothing else — never ` +
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

// Kept as its own test rather than folded into the `candleFailureArtifact` carve-out above, which was
// raised as possible duplication. It is not: that carve-out is a `continue` GUARD, so it can only ever
// weaken the ordering rule, never assert anything. Three claims below have nowhere else to live —
//
//   * the upload step EXISTS. Delete it and the guard simply stops matching, the ordering loop finds no
//     such block, and every assertion up there still passes. The failure-only upload would be gone with
//     nothing red.
//   * the VERIFY step declares `id: verify_candle_capabilities`. Without the id,
//     `steps.verify_candle_capabilities.outcome` resolves to nothing, the condition is false on every
//     run, and the upload never fires — while the guard's literal text match keeps passing.
//   * the artifact CONTENT: name, whole-directory path, if-no-files-found. See the note below on the
//     enumerated two-file spelling that silently produced an unusable artifact.
//
// The `if:` expression is deliberately spelled in both places: up there it is the condition under which
// the carve-out is legitimate, here it is the failure-only guarantee itself.
test("Windows preserves fresh capability facts when content verification fails", async () => {
  const workflow = await source(".github/workflows/windows-candle.yml");
  const verifyAt = workflow.indexOf(
    "- name: Verify capabilities.candle.json content against a fresh dump",
  );
  const uploadAt = workflow.indexOf(
    "- name: Upload fresh Candle capability facts after a verification failure",
  );
  assert.ok(verifyAt > 0 && uploadAt > verifyAt);
  const tail = workflow.slice(verifyAt, uploadAt + 1000);
  assert.match(tail, /id: verify_candle_capabilities/);
  assert.match(
    tail,
    /if: \$\{\{ always\(\) && steps\.verify_candle_capabilities\.outcome == 'failure' \}\}/,
  );
  // The shape, not a review-time SHA — see the note in the calibration-artifact test above.
  assert.match(tail, /actions\/upload-artifact@[0-9a-f]{40}\b/);
  // The whole scratch DIRECTORY, not an enumerated file list, and that distinction has already
  // been load-bearing once. The dumper writes three files — `capabilities.candle.json`,
  // `audio/capabilities.candle.json`, and the rich `runtime/capabilities.candle.json` — and the
  // runtime descriptor is the one the backend capability matrix cannot be rebuilt without. The
  // enumerated two-file spelling this used to assert predates that third file, so an artifact
  // produced under it looks complete and silently cannot repair the matrix. Pin the directory and
  // the artifact name the repair instructions actually tell you to download.
  assert.match(tail, /name: backend-capability-facts-candle/);
  assert.match(tail, /path: \$\{\{ runner\.temp \}\}\/engine-capability-facts-verify\s/);
  assert.match(tail, /if-no-files-found: warn/);
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
        "config/engine-capabilities/capabilities.candle.json", // the native verification step compares it
        "config/engine-capabilities/runtime/capabilities.candle.json", // same step compares rich descriptor + worker facts
        // The audio dump the SAME step also diffs (sc-17593). On BOTH lanes, unlike the media
        // files: AUDIO_BACKEND is candle everywhere, so either box produces this one file and
        // both verify steps open it. That is the test sc-17703 applies — a step here reads it —
        // and not symmetry for its own sake.
        "config/engine-capabilities/audio/capabilities.candle.json",
        "scripts/compare-engine-capability-facts.mjs", // invoked directly by the native verification step
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
        "config/engine-capabilities/capabilities.mlx.json", // the native verification step compares it
        "config/engine-capabilities/runtime/capabilities.mlx.json", // same step compares rich descriptor + worker facts
        // The audio dump the SAME step also diffs (sc-17593). On BOTH lanes, unlike the media
        // files: AUDIO_BACKEND is candle everywhere, so either box produces this one file and
        // both verify steps open it. That is the test sc-17703 applies — a step here reads it —
        // and not symmetry for its own sake.
        "config/engine-capabilities/audio/capabilities.candle.json",
        "scripts/compare-engine-capability-facts.mjs", // invoked directly by the native verification step
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
    //
    // POLARITY MATTERS, and it did not used to (sc-18691). This filter was a bare substring test
    // for `github.event_name == 'workflow_dispatch'` inside the step's `if:`. sc-18691 added the
    // opposite polarity to windows-candle.yml -- `!(github.event_name == 'workflow_dispatch' &&
    // inputs.provision_snapshot && !inputs.run_five_rung_reference &&
    // !inputs.run_sc19057_wan_capture)`, which skips the compile chain for a weights-only dispatch
    // and therefore RUNS on every PR and push. The substring test read
    // those two forms as identical, dropped all six compile steps, and left this audit with zero
    // cargo invocations to reason about -- it survived only because of the `invocations.length > 0`
    // assertion below, which is exactly the vacuity backstop that case is for. Strip negated groups
    // before looking for the positive requirement, so "requires a dispatch" means what it says.
    const requiresDispatch = (block) => {
      const gate = block.match(/^\s*if: ([^\n]*)$/m);
      if (!gate) return false;
      return /github\.event_name == 'workflow_dispatch'/.test(gate[1].replace(/!\([^)]*\)/g, ""));
    };
    const blocks = lane.split(/\n {6}- (?=name: |uses: )/);
    const prSteps = blocks.filter((block) => !requiresDispatch(block)).join("\n");
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

const REQUIRED_WORKFLOWS = [
  ".github/workflows/check.yml",
  ...REQUIRED_LANES.map(({ path }) => path),
];

function stripYamlComment(line) {
  let quote;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quote === '"' && char === "\\") {
      i += 1;
    } else if (quote === "'" && char === "'" && line[i + 1] === "'") {
      i += 1;
    } else if (quote && char === quote) {
      quote = undefined;
    } else if (!quote && (char === '"' || char === "'")) {
      quote = char;
    } else if (!quote && char === "#" && (i === 0 || /\s/.test(line[i - 1]))) {
      return line.slice(0, i).trimEnd();
    }
  }
  return line;
}

function yamlScalar(value, context) {
  const scalar = value.trim();
  assert.ok(scalar.length > 0, `${context}: empty branch pattern`);
  if (scalar.startsWith('"')) {
    assert.ok(scalar.endsWith('"'), `${context}: unterminated double-quoted branch pattern`);
    return JSON.parse(scalar);
  }
  if (scalar.startsWith("'")) {
    assert.ok(scalar.endsWith("'"), `${context}: unterminated single-quoted branch pattern`);
    return scalar.slice(1, -1).replaceAll("''", "'");
  }
  assert.doesNotMatch(
    scalar,
    /[\[\]{},]/,
    `${context}: unsupported branch-pattern syntax; refusing to assume feature PR coverage`,
  );
  return scalar;
}

function splitFlowItems(value, context) {
  const items = [];
  let quote;
  let squareDepth = 0;
  let curlyDepth = 0;
  let start = 0;
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    if (quote === '"' && char === "\\") {
      i += 1;
    } else if (quote === "'" && char === "'" && value[i + 1] === "'") {
      i += 1;
    } else if (quote && char === quote) {
      quote = undefined;
    } else if (!quote && (char === '"' || char === "'")) {
      quote = char;
    } else if (!quote && char === "[") {
      squareDepth += 1;
    } else if (!quote && char === "]") {
      squareDepth -= 1;
    } else if (!quote && char === "{") {
      curlyDepth += 1;
    } else if (!quote && char === "}") {
      curlyDepth -= 1;
    } else if (!quote && char === "," && squareDepth === 0 && curlyDepth === 0) {
      items.push(value.slice(start, i).trim());
      start = i + 1;
    }
    assert.ok(squareDepth >= 0 && curlyDepth >= 0, `${context}: unbalanced flow collection`);
  }
  assert.ok(!quote && squareDepth === 0 && curlyDepth === 0, `${context}: unclosed flow collection`);
  items.push(value.slice(start).trim());
  return items.filter(Boolean);
}

function branchSequence(value, context) {
  const sequence = value.trim();
  assert.match(
    sequence,
    /^\[.*\]$/,
    `${context}: branch filters must use an explicit sequence; refusing to guess`,
  );
  return splitFlowItems(sequence.slice(1, -1), context).map((item) => yamlScalar(item, context));
}

function flowPullRequestFilters(value, context) {
  assert.match(
    value,
    /^\{.*\}$/,
    `${context}: unsupported pull_request value; refusing to assume feature PR coverage`,
  );
  let branches;
  let hasBranchesIgnore = false;
  for (const item of splitFlowItems(value.slice(1, -1), context)) {
    const separator = item.indexOf(":");
    assert.ok(separator > 0, `${context}: malformed pull_request flow mapping`);
    const key = yamlScalar(item.slice(0, separator), context);
    const rawValue = item.slice(separator + 1);
    if (key === "<<") {
      assert.fail(`${context}: inherited pull_request filters cannot prove feature PR coverage`);
    } else if (key === "branches") {
      assert.equal(branches, undefined, `${context}: duplicate branches filter`);
      branches = branchSequence(rawValue, `${context} branches`);
    } else if (key === "branches-ignore") {
      hasBranchesIgnore = true;
    }
  }
  return { branches, hasBranchesIgnore };
}

// This is deliberately a narrow reader, not a general YAML parser. It accepts the block and
// single-line flow forms used by Actions, strips real comments without stripping quoted `#`, and
// throws on aliases or syntax it cannot prove safe. A required check must fail closed here: treating
// an unknown shape as unfiltered could leave every feature-target PR permanently Pending.
function pullRequestBranchFilters(workflow, context) {
  const lines = workflow.split(/\r?\n/).map(stripYamlComment);
  const onDeclarations = lines
    .map((line, index) => (/^on\s*:/.test(line) ? index : -1))
    .filter((index) => index >= 0);
  assert.equal(
    onDeclarations.length,
    1,
    `${context}: required workflow must declare one top-level on mapping`,
  );
  const [onAt] = onDeclarations;
  const onDeclaration = /^on\s*:\s*(.*)$/.exec(lines[onAt]);
  assert.equal(
    onDeclaration[1].trim(),
    "",
    `${context}: flow-style top-level on mappings are unsupported; refusing to guess`,
  );

  let onEnd = lines.length;
  for (let i = onAt + 1; i < lines.length; i += 1) {
    if (lines[i].trim() && /^\S/.test(lines[i])) {
      onEnd = i;
      break;
    }
  }
  const eventLines = lines.slice(onAt + 1, onEnd).filter((line) => line.trim());
  assert.ok(eventLines.length > 0, `${context}: top-level on mapping is empty`);
  const eventIndent = Math.min(...eventLines.map((line) => /^ */.exec(line)[0].length));
  assert.ok(eventIndent > 0, `${context}: malformed top-level on mapping`);

  const pullRequestDeclarations = [];
  for (let i = onAt + 1; i < onEnd; i += 1) {
    const line = lines[i];
    if (!line.trim()) continue;
    const indent = /^ */.exec(line)[0].length;
    assert.ok(indent >= eventIndent, `${context}: inconsistent event indentation`);
    if (indent !== eventIndent) continue;
    const declaration = /^\s*((?:"[^"]*"|'[^']*'|[A-Za-z0-9_-]+))\s*:\s*(.*)$/.exec(line);
    assert.ok(declaration, `${context}: unrecognized event declaration: ${line.trim()}`);
    if (yamlScalar(declaration[1], context) === "pull_request") {
      pullRequestDeclarations.push({ index: i, inline: declaration[2].trim() });
    }
  }
  assert.equal(
    pullRequestDeclarations.length,
    1,
    `${context}: top-level on mapping must declare pull_request exactly once`,
  );
  const [{ index: pullRequestAt, inline }] = pullRequestDeclarations;
  if (inline) return flowPullRequestFilters(inline, context);

  let pullRequestEnd = onEnd;
  for (let i = pullRequestAt + 1; i < onEnd; i += 1) {
    if (!lines[i].trim()) continue;
    const indent = /^ */.exec(lines[i])[0].length;
    if (indent === eventIndent) {
      pullRequestEnd = i;
      break;
    }
    assert.ok(indent > eventIndent, `${context}: inconsistent pull_request indentation`);
  }
  const childLines = lines.slice(pullRequestAt + 1, pullRequestEnd).filter((line) => line.trim());
  if (childLines.length === 0) return { branches: undefined, hasBranchesIgnore: false };
  const childIndent = Math.min(...childLines.map((line) => /^ */.exec(line)[0].length));
  assert.ok(childIndent > eventIndent, `${context}: malformed pull_request mapping`);

  let branches;
  let hasBranchesIgnore = false;
  for (let i = pullRequestAt + 1; i < pullRequestEnd; i += 1) {
    const line = lines[i];
    if (!line.trim()) continue;
    const indent = /^ */.exec(line)[0].length;
    assert.equal(
      indent,
      childIndent,
      `${context}: unrecognized pull_request child line: ${line.trim()}`,
    );
    const keyMatch = /^((?:"[^"]*"|'[^']*'|[A-Za-z0-9_-]+|<<))\s*:\s*(.*)$/.exec(
      line.slice(childIndent),
    );
    assert.ok(keyMatch, `${context}: unrecognized pull_request child line: ${line.trim()}`);
    const key = yamlScalar(keyMatch[1], context);
    if (key === "<<") {
      assert.fail(`${context}: inherited pull_request filters cannot prove feature PR coverage`);
    }

    const values = [];
    if (keyMatch[2].trim()) {
      if (key === "branches" || key === "branches-ignore") {
        values.push(...branchSequence(keyMatch[2], `${context} ${key}`));
      }
    } else {
      let nestedIndent;
      for (let j = i + 1; j < pullRequestEnd; j += 1) {
        if (!lines[j].trim()) continue;
        const indent = /^ */.exec(lines[j])[0].length;
        if (indent <= childIndent) break;
        nestedIndent ??= indent;
        assert.equal(
          indent,
          nestedIndent,
          `${context}: inconsistent indentation under pull_request ${key}`,
        );
        const item = /^-\s+(.+)$/.exec(lines[j].slice(nestedIndent));
        assert.ok(item, `${context}: unrecognized value under pull_request ${key}`);
        if (key === "branches" || key === "branches-ignore") {
          values.push(yamlScalar(item[1], `${context} ${key}`));
        }
        i = j;
      }
    }

    if (key === "branches") {
      assert.equal(branches, undefined, `${context}: duplicate branches filter`);
      branches = values;
    } else if (key === "branches-ignore") {
      hasBranchesIgnore = true;
    }
  }
  return { branches, hasBranchesIgnore };
}

function assertFeaturePullRequestCoverage(workflow, context) {
  const { branches, hasBranchesIgnore } = pullRequestBranchFilters(workflow, context);
  assert.equal(
    hasBranchesIgnore,
    false,
    `${context}: branches-ignore is not allowed on a required workflow because it can exclude ` +
      "feature integration PRs",
  );
  if (!branches) return;
  assert.ok(branches.includes("main"), `${context}: its PR base filter must retain main`);
  assert.ok(
    branches.includes("feature/*"),
    `${context}: its PR base filter must include feature/* so the required check reports on ` +
      "story PRs into a feature integration branch",
  );
  assert.equal(
    branches.some((pattern) => pattern.startsWith("!")),
    false,
    `${context}: negative branch filters cannot prove coverage of every feature integration PR`,
  );
}

function assertWindowsResolvedCacheRuntimeCoverage(workflow, context) {
  const lines = workflow.split("\n");
  const buildStart = lines.findIndex((line) => line === "  build-windows:");
  assert.ok(buildStart >= 0, `${context}: build-windows job must be present`);
  const nextJob = lines
    .slice(buildStart + 1)
    .findIndex((line) => /^  [A-Za-z0-9_-]+:\s*$/.test(line));
  assert.ok(nextJob >= 0, `${context}: build-windows job must have a following job boundary`);
  const requiredBuild = lines
    .slice(buildStart, buildStart + 1 + nextJob)
    .filter((line) => !/^\s*#/.test(line))
    .join("\n");
  assert.match(
    requiredBuild,
    /^ {8}run: "cargo test -p sceneworks-core model_artifacts::resolved_cache:: -- --test-threads=1"$/m,
    `${context}: required build-windows must execute both the durable-store and native ` +
      "materialization safety suites, not only compile or select one half",
  );
}

test("every required workflow reports on feature-target pull requests", async () => {
  for (const path of REQUIRED_WORKFLOWS) {
    assertFeaturePullRequestCoverage(await source(path), path);
  }
});

test("feature-target coverage rejects inline and multiline branches-ignore filters", () => {
  for (const workflow of [
    'on:\n  pull_request:\n    branches-ignore: ["feature/*"]\n',
    'on:\n  pull_request:\n    branches-ignore:\n      - "feature/*"\n',
    'on:\n  pull_request: { branches-ignore: ["feature/*"] }\n',
  ]) {
    assert.throws(
      () => assertFeaturePullRequestCoverage(workflow, "mutated workflow"),
      /branches-ignore/,
    );
  }
});

test("feature-target coverage ignores comments and rejects negative or main-only filters", () => {
  for (const [workflow, error] of [
    ['on:\n  pull_request:\n    branches: [main] # "feature/*"\n', /must include feature\/\*/],
    [
      'on:\n  pull_request:\n    # branches: [main, "feature/*"]\n    branches: [main]\n',
      /must include feature\/\*/,
    ],
    [
      'on:\n  pull_request:\n    branches: [main, "feature/*", "!feature/*"]\n',
      /negative branch filters/,
    ],
    ['on:\n  pull_request: { branches: [main] } # "feature/*"\n', /must include feature\/\*/],
  ]) {
    assert.throws(() => assertFeaturePullRequestCoverage(workflow, "mutated workflow"), error);
  }
});

test("feature-target coverage cannot be supplied by a job named pull_request", () => {
  const workflow = [
    "on:",
    "  push:",
    "jobs:",
    "  pull_request:",
    '    branches: [main, "feature/*"]',
  ].join("\n");
  assert.throws(
    () => assertFeaturePullRequestCoverage(workflow, "mutated workflow"),
    /top-level on mapping must declare pull_request exactly once/,
  );
});

test("feature-target coverage parses arbitrary child indentation and rejects malformed children", () => {
  assert.throws(
    () =>
      assertFeaturePullRequestCoverage(
        "on:\n  pull_request:\n   branches: [main]\n",
        "three-space mutation",
      ),
    /must include feature\/\*/,
  );
  assert.doesNotThrow(() =>
    assertFeaturePullRequestCoverage(
      'on:\n  pull_request:\n   branches: [main, "feature/*"]\n',
      "three-space workflow",
    ),
  );
  assert.throws(
    () =>
      assertFeaturePullRequestCoverage(
        'on:\n  pull_request:\n   branches [main, "feature/*"]\n',
        "malformed workflow",
      ),
    /unrecognized pull_request child line/,
  );
});

test("feature-target coverage accepts unfiltered, block, and flow branch declarations", () => {
  for (const workflow of [
    "on:\n  pull_request:\n  push:\n    branches: [main]\n",
    'on:\n  pull_request:\n    branches:\n      - main\n      - "feature/*"\n',
    'on:\n  pull_request:\n    branches: [main, "feature/*"]\n    types: [opened, synchronize]\n',
    'on:\n  pull_request: { branches: [main, "feature/*"], types: [opened] }\n',
  ]) {
    assert.doesNotThrow(() => assertFeaturePullRequestCoverage(workflow, "valid workflow"));
  }
});

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
  // Pin the two PROPERTIES, not the spelling: nax-worker must be excluded from merge groups AND
  // consult the gate. A spelling-exact assertion broke the moment the fail-open clause was added,
  // which is the wrong kind of brittleness for a guard protecting a developer's daily driver.
  const naxCondition = [...mlx.matchAll(/^ {4}if: (\$\{\{[^\n]*\}\})$/gm)]
    .map((match) => match[1])
    .find((condition) => condition.includes("github.event_name != 'merge_group'"));
  assert.ok(
    naxCondition,
    "nax-worker must stay excluded from merge groups — the two-Mac nax pool is not queue capacity.",
  );
  assert.match(
    naxCondition,
    /needs\.changes/,
    "nax-worker must consult the `changes` gate — without it, every docs-only PR now wakes the " +
      "two-Mac nax pool, because the pull_request path filter that used to do this is gone.",
  );
  assert.match(
    naxCondition,
    /github\.event\.pull_request\.base\.ref == 'main'/,
    "nax-worker must not auto-run for story PRs targeting feature/*; capture NAX evidence with " +
      "an explicit dispatch at the frozen feature head, then run it again on the final PR to main.",
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
  assertWindowsResolvedCacheRuntimeCoverage(desktop, ".github/workflows/desktop-windows.yml");
});

test("Windows resolved-cache coverage rejects either narrowed test module", async () => {
  const workflow = await source(".github/workflows/desktop-windows.yml");
  for (const narrowed of [
    "model_artifacts::resolved_cache::tests::",
    "model_artifacts::resolved_cache::materialization::tests::",
  ]) {
    const mutated = workflow.replace("model_artifacts::resolved_cache:: --", `${narrowed} --`);
    assert.notEqual(mutated, workflow, "the workflow mutation must replace the widened filter");
    assert.throws(
      () => assertWindowsResolvedCacheRuntimeCoverage(mutated, `${narrowed} mutation`),
      /execute both the durable-store and native materialization safety suites/,
    );
  }

  const decoy = workflow
    .replace(
      '        run: "cargo test -p sceneworks-core model_artifacts::resolved_cache:: -- --test-threads=1"',
      '        run: "cargo test -p sceneworks-core model_artifacts::resolved_cache::tests:: -- --test-threads=1"\n' +
        '        # run: "cargo test -p sceneworks-core model_artifacts::resolved_cache:: -- --test-threads=1"',
    )
    .replace(
      "\n  package-windows:",
      '\n  decoy-resolved-cache:\n    steps:\n      - run: "cargo test -p sceneworks-core model_artifacts::resolved_cache:: -- --test-threads=1"\n\n  package-windows:',
    );
  assert.throws(
    () => assertWindowsResolvedCacheRuntimeCoverage(decoy, "comment and sibling decoy mutation"),
    /execute both the durable-store and native materialization safety suites/,
  );
});

test("windows-candle stays out of the queue and out of the required set", async () => {
  // ~24m median, p90 32m (measured 2026-08-05 over 85 runs) against a 60m check-response timeout,
  // on the self-hosted `cuda` pool. Its merge-time stand-in is check.yml's hosted `candle`
  // typecheck. Making candle-worker required would force a merge_group: trigger here, and p90
  // queue wait (18m) + p90 run already reaches ~50m of the 60m budget.
  const candle = await source(".github/workflows/windows-candle.yml");
  assert.doesNotMatch(
    candle,
    /^ {2}merge_group:$/m,
    "windows-candle.yml must stay out of the merge queue; check.yml's `candle` job is its " +
      "merge-time stand-in. See sc-17014 for the (A)/(B)/(C) decision if this changes.",
  );
  const prAt = candle.indexOf("\n  pull_request:");
  assert.ok(prAt > 0, "windows-candle.yml must retain its pull_request trigger for final main PRs");
  const prBlock = candle.slice(prAt + 1).split(/\n {2}(?=[a-z_]+:)/)[0];
  assert.match(prBlock, /^ {4}branches: \[main\]$/m);
  assert.doesNotMatch(
    prBlock,
    /feature\/\*/,
    "windows-candle must not auto-run on feature story PRs; dispatch it at the frozen feature head.",
  );
  assert.match(
    candle.slice(0, candle.indexOf("\njobs:")),
    /^ {2}workflow_dispatch:$/m,
    "windows-candle must remain dispatchable for authoritative final-head evidence.",
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

test("a broken relevance gate runs the lane instead of silently passing its required check", async () => {
  // The false-green one level up. In GitHub Actions a job whose `needs:` FAILED is skipped, and a
  // skipped job reports Success — which SATISFIES a required status check. So a gate that dies
  // (checkout failure, dead runner, a syntax error in the reusable workflow) would skip the lane
  // and turn its required check green without running anything.
  //
  // Every gated job must therefore fail OPEN: `!cancelled()` to survive a failed dependency at all,
  // plus a `result != 'success'` clause so a non-green gate means RUN rather than skip. This
  // mirrors merge-group-relevance.mjs's own internal fallback, where an unparseable anchor or a
  // failed diff also runs the lane.
  for (const { path } of REQUIRED_LANES) {
    const lane = await source(path);
    const conditions = [...lane.matchAll(/^ {4}if: (\$\{\{ [^\n]*needs\.changes[^\n]*\}\})$/gm)].map(
      (match) => match[1],
    );
    assert.ok(
      conditions.length > 0,
      `${path}: expected at least one job conditioned on the changes gate`,
    );
    for (const condition of conditions) {
      assert.match(
        condition,
        /!cancelled\(\)/,
        `${path}: gated job must use !cancelled(), or a FAILED gate skips it — and a skipped job ` +
          "satisfies a required check, so the lane silently never runs.",
      );
      assert.match(
        condition,
        /needs\.changes\.result != 'success'/,
        `${path}: gated job must run when the gate did not succeed. A bare ` +
          "`relevant == 'true'` treats a broken gate as 'not relevant' — a false green.",
      );
    }
  }
});

test("hosted required checks do not skip fork PRs into a vacuous green", async () => {
  // A same-repo guard on a REQUIRED check is a false green: the fork PR matches it, the job is
  // SKIPPED, and a skipped job reports Success. `macos-checks` carried one justified by
  // "SceneWorks/inference is private, a fork cannot read it" — which is not true. The repo is
  // public and was never intended to be private; the pinned rev fetches with no token at all, and
  // also with the EMPTY token a fork PR supplies.
  //
  // Self-hosted jobs are the exception and MUST keep their guards: untrusted code on a persistent
  // box is a real concern that has nothing to do with repo visibility.
  const mlx = await source(".github/workflows/macos-mlx.yml");
  const macosChecks = mlx.slice(mlx.indexOf("  macos-checks:"), mlx.indexOf("  nax-worker:"));
  assert.ok(macosChecks.length > 0, "macos-checks job not found");
  assert.doesNotMatch(
    macosChecks,
    /head\.repo\.full_name/,
    "macos-checks is a HOSTED required check — a same-repo guard makes a fork PR skip it, and a " +
      "skipped job satisfies the required check, so the macOS verdict goes green having run nothing.",
  );
  // Positive half, so this never reads as "guards are bad".
  const nax = mlx.slice(mlx.indexOf("  nax-worker:"));
  assert.match(
    nax,
    /head\.repo\.full_name/,
    "nax-worker MUST keep its same-repo guard — fork code must not execute on the nax pool.",
  );
  assert.match(
    await source(".github/workflows/windows-candle.yml"),
    /head\.repo\.full_name/,
    "candle-worker MUST keep its same-repo guard — fork code must not execute on the cuda pool.",
  );
});

test("the FLUX.2 composition audit still runs, and is still wired into a lane", async () => {
  // sc-17607's composition check answers "which provider is registered under `flux2_dev`" — a
  // question no code digest can answer in either direction, which is why it is a pointer-identity
  // test rather than part of the calibration closure. It is NOT an invalidation mechanism and
  // survived sc-17774's removal of the per-model ones.
  //
  // Its liveness guard did not, at first: it lived in `scripts/inference-artifact-audit.test.mjs`,
  // which sc-17774 deleted along with the flux2-only audit tooling that file existed to grade. The
  // guard itself was never about that tooling, so it is restored here — in a file every lane runs.
  //
  // Why it has to live OUTSIDE the module: `flux2_composition_audit` is
  // `cfg(all(test, not(macos), backend-candle))`, so no macOS or non-candle lane executes anything
  // in it. Deleting its tests, or mistyping its cfg, would fail nothing anywhere.
  //
  // Named `#[test]` functions rather than keywords, because every keyword also appears in that
  // module's own prose — a substring check would pass with all three tests deleted.
  const composition = await source("crates/sceneworks-worker/src/flux2_composition_audit.rs");
  for (const name of [
    "the_bundle_routes_flux2_dev_to_the_provider_the_calibration_measured",
    "the_bundle_keeps_flux2_devs_memory_strategy_route_intact",
    "the_audited_composition_is_the_full_cuda_bundle",
  ]) {
    assert.match(
      composition,
      new RegExp(`#\\[test\\]\\s*\\n\\s*fn ${name}\\(`),
      `the composition audit must still RUN ${name}, not merely mention it`,
    );
  }
  const workerLib = await source("crates/sceneworks-worker/src/lib.rs");
  assert.match(
    workerLib,
    /#\[cfg\(all\(test, not\(target_os = "macos"\), feature = "backend-candle"\)\)\]\s*\nmod flux2_composition_audit;/,
    "an undeclared module is compiled by no lane, so the composition check would vanish in silence",
  );
});

// sc-18808 review. The MLX LTX arm hand-copies four `limits.*` values out of
// config/manifests/builtin.models.jsonc — LTX_RESOLUTIONS, LTX_DURATIONS_SECONDS, LTX_FPS and
// LTX_DIMENSION_MULTIPLE — and DERIVES its accepted frame envelope `[97, 449]` from the durations x
// fps cross product. The derivation was the point: the envelope is not written down. But nothing
// bound the four inputs, so a limits edit in the manifest would leave the derived envelope stale and
// silently change what a real-weight video capture admits, with no test anywhere going red.
//
// The adapter crate deliberately carries two dependencies (serde_json + sha2) and cannot take
// sceneworks-core's bundled SQLite / image codecs just to reach `strip_jsonc_comments`, so the
// binding lives here, where the manifest reader already exists and `npm run check` runs it on every
// PR. Every extraction below asserts it MATCHED before it compares — a renamed constant must red,
// not silently pass with nothing to check.
test("the MLX LTX arm's manifest constants match the shipped ltx_2_3 limits", async () => {
  const manifest = JSON.parse(
    stripJsoncComments(await source("config/manifests/builtin.models.jsonc")),
  );
  const adapter = await source("crates/sceneworks-memory-adapter/src/bin/mlx.rs");

  const rustConst = (name) => {
    const start = adapter.indexOf(`const ${name}`);
    assert.ok(start >= 0, `${name} must still exist in the MLX adapter`);
    const equals = adapter.indexOf("=", start);
    const end = adapter.indexOf(";", equals);
    assert.ok(equals > start && end > equals, `${name} must be a simple const initializer`);
    return adapter.slice(equals + 1, end).trim();
  };
  // `[(768, 512), ...]` is a Rust tuple array; `(`/`)` -> `[`/`]` makes it JSON.
  const rustTuples = (name) => JSON.parse(rustConst(name).replaceAll("(", "[").replaceAll(")", "]"));

  const providerId = JSON.parse(rustConst("LTX_PROVIDER"));
  assert.equal(providerId, "ltx_2_3");
  const model = manifest.models.find((entry) => entry.id === providerId);
  assert.ok(model, `builtin.models.jsonc must still declare ${providerId}`);
  const limits = model.limits;
  assert.ok(limits, `${providerId} must still declare a limits block`);

  assert.deepEqual(
    rustTuples("LTX_RESOLUTIONS"),
    limits.resolutions.map((size) => size.split("x").map(Number)),
    "LTX_RESOLUTIONS is a verbatim copy of limits.resolutions",
  );
  assert.deepEqual(
    JSON.parse(rustConst("LTX_DURATIONS_SECONDS")),
    limits.durations,
    "LTX_DURATIONS_SECONDS is a verbatim copy of limits.durations",
  );
  assert.deepEqual(
    JSON.parse(rustConst("LTX_FPS")),
    limits.fps,
    "LTX_FPS is a verbatim copy of limits.fps",
  );
  assert.equal(
    Number(rustConst("LTX_DIMENSION_MULTIPLE")),
    limits.requiresDimensionsMultipleOf,
    "LTX_DIMENSION_MULTIPLE is limits.requiresDimensionsMultipleOf",
  );

  // ... and the envelope those four inputs feed. Recomputed here from the MANIFEST through the same
  // nearest-8k+1 ladder `sceneworks_core::video_request::ltx_frame_count` implements, so a limits
  // edit is caught as a changed ENVELOPE — what the arm actually admits — and not merely as a
  // changed constant.
  const snap = (raw) => {
    const frames = Math.max(raw, 9);
    const lower = frames - ((frames - 1) % 8);
    const upper = lower + 8;
    if (lower < 9) return upper;
    return frames - lower <= upper - frames ? lower : upper;
  };
  const reachable = limits.durations.flatMap((duration) =>
    limits.fps.map((fps) => snap(duration * fps)),
  );
  const productWidth = Number(rustConst("LTX_PRODUCT_CANARY_WIDTH"));
  const productHeight = Number(rustConst("LTX_PRODUCT_CANARY_HEIGHT"));
  const productFrames = Number(rustConst("LTX_PRODUCT_CANARY_FRAMES"));
  const productFps = Number(rustConst("LTX_PRODUCT_CANARY_FPS"));
  const productResolution = `${productWidth}x${productHeight}`;
  assert.equal(
    model.defaults.resolution,
    productResolution,
    "the product-envelope canary must use the shipped default resolution",
  );
  assert.ok(
    limits.resolutions.includes(productResolution),
    "the product-envelope resolution must remain inside the shipped envelope",
  );
  const defaultDownloads = model.downloads.filter((download) => download.default === true);
  assert.equal(defaultDownloads.length, 1, "LTX must retain exactly one default artifact variant");
  assert.equal(
    defaultDownloads[0].variant,
    "q4",
    "the product-envelope canary must measure the default product artifact variant",
  );
  const productDuration = (productFrames - 1) / productFps;
  assert.ok(Number.isInteger(productDuration), "the product tuple must span an exact duration");
  assert.ok(limits.durations.includes(productDuration), "the product duration must remain shipped");
  assert.ok(limits.fps.includes(productFps), "the product FPS must remain shipped");
  assert.equal(
    snap(productDuration * productFps),
    productFrames,
    "the shipped duration/FPS snap must resolve to the exact product frame count",
  );
  assert.match(
    adapter,
    /const LTX_FRAME_ENVELOPE: \(u32, u32\) = ltx_frame_envelope\(\);/,
    "the frame envelope must stay DERIVED from the declared arrays, not written down",
  );
  const envelopeTestAt = adapter.indexOf(
    "fn ltx_frame_envelope_is_derived_from_the_declared_durations_and_fps(",
  );
  assert.ok(envelopeTestAt >= 0, "the envelope derivation test must still exist");
  assert.match(
    adapter.slice(envelopeTestAt, envelopeTestAt + 600),
    new RegExp(
      `assert_eq!\\(LTX_FRAME_ENVELOPE, \\(${Math.min(...reachable)}, ${Math.max(...reachable)}\\)\\)`,
    ),
    "the pinned envelope must equal the one the shipped limits actually reach",
  );
});

// sc-19057. The Candle twin of the gate above, for the first candle VIDEO arm. Same reasoning: the
// arm hand-copies the `wan_2_2` `limits` block and DERIVES its accepted frame envelope `[93, 189]`
// from durations x the capturable cadence, so an unbound manifest edit would silently move what a
// multi-hour CUDA capture admits. It also carries two checks the LTX gate has no analogue for:
//
//   * `wan_2_2` deliberately declares NO `requiresDimensionsMultipleOf` (core's default floor of 32
//     already equals candle's `SIZE_MULTIPLE`), so this asserts the ELISION. Re-declaring a
//     different stride there reds instead of silently widening the arm.
//   * the catalog model id and the ENGINE id genuinely differ here, and the video-curve fit resolves
//     `modelFamily` from the catalog id — so both constants are bound, to the manifest and to the
//     engine respectively.
test("the Candle Wan arm's manifest constants match the shipped wan_2_2 limits", async () => {
  const manifest = JSON.parse(
    stripJsoncComments(await source("config/manifests/builtin.models.jsonc")),
  );
  const adapter = await source("crates/sceneworks-memory-adapter/src/bin/candle.rs");

  const rustConst = (name) => {
    const start = adapter.indexOf(`const ${name}`);
    assert.ok(start >= 0, `${name} must still exist in the Candle adapter`);
    const equals = adapter.indexOf("=", start);
    const end = adapter.indexOf(";", equals);
    assert.ok(equals > start && end > equals, `${name} must be a simple const initializer`);
    return adapter.slice(equals + 1, end).trim();
  };
  const rustTuples = (name) => JSON.parse(rustConst(name).replaceAll("(", "[").replaceAll(")", "]"));

  const catalogId = JSON.parse(rustConst("WAN_MANIFEST_MODEL_ID"));
  assert.equal(catalogId, "wan_2_2");
  assert.equal(
    JSON.parse(rustConst("WAN_PROVIDER")),
    "wan2_2_ti2v_5b",
    "WAN_PROVIDER is the candle-gen-wan ENGINE id, not the catalog id",
  );
  const model = manifest.models.find((entry) => entry.id === catalogId);
  assert.ok(model, `builtin.models.jsonc must still declare ${catalogId}`);
  assert.equal(model.type, "video", `${catalogId} must remain a video entry`);
  const limits = model.limits;
  assert.ok(limits, `${catalogId} must still declare a limits block`);

  assert.deepEqual(
    rustTuples("WAN_RESOLUTIONS"),
    limits.resolutions.map((size) => size.split("x").map(Number)),
    "WAN_RESOLUTIONS is a verbatim copy of limits.resolutions",
  );
  assert.deepEqual(
    JSON.parse(rustConst("WAN_DURATIONS_SECONDS")),
    limits.durations,
    "WAN_DURATIONS_SECONDS is a verbatim copy of limits.durations",
  );
  assert.deepEqual(
    JSON.parse(rustConst("WAN_FPS")),
    limits.fps,
    "WAN_FPS is a verbatim copy of limits.fps",
  );
  assert.equal(
    Number(rustConst("WAN_MAX_PIXELS").replaceAll("_", "")),
    limits.maxPixels,
    "WAN_MAX_PIXELS is limits.maxPixels",
  );
  assert.equal(
    limits.requiresDimensionsMultipleOf,
    undefined,
    "wan_2_2 deliberately elides the stride; core's default floor of 32 is the binding, so declaring one here means WAN_DIMENSION_MULTIPLE must be re-derived from it",
  );
  assert.equal(
    Number(rustConst("WAN_DIMENSION_MULTIPLE")),
    32,
    "WAN_DIMENSION_MULTIPLE is core's default spatial floor, which equals candle SIZE_MULTIPLE",
  );

  // The one cadence the calibrated route can execute must still be a shipped one, and must still be
  // this model's own default — otherwise the capture measures a geometry the product never renders.
  const calibratedFps = Number(rustConst("WAN_CALIBRATED_FPS"));
  assert.ok(limits.fps.includes(calibratedFps), "the calibrated cadence must remain shipped");
  assert.equal(model.defaults.fps, calibratedFps, "the calibrated cadence is the shipped default");
  assert.ok(
    limits.resolutions.includes(model.defaults.resolution),
    "the shipped default resolution must remain inside the declared envelope",
  );

  // ... and the envelope those inputs feed, recomputed here from the MANIFEST through the same
  // floor-to-4k+1 ladder `sceneworks_core::video_request::wan_frame_count` implements, so a limits
  // edit is caught as a changed ENVELOPE — what the arm actually admits — not merely as a changed
  // constant.
  //
  // sc-19057 review: the cross product is taken over the CAPTURABLE cadence, not over all of
  // `limits.fps`. `WanMemoryRequestScope::validate_request` refuses every cadence but the default,
  // so the 16 fps rungs (61, 77, 109, 125) are geometries no admissible plan row can produce, and
  // an envelope spanning them admitted `832x480 f61 fps24` — the exact drift the fps refusal exists
  // to prevent.
  const snap = (raw) => Math.max(Math.max(raw, 1) - ((Math.max(raw, 1) - 1) % 4), 5);
  assert.equal(
    rustConst("WAN_CAPTURABLE_FPS"),
    "[WAN_CALIBRATED_FPS]",
    "WAN_CAPTURABLE_FPS is the capturable subset of limits.fps, and today that is the calibrated cadence alone",
  );
  const reachable = limits.durations.map((duration) => snap(duration * calibratedFps));
  assert.match(
    adapter,
    /const WAN_FRAME_ENVELOPE: \(u32, u32\) = wan_frame_envelope\(\);/,
    "the frame envelope must stay DERIVED from the declared arrays, not written down",
  );
  const ladderTestAt = adapter.indexOf(
    "fn wan_frame_ladder_port_matches_the_transcribed_shipped_ladder(",
  );
  assert.ok(ladderTestAt >= 0, "the Wan ladder port test must still exist");
  assert.match(
    adapter.slice(ladderTestAt, ladderTestAt + 2400),
    new RegExp(
      `assert_eq!\\(WAN_FRAME_ENVELOPE, \\(${Math.min(...reachable)}, ${Math.max(...reachable)}\\)\\)`,
    ),
    "the pinned envelope must equal the one the capturable cadence actually reaches",
  );
  // The narrowing is only meaningful if it actually moved the floor: the shortest clip the refused
  // cadence reaches must now sit BELOW the envelope. (The envelope is a closed interval, not a set,
  // so counts that fall between capturable rungs stay admissible — the plan-row ladder check further
  // down is what binds the committed rows to the exact rungs.)
  const refusedRungs = limits.durations.flatMap((duration) =>
    limits.fps.filter((fps) => fps !== calibratedFps).map((fps) => snap(duration * fps)),
  );
  if (refusedRungs.length > 0) {
    assert.ok(
      Math.min(...refusedRungs) < Math.min(...reachable),
      "the capture envelope floor must exclude the shortest clip only the refused cadence reaches",
    );
  }

  // The capture plan measures the DEFAULT candle artifact variant, at a geometry and cadence the
  // shipped limits actually reach.
  const plan = JSON.parse(
    await source("docs/calibration/sc-19057/wan-candle-video-capture-plan.json"),
  );
  const candleDefaults = model.downloads.filter(
    (download) => download.default === true && (download.platforms ?? []).includes("windows"),
  );
  assert.equal(candleDefaults.length, 1, "Wan must retain exactly one default candle variant");
  for (const provider of plan.providers) {
    assert.equal(provider.backend, "candle");
    assert.equal(provider.target.provider, "wan2_2_ti2v_5b");
    assert.equal(provider.target.modelId, catalogId);
    assert.equal(
      provider.target.tier,
      candleDefaults[0].variant,
      "the sc-19057 plan measures the default candle artifact variant",
    );
    const { width, height, frames } = provider.target.geometry;
    assert.ok(
      limits.resolutions.includes(`${width}x${height}`),
      `${provider.name} renders an undeclared resolution`,
    );
    assert.ok(reachable.includes(frames), `${provider.name} renders an unreachable frame count`);
    assert.match(
      provider.fixture,
      new RegExp(`^wan2-2-ti2v-5b-candle-${provider.target.tier}-${width}x${height}-f${frames}-fps${calibratedFps}-seed\\d+$`),
      `${provider.name} fixture must bind its tier, geometry and cadence`,
    );
  }
  // Three coefficients need three independent geometry directions: with one pixel count the cross
  // form is singular and the fit reports an unusable slice hours after the capture burned.
  assert.ok(
    new Set(plan.providers.map((provider) => provider.target.geometry.width * provider.target.geometry.height)).size >= 2,
    "the sweep must span at least two pixel counts",
  );
  assert.ok(
    new Set(plan.providers.map((provider) => provider.target.geometry.frames)).size >= 2,
    "the sweep must span at least two frame counts",
  );
  assert.ok(
    plan.providers.filter((provider) => provider._role === "fit").length >= 3,
    "the cross form needs at least three fit points",
  );
  assert.ok(
    plan.providers.some((provider) => (provider._role ?? "").startsWith("held_out")),
    "fit-ltx-temporal-form.mjs discards a slice with no held-out residual",
  );
});

// sc-19057: the first candle video lane needs a closure-table entry before a capture can carry a
// currency term — `memory-calibration-harness.mjs` derives the digest eagerly and fails closed with
// `lane "candle:wan2_2_ti2v_5b" has no entry` BEFORE the capture burns.
test("the candle Wan video lane is declared in the inference closure table", async () => {
  const closures = JSON.parse(await source("config/inference-provider-closures.json"));
  const lane = closures.providers["candle:wan2_2_ti2v_5b"];
  assert.ok(lane, "candle:wan2_2_ti2v_5b must be declared before its capture can be stamped");
  assert.equal(lane.crate, "crates/media/candle-gen/candle-gen-wan");
  assert.match(lane.digest, /^[0-9a-f]{64}$/, "the digest must be a derived sha256, never hand-written");
  assert.ok(
    lane.closureCrates.includes("crates/media/candle-gen/candle-gen-wan"),
    "the closure must contain the crate that owns the memory-strategy contract",
  );
});

// sc-19057 review. `crates/sceneworks-memory-adapter/src/lib.rs` pins the FULL refusal sentences of
// the shared video-geometry guard with `assert_eq!`, on every host — but it builds each envelope
// from a HAND-TRANSCRIBED copy of the arm's `protocol::VideoGeometryEnvelope` literal. A hand copy
// cannot detect drift in the thing it copied: reword `temporal_rationale` in `bin/mlx.rs`, relabel
// the arm, or move its frame envelope, and those tests stay green while the sentence an operator
// reads when a multi-hour capture is refused quietly changes. That mattered because `bin/mlx.rs`
// compiles on exactly one host (macOS + `mlx`) and `bin/candle.rs` on exactly one other (off-macOS
// + `candle` + a CUDA toolchain) — the lib tests were advertised as the always-run cover and were
// not covering the arms at all.
//
// This binds the copy to the original from the outside: parse each arm's struct literal, resolve
// the constants it names, and require the lib copy to be field-for-field identical. `npm run check`
// runs it on any host, so a reworded rationale reds on the PR rather than on one lane or nowhere.
test("the shared video-geometry envelopes are transcribed verbatim into the always-compiled lib tests", async () => {
  const [mlx, candle, lib] = await Promise.all([
    source("crates/sceneworks-memory-adapter/src/bin/mlx.rs"),
    source("crates/sceneworks-memory-adapter/src/bin/candle.rs"),
    source("crates/sceneworks-memory-adapter/src/lib.rs"),
  ]);

  // `//` never appears inside any of these literals' strings, so a per-line strip is exact and keeps
  // a doc-comment comma from being mistaken for a field separator.
  const withoutLineComments = (text) =>
    text
      .split("\n")
      .map((line) => line.replace(/\/\/.*$/, ""))
      .join("\n");

  /** The balanced `VideoGeometryEnvelope { .. }` literal returned by `fn <name>`, as field text. */
  const envelopeFields = (text, fnName, what) => {
    const fnAt = text.indexOf(`fn ${fnName}(`);
    assert.ok(fnAt >= 0, `${what}: fn ${fnName} must still exist`);
    const literalAt = text.indexOf("VideoGeometryEnvelope {", fnAt);
    assert.ok(literalAt > fnAt, `${what}: ${fnName} must still return a struct literal`);
    const open = text.indexOf("{", literalAt);
    let depth = 0;
    let close = -1;
    for (let index = open; index < text.length; index += 1) {
      if (text[index] === "{") depth += 1;
      else if (text[index] === "}" && (depth -= 1) === 0) {
        close = index;
        break;
      }
    }
    assert.ok(close > open, `${what}: ${fnName}'s struct literal is unbalanced`);

    const body = withoutLineComments(text.slice(open + 1, close));
    const fields = {};
    let start = 0;
    let nesting = 0;
    let inString = false;
    let escaped = false;
    const push = (chunk) => {
      const trimmed = chunk.trim();
      if (!trimmed) return;
      const colon = trimmed.indexOf(":");
      assert.ok(colon > 0, `${what}: ${fnName} has a non-field entry ${JSON.stringify(trimmed)}`);
      fields[trimmed.slice(0, colon).trim()] = trimmed.slice(colon + 1).trim();
    };
    for (let index = 0; index < body.length; index += 1) {
      const character = body[index];
      if (inString) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === '"') inString = false;
        continue;
      }
      if (character === '"') inString = true;
      else if ("([{".includes(character)) nesting += 1;
      else if (")]}".includes(character)) nesting -= 1;
      else if (character === "," && nesting === 0) {
        push(body.slice(start, index));
        start = index + 1;
      }
    }
    push(body.slice(start));
    return fields;
  };

  /** The initializer text of `const <name>` in the same file. */
  const constInitializer = (text, name, what) => {
    const at = text.indexOf(`const ${name}`);
    assert.ok(at >= 0, `${what}: const ${name} must still exist`);
    const equals = text.indexOf("=", at);
    const end = text.indexOf(";", equals);
    assert.ok(equals > at && end > equals, `${what}: const ${name} must be a simple initializer`);
    return text.slice(equals + 1, end).trim();
  };

  /** Resolve one field's Rust source text to a comparable JSON value. */
  const resolve = (raw, text, what, depth = 0) => {
    assert.ok(depth < 8, `${what}: ${raw} did not resolve to a literal`);
    let value = raw.trim();
    if (value.startsWith("&")) value = value.slice(1).trim();
    if (value === "None") return null;
    const some = /^Some\(([\s\S]*)\)$/.exec(value);
    if (some) return resolve(some[1], text, what, depth + 1);
    // `\` + newline is Rust's line continuation and carries no characters into the string.
    if (value.startsWith('"')) return JSON.parse(value.replace(/\\\s*\n\s*/g, ""));
    // `(a, b)` and `[(a, b), ...]` become JSON by turning tuples into arrays.
    if (value.startsWith("(")) value = `[${value.slice(1, -1)}]`;
    if (value.startsWith("[")) {
      return JSON.parse(value.replaceAll("(", "[").replaceAll(")", "]").replaceAll("_", ""));
    }
    if (/^[0-9]/.test(value)) return Number(value.replaceAll("_", ""));
    return resolve(constInitializer(text, value, what), text, what, depth + 1);
  };

  const arms = [
    {
      what: "MLX LTX",
      arm: mlx,
      armFn: "ltx_video_envelope",
      libFn: "ltx_envelope",
      envelopeConst: "LTX_FRAME_ENVELOPE",
      // The pinned pair these two tests already bind to builtin.models.jsonc, above.
      pinnedIn: "fn ltx_frame_envelope_is_derived_from_the_declared_durations_and_fps(",
    },
    {
      what: "Candle Wan",
      arm: candle,
      armFn: "wan_video_envelope",
      libFn: "wan_envelope",
      envelopeConst: "WAN_FRAME_ENVELOPE",
      pinnedIn: "fn wan_frame_ladder_port_matches_the_transcribed_shipped_ladder(",
    },
  ];

  for (const { what, arm, armFn, libFn, envelopeConst, pinnedIn } of arms) {
    const armFields = envelopeFields(arm, armFn, what);
    const libFields = envelopeFields(lib, libFn, `${what} lib copy`);
    assert.deepEqual(
      Object.keys(libFields).sort(),
      Object.keys(armFields).sort(),
      `${what}: the lib copy must declare exactly the arm's fields`,
    );

    for (const [field, raw] of Object.entries(armFields)) {
      if (field === "frame_envelope") continue;
      assert.deepEqual(
        resolve(libFields[field], lib, `${what} lib copy`),
        resolve(raw, arm, what),
        `${what}: lib's transcribed ${field} must equal ${armFn}'s`,
      );
    }

    // `frame_envelope` is a derived const (`ltx_frame_envelope()` / `wan_frame_envelope()`), so it
    // has no literal to resolve. Bind the copy to the pair the arm's own derivation test pins —
    // which the two manifest gates above already bind to the shipped `limits`.
    assert.equal(armFields.frame_envelope, envelopeConst, `${what}: the arm must use ${envelopeConst}`);
    const pinnedAt = arm.indexOf(pinnedIn);
    assert.ok(pinnedAt >= 0, `${what}: the envelope derivation test must still exist`);
    const pinned = new RegExp(`assert_eq!\\(${envelopeConst}, \\((\\d+), (\\d+)\\)\\)`).exec(
      arm.slice(pinnedAt, pinnedAt + 2400),
    );
    assert.ok(pinned, `${what}: the derivation test must pin ${envelopeConst} to an explicit pair`);
    assert.deepEqual(
      resolve(libFields.frame_envelope, lib, `${what} lib copy`),
      [Number(pinned[1]), Number(pinned[2])],
      `${what}: lib's transcribed frame_envelope must equal the pinned derived envelope`,
    );
  }

  // The transcriptions are only worth binding because something asserts the full sentences off them.
  for (const pinned of [
    "fn the_video_geometry_guard_reproduces_the_pinned_ltx_wording_on_every_axis(",
    "fn the_video_geometry_guard_reports_the_candle_wan_envelope_in_its_own_words(",
  ]) {
    assert.ok(lib.includes(pinned), `the lib must still assert the pinned wording: ${pinned}`);
  }
});

test("the LTX real-weight safety canary cannot relax or masquerade as campaign evidence", async () => {
  const [adapter, runner] = await Promise.all([
    source("crates/sceneworks-memory-adapter/src/bin/mlx.rs"),
    source("scripts/run-ltx-safety-canary.mjs"),
  ]);
  const historical = JSON.parse(await source("docs/generated/ltx-mlx-video-sc-18808.json"));
  const costaged = historical.records[0].diagnostics.measurements.find(
    (entry) => entry.name === "costagedGiantsBytes",
  )?.value;
  assert.equal(costaged, 53_347_146_863, "SC-18808 historical co-staged bound changed");
  assert.match(
    adapter,
    /const LTX_CANARY_MAX_FOOTPRINT_BYTES: u64 = 53_347_146_863;/,
    "the conservative canary stop must remain byte-bound to SC-18808 co-staged arithmetic",
  );

  const campaign = adapter.slice(
    adapter.indexOf("fn run_ltx_with_admission("),
    adapter.indexOf("fn campaign_entry_diagnostic("),
  );
  const ordinary = adapter.slice(
    adapter.indexOf("fn run_ltx(request:"),
    adapter.indexOf("fn run(request:"),
  );
  assert.match(
    ordinary,
    /run_ltx_with_admission\(request, LtxRunAdmission::Ordinary, &mut phases\)/,
  );
  assert.ok(campaign.indexOf("refuse_unsafe_ltx_capture(") >= 0);
  assert.ok(
    campaign.indexOf("refuse_unsafe_ltx_capture(") < campaign.indexOf("ltx_load_spec("),
    "the campaign must still refuse before model-path/provider/weights access",
  );
  const campaignRows = Object.values(buildLtxPlans()).flatMap((plan) => plan.providers);
  assert.equal(campaignRows.length, 73, "SC-20430 must explicitly inventory 73 rows");
  const ratified = campaignRows.filter((row) => row._role === "bounded_carrier_entry");
  assert.equal(ratified.length, 3, "exactly one matched bounded carrier per tier may be ratified");
  assert.deepEqual(ratified.map((row) => row.target.tier), ["q4", "q8", "bf16"]);
  const crossLayer = {
    q8: {
      logicalCaseId: "implan-d47640caa0c469f2ee13",
      identity: "sc-20430-q8-768x512-f121-fps30-bounded-192-64-authoritative-v1",
      inventorySha256: "bb0bb7577157a158ca39494837d64cb36ded0380ca7ee0c930fea7311f22a247",
    },
    bf16: {
      logicalCaseId: "implan-b3926164bf6bfbee98e1",
      identity: "sc-20430-bf16-768x512-f121-fps30-bounded-192-64-authoritative-v1",
      inventorySha256: "006caeaa9a8638b337cdf5a8622ce8535380b18ebaf90b36c3e2d5d15354f2a8",
    },
  };
  for (const row of ratified.filter(({ target }) => target.tier !== "q4")) {
    const exact = crossLayer[row.target.tier];
    for (const [label, value] of [
      ["provider", row.name], ["fixture", row.fixture],
      ["logical case", exact.logicalCaseId], ["private identity", exact.identity],
      ["inventory SHA", exact.inventorySha256],
    ]) {
      assert.ok(runner.includes(value), `${row.target.tier} runner ${label} changed`);
      if (label !== "provider") {
        assert.ok(adapter.includes(value), `${row.target.tier} adapter ${label} changed`);
      }
    }
  }
  assert.match(runner,
    /process\.argv\[2\] === "--bounded-selector-report"[\s\S]*boundedSelectorReportController/,
    "SC-20430 matched phase anchors must remain executable as a CPU-only report path");
  const originalRows = campaignRows.filter((row) => row._role !== "bounded_carrier_entry");
  assert.equal(originalRows.length, 70, "all original campaign rows must remain present");
  for (const row of originalRows) {
    assert.ok(
      ["incident_forbidden", "arithmetic_unmeasurable", "safety_refused_open"]
        .includes(row._measurementSafety.disposition),
      `${row.name} must retain an explicit refusal disposition`,
    );
  }
  assert.doesNotMatch(
    ordinary,
    /LTX_(?:CANARY|PRODUCT_CANARY)_FIXTURE|diagnostic_(?:product_envelope_)?canary_complete/,
  );
  const campaignEntry = adapter.slice(
    adapter.indexOf("fn run_ltx_campaign_entry("),
    adapter.indexOf("fn run_ltx(request:"),
  );
  for (const required of [
    "prevalidate_ltx_campaign_entry(request)?",
    "consume_ltx_canary_watchdog_attestation(request)?",
    "LtxCanaryLimits::install()?",
    "LtxRunAdmission::CampaignEntry",
    "validate_ltx_campaign_entry_fragment(&fragment)?",
    "validate_ltx_canary_cleanup(",
    '"_campaignEntry"',
    "watchdog_lease.complete()?",
    'watchdog_lease.mark("common_load")?',
    'watchdog_lease.mark("cleanup")?',
  ]) assert.ok(campaignEntry.includes(required), `campaign entry must retain ${required}`);
  for (const phase of [
    "primary_conditioning", "primary_denoise", "primary_decode",
  ]) assert.ok(campaign.includes(`phase_sink.mark("${phase}")`),
    `campaign execution must report ${phase}`);
  for (const phase of [
    "lifecycle_warm_repeat", "lifecycle_cancel", "lifecycle_cancel_recovery",
    "lifecycle_error", "lifecycle_error_recovery",
  ]) assert.ok(adapter.includes(`phase_sink.mark("${phase}")`),
    `campaign lifecycle must report ${phase}`);
  assert.ok(
    campaignEntry.indexOf("prevalidate_ltx_campaign_entry(request)?")
      < campaignEntry.indexOf("consume_ltx_canary_watchdog_attestation(request)?"),
    "the exact campaign row must be validated before the watchdog releases model allocation",
  );
  const boundedCarrier = adapter.slice(
    adapter.indexOf("fn run_ltx_bounded_carrier_proof("),
    adapter.indexOf("fn run_ltx_campaign_entry("),
  );
  for (const required of [
    "prevalidate_ltx_bounded_carrier_proof(request)?",
    "start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?",
    'watchdog_lease.mark("common_load")?',
    'watchdog_lease.mark("primary_conditioning")?',
    'watchdog_lease.mark("primary_denoise")',
    'watchdog_lease.mark("primary_decode")',
    'watchdog_lease.mark("cleanup")?',
    "validate_ltx_bounded_carrier_generation_request(&generation_request)?",
    "scoped_generate(",
    "spatial_decode_tile_count != 24",
    "validate_ltx_canary_cleanup(",
    '"diagnosticOnly": true',
    '"promotable": false',
    '"ingestible": false',
    '"seed": LTX_SEED',
  ]) assert.ok(boundedCarrier.includes(required), `bounded carrier must retain ${required}`);
  assert.doesNotMatch(boundedCarrier, /verify_ltx_lifecycle|LtxRunAdmission::CampaignEntry/,
    "SC-20254 must execute one provider request scope, not the multi-render campaign lifecycle");
  assert.equal((boundedCarrier.match(/scoped_generate\(/g) ?? []).length, 1,
    "SC-20254 must contain exactly one full-A/V render call");
  assert.match(adapter, /LTX_BOUNDED_CARRIER_ACTION => run_ltx_bounded_carrier_proof\(&request\)/);
  const boundedCampaign = adapter.slice(
    adapter.indexOf("fn run_ltx_bounded_campaign_entry("),
    adapter.indexOf("fn run_ltx(request:"),
  );
  for (const required of [
    "prevalidate_ltx_bounded_campaign_entry(request)?",
    "consume_ltx_canary_watchdog_attestation(request)?",
    "start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?",
    "LtxRunAdmission::BoundedCampaignEntry",
    "validate_ltx_bounded_campaign_fragment(&fragment)?",
    "validate_ltx_canary_cleanup(",
    '"_boundedCampaignEntry"',
    'watchdog_lease.mark("common_load")?',
    'watchdog_lease.mark("cleanup")?',
  ]) assert.ok(boundedCampaign.includes(required), `bounded campaign must retain ${required}`);
  assert.match(adapter,
    /LTX_BOUNDED_CAMPAIGN_ACTION => run_ltx_bounded_campaign_entry\(&request\)/);
  const attestationIndex = boundedCampaign.indexOf(
    "consume_ltx_canary_watchdog_attestation(request)?",
  );
  const limitsIndex = boundedCampaign.indexOf("LtxCanaryLimits::install()?");
  const modelResolutionIndex = boundedCampaign.indexOf("run_ltx_with_admission(");
  assert.ok(attestationIndex >= 0 && limitsIndex >= 0 && modelResolutionIndex >= 0,
    "SC-20318 safety ordering anchors must all remain present");
  assert.ok(
    attestationIndex < limitsIndex,
    "SC-20318 must reject a mismatched watchdog phase profile before allocator/model work",
  );
  assert.ok(
    attestationIndex < modelResolutionIndex,
    "SC-20318 must reject a mismatched watchdog phase profile before model resolution",
  );

  const canary = adapter.slice(
    adapter.indexOf("fn validate_ltx_canary_plan_for("),
    adapter.indexOf("/// The `mlx:ltx_2_3` SC-18946 arm"),
  );
  for (const required of [
    "_diagnosticOnly",
    'Some("fixture")',
    "profile.width()",
    "profile.height()",
    "profile.frames()",
    "LTX_CANARY_TILE_EDGE",
    "LTX_CANARY_OVERLAP",
    "profile.video_mode_identity()",
    '"status": profile.completion_status()',
    '"canaryIdentity": profile.identity()',
    '"promotable": false',
    '"ingestible": false',
    'strategy["spatialDecodeTiles"]',
    '"preProviderActiveBytes": pre_provider.active',
    '"preProviderCacheBytes": pre_provider.cache',
    '"identity": LTX_CANARY_ONES_CACHE_IDENTITY',
    '"bytes": expected_persistent_active',
  ]) assert.ok(canary.includes(required), `canary must retain ${required}`);
  const limitsInstalled = canary.indexOf("let limits = LtxCanaryLimits::install()?");
  const baselineCaptured = canary.indexOf("let pre_provider = AllocatorState::capture_current()");
  const providerLoadSpec = canary.indexOf("ltx_load_spec(request, \"q4\", &selection)?");
  assert.ok(limitsInstalled >= 0 && limitsInstalled < baselineCaptured,
    "the allocator baseline must be captured after canary limits are installed");
  assert.ok(baselineCaptured < providerLoadSpec,
    "the allocator baseline must be captured before provider/model resolution");
  assert.match(canary, /clear_cache\(\);\n\s*let pre_provider = AllocatorState::capture_current\(\);/,
    "the pre-provider baseline must exclude reclaimable allocator cache");
  assert.match(canary, /validate_ltx_canary_pre_provider\(pre_provider\)\?;/);
  assert.match(
    canary,
    /validate_ltx_canary_cleanup\(pre_provider, cleanup, expected_persistent_active\)\?;/,
    "successful canary output must require the exact named persistent allocation",
  );
  assert.doesNotMatch(campaign, /validate_ltx_canary_(?:pre_provider|cleanup)/,
    "production generation must not use the diagnostic canary residue exception");

  const profiles = adapter.slice(
    adapter.indexOf("enum LtxCanaryProfile"),
    adapter.indexOf("/// LTX's video VAE"),
  );
  for (const required of [
    'Self::Safety => Some("no_audio")',
    "Self::ProductEnvelope => None",
    'Self::Safety => "diagnostic_canary_complete"',
    'Self::ProductEnvelope => "diagnostic_product_envelope_canary_complete"',
    'Self::Safety => "a sunlit pine branch, static camera"',
    'Self::ProductEnvelope => "sc-20169-product-envelope"',
  ]) assert.ok(profiles.includes(required), `diagnostic profiles must retain ${required}`);
  const generationRequest = adapter.slice(
    adapter.indexOf("fn ltx_canary_generation_request_for("),
    adapter.indexOf("fn ltx_load_spec("),
  );
  assert.match(
    generationRequest,
    /video_mode: profile\.video_mode\(\)\.map\(str::to_owned\)/,
    "the canary must skip the downstream audio decoder and vocoder",
  );
  const admissionBridge = adapter.slice(
    adapter.indexOf("fn ltx_canary_request_for_provider_admission("),
    adapter.indexOf("fn ltx_load_spec("),
  );
  assert.match(admissionBridge, /request\.video_mode = None;/);
  assert.match(admissionBridge, /request\.video_mode = Some\("no_audio"\.to_owned\(\)\);/);
  assert.match(admissionBridge, /scoped_generate_observed_after_configuration\(/);
  const genericScope = adapter.slice(
    adapter.indexOf("fn scoped_generate_observed("),
    adapter.indexOf("fn scoped_generate_observed_after_configuration("),
  );
  assert.match(genericScope, /None,/,
    "ordinary production generation must not gain the canary-only post-configure override");
  const configuredScope = adapter.slice(
    adapter.indexOf("fn scoped_generate_observed_after_configuration("),
    adapter.indexOf("/// Combine the generator and request-scope terminals"),
  );
  assert.ok(
    configuredScope.indexOf(".configure_request(&mut request)")
      < configuredScope.indexOf("after_configuration(&mut request)"),
    "the canary restores no_audio only after ordinary provider configuration",
  );
  assert.ok(
    configuredScope.indexOf("after_configuration(&mut request)")
      < configuredScope.indexOf("generator.generate(&request"),
    "the exact no_audio override must be restored before generation",
  );
  for (const lifecycle of ["enter_phase", "leave_phase", "scope.finish", "settle_scoped_generation"]) {
    assert.ok(configuredScope.includes(lifecycle), `canary scope must retain ${lifecycle}`);
  }
  const diagnosticRun = adapter.slice(
    adapter.indexOf("fn run_ltx_canary_for("),
    adapter.indexOf("/// The `mlx:ltx_2_3` SC-18946 arm"),
  );
  assert.match(
    diagnosticRun,
    /LtxCanaryProfile::ProductEnvelope => scoped_generate\(/,
    "the product-envelope canary must use the ordinary provider request-scope lifecycle",
  );
  assert.doesNotMatch(
    diagnosticRun,
    /LtxCanaryProfile::ProductEnvelope => scoped_generate_ltx_no_audio_canary/,
  );
  for (const required of [
    "LTX_PRODUCT_CANARY_WIDTH",
    "LTX_PRODUCT_CANARY_HEIGHT",
    "LTX_PRODUCT_CANARY_FRAMES",
    "spatial_tile_count",
    "validate_diagnostic_audio",
  ]) assert.ok(adapter.includes(required), `product-envelope canary must retain ${required}`);

  const limitsLifecycle = adapter.slice(
    adapter.indexOf("impl LtxCanaryLimits"),
    adapter.indexOf("impl LtxDecodePlan"),
  );
  assert.match(limitsLifecycle, /set_wired_limit\(self\.previous_wired\);/);
  assert.match(limitsLifecycle, /set_memory_limit\(self\.previous_memory\);/);
  assert.match(limitsLifecycle, /impl Drop for LtxCanaryLimits[\s\S]*self\.restore\(\);/);
});

test("the SC-20318 provider phase profile is exact across runner, watchdog and adapter", async () => {
  const [runner, watchdog, adapter] = await Promise.all([
    source("scripts/run-ltx-safety-canary.mjs"),
    source("scripts/memory-calibration-watchdog.py"),
    source("crates/sceneworks-memory-adapter/src/bin/mlx.rs"),
  ]);
  const campaignPhases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode",
    "lifecycle_warm_repeat", "lifecycle_cancel", "lifecycle_cancel_recovery",
    "lifecycle_error", "lifecycle_error_recovery", "cleanup",
  ];
  const boundedPhases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode", "cleanup",
  ];
  const quotedValues = (sourceText, expression, label) => {
    const match = sourceText.match(expression);
    assert.ok(match, `${label} must remain a literal exact contract`);
    return [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  };

  assert.deepEqual(quotedValues(
    watchdog,
    /^CAMPAIGN_ENTRY_PROVIDER_PHASES = \(([\s\S]*?)^\)$/m,
    "Python campaign-entry phases",
  ), campaignPhases);
  assert.deepEqual(quotedValues(
    watchdog,
    /^BOUNDED_CARRIER_PROVIDER_PHASES = \(([\s\S]*?)^\)$/m,
    "Python bounded-carrier phases",
  ), boundedPhases);
  assert.deepEqual(quotedValues(
    watchdog,
    /^BOUNDED_CAMPAIGN_ENTRY_PROVIDER_PHASES = \(([\s\S]*?)^\)$/m,
    "Python bounded-campaign-entry phases",
  ), boundedPhases);
  const pythonProfiles = watchdog.slice(
    watchdog.indexOf("PROVIDER_PHASE_PROFILES = {"),
    watchdog.indexOf("\n}\n", watchdog.indexOf("PROVIDER_PHASE_PROFILES = {")) + 2,
  );
  assert.match(pythonProfiles,
    /"campaign-entry": CAMPAIGN_ENTRY_PROVIDER_PHASES/);
  assert.match(pythonProfiles,
    /"bounded-carrier": BOUNDED_CARRIER_PROVIDER_PHASES/);
  assert.match(pythonProfiles,
    /"bounded-campaign-entry": BOUNDED_CAMPAIGN_ENTRY_PROVIDER_PHASES/);

  assert.deepEqual(quotedValues(
    runner,
    /export const PROVIDER_PHASES = Object\.freeze\(\[([\s\S]*?)\]\);/,
    "runner campaign-entry phases",
  ), campaignPhases);
  assert.deepEqual(quotedValues(
    runner,
    /export const BOUNDED_CARRIER_PHASES = Object\.freeze\(\[([\s\S]*?)\]\);/,
    "runner bounded phases",
  ), boundedPhases);
  assert.match(runner,
    /export const BOUNDED_CAMPAIGN_ENTRY_PROFILE = "bounded-campaign-entry";/);
  assert.match(runner,
    /export const BOUNDED_CAMPAIGN_ENTRY_Q8_PROFILE = "bounded-campaign-entry-q8";/);
  assert.match(runner,
    /export const BOUNDED_CAMPAIGN_ENTRY_BF16_PROFILE = "bounded-campaign-entry-bf16";/);
  assert.match(runner,
    /phaseProfile: "bounded-campaign-entry",\n\s*childName: `\$\{boundedSpec\.story\}-\$\{boundedSpec\.tier\}-bounded-campaign-entry`/);

  assert.deepEqual(quotedValues(
    adapter,
    /const LTX_PROVIDER_PHASE_NAMES: \[&str; 10\] = \[([\s\S]*?)\];/,
    "Rust campaign-entry phases",
  ), campaignPhases);
  assert.deepEqual(quotedValues(
    adapter,
    /const LTX_BOUNDED_CARRIER_PHASE_NAMES: \[&str; 5\] = \[([\s\S]*?)\];/,
    "Rust bounded phases",
  ), boundedPhases);
  assert.match(adapter,
    /const LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE: &str = "bounded-campaign-entry";/);
  assert.match(adapter,
    /Some\(LTX_BOUNDED_CAMPAIGN_ACTION\) => Some\(\(\n\s*LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,\n\s*&LTX_BOUNDED_CARRIER_PHASE_NAMES,/);
});

test("the MLX FLUX.2-dev calibration arm is bound to the direct reference-free T2I contract", async () => {
  const adapter = await source("crates/sceneworks-memory-adapter/src/bin/mlx.rs");
  const context = adapter.slice(
    adapter.indexOf("fn flux2_admission_context("),
    adapter.indexOf("fn flux2_complete_sweep("),
  );
  const arm = adapter.slice(
    adapter.indexOf("fn run_flux2_dev("),
    adapter.indexOf("fn validate_z_image_batch("),
  );

  assert.ok(context.length > 0 && arm.length > 0, "FLUX.2-dev adapter seams must exist");
  assert.match(context, /mode: MemoryMode::TextToImage/);
  assert.match(context, /has_reference: false/);
  assert.match(context, /reference_count: 0/);
  assert.doesNotMatch(context, /MemoryMode::Edit|reference_count: 2/);

  assert.match(arm, /memory_strategy_contract\(FLUX2_PROVIDER, &spec\)/);
  assert.match(arm, /registered_dev_t2i_safety_check\(/);
  assert.match(arm, /generator\s*\.memory_strategy_contract\(\)/);
  assert.match(arm, /loaded_contract != &contract/);
  assert.doesNotMatch(arm, /registered_dev_safety_check|FLUX2_CONTRACT_PROVIDER/);
});

// =====================================================================================
// sc-18921 — macos-mlx.yml's fatal guards, pinned as fatal.
//
// sc-18691 closed this on the CANDLE lane by counting PowerShell `throw`s per step
// ("every failure mode a weights-only dispatch can hit is still fatal", above). The MLX
// lane had the identical exposure in bash form and nothing closed it: macos-mlx.yml
// carried 18 `exit 1` guards and this file contained zero occurrences of `exit 1`, so
// downgrading ANY of them to a bare `echo` left the whole suite green. That lane is the
// sole producer of config/engine-capabilities/capabilities.mlx.json and of every MLX
// five-rung / memory-calibration capture, so a silently non-fatal guard there does not
// merely miss a break — it publishes a wrong measurement as evidence.
//
// WHY THIS IS NOT A HAND-COUNTED COPY OF THE CANDLE TEST. That one carries per-step
// literals (13, 0, 2, 2, 4) beside the predicate they describe, which is the sc-18932
// defect shape: a literal next to a changed predicate is a new false green. Here NOTHING
// is counted by hand. The lane's own text supplies both sides:
//
//   * fatality  — `exit 1` as a whole statement;
//   * the guard — a FAILURE DIAGNOSTIC: an `echo` that writes to stderr or emits
//                 `::error::`. A downgrade removes the exit and LEAVES the diagnostic,
//                 so "every diagnostic is immediately followed by `exit 1`" goes red on
//                 the downgrade without any number being maintained anywhere.
//
// The enumeration below is therefore not a count. It is the answer to "WHICH failure",
// one row per guard, and it is cross-checked against the file scan in both directions:
// a guard deleted outright (diagnostic AND exit together, which the equality above
// cannot see) drops out of the table lookup, and a guard added anywhere on the lane
// leaves an `exit 1` no row claims.
// =====================================================================================

const MLX_LANE = ".github/workflows/macos-mlx.yml";

// THE lane-wide fatality predicate. A whole statement, so `exit 1` inside a quoted
// message or a comment cannot satisfy it.
const MLX_FATAL_EXIT = "exit 1";

// THE lane-wide guard predicate, and the one that makes a downgrade visible: a downgrade
// deletes the exit and keeps the message. Deliberately narrow — `::warning::` and
// `::notice::` are NOT diagnostics, so a step that genuinely wants to report without
// failing has a spelling available that this contract does not claim.
function isMlxFailureDiagnostic(statement) {
  return /^echo\b/.test(statement) && (/>&2$/.test(statement) || /::error::/.test(statement));
}

// A job's steps, each reduced to LOGICAL statements: comment lines dropped (both YAML and
// bash comment with `#`, and this lane's prose quotes its own guards), backslash
// continuations joined (the two `::error::` messages span four and five lines), blanks
// dropped, indentation normalised.
//
// Scoped to ONE JOB rather than reusing the file-wide `stepBody()` above, because
// macos-mlx.yml has two jobs and "Fetch the pinned inference release" appears in both —
// a file-wide lookup by name silently resolves to whichever comes first.
function mlxJobSteps(workflow, job) {
  const start = workflow.indexOf(`\n  ${job}:\n`);
  assert.ok(start >= 0, `${MLX_LANE} must keep a ${job} job`);
  const rest = workflow.slice(start + 1);
  // Job keys are the only two-space keys in the file; everything inside a job is deeper.
  const end = rest.slice(1).search(/\n {2}[a-z][a-z0-9-]*:\n/);
  const body = end === -1 ? rest : rest.slice(0, end + 1);

  const steps = [];
  const marker = "\n      - ";
  for (let at = body.indexOf(marker); at !== -1; ) {
    const next = body.indexOf(marker, at + 1);
    const chunk = body.slice(at, next === -1 ? undefined : next);
    at = next;
    const named = chunk.match(/^\n {6}- name: (.*)$/m);
    const used = chunk.match(/^\n {6}- uses: (.*)$/m);
    const statements = [];
    let joined = "";
    for (const line of chunk.split("\n")) {
      if (/^\s*#/.test(line)) continue;
      const text = line.trim();
      if (text === "") continue;
      if (text.endsWith("\\")) {
        joined += (joined ? " " : "") + text.slice(0, -1).trim();
        continue;
      }
      statements.push(joined ? `${joined} ${text}` : text);
      joined = "";
    }
    if (joined) statements.push(joined);
    steps.push({ name: named ? named[1] : `uses:${used ? used[1].trim() : "?"}`, statements });
  }
  return steps;
}

// The jobs that carry fatal guards, scanned as ONE set. Both, not just `nax-worker`: main
// moved "Verify capabilities.mlx.json content against a fresh dump" onto the hosted
// `macos-checks` job, because it is a weights-free registry walk that has no business on the
// scarce M5/NAX pool. The guard was not weakened — it still ends every branch in `exit 1` —
// it changed jobs, and the `laneExits` assertion below (which counts the WHOLE FILE) is what
// caught it: that assertion exists precisely so a guard living outside the scanned job cannot
// ship unpinned, and the fix it asks for is to enumerate that job, which is this.
const MLX_GUARD_JOBS = ["macos-checks", "nax-worker"];

function mlxGuardSteps(workflow) {
  return MLX_GUARD_JOBS.flatMap((job) => mlxJobSteps(workflow, job));
}

// ONE ROW PER FATAL GUARD, saying which failure it detects — not merely that the step
// fails somehow. `branch` is the statement chain that reaches the diagnostic, matched
// exactly and in order, which is what distinguishes guards whose MESSAGE is identical:
// "Resolve exact Qwen calibration snapshot" emits the same "not available on this runner"
// string from an `else` fallthrough and from a following `-d` re-check, and the two
// "Validate ..." steps share both the `INFERENCE_PIN` condition and its message.
const MLX_FATAL_GUARDS = [
  {
    step: "Verify capabilities.mlx.json content against a fresh dump",
    detects: "the checked-in MLX facts file differs in capability content from a fresh dump at this pin",
    branch: [
      'if ! node scripts/compare-engine-capability-facts.mjs config/engine-capabilities/capabilities.mlx.json "$scratch/capabilities.mlx.json"; then',
    ],
    diagnostic: /^echo "::error::config\/engine-capabilities\/capabilities\.mlx\.json differs in capability"/,
  },
  {
    step: "Verify capabilities.mlx.json content against a fresh dump",
    detects:
      "the checked-in RUNTIME facts file differs in capability content from inference's fresh snapshot",
    branch: [
      'if ! node scripts/compare-engine-capability-facts.mjs config/engine-capabilities/runtime/capabilities.mlx.json "$scratch/runtime/capabilities.mlx.json"; then',
    ],
    diagnostic:
      /^echo "::error::config\/engine-capabilities\/runtime\/capabilities\.mlx\.json does not"/,
  },
  {
    step: "Verify capabilities.mlx.json content against a fresh dump",
    detects: "the checked-in AUDIO facts file differs in capability content — the one dump BOTH lanes write",
    branch: [
      'if ! node scripts/compare-engine-capability-facts.mjs config/engine-capabilities/audio/capabilities.candle.json "$scratch/audio/capabilities.candle.json"; then',
    ],
    diagnostic:
      /^echo "::error::config\/engine-capabilities\/audio\/capabilities\.candle\.json does not"/,
  },
  {
    step: "Validate Qwen provisioning mode",
    detects: "a ~57 GiB Qwen download requested by a dispatch that will not calibrate",
    branch: [
      'if [[ "$PROVISION_QWEN_SNAPSHOT" == "true" && "$RUN_MEMORY_CALIBRATION" != "true" ]]; then',
    ],
    diagnostic: /^echo "provision_qwen_snapshot requires run_memory_calibration=true" >&2$/,
  },
  {
    step: "Validate Z-Image provisioning mode",
    detects: "a Z-Image download requested by a dispatch that will not capture the reference",
    branch: [
      'if [[ "$PROVISION_Z_IMAGE_SNAPSHOT" == "true" && "$RUN_FIVE_RUNG_REFERENCE" != "true" ]]; then',
    ],
    diagnostic: /^echo "provision_z_image_snapshot requires run_five_rung_reference=true" >&2$/,
  },
  {
    step: "Validate memory-strategy calibration identities",
    detects: "a calibration dispatch whose inference_revision is not an exact 40-hex commit",
    branch: ['if [[ ! "$INFERENCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then'],
    diagnostic: /^echo "inference_revision must be an exact lowercase 40-hex commit" >&2$/,
  },
  {
    step: "Validate memory-strategy calibration identities",
    detects: "a calibration dispatch whose qwen_revision is not an exact 40-hex artifact revision",
    branch: ['if [[ ! "$QWEN_REVISION" =~ ^[0-9a-f]{40}$ ]]; then'],
    diagnostic: /^echo "qwen_revision must be an exact lowercase 40-hex artifact revision" >&2$/,
  },
  {
    step: "Validate memory-strategy calibration identities",
    detects: "calibration pointed at some repository other than the fixed Qwen artifact",
    branch: ['if [[ "$QWEN_REPOSITORY" != "SceneWorks/qwen-image-mlx" ]]; then'],
    diagnostic:
      /^echo "qwen_repository must be the fixed SceneWorks\/qwen-image-mlx calibration artifact" >&2$/,
  },
  {
    step: "Validate memory-strategy calibration identities",
    detects: "a qwen_tier outside the three declared quantization tiers",
    branch: [
      'if [[ "$QWEN_TIER" != "bf16" && "$QWEN_TIER" != "q4" && "$QWEN_TIER" != "q8" ]]; then',
    ],
    diagnostic: /^echo "qwen_tier must be one of bf16, q4, or q8" >&2$/,
  },
  {
    step: "Validate memory-strategy calibration identities",
    detects:
      "calibration evidence stamped with a revision the adapter was NOT compiled against",
    branch: ['if [[ "$PIN" != "$INFERENCE_REVISION" ]]; then'],
    diagnostic:
      /^echo "input inference_revision does not match the adapter's compiled INFERENCE_PIN" >&2$/,
  },
  {
    step: "Validate five-rung reference identities",
    detects: "a five-rung dispatch whose inference_revision is not an exact 40-hex commit",
    branch: ['if [[ ! "$INFERENCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then'],
    diagnostic: /^echo "inference_revision must be an exact lowercase 40-hex commit" >&2$/,
  },
  {
    step: "Validate five-rung reference identities",
    detects:
      "a five-rung dispatch whose z_image_revision is not an exact 40-hex artifact revision",
    branch: ['if [[ ! "$Z_IMAGE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then'],
    diagnostic: /^echo "z_image_revision must be an exact lowercase 40-hex artifact revision" >&2$/,
  },
  {
    step: "Validate five-rung reference identities",
    detects: "a five-rung capture pointed at some repository other than the fixed Z-Image artifact",
    branch: ['if [[ "$Z_IMAGE_REPOSITORY" != "SceneWorks/z-image-turbo-mlx" ]]; then'],
    diagnostic:
      /^echo "z_image_repository must be the fixed SceneWorks\/z-image-turbo-mlx reference artifact" >&2$/,
  },
  {
    step: "Validate five-rung reference identities",
    detects: "five-rung evidence stamped with a revision the adapter was NOT compiled against",
    branch: ['if [[ "$PIN" != "$INFERENCE_REVISION" ]]; then'],
    diagnostic:
      /^echo "input inference_revision does not match the adapter's compiled INFERENCE_PIN" >&2$/,
  },
  {
    step: "Resolve exact Qwen calibration snapshot",
    detects: "the Qwen snapshot is in NEITHER the HF cache nor the Application Support cache",
    branch: [
      'if [[ -d "$QWEN_HF_ROOT" ]]; then',
      'QWEN_ROOT="$QWEN_HF_ROOT"',
      'elif [[ -d "$QWEN_APP_ROOT" ]]; then',
      'QWEN_ROOT="$QWEN_APP_ROOT"',
      "else",
    ],
    diagnostic:
      /^echo "the exact Qwen calibration snapshot is not available on this runner" >&2$/,
  },
  {
    step: "Resolve exact Qwen calibration snapshot",
    detects: "the resolved Qwen root — override included — is not a directory",
    branch: ['if [[ ! -d "$QWEN_ROOT" ]]; then'],
    diagnostic:
      /^echo "the exact Qwen calibration snapshot is not available on this runner" >&2$/,
  },
  {
    step: "Resolve exact Qwen calibration snapshot",
    detects:
      "a real directory that is NOT the requested repository+revision+tier — the dangerous one: " +
      "downgraded, a per-tier measurement runs against the wrong weights and is published as evidence",
    branch: ['if [[ "$QWEN_ROOT" != *"$EXPECTED_SUFFIX" ]]; then'],
    diagnostic:
      /^echo "the Qwen calibration root does not match the fixed repository and exact revision" >&2$/,
  },
  {
    step: "Resolve exact Z-Image reference snapshot",
    detects: "the Z-Image snapshot is in NEITHER the HF cache nor the Application Support cache",
    branch: [
      'if [[ -d "$Z_IMAGE_HF_ROOT" ]]; then',
      'Z_IMAGE_ROOT="$Z_IMAGE_HF_ROOT"',
      'elif [[ -d "$Z_IMAGE_APP_ROOT" ]]; then',
      'Z_IMAGE_ROOT="$Z_IMAGE_APP_ROOT"',
      "else",
    ],
    diagnostic:
      /^echo "the exact Z-Image reference snapshot is not available on this runner" >&2$/,
  },
  {
    step: "Resolve exact Z-Image reference snapshot",
    detects: "a real directory that is NOT the requested Z-Image repository+revision+tier",
    branch: ['if [[ "$Z_IMAGE_ROOT" != *"$EXPECTED_SUFFIX" ]]; then'],
    diagnostic:
      /^echo "the Z-Image reference root does not match the fixed repository and exact revision" >&2$/,
  },
];

// Reachability is the sibling of fatality: a guard whose step never runs is exactly as
// silent as one that never fails. The dispatch-only guards are dispatch-only BY DESIGN, so
// the contract is the exact expression, not its presence. `null` means the step is
// unconditional and must stay that way.
//
// The content-verify step reads `if: ${{ always() }}` since main moved it to `macos-checks`, and
// that is MORE reachable than unconditional, not less: `always()` on a step's `if:` makes it
// run even when an earlier step in the job already failed. It does not swallow this step's own
// failures — only `continue-on-error` and `|| true` do that, and both remain banned below. The
// reason main wants it is stated in the workflow: a pin bump deliberately makes descriptor-
// backed tests fail closed, and `always()` lets the producer and its paired upload still run so
// the fresh facts can be committed, instead of deadlocking the bootstrap.
const MLX_GUARD_STEP_REACHABILITY = {
  "Verify capabilities.mlx.json content against a fresh dump": "if: ${{ always() }}",
  "Validate Qwen provisioning mode": "if: ${{ github.event_name == 'workflow_dispatch' }}",
  "Validate Z-Image provisioning mode": "if: ${{ github.event_name == 'workflow_dispatch' }}",
  "Validate memory-strategy calibration identities":
    "if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_memory_calibration }}",
  "Validate five-rung reference identities":
    "if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_five_rung_reference }}",
  "Resolve exact Qwen calibration snapshot":
    "if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_memory_calibration }}",
  "Resolve exact Z-Image reference snapshot":
    "if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_five_rung_reference }}",
};

test("every failure diagnostic on the MLX lane is fatal, derived from the lane's own text", async () => {
  const workflow = await source(MLX_LANE);
  const steps = mlxGuardSteps(workflow);

  // ANTI-VACUITY. If the splitter stops recognising steps, every loop below is trivially
  // satisfied and this test means nothing — which is exactly how the sibling audit in this
  // file silently emptied itself when sc-18691 changed a guard's polarity.
  assert.ok(
    steps.length >= 20,
    `expected ${MLX_GUARD_JOBS.join(" + ")} to still split into steps, derived ${steps.length}`,
  );
  // Per-job anti-vacuity too: the union above stays over the floor even if one job's splitter
  // silently returns nothing, which would hide every guard that job carries.
  for (const job of MLX_GUARD_JOBS) {
    assert.ok(
      mlxJobSteps(workflow, job).length > 0,
      `expected the ${job} job to still split into steps`,
    );
  }

  const diagnostics = [];
  const exits = [];
  for (const step of steps) {
    step.statements.forEach((statement, at) => {
      if (isMlxFailureDiagnostic(statement)) diagnostics.push({ step, statement, at });
      if (statement === MLX_FATAL_EXIT) exits.push({ step, at });
    });
  }

  // The one number in this test, and it is derived on BOTH sides: the enumeration below is
  // one row per guard, and the scan above is the lane's own text. A guard added without a
  // row, or a row without a guard, breaks this before any message is compared.
  assert.equal(
    diagnostics.length,
    MLX_FATAL_GUARDS.length,
    `${MLX_LANE} carries ${diagnostics.length} failure diagnostics but MLX_FATAL_GUARDS ` +
      `enumerates ${MLX_FATAL_GUARDS.length}. Add or remove the row that says which failure ` +
      "the guard detects — an unenumerated guard is one nothing pins as fatal.",
  );

  // THE DOWNGRADE DETECTOR. `exit 1` -> `echo`, `::warning::`, `exit 0` or deletion all
  // leave the diagnostic standing and remove the exit after it. No count is maintained by
  // hand anywhere in this assertion; both operands come out of the file.
  for (const { step, statement, at } of diagnostics) {
    assert.equal(
      step.statements[at + 1],
      MLX_FATAL_EXIT,
      `${MLX_LANE} / "${step.name}": the diagnostic\n    ${statement}\nmust be followed ` +
        `immediately by \`${MLX_FATAL_EXIT}\`, found ${JSON.stringify(step.statements[at + 1])}. ` +
        "A reported-but-not-fatal failure on this lane publishes a wrong capability dump or a " +
        "wrong memory measurement as evidence. To report without failing, use ::warning::.",
    );
  }

  // The mirror direction, so a fatal exit cannot appear with no diagnostic saying WHY, and
  // so the equality above cannot be satisfied by moving an exit between steps.
  for (const step of steps) {
    const stepDiagnostics = step.statements.filter(isMlxFailureDiagnostic).length;
    const stepExits = step.statements.filter((s) => s === MLX_FATAL_EXIT).length;
    assert.equal(
      stepExits,
      stepDiagnostics,
      `${MLX_LANE} / "${step.name}": ${stepExits} fatal exit(s) against ${stepDiagnostics} ` +
        "failure diagnostic(s). Every fatal exit needs a diagnostic saying which failure it is.",
    );
  }

  // WHOLE FILE, not just the scanned jobs. A guard added anywhere else in the lane would be
  // invisible to the job-scoped scan above and would ship unpinned. This is the assertion that
  // caught main moving the restamp guard into `macos-checks`; the answer it demands is to
  // enumerate that job (see MLX_GUARD_JOBS), never to relax the count. Comment lines stripped
  // so prose cannot move the number either way.
  const laneExits = workflow
    .split("\n")
    .filter((line) => !/^\s*#/.test(line))
    .filter((line) => line.trim() === MLX_FATAL_EXIT).length;
  assert.equal(
    laneExits,
    exits.length,
    `${MLX_LANE} has ${laneExits} fatal exits but only ${exits.length} are inside ` +
      `${MLX_GUARD_JOBS.join(" + ")}. A guard outside those jobs is pinned by nothing here; ` +
      "enumerate its job.",
  );
});

test("each MLX-lane fatal guard is pinned individually, by which failure it detects", async () => {
  const workflow = await source(MLX_LANE);
  const steps = mlxGuardSteps(workflow);
  const byName = new Map(steps.map((step) => [step.name, step]));
  const claimed = new Set();

  for (const guard of MLX_FATAL_GUARDS) {
    const step = byName.get(guard.step);
    assert.ok(step, `${MLX_LANE} must keep a ${MLX_GUARD_JOBS.join(" or ")} step named "${guard.step}"`);

    // Located by the branch chain AND the message together. Either alone is ambiguous on
    // this lane: two guards share the "not available on this runner" message, and two share
    // both the INFERENCE_PIN condition and its message across sibling steps.
    const found = [];
    step.statements.forEach((statement, at) => {
      if (!guard.diagnostic.test(statement)) return;
      const chain = step.statements.slice(at - guard.branch.length, at);
      if (chain.length !== guard.branch.length) return;
      if (chain.every((line, i) => line === guard.branch[i])) found.push(at);
    });
    assert.equal(
      found.length,
      1,
      `${MLX_LANE} / "${guard.step}": expected exactly one guard against ${guard.detects}, ` +
        `matched ${found.length}. Its branch chain is\n    ${guard.branch.join("\n    ")}`,
    );

    const at = found[0];
    assert.equal(
      step.statements[at + 1],
      MLX_FATAL_EXIT,
      `${MLX_LANE} / "${guard.step}": the guard against ${guard.detects} must stay FATAL. ` +
        `Found ${JSON.stringify(step.statements[at + 1])} where \`${MLX_FATAL_EXIT}\` belongs.`,
    );
    claimed.add(`${guard.step}#${at + 1}`);
  }

  // Every fatal exit on the lane is claimed by exactly one row. Without this, deleting a
  // guard outright and adding an unrelated one elsewhere keeps the totals equal.
  for (const step of steps) {
    step.statements.forEach((statement, at) => {
      if (statement !== MLX_FATAL_EXIT) return;
      assert.ok(
        claimed.has(`${step.name}#${at}`),
        `${MLX_LANE} / "${step.name}": a fatal guard at statement ${at} is claimed by no row ` +
          "in MLX_FATAL_GUARDS. Say which failure it detects, so a downgrade names it.",
      );
    });
  }
});

test("MLX-lane guard steps stay reachable and cannot be degraded into warnings", async () => {
  const workflow = await source(MLX_LANE);
  const steps = mlxGuardSteps(workflow);
  const byName = new Map(steps.map((step) => [step.name, step]));

  // Derived from the guards, not hand-listed beside them: every step that carries a row is
  // a step this contract must hold for.
  const guardSteps = [...new Set(MLX_FATAL_GUARDS.map((guard) => guard.step))];
  assert.deepEqual(
    guardSteps.slice().sort(),
    Object.keys(MLX_GUARD_STEP_REACHABILITY).sort(),
    "every step carrying a fatal guard needs a reachability pin, and vice versa",
  );

  for (const name of guardSteps) {
    const step = byName.get(name);
    assert.ok(step, `${MLX_LANE} must keep a ${MLX_GUARD_JOBS.join(" or ")} step named "${name}"`);
    const conditions = step.statements.filter((statement) => statement.startsWith("if: "));
    const expected = MLX_GUARD_STEP_REACHABILITY[name];
    if (expected === null) {
      assert.deepEqual(
        conditions,
        [],
        `${MLX_LANE} / "${name}" must stay unconditional — it is the lane's only check that ` +
          "the checked-in capability dump is real, and it has to run on every PR.",
      );
    } else {
      assert.deepEqual(
        conditions,
        [expected],
        `${MLX_LANE} / "${name}": a guard that never runs is as silent as one that never ` +
          "fails. Pin the exact condition here when the step's reachability changes.",
      );
    }

    // Degrading by SWALLOWING: `continue-on-error` makes every `exit 1` in the step
    // advisory without touching one of them, and `|| true` does it per command. Neither is
    // ever allowed, on any statement.
    for (const statement of step.statements) {
      assert.doesNotMatch(
        statement,
        /continue-on-error|\|\| true/,
        `${MLX_LANE} / "${name}": "${statement}" degrades a guard failure into a warning.`,
      );
    }

    // `always()` is banned everywhere EXCEPT as the step's own pinned `if:` condition, which
    // the `deepEqual` above already fixes to an exact string. The distinction is real:
    // `always()` in a step `if:` makes the step run even after an earlier step in the job
    // failed — strictly more reachable, and it does not touch this step's own exit status.
    // Anywhere else (inside a `run:` body, on a nested expression) it is a swallow. Comparing
    // against `expected` rather than allowing the substring means a step can only carry
    // `always()` if a reachability row deliberately says so.
    for (const statement of step.statements) {
      if (expected !== null && statement === expected) continue;
      assert.doesNotMatch(
        statement,
        /always\(\)/,
        `${MLX_LANE} / "${name}": "${statement}" degrades a guard failure into a warning.`,
      );
    }
  }

  // The content-verify step is the one guard step that runs several commands and a `trap`, and it
  // opts into strictness explicitly. `pipefail` and `-u` are NOT GitHub's defaults (the
  // default shell is `bash -e {0}`), so this is a real declaration, not a restatement of
  // one: without it a failing `cargo run` inside a pipeline, or an unset `$scratch`, reaches
  // the comparison and the guard compares against nothing.
  assert.ok(
    byName
      .get("Verify capabilities.mlx.json content against a fresh dump")
      .statements.includes("set -euo pipefail"),
    `${MLX_LANE}: the content-verify step must keep \`set -euo pipefail\``,
  );
});

// SC-18902 historical acceptance evidence. The real Windows/CUDA baseline proved the former Eros
// Candle route unusable. The published cond_safe LoRA was not a valid candidate for Candle's adapter
// surface (3,320 source keys versus 768 accepted keys), so the retained harness pins the exact
// rejected baseline rather than pretending a partial adapter is usable. The mutation loop is load-bearing:
// each historically plausible drift is injected into an in-memory copy and must make this same
// validator fail, proving the positive assertions are sensitive rather than decorative.
test("the rejected LTX Eros CUDA baseline keeps renderer and product timelines distinct", async () => {
  const documents = {
    manifestText: await source("config/manifests/builtin.models.jsonc"),
    workflow: await source(".github/workflows/windows-candle.yml"),
    harness: await source("crates/sceneworks-worker/src/ltx_eros_gpu_smoke.rs"),
    workerLib: await source("crates/sceneworks-worker/src/lib.rs"),
  };

  const validate = ({ manifestText, workflow, harness, workerLib }) => {
    const manifest = JSON.parse(stripJsoncComments(manifestText));
    const base = manifest.models.find((entry) => entry.id === "ltx_2_3");
    const eros = manifest.models.find((entry) => entry.id === "ltx_2_3_eros");
    assert.ok(base, "the base ltx_2_3 manifest route must remain present");
    assert.ok(eros, "the ltx_2_3_eros manifest route must remain present");
    const shippedDefaults = {
      duration: 6,
      fps: 25,
      resolution: "768x512",
      quality: "balanced",
      steps: 8,
    };
    assert.deepEqual(eros.defaults, shippedDefaults, "Eros manifest defaults must stay fixed");
    assert.deepEqual(base.defaults, shippedDefaults, "base LTX manifest defaults must stay untouched");
    assert.equal(
      base.mlx?.autoDistillLora,
      undefined,
      "the evidence harness must not add the Eros distill recipe to base ltx_2_3",
    );
    assert.deepEqual(
      eros.mlx?.autoDistillLora,
      { stage1Strength: 1, stage2Strength: 0.4 },
      "the existing MLX-only two-pass recipe is context, not a Candle recipe to copy",
    );

    assert.match(
      workflow,
      /run_ltx_eros_acceptance:\s+description:[^\n]+\s+required: false\s+type: boolean\s+default: false/,
      "the expensive real-weight capture must be explicit and off by default",
    );
    assert.doesNotMatch(
      `${workflow}\n${harness}`,
      /AdapterKind|AdapterSpec|LTX_EROS_RECIPE|LTX_EROS_DISTILL_LORA|ltx_eros_recipe|single-pass-distill|DISTILL_(?:REPOSITORY|REVISION|FILE)/,
      "the baseline harness must not expose or construct an unsupported filtered-LoRA candidate",
    );
    assert.match(
      workflow,
      /timeout-minutes: \$\{\{ github\.event_name == 'workflow_dispatch' && \(inputs\.run_ltx_eros_acceptance \|\| inputs\.run_sc19057_wan_capture\) && 360 \|\|/,
      "the real-weight render needs the guarded six-hour ceiling",
    );
    for (const [repository, revision] of [
      ["TenStrip/LTX2.3-10Eros", "84a05a13610d78dbe4340d1be23fd8185e10f697"],
      ["SceneWorks/ltx-2.3-mlx", "01df27d308466533aa09d251e3aebdcc627d07eb"],
    ]) {
      assert.ok(workflow.includes(`repo_id="${repository}"`), `${repository} must be exact`);
      assert.ok(workflow.includes(revision), `${repository} must use its exact revision`);
    }
    const provisioning = workflow.slice(
      workflow.indexOf("- name: Provision exact public LTX Eros acceptance artifacts"),
      workflow.indexOf("- name: Render the fixed LTX Eros CUDA acceptance artifact"),
    );
    assert.equal(
      provisioning.match(/token=False/g)?.length,
      2,
      "both HF downloads must explicitly refuse implicit credentials",
    );
    assert.match(provisioning, /HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"/);
    assert.doesNotMatch(
      provisioning,
      /secrets\.|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|google\/gemma/,
      "the harness must not use a secret or a gated Gemma source",
    );
    assert.match(
      workflow,
      /cargo test -p sceneworks-worker --features backend-candle --release ltx_eros_candle_gpu_smoke -- --ignored --nocapture --test-threads=1/,
    );
    assert.match(
      workflow,
      /ffmpeg [^\n]*-framerate 25[^\n]*-c:v libx264[^\n]*-r 25 \$videoOnly/,
      "the source frames must first be encoded as the product's H.264 video stream",
    );
    assert.match(
      workflow,
      /ffmpeg [^\n]*-i \$videoOnly -i \$audio[^\n]*-c:v copy[^\n]*-c:a aac[^\n]*-shortest[^\n]*-map_metadata 0[^\n]*\$mp4/,
      "the evidence MP4 must use the product's copy-video/AAC/-shortest mux contract",
    );
    assert.match(
      workflow,
      /\$videoOnly = Join-Path \$out 'ltx_eros\.rendered-frames\.mp4'/,
      "the complete 153-frame renderer output must survive beside the product-faithful MP4",
    );
    assert.match(workflow, /stream=index,codec_name,codec_type[^'\n]+duration:format=duration/);
    assert.match(
      workflow,
      /\$video\.codec_name -ne 'h264'/,
      "the MP4 verifier must require the production H.264 video codec",
    );
    assert.match(
      workflow,
      /\$sound\.codec_name -ne 'aac'/,
      "the MP4 verifier must require the production AAC audio codec",
    );
    assert.match(
      workflow,
      /\$wavAudio\.codec_name -ne 'pcm_s16le'/,
      "the source audio verifier must require the worker's PCM16 WAV codec",
    );
    assert.match(
      workflow,
      /\$renderedVideo\.r_frame_rate -ne '25\/1' -or \[int\]\$renderedVideo\.nb_read_frames -ne 153/,
      "the complete renderer output must still prove all 153 frames",
    );
    assert.match(
      workflow,
      /\$productFrames -lt 150 -or \$productFrames -gt 151/,
      "the product-faithful mux must end at the six-second audio boundary",
    );
    assert.match(workflow, /\$renderedDurationSeconds = 153\.0 \/ 25\.0/);
    assert.match(workflow, /\$requestedProductDurationSeconds = 6\.0/);
    assert.match(
      workflow,
      /\$audioDurationSeconds = \[Convert\]::ToDouble\(\$metadata\.result\.audio\.durationSeconds, \[Globalization\.CultureInfo\]::InvariantCulture\)/,
      "the synchronized audio timeline must come from renderer metadata",
    );
    assert.match(
      workflow,
      /\$audioDurationSeconds -lt \$requestedProductDurationSeconds -or \$audioDurationSeconds -gt \$renderedDurationSeconds/,
      "renderer audio must remain bounded by the request and rendered-frame timelines",
    );
    assert.match(workflow, /\$productBoundarySeconds = \[Math\]::Min\(\$renderedDurationSeconds, \$audioDurationSeconds\)/);
    assert.match(workflow, /\$productVideoDurationSeconds = \$productFrames \/ 25\.0/);
    assert.match(
      workflow,
      /\$durationToleranceSeconds = 1\.0 \/ 25\.0/,
      "duration tolerance must remain exactly one output frame",
    );
    for (const invocation of [
      "Assert-DurationNear 'complete rendered-frame video stream' $renderedVideo.duration $renderedDurationSeconds",
      "Assert-DurationNear 'complete rendered-frame container' $renderedProbe.format.duration $renderedDurationSeconds",
      "Assert-DurationNear 'source WAV stream' $wavAudio.duration $audioDurationSeconds",
      "Assert-DurationNear 'source WAV container' $wavProbe.format.duration $audioDurationSeconds",
      "Assert-DurationNear 'product MP4 video stream' $video.duration $productVideoDurationSeconds",
      "Assert-DurationNear 'product MP4 audio stream' $sound.duration $productBoundarySeconds",
      "Assert-DurationNear 'product MP4 container' $probe.format.duration $productBoundarySeconds",
    ]) {
      assert.ok(workflow.includes(invocation), `${invocation} must remain exact`);
    }
    assert.match(workflow, /ffprobe-mp4\.json/);
    assert.match(workflow, /ffprobe-rendered-frames\.json/);
    assert.match(workflow, /ffprobe-wav\.json/);
    assert.match(
      workflow,
      /sc-18902-ltx-eros-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}\/ltx_eros\.rendered-frames\.mp4/,
      "the inspected artifact must include the untrimmed renderer output",
    );
    assert.match(
      workflow,
      /"\$productHash  ltx_eros\.mp4"\s+"\$renderedHash  ltx_eros\.rendered-frames\.mp4"/,
      "the checksum manifest must bind both inspectable MP4s",
    );
    assert.match(
      workflow,
      /\$productHash = \(Get-FileHash -Algorithm SHA256 -LiteralPath \$mp4\)\.Hash\.ToLowerInvariant\(\)/,
      "the product checksum must hash the product-faithful MP4",
    );
    assert.match(
      workflow,
      /\$renderedHash = \(Get-FileHash -Algorithm SHA256 -LiteralPath \$videoOnly\)\.Hash\.ToLowerInvariant\(\)/,
      "the renderer checksum must hash the complete rendered-frame MP4",
    );
    assert.match(
      workflow,
      /\$metadata\.captureKind -ne 'current-candle-route-baseline'/,
      "uploaded metadata must identify the product-neutral current-route baseline",
    );
    assert.match(
      workflow,
      /\$ffmpegVersionLines = @\(& ffmpeg -version\)\s+if \(\$LASTEXITCODE -ne 0\) \{ throw 'ffmpeg version probe failed' \}\s+\$ffmpegVersion = \(\$ffmpegVersionLines \| Select-Object -First 1\)\.Trim\(\)/,
      "the runner probe must consume ffmpeg output before selecting its version line",
    );
    assert.doesNotMatch(
      workflow,
      /ffmpeg -version \| Select-Object -First 1/,
      "an early-closing native pipeline must not leak exit code 1 into the Actions wrapper",
    );
    assert.match(workflow, /ffmpeg = \$ffmpegVersion/);
    assert.match(
      workflow,
      /if: \$\{\{ success\(\) && github\.event_name == 'workflow_dispatch' && inputs\.run_ltx_eros_acceptance \}\}\s+uses: actions\/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a/,
      "only a verified capture may publish the MP4/stills/metadata bundle",
    );

    for (const declaration of [
      "const WIDTH: u32 = 768;",
      "const HEIGHT: u32 = 512;",
      "const DURATION_SECONDS: u32 = 6;",
      "const FPS: u32 = 25;",
      "const STEPS: u32 = 8;",
      "const RENDERED_FRAMES: u32 = 153;",
      "const SEED: u64 = 18_902;",
      "assert_eq!(request.sampler, None);",
      'const CAPTURE_KIND: &str = "current-candle-route-baseline";',
    ]) {
      assert.ok(harness.includes(declaration), `fixed harness contract missing: ${declaration}`);
    }
    assert.doesNotMatch(
      harness,
      /sampler:\s*Some\(/,
      "the current-route baseline must preserve the manifest-default absent sampler",
    );
    assert.match(
      harness,
      /"sampler": null/,
      "capture metadata must record the manifest-default absent sampler",
    );
    assert.match(harness, /const PROMPT: &str = "A wide locked camera shot of a red fox/);
    assert.doesNotMatch(
      `${workflow}\n${harness}`,
      /LTX_EROS_(?:PROMPT|SEED|WIDTH|HEIGHT|DURATION|FPS|STEPS)/,
      "evidence-defining request fields must not be environment-overridable",
    );
    assert.match(harness, /fn load_spec\(eros_dir: PathBuf, gemma_dir: PathBuf\) -> LoadSpec/);
    assert.doesNotMatch(harness, /\.with_adapters\(/);
    assert.match(harness, /assert!\(spec\.adapters\.is_empty\(\)\)/);
    assert.match(harness, /"captureKind": CAPTURE_KIND/);
    assert.match(harness, /adapter_reports\.is_empty\(\)/);
    assert.match(harness, /"pairDeltas": pair_deltas/);
    assert.match(harness, /"frameStats": frame_stats/);
    assert.match(harness, /"renderedFramesDurationSeconds": RENDERED_FRAMES as f64 \/ FPS as f64/);
    assert.match(harness, /"productDurationSeconds": DURATION_SECONDS/);
    assert.doesNotMatch(harness, /"encodedDurationSeconds"/);
    assert.match(harness, /audio\.expect\("Candle LTX Eros must return its synchronized audio track"\)/);
    assert.match(
      workerLib,
      /#\[cfg\(all\(test, not\(target_os = "macos"\), feature = "backend-candle"\)\)\]\s*\nmod ltx_eros_gpu_smoke;/,
      "the ignored real-weight test must compile on the actual Windows Candle lane",
    );
  };

  assert.doesNotThrow(() => validate(documents));

  const manifestObject = JSON.parse(stripJsoncComments(documents.manifestText));
  const manifestMutation = (mutate) => {
    const copy = structuredClone(manifestObject);
    mutate(copy);
    return JSON.stringify(copy);
  };
  const mutations = [
    {
      name: "manifest duration drift",
      expected: /Eros manifest defaults must stay fixed/,
      documents: {
        ...documents,
        manifestText: manifestMutation((copy) => {
          copy.models.find((entry) => entry.id === "ltx_2_3_eros").defaults.duration = 5;
        }),
      },
    },
    {
      name: "base route contamination",
      expected: /must not add the Eros distill recipe to base/,
      documents: {
        ...documents,
        manifestText: manifestMutation((copy) => {
          copy.models.find((entry) => entry.id === "ltx_2_3").mlx.autoDistillLora = {
            stage1Strength: 1,
            stage2Strength: 0.4,
          };
        }),
      },
    },
    {
      name: "auto-dispatch regression",
      expected: /must be explicit and off by default/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          /(run_ltx_eros_acceptance:[\s\S]*?default:) false/,
          "$1 true",
        ),
      },
    },
    {
      name: "credentialed HF regression",
      expected: /both HF downloads must explicitly refuse implicit credentials/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          /(repo_id="TenStrip\/LTX2\.3-10Eros"[\s\S]*?)token=False/,
          "$1token=True",
        ),
      },
    },
    {
      name: "unsupported candidate mode introduced",
      expected: /must not expose or construct an unsupported filtered-LoRA candidate/,
      documents: {
        ...documents,
        workflow: `${documents.workflow}\nltx_eros_recipe: single-pass-distill`,
      },
    },
    {
      name: "production shortest mux removed",
      expected: /copy-video\/AAC\/-shortest mux contract/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "-b:a 192k -shortest -map_metadata",
          "-b:a 192k -map_metadata",
        ),
      },
    },
    {
      name: "H.264 verification weakened",
      expected: /must require the production H\.264 video codec/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "$video.codec_name -ne 'h264'",
          "$video.codec_name -ne 'hevc'",
        ),
      },
    },
    {
      name: "duration tolerance widened",
      expected: /duration tolerance must remain exactly one output frame/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "$durationToleranceSeconds = 1.0 / 25.0",
          "$durationToleranceSeconds = 2.0 / 25.0",
        ),
      },
    },
    {
      name: "product MP4 audio duration verification removed",
      expected: /product MP4 audio stream.*must remain exact/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "Assert-DurationNear 'product MP4 audio stream' $sound.duration $productBoundarySeconds",
          "",
        ),
      },
    },
    {
      name: "renderer duration wired to the product timeline",
      expected: /complete rendered-frame video stream.*must remain exact/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "Assert-DurationNear 'complete rendered-frame video stream' $renderedVideo.duration $renderedDurationSeconds",
          "Assert-DurationNear 'complete rendered-frame video stream' $renderedVideo.duration $productBoundarySeconds",
        ),
      },
    },
    {
      name: "product duration wired to the renderer timeline",
      expected: /product MP4 container.*must remain exact/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "Assert-DurationNear 'product MP4 container' $probe.format.duration $productBoundarySeconds",
          "Assert-DurationNear 'product MP4 container' $probe.format.duration $renderedDurationSeconds",
        ),
      },
    },
    {
      name: "early-closing ffmpeg version pipeline restored",
      expected: /must consume ffmpeg output before selecting its version line|must not leak exit code 1/,
      documents: {
        ...documents,
        workflow: documents.workflow
          .replace("$ffmpegVersionLines = @(& ffmpeg -version)", "$ffmpegVersionLines = & ffmpeg -version | Select-Object -First 1")
          .replace("$ffmpegVersion = ($ffmpegVersionLines | Select-Object -First 1).Trim()", "$ffmpegVersion = $ffmpegVersionLines.Trim()"),
      },
    },
    {
      name: "renderer checksum omitted",
      expected: /checksum manifest must bind both inspectable MP4s/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          '            "$renderedHash  ltx_eros.rendered-frames.mp4"\n',
          "",
        ),
      },
    },
    {
      name: "renderer checksum hashes the product MP4",
      expected: /renderer checksum must hash the complete rendered-frame MP4/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "$renderedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $videoOnly)",
          "$renderedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $mp4)",
        ),
      },
    },
    {
      name: "complete renderer frame assertion weakened",
      expected: /must still prove all 153 frames/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "[int]$renderedVideo.nb_read_frames -ne 153",
          "[int]$renderedVideo.nb_read_frames -lt 150",
        ),
      },
    },
    {
      name: "product mux accepts the untrimmed lattice",
      expected: /must end at the six-second audio boundary/,
      documents: {
        ...documents,
        workflow: documents.workflow.replace(
          "$productFrames -gt 151",
          "$productFrames -gt 153",
        ),
      },
    },
    {
      name: "explicit sampler alias reintroduced",
      expected: /manifest-default absent sampler/,
      documents: {
        ...documents,
        harness: documents.harness.replace(
          "steps: Some(STEPS),",
          'steps: Some(STEPS),\n        sampler: Some("rectified-flow".to_owned()),',
        ),
      },
    },
  ];
  for (const mutation of mutations) {
    assert.throws(
      () => validate(mutation.documents),
      mutation.expected,
      `${mutation.name} must be killed by the acceptance contract`,
    );
  }
});
