import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const WORKFLOW_URL = new URL("../.github/workflows/windows-candle.yml", import.meta.url);

function stepBody(workflow, name) {
  const start = workflow.indexOf(`      - name: ${name}\n`);
  assert.ok(start >= 0, `missing workflow step ${name}`);
  const end = workflow.indexOf("\n      - ", start + 1);
  return workflow.slice(start, end < 0 ? undefined : end);
}

function dispatchInputNames(workflow) {
  const start = workflow.indexOf("  workflow_dispatch:\n    inputs:\n");
  assert.ok(start >= 0, "missing workflow_dispatch inputs");
  const end = workflow.indexOf("\npermissions:", start);
  assert.ok(end > start, "workflow_dispatch inputs must end before permissions");
  return [...workflow.slice(start, end).matchAll(/^ {6}([a-z][a-z0-9_]+):$/gm)].map((match) => match[1]);
}

function escaped(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function assertWorkflowContract(workflow) {
  assert.match(workflow, /^permissions:\n {2}contents: read$/m);
  assert.match(workflow, /^ {6}run_sc19057_wan_capture:$/m);
  assert.match(
    workflow,
    /runs-on: \$\{\{ \(github\.event_name == 'workflow_dispatch' && \(inputs\.provision_snapshot \|\| inputs\.run_five_rung_reference \|\| inputs\.run_sc19054_flux_acceptance \|\| inputs\.run_sc19057_wan_capture\)\)/,
  );
  assert.match(
    workflow,
    /timeout-minutes: \$\{\{ github\.event_name == 'workflow_dispatch' && \(inputs\.run_ltx_eros_acceptance \|\| inputs\.run_sc19057_wan_capture\) && 360/,
  );

  const validate = stepBody(workflow, "Validate dispatch inputs");
  const realModes = dispatchInputNames(workflow).filter((name) => name.startsWith("run_"));
  assert.deepEqual(realModes, [
    "run_five_rung_reference",
    "run_ltx_eros_acceptance",
    "run_sc19054_flux_acceptance",
    "run_sc19057_wan_capture",
  ]);
  for (const mode of realModes) {
    const envName = mode.toUpperCase();
    assert.match(validate, new RegExp(`${envName}: \\$\\{\\{ inputs\\.${escaped(mode)} \\}\\}`));
    assert.match(
      validate,
      new RegExp(`if \\(\\$env:${envName} -eq 'true'\\) \\{ \\$selectedModes \\+= '${escaped(mode)}' \\}`),
    );
  }
  assert.match(validate, /if \(\$selectedModes\.Count -gt 1\)/);
  assert.match(validate, /real-weight execution modes are mutually exclusive/);
  for (const exact of [
    "4013049764172ee7dc707101c7da8c83c1483f2d",
    "SceneWorks/wan2.2-ti2v-5b-candle",
    "9b173dc8660334a87a11e67de58939afe68f8cb2",
    "q4/**",
  ]) assert.match(validate, new RegExp(escaped(exact)));
  assert.match(validate, /RUN_SC19057_WAN_CAPTURE -eq 'true'/);
  assert.match(validate, /PROVISION_SNAPSHOT -ne 'true'/);
  assert.match(validate, /PROVISION_SUBDIR -ne 'q4'/);

  const exactGate = /if: \$\{\{ (?:success\(\) && )?github\.event_name == 'workflow_dispatch' && inputs\.run_sc19057_wan_capture \}\}/;
  for (const name of [
    "Inventory the exact SC-19057 Wan q4 artifact",
    "Verify the exact SC-19057 paired source closure",
    "Build the release SC-19057 Candle memory adapter",
    "Capture and accept the exact SC-19057 six-row terminal bundle",
    "Seal the SC-19057 source runner and receipt identity",
  ]) assert.match(stepBody(workflow, name), exactGate, `${name} must remain dispatch-only`);

  const inventory = stepBody(workflow, "Inventory the exact SC-19057 Wan q4 artifact");
  assert.match(inventory, /\$files\.Count -ne 25/);
  assert.match(inventory, /\$total -ne 17338835457/);
  assert.match(inventory, /Get-FileHash -Algorithm SHA256/);
  assert.match(inventory, /wan-q4-artifact-inventory\.json/);

  const checkout = stepBody(workflow, "Check out the exact inference reference source");
  assert.match(checkout, /inputs\.run_five_rung_reference \|\| inputs\.run_sc19057_wan_capture/);
  assert.match(checkout, /repository: SceneWorks\/inference/);
  assert.match(checkout, /ref: \$\{\{ inputs\.inference_revision \}\}/);
  assert.match(checkout, /persist-credentials: false/);

  const closure = stepBody(workflow, "Verify the exact SC-19057 paired source closure");
  assert.match(closure, /\$sceneHead -ne \$env:GITHUB_SHA/);
  assert.match(closure, /\$inferenceHead -ne \$env:INFERENCE_REVISION/);
  assert.match(closure, /git status --porcelain/);
  assert.match(closure, /git -C '\.calibration\\inference' status --porcelain/);
  assert.match(closure, /inference-closure-digest\.mjs .* --check/);

  const build = stepBody(workflow, "Build the release SC-19057 Candle memory adapter");
  assert.match(build, /cargo build --release --locked -p sceneworks-memory-adapter --features candle --bin memory-candle-adapter/);

  const capture = stepBody(workflow, "Capture and accept the exact SC-19057 six-row terminal bundle");
  assert.match(capture, /SCENEWORKS_WAN_REPOSITORY: \$\{\{ inputs\.provision_repository \}\}/);
  assert.match(capture, /SCENEWORKS_WAN_REVISION: \$\{\{ inputs\.provision_revision \}\}/);
  assert.match(capture, /\$env:SCENEWORKS_WAN_ROOT = \$env:SCENEWORKS_PROVISIONED_ROOT/);
  assert.match(capture, /ConvertTo-Json -Compress -InputObject @\(\$adapter\)/);
  assert.match(capture, /--config docs\\calibration\\sc-19057\\wan-candle-video-capture-plan\.json/);
  assert.match(capture, /--backend candle/);
  assert.match(capture, /--fresh-per-case/);
  assert.match(capture, /memory-calibration-harness\.mjs check --input \$capture/);
  assert.match(capture, /validate-sc19057-wan-capture\.mjs/);
  assert.match(capture, /--sceneworks-revision \$env:GITHUB_SHA/);
  assert.match(capture, /--inference-revision \$env:INFERENCE_REVISION/);
  assert.match(capture, /--write-receipt \$receipt/);
  assert.doesNotMatch(capture, /--resume|--raw-log-dir|--source-path-prefix|--batch-rungs/);

  const seal = stepBody(workflow, "Seal the SC-19057 source runner and receipt identity");
  assert.match(seal, /git status --porcelain/);
  assert.match(seal, /Get-FileHash -Algorithm SHA256/);
  assert.match(seal, /run-identity\.json/);
  assert.match(seal, /sha256\.txt/);

  const upload = stepBody(workflow, "Upload the sealed SC-19057 capture attempt");
  assert.match(upload, /if: \$\{\{ always\(\) && github\.event_name == 'workflow_dispatch' && inputs\.run_sc19057_wan_capture \}\}/);
  assert.match(upload, /actions\/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a/);
  assert.match(upload, /if-no-files-found: error/);
  assert.match(upload, /retention-days: 30/);

  const cleanup = stepBody(workflow, "Clean bounded SC-19057 job scratch");
  assert.match(cleanup, /if: \$\{\{ always\(\) && github\.event_name == 'workflow_dispatch' && inputs\.run_sc19057_wan_capture \}\}/);
  assert.match(cleanup, /refusing cleanup outside bounded job scratch/);
  assert.match(cleanup, /Remove-Item -LiteralPath \$full -Recurse -Force/);
  assert.doesNotMatch(cleanup, /SCENEWORKS_PROVISIONED_ROOT|PROVISION_CACHE_DIR/);

  const acceptAt = workflow.indexOf("validate-sc19057-wan-capture.mjs");
  const uploadAt = workflow.indexOf("name: Upload the sealed SC-19057 capture attempt");
  assert.ok(acceptAt >= 0 && acceptAt < uploadAt, "6/6 acceptance must precede evidence upload");
}

test("windows-candle exposes one exact mutually-exclusive manual SC-19057 mode", async () => {
  assertWorkflowContract(await readFile(WORKFLOW_URL, "utf8"));
});

test("the workflow contract kills routing artifact capture and cleanup disconnects", async () => {
  const workflow = await readFile(WORKFLOW_URL, "utf8");
  const mutations = [
    ["wrong runner", (text) => text.replace("inputs.run_sc19057_wan_capture)) && fromJSON", "false)) && fromJSON")],
    ["wrong timeout", (text) => text.replace("inputs.run_ltx_eros_acceptance || inputs.run_sc19057_wan_capture", "inputs.run_ltx_eros_acceptance")],
    ["missing mutual exclusion", (text) => text.replace("if ($selectedModes.Count -gt 1)", "if ($selectedModes.Count -gt 99)")],
    ["missing mode dispatch", (text) => text.replace("$selectedModes += 'run_sc19057_wan_capture'", "$null = 'run_sc19057_wan_capture'")],
    ["wrong inference pin", (text) => text.replaceAll("4013049764172ee7dc707101c7da8c83c1483f2d", "a".repeat(40))],
    ["wrong artifact repo", (text) => text.replaceAll("SceneWorks/wan2.2-ti2v-5b-candle", "SceneWorks/lookalike")],
    ["wrong artifact revision", (text) => text.replaceAll("9b173dc8660334a87a11e67de58939afe68f8cb2", "b".repeat(40))],
    ["wrong artifact count", (text) => text.replace("$files.Count -ne 25", "$files.Count -ne 24")],
    ["wrong artifact bytes", (text) => text.replace("$total -ne 17338835457", "$total -ne 1")],
    ["non-release adapter", (text) => text.replaceAll("cargo build --release --locked", "cargo build --locked")],
    ["missing fresh isolation", (text) => text.replace("            --fresh-per-case `\n", "")],
    ["missing 6/6 validator", (text) => text.replace("validate-sc19057-wan-capture.mjs", "accept-any-capture.mjs")],
    ["unsafe raw-log arm", (text) => text.replace("            --fresh-per-case `\n", "            --fresh-per-case --raw-log-dir logs `\n")],
    ["automatic capture", (text) => text.replace("github.event_name == 'workflow_dispatch' && inputs.run_sc19057_wan_capture", "github.event_name == 'push'")],
    ["unbounded cleanup", (text) => text.replace("Remove-Item -LiteralPath $full -Recurse -Force", "Remove-Item -Path $env:RUNNER_TEMP\\* -Recurse -Force")],
  ];
  for (const [label, mutate] of mutations) {
    assert.throws(() => assertWorkflowContract(mutate(workflow)), undefined, label);
  }
});
