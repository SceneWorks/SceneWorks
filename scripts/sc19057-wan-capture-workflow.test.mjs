import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const WORKFLOW_URL = new URL("../.github/workflows/windows-candle.yml", import.meta.url);
const INVENTORY_URL = new URL("./inventory-sc19057-wan-artifact.mjs", import.meta.url);
const PROVIDER_HARNESS_URL = new URL("./memory-calibration-harness.mjs", import.meta.url);

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

function assertInventorySourceContract(source) {
  assert.match(source, /export const SC19057_WAN_TOTAL_BYTES = 17_338_835_457/);
  assert.match(source, /export const SC19057_WAN_FILES = Object\.freeze\(\{/);
  assert.match(source, /path\.posix\.isAbsolute\(target\)/);
  assert.match(source, /path\.win32\.isAbsolute\(target\)/);
  assert.match(source, /\/\^\[A-Za-z\]\:\/\.test\(target\)/);
  assert.match(source, /const metadata = await lstat\(entryPath\)/);
  assert.match(source, /const rawTarget = await readlink\(logicalPath\)/);
  assert.match(source, /const lexicalTarget = validateRawLinkTarget\(\{ logicalPath, rawTarget, repositoryRoot, expectedObject \}\)/);
  assert.match(source, /path\.join\(repositoryRoot, "blobs", expectedObject\)/);
  assert.match(source, /samePath\(actualLexicalTarget, expectedLexicalTarget\)/);
  assert.match(source, /const targetMetadata = await lstat\(lexicalTarget\)/);
  assert.match(source, /!targetMetadata\.isFile\(\) \|\| targetMetadata\.isSymbolicLink\(\)/);
  assert.match(source, /const physicalPath = await realpath\(logicalPath\)/);
  assert.match(source, /if \(logicalMetadata\.isSymbolicLink\(\)\)/);
  assert.match(source, /!isInside\(canonicalBlobsRoot, physicalPath\)/);
  assert.match(source, /!samePath\(path\.dirname\(physicalPath\), canonicalBlobsRoot\)/);
  assert.match(source, /path\.basename\(physicalPath\)\.toLowerCase\(\) !== expectedObject/);
  assert.match(source, /seenPhysicalPaths\.has\(pathKey\(physicalPath\)\)/);
  assert.match(source, /seenPhysicalPaths\.add\(pathKey\(physicalPath\)\)/);
  assert.match(source, /const handle = await open\(file, "r"\)/);
  assert.match(source, /const metadata = await handle\.stat\(\)/);
  assert.match(source, /const \{ bytesRead \} = await handle\.read\(buffer, 0, buffer\.length, streamedBytes\)/);
  assert.match(source, /await handle\.close\(\)/);
  assert.match(source, /inspected\.streamedBytes !== inspected\.metadata\.size/);
  assert.match(source, /inspected\.streamedBytes !== expectedBytes/);
  assert.match(source, /expectedObject\.length === 64 \? inspected\.sha256 : inspected\.gitBlob/);
  assert.doesNotMatch(source, /createReadStream|await stat\(physicalPath\)/);
  const readlinkAt = source.indexOf("await readlink(logicalPath)");
  const targetLstatAt = source.indexOf("await lstat(lexicalTarget)");
  const realpathAt = source.indexOf("await realpath(logicalPath)");
  assert.ok(readlinkAt >= 0 && readlinkAt < targetLstatAt && targetLstatAt < realpathAt, "raw link authentication must precede dereference");
  assert.match(source, /status: "STARTED"/);
  assert.match(source, /status: "PASS"/);
  assert.match(source, /status: "FAIL"/);
}

function assertProviderTransportSourceContract(source) {
  assert.match(source, /export async function readExactProviderCommandFile/);
  assert.match(source, /if \(!file \|\| !path\.isAbsolute\(file\)\)/);
  assert.match(source, /--provider-cmd-json-file must be an absolute path/);
  assert.match(source, /--provider-executable must be an absolute path with --provider-cmd-json-file/);
  assert.match(source, /lstat\(commandFile\)/);
  assert.match(source, /!metadata\.isFile\(\) \|\| metadata\.isSymbolicLink\(\)/);
  assert.match(source, /canonicalCommandFile.*realpath\(commandFile\)/s);
  assert.match(source, /sameFilesystemPath\(canonicalCommandFile, commandFile\)/);
  assert.match(source, /canonicalRoot = await realpath\(root\)/);
  assert.match(source, /isWithin\(canonicalRoot, canonicalCommandFile\)/);
  assert.match(source, /!Array\.isArray\(parsed\) \|\| parsed\.length !== 1/);
  assert.match(source, /parsed\[0\]\.includes\("\\0"\)/);
  assert.match(source, /path\.isAbsolute\(parsed\[0\]\)/);
  assert.match(source, /lstat\(commandPath\)/);
  assert.match(source, /realpath\(commandPath\)/);
  assert.match(source, /sameFilesystemPath\(commandPath, expectedPath\)/);
  assert.match(source, /sameFilesystemPath\(canonicalCommand, canonicalExpected\)/);
  assert.match(source, /forbiddenRoots: \[sceneWorksRepo, inferenceRepo, path\.dirname\(path\.resolve\(outputPath\)\)\]/);
  assert.match(source, /indexes\.length > 1/);
  assert.match(source, /candidate\.startsWith\("--"\)/);
  assert.match(source, /Boolean\(inline\) === Boolean\(file\)/);
  assert.match(
    source,
    /const providerCommand = await providerCommandFromArgs\([\s\S]*?const output = await runProviderPlan\(\{\s*config:[\s\S]*?providerCommand,\s*sceneWorksRepo,\s*inferenceRepo,/,
  );
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
  assert.match(inventory, /New-Item -ItemType Directory -Force -Path \$evidence/);
  assert.match(inventory, /wan-q4-inventory-preflight\.json/);
  assert.match(inventory, /SC19057_EVIDENCE_DIR=\$evidence/);
  assert.match(inventory, /node scripts\\inventory-sc19057-wan-artifact\.mjs --root "\$env:SCENEWORKS_PROVISIONED_ROOT" --evidence "\$evidence"/);
  const directoryAt = inventory.indexOf("New-Item -ItemType Directory -Force -Path $evidence");
  const preflightAt = inventory.indexOf("wan-q4-inventory-preflight.json");
  const exportAt = inventory.indexOf("SC19057_EVIDENCE_DIR=$evidence");
  const inventoryAt = inventory.indexOf("node scripts\\inventory-sc19057-wan-artifact.mjs");
  assert.ok(directoryAt >= 0 && directoryAt < preflightAt, "evidence directory must precede the initial receipt");
  assert.ok(preflightAt < inventoryAt, "initial receipt must precede every artifact inventory gate");
  assert.ok(exportAt < inventoryAt, "always-upload evidence path must be exported before inventory can fail");

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
  assert.match(capture, /\$providerCommandFile = Join-Path \$env:RUNNER_TEMP "sc-19057-provider-command-\$\(\$env:GITHUB_RUN_ID\)-\$\(\$env:GITHUB_RUN_ATTEMPT\)\.json"/);
  assert.match(capture, /\$providerCommandTemporary = "\$providerCommandFile\.tmp"/);
  assert.match(capture, /provider-command-transport\.json/);
  assert.match(capture, /status = 'STARTED'/);
  assert.match(capture, /ConvertTo-Json -Compress -InputObject @\(\[string\]\$adapter\)/);
  assert.match(capture, /New-Object System\.Text\.UTF8Encoding\(\$false\)/);
  assert.match(capture, /\[System\.IO\.File\]::WriteAllText\(\$providerCommandTemporary, "\$providerJson`n", \$utf8WithoutBom\)/);
  assert.match(capture, /Move-Item -LiteralPath \$providerCommandTemporary -Destination \$providerCommandFile/);
  assert.match(capture, /providerCommandFileSha256 = \(Get-FileHash -Algorithm SHA256 -LiteralPath \$providerCommandFile\)/);
  assert.match(capture, /status = 'FAIL'/);
  assert.match(capture, /--config docs\\calibration\\sc-19057\\wan-candle-video-capture-plan\.json/);
  assert.match(capture, /--backend candle/);
  assert.match(capture, /--fresh-per-case/);
  assert.match(capture, /--provider-cmd-json-file \$providerCommandFile/);
  assert.match(capture, /--provider-executable \$adapter/);
  assert.doesNotMatch(capture, /--provider-command(?:\s|$)/);
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
  assert.match(cleanup, /sc-19057-provider-command-\$\(\$env:GITHUB_RUN_ID\)-\$\(\$env:GITHUB_RUN_ATTEMPT\)\.json/);
  assert.match(cleanup, /sc-19057-provider-command-\$\(\$env:GITHUB_RUN_ID\)-\$\(\$env:GITHUB_RUN_ATTEMPT\)\.json\.tmp/);
  assert.doesNotMatch(cleanup, /SCENEWORKS_PROVISIONED_ROOT|PROVISION_CACHE_DIR/);

  const receiptAt = capture.indexOf("$transport | ConvertTo-Json -Depth 3 | Out-File -LiteralPath $providerTransportReceipt -Encoding utf8");
  const adapterResolveAt = capture.indexOf("$adapter = (Resolve-Path -LiteralPath $expectedAdapter).Path");
  const transportWriteAt = capture.indexOf("[System.IO.File]::WriteAllText");
  const harnessAt = capture.indexOf("node scripts\\memory-calibration-harness.mjs run");
  assert.ok(receiptAt >= 0 && receiptAt < adapterResolveAt, "transport failure receipt must precede adapter resolution");
  assert.ok(adapterResolveAt < transportWriteAt, "adapter identity must be resolved before the atomic write");
  assert.ok(transportWriteAt < harnessAt, "atomic provider transport must be complete before capture starts");

  const acceptAt = workflow.indexOf("validate-sc19057-wan-capture.mjs");
  const uploadAt = workflow.indexOf("name: Upload the sealed SC-19057 capture attempt");
  assert.ok(acceptAt >= 0 && acceptAt < uploadAt, "6/6 acceptance must precede evidence upload");
}

test("windows-candle exposes one exact mutually-exclusive manual SC-19057 mode", async () => {
  assertWorkflowContract(await readFile(WORKFLOW_URL, "utf8"));
  assertInventorySourceContract(await readFile(INVENTORY_URL, "utf8"));
  assertProviderTransportSourceContract(await readFile(PROVIDER_HARNESS_URL, "utf8"));
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
    ["missing early inventory receipt", (text) => text.replace("wan-q4-inventory-preflight.json", "late-only.json")],
    ["missing early evidence export", (text) => text.replace("SC19057_EVIDENCE_DIR=$evidence", "SC19057_EVIDENCE_DIR=late")],
    ["disconnected resolved inventory", (text) => text.replace("node scripts\\inventory-sc19057-wan-artifact.mjs", "node scripts\\unsafe-link-length.mjs")],
    ["non-release adapter", (text) => text.replaceAll("cargo build --release --locked", "cargo build --locked")],
    ["missing fresh isolation", (text) => text.replace("            --fresh-per-case `\n", "")],
    ["inline JSON native argv", (text) => text.replace("            --provider-cmd-json-file $providerCommandFile `", "            --provider-command $providerJson `")],
    ["multiple provider argv", (text) => text.replace("@([string]$adapter)", "@([string]$adapter, '--unexpected')")],
    ["BOM provider JSON", (text) => text.replace("System.Text.UTF8Encoding($false)", "System.Text.UTF8Encoding($true)")],
    ["provider path substitution", (text) => text.replace("--provider-executable $adapter", "--provider-executable 'lookalike.exe'")],
    ["non-atomic provider transport", (text) => text.replace("Move-Item -LiteralPath $providerCommandTemporary -Destination $providerCommandFile", "Copy-Item -LiteralPath $providerCommandTemporary -Destination $providerCommandFile")],
    ["late provider failure evidence", (text) => text.replace("$transport | ConvertTo-Json -Depth 3 | Out-File -LiteralPath $providerTransportReceipt -Encoding utf8\n          try {", "try {")],
    ["transport cleanup disconnected", (text) => text.replace("sc-19057-provider-command-$($env:GITHUB_RUN_ID)-$($env:GITHUB_RUN_ATTEMPT).json.tmp", "disconnected-provider-command.tmp")],
    ["missing 6/6 validator", (text) => text.replace("validate-sc19057-wan-capture.mjs", "accept-any-capture.mjs")],
    ["unsafe raw-log arm", (text) => text.replace("            --fresh-per-case `\n", "            --fresh-per-case --raw-log-dir logs `\n")],
    ["automatic capture", (text) => text.replace("github.event_name == 'workflow_dispatch' && inputs.run_sc19057_wan_capture", "github.event_name == 'push'")],
    ["unbounded cleanup", (text) => text.replace("Remove-Item -LiteralPath $full -Recurse -Force", "Remove-Item -Path $env:RUNNER_TEMP\\* -Recurse -Force")],
  ];
  for (const [label, mutate] of mutations) {
    assert.throws(() => assertWorkflowContract(mutate(workflow)), undefined, label);
  }
});

test("the harness source contract kills provider schema path and identity bypasses", async () => {
  const source = await readFile(PROVIDER_HARNESS_URL, "utf8");
  const mutations = [
    ["relative command file", (text) => text.replace("!file || !path.isAbsolute(file)", "!file")],
    ["linked command file", (text) => text.replace("!metadata.isFile() || metadata.isSymbolicLink()", "!metadata.isFile()")],
    ["linked command parent", (text) => text.replace("!sameFilesystemPath(canonicalCommandFile, commandFile)", "false")],
    ["lexical forbidden root", (text) => text.replace("isWithin(canonicalRoot, canonicalCommandFile)", "isWithin(root, commandFile)")],
    ["multiple argv", (text) => text.replace("parsed.length !== 1", "parsed.length === 0")],
    ["NUL argv", (text) => text.replace('parsed[0].includes("\\0")', "false")],
    ["relative executable", (text) => text.replace("!path.isAbsolute(parsed[0])", "false")],
    ["unresolved identity", (text) => text.replace("realpath(commandPath)", "commandPath")],
    ["lexical identity alias", (text) => text.replace("!sameFilesystemPath(commandPath, expectedPath)", "false")],
    ["identity mismatch", (text) => text.replace("!sameFilesystemPath(canonicalCommand, canonicalExpected)", "false")],
    ["checkout transport", (text) => text.replace("forbiddenRoots: [sceneWorksRepo, inferenceRepo, path.dirname(path.resolve(outputPath))]", "forbiddenRoots: []")],
    ["duplicate flags", (text) => text.replace("indexes.length > 1", "false")],
    ["missing flag value", (text) => text.replace('candidate.startsWith("--")', "false")],
    ["ambiguous provider modes", (text) => text.replace("Boolean(inline) === Boolean(file)", "false")],
    ["provider command disconnected", (text) => text.replace("      providerCommand,\n      sceneWorksRepo,", "      providerCommand: [],\n      sceneWorksRepo,")],
    ["provider command substituted", (text) => text.replace("      providerCommand,\n      sceneWorksRepo,", "      providerCommand: [process.execPath],\n      sceneWorksRepo,")],
  ];
  for (const [label, mutate] of mutations) {
    assert.throws(() => assertProviderTransportSourceContract(mutate(source)), undefined, label);
  }
});

test("the source contract kills link-metadata, escape, duplicate, hash, and early-receipt mutations", async () => {
  const source = await readFile(INVENTORY_URL, "utf8");
  const mutations = [
    ["POSIX absolute target", (text) => text.replace("path.posix.isAbsolute(target)", "false")],
    ["Windows absolute target", (text) => text.replace("path.win32.isAbsolute(target)", "false")],
    ["drive-qualified relative target", (text) => text.replace("/^[A-Za-z]:/.test(target)", "false")],
    ["followed enumeration metadata", (text) => text.replace("await lstat(entryPath)", "await realpath(entryPath)")],
    ["unread raw link", (text) => text.replace("await readlink(logicalPath)", '"unchecked"')],
    ["wrong lexical target", (text) => text.replace('path.join(repositoryRoot, "blobs", expectedObject)', "logicalPath")],
    ["lexical equality bypass", (text) => text.replace("samePath(actualLexicalTarget, expectedLexicalTarget)", "true")],
    ["followed target metadata", (text) => text.replace("await lstat(lexicalTarget)", "await stat(lexicalTarget)")],
    ["directory target", (text) => text.replace("!targetMetadata.isFile() || targetMetadata.isSymbolicLink()", "false")],
    ["outside blob root", (text) => text.replace("!isInside(canonicalBlobsRoot, physicalPath)", "false")],
    ["nested lookalike blob root", (text) => text.replace("!samePath(path.dirname(physicalPath), canonicalBlobsRoot)", "false")],
    ["wrong content object", (text) => text.replace("path.basename(physicalPath).toLowerCase() !== expectedObject", "false")],
    ["duplicate resolved file", (text) => text.replace("seenPhysicalPaths.has(pathKey(physicalPath))", "false")],
    ["split path stat", (text) => text.replace("await handle.stat()", "await stat(file)")],
    ["split path stream", (text) => text.replace("await handle.read(buffer, 0, buffer.length, streamedBytes)", "await createReadStream(file)")],
    ["unclosed payload", (text) => text.replace("await handle.close()", "return")],
    ["no stream-fstat parity", (text) => text.replace("inspected.streamedBytes !== inspected.metadata.size", "false")],
    ["link metadata bytes", (text) => text.replace("inspected.streamedBytes !== expectedBytes", "logicalMetadata.size !== expectedBytes")],
    ["no content hash authority", (text) => text.replace("expectedObject.length === 64 ? inspected.sha256 : inspected.gitBlob", "expectedObject")],
    ["added split path stream", (text) => `${text}\ncreateReadStream(physicalPath);\n`],
    ["no started receipt", (text) => text.replace('status: "STARTED"', 'status: "UNKNOWN"')],
    ["no failure receipt", (text) => text.replace('status: "FAIL"', 'status: "UNKNOWN"')],
  ];
  for (const [label, mutate] of mutations) {
    assert.throws(() => assertInventorySourceContract(mutate(source)), undefined, label);
  }
});
