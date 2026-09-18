import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { INFERENCE_SOURCE_REVISION, readPlanAndLock, validateInferenceValidatorSource, validateMetricsLock, validatePlan, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  const workerManifest = readFileSync("crates/sceneworks-worker/Cargo.toml", "utf8");
  const nativePin = workerManifest.match(/sceneworks-gen-core = \{ git = "https:\/\/github.com\/SceneWorks\/inference", rev = "([a-f0-9]{40})"/)[1];
  const validatorSource = validateInferenceValidatorSource(validated.inference_contract);
  assert.equal(validatorSource.revision, nativePin, "campaign follows the actual shipping provider source");
  assert.equal(validated.inference_preflight.head_sha, validatorSource.base_revision);
  assert.equal(validated.inference_preflight.artifact.name,
    `starvector-terminal-preflight-${nativePin}-${validated.inference_preflight.workflow_run_id}-${validated.inference_preflight.workflow_run_attempt}`);
  assert.match((await readPlanAndLock("release/starvector-terminal-campaign-v1.json")).metrics_lock_sha256, /^[0-9a-f]{64}$/);
});

test("terminal campaign rejects count, lock, and LPIPS weight drift", () => {
  const badPlan = structuredClone(plan); badPlan.counts.hostile_sanitizer = 199;
  assert.throws(() => validatePlan(badPlan), /counts/);
  const legacySchema = structuredClone(plan); legacySchema.inference_contract.receipt_schema = "release/starvector-terminal-receipt-v2.schema.json";
  assert.throws(() => validatePlan(legacySchema), /outcome-parity receipt schema/);
  const wrongSchemaHash = structuredClone(plan); wrongSchemaHash.inference_contract.receipt_schema_sha256 = "0".repeat(64);
  assert.throws(() => validatePlan(wrongSchemaHash), /outcome-parity receipt schema/);
  const badLock = structuredClone(lock); badLock.lpips.alexnet_weights_sha256 = "0".repeat(64);
  assert.throws(() => validateMetricsLock(badLock), /LPIPS/);
});

test("terminal campaign seals validator source compatibility separately from historical inference evidence", () => {
  const source = validateInferenceValidatorSource(plan.inference_contract);
  assert.equal(source.revision, INFERENCE_SOURCE_REVISION);
  assert.deepEqual(source.changed_paths, [...source.changed_paths].sort());
  const changed = structuredClone(plan);
  changed.inference_contract.validator_source = {
    base_revision: changed.inference_contract.revision,
    revision: "0".repeat(40),
    changed_paths: ["scripts/release/starvector_terminal_evidence.mjs"],
  };
  assert.throws(() => validatePlan(changed), /validator source compatibility (?:is invalid|changed|is outside)/);
  const unsafe = structuredClone(plan);
  unsafe.inference_contract.validator_source = { ...source, changed_paths: ["../escape"] };
  assert.throws(() => validatePlan(unsafe), /validator source compatibility is invalid/);
});

test("a validator-only profile may reuse the authentic historical preflight capture", () => {
  const validatorSource = {
    base_revision: plan.inference_contract.revision,
    revision: "c58fa52dedba8f3afa38244a5de8aa4fa22813d5",
    changed_paths: [
      "release/starvector-terminal-README.md",
      "release/starvector-terminal-receipt-v2-outcome-parity-limit-behaviors.schema.json",
      "scripts/release/starvector_terminal_evidence.mjs",
      "scripts/tests/starvector_terminal_evidence.test.mjs",
    ],
  };
  const contract = { ...plan.inference_contract, revision: validatorSource.revision, validator_source: validatorSource };
  assert.deepEqual(validateInferenceValidatorSource(contract, validatorSource), validatorSource);
  assert.equal(plan.inference_preflight.head_sha, validatorSource.base_revision);
  const incomplete = { ...validatorSource, changed_paths: validatorSource.changed_paths.slice(1) };
  assert.throws(() => validateInferenceValidatorSource({ ...contract, validator_source: incomplete }, incomplete), /outside the reviewed receipt-tooling closure/);
  const legacyMismatch = structuredClone(plan);
  legacyMismatch.inference_preflight.head_sha = validatorSource.revision;
  assert.throws(() => validatePlan(legacyMismatch), /preflight provenance/);
});

test("plan validates preflight structure while allowing newly recorded exact evidence", () => {
  const next = structuredClone(plan);
  next.inference_preflight.artifact.id += 1;
  next.inference_preflight.artifact.digest = `sha256:${"a".repeat(64)}`;
  assert.equal(validatePlan(next), next);
  for (const change of [p => p.inference_preflight.head_sha = "0".repeat(40), p => p.inference_preflight.artifact.digest = "invalid", p => p.inference_preflight.workflow_run_attempt = 0, p => p.inference_preflight.hook_logs.pop()]) {
    const invalid = structuredClone(plan); change(invalid); assert.throws(() => validatePlan(invalid), /preflight provenance/);
  }
});

test("dispatch identities reject Bash and PowerShell injection-shaped payloads", () => {
  const pin = INFERENCE_SOURCE_REVISION;
  assert.deepEqual(validateTerminalDispatchInputs(plan, pin, "campaign-safe-1"), {
    permanent_pin: pin,
    campaign_run_id: "campaign-safe-1",
  });
  for (const value of [
    "$(touch /tmp/starvector-shell-injection)",
    "campaign'; Write-Output pwned; #",
    'campaign\"; Start-Process calc; #',
    "campaign;rm",
  ]) {
    assert.throws(
      () => validateTerminalDispatchInputs(plan, pin, value),
      /bounded portable identifier/,
      value,
    );
  }
  assert.throws(
    () => validateTerminalDispatchInputs(plan, `$(printf ${pin})`, "campaign"),
    /permanent pin/,
  );
});
