#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CALIBRATION_SCHEMA = JSON.parse(
  readFileSync(path.join(ROOT, "packages/schemas/memory-calibration.schema.json"), "utf8"),
);
export const HARNESS_VERSION = "sceneworks-memory-v4";
export const REQUIRED_SCENARIOS = [
  "exact_fit", "unknown_budget", "stale_evidence", "warm_repeat",
  "cancel", "error", "loadability", "overlay",
];
const RUNGS = [
  "resident", "staged_residency", "bounded_decode",
  "bounded_attention", "bounded_transformer_residency",
];
const RUNG_SET = new Set(RUNGS);
export const RUNG_REUSE_TOLERANCE = Object.freeze({
  absoluteBytes: 256 * 1024 * 1024,
  relative: 0.05,
});

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  return value;
}
export function canonicalJson(value) {
  return `${JSON.stringify(stable(value), null, 2)}\n`;
}
function digest(value) {
  return createHash("sha256").update(JSON.stringify(stable(value))).digest("hex");
}
function fail(message) {
  throw new Error(`memory-strategy calibration: ${message}`);
}
function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
}
function text(value, label) {
  if (typeof value !== "string" || !value.trim()) fail(`${label} must be a non-empty string`);
}
function number(value, label, positive = false) {
  if (!Number.isFinite(value) || (positive ? value <= 0 : value < 0)) {
    fail(`${label} must be a ${positive ? "positive" : "nonnegative"} finite number`);
  }
}
function equal(left, right) {
  return JSON.stringify(stable(left)) === JSON.stringify(stable(right));
}

function schemaTypeMatches(value, type) {
  if (type === "null") return value === null;
  if (type === "array") return Array.isArray(value);
  if (type === "object") return value !== null && typeof value === "object" && !Array.isArray(value);
  if (type === "integer") return Number.isInteger(value);
  if (type === "number") return typeof value === "number" && Number.isFinite(value);
  return typeof value === type;
}

function resolveLocalRef(root, reference) {
  if (!reference.startsWith("#/")) fail(`runtime schema validator supports only local refs, got ${reference}`);
  return reference
    .slice(2)
    .split("/")
    .reduce((value, token) => value[token.replaceAll("~1", "/").replaceAll("~0", "~")], root);
}

function schemaErrors(value, schema, root = schema, location = "$") {
  if (schema.$ref) return schemaErrors(value, resolveLocalRef(root, schema.$ref), root, location);
  const errors = [];
  if (schema.allOf) {
    for (const part of schema.allOf) errors.push(...schemaErrors(value, part, root, location));
  }
  if (schema.oneOf) {
    const matching = schema.oneOf.filter((part) => schemaErrors(value, part, root, location).length === 0);
    if (matching.length !== 1) errors.push(`${location}: expected exactly one schema branch, matched ${matching.length}`);
  }
  if (schema.if && schemaErrors(value, schema.if, root, location).length === 0 && schema.then) {
    errors.push(...schemaErrors(value, schema.then, root, location));
  }
  if (schema.const !== undefined && !equal(value, schema.const)) errors.push(`${location}: value does not equal const`);
  if (schema.enum && !schema.enum.some((candidate) => equal(value, candidate))) errors.push(`${location}: value is outside enum`);
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((type) => schemaTypeMatches(value, type))) {
      errors.push(`${location}: expected type ${types.join("|")}`);
      return errors;
    }
  }
  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) errors.push(`${location}: string is too short`);
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) errors.push(`${location}: string does not match pattern`);
  }
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${location}: number is below minimum`);
    if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) {
      errors.push(`${location}: number is below exclusive minimum`);
    }
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${location}: array has too few items`);
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${location}: array has too many items`);
    if (schema.uniqueItems && new Set(value.map((item) => JSON.stringify(stable(item)))).size !== value.length) {
      errors.push(`${location}: array items must be unique`);
    }
    if (schema.items) value.forEach((item, index) => errors.push(...schemaErrors(item, schema.items, root, `${location}[${index}]`)));
    if (schema.contains && !value.some((item, index) => schemaErrors(item, schema.contains, root, `${location}[${index}]`).length === 0)) {
      errors.push(`${location}: array does not contain required item`);
    }
  }
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const required of schema.required ?? []) {
      if (!Object.hasOwn(value, required)) errors.push(`${location}.${required}: required property is missing`);
    }
    const properties = schema.properties ?? {};
    for (const [key, child] of Object.entries(value)) {
      if (properties[key]) errors.push(...schemaErrors(child, properties[key], root, `${location}.${key}`));
      else if (schema.additionalProperties === false) errors.push(`${location}.${key}: unexpected property`);
    }
  }
  return errors;
}

function validateSchema(value) {
  const errors = schemaErrors(value, CALIBRATION_SCHEMA);
  if (errors.length) fail(`schema validation failed: ${errors.slice(0, 8).join("; ")}`);
}

export function logicalCaseId(spec) {
  return `implan-${digest({
    harnessVersion: HARNESS_VERSION,
    evidenceScope: spec.evidenceScope,
    backend: spec.backend,
    target: spec.target,
    strategy: spec.strategy,
    calibrationFingerprint: spec.calibrationFingerprint,
    fixture: spec.fixture,
    negative: spec.negative === true || spec.status === "negative_complete",
  }).slice(0, 20)}`;
}

export function recordId(record) {
  return `imc-${digest({
    harnessVersion: record.harnessVersion,
    evidenceScope: record.evidenceScope,
    repositories: record.repositories,
    backend: record.backend,
    hardware: record.hardware,
    artifact: record.artifact,
    target: record.target,
    strategy: record.strategy,
    calibrationFingerprint: record.calibrationFingerprint,
    fixture: record.fixture,
  }).slice(0, 20)}`;
}

function validateEngagedRungs(strategy, label) {
  const engaged = strategy.engagedRungs;
  if (!Array.isArray(engaged) || engaged.length === 0) {
    fail(`${label}.engagedRungs must be a non-empty canonical rung set`);
  }
  if (engaged.some((rung) => !RUNG_SET.has(rung))) {
    fail(`${label}.engagedRungs contains an invalid rung`);
  }
  const canonical = RUNGS.filter((rung) => engaged.includes(rung));
  if (!equal(engaged, canonical)) {
    fail(`${label}.engagedRungs must be unique and in canonical rung order`);
  }
  if (!engaged.includes("resident") || !engaged.includes(strategy.rung)) {
    fail(`${label}.engagedRungs must include resident and the selected rung`);
  }
}

function validateRepositories(record) {
  object(record.repositories, `${record.id}.repositories`);
  for (const name of ["sceneWorks", "inference"]) {
    const repo = record.repositories[name];
    object(repo, `${record.id}.repositories.${name}`);
    if (!/^[0-9a-f]{7,40}$/.test(repo.revision)) fail(`${record.id}: ${name} revision must be a git SHA`);
    if (typeof repo.dirty !== "boolean") fail(`${record.id}: ${name}.dirty must be boolean`);
  }
  text(record.repositories.sceneWorks.matrixSourceRevision, `${record.id}.repositories.sceneWorks.matrixSourceRevision`);
}

function validateHardware(record) {
  const hardware = record.hardware;
  object(hardware, `${record.id}.hardware`);
  text(hardware.probe, `${record.id}.hardware.probe`);
  number(hardware.memoryBytes, `${record.id}.hardware.memoryBytes`, true);
  if (record.backend === "candle") {
    for (const key of ["deviceId", "name", "computeCapability", "driverVersion", "runtimeVersion"]) {
      text(hardware[key], `${record.id}.hardware.${key}`);
    }
  } else {
    for (const key of ["model", "chip", "osVersion", "metalDevice"]) {
      text(hardware[key], `${record.id}.hardware.${key}`);
    }
    number(hardware.mlxMemoryLimitBytes, `${record.id}.hardware.mlxMemoryLimitBytes`, true);
    number(hardware.wiredLimitBytes, `${record.id}.hardware.wiredLimitBytes`, true);
  }
}

function validatePhaseMetrics(metrics, label) {
  object(metrics, label);
  for (const phase of ["conditioning", "denoise", "decode", "overall"]) {
    const values = metrics[phase];
    object(values, `${label}.${phase}`);
    for (const metric of ["activeBytes", "allocatorBytes", "deviceBytes", "wiredBytes", "reclaimableBytes"]) {
      number(values[metric], `${label}.${phase}.${metric}`);
    }
    if (values.allocatorBytes < values.activeBytes || values.deviceBytes < values.activeBytes) {
      fail(`${label}.${phase}: allocator/device must cover active bytes`);
    }
    if (values.wiredBytes < values.activeBytes || values.reclaimableBytes > values.allocatorBytes) {
      fail(`${label}.${phase}: wired must cover active and reclaimable cannot exceed allocator bytes`);
    }
  }
  for (const metric of ["activeBytes", "allocatorBytes", "deviceBytes", "wiredBytes", "reclaimableBytes"]) {
    const phaseMax = Math.max(metrics.conditioning[metric], metrics.denoise[metric], metrics.decode[metric]);
    if (metrics.overall[metric] < phaseMax) fail(`${label}.overall.${metric} must cover phase peaks`);
  }
}

function validatePredicted(predicted, label) {
  object(predicted, label);
  for (const phase of ["conditioning", "denoise", "decode", "overall"]) number(predicted[phase], `${label}.${phase}`);
  if (predicted.overall < Math.max(predicted.conditioning, predicted.denoise, predicted.decode)) {
    fail(`${label}.overall must cover phase peaks`);
  }
}

function validateComplete(record) {
  if (record.repositories.sceneWorks.dirty || record.repositories.inference.dirty) {
    fail(`${record.id}: complete evidence cannot come from a dirty repository`);
  }
  validatePredicted(record.predictedPeakBytes, `${record.id}.predictedPeakBytes`);
  validatePhaseMetrics(record.observedMemory, `${record.id}.observedMemory`);
  object(record.sweep, `${record.id}.sweep`);
  if (!Array.isArray(record.sweep.axes) || record.sweep.axes.length === 0) {
    fail(`${record.id}: sweep axes must not be empty`);
  }
  const axisNames = record.sweep.axes.map((axis) => axis.parameter);
  if (new Set(axisNames).size !== axisNames.length) fail(`${record.id}: sweep axes must be unique`);
  for (const axis of record.sweep.axes) {
    text(axis.parameter, `${record.id}.sweep.axis.parameter`);
    if (
      !Array.isArray(axis.testedValues) ||
      axis.testedValues.length < 1 ||
      new Set(axis.testedValues).size !== axis.testedValues.length
    ) fail(`${record.id}: ${axis.parameter} tested values must be nonempty and unique`);
  }
  if (record.sweep.rangeVerified !== true) fail(`${record.id}: complete evidence must verify its range`);
  if (!Array.isArray(record.sweep.cases) || record.sweep.cases.length < 1) {
    fail(`${record.id}: complete evidence needs at least one executed case`);
  }
  const passedCases = record.sweep.cases.filter((item) => item.result === "passed");
  const caseKeys = record.sweep.cases.map((item) => JSON.stringify(stable(item.parameters)));
  if (new Set(caseKeys).size !== caseKeys.length) fail(`${record.id}: executed sweep cases must be unique`);
  if (!passedCases.some((item) => equal(item.parameters, record.strategy.parameters))) {
    fail(`${record.id}: exact strategy parameters were not a passed executed case`);
  }
  for (const axis of record.sweep.axes) {
    const actual = [...new Set(passedCases.map((item) => item.parameters[axis.parameter]))].sort();
    const declared = [...axis.testedValues].sort();
    if (!equal(actual, declared) || actual.length < 1) {
      fail(`${record.id}: ${axis.parameter} range is not derived from passed executed cases`);
    }
  }
  const scenarios = new Map(record.scenarios.map((item) => [item.name, item]));
  if (record.scenarios.length !== REQUIRED_SCENARIOS.length || scenarios.size !== REQUIRED_SCENARIOS.length) {
    fail(`${record.id}: scenarios must be unique`);
  }
  for (const name of REQUIRED_SCENARIOS) {
    const scenario = scenarios.get(name);
    if (!scenario) fail(`${record.id}: missing ${name} scenario`);
    if (name === "overlay") {
      if (!["passed", "not_applicable"].includes(scenario.result)) fail(`${record.id}: overlay did not pass`);
      if (scenario.result === "not_applicable") text(scenario.reason, `${record.id}.overlay.reason`);
    } else if (scenario.result !== "passed") {
      fail(`${record.id}: ${name} scenario did not pass`);
    }
  }
  const exact = scenarios.get("exact_fit");
  number(exact.predictedBytes, `${record.id}.exact_fit.predictedBytes`);
  number(exact.effectiveBudgetBytes, `${record.id}.exact_fit.effectiveBudgetBytes`);
  if (exact.predictedBytes !== exact.effectiveBudgetBytes) fail(`${record.id}: exact_fit must exercise equality`);
  for (const name of ["cancel", "error"]) {
    if (scenarios.get(name).cleanupVerified !== true || scenarios.get(name).warmFollowUpPassed !== true) {
      fail(`${record.id}: ${name} must clean up and pass a warm follow-up`);
    }
  }
  if (record.quality.result !== "passed" || record.quality.identicalLatents !== true) {
    fail(`${record.id}: complete quality evidence must pass with identical latents`);
  }
  text(record.quality.contract, `${record.id}.quality.contract`);
  for (const metric of ["maximumError", "meanError", "maximumErrorThreshold", "meanErrorThreshold"]) {
    number(record.quality[metric], `${record.id}.quality.${metric}`);
  }
  if (
    record.quality.maximumError > record.quality.maximumErrorThreshold ||
    record.quality.meanError > record.quality.meanErrorThreshold
  ) fail(`${record.id}: quality threshold exceeded`);
  const mutation = record.negativeMutation;
  object(mutation, `${record.id}.negativeMutation`);
  if (mutation.result !== "failed_as_expected" || !mutation.measured) {
    fail(`${record.id}: negative mutation must be measured and fail as expected`);
  }
  number(mutation.maximumError, `${record.id}.negativeMutation.maximumError`);
  number(mutation.meanError, `${record.id}.negativeMutation.meanError`);
  if (
    mutation.maximumError <= record.quality.maximumErrorThreshold &&
    mutation.meanError <= record.quality.meanErrorThreshold
  ) fail(`${record.id}: negative mutation did not breach a threshold`);
  if (record.loadability.result !== "passed") fail(`${record.id}: loadability did not pass`);
  text(record.loadability.resolvedPathFingerprint, `${record.id}.loadability.resolvedPathFingerprint`);
  if (record.observedMemory.overall.deviceBytes > record.hardware.memoryBytes) {
    fail(`${record.id}: overall device bytes exceed probed hardware memory`);
  }
  if (record.backend === "mlx" && record.observedMemory.overall.wiredBytes > record.hardware.wiredLimitBytes) {
    fail(`${record.id}: overall wired bytes exceed the probed wired ceiling`);
  }
}

function validateNegative(record) {
  const mutation = record.negativeMutation;
  object(mutation, `${record.id}.negativeMutation`);
  if (mutation.result !== "failed_as_expected" || mutation.measured !== true) {
    fail(`${record.id}: negative case was not a measured expected failure`);
  }
  if (!equal(mutation.parameters, record.strategy.parameters)) {
    fail(`${record.id}: negative mutation parameters do not match the planned strategy parameters`);
  }
  text(record.quality.contract, `${record.id}.quality.contract`);
  for (const metric of ["maximumErrorThreshold", "meanErrorThreshold"]) {
    number(record.quality[metric], `${record.id}.quality.${metric}`);
  }
  number(mutation.maximumError, `${record.id}.negativeMutation.maximumError`);
  number(mutation.meanError, `${record.id}.negativeMutation.meanError`);
  if (
    mutation.maximumError <= record.quality.maximumErrorThreshold &&
    mutation.meanError <= record.quality.meanErrorThreshold
  ) fail(`${record.id}: negative case did not breach a threshold`);
}

export function validateRecord(record) {
  object(record, "record");
  text(record.id, "record.id");
  text(record.logicalCaseId, `${record.id}.logicalCaseId`);
  if (!["complete", "gated", "negative_complete"].includes(record.status)) fail(`${record.id}: invalid status`);
  if (!["authoritative", "candidate", "fixture"].includes(record.evidenceScope)) {
    fail(`${record.id}: invalid evidenceScope`);
  }
  if (!["mlx", "candle"].includes(record.backend)) fail(`${record.id}: invalid backend`);
  validateRepositories(record);
  validateHardware(record);
  object(record.artifact, `${record.id}.artifact`);
  for (const key of ["repository", "resolvedRevision", "variant"]) text(record.artifact[key], `${record.id}.artifact.${key}`);
  object(record.target, `${record.id}.target`);
  for (const key of ["modelId", "provider", "tier", "mode", "overlay"]) text(record.target[key], `${record.id}.target.${key}`);
  for (const key of ["width", "height", "batch", "frames"]) number(record.target.geometry[key], `${record.id}.target.geometry.${key}`, true);
  object(record.strategy, `${record.id}.strategy`);
  if (!RUNG_SET.has(record.strategy.rung)) fail(`${record.id}: invalid rung`);
  validateEngagedRungs(record.strategy, `${record.id}.strategy`);
  object(record.strategy.parameters, `${record.id}.strategy.parameters`);
  text(record.fixture, `${record.id}.fixture`);
  text(record.calibrationFingerprint, `${record.id}.calibrationFingerprint`);
  text(record.capturedAt, `${record.id}.capturedAt`);
  if (Number.isNaN(Date.parse(record.capturedAt))) fail(`${record.id}: invalid capturedAt`);
  if (record.harnessVersion !== HARNESS_VERSION) fail(`${record.id}: invalid harnessVersion`);
  if (record.logicalCaseId !== logicalCaseId(record)) fail(`${record.id}: logical identity mismatch`);
  if (record.id !== recordId(record)) fail(`${record.id}: deterministic identity mismatch`);
  object(record.loadability, `${record.id}.loadability`);
  object(record.quality, `${record.id}.quality`);
  if (!Array.isArray(record.scenarios)) fail(`${record.id}: scenarios must be an array`);
  if (record.status === "complete") validateComplete(record);
  if (record.status === "negative_complete") validateNegative(record);
  return record;
}

export function validateBundle(bundle) {
  validateSchema(bundle);
  object(bundle, "bundle");
  if (bundle.schemaVersion !== 3 || bundle.harnessVersion !== HARNESS_VERSION || !Array.isArray(bundle.records)) {
    fail("invalid bundle envelope");
  }
  const ids = new Set();
  for (const record of bundle.records) {
    validateRecord(record);
    if (ids.has(record.id)) fail(`duplicate record ${record.id}`);
    ids.add(record.id);
  }
  return bundle;
}

export function evidenceSemantics(record, revisions) {
  validateRecord(record);
  if (record.evidenceScope === "fixture") return "fixture";
  if (record.evidenceScope === "candidate") return "candidate";
  if (record.status === "negative_complete") return "negative";
  if (record.status !== "complete") return "gated";
  // SceneWorks invalidation is owned by calibrationBinding's provider ABI fingerprint. The exact
  // matrixSourceRevision remains captured provenance, but treating it as a second invalidation gate
  // would stale every measurement on comments, formatting, or unrelated source edits.
  return record.repositories.inference.revision === revisions.inference
    ? "current" : "historical";
}

export function mergeBundles(left, right) {
  validateBundle(left);
  validateBundle(right);
  const records = new Map(left.records.map((record) => [record.id, record]));
  for (const record of right.records) {
    const existing = records.get(record.id);
    if (existing && !equal(existing, record)) fail(`conflicting record with exact identity ${record.id}`);
    records.set(record.id, record);
  }
  return { schemaVersion: 3, harnessVersion: HARNESS_VERSION, records: [...records.values()].sort((a, b) => a.id.localeCompare(b.id)) };
}

export function compareRungReuse(fresh, reused, tolerance = RUNG_REUSE_TOLERANCE) {
  validateBundle(fresh);
  validateBundle(reused);
  const freshByLogicalId = new Map(fresh.records.map((record) => [record.logicalCaseId, record]));
  const reusedByLogicalId = new Map(reused.records.map((record) => [record.logicalCaseId, record]));
  if (freshByLogicalId.size !== fresh.records.length || reusedByLogicalId.size !== reused.records.length) {
    fail("fresh/reused comparison cannot contain duplicate logical cases");
  }
  if (freshByLogicalId.size !== reusedByLogicalId.size) {
    fail(`fresh/reused comparison cardinality differs: ${freshByLogicalId.size} != ${reusedByLogicalId.size}`);
  }
  const comparisons = [];
  for (const [logicalCaseId, freshRecord] of freshByLogicalId) {
    const reusedRecord = reusedByLogicalId.get(logicalCaseId);
    if (!reusedRecord) fail(`reused capture is missing ${logicalCaseId}`);
    if (freshRecord.id !== reusedRecord.id) {
      fail(`${logicalCaseId}: fresh/reused comparison domain differs in repository, hardware, or artifact provenance`);
    }
    if (!freshRecord.observedMemory || !reusedRecord.observedMemory) {
      fail(`${logicalCaseId}: fresh/reused comparison requires observedMemory on both records`);
    }
    const metrics = [];
    for (const phase of ["conditioning", "denoise", "decode", "overall"]) {
      for (const metric of ["activeBytes", "allocatorBytes", "deviceBytes", "wiredBytes", "reclaimableBytes"]) {
        const freshBytes = freshRecord.observedMemory[phase][metric];
        const reusedBytes = reusedRecord.observedMemory[phase][metric];
        const differenceBytes = Math.abs(reusedBytes - freshBytes);
        const allowedBytes = Math.max(tolerance.absoluteBytes, Math.ceil(freshBytes * tolerance.relative));
        metrics.push({ phase, metric, freshBytes, reusedBytes, differenceBytes, allowedBytes,
          passed: differenceBytes <= allowedBytes });
      }
    }
    comparisons.push({
      logicalCaseId,
      rung: freshRecord.strategy.rung,
      passed: metrics.every((metric) => metric.passed),
      metrics,
    });
  }
  const backend = fresh.records[0]?.backend;
  if (
    !backend ||
    fresh.records.some((record) => record.backend !== backend) ||
    reused.records.some((record) => record.backend !== backend)
  ) {
    fail("fresh/reused comparison must contain exactly one backend");
  }
  return {
    schemaVersion: 1,
    backend,
    tolerance,
    verdict: comparisons.every((comparison) => comparison.passed)
      ? "amortizable"
      : "unable_to_amortize",
    comparisons,
  };
}

function completedLogicalIds(record) {
  if (record.status === "negative_complete") return [record.logicalCaseId];
  if (record.status !== "complete") return [];
  return record.sweep.cases.filter((item) => item.result === "passed").map((item) =>
    logicalCaseId({
      evidenceScope: record.evidenceScope,
      backend: record.backend,
      target: record.target,
      strategy: {
        rung: record.strategy.rung,
        engagedRungs: record.strategy.engagedRungs,
        parameters: item.parameters,
      },
      calibrationFingerprint: record.calibrationFingerprint,
      fixture: record.fixture,
      negative: item.result === "failed",
    }),
  );
}

export function expandPlan(config, completed = []) {
  object(config, "plan config");
  const completedLogical = new Set(
    completed.flatMap(completedLogicalIds),
  );
  const cases = [];
  for (const provider of config.providers) {
    for (const candidate of provider.cases) {
      if (!["passed", "failed"].includes(candidate.expectedResult)) {
        fail(`${provider.name ?? provider.target.provider}: plan case requires expectedResult`);
      }
      if ((candidate.expectedResult === "failed") !== (candidate.negative === true)) {
        fail(`${provider.name ?? provider.target.provider}: failed cases must be explicitly negative`);
      }
      const spec = {
        evidenceScope: provider.evidenceScope,
        backend: provider.backend,
        target: provider.target,
        strategy: {
          rung: provider.rung,
          engagedRungs: provider.engagedRungs,
          parameters: candidate.parameters,
        },
        calibrationFingerprint: provider.calibrationFingerprint,
        fixture: provider.fixture,
        negative: candidate.negative === true,
      };
      const id = logicalCaseId(spec);
      if (!completedLogical.has(id)) cases.push({
        logicalCaseId: id,
        ...spec,
        expectedResult: candidate.expectedResult,
        modelLoadPolicy: provider.modelLoadPolicy ?? "fresh_per_case",
        modelLoadGroup: provider.modelLoadGroup ?? null,
      });
    }
  }
  return cases.sort((a, b) => a.logicalCaseId.localeCompare(b.logicalCaseId));
}

export async function assessProviderReuse({ config, providerCommand, backend, fixture }) {
  if (!Array.isArray(providerCommand) || !providerCommand.length) fail("provider command must be a JSON argv array");
  const planned = expandPlan(config).filter(
    (candidate) => candidate.backend === backend && (!fixture || candidate.fixture === fixture),
  ).sort((left, right) => RUNGS.indexOf(left.strategy.rung) - RUNGS.indexOf(right.strategy.rung));
  if (planned.length === 0) fail(`reuse assessment selected no ${backend} cases`);
  const response = JSON.parse(await execute(
    providerCommand[0],
    providerCommand.slice(1),
    canonicalJson({ action: "assess_batch", planned }),
  ));
  if (!["eligible_for_measurement", "unable_to_amortize"].includes(response.verdict)) {
    fail(`provider returned invalid reuse-assessment verdict ${JSON.stringify(response.verdict)}`);
  }
  text(response.reason, "reuse assessment reason");
  return {
    schemaVersion: 1,
    backend,
    fixture: fixture ?? null,
    tolerance: RUNG_REUSE_TOLERANCE,
    ...response,
  };
}

function execute(command, args, input) {
  return new Promise((resolve, reject) => {
    const hasInput = input !== undefined;
    const child = spawn(command, args, {
      stdio: [hasInput ? "pipe" : "ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    let spawnError;
    let stdinError;
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", (error) => (spawnError = error));
    child.stdin?.on("error", (error) => (stdinError = error));
    child.on("close", (code) => {
      if (spawnError) return reject(new Error(`could not start ${command}: ${spawnError.message}`));
      if (stdinError) {
        return reject(
          new Error(
            `${command} closed its provider-protocol stdin before consuming the JSON request: ${stdinError.message}`,
          ),
        );
      }
      return code === 0
        ? resolve(stdout)
        : reject(new Error(`${command} exited ${code}: ${stderr.trim() || "no stderr"}`));
    });
    if (hasInput) child.stdin.end(input);
  });
}

export async function runProviderPlan({
  config, providerCommand, sceneWorksRepo, inferenceRepo, resume, backend, providerName, fixture,
  onProviderInvocation, forceFreshPerCase = false, forceBatchRungs = false,
}) {
  if (!Array.isArray(providerCommand) || !providerCommand.length) fail("provider command must be a JSON argv array");
  const gitState = async (repo, sceneWorks = false) => ({
    revision: (await execute("git", ["-C", repo, "rev-parse", "HEAD"])).trim(),
    dirty: Boolean((await execute("git", ["-C", repo, "status", "--porcelain"])).trim()),
    ...(sceneWorks
      ? {
          matrixSourceRevision: JSON.parse(
            await readFile(path.join(repo, "docs/generated/memory-matrix.json"), "utf8"),
          ).generatedFrom.sceneWorksRevision,
        }
      : {}),
  });
  const probeRepositories = async () => ({
    sceneWorks: await gitState(sceneWorksRepo, true),
    inference: await gitState(inferenceRepo),
  });
  const repositories = await probeRepositories();
  const assertRepositoriesStable = async () => {
    const after = await probeRepositories();
    if (!equal(repositories, after)) fail("repository HEAD or dirty state changed during provider execution");
  };
  const existing = resume ? validateBundle(resume) : { schemaVersion: 3, harnessVersion: HARNESS_VERSION, records: [] };
  const selectedConfig = {
    ...config,
    providers: config.providers.filter(
      (provider) => (!providerName || provider.name === providerName) && (!fixture || provider.fixture === fixture),
    ),
  };
  if (providerName && selectedConfig.providers.length === 0) {
    fail(`provider run selected no plan provider named ${providerName}`);
  }
  if (fixture && selectedConfig.providers.length === 0) {
    fail(`provider run selected no plan provider with fixture ${fixture}`);
  }
  let expanded = expandPlan(selectedConfig, existing.records);
  if (forceFreshPerCase && forceBatchRungs) fail("cannot force both fresh and batched provider execution");
  if (forceFreshPerCase) {
    expanded = expanded.map((planned) => ({ ...planned, modelLoadPolicy: "fresh_per_case", modelLoadGroup: null }));
  }
  if (forceBatchRungs) {
    expanded = expanded.map((planned) => ({
      ...planned,
      modelLoadPolicy: "batch_rungs",
      modelLoadGroup: `forced-${digest({ backend: planned.backend, target: planned.target, fixture: planned.fixture })}`,
    }));
  }
  const cases = backend ? expanded.filter((planned) => planned.backend === backend) : expanded;
  if (cases.length === 0) fail(`provider run selected no ${backend ?? "remaining"} cases`);
  const backends = new Set(cases.map((planned) => planned.backend));
  if (backends.size !== 1) {
    fail(`provider run must select exactly one backend; pass --backend mlx|candle (selected: ${[...backends].join(", ")})`);
  }
  const probe = JSON.parse(await execute(
    providerCommand[0],
    providerCommand.slice(1),
    canonicalJson({ action: "probe", repositories }),
  ));
  await assertRepositoriesStable();
  const incoming = [];
  let remaining = cases;
  const sameBatch = (left, right) =>
    left.modelLoadPolicy === "batch_rungs" &&
    right.modelLoadPolicy === "batch_rungs" &&
    left.modelLoadGroup &&
    left.modelLoadGroup === right.modelLoadGroup &&
    left.backend === right.backend &&
    equal(left.target, right.target) &&
    left.fixture === right.fixture;
  while (remaining.length > 0) {
    const first = remaining[0];
    const batchRungs = new Set();
    const invocation = first.modelLoadPolicy === "batch_rungs"
      ? remaining
          .filter((planned) => sameBatch(first, planned))
          .sort((left, right) => RUNGS.indexOf(left.strategy.rung) - RUNGS.indexOf(right.strategy.rung))
          .filter((planned) => {
            if (batchRungs.has(planned.strategy.rung)) return false;
            batchRungs.add(planned.strategy.rung);
            return true;
          })
      : [first];
    const uniqueRungs = new Set(invocation.map((planned) => planned.strategy.rung));
    if (first.modelLoadPolicy === "batch_rungs" && uniqueRungs.size !== invocation.length) {
      fail(`${first.modelLoadGroup}: a rung batch may contain only one pending case per rung`);
    }
    const action = invocation.length > 1 ? "run_batch" : "run";
    onProviderInvocation?.({ action, cases: invocation });
    const providerOutput = await execute(
      providerCommand[0],
      providerCommand.slice(1),
      canonicalJson({
        action,
        ...(action === "run" ? { planned: first } : { planned: invocation }),
        repositories,
        repositoryPaths: { sceneWorks: sceneWorksRepo, inference: inferenceRepo },
        hardware: probe.hardware,
      }),
    );
    await assertRepositoriesStable();
    const response = JSON.parse(providerOutput);
    const fragments = action === "run_batch" ? response.fragments : [response];
    if (!Array.isArray(fragments) || fragments.length !== invocation.length) {
      fail(`${first.modelLoadGroup ?? first.logicalCaseId}: provider returned ${
        Array.isArray(fragments) ? fragments.length : "a non-array"
      } fragments for ${invocation.length} planned cases`);
    }
    if (action === "run_batch" && response.modelLoads !== 1) {
      fail(`${first.modelLoadGroup}: batched provider must attest exactly one model load`);
    }
    for (const [index, fragment] of fragments.entries()) {
      const planned = invocation[index];
      if (!fragment.strategy || typeof fragment.strategy !== "object" || Array.isArray(fragment.strategy)) {
        fail(`${planned.logicalCaseId}: provider fragment.strategy must attest the executed strategy`);
      }
      validateEngagedRungs(fragment.strategy, `${planned.logicalCaseId}.provider strategy`);
      if (!equal(fragment.strategy, planned.strategy)) {
        fail(`${planned.logicalCaseId}: adapter measured strategy does not match planned strategy`);
      }
      const record = {
        ...fragment,
        logicalCaseId: planned.logicalCaseId,
        evidenceScope: planned.evidenceScope,
        backend: planned.backend,
        repositories,
        hardware: probe.hardware,
        target: planned.target,
        strategy: fragment.strategy,
        calibrationFingerprint: planned.calibrationFingerprint,
        fixture: planned.fixture,
        harnessVersion: HARNESS_VERSION,
      };
      if (planned.expectedResult === "failed" && record.status !== "negative_complete") {
        fail(`${planned.logicalCaseId}: negative plan case must return status=negative_complete`);
      }
      if (planned.expectedResult === "passed" && record.status === "negative_complete") {
        fail(`${planned.logicalCaseId}: passing plan case returned a negative result`);
      }
      record.id = recordId(record);
      validateRecord(record);
      incoming.push(record);
    }
    const completed = new Set([...existing.records, ...incoming].flatMap(completedLogicalIds));
    const invoked = new Set(invocation.map((planned) => planned.logicalCaseId));
    remaining = remaining.filter(
      (planned) => !invoked.has(planned.logicalCaseId) && !completed.has(planned.logicalCaseId),
    );
  }
  return mergeBundles(existing, { schemaVersion: 3, harnessVersion: HARNESS_VERSION, records: incoming });
}

async function readJson(file) {
  return JSON.parse(await readFile(path.resolve(ROOT, file), "utf8"));
}
async function atomicWrite(file, value) {
  const destination = path.resolve(ROOT, file);
  const temporary = `${destination}.tmp-${process.pid}`;
  await writeFile(temporary, canonicalJson(value));
  await rename(temporary, destination);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  const value = (flag) => {
    const index = args.indexOf(flag);
    return index < 0 ? undefined : args[index + 1];
  };
  if (command === "check") return void validateBundle(await readJson(value("--input")));
  if (command === "plan") {
    const resume = value("--resume") ? validateBundle(await readJson(value("--resume"))).records : [];
    return void await atomicWrite(value("--output"), { harnessVersion: HARNESS_VERSION, cases: expandPlan(await readJson(value("--config")), resume) });
  }
  if (command === "ingest") {
    const incoming = validateBundle(await readJson(value("--input")));
    const output = value("--resume") ? mergeBundles(validateBundle(await readJson(value("--resume"))), incoming) : incoming;
    return void await atomicWrite(value("--output"), output);
  }
  if (command === "compare-reuse") {
    return void await atomicWrite(
      value("--output"),
      compareRungReuse(await readJson(value("--fresh")), await readJson(value("--reused"))),
    );
  }
  if (command === "assess-reuse") {
    return void await atomicWrite(value("--output"), await assessProviderReuse({
      config: await readJson(value("--config")),
      providerCommand: JSON.parse(value("--provider-command")),
      backend: value("--backend"),
      fixture: value("--fixture"),
    }));
  }
  if (command === "run") {
    const output = await runProviderPlan({
      config: await readJson(value("--config")),
      providerCommand: JSON.parse(value("--provider-command")),
      sceneWorksRepo: path.resolve(value("--sceneworks-repo")),
      inferenceRepo: path.resolve(value("--inference-repo")),
      resume: value("--resume") ? await readJson(value("--resume")) : undefined,
      backend: value("--backend"),
      providerName: value("--provider"),
      fixture: value("--fixture"),
      forceFreshPerCase: args.includes("--fresh-per-case"),
      forceBatchRungs: args.includes("--batch-rungs"),
    });
    return void await atomicWrite(value("--output"), output);
  }
  fail("usage: check|plan|ingest|assess-reuse|compare-reuse|run (see docs/memory-calibration-harness.md)");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
