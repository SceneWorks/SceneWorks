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
  readFileSync(path.join(ROOT, "packages/schemas/image-memory-calibration.schema.json"), "utf8"),
);
export const HARNESS_VERSION = "sceneworks-image-memory-v2";
export const REQUIRED_SCENARIOS = [
  "exact_fit", "unknown_budget", "stale_evidence", "warm_repeat",
  "cancel", "error", "loadability", "overlay",
];
const RUNGS = new Set([
  "resident", "staged_residency", "bounded_decode",
  "bounded_attention", "bounded_transformer_residency",
]);

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
  throw new Error(`image-memory calibration: ${message}`);
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
      axis.testedValues.length < 2 ||
      new Set(axis.testedValues).size !== axis.testedValues.length
    ) fail(`${record.id}: ${axis.parameter} tested values must be nonempty and unique`);
  }
  if (record.sweep.rangeVerified !== true) fail(`${record.id}: complete evidence must verify its range`);
  if (!Array.isArray(record.sweep.cases) || record.sweep.cases.length < 2) {
    fail(`${record.id}: complete range evidence needs at least two executed cases`);
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
    if (!equal(actual, declared) || actual.length < 2) {
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
  if (!["authoritative", "fixture"].includes(record.evidenceScope)) fail(`${record.id}: invalid evidenceScope`);
  if (!["mlx", "candle"].includes(record.backend)) fail(`${record.id}: invalid backend`);
  validateRepositories(record);
  validateHardware(record);
  object(record.artifact, `${record.id}.artifact`);
  for (const key of ["repository", "resolvedRevision", "variant"]) text(record.artifact[key], `${record.id}.artifact.${key}`);
  object(record.target, `${record.id}.target`);
  for (const key of ["modelId", "provider", "tier", "mode", "overlay"]) text(record.target[key], `${record.id}.target.${key}`);
  for (const key of ["width", "height", "batch", "frames"]) number(record.target.geometry[key], `${record.id}.target.geometry.${key}`, true);
  object(record.strategy, `${record.id}.strategy`);
  if (!RUNGS.has(record.strategy.rung)) fail(`${record.id}: invalid rung`);
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
  if (bundle.schemaVersion !== 2 || bundle.harnessVersion !== HARNESS_VERSION || !Array.isArray(bundle.records)) {
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
  if (record.status === "negative_complete") return "negative";
  if (record.status !== "complete") return "gated";
  return record.repositories.sceneWorks.matrixSourceRevision === revisions.sceneWorks &&
    record.repositories.inference.revision === revisions.inference
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
  return { schemaVersion: 2, harnessVersion: HARNESS_VERSION, records: [...records.values()].sort((a, b) => a.id.localeCompare(b.id)) };
}

function completedLogicalIds(record) {
  if (record.status === "negative_complete") return [record.logicalCaseId];
  if (record.status !== "complete") return [];
  return record.sweep.cases.filter((item) => item.result === "passed").map((item) =>
    logicalCaseId({
      evidenceScope: record.evidenceScope,
      backend: record.backend,
      target: record.target,
      strategy: { rung: record.strategy.rung, parameters: item.parameters },
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
        strategy: { rung: provider.rung, parameters: candidate.parameters },
        calibrationFingerprint: provider.calibrationFingerprint,
        fixture: provider.fixture,
        negative: candidate.negative === true,
      };
      const id = logicalCaseId(spec);
      if (!completedLogical.has(id)) cases.push({ logicalCaseId: id, ...spec, expectedResult: candidate.expectedResult });
    }
  }
  return cases.sort((a, b) => a.logicalCaseId.localeCompare(b.logicalCaseId));
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

export async function runProviderPlan({ config, providerCommand, sceneWorksRepo, inferenceRepo, resume }) {
  if (!Array.isArray(providerCommand) || !providerCommand.length) fail("provider command must be a JSON argv array");
  const gitState = async (repo, sceneWorks = false) => ({
    revision: (await execute("git", ["-C", repo, "rev-parse", "HEAD"])).trim(),
    dirty: Boolean((await execute("git", ["-C", repo, "status", "--porcelain"])).trim()),
    ...(sceneWorks
      ? {
          matrixSourceRevision: JSON.parse(
            await readFile(path.join(repo, "docs/generated/image-memory-matrix.json"), "utf8"),
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
  const existing = resume ? validateBundle(resume) : { schemaVersion: 2, harnessVersion: HARNESS_VERSION, records: [] };
  const cases = expandPlan(config, existing.records);
  const probe = JSON.parse(await execute(
    providerCommand[0],
    providerCommand.slice(1),
    canonicalJson({ action: "probe", repositories }),
  ));
  await assertRepositoriesStable();
  const incoming = [];
  for (const planned of cases) {
    const providerOutput = await execute(
      providerCommand[0],
      providerCommand.slice(1),
      canonicalJson({
        action: "run",
        planned,
        repositories,
        repositoryPaths: { sceneWorks: sceneWorksRepo, inference: inferenceRepo },
        hardware: probe.hardware,
      }),
    );
    await assertRepositoriesStable();
    const fragment = JSON.parse(providerOutput);
    const record = {
      ...fragment,
      logicalCaseId: planned.logicalCaseId,
      evidenceScope: planned.evidenceScope,
      backend: planned.backend,
      repositories,
      hardware: probe.hardware,
      target: planned.target,
      strategy: planned.strategy,
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
  return mergeBundles(existing, { schemaVersion: 2, harnessVersion: HARNESS_VERSION, records: incoming });
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
  if (command === "run") {
    const output = await runProviderPlan({
      config: await readJson(value("--config")),
      providerCommand: JSON.parse(value("--provider-command")),
      sceneWorksRepo: path.resolve(value("--sceneworks-repo")),
      inferenceRepo: path.resolve(value("--inference-repo")),
      resume: value("--resume") ? await readJson(value("--resume")) : undefined,
    });
    return void await atomicWrite(value("--output"), output);
  }
  fail("usage: check|plan|ingest|run (see docs/image-memory-calibration-harness.md)");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
