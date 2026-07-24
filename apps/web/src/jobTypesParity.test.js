import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GPU_REQUIRED_JOB_TYPES, NON_GPU_JOB_TYPES } from "./jobTypes.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const CORE_SRC = resolve(HERE, "../../../crates/sceneworks-core/src");
const contracts = readFileSync(resolve(CORE_SRC, "contracts.rs"), "utf8");
const jobsStore = readFileSync(resolve(CORE_SRC, "jobs_store.rs"), "utf8");

function jobTypeNamesByVariant() {
  const block = contracts.match(/pub enum JobType \{([\s\S]*?)\n\s{4}\}\n\}/)?.[1];
  expect(block, "contracts.rs must contain the JobType string enum").toBeTruthy();
  return new Map(
    [...block.matchAll(/^\s*(\w+)\s*=>\s*"([^"]+)",/gm)].map(([, variant, name]) => [
      variant,
      name,
    ]),
  );
}

function backendGpuJobTypes() {
  const body = jobsStore.match(
    /fn job_requires_gpu\(job_type: &JobType\) -> bool \{([\s\S]*?)\n\}/,
  )?.[1];
  expect(body, "jobs_store.rs must contain job_requires_gpu").toBeTruthy();
  const names = jobTypeNamesByVariant();
  return new Set(
    [...body.matchAll(/JobType::(\w+)/g)].map(([, variant]) => {
      expect(names.has(variant), `JobType::${variant} must exist in contracts.rs`).toBe(true);
      return names.get(variant);
    }),
  );
}

function backendNonGpuUtilityJobTypes() {
  const block = jobsStore.match(
    /pub const NON_GPU_JOB_TYPES: &\[&str\] = &\[([\s\S]*?)\n\];/,
  )?.[1];
  expect(block, "jobs_store.rs must contain NON_GPU_JOB_TYPES").toBeTruthy();
  return new Set([...block.matchAll(/"([^"]+)"/g)].map(([, name]) => name));
}

describe("web ↔ backend job placement parity (sc-13631)", () => {
  it("mirrors the backend GPU-required predicate", () => {
    expect([...GPU_REQUIRED_JOB_TYPES].sort()).toEqual([...backendGpuJobTypes()].sort());
  });

  it("mirrors backend utility jobs that ignore GPU placement", () => {
    expect([...NON_GPU_JOB_TYPES].sort()).toEqual(
      [...backendNonGpuUtilityJobTypes()].sort(),
    );
  });

  it("leaves capability-routed audio outside both placement overrides", () => {
    expect(GPU_REQUIRED_JOB_TYPES.has("audio_generate")).toBe(false);
    expect(NON_GPU_JOB_TYPES.has("audio_generate")).toBe(false);
  });
});
