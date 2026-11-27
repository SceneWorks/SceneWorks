#!/usr/bin/env node
// Compare two canonical `sga.capabilities/2` capability exports for pure semantic equality
// (`SW-6983`, `REG-3`, `M8`).
//
// `sga.capabilities/2` documents contain zero git SHA, pin hash, wall-clock timestamp, or host
// metadata — identical semantic capability truth across a pin bump emits identical canonical
// bytes and requires zero artifact churn.

import { isDeepStrictEqual } from "node:util";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const SGA_CAPABILITIES_SCHEMA_VERSION = "sga.capabilities/2";

export function canonicalizeValue(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalizeValue);
  }
  if (value !== null && typeof value === "object") {
    const sorted = {};
    for (const key of Object.keys(value).sort()) {
      sorted[key] = canonicalizeValue(value[key]);
    }
    return sorted;
  }
  return value;
}

export function validateSgaCapabilitiesExport(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must contain a JSON object`);
  }
  if (value.schema_version !== SGA_CAPABILITIES_SCHEMA_VERSION) {
    throw new Error(
      `${label}.schema_version must be ${JSON.stringify(SGA_CAPABILITIES_SCHEMA_VERSION)}, got ${JSON.stringify(value.schema_version)}`,
    );
  }
  if (typeof value.backend !== "string" || value.backend.length === 0) {
    throw new Error(`${label}.backend must be a non-empty string`);
  }
  if (typeof value.feature_set !== "string" || value.feature_set.length === 0) {
    throw new Error(`${label}.feature_set must be a non-empty string`);
  }
  if ("generatedFrom" in value || "inferenceRevision" in value) {
    throw new Error(
      `${label} must be pin- and host-metadata-free (M8): found forbidden provenance field`,
    );
  }
  if (!value.routes || typeof value.routes !== "object" || Array.isArray(value.routes)) {
    throw new Error(`${label}.routes must be an object map`);
  }
  return canonicalizeValue(value);
}

export async function compareEngineCapabilityFacts(checkedInPath, freshPath) {
  const checkedInRaw = JSON.parse(await readFile(checkedInPath, "utf8"));
  const freshRaw = JSON.parse(await readFile(freshPath, "utf8"));
  const checkedIn = validateSgaCapabilitiesExport(checkedInRaw, checkedInPath);
  const fresh = validateSgaCapabilitiesExport(freshRaw, freshPath);

  return {
    matches: isDeepStrictEqual(checkedIn, fresh),
    schemaVersion: checkedIn.schema_version,
    featureSet: checkedIn.feature_set,
    freshFeatureSet: fresh.feature_set,
  };
}

async function main() {
  const [checkedInPath, freshPath, ...extra] = process.argv.slice(2);
  if (!checkedInPath || !freshPath || extra.length > 0) {
    throw new Error(
      "usage: node scripts/compare-engine-capability-facts.mjs <checked-in.json> <fresh.json>",
    );
  }

  const result = await compareEngineCapabilityFacts(checkedInPath, freshPath);
  if (!result.matches) {
    throw new Error(
      `sga.capabilities/2 exports differ semantically: ${checkedInPath} != ${freshPath}`,
    );
  }
  console.log(
    `sga.capabilities/2 exports match semantically (${result.featureSet}, ${result.schemaVersion}).`,
  );
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
