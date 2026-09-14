#!/usr/bin/env node

import { isDeepStrictEqual } from "node:util";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const REVISION_PATTERN = /^[0-9a-f]{40}$/;

function withoutInferenceRevision(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must contain a JSON object`);
  }
  const generatedFrom = value.generatedFrom;
  if (
    generatedFrom === null ||
    typeof generatedFrom !== "object" ||
    Array.isArray(generatedFrom)
  ) {
    throw new Error(`${label}.generatedFrom must contain a JSON object`);
  }
  const revision = generatedFrom.inferenceRevision;
  if (typeof revision !== "string" || !REVISION_PATTERN.test(revision)) {
    throw new Error(`${label}.generatedFrom.inferenceRevision must be a 40-character lowercase SHA`);
  }

  return {
    revision,
    comparable: {
      ...value,
      generatedFrom: Object.fromEntries(
        Object.entries(generatedFrom).filter(([key]) => key !== "inferenceRevision"),
      ),
    },
  };
}

export async function compareEngineCapabilityFacts(checkedInPath, freshPath) {
  const checkedIn = withoutInferenceRevision(
    JSON.parse(await readFile(checkedInPath, "utf8")),
    checkedInPath,
  );
  const fresh = withoutInferenceRevision(JSON.parse(await readFile(freshPath, "utf8")), freshPath);

  return {
    matches: isDeepStrictEqual(checkedIn.comparable, fresh.comparable),
    checkedInRevision: checkedIn.revision,
    freshRevision: fresh.revision,
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
      `capability facts differ beyond generatedFrom.inferenceRevision: ${checkedInPath} != ${freshPath}`,
    );
  }
  console.log(
    `Capability facts match (revision labels ${result.checkedInRevision} and ${result.freshRevision}).`,
  );
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
