import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  compareEngineCapabilityFacts,
  SGA_CAPABILITIES_SCHEMA_VERSION,
} from "./compare-engine-capability-facts.mjs";

const CHECKER = join(
  dirname(fileURLToPath(import.meta.url)),
  "compare-engine-capability-facts.mjs",
);

async function withExports(t, checkedIn, fresh) {
  const directory = await mkdtemp(join(tmpdir(), "sceneworks-sga-capabilities-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const checkedInPath = join(directory, "checked-in.json");
  const freshPath = join(directory, "fresh.json");
  await Promise.all([
    writeFile(checkedInPath, JSON.stringify(checkedIn)),
    writeFile(freshPath, JSON.stringify(fresh)),
  ]);
  return { checkedInPath, freshPath };
}

function sgaExport(supportsPreview = true, keyOrder = "normal") {
  if (keyOrder === "reversed") {
    return {
      routes: {
        boogu_image: {
          capabilities: { supports_preview: supportsPreview },
          route_id: "boogu_image",
        },
      },
      feature_set: "candle-cuda",
      backend: "candle",
      schema_version: SGA_CAPABILITIES_SCHEMA_VERSION,
    };
  }
  return {
    schema_version: SGA_CAPABILITIES_SCHEMA_VERSION,
    backend: "candle",
    feature_set: "candle-cuda",
    routes: {
      boogu_image: {
        route_id: "boogu_image",
        capabilities: { supports_preview: supportsPreview },
      },
    },
  };
}

test("accepts semantically identical sga.capabilities/2 exports regardless of key order (M8)", async (t) => {
  const paths = await withExports(t, sgaExport(true, "normal"), sgaExport(true, "reversed"));
  const result = await compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath);
  assert.equal(result.matches, true);
  assert.equal(result.schemaVersion, SGA_CAPABILITIES_SCHEMA_VERSION);
  assert.equal(result.featureSet, "candle-cuda");
});

test("rejects semantic capability drift", async (t) => {
  const paths = await withExports(t, sgaExport(true), sgaExport(false));
  const result = await compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath);
  assert.equal(result.matches, false);
});

test("CLI reports semantic drift as a failed check", async (t) => {
  const paths = await withExports(t, sgaExport(true), sgaExport(false));
  const result = spawnSync(process.execPath, [CHECKER, paths.checkedInPath, paths.freshPath], {
    encoding: "utf8",
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /sga\.capabilities\/2 exports differ semantically/);
});

test("rejects legacy or wrong schema_version", async (t) => {
  const legacy = { ...sgaExport(true), schema_version: "sga.capabilities/1" };
  const paths = await withExports(t, legacy, sgaExport(true));
  await assert.rejects(
    compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath),
    /schema_version must be "sga\.capabilities\/2"/,
  );
});

test("rejects forbidden pin/host provenance fields (M8)", async (t) => {
  const tainted = { ...sgaExport(true), generatedFrom: { inferenceRevision: "a".repeat(40) } };
  const paths = await withExports(t, tainted, sgaExport(true));
  await assert.rejects(
    compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath),
    /must be pin- and host-metadata-free/,
  );
});
