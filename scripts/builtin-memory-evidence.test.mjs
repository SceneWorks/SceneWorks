import assert from "node:assert/strict";
import test from "node:test";

import { projectBuiltinMemoryEvidence } from "./lib/builtin-memory-evidence.mjs";

const identity = {
  provider: "huggingface",
  repo: "owner/model",
  revision: "a".repeat(40),
  variant: "q4",
  files: ["model.safetensors", "config.json"],
};

function fixture() {
  return {
    models: [{
      id: "example",
      mlx: { minMemoryGb: 99, quantize: "q4" },
      downloads: [{ ...identity, footprint: {
        diskSizeBytes: 123,
        residentMemoryBytes: 999,
      } }],
    }],
  };
}

test("evidence replaces measured facts without changing product identity or disk size", () => {
  const manifest = fixture();
  const evidence = { example: {
    mlx: { minMemoryGb: 42 },
    downloads: [{ ...identity, footprint: {
      residentMemoryBytes: 456,
      peakMemoryBytes: 789,
    } }],
  } };
  const projected = projectBuiltinMemoryEvidence(manifest, evidence);
  assert.equal(projected.models[0].mlx.minMemoryGb, 42);
  assert.equal(projected.models[0].mlx.quantize, "q4");
  assert.deepEqual(projected.models[0].downloads[0].footprint, {
    diskSizeBytes: 123,
    residentMemoryBytes: 456,
    peakMemoryBytes: 789,
  });
  assert.equal(manifest.models[0].mlx.minMemoryGb, 99);
  assert.equal(manifest.models[0].downloads[0].footprint.residentMemoryBytes, 999);
});

test("download evidence refuses changed file order, duplicate identity, and product disk size", () => {
  const manifest = fixture();
  const row = { ...identity, footprint: { residentMemoryBytes: 456 } };
  assert.throws(() => projectBuiltinMemoryEvidence(manifest, {
    example: { downloads: [{ ...row, files: [...identity.files].reverse() }] },
  }), /matches 0 product rows/);
  assert.throws(() => projectBuiltinMemoryEvidence(manifest, {
    example: { downloads: [row, row] },
  }), /duplicate memory evidence download/);
  assert.throws(() => projectBuiltinMemoryEvidence(manifest, {
    example: { downloads: [{ ...identity, footprint: { diskSizeBytes: 999 } }] },
  }), /cannot own product download field/);
});

test("platforms distinguish otherwise identical downloads and explicit null measurements survive", () => {
  const manifest = fixture();
  manifest.models[0].downloads = [
    { ...identity, platforms: ["macos"], footprint: { diskSizeBytes: 12 } },
    { ...identity, platforms: ["windows"], footprint: { diskSizeBytes: 13 } },
  ];
  const evidence = { example: { downloads: [
    { ...identity, platforms: ["macos"], footprint: { residentMemoryBytes: null, peakMemoryBytes: null } },
    { ...identity, platforms: ["windows"], footprint: { residentMemoryBytes: 24 } },
  ] } };
  const projected = projectBuiltinMemoryEvidence(manifest, evidence);
  assert.deepEqual(projected.models[0].downloads.map((row) => row.footprint), [
    { diskSizeBytes: 12, residentMemoryBytes: null, peakMemoryBytes: null },
    { diskSizeBytes: 13, residentMemoryBytes: 24 },
  ]);
  assert.throws(() => projectBuiltinMemoryEvidence(manifest, {
    example: { downloads: [{ ...identity, footprint: { residentMemoryBytes: 24 } }] },
  }), /matches 0 product rows|matches 2 product rows/);
});

test("a unique five-field identity does not require a platforms discriminator", () => {
  const manifest = fixture();
  manifest.models[0].downloads[0].platforms = ["macos"];
  const projected = projectBuiltinMemoryEvidence(manifest, {
    example: { downloads: [{ ...identity, footprint: { residentMemoryBytes: null } }] },
  });
  assert.equal(projected.models[0].downloads[0].footprint.residentMemoryBytes, null);
});
