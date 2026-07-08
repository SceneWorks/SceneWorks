import { describe, expect, it } from "vitest";

import {
  PROMPT_BATCH_EXPORT_VERSION,
  fromPromptBatchImport,
  serializePromptBatchExport,
  toPromptBatchExport,
} from "./promptBatchIO.js";

const sampleBatch = {
  id: "character_turnaround",
  scope: "global",
  manifestPath: "/home/user/.config/sceneworks/manifests/user.prompt-batches.jsonc",
  archived: false,
  createdAt: "2026-07-05T00:00:00Z",
  updatedAt: "2026-07-05T00:00:00Z",
  name: "Character Turnaround",
  prompts: ["{{name}} with {{hair}} hair, front view", "{{name}} profile"],
  variables: [
    { key: "name", values: ["Alice"] },
    { key: "hair", values: ["red", "blue"] },
  ],
  lastValues: { name: ["Alice"], hair: ["red", "blue"] },
};

describe("toPromptBatchExport", () => {
  it("carries only authored content, never server-managed fields", () => {
    const exported = toPromptBatchExport(sampleBatch);
    expect(exported).toEqual({
      sceneworksPromptBatch: PROMPT_BATCH_EXPORT_VERSION,
      name: "Character Turnaround",
      prompts: sampleBatch.prompts,
      variables: sampleBatch.variables,
      lastValues: sampleBatch.lastValues,
    });
    for (const field of ["id", "scope", "manifestPath", "archived", "createdAt", "updatedAt"]) {
      expect(exported).not.toHaveProperty(field);
    }
  });

  it("omits lastValues when there are none", () => {
    const exported = toPromptBatchExport({ name: "Plain", prompts: ["a"], variables: [] });
    expect(exported).not.toHaveProperty("lastValues");
  });
});

describe("import round-trip", () => {
  it("export → serialize → import yields the authored core", () => {
    const json = serializePromptBatchExport(sampleBatch);
    const imported = fromPromptBatchImport(json);
    expect(imported).toEqual({
      name: "Character Turnaround",
      prompts: sampleBatch.prompts,
      variables: sampleBatch.variables,
      lastValues: sampleBatch.lastValues,
    });
  });

  it("accepts an already-parsed object", () => {
    const imported = fromPromptBatchImport(toPromptBatchExport(sampleBatch));
    expect(imported.name).toBe("Character Turnaround");
    expect(imported.prompts).toHaveLength(2);
  });
});

describe("fromPromptBatchImport validation", () => {
  it("rejects non-JSON strings", () => {
    expect(() => fromPromptBatchImport("{not json")).toThrow(/valid JSON/);
  });

  it("rejects non-objects", () => {
    expect(() => fromPromptBatchImport("[1,2,3]")).toThrow(/JSON object/);
    expect(() => fromPromptBatchImport("42")).toThrow(/JSON object/);
  });

  it("rejects files that are not prompt batches", () => {
    expect(() => fromPromptBatchImport({ some: "thing" })).toThrow(/not a SceneWorks prompt batch/);
  });

  it("rejects a wrong prompts type", () => {
    expect(() => fromPromptBatchImport({ prompts: "nope" })).toThrow(/prompts must be an array/);
  });

  it("rejects a wrong variables type", () => {
    expect(() =>
      fromPromptBatchImport({ prompts: [], variables: "nope" }),
    ).toThrow(/variables must be an array/);
  });

  it("sanitizes junk entries within otherwise-valid arrays", () => {
    const imported = fromPromptBatchImport({
      name: "  Messy  ",
      prompts: ["good", 42, null, "also good"],
      variables: [
        { key: "name", values: ["Alice", 7, "Bob"] },
        { key: "", values: ["dropped"] },
        { notAKey: true },
      ],
    });
    expect(imported.name).toBe("Messy");
    expect(imported.prompts).toEqual(["good", "also good"]);
    expect(imported.variables).toEqual([{ key: "name", values: ["Alice", "Bob"] }]);
  });
});
