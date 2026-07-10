import { describe, expect, it } from "vitest";

import { loraImportValidation, modelImportValidation } from "./modelManagerValidation.js";
import { summarize } from "./validation/issues.js";

// The Model Manager import gates (epic 10651). Both emit only silent requirements — the
// empty file/URL inputs show what's missing — so `ready` is the whole contract and nothing
// ever surfaces.
describe("loraImportValidation", () => {
  it("passes a global URL import with a URL", () => {
    expect(summarize(loraImportValidation({ scope: "global", isFileImport: false, sourceUrl: "https://x" })).ready).toBe(true);
  });

  it("requires a project for a project-scoped import, silently", () => {
    const summary = summarize(loraImportValidation({ scope: "project", activeProject: null, isFileImport: false, sourceUrl: "https://x" }));
    expect(summary.ready).toBe(false);
    expect(summary.surfaced).toEqual([]);
  });

  it("requires a file in file mode, and a URL otherwise — both silent", () => {
    expect(summarize(loraImportValidation({ scope: "global", isFileImport: true, file: null })).ready).toBe(false);
    expect(summarize(loraImportValidation({ scope: "global", isFileImport: false, sourceUrl: "  " })).ready).toBe(false);
    expect(summarize(loraImportValidation({ scope: "global", isFileImport: true, file: {} })).surfaced).toEqual([]);
  });
});

describe("modelImportValidation", () => {
  it("passes when a source is present", () => {
    expect(summarize(modelImportValidation({ isFileImport: true, file: {} })).ready).toBe(true);
    expect(summarize(modelImportValidation({ isFileImport: false, sourceUrl: "https://x" })).ready).toBe(true);
  });

  it("requires a file or URL, silently", () => {
    const summary = summarize(modelImportValidation({ isFileImport: false, sourceUrl: "" }));
    expect(summary.ready).toBe(false);
    expect(summary.surfaced).toEqual([]);
  });
});
