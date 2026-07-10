import { describe, expect, it } from "vitest";

import { isBlocking, isSurfaced, issue, summarize } from "./issues.js";

// The two derivations are the whole contract, so they're pinned exhaustively over
// every kind rather than sampled — a table that omits a kind cannot catch a
// predicate that special-cases it (epic 10644, R1).
describe("issue kinds derive blocking and surfaced", () => {
  const cases = [
    { kind: "requirement", blocking: true, surfaced: false },
    { kind: "error", blocking: true, surfaced: true },
    { kind: "advisory", blocking: false, surfaced: true },
  ];

  for (const { kind, blocking, surfaced } of cases) {
    it(`${kind}: blocks=${blocking}, surfaces=${surfaced}`, () => {
      const item = issue[kind]("rank", "message");
      expect(item.kind).toBe(kind);
      expect(isBlocking(item)).toBe(blocking);
      expect(isSurfaced(item)).toBe(surfaced);
    });
  }

  it("carries the field it belongs to, and null for a form-scoped issue", () => {
    expect(issue.error("rank", "Rank must be greater than zero").field).toBe("rank");
    expect(issue.error(null, "Not enough photos to train").field).toBeNull();
  });
});

describe("summarize: ready", () => {
  it("is true for no issues at all", () => {
    expect(summarize([]).ready).toBe(true);
  });

  // The silent kind still blocks. An implementation that gates `ready` on the
  // *surfaced* issues — the natural mistake, since those are what you can see —
  // would let Start training fire with no dataset picked.
  it("is false for a lone requirement, which blocks without surfacing", () => {
    expect(summarize([issue.requirement("dataset", "Select a saved dataset")]).ready).toBe(false);
  });

  it("is false for a lone error", () => {
    expect(summarize([issue.error("rank", "Rank must be greater than zero")]).ready).toBe(false);
  });

  // The mirror case: an advisory surfaces, so an implementation that conflates
  // "has something to say" with "not ready" would disable a CTA it must not.
  it("is true for a lone advisory, which surfaces without blocking", () => {
    expect(summarize([issue.advisory("captions", "3 captions may not match")]).ready).toBe(true);
  });

  it("is false when an advisory sits alongside a blocking issue", () => {
    const summary = summarize([
      issue.advisory("captions", "3 captions may not match"),
      issue.requirement("dataset", "Select a saved dataset"),
    ]);
    expect(summary.ready).toBe(false);
  });

  it("is true when every issue is advisory", () => {
    const summary = summarize([
      issue.advisory("captions", "3 captions may not match"),
      issue.advisory(null, "The photos are quite similar"),
    ]);
    expect(summary.ready).toBe(true);
  });
});

describe("summarize: surfaced is the one message channel", () => {
  it("drops requirements and keeps errors and advisories, in rule order", () => {
    const summary = summarize([
      issue.requirement("dataset", "Select a saved dataset"),
      issue.error("rank", "Rank must be greater than zero"),
      issue.requirement("outputName", "Name the LoRA output"),
      issue.advisory("captions", "3 captions may not match"),
    ]);
    expect(summary.surfaced.map((item) => item.message)).toEqual([
      "Rank must be greater than zero",
      "3 captions may not match",
    ]);
  });

  it("carries field-scoped and form-scoped messages alike", () => {
    const summary = summarize([
      issue.error(null, "This dataset isn't ready to train yet"),
      issue.error("rank", "Rank must be greater than zero"),
    ]);
    expect(summary.surfaced).toHaveLength(2);
  });

  it("is empty when a form is merely unfilled", () => {
    const summary = summarize([
      issue.requirement("dataset", "Select a saved dataset"),
      issue.requirement("outputName", "Name the LoRA output"),
    ]);
    expect(summary.surfaced).toEqual([]);
    // Nothing to say, and still not ready. That pairing is the point of the
    // requirement kind, so assert both halves together.
    expect(summary.ready).toBe(false);
  });
});

// The mark, never the message. `invalidFields` holds no text, so an error cannot
// render both as a chip and under its own input.
describe("summarize: invalidFields marks inputs without repeating them", () => {
  it("holds field names only — never anything renderable as a message", () => {
    const summary = summarize([issue.error("rank", "Rank must be greater than zero")]);
    expect(summary.invalidFields).toBeInstanceOf(Set);
    expect([...summary.invalidFields]).toEqual(["rank"]);
  });

  it("marks every field carrying an error, once each", () => {
    const summary = summarize([
      issue.error("lora", "Preset LoRA has not finished importing"),
      issue.error("lora", "Preset LoRA is incompatible with the selected model"),
      issue.error("rank", "Rank must be greater than zero"),
    ]);
    expect(summary.invalidFields.size).toBe(2);
    expect(summary.invalidFields.has("lora")).toBe(true);
    expect(summary.invalidFields.has("rank")).toBe(true);
  });

  // A fresh form must not paint itself red before the user has typed a character.
  it("does not mark a requirement's field", () => {
    const summary = summarize([issue.requirement("dataset", "Select a saved dataset")]);
    expect(summary.invalidFields.size).toBe(0);
    expect(summary.ready).toBe(false);
  });

  // An advisory doesn't block, so outlining its input would overstate it.
  it("does not mark an advisory's field, though the advisory still surfaces", () => {
    const summary = summarize([issue.advisory("captions", "3 captions may not match")]);
    expect(summary.invalidFields.size).toBe(0);
    expect(summary.surfaced).toHaveLength(1);
  });

  it("ignores a form-scoped error, which names no input to mark", () => {
    const summary = summarize([issue.error(null, "This dataset isn't ready to train yet")]);
    expect(summary.invalidFields.size).toBe(0);
    expect(summary.surfaced).toHaveLength(1);
  });

  it("is empty for a valid draft", () => {
    expect(summarize([]).invalidFields.size).toBe(0);
  });
});
