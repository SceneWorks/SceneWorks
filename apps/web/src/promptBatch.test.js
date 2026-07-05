import { describe, expect, it } from "vitest";

import {
  cardinality,
  expandBatch,
  extractKeys,
  missingKeys,
  resolvePrompt,
  splitPromptLines,
} from "./promptBatch.js";

describe("splitPromptLines", () => {
  it("splits one prompt per line, ignoring blank lines", () => {
    expect(splitPromptLines("cat\n\n  dog  \nbird\n")).toEqual(["cat", "dog", "bird"]);
  });

  it("does NOT merge newline-separated lines into one prompt", () => {
    expect(splitPromptLines("a\nb\nc")).toHaveLength(3);
  });

  it("switches to block mode when a --- delimiter is present", () => {
    const text = "line one\nline two\n---\nsecond prompt\n---\nthird";
    expect(splitPromptLines(text)).toEqual(["line one\nline two", "second prompt", "third"]);
  });

  it("drops empty blocks and trims each block in delimiter mode", () => {
    expect(splitPromptLines("---\n  only  \n---")).toEqual(["only"]);
  });

  it("returns [] for blank or non-string input", () => {
    expect(splitPromptLines("   \n  ")).toEqual([]);
    expect(splitPromptLines(null)).toEqual([]);
  });
});

describe("extractKeys", () => {
  it("collects unique keys in first-seen order", () => {
    expect(extractKeys(["{{name}} has {{hair_color}} hair", "{{name}} smiling"])).toEqual([
      "name",
      "hair_color",
    ]);
  });

  it("trims whitespace so {{ name }} equals {{name}}", () => {
    expect(extractKeys(["{{ name }} and {{name}}"])).toEqual(["name"]);
  });

  it("supports arbitrary key names and adjacent placeholders", () => {
    expect(extractKeys(["{{a}}{{b-2}}{{ c d }}"])).toEqual(["a", "b-2", "c d"]);
  });

  it("ignores empty and brace-only placeholders", () => {
    expect(extractKeys(["{{}} {{ }} plain"])).toEqual([]);
  });

  it("skips blank prompt lines and non-string/garbage input", () => {
    expect(extractKeys(["  ", "{{x}}", 42, null])).toEqual(["x"]);
    expect(extractKeys("not an array")).toEqual([]);
  });
});

describe("resolvePrompt", () => {
  it("substitutes provided values, honoring trimmed keys", () => {
    expect(resolvePrompt("{{ name }} with {{hair_color}} hair", { name: "Alice", hair_color: "red" })).toBe(
      "Alice with red hair",
    );
  });

  it("leaves unknown keys as their literal placeholder", () => {
    expect(resolvePrompt("{{name}} in {{place}}", { name: "Bob" })).toBe("Bob in {{place}}");
  });

  it("returns empty string for non-string templates", () => {
    expect(resolvePrompt(null, { name: "Alice" })).toBe("");
  });
});

describe("expandBatch", () => {
  it("increments a single variable per render", () => {
    const out = expandBatch(["{{hair_color}} hair"], [{ key: "hair_color", values: ["red", "blue", "blonde"] }]);
    expect(out.map((entry) => entry.prompt)).toEqual(["red hair", "blue hair", "blonde hair"]);
  });

  it("cross-products multiple variables (prompts slowest, last variable fastest)", () => {
    const out = expandBatch(
      ["{{name}}/{{hair}}"],
      [
        { key: "name", values: ["Alice", "Bob"] },
        { key: "hair", values: ["red", "blue"] },
      ],
    );
    expect(out.map((entry) => entry.prompt)).toEqual([
      "Alice/red",
      "Alice/blue",
      "Bob/red",
      "Bob/blue",
    ]);
    expect(out[0].values).toEqual({ name: "Alice", hair: "red" });
  });

  it("multiplies the cross-product across every prompt line", () => {
    const out = expandBatch(
      ["{{name}} smiling", "{{name}} waving"],
      [{ key: "name", values: ["Alice", "Bob"] }],
    );
    expect(out.map((entry) => entry.prompt)).toEqual([
      "Alice smiling",
      "Bob smiling",
      "Alice waving",
      "Bob waving",
    ]);
  });

  it("ignores an unreferenced variable instead of duplicating renders", () => {
    const out = expandBatch(["a plain prompt"], [{ key: "mood", values: ["happy", "sad"] }]);
    expect(out).toEqual([{ prompt: "a plain prompt", values: {} }]);
  });

  it("skips a referenced variable that has no usable values, keeping the placeholder", () => {
    const out = expandBatch(["{{name}} in {{place}}"], [
      { key: "name", values: ["Alice"] },
      { key: "place", values: ["", "   "] },
    ]);
    expect(out).toEqual([{ prompt: "Alice in {{place}}", values: { name: "Alice" } }]);
  });

  it("trims values and drops blanks before expanding", () => {
    const out = expandBatch(["{{c}}"], [{ key: "c", values: [" red ", "", "blue"] }]);
    expect(out.map((entry) => entry.prompt)).toEqual(["red", "blue"]);
  });

  it("takes the first entry when a key is defined twice", () => {
    const out = expandBatch(["{{k}}"], [
      { key: "k", values: ["one"] },
      { key: "k", values: ["two"] },
    ]);
    expect(out.map((entry) => entry.prompt)).toEqual(["one"]);
  });

  it("returns nothing for an all-blank prompt list", () => {
    expect(expandBatch(["", "   "], [{ key: "x", values: ["a"] }])).toEqual([]);
  });
});

describe("cardinality", () => {
  it("sums per-line products × count (each line references only {{name}})", () => {
    const prompts = ["{{name}} smiling", "{{name}} waving", "{{name}} jumping"];
    const variables = [
      { key: "name", values: ["Alice", "Bob"] },
      { key: "hair", values: ["red", "blue"] },
    ];
    // {{hair}} is referenced nowhere, so it never multiplies: 3 lines × 2 names × 4 = 24.
    expect(cardinality(prompts, variables, 4)).toBe(3 * 2 * 4);
  });

  it("adds up uneven per-line fan-out rather than one global product", () => {
    const prompts = ["{{name}} with {{hair}} hair", "{{name}} alone"];
    const variables = [
      { key: "name", values: ["Alice", "Bob"] },
      { key: "hair", values: ["red", "blue"] },
    ];
    // line 1 fans out name×hair = 4; line 2 fans out name = 2; total 6, not 2×2×2=8.
    expect(cardinality(prompts, variables, 1)).toBe(4 + 2);
  });

  it("counts a hair axis once it is referenced", () => {
    const prompts = ["{{name}} with {{hair}} hair"];
    const variables = [
      { key: "name", values: ["Alice", "Bob"] },
      { key: "hair", values: ["red", "blue", "blonde"] },
    ];
    expect(cardinality(prompts, variables, 4)).toBe(1 * 2 * 3 * 4);
  });

  it("is 0 for an empty or all-blank prompt list", () => {
    expect(cardinality([], [], 4)).toBe(0);
    expect(cardinality(["  "], [], 4)).toBe(0);
  });

  it("treats an unfilled referenced variable as a factor of 1", () => {
    expect(cardinality(["{{name}} in {{place}}"], [{ key: "name", values: ["Alice"] }], 1)).toBe(1);
  });

  it("defaults a missing or invalid count to 1", () => {
    expect(cardinality(["{{name}}"], [{ key: "name", values: ["Alice", "Bob"] }])).toBe(2);
    expect(cardinality(["{{name}}"], [{ key: "name", values: ["Alice", "Bob"] }], 0)).toBe(2);
    expect(cardinality(["{{name}}"], [{ key: "name", values: ["Alice", "Bob"] }], "3")).toBe(6);
  });

  it("equals the expandBatch job count times the variation count (the pinned invariant)", () => {
    const prompts = ["{{name}}/{{hair}}", "{{name}} alone", "a plain line"];
    const variables = [
      { key: "name", values: ["Alice", "Bob"] },
      { key: "hair", values: ["red", "blue"] },
    ];
    const jobs = expandBatch(prompts, variables).length; // 4 + 2 + 1 = 7
    expect(jobs).toBe(7);
    expect(cardinality(prompts, variables, 4)).toBe(jobs * 4);
  });

  it("does not emit duplicate renders for a line missing a batch variable", () => {
    const out = expandBatch(["{{name}}/{{hair}}", "{{name}} alone"], [
      { key: "name", values: ["Alice", "Bob"] },
      { key: "hair", values: ["red", "blue"] },
    ]);
    expect(out.map((entry) => entry.prompt)).toEqual([
      "Alice/red",
      "Alice/blue",
      "Bob/red",
      "Bob/blue",
      "Alice alone",
      "Bob alone",
    ]);
  });
});

describe("missingKeys", () => {
  it("names referenced keys that have no usable value", () => {
    expect(
      missingKeys(["{{name}} in {{place}} at {{time}}"], [
        { key: "name", values: ["Alice"] },
        { key: "place", values: [""] },
      ]),
    ).toEqual(["place", "time"]);
  });

  it("returns nothing when every referenced key is filled", () => {
    expect(missingKeys(["{{name}}"], [{ key: "name", values: ["Alice"] }])).toEqual([]);
  });

  it("ignores unreferenced variables", () => {
    expect(missingKeys(["plain"], [{ key: "mood", values: [] }])).toEqual([]);
  });
});
