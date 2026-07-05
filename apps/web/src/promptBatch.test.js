import { describe, expect, it } from "vitest";

import {
  cardinality,
  expandBatch,
  extractKeys,
  firstResolvedPrompt,
  linkedGroupIssues,
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

  it("ignores inline and linked placeholders (their values are inline)", () => {
    expect(missingKeys(["{{a|b}} and {{p:he|she}} {{p:his|her}}"], [])).toEqual([]);
  });
});

describe("inline alternation {{a|b|c}}", () => {
  it("expands an inline placeholder into one render per option", () => {
    expect(expandBatch(["a {{small|large}} cat"], []).map((entry) => entry.prompt)).toEqual([
      "a small cat",
      "a large cat",
    ]);
  });

  it("is not a named key — no chip editor, never missing", () => {
    expect(extractKeys(["{{he|she|they}} waves"])).toEqual([]);
    expect(missingKeys(["{{he|she|they}} waves"], [])).toEqual([]);
  });

  it("treats each occurrence as an independent axis (cross-product)", () => {
    const out = expandBatch(["{{a|b}}-{{a|b}}"], []).map((entry) => entry.prompt);
    expect(out).toEqual(["a-a", "a-b", "b-a", "b-b"]);
  });

  it("cross-products inline with named variables (named first, then inline fastest)", () => {
    const out = expandBatch(["{{name}} the {{small|big}} one"], [
      { key: "name", values: ["Alice", "Bob"] },
    ]).map((entry) => entry.prompt);
    expect(out).toEqual([
      "Alice the small one",
      "Alice the big one",
      "Bob the small one",
      "Bob the big one",
    ]);
  });

  it("keeps empty options as a valid empty choice", () => {
    expect(expandBatch(["a{{|-large}} cat"], []).map((entry) => entry.prompt)).toEqual([
      "a cat",
      "a-large cat",
    ]);
  });

  it("counts inline options in cardinality", () => {
    expect(cardinality(["{{a|b|c}} {{x|y}}"], [], 2)).toBe(3 * 2 * 2);
  });
});

describe("linked groups {{label:a|b|c}} (zip)", () => {
  it("advances same-label placeholders together (pronouns stay grammatical)", () => {
    const out = expandBatch(
      ["{{p:he|she|they}} took {{p:his|her|their}} bag; it was {{p:his|hers|theirs}}."],
      [],
    ).map((entry) => entry.prompt);
    expect(out).toEqual([
      "he took his bag; it was his.",
      "she took her bag; it was hers.",
      "they took their bag; it was theirs.",
    ]);
  });

  it("zips (not cross-products) — 3 renders, not 27", () => {
    expect(cardinality(["{{p:he|she|they}} {{p:his|her|their}} {{p:him|her|them}}"], [], 1)).toBe(3);
  });

  it("cross-products DIFFERENT labels but zips within each", () => {
    const out = expandBatch(["{{p:he|she}} likes {{c:red|blue}}"], []).map((entry) => entry.prompt);
    // p (2) × c (2) = 4, but each label zips internally
    expect(out).toEqual(["he likes red", "he likes blue", "she likes red", "she likes blue"]);
  });

  it("a label:value with no pipe is still a named key, not a group", () => {
    expect(extractKeys(["{{time:noon}}"])).toEqual(["time:noon"]);
  });

  it("clamps a mismatched group to the shortest for expansion", () => {
    const out = expandBatch(["{{p:he|she|they}}/{{p:his|her}}"], []).map((entry) => entry.prompt);
    expect(out).toEqual(["he/his", "she/her"]);
  });
});

describe("linkedGroupIssues", () => {
  it("reports a same-label group with differing option counts", () => {
    expect(linkedGroupIssues(["{{p:he|she|they}} {{p:his|her}}"])).toEqual([
      { label: "p", lengths: [2, 3] },
    ]);
  });

  it("is empty when every same-label group matches", () => {
    expect(linkedGroupIssues(["{{p:he|she}} {{p:his|her}}"])).toEqual([]);
    expect(linkedGroupIssues(["{{a|b}} plain {{name}}"])).toEqual([]);
  });
});

describe("firstResolvedPrompt", () => {
  it("resolves line 1 with the first choice of every axis, no full materialization", () => {
    expect(
      firstResolvedPrompt(["{{name}} in {{a|b|c}} {{p:he|she}}/{{p:his|her}}", "second"], [
        { key: "name", values: ["Alice", "Bob"] },
      ]),
    ).toBe("Alice in a he/his");
  });

  it("is empty for a blank batch", () => {
    expect(firstResolvedPrompt([], [])).toBe("");
  });

  it("leaves an unfilled named key literal", () => {
    expect(firstResolvedPrompt(["{{name}} {{a|b}}"], [])).toBe("{{name}} a");
  });
});
