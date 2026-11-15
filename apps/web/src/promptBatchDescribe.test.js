import { describe, expect, it } from "vitest";
import { extractKeys, parsePromptResolution, splitPromptLines } from "./promptBatch.js";
import {
  MIN_FIT_TO_PROMPT_TOKENS,
  condenseChunks,
  condensedSizeText,
  fitToTargets,
  formatPromptBatch,
  snapRatio,
  wrapLabels,
} from "./promptBatchDescribe.js";

const fields = (over = {}) => ({
  description: "man_1 walks past woman_1",
  style: "Photo",
  camera: "",
  composition: "  ",
  lighting: "Soft",
  environment: "Street",
  ...over,
});

describe("snapRatio", () => {
  it("snaps to the nearest standard ratio", () => {
    expect(snapRatio(1500, 1001)).toBe("3:2");
    expect(snapRatio(1000, 800)).toBe("5:4");
    expect(snapRatio(768, 1024)).toBe("3:4");
    expect(snapRatio(1000, 1000)).toBe("1:1");
    expect(snapRatio(2560, 1080)).toBe("21:9");
    expect(snapRatio(1080, 1920)).toBe("9:16");
    expect(snapRatio(0, 10)).toBeNull();
  });
});

describe("wrapLabels", () => {
  it("wraps bare labels once and leaves wrapped ones", () => {
    expect(wrapLabels("woman_1 pets dog_2; {{woman_1}} and {{a|b_2}}")).toBe(
      "{{woman_1}} pets {{dog_2}}; {{woman_1}} and {{a|b_2}}",
    );
    expect(wrapLabels("Sign reads FOO_1 and snake_case")).toBe("Sign reads FOO_1 and snake_case");
  });
});

describe("formatPromptBatch", () => {
  const results = [
    { itemId: "a", width: 1500, height: 1001, fields: fields() },
    { itemId: "b", width: 1, height: 1, failed: true, error: "x", fields: fields() },
    {
      itemId: "c",
      width: 768,
      height: 1024,
      fields: fields({ description: "girl_1\n---\nwaves at cat_1", camera: "Wide" }),
    },
  ];

  it("round-trips through the existing parser", () => {
    const { text, skipped } = formatPromptBatch(results);
    expect(skipped.map((r) => r.itemId)).toEqual(["b"]);
    expect(text.startsWith(
      "[3:2]\nDescription: {{man_1}} walks past {{woman_1}}\nStyle: Photo\nLighting: Soft\nEnvironment: Street",
    )).toBe(true);
    const prompts = splitPromptLines(text);
    expect(prompts).toHaveLength(2);
    const options = ["1536x1024", "1024x1536"];
    expect(parsePromptResolution(prompts[0], options).resolution).toEqual({ width: 1536, height: 1024 });
    const second = parsePromptResolution(prompts[1], options);
    expect(second.resolution).toEqual({ width: 1024, height: 1536 });
    expect(second.prompt).toBe(
      "Description: {{girl_1}} --- waves at {{cat_1}}\nStyle: Photo\nCamera: Wide\nLighting: Soft\nEnvironment: Street",
    );
    expect(extractKeys(prompts)).toEqual(["man_1", "woman_1", "girl_1", "cat_1"]);
  });

  it("emits one Pose line per person after Description, with labels as batch variables", () => {
    const pose =
      "woman_1 — Body line: upright; Stance: standing on their right leg; Facing: front; Torso: straight; " +
      "Head: tilted toward their left shoulder; Left arm: hidden; Right arm: hand on man_1's shoulder; " +
      "Left leg: knee bent; Right leg: straight; Depth: their left side is nearer the camera; Placement: frame-left of man_1\n" +
      "man_1 — Body line: upright; Stance: seated; Facing: profile toward frame-left; Torso: leaning back; " +
      "Head: turned toward woman_1; Left arm: on the table; Right arm: hidden; Left leg: hidden; Right leg: hidden; " +
      "Depth: not stated; Placement: frame-right";
    const { text } = formatPromptBatch([{ itemId: "p", width: 1000, height: 1000, fields: fields({ pose }) }]);
    const lines = text.split("\n");
    expect(lines.slice(0, 4)).toEqual([
      "[1:1]",
      "Description: {{man_1}} walks past {{woman_1}}",
      "Pose: {{woman_1}} — Body line: upright; Stance: standing on their right leg; Facing: front; Torso: straight; " +
        "Head: tilted toward their left shoulder; Left arm: hidden; Right arm: hand on {{man_1}}'s shoulder; " +
        "Left leg: knee bent; Right leg: straight; Depth: their left side is nearer the camera; Placement: frame-left of {{man_1}}",
      "Pose: {{man_1}} — Body line: upright; Stance: seated; Facing: profile toward frame-left; Torso: leaning back; " +
        "Head: turned toward {{woman_1}}; Left arm: on the table; Right arm: hidden; Left leg: hidden; Right leg: hidden; " +
        "Depth: not stated; Placement: frame-right",
    ]);
    expect(lines[4]).toBe("Style: Photo");
    expect(extractKeys(splitPromptLines(text))).toEqual(["man_1", "woman_1"]);
  });

  it("keeps a plain-text pose, skips blank pose lines, and formats a six-field result without a Pose line", () => {
    const plain = formatPromptBatch([
      { itemId: "j", width: 1000, height: 1000, fields: fields({ pose: "  man_1 stands upright, arms folded.\n\n  woman_1 sits,   facing frame-left.  " }) },
    ]).text;
    expect(plain).toBe(
      "[1:1]\nDescription: {{man_1}} walks past {{woman_1}}\nPose: {{man_1}} stands upright, arms folded.\n" +
        "Pose: {{woman_1}} sits, facing frame-left.\n" +
        "Style: Photo\nLighting: Soft\nEnvironment: Street",
    );
    const empty = formatPromptBatch([{ itemId: "e", width: 1000, height: 1000, fields: fields({ pose: "" }) }]).text;
    const legacy = formatPromptBatch([{ itemId: "l", width: 1000, height: 1000, fields: fields() }]).text;
    expect("pose" in fields()).toBe(false);
    expect(legacy).toBe("[1:1]\nDescription: {{man_1}} walks past {{woman_1}}\nStyle: Photo\nLighting: Soft\nEnvironment: Street");
    expect(empty).toBe(legacy);
  });

  it("skips a row whose fields are all empty instead of emitting a ratio-only prompt", () => {
    const empty = { itemId: "d", width: 1000, height: 1000, fields: fields({ description: "", style: " ", lighting: "", environment: "" }) };
    const { text, skipped } = formatPromptBatch([empty, results[0]]);
    expect(skipped).toEqual([empty]);
    expect(text).toBe(formatPromptBatch([results[0]]).text);
  });
});

describe("Fit to (SW-7163)", () => {
  const condensed = (over = {}) => ({
    target: "krea_2_turbo",
    limitTokens: 1024,
    text: "A man, man_1, walks past woman_1\n on a street.",
    tokens: 612,
    tokenCount: "exact",
    characters: 46,
    over: false,
    failed: false,
    notice: null,
    ...over,
  });
  const row = (over = {}) => ({ itemId: "a", width: 1500, height: 1000, fields: fields(), condensed: condensed(), ...over });

  it("offers only catalog models whose prompt limit can hold the instruction (at least 512 tokens)", () => {
    expect(
      fitToTargets([
        { id: "krea_2_turbo", name: "Krea 2 Turbo", image: { maxPromptTokens: 1024 } },
        { id: "z_image_turbo", name: "Z-Image Turbo", image: {} },
        { id: "supra2_img", name: "Supra2-IMG", image: { maxPromptTokens: 128 } },
        { id: "broken", image: { maxPromptTokens: "1024" } },
      ]),
    ).toEqual([
      { id: "krea_2_turbo", name: "Krea 2 Turbo", limit: 1024 },
    ]);
    expect(MIN_FIT_TO_PROMPT_TOKENS).toBe(512);
    expect(fitToTargets([{ id: "edge", name: "Edge", image: { maxPromptTokens: 512 } }])).toHaveLength(1);
    expect(fitToTargets(undefined)).toEqual([]);
  });

  it("uses the condensed prompt for the selected target, one line with labels wrapped", () => {
    expect(formatPromptBatch([row()], { fitTo: "krea_2_turbo" }).text).toBe(
      "[3:2]\nA man, {{man_1}}, walks past {{woman_1}} on a street.",
    );
    expect(extractKeys(formatPromptBatch([row()], { fitTo: "krea_2_turbo" }).text.split("\n---\n"))).toEqual([
      "man_1",
      "woman_1",
    ]);
  });

  it("keeps the seven-field block with Fit to off, for another target, or after a failed condense", () => {
    const fieldsBlock = formatPromptBatch([row({ condensed: null })]).text;
    expect(fieldsBlock.startsWith("[3:2]\nDescription: {{man_1}} walks past {{woman_1}}")).toBe(true);
    expect(formatPromptBatch([row()]).text).toBe(fieldsBlock);
    expect(formatPromptBatch([row()], { fitTo: "krea_2_raw" }).text).toBe(fieldsBlock);
    expect(formatPromptBatch([row({ condensed: condensed({ failed: true }) })], { fitTo: "krea_2_turbo" }).text).toBe(
      fieldsBlock,
    );
    // An item stored before the step existed has no `condensed` at all.
    const { condensed: _none, ...old } = row();
    expect(formatPromptBatch([old], { fitTo: "krea_2_turbo" }).text).toBe(fieldsBlock);
  });

  it("states an exact size plainly and an estimate as about", () => {
    expect(condensedSizeText(condensed())).toBe("612 of 1,024 tokens");
    expect(condensedSizeText(condensed({ tokenCount: "estimate", tokens: 1100 }))).toBe("about 1,100 of 1,024 tokens");
    expect(condensedSizeText(condensed({ tokens: null }))).toBeNull();
    expect(condensedSizeText(null)).toBeNull();
  });

  it("never uses a condensed prompt the row's fields have changed since", () => {
    const fieldsBlock = formatPromptBatch([row({ condensed: null })]).text;
    expect(formatPromptBatch([row({ fieldsEdited: true })], { fitTo: "krea_2_turbo" }).text).toBe(fieldsBlock);
  });

  it("states the room kept for variable values and nothing once the prompt is edited", () => {
    expect(condensedSizeText(condensed({ labelMentions: 8 }))).toBe("612 of 1,024 tokens, leaving room for 8 variable values");
    expect(condensedSizeText(condensed({ labelMentions: 1 }))).toBe("612 of 1,024 tokens, leaving room for 1 variable value");
    expect(condensedSizeText(condensed({ edited: true }))).toBeNull();
  });

  it("splits condense-only work into parts of one describer, bounded by size and row count", () => {
    const rows = Array.from({ length: 5 }, (_, index) => ({
      itemId: `i${index}`,
      model: index === 4 ? "other/describer" : "vision",
      text: "x".repeat(100),
    }));
    const toRequest = (r) => ({ itemId: r.itemId, text: r.text });
    const byRows = condenseChunks(rows, toRequest, { maxRows: 2 });
    expect(byRows.map((part) => [part.model, part.rows.map((r) => r.itemId)])).toEqual([
      ["vision", ["i0", "i1"]],
      ["vision", ["i2", "i3"]],
      ["other/describer", ["i4"]],
    ]);
    expect(byRows.every((part) => part.variant === null)).toBe(true);
    // A tiered describer's rows condense at the tier each was described with: one part per tier.
    const tiered = [
      { itemId: "a", model: "big", variant: "q4", text: "x" },
      { itemId: "b", model: "big", variant: "q8", text: "x" },
      { itemId: "c", model: "big", variant: "q4", text: "x" },
    ];
    expect(condenseChunks(tiered, toRequest).map((part) => [part.model, part.variant, part.rows.map((r) => r.itemId)])).toEqual([
      ["big", "q4", ["a", "c"]],
      ["big", "q8", ["b"]],
    ]);
    const oneRowBytes = new TextEncoder().encode(JSON.stringify(toRequest(rows[0]))).length + 1;
    const bySize = condenseChunks(rows.slice(0, 4), toRequest, { maxBytes: oneRowBytes * 3 - 1 });
    expect(bySize.map((part) => part.rows.length)).toEqual([2, 2]);
    // The default part stays well under the API's 10 MiB JSON body limit.
    const big = Array.from({ length: 2500 }, (_, index) => ({ itemId: `r${index}`, model: "vision", text: "y".repeat(4000) }));
    const parts = condenseChunks(big, toRequest);
    expect(parts.every((part) => new TextEncoder().encode(JSON.stringify(part.rows)).length < 10 * 1024 * 1024 / 4)).toBe(true);
    expect(parts.flatMap((part) => part.rows)).toHaveLength(2500);
  });
});
