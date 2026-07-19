import { describe, expect, it } from "vitest";

import {
  composeStyledPrompt,
  parseDirectiveLines,
  promptBudget,
  PROMPT_MAX_CHARS,
  STYLE_DIRECTIVE_KEYS,
} from "./styleComposer.js";

// sc-13129: the style composer wraps an already-preset-folded userPrompt in a Style/Description
// template, splicing around any directive lines the user typed. These pin the exact composed
// string the studio sends. Table-driven: {name, styleText, userPrompt, want}.
describe("composeStyledPrompt", () => {
  const cases = [
    {
      name: "no style selected → passthrough unchanged",
      styleText: "",
      userPrompt: "a fox in the snow",
      want: "a fox in the snow",
    },
    {
      name: "undefined styleText behaves as no style → passthrough",
      styleText: undefined,
      userPrompt: "a fox in the snow",
      want: "a fox in the snow",
    },
    {
      name: "whitespace-only styleText behaves as no style → passthrough",
      styleText: "   \n\t ",
      userPrompt: "a fox in the snow",
      want: "a fox in the snow",
    },
    {
      name: "clean prompt → basic Style/Description",
      styleText: "cinematic watercolor",
      userPrompt: "a fox in the snow",
      want: "Style: cinematic watercolor\nDescription: a fox in the snow",
    },
    {
      name: "mid-sentence colon is NOT a directive",
      styleText: "oil painting",
      userPrompt: "a portrait: dramatic lighting on her face",
      want: "Style: oil painting\nDescription: a portrait: dramatic lighting on her face",
    },
    {
      name: "lowercase line-start colon is NOT a directive",
      styleText: "oil painting",
      userPrompt: "style: not really a directive",
      want: "Style: oil painting\nDescription: style: not really a directive",
    },
    {
      name: "unrecognized capitalized key is NOT a directive (folds into Description)",
      styleText: "oil painting",
      userPrompt: "Note: this is a plain sentence",
      want: "Style: oil painting\nDescription: Note: this is a plain sentence",
    },
    {
      // Discriminates the LINE-ANCHOR (^) rule for a RECOGNIZED key: "Lighting" appears
      // mid-line, so it must stay prose and fold into Description, never become a sibling
      // directive. Fails if the `^` anchor is dropped from DIRECTIVE_LINE_RE.
      name: "recognized key mid-line is NOT a directive (stays prose)",
      styleText: "oil painting",
      userPrompt: "a photo with dramatic Lighting: soft",
      want: "Style: oil painting\nDescription: a photo with dramatic Lighting: soft",
    },
    {
      // Same anchor rule with "Setting" preceded by a word on the same line.
      name: "recognized key after leading word is NOT a directive (stays prose)",
      styleText: "oil painting",
      userPrompt: "The Setting: was perfect",
      want: "Style: oil painting\nDescription: The Setting: was perfect",
    },
    {
      // CRLF (\r\n) line breaks must classify recognized keys as directives (issue 1). A
      // trailing "\r" on the non-final lines would break the anchored regex and leak into
      // Description. Fails if the newline normalization in parseDirectiveLines is reverted.
      name: "CRLF directive lines classify correctly, no stray carriage return",
      styleText: "noir",
      userPrompt: "Setting: alley\r\nAngle: low\r\nsomething",
      want: "Style: noir\nSetting: alley\nAngle: low\nDescription: something",
    },
    {
      name: "CRLF prose-then-directive splices cleanly",
      styleText: "noir",
      userPrompt: "a detective\r\nSetting: a foggy alley",
      want: "Style: noir\nSetting: a foggy alley\nDescription: a detective",
    },
    {
      name: "foreign directive preserved as sibling, remainder → Description",
      styleText: "noir",
      userPrompt: "a detective in the rain\nSetting: a foggy alley",
      want: "Style: noir\nSetting: a foggy alley\nDescription: a detective in the rain",
    },
    {
      name: "multiple foreign directives preserved as siblings in order",
      styleText: "noir",
      userPrompt: "a detective\nSetting: a foggy alley\nAngle: low angle\nLighting: single streetlamp",
      want: "Style: noir\nSetting: a foggy alley\nAngle: low angle\nLighting: single streetlamp\nDescription: a detective",
    },
    {
      name: "own Style line merges — catalog first, user words appended after comma",
      styleText: "oil painting",
      userPrompt: "Style: loose brushwork\na castle on a hill",
      want: "Style: oil painting, loose brushwork\nDescription: a castle on a hill",
    },
    {
      name: "own Style line merges alongside other foreign directives",
      styleText: "noir",
      userPrompt: "Style: high contrast\nSetting: alley\na detective",
      want: "Style: noir, high contrast\nSetting: alley\nDescription: a detective",
    },
    {
      name: "empty userPrompt with a style → Style only, no Description",
      styleText: "cinematic",
      userPrompt: "",
      want: "Style: cinematic",
    },
    {
      name: "whitespace-only userPrompt with a style → Style only, no Description",
      styleText: "cinematic",
      userPrompt: "   \n  \t",
      want: "Style: cinematic",
    },
    {
      name: "styleText is trimmed before templating",
      styleText: "  cinematic watercolor  ",
      userPrompt: "a fox",
      want: "Style: cinematic watercolor\nDescription: a fox",
    },
    {
      name: "unicode content is preserved in Description",
      styleText: "浮世絵",
      userPrompt: "富士山と桜、朝の光",
      want: "Style: 浮世絵\nDescription: 富士山と桜、朝の光",
    },
    {
      name: "unicode prose alongside a foreign directive",
      styleText: "浮世絵",
      userPrompt: "富士山と桜\nSetting: 早朝",
      want: "Style: 浮世絵\nSetting: 早朝\nDescription: 富士山と桜",
    },
    {
      name: "leading whitespace before a directive key still detects it",
      styleText: "noir",
      userPrompt: "a detective\n   Angle: low",
      want: "Style: noir\nAngle: low\nDescription: a detective",
    },
    {
      name: "directive-only prompt (no prose) → Style + siblings, no Description",
      styleText: "noir",
      userPrompt: "Setting: alley\nAngle: low",
      want: "Style: noir\nSetting: alley\nAngle: low",
    },
    {
      name: "own bare Style line (no user words) does not append a trailing comma",
      styleText: "oil painting",
      userPrompt: "Style:\na castle",
      want: "Style: oil painting\nDescription: a castle",
    },
    {
      name: "multi-line prose remainder is preserved across lines",
      styleText: "noir",
      userPrompt: "a detective\nunder a streetlamp\nSetting: alley",
      want: "Style: noir\nSetting: alley\nDescription: a detective\nunder a streetlamp",
    },
  ];

  for (const { name, styleText, userPrompt, want } of cases) {
    it(name, () => {
      expect(composeStyledPrompt({ styleText, userPrompt })).toBe(want);
    });
  }

  it("returns the untouched userPrompt value when styleText is empty (identity, not coercion)", () => {
    const userPrompt = "a fox";
    expect(composeStyledPrompt({ styleText: "", userPrompt })).toBe(userPrompt);
  });
});

describe("STYLE_DIRECTIVE_KEYS", () => {
  it("exposes the recognized directive set", () => {
    expect(STYLE_DIRECTIVE_KEYS).toEqual([
      "Style",
      "Description",
      "Setting",
      "Environment",
      "Angle",
      "Lighting",
      "Camera",
      "Mood",
      "Composition",
      "Negative",
    ]);
  });
});

describe("parseDirectiveLines", () => {
  it("classifies directive vs prose lines", () => {
    const parsed = parseDirectiveLines("a detective\nSetting: alley\nAngle: low");
    expect(parsed).toEqual([
      { type: "prose", key: null, content: "a detective", raw: "a detective" },
      { type: "directive", key: "Setting", content: "alley", raw: "Setting: alley" },
      { type: "directive", key: "Angle", content: "low", raw: "Angle: low" },
    ]);
  });

  it("does not treat a mid-sentence colon as a directive", () => {
    const parsed = parseDirectiveLines("a portrait: dramatic");
    expect(parsed).toEqual([{ type: "prose", key: null, content: "a portrait: dramatic", raw: "a portrait: dramatic" }]);
  });

  it("does not treat an unrecognized capitalized key as a directive", () => {
    const parsed = parseDirectiveLines("Note: hello");
    expect(parsed[0].type).toBe("prose");
  });

  it("does not treat a RECOGNIZED key mid-line as a directive (line-anchor)", () => {
    // "Lighting" is recognized but not at line start → must stay prose. This discriminates the
    // `^` anchor: with the anchor removed the regex would match mid-line and misclassify.
    const parsed = parseDirectiveLines("a photo with dramatic Lighting: soft");
    expect(parsed).toEqual([
      {
        type: "prose",
        key: null,
        content: "a photo with dramatic Lighting: soft",
        raw: "a photo with dramatic Lighting: soft",
      },
    ]);
  });

  it("classifies CRLF-separated directive lines without a trailing carriage return", () => {
    // Non-final CRLF lines would carry a trailing "\r" if newlines were split on "\n" only,
    // failing the anchored regex and leaking "\r" into content. This pins issue 1.
    const parsed = parseDirectiveLines("Setting: alley\r\nAngle: low\r\nsomething");
    expect(parsed).toEqual([
      { type: "directive", key: "Setting", content: "alley", raw: "Setting: alley" },
      { type: "directive", key: "Angle", content: "low", raw: "Angle: low" },
      { type: "prose", key: null, content: "something", raw: "something" },
    ]);
  });
});

// sc-13133: the composed-prompt budget the studio spends against the backend cap. The value is
// measured on the COMPOSED string (what is sent), and length is counted in Unicode scalar values
// to match rust-api's `prompt.chars().count()` bound.
describe("promptBudget", () => {
  it("pins the backend cap value", () => {
    // The number rust-api enforces (apps/rust-api/src/generation.rs — `> 4000`). A drift here means
    // the studio would gate at a different length than the server rejects at.
    expect(PROMPT_MAX_CHARS).toBe(4000);
  });

  it("reports length, remaining, and not-over for an under-cap string", () => {
    const budget = promptBudget("a".repeat(3820));
    expect(budget).toEqual({ length: 3820, max: 4000, remaining: 180, over: false });
  });

  it("is exactly at the cap → remaining 0, not over (the cap is inclusive)", () => {
    const budget = promptBudget("a".repeat(4000));
    expect(budget.length).toBe(4000);
    expect(budget.remaining).toBe(0);
    expect(budget.over).toBe(false);
  });

  it("one past the cap → over, negative remaining", () => {
    const budget = promptBudget("a".repeat(4001));
    expect(budget.length).toBe(4001);
    expect(budget.remaining).toBe(-1);
    expect(budget.over).toBe(true);
  });

  it("counts Unicode scalar values, not UTF-16 code units (matches Rust chars().count())", () => {
    // An astral emoji is ONE scalar value but TWO UTF-16 code units. A plain `.length` would report
    // 2 and over-count; the budget must report 1 so it agrees with the backend's char count.
    expect(promptBudget("😀").length).toBe(1);
    expect(promptBudget("😀".repeat(10)).length).toBe(10);
  });

  it("treats a null/undefined composed prompt as empty", () => {
    expect(promptBudget(undefined)).toEqual({ length: 0, max: 4000, remaining: 4000, over: false });
    expect(promptBudget(null).length).toBe(0);
  });
});
