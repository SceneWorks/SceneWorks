import { describe, expect, it } from "vitest";

import { ACCENTS } from "./accents.js";
import { extractAccentIds, renderThemeInit } from "./accentIds.js";
// Raw source text (Vite `?raw`) so the test parses the same bytes the build does.
import accentsSource from "./accents.js?raw";
import templateSource from "./theme-init.template.js?raw";

// Guards sc-8956: the pre-paint theme script's accent-id list must stay a
// derived copy of accents.js — never hand-maintained. If accents.js changes,
// the generated theme-init.js must change with it.

const EXPECTED_IDS = ACCENTS.map((a) => a.id);

describe("accent-id single source of truth", () => {
  it("extractAccentIds parses the same ids (and order) as ACCENTS", () => {
    expect(extractAccentIds(accentsSource)).toEqual(EXPECTED_IDS);
  });

  it("generated theme-init.js embeds exactly the accents.js id list", () => {
    const generated = renderThemeInit(templateSource, accentsSource);
    const match = generated.match(/const ACCENT_IDS = (\[[^\]]*\]);/);
    expect(match).not.toBeNull();
    expect(JSON.parse(match[1])).toEqual(EXPECTED_IDS);
  });

  it("leaves no unsubstituted accent-ids marker in the generated script", () => {
    expect(renderThemeInit(templateSource, accentsSource)).not.toContain("@accent-ids");
  });
});
