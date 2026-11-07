import { describe, expect, it } from "vitest";
import { APPEARANCE_CATALOG, appearanceOptions, resolveAppearance, pruneAppearance, composeNegativePrompt, appearanceWarnings } from "./appearanceCatalog.js";

describe("model-aware appearance catalog", () => {
  it("covers both complete guide catalogs and all lighting setups, including neon alternatives", () => {
    for (const [model, count] of [["moss_image_base", 52], ["z_image_turbo", 42], ["krea_2_turbo", 42]]) {
      expect(["camera", "film", "look"].flatMap((axis) => appearanceOptions(model, axis))).toHaveLength(count);
      expect(appearanceOptions(model, "lighting")).toHaveLength(27);
    }
    const ids = APPEARANCE_CATALOG.entries.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const entry of APPEARANCE_CATALOG.entries) {
      for (const variant of Object.values(entry.variants)) {
        expect(variant.prompt.trim()).not.toBe("");
        expect(APPEARANCE_CATALOG.sources[variant.source]).toMatch(/^https:/);
      }
    }
  });
  it("shares Z-Image/Krea recipes and retains MOSS-specific words and negatives", () => {
    const selection = { film: "film-kodak-tri-x-400", looks: ["look-moody-dark", "look-moody-dark"] };
    expect(resolveAppearance(selection, "z_image_turbo")).toEqual(resolveAppearance(selection, "krea_2_turbo"));
    expect(resolveAppearance(selection, { id: "my-finetune", family: "krea_2" }).directives).toEqual(resolveAppearance(selection, "krea_2_turbo").directives);
    const moss = resolveAppearance(selection, "moss_image_base");
    expect(moss.directives.Film).toContain("Make these visual traits distinct");
    expect(moss.entries.filter((entry) => entry.axis === "look")).toHaveLength(1);
    expect(moss.negative).toContain("chromatic color");
    expect(composeNegativePrompt("bad anatomy", moss.negative)).toBe(`bad anatomy, ${moss.negative}`);
    expect(appearanceWarnings(moss, "a red coat")[0]).toContain("monochrome");
  });
  it("retains compatible IDs and removes unsupported choices across models", () => {
    const selection = { film: "film-kodak-portra-400", looks: ["look-light-leaks"] };
    const { selection: next, removed } = pruneAppearance(selection, "krea_2_turbo");
    expect(next.film).toBe(selection.film);
    expect(next.looks).toEqual([]);
    expect(removed).toEqual(["look-light-leaks"]);
    expect(pruneAppearance(selection, { id: "anima", captionStyle: "tags" }).removed).toHaveLength(2);
  });
  it("refuses invalid IDs, wrong axes and models with no catalog", () => {
    expect(() => resolveAppearance({ camera: "film-kodak-portra-400" }, "z_image_turbo")).toThrow("unavailable");
    expect(() => resolveAppearance({ film: "missing" }, "moss_image_base")).toThrow("unavailable");
    expect(() => resolveAppearance({ film: "film-kodak-portra-400" }, "ltx_2_3")).toThrow("unavailable");
  });
});
