import { describe, expect, it } from "vitest";
import { presetMatchesModel } from "./presetUtils.js";

const ltx = { id: "ltx_2_3", family: "ltx-video" };
const ltxEros = { id: "ltx_2_3_eros", family: "ltx-video" };
const sdxl = { id: "sdxl", family: "sdxl" };
const catalog = [ltx, ltxEros, sdxl];

describe("presetMatchesModel", () => {
  it("matches when the preset pins no model", () => {
    expect(presetMatchesModel({ id: "p" }, ltxEros, catalog)).toBe(true);
  });

  it("matches when the selected model has no id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, {}, catalog)).toBe(true);
  });

  it("matches on exact model id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltx, catalog)).toBe(true);
  });

  it("matches a sibling model in the same family (ltx_2_3 preset under ltx_2_3_eros)", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros, catalog)).toBe(true);
  });

  it("does not match across families", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, sdxl, catalog)).toBe(false);
  });

  it("stays strict (no family fallback) when the catalog is unavailable", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros)).toBe(false);
  });
});
