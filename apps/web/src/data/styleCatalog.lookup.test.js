import { describe, expect, it } from "vitest";

import catalog from "./styles.json";
import { STYLE_GROUPS, findStyleById, styleTextForId } from "./styleCatalog.js";

// sc-13130: the runtime lookup that bridges the studio's `styleId` selection to the free-text
// `prompt` the composer consumes. (Catalog-vs-source drift is guarded separately in
// styleCatalog.test.js; this pins the accessor behavior the picker + payload fold rely on.)
describe("styleCatalog runtime lookup", () => {
  const firstStyle = catalog.groups[0].styles[0];

  it("exposes the 8 authored groups", () => {
    expect(STYLE_GROUPS).toBe(catalog.groups);
    expect(STYLE_GROUPS).toHaveLength(8);
  });

  it("resolves a known style id to its entry and prompt text", () => {
    expect(findStyleById(firstStyle.id)).toEqual(firstStyle);
    expect(styleTextForId(firstStyle.id)).toBe(firstStyle.prompt);
  });

  it("treats null / empty / unknown ids as pass-through (no style)", () => {
    expect(findStyleById(null)).toBeNull();
    expect(findStyleById("")).toBeNull();
    expect(findStyleById(undefined)).toBeNull();
    expect(findStyleById("not-a-real-style-id")).toBeNull();
    expect(styleTextForId(null)).toBeNull();
    expect(styleTextForId("not-a-real-style-id")).toBeNull();
  });

  it("indexes every style across every group (unique ids)", () => {
    const allIds = catalog.groups.flatMap((group) => group.styles.map((style) => style.id));
    expect(new Set(allIds).size).toBe(allIds.length);
    for (const id of allIds) {
      expect(styleTextForId(id)).toBeTruthy();
    }
  });
});
