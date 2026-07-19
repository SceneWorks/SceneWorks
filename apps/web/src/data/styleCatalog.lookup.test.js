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

  // sc-13171: a GROUP id is now a first-class, selectable style — the two-level picker stores it
  // when the user picks a group's "overall" style. styleTextForId must resolve it to the group's
  // `description`, and findStyleById must expose enough for the breadcrumb.
  it("resolves a GROUP id to its description as the group-level style text", () => {
    for (const group of catalog.groups) {
      expect(styleTextForId(group.id)).toBe(group.description);
      const entry = findStyleById(group.id);
      expect(entry).toMatchObject({ id: group.id, name: group.name, prompt: group.description, isGroup: true });
    }
  });

  // The single stored `styleId` must resolve unambiguously to exactly ONE style, so group ids and
  // sub-style ids share one global id-space and must never collide. (styleCatalog.js also enforces
  // this at module load; this pins it on the shipped data.)
  it("keeps group ids globally unique from every sub-style id", () => {
    const groupIds = catalog.groups.map((group) => group.id);
    const subStyleIds = new Set(catalog.groups.flatMap((group) => group.styles.map((s) => s.id)));
    const collisions = groupIds.filter((id) => subStyleIds.has(id));
    expect(collisions).toEqual([]);
    // Combined id-space is fully unique too.
    const combined = [...groupIds, ...subStyleIds];
    expect(new Set(combined).size).toBe(combined.length);
  });
});
