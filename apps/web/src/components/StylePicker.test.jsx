import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StylePicker } from "./StylePicker.jsx";
import { STYLE_GROUPS } from "../data/styleCatalog.js";

// Set a controlled input's value the way React expects (bypass its value tracker), then fire the
// events onChange listens for. Mirrors the shared helper in BatchPromptPanel.test.jsx.
function setInputValue(el, value) {
  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

// sc-13171: the Style Catalog picker is now a TWO-LEVEL cascade — pick a group, then a style within
// it. The group's own "overall" style is the first (selectable) option inside the group and is
// stored as the GROUP id. "None" (pass-through) and the single value/onSelect(styleId) contract are
// preserved from the sc-13130 flat picker.
const GROUPS = [
  {
    id: "anime-style",
    name: "Anime Style",
    description: "broad anime look",
    styles: [
      { id: "ghibli-style", name: "Ghibli Style", prompt: "gentle watercolor animation" },
      { id: "shonen-style", name: "Shonen Style", prompt: "bold action lines" },
    ],
  },
  {
    id: "photography",
    name: "Photography",
    description: "broad photographic look",
    styles: [{ id: "film-noir", name: "Film Noir", prompt: "high-contrast monochrome" }],
  },
];

describe("StylePicker (sc-13171 two-level)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  function render(props = {}) {
    return act(async () =>
      root.render(<StylePicker groups={GROUPS} selectedId={null} onSelect={() => {}} {...props} />),
    );
  }

  const pill = () => container.querySelector(".compact-selector-pill");
  const openMenu = async () => act(async () => pill().click());
  const optionButtons = () => [...container.querySelectorAll('.style-picker-menu [role="option"]')];
  const groupNavs = () => [...container.querySelectorAll(".style-picker-group-nav")];
  const clickText = async (nodes, text) => {
    const el = nodes.find((b) => b.textContent.includes(text));
    await act(async () => el.click());
  };

  it("shows 'None' pass-through on the closed pill when nothing is selected", async () => {
    await render();
    expect(pill().textContent).toContain("None");
    expect(pill().getAttribute("aria-expanded")).toBe("false");
  });

  it("shows a group › style breadcrumb on the pill for a sub-style selection", async () => {
    await render({ selectedId: "ghibli-style" });
    expect(pill().textContent).toContain("Ghibli Style");
    expect(pill().textContent).toContain("Anime Style › Ghibli Style");
  });

  it("shows a group — general breadcrumb on the pill for a group-level selection", async () => {
    await render({ selectedId: "anime-style" });
    expect(pill().textContent).toContain("Anime Style (overall)");
    expect(pill().textContent).toContain("Anime Style — general");
  });

  it("level 1: shows None plus one nav row per group (no sub-styles yet)", async () => {
    await render();
    await openMenu();
    expect(pill().getAttribute("aria-expanded")).toBe("true");
    const navLabels = groupNavs().map((b) => b.textContent);
    expect(navLabels.some((t) => t.includes("Anime Style"))).toBe(true);
    expect(navLabels.some((t) => t.includes("Photography"))).toBe(true);
    // Sub-styles are NOT shown at level 1.
    expect(container.textContent).not.toContain("Ghibli Style");
    // Only "None" is a selectable option at level 1.
    expect(optionButtons().map((b) => b.textContent)).toEqual([
      expect.stringContaining("None"),
    ]);
  });

  it("level 2: entering a group reveals its 'overall' first, then its sub-styles", async () => {
    await render();
    await openMenu();
    await clickText(groupNavs(), "Anime Style");
    const labels = optionButtons().map((b) => b.textContent);
    // First option is the group-level "overall"; then the sub-styles.
    expect(labels[0]).toContain("Anime Style (overall)");
    expect(labels.some((t) => t.includes("Ghibli Style"))).toBe(true);
    expect(labels.some((t) => t.includes("Shonen Style"))).toBe(true);
    // Photography's style is NOT present — we scoped to one group.
    expect(labels.some((t) => t.includes("Film Noir"))).toBe(false);
    // Breadcrumb header names the group.
    expect(container.querySelector(".style-picker-crumb").textContent).toBe("Anime Style");
  });

  it("selects a group's 'overall' style using the GROUP id", async () => {
    const onSelect = vi.fn();
    await render({ onSelect });
    await openMenu();
    await clickText(groupNavs(), "Anime Style");
    await clickText(optionButtons(), "Anime Style (overall)");
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith("anime-style");
    expect(container.querySelector(".style-picker-menu")).toBeNull();
  });

  it("single-selects a sub-style by id and closes the menu", async () => {
    const onSelect = vi.fn();
    await render({ onSelect });
    await openMenu();
    await clickText(groupNavs(), "Anime Style");
    await clickText(optionButtons(), "Shonen Style");
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith("shonen-style");
    expect(container.querySelector(".style-picker-menu")).toBeNull();
  });

  it("has a back control that returns from a group to the group list", async () => {
    await render();
    await openMenu();
    await clickText(groupNavs(), "Anime Style");
    expect(container.querySelector(".style-picker-crumb-row")).toBeTruthy();
    const back = container.querySelector(".style-picker-back");
    await act(async () => back.click());
    // Back at level 1: group navs visible again, no crumb header.
    expect(container.querySelector(".style-picker-crumb-row")).toBeNull();
    expect(groupNavs().length).toBe(2);
  });

  it("opens directly into the selected style's group for easy change", async () => {
    await render({ selectedId: "ghibli-style" });
    await openMenu();
    // Jumps straight to level 2 of Anime Style; the active option is marked.
    expect(container.querySelector(".style-picker-crumb").textContent).toBe("Anime Style");
    const active = optionButtons().find((b) => b.getAttribute("aria-selected") === "true");
    expect(active.textContent).toContain("Ghibli Style");
  });

  it("clears to None (null) via the None option at level 1", async () => {
    const onSelect = vi.fn();
    // selectedId null so we open at level 1 where None lives.
    await render({ onSelect });
    await openMenu();
    await clickText(optionButtons(), "None");
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("global search jumps across all groups, including group 'overall' entries", async () => {
    await render();
    await openMenu();
    const search = container.querySelector(".style-picker-search");
    await act(async () => setInputValue(search, "noir"));
    const labels = optionButtons().map((b) => b.textContent);
    expect(labels.some((t) => t.includes("Film Noir"))).toBe(true);
    expect(labels.some((t) => t.includes("Ghibli"))).toBe(false);
    // The result row shows its owning group as a breadcrumb subtitle.
    const noir = optionButtons().find((b) => b.textContent.includes("Film Noir"));
    expect(noir.textContent).toContain("Photography");
  });

  it("search matches a group's 'overall' entry and selects the GROUP id", async () => {
    const onSelect = vi.fn();
    await render({ onSelect });
    await openMenu();
    const search = container.querySelector(".style-picker-search");
    await act(async () => setInputValue(search, "overall"));
    await clickText(optionButtons(), "Anime Style (overall)");
    expect(onSelect).toHaveBeenCalledWith("anime-style");
  });

  it("shows an empty state when nothing matches the search", async () => {
    await render();
    await openMenu();
    const search = container.querySelector(".style-picker-search");
    await act(async () => setInputValue(search, "zzzznope"));
    expect(container.querySelector(".compact-selector-empty")).toBeTruthy();
  });

  it("is keyboard-usable: focuses search on open and closes on Escape", async () => {
    await render();
    await openMenu();
    expect(document.activeElement).toBe(container.querySelector(".style-picker-search"));
    await act(async () => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(container.querySelector(".style-picker-menu")).toBeNull();
    expect(pill().getAttribute("aria-expanded")).toBe("false");
  });

  it("handles the full shipped catalog (8 groups) at level 1", async () => {
    await render({ groups: STYLE_GROUPS });
    await openMenu();
    expect(groupNavs()).toHaveLength(8);
  });
});
