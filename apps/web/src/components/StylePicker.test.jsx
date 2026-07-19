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

// sc-13130: the Style Catalog picker — searchable, grouped, single-select, clearable to "None".
const GROUPS = [
  {
    id: "anime-style",
    name: "Anime Style",
    styles: [
      { id: "ghibli-style", name: "Ghibli Style", prompt: "gentle watercolor animation" },
      { id: "shonen-style", name: "Shonen Style", prompt: "bold action lines" },
    ],
  },
  {
    id: "photography",
    name: "Photography",
    styles: [{ id: "film-noir", name: "Film Noir", prompt: "high-contrast monochrome" }],
  },
];

describe("StylePicker (sc-13130)", () => {
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
  const optionButtons = () =>
    [...container.querySelectorAll('.style-picker-menu [role="option"]')];
  const groupLabels = () =>
    [...container.querySelectorAll(".style-picker-group-label")].map((el) => el.textContent);

  it("shows 'None' pass-through on the closed pill when nothing is selected", async () => {
    await render();
    expect(pill().textContent).toContain("None");
    expect(pill().getAttribute("aria-expanded")).toBe("false");
  });

  it("reflects the selected style's name on the pill", async () => {
    await render({ selectedId: "ghibli-style" });
    expect(pill().textContent).toContain("Ghibli Style");
  });

  it("renders group headers and every style as an option when opened", async () => {
    await render();
    await openMenu();
    expect(pill().getAttribute("aria-expanded")).toBe("true");
    expect(groupLabels()).toEqual(["Anime Style", "Photography"]);
    const labels = optionButtons().map((b) => b.textContent);
    // "None" + the three catalog styles.
    expect(labels.some((t) => t.includes("None"))).toBe(true);
    expect(labels.some((t) => t.includes("Ghibli Style"))).toBe(true);
    expect(labels.some((t) => t.includes("Shonen Style"))).toBe(true);
    expect(labels.some((t) => t.includes("Film Noir"))).toBe(true);
  });

  it("filters styles by name via the search box (and drops empty groups)", async () => {
    await render();
    await openMenu();
    const search = container.querySelector(".style-picker-search");
    await act(async () => setInputValue(search, "noir"));
    // Only Photography survives; Anime Style group header is gone.
    expect(groupLabels()).toEqual(["Photography"]);
    const labels = optionButtons().map((b) => b.textContent);
    expect(labels.some((t) => t.includes("Film Noir"))).toBe(true);
    expect(labels.some((t) => t.includes("Ghibli"))).toBe(false);
  });

  it("shows an empty state when nothing matches the search", async () => {
    await render();
    await openMenu();
    const search = container.querySelector(".style-picker-search");
    await act(async () => setInputValue(search, "zzzznope"));
    expect(container.querySelector(".compact-selector-empty")).toBeTruthy();
    // "None" is still offered even with no matches.
    expect(optionButtons().some((b) => b.textContent.includes("None"))).toBe(true);
  });

  it("single-selects a style by id and closes the menu", async () => {
    const onSelect = vi.fn();
    await render({ onSelect });
    await openMenu();
    const shonen = optionButtons().find((b) => b.textContent.includes("Shonen Style"));
    await act(async () => shonen.click());
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith("shonen-style");
    // Menu closed after selection.
    expect(container.querySelector(".style-picker-menu")).toBeNull();
  });

  it("clears to None (null) via the None option", async () => {
    const onSelect = vi.fn();
    await render({ selectedId: "ghibli-style", onSelect });
    await openMenu();
    const none = optionButtons().find((b) => b.textContent.includes("None"));
    await act(async () => none.click());
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("marks the active option with aria-selected", async () => {
    await render({ selectedId: "film-noir" });
    await openMenu();
    const active = optionButtons().find((b) => b.getAttribute("aria-selected") === "true");
    expect(active.textContent).toContain("Film Noir");
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

  it("handles the full shipped catalog (8 groups) without error", async () => {
    await render({ groups: STYLE_GROUPS });
    await openMenu();
    expect(groupLabels()).toHaveLength(8);
  });
});
