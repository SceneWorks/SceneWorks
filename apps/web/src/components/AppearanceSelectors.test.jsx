import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { useAppearance } from "../hooks/useAppearance.js";
import { AppearanceSelectors, AppearanceNotes } from "./AppearanceSelectors.jsx";

let root, container;
function Harness({ model }) {
  const control = useAppearance({ model });
  return <><AppearanceSelectors control={control} /><AppearanceNotes control={control} /><output>{JSON.stringify(control.selection)}</output></>;
}
afterEach(() => { act(() => root?.unmount()); container?.remove(); });
async function render(model) {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  if (!root) { container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container); }
  await act(async () => root.render(<Harness model={model} />));
}
async function click(selector) { await act(async () => container.querySelector(selector).click()); }
describe("Appearance selector interactions", () => {
  it("supports searching, multiple looks, keyboard dismissal and model-compatible clearing", async () => {
    root = null;
    await render({ id: "moss_image_base" });
    await click('button[aria-label="Looks"]');
    const search = container.querySelector('input[aria-label="Search Looks"]');
    expect(document.activeElement).toBe(search);
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(search, "light leaks");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(container.querySelectorAll('.appearance-option')).toHaveLength(1);
    await click('input[type="checkbox"]');
    expect(container.querySelector("output").textContent).toContain("look-light-leaks");
    await act(async () => search.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(container.querySelector('button[aria-label="Looks"]'));
    await click('button[aria-label="Looks"]');
    const muted = [...container.querySelectorAll('.appearance-option')].find((label) => label.textContent.includes("Cool muted"));
    await act(async () => muted.querySelector("input").click());
    expect(JSON.parse(container.querySelector("output").textContent).looks).toHaveLength(2);
    await render({ id: "krea_2_turbo" });
    expect(JSON.parse(container.querySelector("output").textContent).looks).toEqual(["look-cool-muted"]);
    expect(container.querySelector('[role="status"]').textContent).toContain("Cleared");
    await render({ id: "anima", captionStyle: "tags" });
    expect(container.querySelector('button[aria-label="Looks"]')).toBeNull();
    expect(JSON.parse(container.querySelector("output").textContent).looks).toEqual([]);
  });
});
