import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { useAppearance } from "../hooks/useAppearance.js";
import { AppearanceSelectors, AppearanceNotes } from "./AppearanceSelectors.jsx";

let root, container;
function Harness({ model }) {
  const control = useAppearance({ model });
  return <form onSubmit={(event) => event.preventDefault()}><AppearanceSelectors control={control} /><AppearanceNotes control={control} /><output>{JSON.stringify(control.selection)}</output></form>;
}
afterEach(() => { act(() => root?.unmount()); container?.remove(); });
async function render(model) {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  if (!root) { container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container); }
  await act(async () => root.render(<Harness model={model} />));
}
async function click(selector) { await act(async () => container.querySelector(selector).click()); }
describe("Appearance selector interactions", () => {
  it("keeps Enter inside search and activates recipe inputs without submitting the generation form", async () => {
    root = null;
    await render({ id: "krea_2_turbo" });
    async function enter(input) {
      const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
      await act(async () => input.dispatchEvent(event));
      expect(event.defaultPrevented).toBe(true);
    }
    await click('button[aria-label="Camera"]');
    await enter(container.querySelector('input[type="search"]'));
    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(JSON.parse(container.querySelector("output").textContent).camera).toBeNull();
    const radio = container.querySelector('input[type="radio"]');
    const cameraName = radio.closest("label").querySelector("strong").textContent;
    await enter(radio);
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(container.querySelector('button[aria-label="Camera"]').textContent).toContain(cameraName);
    expect(document.activeElement).toBe(container.querySelector('button[aria-label="Camera"]'));
    await click('button[aria-label="Looks"]');
    const [first, second] = container.querySelectorAll('input[type="checkbox"]');
    await enter(first);
    await enter(second);
    expect(JSON.parse(container.querySelector("output").textContent).looks).toHaveLength(2);
    await enter(first);
    expect(JSON.parse(container.querySelector("output").textContent).looks).toHaveLength(1);
    const done = [...container.querySelectorAll("button")].find((button) => button.textContent === "Done");
    const doneEnter = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    await act(async () => done.dispatchEvent(doneEnter));
    expect(doneEnter.defaultPrevented).toBe(false);
  });
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
