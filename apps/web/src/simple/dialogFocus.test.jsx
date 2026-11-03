import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleSheet } from "./SimpleSheet.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { SimplePreview } from "./SimplePreview.jsx";

let container, root, opener;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  opener = document.createElement("button");
  document.body.appendChild(opener);
  opener.focus();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  opener.remove();
});

describe.each(["sheet", "preview"])("%s dialog focus", (kind) => {
  it("traps Tab, closes on Escape and restores the opener", () => {
    const onClose = vi.fn();
    act(() => root.render(kind === "sheet" ? (
      <SimpleSheet title="Choose" onClose={onClose}><button>First</button><button>Last</button></SimpleSheet>
    ) : (
      <AppContext.Provider value={{ audioModels: [] }}>
        <SimpleUiContext.Provider value={{ toast: () => {} }}><SimplePreview asset={{ id: "a", type: "image", status: {} }} onClose={onClose} onUseAsReference={() => {}} onToggleFavorite={() => {}} onDelete={() => {}} /></SimpleUiContext.Provider>
      </AppContext.Provider>
    )));
    const dialog = container.querySelector('[role="dialog"]');
    const buttons = [...dialog.querySelectorAll('button:not([disabled])')];
    expect(document.activeElement).toBe(dialog);
    buttons.at(-1).focus();
    act(() => buttons.at(-1).dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true })));
    expect(document.activeElement).toBe(buttons[0]);
    act(() => buttons[0].dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true })));
    expect(document.activeElement).toBe(buttons.at(-1));
    act(() => dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true })));
    expect(onClose).toHaveBeenCalledOnce();
    act(() => root.render(null));
    expect(document.activeElement).toBe(opener);
  });
});
