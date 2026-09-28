import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EditorGuides } from "./EditorGuides.jsx";

let root;
let container;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.unstubAllGlobals(); });
const click = (element) => act(() => element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true })));

describe("Video Editor guides", () => {
  it("opens bundled script help offline, switches guides, and keeps document links inside the dialog", () => {
    const fetch = vi.fn(() => Promise.reject(new Error("offline")));
    vi.stubGlobal("fetch", fetch);
    act(() => root.render(<EditorGuides />));
    click(container.querySelector("button"));
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.textContent).toContain("Copy-and-paste first test");
    expect(dialog.querySelector("pre").textContent).toContain("small red box");
    click(dialog.querySelector('a[href="#editor-guide-operator"]'));
    expect(dialog.querySelector("h1").textContent).toBe("Film Editor operator guide");
    expect(dialog.querySelector('button[aria-pressed="true"]').textContent).toBe("Film Editor operator guide");
    click(dialog.querySelector('a[href="#editor-guide-script"]'));
    expect(dialog.querySelector("h1").textContent).toBe("Writing your first film script");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("traps keyboard focus, closes with Escape, and returns focus to the guide link", () => {
    act(() => root.render(<EditorGuides />));
    const trigger = container.querySelector("button");
    trigger.focus(); click(trigger);
    const dialog = document.querySelector('[role="dialog"]');
    expect(document.activeElement).toBe(dialog);
    act(() => dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true })));
    expect(document.activeElement.textContent).toBe("Close");
    act(() => dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true })));
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
    click(trigger);
    click(document.querySelector(".modal-close"));
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });
});
