import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { DatasetCaptionWindow } from "./DatasetCaptionWindow.jsx";

it("measures responsive variable-height rows, anchors the viewport and disconnects observers", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const observers = new Set();
  let width = 1050;
  let changedHeight = 340;
  class Observer {
    constructor(callback) { this.callback = callback; observers.add(this); }
    observe(element) { this.element = element; }
    disconnect() { observers.delete(this); }
  }
  vi.stubGlobal("ResizeObserver", Observer);
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function () { return this.classList.contains("training-caption-window") ? width : 0; });
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(700);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function () {
    return { height: this.dataset.captionId === "0" ? changedHeight : 340 };
  });
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const assets = Array.from({ length: 300 }, (_, index) => ({ id: String(index) }));
  try {
    await act(async () => root.render(<DatasetCaptionWindow assets={assets} renderItem={(asset) => <article className="training-caption-card" data-caption-id={asset.id}><textarea aria-label={asset.id} /></article>} />));
    const item = (id) => [...container.querySelectorAll("div[data-caption-id]")].find((node) => node.dataset.captionId === String(id));
    expect(item(1).style.top).toBe("0px");
    expect(item(3).style.top).toBe("352px");
    expect(container.querySelectorAll("article").length).toBeLessThan(24);
    // Three columns share a row; a taller first card pushes the next row as a unit.
    changedHeight = 540;
    await act(async () => [...observers].forEach((observer) => observer.callback()));
    expect(item(3).style.top).toBe("552px");
    const viewport = container.querySelector(".training-caption-window");
    await act(async () => { viewport.scrollTop = 552 + 352 * 30; viewport.dispatchEvent(new Event("scroll", { bubbles: true })); });
    const anchor = item(93);
    expect(anchor).toBeDefined();
    // Resize to one column. The first visible image stays at the viewport anchor.
    width = 340;
    await act(async () => [...observers].forEach((observer) => observer.callback()));
    expect(item(93)).toBeDefined();
    expect(viewport.scrollTop).toBe(552 + 352 * 92);
    expect(container.querySelectorAll("article").length).toBeLessThan(12);
    // Filtering across the small/large threshold must preserve the actual input
    // node and browser focus, not merely restore its text after remounting.
    const render = (items) => <DatasetCaptionWindow assets={items} renderItem={(asset) =>
      <article className="training-caption-card" data-caption-id={asset.id}><textarea aria-label={asset.id} /><button disabled>Disabled action</button></article>} />;
    await act(async () => root.render(render(assets.slice(0, 25))));
    viewport.scrollTop = 0;
    await act(async () => viewport.dispatchEvent(new Event("scroll", { bubbles: true })));
    const input = container.querySelector('textarea[aria-label="0"]');
    await act(async () => input.focus());
    await act(async () => root.render(render(assets.slice(0, 24))));
    expect(document.activeElement).toBe(input);
    expect(input.isConnected).toBe(true);
    await act(async () => input.blur());
    await act(async () => root.render(render(assets)));
    await act(async () => {
      viewport.scrollTop = 352 * 50;
      viewport.dispatchEvent(new Event("scroll", { bubbles: true }));
    });
    const firstVisible = container.querySelector("article textarea");
    const priorId = String(Number(firstVisible.getAttribute("aria-label")) - 1);
    await act(async () => {
      firstVisible.focus();
      firstVisible.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true }));
    });
    expect(document.activeElement.getAttribute("aria-label")).toBe(priorId);


  } finally {
    act(() => root.unmount());
    expect(observers.size).toBe(0);
    container.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  }
});
