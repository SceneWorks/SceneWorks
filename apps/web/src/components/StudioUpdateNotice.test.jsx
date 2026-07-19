import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot } from "../testUtils/dom.js";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "./StudioUpdateNotice.jsx";

describe("Studio update indicators", () => {
  let container;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("marks stale picker entries without disabling selection", () => {
    expect(updateOptionLabel({ name: "Current", updateAvailable: false })).toBe("Current");
    expect(updateOptionLabel({ name: "Stale", updateAvailable: true })).toBe("Stale • update");
  });

  it("shows a dismissible notice and invokes the existing download action", async () => {
    const onUpdate = vi.fn();
    const stale = { id: "stale", name: "Stale", updateAvailable: true };
    await act(async () => root.render(<><StudioUpdateBadge item={stale} /><StudioUpdateNotice item={stale} onUpdate={onUpdate} /></>));
    expect(container.textContent).toContain("keep generating");
    const buttons = container.querySelectorAll("button");
    await click(buttons[0]);
    expect(onUpdate).toHaveBeenCalledWith(stale);
    await click(buttons[1]);
    expect(container.querySelector(".studio-update-notice")).toBeNull();
  });

  it("clears both indicators when the live catalog clears updateAvailable", async () => {
    const render = (item) => root.render(<><StudioUpdateBadge item={item} /><StudioUpdateNotice item={item} onUpdate={() => {}} /></>);
    await act(async () => render({ id: "m", name: "Model", updateAvailable: true }));
    expect(container.querySelector(".studio-update-badge")).toBeTruthy();
    await click(container.querySelector('[aria-label="Dismiss Model update notice"]'));
    await act(async () => render({ id: "m", name: "Model", updateAvailable: false }));
    expect(container.querySelector(".studio-update-badge")).toBeNull();
    expect(container.querySelector(".studio-update-notice")).toBeNull();
    await act(async () => render({ id: "m", name: "Model", updateAvailable: true }));
    expect(container.querySelector(".studio-update-notice")).toBeTruthy();
  });

  it("shows a fresh notice when selection moves between stale items", async () => {
    const render = (item) => root.render(<StudioUpdateNotice item={item} onUpdate={() => {}} />);
    await act(async () => render({ id: "a", name: "A", updateAvailable: true }));
    await click(container.querySelector('[aria-label="Dismiss A update notice"]'));
    await act(async () => render({ id: "b", name: "B", updateAvailable: true }));
    expect(container.textContent).toContain("B has an update");
  });
});
