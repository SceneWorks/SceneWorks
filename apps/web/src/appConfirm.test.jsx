import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { appConfirm, useConfirm, ConfirmHost, normalizeConfirmOptions } from "./appConfirm.jsx";

// sc-11968: the shared, desktop-safe confirm. window.confirm silently no-ops / returns
// undefined inside the Tauri WebView, so these tests prove the promise-returning appConfirm
// resolves through a REAL React dialog (the shared Modal, portaled to <body>) — including
// when window.confirm is stubbed to a desktop-style no-op.

const flush = async () => {
  await act(async () => {
    for (let i = 0; i < 4; i += 1) await Promise.resolve();
  });
};

// Kick off a confirm INSIDE act (appConfirm synchronously calls setRequest on the host) and
// let the dialog render. The still-pending promise is stashed in a shared variable rather
// than RETURNED, so `await askConfirm(...)` can't flatten onto (and block on) it — awaiting
// an async fn that returns a pending promise would deadlock the test until a click.
let pending;
const askConfirm = async (options) => {
  await act(async () => {
    pending = appConfirm(options);
    for (let i = 0; i < 4; i += 1) await Promise.resolve();
  });
};

const dialog = () => document.body.querySelector(".app-confirm-modal");
const dialogButton = (label) =>
  [...document.body.querySelectorAll(".app-confirm-modal button")].find((b) => b.textContent === label);

describe("normalizeConfirmOptions", () => {
  it("treats a bare string as the message and fills sane defaults", () => {
    expect(normalizeConfirmOptions("Discard?")).toEqual({
      title: "Are you sure?",
      message: "Discard?",
      confirmLabel: "Confirm",
      cancelLabel: "Cancel",
      tone: "default",
    });
  });

  it("passes through provided fields and only honors the 'danger' tone", () => {
    expect(normalizeConfirmOptions({ title: "T", message: "M", confirmLabel: "Go", cancelLabel: "No", tone: "danger" })).toEqual({
      title: "T",
      message: "M",
      confirmLabel: "Go",
      cancelLabel: "No",
      tone: "danger",
    });
    expect(normalizeConfirmOptions({ tone: "weird" }).tone).toBe("default");
  });
});

describe("appConfirm with a mounted ConfirmHost (sc-11968)", () => {
  let container;
  let root;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => {
      root.render(<ConfirmHost />);
    });
    await flush();
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("renders nothing until a confirm is requested", () => {
    expect(dialog()).toBeNull();
  });

  it("shows a dialog with the title/message/labels and resolves true on Confirm", async () => {
    await askConfirm({
      title: "Close image?",
      message: "Discard and close?",
      confirmLabel: "Discard & close",
      cancelLabel: "Keep editing",
    });
    const p = pending;

    expect(dialog()).not.toBeNull();
    expect(document.body.textContent).toContain("Close image?");
    expect(document.body.textContent).toContain("Discard and close?");
    expect(dialogButton("Discard & close")).toBeTruthy();
    expect(dialogButton("Keep editing")).toBeTruthy();

    await act(async () => dialogButton("Discard & close").click());
    expect(await p).toBe(true);
    // The dialog closes once answered.
    expect(dialog()).toBeNull();
  });

  it("resolves false on Cancel", async () => {
    await askConfirm("Proceed?");
    const p = pending;
    await act(async () => dialogButton("Cancel").click());
    expect(await p).toBe(false);
    expect(dialog()).toBeNull();
  });

  it("resolves false on Escape and on a backdrop click", async () => {
    await askConfirm("Proceed?");
    const pEsc = pending;
    await act(async () => {
      document.body
        .querySelector(".app-confirm-modal")
        .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(await pEsc).toBe(false);

    await askConfirm("Proceed?");
    const pBackdrop = pending;
    await act(async () => {
      const backdrop = document.body.querySelector(".modal-backdrop");
      backdrop.dispatchEvent(new window.MouseEvent("mousedown", { bubbles: true }));
    });
    expect(await pBackdrop).toBe(false);
  });

  it("works even when window.confirm is a desktop-style no-op returning undefined", async () => {
    // Simulate the Tauri WebView: confirm exists but silently returns undefined. The dialog
    // path must be used instead, and Confirm must still resolve true.
    const noop = vi.fn(() => undefined);
    vi.stubGlobal("confirm", noop);
    window.confirm = noop;

    await askConfirm({ message: "Leave?", tone: "danger" });
    const p = pending;
    expect(dialog()).not.toBeNull();
    // The destructive tone styles the confirm button.
    expect(document.body.querySelector(".app-confirm-modal .danger-action")).toBeTruthy();

    await act(async () => dialogButton("Confirm").click());
    expect(await p).toBe(true);
    // The broken window.confirm was never consulted.
    expect(noop).not.toHaveBeenCalled();
  });

  it("cancels a superseded request so its awaiter never hangs", async () => {
    await askConfirm("First?");
    const first = pending;
    await askConfirm("Second?");
    const second = pending;
    // Opening the second resolves the first as cancelled.
    expect(await first).toBe(false);
    await act(async () => dialogButton("Confirm").click());
    expect(await second).toBe(true);
  });

  it("useConfirm() returns the same appConfirm function", async () => {
    let hookValue;
    function Probe() {
      hookValue = useConfirm();
      return null;
    }
    const probeContainer = document.createElement("div");
    document.body.appendChild(probeContainer);
    const probeRoot = createRoot(probeContainer);
    await act(async () => probeRoot.render(<Probe />));
    expect(hookValue).toBe(appConfirm);
    await act(async () => probeRoot.unmount());
    probeContainer.remove();
  });
});

describe("appConfirm fallback with NO host mounted", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("falls back to window.confirm's answer", async () => {
    window.confirm = vi.fn(() => true);
    expect(await appConfirm("Proceed?")).toBe(true);
    window.confirm = vi.fn(() => false);
    expect(await appConfirm("Proceed?")).toBe(false);
  });

  it("proceeds (resolves true) when no confirm is available at all", async () => {
    const original = window.confirm;
    window.confirm = undefined;
    expect(await appConfirm("Proceed?")).toBe(true);
    window.confirm = original;
  });
});
