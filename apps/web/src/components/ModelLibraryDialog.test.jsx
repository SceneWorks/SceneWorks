import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModelLibraryDialog } from "./ModelLibraryDialog.jsx";

const CONTEXT = {
  schemaVersion: 1,
  availability: "installed_external_unavailable",
  modelId: "z_image",
  modelName: "Z-Image Turbo",
  configuredLibraryPath: "/Volumes/Models/hf/hub",
  expectedLibraryPath: "/Volumes/Models/hf/hub",
  expectedVolumeId: "macos-volume:abc",
};

function blocked(overrides = {}) {
  return { status: "blocked", context: CONTEXT, hint: "", error: "", ...overrides };
}

describe("ModelLibraryDialog", () => {
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
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  async function render(props) {
    await act(async () => {
      root.render(
        <ModelLibraryDialog
          autoProbeMs={0}
          canRelocate
          onCancel={() => {}}
          onRelocate={() => {}}
          onRetry={() => {}}
          {...props}
        />,
      );
    });
    return document.body.querySelector(".model-library-modal");
  }

  function buttons(dialog) {
    return [...dialog.querySelectorAll("button")].reduce((map, button) => {
      map[button.textContent] = button;
      return map;
    }, {});
  }

  it("renders nothing while the gate is idle", async () => {
    const dialog = await render({ state: { status: "idle", context: null } });
    expect(dialog).toBeNull();
  });

  it("names the model and the expected library, and never shows raw error text", async () => {
    const dialog = await render({ state: blocked() });
    expect(dialog.textContent).toContain("Z-Image Turbo");
    expect(dialog.textContent).toContain("/Volumes/Models/hf/hub");
    expect(dialog.textContent).not.toMatch(/ENOENT|No such file|os error/i);
    // Accessible dialog wiring comes from the shared Modal primitive.
    expect(dialog.getAttribute("role")).toBe("dialog");
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.getAttribute("aria-labelledby")).toBe("model-library-title");
    expect(document.getElementById("model-library-title")).not.toBeNull();
    expect(dialog.querySelector('[role="status"]').getAttribute("aria-live")).toBe(
      "polite",
    );
    expect(document.activeElement).toBe(dialog);
  });

  it("offers exactly the three recovery exits, and wires each to the gate", async () => {
    const onRetry = vi.fn();
    const onRelocate = vi.fn();
    const onCancel = vi.fn();
    const dialog = await render({ state: blocked(), onRetry, onRelocate, onCancel });
    const named = buttons(dialog);
    expect(Object.keys(named)).toEqual([
      "Cancel",
      "Choose a different library location",
      "Connect drive and retry",
    ]);

    await act(async () => named["Connect drive and retry"].click());
    await act(async () => named["Choose a different library location"].click());
    await act(async () => named.Cancel.click());
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRelocate).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("closes on Escape, which the gate treats as cancel", async () => {
    const onCancel = vi.fn();
    const dialog = await render({ state: blocked(), onCancel });
    await act(async () => {
      dialog.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
    });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("disables every exit while an attempt is in flight, so a click cannot double-fire", async () => {
    const onRetry = vi.fn();
    const dialog = await render({
      state: { ...blocked(), status: "retrying" },
      onRetry,
    });
    const named = buttons(dialog);
    expect(named["Connect drive and retry"].disabled).toBe(true);
    expect(named.Cancel.disabled).toBe(true);
    expect(dialog.querySelector('[role="status"]').textContent).toContain("Checking");
    await act(async () => named["Connect drive and retry"].click());
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("announces the seam's guidance for a still-missing library or a rejected folder", async () => {
    let dialog = await render({
      state: blocked({ hint: "That library is still not connected." }),
    });
    expect(dialog.querySelector('[role="status"]').textContent).toBe(
      "That library is still not connected.",
    );
    dialog = await render({
      state: blocked({ error: "That folder does not contain a SceneWorks model library." }),
    });
    expect(dialog.querySelector('[role="status"]').textContent).toContain(
      "does not contain a SceneWorks model library",
    );
  });

  // A library that is present but whose identity disagrees is not fixed by reconnecting anything,
  // so the prompt must not tell the user to. Same three exits; relocation leads.
  it("leads with relocation when the library is present but is not the recorded one", async () => {
    const dialog = await render({
      state: blocked({ context: { ...CONTEXT, libraryPresent: true } }),
    });
    expect(dialog.textContent).toContain("is on a different model library");
    expect(dialog.textContent).toContain("point SceneWorks at the library holding your models");
    expect(dialog.textContent).not.toContain("reconnect the drive");
    const named = buttons(dialog);
    expect(named["Choose a different library location"].className).toContain(
      "primary-action",
    );
    expect(named["Connect drive and retry"].className).not.toContain("primary-action");
    // Retrying is still offered — it is simply no longer the recommended answer.
    expect(named["Connect drive and retry"]).toBeTruthy();
  });

  it("keeps reconnect as the lead when the library is genuinely disconnected", async () => {
    const dialog = await render({ state: blocked() });
    expect(dialog.textContent).toContain("needs its model library");
    const named = buttons(dialog);
    expect(named["Connect drive and retry"].className).toContain("primary-action");
    expect(named["Choose a different library location"].className).not.toContain(
      "primary-action",
    );
  });

  it("replaces the relocate action with server guidance when relocation is not this app's to do", async () => {
    const dialog = await render({ state: blocked(), canRelocate: false });
    expect(Object.keys(buttons(dialog))).toEqual([
      "Cancel",
      "Connect drive and retry",
    ]);
    expect(dialog.textContent).toContain("desktop setting");
  });

  it("re-probes on a timer while blocked, and stops the moment an attempt is in flight", async () => {
    vi.useFakeTimers();
    const onRetry = vi.fn();
    await act(async () => {
      root.render(
        <ModelLibraryDialog
          autoProbeMs={1000}
          canRelocate
          onCancel={() => {}}
          onRelocate={() => {}}
          onRetry={onRetry}
          state={blocked()}
        />,
      );
    });
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });
    expect(onRetry).toHaveBeenCalledTimes(2);
    expect(onRetry).toHaveBeenCalledWith({ auto: true });

    // Once an attempt is running the poller is torn down: the gate's own guard would reject a
    // concurrent tick anyway, but not scheduling it keeps the two from racing at all.
    onRetry.mockClear();
    await act(async () => {
      root.render(
        <ModelLibraryDialog
          autoProbeMs={1000}
          canRelocate
          onCancel={() => {}}
          onRelocate={() => {}}
          onRetry={onRetry}
          state={{ ...blocked(), status: "retrying" }}
        />,
      );
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(onRetry).not.toHaveBeenCalled();
  });
});
