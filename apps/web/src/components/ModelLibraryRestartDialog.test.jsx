import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModelLibraryRestartDialog } from "./ModelLibraryRestartDialog.jsx";

const RELOCATION = {
  libraryRoot: "/Volumes/Models 1/hf/hub",
  hfHome: "/Volumes/Models 1/hf",
  status: { available: true },
  // The prompt path: the user was submitting something when the library was found unavailable.
  droppedSubmission: true,
};

describe("ModelLibraryRestartDialog (sc-19709)", () => {
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
  });

  async function render(props) {
    await act(async () => {
      root.render(
        <ModelLibraryRestartDialog
          canRestart
          onLater={() => {}}
          onRestart={() => {}}
          relocation={RELOCATION}
          {...props}
        />,
      );
    });
    return document.body.querySelector(".model-library-modal");
  }

  function button(dialog, label) {
    return [...dialog.querySelectorAll("button")].find(
      (item) => item.textContent === label,
    );
  }

  it("renders nothing until a relocation actually succeeded", async () => {
    expect(await render({ relocation: null })).toBeNull();
  });

  it("discloses the next-launch behavior and names the new location", async () => {
    const dialog = await render({});
    expect(dialog.textContent).toContain("after it restarts");
    expect(dialog.textContent).toContain("/Volumes/Models 1/hf");
    // The user must not think anything was copied or re-downloaded.
    expect(dialog.textContent).toContain("nothing was moved, copied, or downloaded");
    // …and must be told what became of the generation they actually started. Without this, the
    // missing job reads as a bug rather than the consequence of relocating.
    expect(dialog.textContent).toContain("was not queued");
    expect(dialog.textContent).toContain("Submit it again after SceneWorks restarts");
    expect(dialog.getAttribute("role")).toBe("dialog");
    expect(dialog.getAttribute("aria-labelledby")).toBe("model-library-restart-title");
    expect(dialog.querySelector('[role="status"]').getAttribute("aria-live")).toBe(
      "polite",
    );
  });

  // A relocation started from Settings → Storage dropped nothing: telling that user a generation
  // "was not queued" would invent a job they never started.
  it("omits the dropped-submission line for a relocation that had no submission", async () => {
    const dialog = await render({
      relocation: { ...RELOCATION, droppedSubmission: false },
    });
    expect(dialog.textContent).toContain("after it restarts");
    expect(dialog.textContent).toContain("/Volumes/Models 1/hf");
    expect(dialog.textContent).not.toContain("was not queued");
  });

  it("invokes the relaunch exactly once, however many times the button is clicked", async () => {
    const onRestart = vi.fn();
    const dialog = await render({ onRestart });
    const restart = button(dialog, "Restart now");
    await act(async () => {
      restart.click();
      restart.click();
    });
    await act(async () => restart.click());
    expect(onRestart).toHaveBeenCalledTimes(1);
    expect(button(dialog, "Restart now").disabled).toBe(true);
    expect(dialog.querySelector('[role="status"]').textContent).toContain("Restarting");
  });

  // The dialog stays mounted across relocations, so a restart that never launched must not latch it
  // permanently into "Restarting…" with both buttons disabled.
  it("leaves the restarting state when the relaunch rejects, and can be retried", async () => {
    const onRestart = vi.fn().mockRejectedValueOnce(new Error("relaunch refused"));
    const dialog = await render({ onRestart });
    await act(async () => button(dialog, "Restart now").click());
    await act(async () => {});

    expect(dialog.querySelector('[role="status"]').textContent).not.toContain("Restarting");
    expect(button(dialog, "Restart now").disabled).toBe(false);
    expect(button(dialog, "Later").disabled).toBe(false);
    await act(async () => button(dialog, "Restart now").click());
    expect(onRestart).toHaveBeenCalledTimes(2);
  });

  it("treats a false result from the relaunch as a failure too", async () => {
    const onRestart = vi.fn(async () => false);
    const dialog = await render({ onRestart });
    await act(async () => button(dialog, "Restart now").click());
    await act(async () => {});
    expect(button(dialog, "Restart now").disabled).toBe(false);
  });

  // Both the once-only latch and the restarting state are per-RELOCATION, not per-mount: this
  // component stays mounted, so without a reset the dead restarting state carries straight into the
  // NEXT relocation and restart can never fire again without reloading the app. Driven here with a
  // relaunch that never settles, so the reset is proven independently of the rejection path.
  it("resets for a new relocation after a restart that never completed", async () => {
    const onRestart = vi.fn(() => new Promise(() => {}));
    let dialog = await render({ onRestart });
    await act(async () => button(dialog, "Restart now").click());
    expect(dialog.querySelector('[role="status"]').textContent).toContain("Restarting");

    // The disclosure is dismissed, then a fresh relocation opens it again.
    await render({ onRestart, relocation: null });
    dialog = await render({
      onRestart,
      relocation: { ...RELOCATION, hfHome: "/Volumes/Models 2/hf" },
    });

    expect(dialog.textContent).toContain("/Volumes/Models 2/hf");
    expect(dialog.querySelector('[role="status"]').textContent).not.toContain("Restarting");
    expect(button(dialog, "Restart now").disabled).toBe(false);
    await act(async () => button(dialog, "Restart now").click());
    expect(onRestart).toHaveBeenCalledTimes(2);
    expect(dialog.querySelector('[role="status"]').textContent).toContain("Restarting");
  });

  it("withholds the restart while a job is running, and says why", async () => {
    const onRestart = vi.fn();
    const dialog = await render({ onRestart, runningJobCount: 1 });
    const restart = button(dialog, "Restart now");
    expect(restart.disabled).toBe(true);
    await act(async () => restart.click());
    expect(onRestart).not.toHaveBeenCalled();
    expect(dialog.querySelector('[role="status"]').textContent).toContain(
      "A job is still running",
    );
    // Deferring is always available.
    expect(button(dialog, "Later").disabled).toBe(false);
  });

  it("pluralizes the running-job warning", async () => {
    const dialog = await render({ runningJobCount: 3 });
    expect(dialog.querySelector('[role="status"]').textContent).toContain(
      "3 jobs are still running",
    );
  });

  it("dismisses on Later and on Escape without restarting", async () => {
    const onLater = vi.fn();
    const onRestart = vi.fn();
    let dialog = await render({ onLater, onRestart });
    await act(async () => button(dialog, "Later").click());
    expect(onLater).toHaveBeenCalledTimes(1);

    dialog = await render({ onLater, onRestart });
    await act(async () => {
      dialog.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
    });
    expect(onLater).toHaveBeenCalledTimes(2);
    expect(onRestart).not.toHaveBeenCalled();
  });

  it("degrades to guidance with no restart control outside the desktop shell", async () => {
    const dialog = await render({ canRestart: false });
    expect(button(dialog, "Restart now")).toBeUndefined();
    expect(button(dialog, "Later")).toBeTruthy();
    expect(dialog.querySelector('[role="status"]').textContent).toContain(
      "Restart the SceneWorks server",
    );
  });
});
