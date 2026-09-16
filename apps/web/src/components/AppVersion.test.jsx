import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", () => ({ apiFetch: vi.fn() }));
vi.mock("../appConfirm.jsx", () => ({ appConfirm: vi.fn() }));
vi.mock("../runtime.js", () => ({ isDesktop: true, tauriInvoke: vi.fn() }));
import { apiFetch } from "../api.js";
import { appConfirm } from "../appConfirm.jsx";
import { tauriInvoke } from "../runtime.js";
import { AppVersion, prepareAppUpdate, useAppUpdate } from "./AppVersion.jsx";

function Harness({ ready = true }) {
  const update = useAppUpdate("token", ready);
  return <AppVersion update={update} />;
}

describe("app update", () => {
  let root;
  let container;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    vi.resetAllMocks();
    ({ root, container } = mountRoot());
    apiFetch.mockResolvedValue({ activeJobs: [] });
    tauriInvoke.mockImplementation(async (command) => command === "get_update_status" ? { version: "1.2.3" } : undefined);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.useRealTimers();
  });

  it("installs without a confirmation when the live queue is empty", async () => {
    await act(async () => root.render(<Harness />));
    expect(container.textContent).toContain("Update to 1.2.3");
    await click(container.querySelector("button"));
    expect(appConfirm).not.toHaveBeenCalled();
    expect(apiFetch).toHaveBeenCalledTimes(2);
    expect(tauriInvoke.mock.calls.map(([command]) => command)).toEqual([
      "get_update_status", "download_app_update", "install_app_update", "discard_app_update",
    ]);
  });

  it("uses the guarded update flow after accepting the startup prompt", async () => {
    tauriInvoke.mockImplementation(async (command, args) => command === "get_update_status" ? {
      version: "1.2.3", startupRequested: args.ready,
    } : undefined);
    await act(async () => root.render(<Harness ready={false} />));
    expect(tauriInvoke).not.toHaveBeenCalledWith("download_app_update");
    await act(async () => root.render(<Harness ready />));
    expect(tauriInvoke).toHaveBeenCalledWith("download_app_update");
    expect(tauriInvoke).toHaveBeenCalledWith("install_app_update");
    expect(appConfirm).not.toHaveBeenCalled();
  });

  it("declining preserves work and does not download or install", async () => {
    apiFetch.mockResolvedValue({ activeJobs: [{ id: "running" }] });
    appConfirm.mockResolvedValue(false);
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    expect(appConfirm).toHaveBeenCalledWith(expect.objectContaining({
      message: "There are work items in progress. Are you sure you want to update now?",
    }));
    expect(tauriInvoke).toHaveBeenCalledTimes(1);
    expect(apiFetch).toHaveBeenCalledTimes(1);
  });

  it("cancels every live item, including work added during confirmation", async () => {
    apiFetch.mockResolvedValueOnce({ activeJobs: [{ id: "a" }] })
      .mockResolvedValueOnce({ activeJobs: [{ id: "a" }, { id: "b" }] });
    appConfirm.mockResolvedValue(true);
    expect(await prepareAppUpdate("token")).toEqual({ proceed: true, confirmed: true });
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/jobs/a/cancel", "token", { method: "POST" });
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/jobs/b/cancel", "token", { method: "POST" });
  });

  it("prompts if work starts during the download and declining prevents installation", async () => {
    apiFetch.mockResolvedValueOnce({ activeJobs: [] }).mockResolvedValueOnce({ activeJobs: [{ id: "new" }] });
    appConfirm.mockResolvedValue(false);
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    expect(tauriInvoke).toHaveBeenCalledWith("download_app_update");
    expect(tauriInvoke).not.toHaveBeenCalledWith("install_app_update");
    expect(tauriInvoke).toHaveBeenCalledWith("discard_app_update");
  });

  it("does not ask again after consent, but cancels new work", async () => {
    apiFetch.mockResolvedValueOnce({ activeJobs: [{ id: "new" }] })
      .mockResolvedValueOnce({ activeJobs: [{ id: "new" }] });
    await prepareAppUpdate("token", true);
    expect(appConfirm).not.toHaveBeenCalled();
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/jobs/new/cancel", "token", { method: "POST" });
  });

  it.each(["unreachable", "malformed", "cancellation"])("stops on %s queue errors", async (failure) => {
    if (failure === "unreachable") apiFetch.mockRejectedValue(new Error("API offline"));
    if (failure === "malformed") apiFetch.mockResolvedValue(null);
    if (failure === "cancellation") {
      apiFetch.mockResolvedValueOnce({ activeJobs: [{ id: "a" }] })
        .mockResolvedValueOnce({ activeJobs: [{ id: "a" }] })
        .mockRejectedValue(new Error("Cancel failed"));
      appConfirm.mockResolvedValue(true);
    }
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    expect(container.querySelector('[role="alert"]').textContent).toContain("Update failed");
    expect(tauriInvoke).toHaveBeenCalledTimes(1);
  });

  it("waits for worker acknowledgement before installing", async () => {
    vi.useFakeTimers();
    let stopped = false;
    apiFetch.mockImplementation(async (path) => path.endsWith("/cancel") ? {} : {
      activeJobs: stopped ? [] : [{ id: "running" }],
    });
    appConfirm.mockResolvedValue(true);
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    expect(tauriInvoke).not.toHaveBeenCalledWith("download_app_update");
    expect(document.body.textContent).toContain("Canceling work…");
    stopped = true;
    await act(async () => vi.advanceTimersByTimeAsync(250));
    expect(tauriInvoke).toHaveBeenCalledWith("install_app_update");
  });

  it("keeps the app open if workers do not stop", async () => {
    vi.useFakeTimers();
    apiFetch.mockResolvedValue({ activeJobs: [{ id: "running" }] });
    appConfirm.mockResolvedValue(true);
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    await act(async () => vi.advanceTimersByTimeAsync(60_000));
    expect(container.querySelector('[role="alert"]').textContent).toContain("Work is still stopping");
    expect(tauriInvoke).not.toHaveBeenCalledWith("install_app_update");
    expect(apiFetch.mock.calls.filter(([path]) => path.endsWith("/cancel"))).toHaveLength(1);
  });

  it("shows download errors and allows retry without installing", async () => {
    tauriInvoke.mockImplementation(async (command) => {
      if (command === "get_update_status") return { version: "1.2.3" };
      throw new Error("Download failed");
    });
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    expect(container.querySelector('[role="alert"]').textContent).toContain("Download failed");
    expect(container.querySelector("button").disabled).toBe(false);
    expect(tauriInvoke).not.toHaveBeenCalledWith("install_app_update");
  });

  it("refreshes availability without overlapping reads and stops after unmount", async () => {
    vi.useFakeTimers();
    tauriInvoke.mockResolvedValueOnce({ version: null }).mockResolvedValue({ version: "1.2.4" });
    await act(async () => root.render(<Harness />));
    expect(container.querySelector("button")).toBeNull();
    await act(async () => vi.advanceTimersByTimeAsync(15_000));
    expect(container.textContent).toContain("Update to 1.2.4");
    expect(appConfirm).not.toHaveBeenCalled();
    expect(tauriInvoke).not.toHaveBeenCalledWith("download_app_update");
    await act(async () => root.unmount());
    const calls = tauriInvoke.mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(tauriInvoke).toHaveBeenCalledTimes(calls);
  });

  it("prevents duplicate downloads while the update is busy", async () => {
    let finishDownload;
    tauriInvoke.mockImplementation(async (command) => {
      if (command === "get_update_status") return { version: "1.2.3" };
      if (command === "download_app_update") return new Promise((resolve) => { finishDownload = resolve; });
    });
    await act(async () => root.render(<Harness />));
    await click(container.querySelector("button"));
    await click(container.querySelector("button"));
    expect(tauriInvoke.mock.calls.filter(([command]) => command === "download_app_update")).toHaveLength(1);
    await act(async () => finishDownload());
  });
});
