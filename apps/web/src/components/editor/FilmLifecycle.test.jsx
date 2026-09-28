import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../../api.js", () => ({ apiFetch: apiFetchMock }));

import { FilmLifecycle } from "./FilmLifecycle.jsx";

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  apiFetchMock.mockReset();
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function renderLifecycle(setNotice = vi.fn(), onRunChange = vi.fn(), props = {}) {
  root = createRoot(container);
  await act(async () => {
    root.render(<FilmLifecycle draftId="film_1" onRunChange={onRunChange} projectId="project_1" setNotice={setNotice} token="token" {...props} />);
    await Promise.resolve();
  });
  return { onRunChange, setNotice };
}

describe("FilmLifecycle", () => {
  it("reloads a refused resume without replacing the saved canceled verdict and keeps retry available", async () => {
    apiFetchMock.mockImplementation((url) => url.endsWith("/planning")
      ? Promise.reject(new Error("no planning operation"))
      : Promise.resolve([{
        locator: { id: "filmrun_1", draftId: "film_1" }, controllerActive: false,
        record: { state: "finished", outcome: "canceled", stop: { reason: "canceled", resumable: true, detail: "Operator canceled" } },
        actionOperation: { action: "resume", status: "failed", detail: "No video_generate worker is available" },
      }]));
    await renderLifecycle();
    expect(container.textContent).toContain("Render · canceled");
    expect(container.querySelector('[role="alert"]').textContent).toContain("resume failed: No video_generate worker");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Resume").disabled).toBe(false);
  });

  it("offers retry after an initial transport failure before a run record exists", async () => {
    apiFetchMock.mockImplementation((url) => url.endsWith("/planning")
      ? Promise.reject(new Error("no planning operation"))
      : Promise.resolve([{
        locator: { id: "filmrun_1", draftId: "film_1" }, controllerActive: false,
        actionOperation: { action: "start", status: "failed", detail: "Host connection refused" },
      }]));
    await renderLifecycle();
    expect(container.querySelector('[role="alert"]').textContent).toContain("start failed: Host connection refused");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Retry start").disabled).toBe(false);
  });

  it("lists newest and older runs and selects either durable record explicitly", async () => {
    const newest = {
      locator: { id: "filmrun_new", draftId: "film_1" }, controllerActive: false,
      record: { state: "finished", outcome: "completed" },
    };
    const older = {
      locator: { id: "filmrun_old", draftId: "film_1" }, controllerActive: false,
      record: { state: "finished", outcome: "failed" },
    };
    apiFetchMock.mockImplementation((url) => url.endsWith("/planning")
      ? Promise.reject(new Error("no planning operation"))
      : Promise.resolve([newest, older]));
    const { onRunChange } = await renderLifecycle(vi.fn(), vi.fn(), { selectedRunId: "filmrun_new" });

    expect(onRunChange).toHaveBeenCalledWith(newest, { latest: true, select: false });
    expect(onRunChange).toHaveBeenCalledWith(older, { latest: false, select: false });
    const selector = container.querySelector('select[aria-label="Review run"]');
    expect([...selector.options].map((option) => option.value)).toEqual(["filmrun_new", "filmrun_old"]);
    expect(selector.value).toBe("filmrun_new");

    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set.call(selector, "filmrun_old");
      selector.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onRunChange).toHaveBeenLastCalledWith(older, { select: true });
  });

  it("rediscovers running planning and render operations and requests bounded cancellation", async () => {
    apiFetchMock.mockImplementation((url, _token, options) => {
      if (url.endsWith("/planning")) return Promise.resolve({ status: "running", stage: "generating", progress: 0.4 });
      if (url.endsWith("/cancel") && options?.method === "POST") return Promise.resolve({});
      return Promise.resolve([{
        locator: { id: "filmrun_1", draftId: "film_1" }, controllerActive: true,
        controllerOwner: "api:filmrun_1", record: { state: "running", outcome: "failed" },
      }]);
    });
    const { setNotice } = await renderLifecycle();

    expect(container.textContent).toContain("Planning · generating");
    expect(container.textContent).toContain("Render · running");
    const cancel = [...container.querySelectorAll("button")].find((button) => button.textContent === "Cancel");
    await act(async () => { cancel.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/film-runs/filmrun_1/cancel",
      "token",
      { method: "POST" },
    );
    expect(setNotice).toHaveBeenCalledWith(expect.stringContaining("Completed takes"));
  });

  it("shows actionable durable stops and resumes the same run record", async () => {
    const resumed = {
      locator: { id: "filmrun_2", draftId: "film_1" }, controllerActive: true,
      controllerOwner: "api-resume:filmrun_2", record: { state: "running", outcome: "failed" },
    };
    apiFetchMock.mockImplementation((url, _token, options) => {
      if (url.endsWith("/planning")) return Promise.reject(new Error("no planning operation"));
      if (url.endsWith("/resume") && options?.method === "POST") return Promise.resolve(resumed);
      return Promise.resolve([{
        locator: { id: "filmrun_2", draftId: "film_1" }, controllerActive: false,
        record: { state: "finished", outcome: "failed", stop: { reason: "interrupted", resumable: true, detail: "Worker stopped; start a video worker and resume." } },
      }]);
    });
    const { onRunChange } = await renderLifecycle();

    expect(container.textContent).toContain("Worker stopped; start a video worker and resume.");
    const resume = [...container.querySelectorAll("button")].find((button) => button.textContent === "Resume");
    await act(async () => { resume.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/film-runs/filmrun_2/resume",
      "token",
      { method: "POST" },
    );
    expect(onRunChange).toHaveBeenCalledWith(resumed, { select: true });
  });
});
