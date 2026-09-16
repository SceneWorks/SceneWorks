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

async function renderLifecycle(setNotice = vi.fn()) {
  root = createRoot(container);
  await act(async () => {
    root.render(<FilmLifecycle draftId="film_1" projectId="project_1" setNotice={setNotice} token="token" />);
    await Promise.resolve();
  });
  return setNotice;
}

describe("FilmLifecycle", () => {
  it("rediscovers running planning and render operations and requests bounded cancellation", async () => {
    apiFetchMock.mockImplementation((url, _token, options) => {
      if (url.endsWith("/planning")) return Promise.resolve({ status: "running", stage: "generating", progress: 0.4 });
      if (url.endsWith("/cancel") && options?.method === "POST") return Promise.resolve({});
      return Promise.resolve([{
        locator: { id: "filmrun_1", draftId: "film_1" }, controllerActive: true,
        controllerOwner: "api:filmrun_1", record: { state: "running", outcome: "failed" },
      }]);
    });
    const setNotice = await renderLifecycle();

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
    apiFetchMock.mockImplementation((url, _token, options) => {
      if (url.endsWith("/planning")) return Promise.reject(new Error("no planning operation"));
      if (url.endsWith("/resume") && options?.method === "POST") return Promise.resolve({});
      return Promise.resolve([{
        locator: { id: "filmrun_2", draftId: "film_1" }, controllerActive: false,
        record: { state: "finished", outcome: "failed", stop: { reason: "interrupted", resumable: true, detail: "Worker stopped; start a video worker and resume." } },
      }]);
    });
    await renderLifecycle();

    expect(container.textContent).toContain("Worker stopped; start a video worker and resume.");
    const resume = [...container.querySelectorAll("button")].find((button) => button.textContent === "Resume");
    await act(async () => { resume.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/film-runs/filmrun_2/resume",
      "token",
      { method: "POST" },
    );
  });
});
