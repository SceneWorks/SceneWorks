import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "../api.js";
import { useGenerationMetrics } from "./useGenerationMetrics.js";

vi.mock("../api.js", () => ({
  apiFetch: vi.fn(),
  isAbortError: (error) => error?.name === "AbortError",
}));

function deferred() {
  let resolve;
  const promise = new Promise((accept) => {
    resolve = accept;
  });
  return { promise, resolve };
}

function Probe({ params, onValue }) {
  const value = useGenerationMetrics({ token: "token", params });
  onValue(value);
  return null;
}

describe("useGenerationMetrics request supersession", () => {
  let container;
  let root;
  let requests;
  let current;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    requests = [];
    current = null;
    apiFetch.mockReset();
    apiFetch.mockImplementation((_path, _token, { signal }) => {
      const request = deferred();
      requests.push({ ...request, signal });
      return request.promise;
    });
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("aborts manual/filter-superseded loads and commits only the latest filter", async () => {
    await act(async () => {
      root.render(<Probe params={{ model: "old" }} onValue={(value) => { current = value; }} />);
    });
    expect(requests).toHaveLength(1);

    await act(async () => {
      current.refresh();
    });
    expect(requests).toHaveLength(2);
    expect(requests[0].signal.aborted).toBe(true);

    await act(async () => {
      root.render(<Probe params={{ model: "new" }} onValue={(value) => { current = value; }} />);
    });
    expect(requests).toHaveLength(3);
    expect(requests[1].signal.aborted).toBe(true);
    expect(apiFetch.mock.calls[2][0]).toBe("/api/v1/metrics?model=new");

    await act(async () => {
      requests[2].resolve([{ jobId: "new-row" }]);
      await requests[2].promise;
    });
    expect(current.rows).toEqual([{ jobId: "new-row" }]);

    await act(async () => {
      requests[0].resolve([{ jobId: "stale-effect-row" }]);
      requests[1].resolve([{ jobId: "stale-manual-row" }]);
      await Promise.all([requests[0].promise, requests[1].promise]);
    });
    expect(current.rows).toEqual([{ jobId: "new-row" }]);
  });
});
