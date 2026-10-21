import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useIrisDataWalk } from "./useIrisDataWalk.js";

let container;
let root;
let latest;

function Probe(props) {
  latest = useIrisDataWalk(props);
  return null;
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  latest = undefined;
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

const WARNING = "Only 640 of 700 items will be used.";

function stubFetch(body) {
  const fetchMock = vi.fn(async () => ({ ok: true, status: 200, json: async () => body }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("useIrisDataWalk", () => {
  it("asks the API for the selected dataset and batch and returns its warning", async () => {
    const fetchMock = stubFetch({ applies: true, items: 700, covered: 640, unusedItems: 60, chunks: 640, warning: WARNING });
    await act(async () => {
      root.render(<Probe enabled targetId="iris_3b_generation" items={700} batchSize={2} token="tok" />);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/v1/training/data-walk?targetId=iris_3b_generation&items=700&batchSize=2");
    expect(options.headers.get("X-SceneWorks-Token")).toBe("tok");
    expect(latest?.warning).toBe(WARNING);
  });

  it("re-asks when the batch size changes", async () => {
    const fetchMock = stubFetch({ applies: true, warning: null });
    await act(async () => {
      root.render(<Probe enabled targetId="iris_3b_generation" items={700} batchSize={2} />);
    });
    await act(async () => {
      root.render(<Probe enabled targetId="iris_3b_generation" items={700} batchSize={4} />);
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][0]).toContain("batchSize=4");
  });

  it("does not ask for a non-Iris target or an empty dataset", async () => {
    const fetchMock = stubFetch({ applies: true, warning: WARNING });
    await act(async () => {
      root.render(<Probe enabled={false} targetId="sdxl_lora" items={700} batchSize={2} />);
    });
    await act(async () => {
      root.render(<Probe enabled targetId="iris_3b_generation" items={0} batchSize={2} />);
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(latest).toBeNull();
  });

  it("shows nothing when the preflight fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500, json: async () => ({ detail: "boom" }) })));
    await act(async () => {
      root.render(<Probe enabled targetId="iris_3b_generation" items={700} batchSize={2} />);
    });
    expect(latest).toBeNull();
  });
});
