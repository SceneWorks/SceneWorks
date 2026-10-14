// sc-25908 (epic 25899 E6): the job card says which prompt path a Qwen Image 2.1 / Turbo render ran.
//
// The records are the worker's `result.prefixKvCache` shape (`execution_planner::
// prefix_kv_cache_record`): `{selected, pricedBytes, path, cacheBytes, overflow}`.

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { WorkerProgressCard } from "./WorkerProgressCard.jsx";
import { AppContext } from "../context/AppContext.js";

const GIB = 1024 ** 3;

function job(records) {
  return {
    id: "job-turbo-1",
    type: "image_generate",
    status: "completed",
    payload: { model: "qwen_image_2_1_turbo", prompt: "a lighthouse" },
    result: records === undefined ? { assetIds: ["a1"] } : { assetIds: ["a1"], prefixKvCache: records },
  };
}

let mounted = null;
function render(card) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <AppContext.Provider value={{ visibleWorkers: [], workersById: new Map(), models: [] }}>
        <WorkerProgressCard job={card} />
      </AppContext.Provider>,
    );
  });
  mounted = { root, container };
  return container;
}

afterEach(() => {
  if (mounted) {
    act(() => mounted.root.unmount());
    mounted.container.remove();
    mounted = null;
  }
});

const detail = (container) => container.querySelector("details.worker-progress-card__prefix-cache");

describe("WorkerProgressCard prefix KV cache detail (sc-25908)", () => {
  // *Mutation that reds this:* rendering the path only inside the collapsed body (or not at all).
  it("shows a cached render in the summary, collapsed, with the cache bytes behind it", () => {
    const container = render(
      job([{ selected: "cached", pricedBytes: 3 * GIB, path: "cached", cacheBytes: 2 * GIB, overflow: null }]),
    );
    const disclosure = detail(container);
    expect(disclosure).toBeTruthy();
    expect(disclosure.open, "detail-level disclosure defaults collapsed").toBe(false);
    expect(disclosure.querySelector("summary").textContent).toBe("Prompt cache: cached");
    const body = disclosure.querySelector("li").textContent;
    expect(body).toContain("ran cached");
    expect(body).toContain("2.0 GiB cache");
    expect(body).toContain("3.0 GiB priced");
    expect(disclosure.querySelector(".worker-progress-card__prefix-cache-overflow")).toBeNull();
  });

  it("shows a prefill render, and that the admission chose it", () => {
    const container = render(
      job([{ selected: "prefill", pricedBytes: 3 * GIB, path: "prefill", cacheBytes: 0, overflow: null }]),
    );
    expect(detail(container).querySelector("summary").textContent).toBe("Prompt cache: prefill");
    const body = detail(container).querySelector("li").textContent;
    expect(body).toContain("ran prefill");
    // A prefill render holds no cache, so no cache size is quoted.
    expect(body).not.toContain("cache ·");
    expect(body).not.toMatch(/GiB cache/);
  });

  // *Mutation that reds this:* dropping the overflow notice, or reading `overflow` off the
  // wrong key — the engine-shed render would then read as an ordinary prefill.
  it("explains an engine shed over the priced bound", () => {
    const container = render(
      job([
        {
          selected: "cached",
          pricedBytes: 2 * GIB,
          path: "prefill",
          cacheBytes: 0,
          overflow: { actualBytes: 3 * GIB, pricedBytes: 2 * GIB },
        },
      ]),
    );
    const disclosure = detail(container);
    expect(disclosure.querySelector("summary").textContent).toBe("Prompt cache: prefill · cache shed");
    expect(disclosure.querySelector("li").textContent).toContain("admission selected cached");
    const notice = disclosure.querySelector(".worker-progress-card__prefix-cache-overflow");
    expect(notice?.textContent).toContain("The engine shed the prompt cache");
    expect(notice.textContent).toContain("3.0 GiB");
    expect(notice.textContent).toContain("2.0 GiB the memory admission priced");
  });

  it("counts each path across a multi-image job", () => {
    const container = render(
      job([
        { selected: "cached", pricedBytes: GIB, path: "cached", cacheBytes: GIB, overflow: null },
        { selected: "cached", pricedBytes: GIB, path: "cached", cacheBytes: GIB, overflow: null },
        { selected: "cached", pricedBytes: GIB, path: "prefill", cacheBytes: 0, overflow: { actualBytes: 2 * GIB, pricedBytes: GIB } },
      ]),
    );
    const disclosure = detail(container);
    expect(disclosure.querySelector("summary").textContent).toBe("Prompt cache: 2 cached, 1 prefill · cache shed");
    const rows = [...disclosure.querySelectorAll("li")].map((row) => row.textContent);
    expect(rows[0]).toMatch(/^Image 1: ran cached/);
    expect(rows[2]).toMatch(/^Image 3: ran prefill/);
  });

  it("renders nothing for a job whose provider prices no cache", () => {
    const container = render(job(undefined));
    expect(detail(container)).toBeNull();
  });
});
