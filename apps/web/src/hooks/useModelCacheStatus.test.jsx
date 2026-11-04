import React, { act } from "react";
import { expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { deferred } from "../testUtils/async.js";

vi.mock("../modelCache.js", async (original) => ({ ...(await original()), fetchModelCache: vi.fn() }));
import { CACHE_CONVERGENCE_POLL_MS, CACHE_ENTRY_STATE, MAX_CACHE_CONVERGENCE_POLLS, fetchModelCache } from "../modelCache.js";
import { useModelCacheStatus } from "./useModelCacheStatus.js";

it("discards an old token's response and exposes the current snapshot and error", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const first = deferred();
  const second = deferred();
  vi.mocked(fetchModelCache).mockReset().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  let status;
  function Harness({ token }) {
    status = useModelCacheStatus(token, { reportSnapshotError: true });
    return null;
  }
  const { root, container } = mountRoot();
  await act(async () => root.render(<Harness token="old" />));
  await act(async () => root.render(<Harness token="new" />));
  await act(async () => first.resolve({ policy: { enabled: true }, error: "old" }));
  expect(status.cache).toBeNull();
  await act(async () => second.resolve({ policy: { enabled: false }, error: "unavailable" }));
  expect(status.cache.policy.enabled).toBe(false);
  expect(status.error).toBe("unavailable");
  expect(fetchModelCache.mock.calls.map(([token]) => token)).toEqual(["old", "new"]);
  await unmountRoot(root, container);
});

it.each([
  ["Settings", {}],
  ["Model Manager", { pollWhenDisabled: false, reportSnapshotError: true }],
])("%s refuses a refresh continuation captured before a token change", async (_screen, options) => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const current = deferred();
  const oldSnapshot = { policy: { enabled: true }, entries: [], owner: "old" };
  const newSnapshot = { policy: { enabled: true }, entries: [], owner: "new" };
  vi.mocked(fetchModelCache).mockReset().mockImplementation((token) =>
    token === "new" ? current.promise : Promise.resolve(oldSnapshot));
  let status;
  function Harness({ token }) {
    status = useModelCacheStatus(token, options);
    return null;
  }
  const { root, container } = mountRoot();
  try {
    await act(async () => root.render(<Harness token="old" />));
    const obsoleteRefresh = status.refresh;
    await act(async () => root.render(<Harness token="new" />));
    await act(async () => { await obsoleteRefresh(); });
    expect(fetchModelCache.mock.calls.map(([token]) => token)).toEqual(["old", "new"]);
    expect(status.cache).toBeNull();
    await act(async () => current.resolve(newSnapshot));
    expect(status.cache).toEqual(newSnapshot);
    expect(status.error).toBe("");
  } finally { await unmountRoot(root, container); }
});

it.each([
  ["Settings", {}],
  ["Model Manager", { pollWhenDisabled: false, reportSnapshotError: true }],
])("%s accepts a settled response when cache reads outlast the polling interval", async (_screen, options) => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  const copying = { policy: { enabled: true }, entries: [{ state: CACHE_ENTRY_STATE.MATERIALIZING }] };
  const settled = { policy: { enabled: true }, entries: [{ state: CACHE_ENTRY_STATE.COMPLETE }] };
  const pending = [];
  vi.mocked(fetchModelCache).mockReset().mockResolvedValueOnce(copying).mockImplementation(() => {
    const request = deferred();
    pending.push(request);
    return request.promise;
  });
  let status;
  function Harness() {
    status = useModelCacheStatus("current", options);
    return null;
  }
  const { root, container } = mountRoot();
  try {
    await act(async () => root.render(<Harness />));
    for (let poll = 0; poll < 3; poll += 1) {
      await act(async () => { await vi.advanceTimersByTimeAsync(CACHE_CONVERGENCE_POLL_MS); });
    }
    expect(pending.length).toBeGreaterThan(0);
    await act(async () => pending[0].resolve(settled));
    expect(status.cache).toEqual(settled);
    expect(status.stalled).toBe(false);
    expect(fetchModelCache).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(CACHE_CONVERGENCE_POLL_MS * MAX_CACHE_CONVERGENCE_POLLS); });
    expect(fetchModelCache).toHaveBeenCalledTimes(2);
    expect(status.stalled).toBe(false);
  } finally {
    await unmountRoot(root, container);
    vi.useRealTimers();
  }
});
