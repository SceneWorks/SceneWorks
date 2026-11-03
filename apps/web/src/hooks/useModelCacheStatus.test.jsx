import React, { act } from "react";
import { expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { deferred } from "../testUtils/async.js";

vi.mock("../modelCache.js", async (original) => ({ ...(await original()), fetchModelCache: vi.fn() }));
import { fetchModelCache } from "../modelCache.js";
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
