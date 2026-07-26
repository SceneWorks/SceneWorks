import { afterEach, expect, it, vi } from "vitest";
import {
  persistNavigationPreferences,
  resetNavigationPreferenceQueueForTests,
} from "./uiPreferences.js";

function deferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function response(payload = {}) {
  return {
    ok: true,
    status: 200,
    json: () => Promise.resolve(payload),
  };
}

afterEach(async () => {
  await resetNavigationPreferenceQueueForTests();
  vi.unstubAllGlobals();
});

it("serializes delayed writes and coalesces the latest navigation intent", async () => {
  const first = deferred();
  const second = deferred();
  const bodies = [];
  vi.stubGlobal("fetch", vi.fn((_url, options) => {
    bodies.push(JSON.parse(options.body));
    return bodies.length === 1 ? first.promise : second.promise;
  }));

  const all = persistNavigationPreferences({ activeView: "Library" }, "token");
  await Promise.resolve();
  await Promise.resolve();
  expect(fetch).toHaveBeenCalledTimes(1);

  persistNavigationPreferences({ activeView: "DatasetCatalogs" }, "token");
  persistNavigationPreferences({ selectedCatalogId: "older" }, "token");
  persistNavigationPreferences({ selectedCatalogId: "newest" }, "token");
  expect(fetch).toHaveBeenCalledTimes(1);

  first.resolve(response());
  await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(bodies[1]).toEqual({
    activeView: "DatasetCatalogs",
    selectedCatalogId: "newest",
  });

  second.resolve(response());
  await all;
  expect(bodies).toEqual([
    { activeView: "Library" },
    { activeView: "DatasetCatalogs", selectedCatalogId: "newest" },
  ]);
});
