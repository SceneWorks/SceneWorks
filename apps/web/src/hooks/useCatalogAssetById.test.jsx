import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCatalogAssetById } from "./useCatalogAssetById.js";

const result = { current: undefined };
function Probe(props) {
  result.current = useCatalogAssetById(props);
  return null;
}

let root;
let container;
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

const respond = (status, body) => ({ ok: status < 400, status, json: async () => body });

it("returns the catalog copy without a request", async () => {
  global.fetch = vi.fn();
  const source = { id: "source", projectId: "p" };
  await act(async () => root.render(<Probe assets={[source]} assetId="source" projectId="p" token="" />));
  expect(result.current).toBe(source);
  expect(fetch).not.toHaveBeenCalled();
});

it("fetches an asset the catalog slice does not hold, and prefers the catalog once it arrives", async () => {
  const fetched = { id: "source", projectId: "p", origin: "character_studio" };
  global.fetch = vi.fn(async () => respond(200, fetched));
  await act(async () => root.render(<Probe assets={[]} assetId="source" projectId="p" token="" />));
  await vi.waitFor(() => expect(result.current).toEqual(fetched));
  expect(new URL(fetch.mock.calls[0][0]).pathname).toMatch(/\/projects\/p\/assets\/source$/);

  const listed = { id: "source", projectId: "p", listView: "summary" };
  await act(async () => root.render(<Probe assets={[listed]} assetId="source" projectId="p" token="" />));
  expect(result.current).toBe(listed);
});

it("resolves a purged asset to null", async () => {
  global.fetch = vi.fn(async () => respond(404, { detail: "Asset not found" }));
  await act(async () => root.render(<Probe assets={[]} assetId="gone" projectId="p" token="" />));
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  await act(async () => {});
  expect(result.current).toBeNull();
});

it("forgets a fetched source that a later lookup finds purged", async () => {
  const fetched = { id: "source", projectId: "p", origin: "character_studio" };
  global.fetch = vi.fn(async () => respond(200, fetched));
  await act(async () => root.render(<Probe assets={[]} assetId="source" projectId="p" token="" />));
  await vi.waitFor(() => expect(result.current).toEqual(fetched));

  // The preview closes, the source is purged, and the same edit is reopened.
  await act(async () => root.render(<Probe assets={[]} assetId={null} projectId="p" token="" />));
  global.fetch = vi.fn(async () => respond(404, { detail: "Asset not found" }));
  await act(async () => root.render(<Probe assets={[]} assetId="source" projectId="p" token="" />));
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  await vi.waitFor(() => expect(result.current).toBeNull());
});

it("keeps the last known source through a transient lookup failure", async () => {
  const fetched = { id: "source", projectId: "p" };
  global.fetch = vi.fn(async () => respond(200, fetched));
  await act(async () => root.render(<Probe assets={[]} assetId="source" projectId="p" token="" />));
  await vi.waitFor(() => expect(result.current).toEqual(fetched));

  await act(async () => root.render(<Probe assets={[]} assetId={null} projectId="p" token="" />));
  global.fetch = vi.fn(async () => respond(503, { detail: "busy" }));
  await act(async () => root.render(<Probe assets={[]} assetId="source" projectId="p" token="" />));
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  await act(async () => {});
  expect(result.current).toEqual(fetched);
});

it("never returns a fetch made for another asset", async () => {
  global.fetch = vi.fn(async (url) => respond(200, { id: new URL(url).pathname.split("/").pop(), projectId: "p" }));
  await act(async () => root.render(<Probe assets={[]} assetId="first" projectId="p" token="" />));
  await vi.waitFor(() => expect(result.current?.id).toBe("first"));
  global.fetch = vi.fn(() => new Promise(() => {}));
  await act(async () => root.render(<Probe assets={[]} assetId="second" projectId="p" token="" />));
  expect(result.current).toBeNull();
});
