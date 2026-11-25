import React, { act, useRef } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { deferred } from "../testUtils/async.js";
import { useResidentState } from "./useResidentState.js";
import { useProjectScopedList } from "./useProjectScopedList.js";

let root, container, current;
beforeEach(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; ({ root, container } = mountRoot()); });
afterEach(() => unmountRoot(root, container));

function Catalog({ resident, load }) {
  const activeProjectRef = useRef({ id: "project" });
  const [items, setItems] = useResidentState([], resident);
  const refresh = useProjectScopedList({
    token: "", activeProject: activeProjectRef.current, activeProjectRef,
    resident, load, setItems,
  });
  current = { items, refresh, setItems };
  return null;
}

it("evicts loaded records and refuses background refreshes and late mutation setters", async () => {
  const load = vi.fn(async () => [{ id: "asset", recipe: "x".repeat(100_000) }]);
  await act(async () => root.render(<Catalog resident load={load} />));
  await act(async () => current.refresh());
  expect(current.items).toHaveLength(1);
  const oldSetter = current.setItems;
  await act(async () => root.render(<Catalog resident={false} load={load} />));
  expect(current.items).toEqual([]);
  await act(async () => oldSetter([{ id: "late" }]));
  expect(current.items).toEqual([]);
  expect(await current.refresh()).toMatchObject({ ok: false, reason: "inactive" });
  expect(load).toHaveBeenCalledTimes(1);
});

it("aborts page-owned loads and ignores their responses even after a quick return", async () => {
  const old = deferred();
  const load = vi.fn().mockImplementationOnce(() => old.promise).mockResolvedValue([{ id: "fresh" }]);
  await act(async () => root.render(<Catalog resident load={load} />));
  let pending;
  await act(async () => { pending = current.refresh(); });
  const signal = load.mock.calls[0][2].signal;
  await act(async () => root.render(<Catalog resident={false} load={load} />));
  expect(signal.aborted).toBe(true);
  await act(async () => root.render(<Catalog resident load={load} />));
  await act(async () => current.refresh());
  await act(async () => { old.resolve([{ id: "obsolete" }]); await pending; });
  expect(current.items).toEqual([{ id: "fresh" }]);
});

it("bounds overlapping batch-triggered refreshes to the newest live request", async () => {
  const old = deferred();
  const load = vi.fn().mockImplementationOnce(() => old.promise).mockResolvedValue([{ id: "newest" }]);
  await act(async () => root.render(<Catalog resident load={load} />));
  let pending;
  await act(async () => { pending = current.refresh(); });
  const firstSignal = load.mock.calls[0][2].signal;
  await act(async () => current.refresh());
  expect(firstSignal.aborted).toBe(true);
  await act(async () => { old.resolve([{ id: "superseded" }]); await pending; });
  expect(current.items).toEqual([{ id: "newest" }]);
});
