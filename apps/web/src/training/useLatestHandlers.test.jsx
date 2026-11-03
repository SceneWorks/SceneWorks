import React, { act, startTransition, Suspense, useLayoutEffect } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { useLatestHandlers } from "./useLatestHandlers.js";

it("exposes only committed owners, including to descendant layout effects", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const calls = [];
  let visibleHandlers;
  const pending = new Promise(() => {});
  function LayoutConsumer({ handlers, value }) {
    useLayoutEffect(() => { handlers.run(); }, [handlers, value]);
    return null;
  }
  function Suspend({ suspend }) { if (suspend) throw pending; return null; }
  function Owner({ value, suspend }) {
    const handlers = useLatestHandlers({ run: () => calls.push(value) });
    useLayoutEffect(() => { visibleHandlers = handlers; }, [handlers]);
    return <><LayoutConsumer handlers={handlers} value={value} /><Suspend suspend={suspend} /></>;
  }
  try {
    await act(async () => root.render(<Suspense fallback="loading"><Owner value="A" /></Suspense>));
    expect(calls).toEqual(["A"]);
    const first = visibleHandlers;
    await act(async () => startTransition(() => root.render(<Suspense fallback="loading"><Owner value="B" suspend /></Suspense>)));
    first.run();
    expect(calls).toEqual(["A", "A"]);
    await act(async () => root.render(<Suspense fallback="loading"><Owner value="B" /></Suspense>));
    expect(visibleHandlers).toBe(first);
    // The descendant's layout effect must already see B at this commit.
    expect(calls).toEqual(["A", "A", "B"]);
    first.run();
    expect(calls).toEqual(["A", "A", "B", "B"]);
  } finally {
    act(() => root.unmount());
    container.remove();
  }
});
