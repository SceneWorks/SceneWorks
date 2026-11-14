// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";

import { usePortableBatchFeedback, usePortableFeedback } from "./usePortableFeedback.js";
import { validatePortableRequest } from "../portableValidation.js";

vi.mock("../portableValidation.js", () => ({
  validatePortableRequest: vi.fn(async ({ payload }) => payload.value === 3
    ? { state: "invalid", refusal: { field: "value", detail: "Third item refused" } }
    : { state: "valid" }),
}));

let root;
let container;
afterEach(async () => {
  if (root) await act(async () => root.unmount());
  container?.remove();
  root = null;
  vi.mocked(validatePortableRequest).mockReset();
  vi.mocked(validatePortableRequest).mockImplementation(async ({ payload }) => payload.value === 3
    ? { state: "invalid", refusal: { field: "value", detail: "Third item refused" } }
    : { state: "valid" });
});

function SingleHarness({ value }) {
  const feedback = usePortableFeedback("image", { value }, null);
  return <p>{feedback?.state === "invalid" ? feedback.refusal.detail : feedback?.state ?? "pending"}</p>;
}

function ModelHarness({ modelEntry }) {
  const feedback = usePortableFeedback("image", { model: modelEntry.id, value: 1 }, modelEntry);
  return <p>{feedback?.state ?? "pending"}</p>;
}

it("keeps feedback when the same catalog facts arrive in a new object", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<ModelHarness modelEntry={{ id: "m", limits: { count: [1, 2] } }} />));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  const calls = vi.mocked(validatePortableRequest).mock.calls.length;
  await act(async () => root.render(<ModelHarness modelEntry={{ id: "m", limits: { count: [1, 2] } }} />));
  expect(container.textContent).toBe("valid");
  expect(vi.mocked(validatePortableRequest).mock.calls).toHaveLength(calls);
});

it("does not show a refusal returned after the input has changed", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  let finish;
  vi.mocked(validatePortableRequest).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<SingleHarness value={1} />));
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  await act(async () => root.render(<SingleHarness value={2} />));
  await act(async () => finish({ state: "invalid", refusal: { detail: "Old draft refused" } }));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(container.textContent).not.toContain("Old draft refused");
});

function Harness({ revision, values, modelEntry = null }) {
  const feedback = usePortableBatchFeedback(
    "image", revision, function* requests() {
      for (const value of values) yield { value };
    }, modelEntry,
  );
  return <p>{feedback?.result.state === "invalid"
    ? `${feedback.index}: ${feedback.result.refusal.detail}`
    : feedback?.result.state ?? "pending"}</p>;
}

it("keeps a completed batch when equivalent catalog facts are reallocated", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness revision="a" values={[1, 2]} modelEntry={{ id: "m" }} />));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  const calls = vi.mocked(validatePortableRequest).mock.calls.length;
  await act(async () => root.render(<Harness revision="a" values={[1, 2]} modelEntry={{ id: "m" }} />));
  expect(container.textContent).toBe("valid");
  expect(vi.mocked(validatePortableRequest).mock.calls).toHaveLength(calls);
});

it("checks every batch item and hides a refusal when the authored draft changes", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness revision="a" values={[1, 2, 3]} />));
  await vi.waitFor(() => expect(container.textContent).toBe("3: Third item refused"));
  await act(async () => root.render(<Harness revision="b" values={[1, 2]} />));
  expect(container.textContent).not.toContain("Third item refused");
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
});
