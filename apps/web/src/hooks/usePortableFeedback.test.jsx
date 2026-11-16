// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";

import { usePortableBatchFeedback, usePortableFeedback } from "./usePortableFeedback.js";
import { validatePortableBatch, validatePortableRequest } from "../portableValidation.js";
import { AppContext } from "../context/AppContext.js";

vi.mock("../portableValidation.js", () => ({
  MAX_VALIDATION_BATCH: 32,
  validatePortableRequest: vi.fn(async ({ payload }) => payload.value === 3
    ? { state: "invalid", refusal: { field: "value", detail: "Third item refused" } }
    : { state: "valid" }),
  validatePortableBatch: vi.fn(async ({ requests }) => requests.map((payload) => payload.value === 3
    ? { state: "invalid", refusal: { field: "value", detail: "Third item refused" } }
    : { state: "valid" })),
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
  vi.mocked(validatePortableBatch).mockReset();
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => requests.map((payload) => payload.value === 3
    ? { state: "invalid", refusal: { field: "value", detail: "Third item refused" } }
    : { state: "valid" }));
});

function SingleHarness({ value }) {
  const feedback = usePortableFeedback("image", { value }, null);
  return <p>{feedback?.state === "invalid" ? feedback.refusal.detail : feedback?.state ?? "pending"}</p>;
}

function Yue2Harness() {
  const feedback = usePortableFeedback("yue2", { mode: "compose" }, null);
  return <p>{feedback?.state ?? "pending"}</p>;
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

it("validates the submitted project scope and drops feedback after a project switch", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  let finish;
  vi.mocked(validatePortableRequest).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  const render = (id) => <AppContext.Provider value={{ activeProject: { id } }}>
    <SingleHarness value={1} />
  </AppContext.Provider>;
  await act(async () => root.render(render("first")));
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  expect(validatePortableRequest).toHaveBeenLastCalledWith(expect.objectContaining({
    kind: "image", payload: { value: 1, projectId: "first" },
  }));
  await act(async () => root.render(render("second")));
  await act(async () => finish({ state: "invalid", refusal: { detail: "Old project refused" } }));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(container.textContent).not.toContain("Old project refused");
  expect(validatePortableRequest).toHaveBeenLastCalledWith(expect.objectContaining({
    kind: "image", payload: { value: 1, projectId: "second" },
  }));
});

it("keeps YuE2 project identity out of the job spec while revalidating after a project switch", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  const render = (id) => <AppContext.Provider value={{ activeProject: { id } }}><Yue2Harness /></AppContext.Provider>;
  await act(async () => root.render(render("first")));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  await act(async () => root.render(render("second")));
  await vi.waitFor(() => expect(vi.mocked(validatePortableRequest).mock.calls).toHaveLength(2));
  expect(vi.mocked(validatePortableRequest).mock.calls.map(([request]) => request.payload))
    .toEqual([{ mode: "compose" }, { mode: "compose" }]);
});

it("revalidates an unchanged draft when connectivity returns", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(validatePortableRequest).mockRejectedValueOnce(new Error("offline"));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<SingleHarness value={1} />));
  await vi.waitFor(() => expect(container.textContent).toBe("server-required"));
  await act(async () => window.dispatchEvent(new Event("online")));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(validatePortableRequest).toHaveBeenCalledTimes(2);
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
  const calls = vi.mocked(validatePortableBatch).mock.calls.length;
  await act(async () => root.render(<Harness revision="a" values={[1, 2]} modelEntry={{ id: "m" }} />));
  expect(container.textContent).toBe("valid");
  expect(vi.mocked(validatePortableBatch).mock.calls).toHaveLength(calls);
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

it("checks each batch request in the active project and restarts on a project switch", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  const render = (id) => <AppContext.Provider value={{ activeProject: { id } }}>
    <Harness revision="same draft" values={[1, 2]} />
  </AppContext.Provider>;
  await act(async () => root.render(render("first")));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(vi.mocked(validatePortableBatch).mock.calls.flatMap(([request]) => request.requests.map((payload) => payload.projectId)))
    .toEqual(["first", "first"]);
  await act(async () => root.render(render("second")));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(vi.mocked(validatePortableBatch).mock.calls.flatMap(([request]) => request.requests.map((payload) => payload.projectId)))
    .toEqual(["first", "first", "second", "second"]);
});

it("revalidates an unchanged batch when connectivity returns", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(validatePortableBatch).mockRejectedValueOnce(new Error("offline"));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness revision="same" values={[1]} />));
  await vi.waitFor(() => expect(container.textContent).toBe("server-required"));
  await act(async () => window.dispatchEvent(new Event("online")));
  await vi.waitFor(() => expect(container.textContent).toBe("valid"));
  expect(validatePortableBatch).toHaveBeenCalledTimes(2);
});
