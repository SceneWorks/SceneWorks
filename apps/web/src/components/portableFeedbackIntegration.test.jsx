import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const validate = vi.hoisted(() => vi.fn());
vi.mock("../portableValidation.js", () => ({ validatePortableRequest: validate }));

import { RefinePromptControl } from "./RefinePromptControl.jsx";

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

let host;
let root;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  validate.mockReset();
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

it("shows the current typed refusal and ignores a slower result for an older prompt", async () => {
  const first = deferred();
  const second = deferred();
  validate.mockImplementation(({ payload }) => payload.prompt === "old" ? first.promise : second.promise);
  const props = { modelId: "z", workflow: "image", refinePrompt: vi.fn(), onApply: vi.fn() };
  await act(async () => root.render(<RefinePromptControl {...props} prompt="old" />));
  await act(async () => root.render(<RefinePromptControl {...props} prompt="new" />));

  await act(async () => first.resolve({ state: "invalid", refusal: { code: "invalid_generation_option", field: "prompt", detail: "Old refusal" } }));
  expect(host.textContent).not.toContain("Old refusal");

  await act(async () => second.resolve({ state: "invalid", refusal: { code: "invalid_generation_option", field: "prompt", detail: "Current refusal" } }));
  expect(host.querySelector('[aria-label="Prompt refinement errors"]').textContent).toContain("Current refusal");
  expect(host.querySelector("button.refine-button").disabled).toBe(true);
});
