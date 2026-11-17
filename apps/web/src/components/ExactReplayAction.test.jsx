import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import ExactReplayAction from "./ExactReplayAction.jsx";

let root;
let container;
function render(ui) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(ui));
  return container;
}
afterEach(() => {
  if (root) act(() => root.unmount());
  container?.remove();
  root = null;
  container = null;
});

it("offers exact replay for a completed image job and explains a typed refusal", async () => {
  const job = { id: "render-one", type: "image_generate", status: "completed" };
  const onExactReplay = vi.fn(async () => ({
    code: "replay_unavailable", context: { reason: "deterministic_replay_unavailable" },
  }));
  const element = render(<ExactReplayAction job={job} onExactReplay={onExactReplay} />);
  await act(async () => element.querySelector("button").click());
  expect(onExactReplay).toHaveBeenCalledWith(job);
  expect(element.querySelector('[role="alert"]').textContent)
    .toBe("The provider has not verified exact replay for this run.");
});

it("shows a persisted refusal without offering replay on a failed job", () => {
  const element = render(<ExactReplayAction job={{
    id: "replay-failed", type: "image_generate", status: "failed",
    result: { replayUnavailable: { code: "replay_unavailable", reason: "source_unavailable" } },
  }} onExactReplay={vi.fn()} />);
  expect(element.querySelector('[role="alert"]').textContent)
    .toBe("The original model source is unavailable.");
  expect(element.querySelector("button")).toBeNull();
});
