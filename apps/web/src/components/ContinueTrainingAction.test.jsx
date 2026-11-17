import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { canContinueTraining, completedTrainingSteps, ContinueTrainingAction } from "./ContinueTrainingAction.jsx";

const job = {
  id: "job_original", type: "lora_train", status: "completed",
  payload: { dryRun: false, plan: { target: { outputKind: "lora" }, config: { steps: 3000 } } },
};
let root;
let container;
afterEach(() => {
  if (root) act(() => root.unmount());
  container?.remove();
});
function render(onContinue) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(<ContinueTrainingAction job={job} onContinue={onContinue} />));
  act(() => container.querySelector("button").click());
}
function setSteps(value) {
  const input = container.querySelector("input");
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
it("offers continuation only for real completed LoRA outputs", () => {
  expect(canContinueTraining(job)).toBe(true);
  expect(completedTrainingSteps({ ...job, result: { stepsCompleted: 2500 } })).toBe(2500);
  expect(canContinueTraining({ ...job, result: { stepsCompleted: 0 } })).toBe(false);
  for (const status of ["running", "canceled", "failed", "interrupted"]) {
    expect(canContinueTraining({ ...job, status })).toBe(false);
  }
  expect(canContinueTraining({ ...job, type: "control_training" })).toBe(false);
  expect(canContinueTraining({ ...job, payload: { ...job.payload, dryRun: true } })).toBe(false);
  expect(canContinueTraining({ ...job, payload: { plan: { target: { outputKind: "base_checkpoint" }, config: { steps: 3000 } } } })).toBe(false);
});
it("shows cumulative steps, blocks invalid counts, and submits additional steps once", async () => {
  let complete;
  const onContinue = vi.fn(() => new Promise((resolve) => { complete = resolve; }));
  render(onContinue);
  expect(container.textContent).toContain("3,000 → 4,000 total steps");
  for (const invalid of ["", "0", "-1", "1.5", "4294967295"]) {
    setSteps(invalid);
    expect(container.querySelector('[type="submit"]').disabled).toBe(true);
  }
  setSteps("2000");
  expect(container.textContent).toContain("3,000 → 5,000 total steps");
  await act(async () => container.querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(onContinue).toHaveBeenCalledExactlyOnceWith(job, 2000);
  expect(container.querySelector('[type="submit"]').disabled).toBe(true);
  await act(async () => complete({ id: "job_continued" }));
  expect(container.querySelector("form")).toBeNull();
});
it("keeps the form and count when the server refuses continuation", async () => {
  render(vi.fn().mockRejectedValue(new Error("No final checkpoint.")));
  await act(async () => container.querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(container.querySelector('[role="alert"]').textContent).toBe("No final checkpoint.");
  expect(container.querySelector("input").value).toBe("1000");
});
