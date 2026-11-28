// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import ExactReplayAction, { checkExactReplayHostSupport } from "./ExactReplayAction.jsx";

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
  const job = { id: "render-one", type: "image_generate", status: "completed", exactReplay: { supported: true } };
  const onExactReplay = vi.fn(async () => ({
    code: "replay_unavailable", context: { reason: "deterministic_replay_unavailable" },
  }));
  const element = render(<ExactReplayAction job={job} onExactReplay={onExactReplay} />);
  await act(async () => element.querySelector("button").click());
  expect(onExactReplay).toHaveBeenCalledWith(job);
  expect(element.querySelector('[role="alert"]').textContent)
    .toBe("The provider has not verified exact replay for this run.");
});

it.each([
  [undefined, "Exact replay support is unavailable for this job. Use Duplicate to generate again."],
  [{ supported: false, reason: "receipt_missing" }, "This job has no complete execution receipt."],
  [{ supported: false, reason: "execution_unavailable" }, "This run did not record the execution details required for exact replay. Use Duplicate to generate again."],
  [{ supported: false, reason: "deterministic_replay_unavailable" }, "The provider has not verified exact replay for this run."],
])("disables replay when recorded support is unavailable (%j)", (exactReplay, explanation) => {
  const onExactReplay = vi.fn();
  const element = render(<ExactReplayAction job={{
    id: "unsupported", type: "image_generate", status: "completed", exactReplay,
  }} onExactReplay={onExactReplay} />);
  const button = element.querySelector("button");
  expect(button.disabled).toBe(true);
  expect(button.title).toBe(explanation);
  button.click();
  expect(onExactReplay).not.toHaveBeenCalled();
});

it("queues supported replay once while pending and permits retry after a host refusal", async () => {
  let finish;
  const onExactReplay = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
  const element = render(<ExactReplayAction job={{
    id: "supported", type: "image_edit", status: "completed", exactReplay: { supported: true },
  }} onExactReplay={onExactReplay} />);
  act(() => element.querySelector("button").click());
  expect(element.querySelector("button").disabled).toBe(true);
  expect(element.querySelector("button").textContent).toBe("Queuing replay…");
  act(() => element.querySelector("button").click());
  expect(onExactReplay).toHaveBeenCalledTimes(1);
  await act(async () => finish({ code: "replay_unavailable", context: { reason: "device_unavailable" } }));
  expect(element.querySelector('[role="alert"]').textContent).toBe("The original worker is unavailable.");
  expect(element.querySelector("button").disabled).toBe(false);
  onExactReplay.mockResolvedValueOnce({ id: "new-replay", status: "queued" });
  await act(async () => element.querySelector("button").click());
  expect(onExactReplay).toHaveBeenCalledTimes(2);
  expect(element.querySelector('[role="alert"]')).toBeNull();
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

it("explains quality-affecting rung and low-res estimator replay refusals (SW-6943)", () => {
  for (const [reason, expected] of [
    ["quality_rung_unavailable", "The recorded quality-affecting memory strategy is unavailable on this host."],
    ["estimator_unavailable", "The recorded Low Res Start transition estimator is unavailable on this host."],
    ["unsupported_low_res_recipe_version", "The recorded Low Res Start recipe version is unsupported by this build."],
  ]) {
    const element = render(<ExactReplayAction job={{
      id: `replay-${reason}`, type: "image_generate", status: "failed",
      result: { replayUnavailable: { code: "replay_unavailable", reason } },
    }} onExactReplay={vi.fn()} />);
    expect(element.querySelector('[role="alert"]').textContent).toBe(expected);
  }
});

it("checkExactReplayHostSupport validates active lowResStart receipt, estimator, and quality rung (SW-6943)", () => {
  const payload = {
    advanced: {
      steps: 9,
      lowResStart: { version: 1, k: 3, scale: 0.5 },
    },
  };
  const receipt = {
    acceptedBackend: "mlx",
    execution: {
      identity: {
        quality_policy: { value: { rung: "resident" } },
      },
    },
    lowResStart: {
      version: 1,
      k: 3,
      scale: 0.5,
      estimator: "endpoint_evaluation_v1",
      selectedRung: "resident",
    },
  };
  const host = {
    backend: "mlx",
    supportsLowResStart: true,
    availableEstimators: ["endpoint_evaluation_v1"],
    availableQualityRungs: ["resident"],
  };
  expect(checkExactReplayHostSupport(payload, { receipt, host })).toEqual({ ok: true });

  expect(
    checkExactReplayHostSupport(payload, {
      receipt,
      host: { ...host, availableQualityRungs: ["tiled_vae"] },
    }),
  ).toMatchObject({ ok: false, reason: "quality_rung_unavailable" });

  expect(
    checkExactReplayHostSupport(payload, {
      receipt,
      host: { ...host, availableEstimators: ["previous_evaluation_v1"] },
    }),
  ).toMatchObject({ ok: false, reason: "estimator_unavailable" });

  expect(
    checkExactReplayHostSupport(
      { advanced: { lowResStart: { version: 2, k: 3, scale: 0.5 } } },
      { receipt, host },
    ),
  ).toMatchObject({ ok: false, reason: "unsupported_low_res_recipe_version" });
});
