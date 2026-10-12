import { describe, expect, it } from "vitest";

import { candleTrainingKernelBlocked } from "./candleGating.js";

// sc-24161: off-Mac, a training target whose kernel has no candle trainer is blocked, read from the
// server's candle-routed set (never a hardcoded kernel), and the gate is inert on Mac / when the
// field is absent.
describe("candleTrainingKernelBlocked", () => {
  const offMac = {
    candleGatingActive: true,
    training: { candleSupportedKernels: ["qwen_image_2_1_lora", "z_image_lora"] },
  };

  it("blocks a kernel missing from the candle-routed set off-Mac only", () => {
    expect(candleTrainingKernelBlocked(offMac, "qwen_image_2_1_edit_lora")).toBe(true);
    expect(candleTrainingKernelBlocked(offMac, "qwen_image_2_1_lora")).toBe(false);
    expect(
      candleTrainingKernelBlocked({ ...offMac, candleGatingActive: false }, "qwen_image_2_1_edit_lora"),
    ).toBe(false);
    expect(candleTrainingKernelBlocked({ candleGatingActive: true, training: {} }, "qwen_image_2_1_edit_lora")).toBe(
      false,
    );
    expect(candleTrainingKernelBlocked(offMac, "")).toBe(false);
  });

  // sc-24162: once the server lists the edit kernel as candle-routed, it is selectable off-Mac with
  // no client change.
  it("unblocks a kernel as soon as the server lists it as candle-routed", () => {
    const withEdit = {
      ...offMac,
      training: { candleSupportedKernels: [...offMac.training.candleSupportedKernels, "qwen_image_2_1_edit_lora"] },
    };
    expect(candleTrainingKernelBlocked(withEdit, "qwen_image_2_1_edit_lora")).toBe(false);
  });
});
