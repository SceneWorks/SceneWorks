import { describe, expect, it } from "vitest";
import {
  DEFAULT_SOL_TOPK_RATIO,
  attentionExecutionSummary,
  lossyAttentionAvailable,
  lossyAttentionDefaults,
  lossyAttentionFromRecipe,
  resolveLossyAttention,
  servedLossyKernels,
} from "./lossyAttention.js";

const BOTH = { lossyAttention: { int8: { served: true }, sol: { served: true } } };
const INT8_ONLY = {
  lossyAttention: {
    int8: { served: true },
    sol: { served: false, reason: "attends block-causally" },
  },
};
const NEITHER = {
  lossyAttention: {
    int8: { served: false, reason: "has 256-wide attention heads" },
    sol: { served: false, reason: "has 256-wide attention heads" },
  },
};

describe("lossy attention opt-in (SW-6132)", () => {
  it("is off by default with INT8 as the recommended kernel", () => {
    expect(lossyAttentionDefaults()).toEqual({
      enabled: false,
      kernel: "int8",
      pvPrecision: "value_dtype",
      topkRatio: DEFAULT_SOL_TOPK_RATIO,
    });
    expect(DEFAULT_SOL_TOPK_RATIO).toBe(0.5);
    expect(resolveLossyAttention({ backend: "candle", model: BOTH, options: undefined })).toEqual({
      block: null,
      error: null,
    });
  });

  it("is offered only on candle for a model that serves a kernel", () => {
    expect(lossyAttentionAvailable("candle", BOTH)).toBe(true);
    expect(lossyAttentionAvailable("candle", INT8_ONLY)).toBe(true);
    expect(lossyAttentionAvailable("mlx", BOTH)).toBe(false);
    expect(lossyAttentionAvailable("candle", NEITHER)).toBe(false);
    expect(lossyAttentionAvailable("candle", { id: "moss_image_base" })).toBe(false);
    expect(servedLossyKernels(INT8_ONLY)).toEqual(["int8"]);
    expect(servedLossyKernels(BOTH)).toEqual(["int8", "sol"]);
  });

  it("resolves the wire the API validates, never a kernel the model refuses", () => {
    const on = (patch) => ({ ...lossyAttentionDefaults(), enabled: true, ...patch });
    expect(resolveLossyAttention({ backend: "candle", model: BOTH, options: on() }).block).toEqual({ int8: {} });
    expect(
      resolveLossyAttention({ backend: "candle", model: BOTH, options: on({ pvPrecision: "int8" }) }).block,
    ).toEqual({ int8: { pvPrecision: "int8" } });
    expect(
      resolveLossyAttention({ backend: "candle", model: BOTH, options: on({ kernel: "sol", topkRatio: "0.25" }) })
        .block,
    ).toEqual({ sol: { topkRatio: 0.25 } });
    // A sticky Sol choice from another model falls back to the served kernel.
    expect(
      resolveLossyAttention({ backend: "candle", model: INT8_ONLY, options: on({ kernel: "sol" }) }).block,
    ).toEqual({ int8: {} });
    // MLX or an unsupported model: nothing on the wire.
    expect(resolveLossyAttention({ backend: "mlx", model: BOTH, options: on() }).block).toBeNull();
    expect(resolveLossyAttention({ backend: "candle", model: NEITHER, options: on() }).block).toBeNull();
    for (const ratio of [0, -1, 1.5, "abc", ""]) {
      const result = resolveLossyAttention({
        backend: "candle",
        model: BOTH,
        options: on({ kernel: "sol", topkRatio: ratio }),
      });
      expect(result.block).toBeNull();
      expect(result.error).toMatch(/top-k ratio/);
    }
  });

  it("restores the studio state from a recipe", () => {
    expect(lossyAttentionFromRecipe(undefined)).toEqual(lossyAttentionDefaults());
    expect(lossyAttentionFromRecipe({ int8: { pvPrecision: "int8" } })).toMatchObject({
      enabled: true,
      kernel: "int8",
      pvPrecision: "int8",
    });
    expect(lossyAttentionFromRecipe({ sol: { topkRatio: 0.25 } })).toMatchObject({
      enabled: true,
      kernel: "sol",
      topkRatio: 0.25,
    });
  });

  it("summarizes whether the kernel actually ran", () => {
    expect(attentionExecutionSummary({ int8: 228, sol: 0, total: 264 })).toBe("INT8 228/264 attention calls");
    expect(attentionExecutionSummary({ int8: 0, sol: 0, total: 40 })).toBe("no lossy kernel ran (40 attention calls)");
    expect(attentionExecutionSummary(null)).toBeNull();
  });
});
