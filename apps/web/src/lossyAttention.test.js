import { describe, expect, it } from "vitest";
import {
  DEFAULT_SOL_TOPK_RATIO,
  attentionExecutionSummary,
  lossyAttentionAvailable,
  lossyAttentionDefaults,
  effectiveLossyKernels,
  lossyAttentionFromRecipe,
  lossyAttentionPreference,
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
      recorded: null,
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

  // Codex P2 on #3528: replay must re-send the recorded configuration, not the control's subset.
  it("replays a recorded recipe unchanged, including fields the control does not expose", () => {
    const replay = (raw) =>
      resolveLossyAttention({ backend: "candle", model: BOTH, options: lossyAttentionFromRecipe(raw) }).block;
    const tauSol = { sol: { tau: 1, tail: false, sinkBlocks: [0, 2], sinkQ: [0, 0], queryChunkBlocks: 0 } };
    expect(replay(tauSol)).toEqual(tauSol);
    const centered = { int8: { pvPrecision: "value_dtype", kCentering: "always" } };
    expect(replay(centered)).toEqual(centered);
    const both = { int8: { kCentering: "off" }, sol: { topkRatio: 0.25, tail: false } };
    expect(replay(both)).toEqual(both);
    // The replayed recipe is not mutated by resolution.
    expect(tauSol.sol).not.toHaveProperty("topkRatio");
  });

  it("overlays only the fields the user edits onto a replayed recipe", () => {
    const state = lossyAttentionFromRecipe({ sol: { tau: 1, tail: false, sinkBlocks: [0, 2] } });
    expect(state).toMatchObject({ enabled: true, kernel: "sol", topkRatio: null });
    // Typing a top-k ratio replaces the tau selection and keeps every other recorded field.
    expect(
      resolveLossyAttention({ backend: "candle", model: BOTH, options: { ...state, topkRatio: "0.3" } }).block,
    ).toEqual({ sol: { topkRatio: 0.3, tail: false, sinkBlocks: [0, 2] } });
    const int8 = lossyAttentionFromRecipe({ int8: { kCentering: "always" } });
    expect(
      resolveLossyAttention({ backend: "candle", model: BOTH, options: { ...int8, pvPrecision: "int8" } }).block,
    ).toEqual({ int8: { kCentering: "always", pvPrecision: "int8" } });
    // Choosing one kernel of a two-kernel recipe keeps that kernel's recorded settings.
    const both = lossyAttentionFromRecipe({ int8: { kCentering: "off" }, sol: { tau: 2 } });
    expect(both.kernel).toBe("both");
    expect(
      resolveLossyAttention({ backend: "candle", model: BOTH, options: { ...both, kernel: "sol" } }).block,
    ).toEqual({ sol: { tau: 2 } });
  });

  it("refuses a replayed kernel the selected model cannot take instead of substituting one", () => {
    const state = lossyAttentionFromRecipe({ sol: { tau: 1 } });
    const result = resolveLossyAttention({ backend: "candle", model: INT8_ONLY, options: state });
    expect(result.block).toBeNull();
    expect(result.error).toMatch(/Sol-Attn/);
  });

  // Codex P2 on #3538: a replayed recipe that cannot run here blocks; a sticky preference is quiet.
  it("blocks a replayed recipe on MLX or a model with neither kernel, and drops a sticky preference", () => {
    const replay = lossyAttentionFromRecipe({ int8: { kCentering: "always" } });
    for (const [backend, model, pattern] of [
      ["mlx", BOTH, /CUDA/],
      ["candle", NEITHER, /cannot take/],
      ["candle", null, /cannot take/],
      ["candle", { id: "moss_image_base" }, /cannot take/],
    ]) {
      const result = resolveLossyAttention({ backend, model, options: replay });
      expect(result.block).toBeNull();
      expect(result.error).toMatch(pattern);
      expect(result.error).toMatch(/turn lossy attention off/);
    }
    // Turning the control off clears the block: the recipe runs on the exact path by choice.
    expect(resolveLossyAttention({ backend: "mlx", model: BOTH, options: { ...replay, enabled: false } })).toEqual({
      block: null,
      error: null,
    });
    // A sticky preference (no replayed recipe) is silently not offered.
    const sticky = { ...lossyAttentionDefaults(), enabled: true, kernel: "sol" };
    for (const [backend, model] of [["mlx", BOTH], ["candle", NEITHER], ["candle", null]]) {
      expect(resolveLossyAttention({ backend, model, options: sticky })).toEqual({ block: null, error: null });
    }
    // Unchanged: the same replay on a model serving its kernel re-sends it whole.
    expect(resolveLossyAttention({ backend: "candle", model: BOTH, options: replay }).block).toEqual({
      int8: { kCentering: "always" },
    });
  });

  // Codex P2 on #3538: what the control shows is exactly what is sent, across replay -> persist -> reload.
  it("sends exactly the displayed kernels after a two-kernel replay is persisted and reloaded", () => {
    const replay = lossyAttentionFromRecipe({ int8: { kCentering: "off" }, sol: { tau: 2 } });
    // While replaying, both kernels are shown and sent.
    expect(effectiveLossyKernels(replay, servedLossyKernels(BOTH))).toEqual({ kernels: ["int8", "sol"], kernel: "both" });
    expect(Object.keys(resolveLossyAttention({ backend: "candle", model: BOTH, options: replay }).block)).toEqual(["int8", "sol"]);
    // The studio persists that state; a reload turns it into a single-kernel preference.
    const reloaded = lossyAttentionPreference(JSON.parse(JSON.stringify(replay)));
    expect(reloaded).toMatchObject({ enabled: true, kernel: "int8", recorded: null });
    const shown = effectiveLossyKernels(reloaded, servedLossyKernels(BOTH));
    const sent = resolveLossyAttention({ backend: "candle", model: BOTH, options: reloaded });
    expect(shown).toEqual({ kernels: ["int8"], kernel: "int8" });
    expect(sent).toEqual({ block: { int8: {} }, error: null });
    expect(Object.keys(sent.block)).toEqual(shown.kernels);
  });

  it("normalizes an old persisted kernel: both into one coherent, unblocked kernel", () => {
    const persisted = lossyAttentionPreference({ enabled: true, kernel: "both", topkRatio: 0.3 });
    expect(persisted.kernel).toBe("int8");
    for (const model of [BOTH, INT8_ONLY]) {
      const shown = effectiveLossyKernels(persisted, servedLossyKernels(model));
      const sent = resolveLossyAttention({ backend: "candle", model, options: persisted });
      expect(sent.error).toBeNull();
      expect(Object.keys(sent.block)).toEqual(shown.kernels);
      expect(shown.kernels).toEqual(["int8"]);
    }
    // Even an un-normalized "both" preference (no recorded recipe) is one served kernel, never two.
    const raw = { ...lossyAttentionDefaults(), enabled: true, kernel: "both" };
    expect(effectiveLossyKernels(raw, servedLossyKernels(INT8_ONLY))).toEqual({ kernels: ["int8"], kernel: "int8" });
    expect(resolveLossyAttention({ backend: "candle", model: INT8_ONLY, options: raw })).toEqual({
      block: { int8: {} },
      error: null,
    });
    // A Sol-only preference stays Sol where it is served and falls back where it is not.
    const sol = lossyAttentionPreference({ enabled: true, kernel: "sol", topkRatio: 0.25 });
    expect(resolveLossyAttention({ backend: "candle", model: BOTH, options: sol }).block).toEqual({ sol: { topkRatio: 0.25 } });
    expect(resolveLossyAttention({ backend: "candle", model: INT8_ONLY, options: sol }).block).toEqual({ int8: {} });
  });
});
