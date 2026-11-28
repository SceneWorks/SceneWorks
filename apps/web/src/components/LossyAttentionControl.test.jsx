import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { LOSSY_ATTENTION_TOOLTIP, LossyAttentionControl } from "./LossyAttentionControl.jsx";
import AttentionExecutionNote from "./AttentionExecutionNote.jsx";
import { lossyAttentionDefaults, lossyAttentionFromRecipe } from "../lossyAttention.js";

const BOTH = { id: "krea2", lossyAttention: { int8: { served: true }, sol: { served: true } } };
const INT8_ONLY = {
  id: "qwen_image_2_1",
  lossyAttention: { int8: { served: true }, sol: { served: false, reason: "attends block-causally" } },
};

let root;
let container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));

function render(props) {
  act(() =>
    root.render(
      <LossyAttentionControl backend="candle" model={BOTH} onChange={() => {}} value={lossyAttentionDefaults()} {...props} />,
    ),
  );
}

describe("LossyAttentionControl (SW-6132)", () => {
  it("is hidden on MLX and for a model whose attention takes neither kernel", () => {
    render({ backend: "mlx" });
    expect(container.querySelector(".lossy-attention-control")).toBeNull();
    render({ model: { id: "moss_image_base" } });
    expect(container.querySelector(".lossy-attention-control")).toBeNull();
  });

  it("is collapsed and off by default, labelled lossy and experimental", () => {
    render({});
    const details = container.querySelector("details.lossy-attention-control");
    expect(details).not.toBeNull();
    expect(details.open).toBe(false);
    expect(details.querySelector("summary").textContent).toBe("Experimental: lossy attention");
    expect(details.querySelector("summary").getAttribute("title")).toBe(LOSSY_ATTENTION_TOOLTIP);
    expect(LOSSY_ATTENTION_TOOLTIP).toMatch(/lossy/i);
    const toggle = container.querySelector('input[aria-label="Lossy attention (experimental)"]');
    expect(toggle.checked).toBe(false);
    expect(container.querySelector('select[aria-label="Lossy attention kernel"]')).toBeNull();
  });

  it("enables with INT8 recommended and offers only the kernels the model serves", () => {
    const onChange = vi.fn();
    render({ onChange });
    act(() => container.querySelector('input[aria-label="Lossy attention (experimental)"]').click());
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, kernel: "int8" }));

    render({ model: BOTH, value: { ...lossyAttentionDefaults(), enabled: true } });
    const kernels = [...container.querySelectorAll('select[aria-label="Lossy attention kernel"] option')];
    expect(kernels.map((option) => option.value)).toEqual(["int8", "sol"]);
    expect(kernels[0].textContent).toContain("recommended");
    expect(container.querySelector('select[aria-label="INT8 attention P·V precision"]').value).toBe("value_dtype");

    render({ model: INT8_ONLY, value: { ...lossyAttentionDefaults(), enabled: true, kernel: "sol" } });
    const served = [...container.querySelectorAll('select[aria-label="Lossy attention kernel"] option')];
    expect(served.map((option) => option.value)).toEqual(["int8"]);
  });

  it("shows the Sol-Attn top-k ratio with its default and surfaces a range error", () => {
    render({ value: { ...lossyAttentionDefaults(), enabled: true, kernel: "sol" } });
    const ratio = container.querySelector('input[aria-label="Sol-Attn top-k ratio"]');
    expect(ratio).not.toBeNull();
    expect(ratio.value).toBe("0.5");
    expect(ratio.getAttribute("max")).toBe("1");
    render({
      value: { ...lossyAttentionDefaults(), enabled: true, kernel: "sol", topkRatio: 2 },
      error: "Sol-Attn top-k ratio must be a number above 0 and at most 1.",
    });
    expect(container.querySelector(".field-error").textContent).toMatch(/top-k ratio/);
  });

  it("shows an opted-in asset's receipt and nothing for a plain asset", () => {
    const asset = { recipe: { rawAdapterSettings: { attentionExecution: { int8: 228, sol: 0, total: 264 } } } };
    act(() => root.render(<AttentionExecutionNote asset={asset} />));
    expect(container.querySelector(".attention-execution-note").textContent).toBe(
      "Lossy attention: INT8 228/264 attention calls",
    );
    act(() => root.render(<AttentionExecutionNote asset={{ recipe: { rawAdapterSettings: {} } }} />));
    expect(container.querySelector(".attention-execution-note")).toBeNull();
  });

  it("shows a replayed recipe's own selection and keeps what it does not expose", () => {
    render({ value: lossyAttentionFromRecipe({ int8: { kCentering: "always" }, sol: { tau: 1, tail: false } }) });
    expect(container.querySelector('select[aria-label="Lossy attention kernel"]').value).toBe("both");
    const ratio = container.querySelector('input[aria-label="Sol-Attn top-k ratio"]');
    expect(ratio.value).toBe("");
    expect(ratio.getAttribute("placeholder")).toBe("tau 1 (from the recipe)");
    expect(container.querySelector('select[aria-label="INT8 attention P·V precision"]')).not.toBeNull();
    expect(container.textContent).toContain("Recipe settings not shown here are kept as recorded.");
  });

  it("keeps a blocked replay visible with its error and an off switch", () => {
    const onChange = vi.fn();
    const replay = lossyAttentionFromRecipe({ sol: { tau: 1 } });
    render({ backend: "mlx", value: replay, onChange, error: "This recipe used lossy attention, which runs only on CUDA (candle) workers; turn lossy attention off to run it on the exact path." });
    const details = container.querySelector("details.lossy-attention-control");
    expect(details.open).toBe(true);
    expect(container.querySelector(".field-error").textContent).toMatch(/CUDA/);
    expect(container.querySelector('select[aria-label="Lossy attention kernel"]')).toBeNull();
    act(() => container.querySelector('input[aria-label="Lossy attention (experimental)"]').click());
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ enabled: false }));
    // A sticky preference on MLX stays hidden.
    render({ backend: "mlx", value: { ...lossyAttentionDefaults(), enabled: true } });
    expect(container.querySelector(".lossy-attention-control")).toBeNull();
  });
});
