import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import {
  FLASH_ATTENTION_TOOLTIP,
  FlashAttentionToggle,
  flashAttentionAvailable,
} from "./FlashAttentionToggle.jsx";

let root;
let container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));

function render(props) {
  act(() => root.render(<FlashAttentionToggle checked onChange={() => {}} {...props} />));
}

describe("FlashAttentionToggle (SW-7276)", () => {
  it("is hidden on the Mac/MLX route", () => {
    render({ backend: "mlx" });
    expect(container.querySelector(".flash-attn-toggle")).toBeNull();
    expect(container.textContent).not.toContain("Flash attention");
    expect(flashAttentionAvailable("mlx")).toBe(false);
  });

  it("is shown on the candle route with the shared-kernel tooltip", () => {
    render({ backend: "candle" });
    const label = container.querySelector(".flash-attn-toggle");
    expect(label).not.toBeNull();
    expect(label.textContent).toContain("Flash attention");
    expect(label.getAttribute("title")).toBe(
      "FlashAttention-2 for this model's attention on CUDA (candle) — faster and lower VRAM. " +
        "Calls the kernel can't take (masked text encoders, f32 paths) stay on the standard path.",
    );
    expect(FLASH_ATTENTION_TOOLTIP).toBe(label.getAttribute("title"));
    expect(label.getAttribute("title")).not.toMatch(/SDXL|Ignored on other backends/);
  });

  it("reports the new checked state", () => {
    const onChange = vi.fn();
    render({ backend: "candle", checked: true, onChange });
    const input = container.querySelector("input[type=checkbox]");
    expect(input.checked).toBe(true);
    act(() => input.click());
    expect(onChange).toHaveBeenCalledWith(false);
  });
});
