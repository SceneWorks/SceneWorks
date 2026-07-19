import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { StyledPromptPreview } from "./StyledPromptPreview.jsx";

// sc-13131 — The preview component's ONE job: display the composed prompt string it is handed,
// byte-for-byte (including the Style:/Description: newlines), and render nothing when inactive.
// The anti-drift guarantee lives one level up (ImageStudio feeds it the same buildJobRequest output
// that is submitted); this file pins that the component itself never mangles or hides that string.
describe("StyledPromptPreview", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => {
      root.render(ui);
    });
  }

  const node = () => container.querySelector('[data-testid="styled-prompt-preview"]');

  it("renders nothing when no style is active", () => {
    render(<StyledPromptPreview active={false} composedPrompt={"Style: x\nDescription: y"} />);
    expect(node()).toBeNull();
    expect(container.textContent).toBe("");
  });

  it("shows the composed prompt byte-for-byte, preserving Style/Description line breaks", () => {
    const composed = "Style: cinematic watercolor\nDescription: a fox in the snow";
    render(<StyledPromptPreview active composedPrompt={composed} />);
    const paragraph = node().querySelector(".preset-stack-prompt p");
    // textContent must equal the source string exactly — no trimming, no whitespace collapse,
    // no HTML-escaping. Discriminates a component that mangles the payload string.
    expect(paragraph.textContent).toBe(composed);
  });

  it("reflects a multi-directive merge composition verbatim", () => {
    const composed = "Style: oil painting, moody\nLighting: soft\nDescription: a portrait";
    render(<StyledPromptPreview active composedPrompt={composed} />);
    expect(node().querySelector(".preset-stack-prompt p").textContent).toBe(composed);
  });

  it("falls back to a placeholder token when the composed prompt is empty", () => {
    render(<StyledPromptPreview active composedPrompt="" />);
    expect(node().querySelector(".token").textContent).toBe("your prompt");
  });

  // sc-13133: the composed-prompt budget readout lives here because a style is active exactly when
  // this preview renders. The count is measured on the SAME composed string shown above.
  const budget = () => container.querySelector('[data-testid="styled-prompt-budget"]');

  it("shows the composed length / cap for an under-budget prompt, with no warning", () => {
    const composed = "Style: cinematic\nDescription: a fox in the snow";
    render(<StyledPromptPreview active composedPrompt={composed} />);
    // Measured on the composed string the component was handed (Unicode scalar values).
    expect(budget().textContent).toContain(`${[...composed].length} / 4000`);
    expect(budget().classList.contains("over")).toBe(false);
    expect(budget().getAttribute("role")).toBeNull();
  });

  it("flips to an over-budget warning (with alert role) when the composed prompt exceeds the cap", () => {
    const composed = `Style: ${"x".repeat(4000)}\nDescription: a fox`;
    render(<StyledPromptPreview active composedPrompt={composed} />);
    const over = [...composed].length;
    expect(over).toBeGreaterThan(4000);
    expect(budget().classList.contains("over")).toBe(true);
    expect(budget().getAttribute("role")).toBe("alert");
    expect(budget().textContent).toContain(`${over} / 4000`);
    expect(budget().textContent).toContain("shorten your prompt or pick a shorter style");
  });

  it("renders no budget readout when inactive", () => {
    render(<StyledPromptPreview active={false} composedPrompt={"x".repeat(5000)} />);
    expect(budget()).toBeNull();
  });
});
