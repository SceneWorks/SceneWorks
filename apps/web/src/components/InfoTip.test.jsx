import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { InfoTip, withoutInfoTips } from "./InfoTip.jsx";

// InfoTip moves a field's explanation off the grid into a tooltip: the text stays in the DOM and
// describes the (i) button, a click pins it open, an outside click or Escape closes it, and a click
// inside a <label> never reaches the labelled control.

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("InfoTip", () => {
  it("describes its button with the tooltip text and renders nothing without content", () => {
    act(() => root.render(<><InfoTip label="About Duration">Lengths follow the frame lattice.</InfoTip><InfoTip>{null}</InfoTip></>));
    const buttons = container.querySelectorAll(".info-tip-button");
    expect(buttons).toHaveLength(1);
    const tip = document.getElementById(buttons[0].getAttribute("aria-describedby"));
    expect(tip.getAttribute("role")).toBe("tooltip");
    expect(tip.textContent).toBe("Lengths follow the frame lattice.");
    expect(buttons[0].getAttribute("aria-label")).toBe("About Duration");
    // Kept out of an enclosing label's name; still the button's description.
    expect(tip.getAttribute("aria-hidden")).toBe("true");
  });

  it("strips tips from a caption with withoutInfoTips", () => {
    act(() => root.render(<label><span>Duration<InfoTip>Lattice note.</InfoTip></span><select /></label>));
    expect(withoutInfoTips(container.querySelector("label")).trim()).toBe("Duration");
  });

  it("pins open on click and closes on an outside click or Escape", () => {
    act(() => root.render(<InfoTip>Explanation</InfoTip>));
    const button = container.querySelector(".info-tip-button");
    act(() => button.click());
    expect(container.querySelector(".info-tip.pinned")).toBeTruthy();
    act(() => document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
    expect(container.querySelector(".info-tip.pinned")).toBeNull();
    act(() => button.click());
    act(() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })));
    expect(container.querySelector(".info-tip.pinned")).toBeNull();
  });

  it("does not toggle the control of the label it sits in", () => {
    const onChange = vi.fn();
    act(() =>
      root.render(
        <label>
          <input onChange={onChange} type="checkbox" />
          Enhance prompt
          <InfoTip tone="warn">Rewrites the prompt.</InfoTip>
        </label>,
      ),
    );
    act(() => container.querySelector(".info-tip-button").click());
    expect(container.querySelector("input").checked).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
    expect(container.querySelector(".info-tip.warn")).toBeTruthy();
  });
});
