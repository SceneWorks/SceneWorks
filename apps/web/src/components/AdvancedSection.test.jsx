import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AdvancedSection } from "./AdvancedSection.jsx";

// AdvancedSection expands in place below the work-panel (sc-10436). Proves the AC:
// the body only renders when open, the caret/label toggle calls onToggle, and
// `actions` render alongside their own click handler.

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

function render(ui) {
  act(() => root.render(ui));
}

describe("AdvancedSection (sc-10436)", () => {
  it("hides the body when collapsed and shows it when open", () => {
    render(
      <AdvancedSection open={false} onToggle={() => {}}>
        <div>knobs</div>
      </AdvancedSection>,
    );
    expect(container.querySelector(".advanced-section-body")).toBeNull();
    expect(container.querySelector(".advanced-section.open")).toBeNull();

    render(
      <AdvancedSection open onToggle={() => {}}>
        <div>knobs</div>
      </AdvancedSection>,
    );
    expect(container.querySelector(".advanced-section-body").textContent).toContain("knobs");
    expect(container.querySelector(".advanced-section.open")).toBeTruthy();
  });

  it("calls onToggle when the header toggle is clicked", () => {
    let toggles = 0;
    render(
      <AdvancedSection open={false} onToggle={() => (toggles += 1)}>
        <div>knobs</div>
      </AdvancedSection>,
    );
    act(() => container.querySelector(".advanced-section-toggle").click());
    act(() => container.querySelector(".advanced-section-caret-btn").click());
    expect(toggles).toBe(2);
  });

  it("renders actions with their own handler", () => {
    let reset = 0;
    render(
      <AdvancedSection
        open
        onToggle={() => {}}
        actions={
          <button type="button" className="reset-defaults" onClick={() => (reset += 1)}>
            Reset
          </button>
        }
      >
        <div>knobs</div>
      </AdvancedSection>,
    );
    act(() => container.querySelector(".reset-defaults").click());
    expect(reset).toBe(1);
  });
});
