import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AdvancedSection } from "./AdvancedSection.jsx";

// AdvancedSection is the canonical Advanced disclosure (sc-10436, sc-10474): a
// bordered block whose header row is the toggle and whose controls open
// contiguously beneath that header. Proves the AC: the body only renders when
// open, both header affordances call onToggle, the Show/Hide label flips, and
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

  it("flips the Show/Hide label with the open state", () => {
    render(
      <AdvancedSection open={false} onToggle={() => {}}>
        <div>knobs</div>
      </AdvancedSection>,
    );
    expect(container.querySelector(".advanced-section-caret-label").textContent).toBe("Show");

    render(
      <AdvancedSection open onToggle={() => {}}>
        <div>knobs</div>
      </AdvancedSection>,
    );
    expect(container.querySelector(".advanced-section-caret-label").textContent).toBe("Hide");
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
