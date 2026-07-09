import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { WorkPanel } from "./WorkPanel.jsx";

// WorkPanel is the one elevated card per page (Page-Frame standard, sc-10436).
// Proves the AC: it renders its children, always paints the 3px accent top-rule,
// and only shows the head when an eyebrow/hint/actions is supplied.

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

describe("WorkPanel (sc-10436)", () => {
  it("renders children and the accent top-rule", () => {
    render(
      <WorkPanel>
        <button type="button">Generate</button>
      </WorkPanel>,
    );
    const panel = container.querySelector(".work-panel");
    expect(panel).toBeTruthy();
    expect(panel.querySelector(".work-panel-rule")).toBeTruthy();
    expect(panel.textContent).toContain("Generate");
  });

  it("shows the head with an accent eyebrow only when provided", () => {
    render(
      <WorkPanel eyebrow="Add a job" hint="Queue a prompt to a GPU">
        <div>body</div>
      </WorkPanel>,
    );
    const eyebrow = container.querySelector(".work-panel-eyebrow");
    expect(eyebrow.textContent).toBe("Add a job");
    expect(container.querySelector(".work-panel-hint").textContent).toBe("Queue a prompt to a GPU");
  });

  it("omits the head when no eyebrow/hint/actions is given", () => {
    render(
      <WorkPanel>
        <div>tabs first</div>
      </WorkPanel>,
    );
    expect(container.querySelector(".work-panel-head")).toBeNull();
  });

  it("passes through className", () => {
    render(<WorkPanel className="queue-composer">x</WorkPanel>);
    const panel = container.querySelector(".work-panel");
    expect(panel.classList.contains("queue-composer")).toBe(true);
  });
});
