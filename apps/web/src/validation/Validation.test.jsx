import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { issue, summarize } from "./issues.js";
import { invalidProps, ReadyPill, ValidationSummary } from "./Validation.jsx";

// The rendering layer of the app-wide validation core (epic 10644, sc-10646). Proves
// the AC: the chip row is the only place a message appears, requirements never reach
// it, the two kinds are visually distinct, the readiness pill carries a different
// class per state (its predecessor carried the same accent tone in both), and a broken
// input is marked with an attribute rather than wrapped in a node.

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

function render(element) {
  act(() => root.render(element));
}

describe("ValidationSummary", () => {
  it("renders nothing at all when there is nothing to say", () => {
    render(<ValidationSummary issues={[]} />);
    expect(container.querySelector(".validation-chips")).toBeNull();
    // An empty bordered box is worse than silence, so not even the container.
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when handed undefined", () => {
    render(<ValidationSummary issues={undefined} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders one chip per message, in rule order", () => {
    render(
      <ValidationSummary
        issues={[
          issue.error("rank", "Rank must be greater than zero"),
          issue.advisory("captions", "3 captions may not match"),
        ]}
      />,
    );
    const chips = [...container.querySelectorAll(".validation-chip")];
    expect(chips.map((chip) => chip.textContent)).toEqual([
      "Rank must be greater than zero",
      "3 captions may not match",
    ]);
  });

  it("tones an error and an advisory differently", () => {
    render(
      <ValidationSummary
        issues={[issue.error("rank", "broken"), issue.advisory("captions", "heads up")]}
      />,
    );
    const chips = [...container.querySelectorAll(".validation-chip")];
    expect(chips[0].className).toContain("tone-error");
    expect(chips[1].className).toContain("tone-advisory");
    expect(chips[0].className).not.toBe(chips[1].className);
  });

  // The whole point of the requirement kind. The core drops them, so a chip row fed
  // from `summarize().surfaced` cannot show one — assert through the real seam rather
  // than by hand-filtering, or the test proves nothing about the pipeline.
  it("shows no chip for a form that is merely unfilled", () => {
    const summary = summarize([
      issue.requirement("dataset", "Select a saved dataset"),
      issue.requirement("outputName", "Name the LoRA output"),
    ]);
    render(<ValidationSummary issues={summary.surfaced} />);
    expect(container.innerHTML).toBe("");
    expect(summary.ready).toBe(false);
  });

  it("shows the broken value but not the unfilled field alongside it", () => {
    const summary = summarize([
      issue.requirement("dataset", "Select a saved dataset"),
      issue.error("rank", "Rank must be greater than zero"),
    ]);
    render(<ValidationSummary issues={summary.surfaced} />);
    const chips = [...container.querySelectorAll(".validation-chip")];
    expect(chips).toHaveLength(1);
    expect(chips[0].textContent).toBe("Rank must be greater than zero");
  });

  it("labels the row for assistive tech", () => {
    render(<ValidationSummary issues={[issue.error(null, "boom")]} label="Configuration errors" />);
    expect(container.querySelector(".validation-chips").getAttribute("aria-label")).toBe(
      "Configuration errors",
    );
  });
});

describe("ReadyPill", () => {
  it("says Ready and carries the ready class", () => {
    render(<ReadyPill ready />);
    const pill = container.querySelector(".ready-pill");
    expect(pill.textContent).toBe("Ready");
    expect(pill.className).toContain("is-ready");
  });

  it("says Needs input and carries a different class", () => {
    render(<ReadyPill ready={false} />);
    const pill = container.querySelector(".ready-pill");
    expect(pill.textContent).toBe("Needs input");
    expect(pill.className).toContain("is-pending");
  });

  // The predecessor `.training-status-pill` rendered both states with the same accent
  // styling, so only the text disagreed. A test asserting the text would have passed
  // against that bug. This asserts the two states are distinguishable by class; that
  // the classes are also *coloured* differently is pinned in styles.test.js, since no
  // JS assertion can see a stylesheet vitest never loads.
  it("distinguishes its two states by more than text", () => {
    render(<ReadyPill ready />);
    const readyClass = container.querySelector(".ready-pill").className;
    render(<ReadyPill ready={false} />);
    const pendingClass = container.querySelector(".ready-pill").className;
    expect(readyClass).not.toBe(pendingClass);
  });
});

describe("invalidProps", () => {
  it("marks a field carrying an error", () => {
    const summary = summarize([issue.error("rank", "Rank must be greater than zero")]);
    expect(invalidProps(summary, "rank")).toEqual({ "aria-invalid": "true" });
  });

  it("leaves an untouched required field unmarked", () => {
    const summary = summarize([issue.requirement("dataset", "Select a saved dataset")]);
    expect(invalidProps(summary, "dataset")).toEqual({});
  });

  it("leaves a clean field unmarked", () => {
    expect(invalidProps(summarize([]), "rank")).toEqual({});
  });

  it("tolerates a missing summary", () => {
    expect(invalidProps(undefined, "rank")).toEqual({});
  });

  // It returns an attribute, never text. If it ever returned a message the field would
  // repeat what the chip already said, and the two could drift apart.
  it("returns nothing renderable as a message", () => {
    const summary = summarize([issue.error("rank", "Rank must be greater than zero")]);
    const props = invalidProps(summary, "rank");
    expect(Object.keys(props)).toEqual(["aria-invalid"]);
    expect(JSON.stringify(props)).not.toContain("Rank must be");
  });

  // Spread onto a real input, the mark lands on the element itself — no wrapper node,
  // which in a CSS-grid form body would become a grid item and reflow the row.
  it("spreads onto an input without adding a node", () => {
    const summary = summarize([issue.error("rank", "broken")]);
    render(
      <label>
        Rank
        <input readOnly value="0" {...invalidProps(summary, "rank")} />
      </label>,
    );
    const input = container.querySelector("input");
    expect(input.getAttribute("aria-invalid")).toBe("true");
    expect(container.querySelectorAll("label > *")).toHaveLength(1);
  });
});
