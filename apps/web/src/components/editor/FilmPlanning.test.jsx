import { readFileSync } from "node:fs";
import { join } from "node:path";

import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FilmPlanning } from "./FilmPlanning.jsx";

const CSS = readFileSync(join(process.cwd(), "src/styles.css"), "utf8");

function ruleBody(selector) {
  const escaped = selector.replaceAll(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
  const match = CSS.match(new RegExp(`^${escaped}\\s*\\{([^}]*)\\}`, "m"));
  if (!match) throw new Error(`No CSS rule found for selector: ${selector}`);
  return match[1];
}

function declaration(body, property) {
  const match = body.match(new RegExp(`(?:^|;)\\s*${property}\\s*:\\s*([^;]+)`));
  return match ? match[1].trim() : null;
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

describe("FilmPlanning", () => {
  it("renders long failed-plan findings as a labeled list with shot and field context", async () => {
    const referencePack = "film_a96f0c96c6a64e1da49bbbb6a5b9f976-references";
    const rejectedPath = `/Volumes/Models/codex/sc-23730/runtime/terminal/evidence/projects/example/films/planning/operations/filmplan_4af8883889984985823abb5c53ff7a0b/planner-rejected.txt`;
    const draft = {
      originalScript: "A courier enters.",
      planning: { provider: "prompt_refiner", thinkingMode: "disabled", refinePrompts: false },
      productionPlan: { model: { id: "minimax_h3" } },
    };
    const operation = {
      status: "failed",
      stage: "failed",
      detail: "The planner returned an invalid plan.",
      plannerModel: "TheDrummer/Anubis-Mini-8B-v1",
      videoModelId: "minimax_h3",
      findings: [
        { shotId: "SH010", field: "continuityRoles", message: `reference role "courier" is not in reference pack "${referencePack}" (roles: )` },
        { field: "planner.repair", message: `the refused answer is at ${rejectedPath}` },
      ],
    };

    root = createRoot(container);
    await act(async () => root.render(<FilmPlanning
      availability={{ providers: [] }}
      disabled={false}
      draft={draft}
      models={[]}
      onApply={vi.fn()}
      onCancel={vi.fn()}
      onChange={vi.fn()}
      onInstall={vi.fn()}
      onNotice={vi.fn()}
      onStart={vi.fn()}
      operation={operation}
      token=""
    />));

    const findings = container.querySelector('ul[aria-label="Planning findings"]');
    const items = [...findings.querySelectorAll("li")];
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain("SH010 · continuityRoles:");
    expect(items[0].textContent).toContain(referencePack);
    expect(items[1].textContent).toContain("planner.repair:");
    expect(items[1].textContent).toContain(rejectedPath);
    expect(container.querySelector(".ve-film-operation").getAttribute("aria-live")).toBe("polite");
  });

  it("keeps operation metadata and findings shrinkable and breakable", () => {
    const operation = ruleBody(".ve-film-operation");
    const metadata = ruleBody(".ve-film-operation > span");
    const finding = ruleBody(".ve-film-planning-findings li");

    expect(declaration(operation, "box-sizing")).toBe("border-box");
    expect(declaration(operation, "min-width")).toBe("0");
    expect(declaration(operation, "max-width")).toBe("100%");
    expect(declaration(metadata, "overflow-wrap")).toBe("anywhere");
    expect(declaration(finding, "white-space")).toBe("normal");
    expect(declaration(finding, "overflow-wrap")).toBe("anywhere");
  });
});
