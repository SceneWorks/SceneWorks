import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

// The readiness pill's predecessor, `.training-status-pill`, set
// `background: var(--accent-soft)` unconditionally — so "Needs input" was painted with
// the success accent and only its text said otherwise. No component test could see
// that: vitest never loads styles.css, so getComputedStyle reports nothing.
//
// The only assertion that catches a CSS-only bug is one that reads the CSS. This
// parses the two rules and pins them apart. It is narrow on purpose — it checks that
// the states differ, not what colour either one is.

// Resolved from the vitest root (apps/web) — `import.meta.url` is not a file: URL once
// vitest has transformed the module.
const CSS = readFileSync(join(process.cwd(), "src/styles.css"), "utf8");

// Body of the first rule whose selector list matches `selector` exactly.
function ruleBody(selector) {
  const escaped = selector.replaceAll(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
  const match = CSS.match(new RegExp(`^${escaped}\\s*\\{([^}]*)\\}`, "m"));
  if (!match) {
    throw new Error(`No CSS rule found for selector: ${selector}`);
  }
  return match[1];
}

function declaration(body, property) {
  const match = body.match(new RegExp(`(?:^|;)\\s*${property}\\s*:\\s*([^;]+)`));
  return match ? match[1].trim() : null;
}

describe("styles.css: the readiness pill tones its two states apart", () => {
  const ready = ruleBody(".ready-pill.is-ready");
  const pending = ruleBody(".ready-pill.is-pending");

  it("declares a background and colour for each state", () => {
    for (const [name, body] of [
      ["is-ready", ready],
      ["is-pending", pending],
    ]) {
      expect(declaration(body, "background"), `${name} background`).toBeTruthy();
      expect(declaration(body, "color"), `${name} color`).toBeTruthy();
    }
  });

  it("does not paint 'Needs input' with the 'Ready' background", () => {
    expect(declaration(pending, "background")).not.toBe(declaration(ready, "background"));
  });

  it("does not paint 'Needs input' with the 'Ready' text colour", () => {
    expect(declaration(pending, "color")).not.toBe(declaration(ready, "color"));
  });

  // The specific regression: the old pill used the success accent for both states.
  it("keeps the success accent off the pending state", () => {
    expect(declaration(ready, "background")).toContain("--accent-soft");
    expect(declaration(pending, "background")).not.toContain("--accent");
    expect(declaration(pending, "color")).not.toContain("--accent");
  });
});

describe("styles.css: the two chip kinds are visually distinct", () => {
  const error = ruleBody(".validation-chip.tone-error");
  const advisory = ruleBody(".validation-chip.tone-advisory");

  it("gives an error and an advisory different backgrounds and borders", () => {
    expect(declaration(advisory, "background")).not.toBe(declaration(error, "background"));
    expect(declaration(advisory, "border")).not.toBe(declaration(error, "border"));
  });

  // An advisory doesn't block. Dressing it in the danger colour would overstate it.
  it("reserves the danger hue for errors", () => {
    expect(declaration(error, "background")).toContain("--danger");
    expect(declaration(advisory, "background")).not.toContain("--danger");
    expect(declaration(advisory, "background")).toContain("--warn");
  });
});
