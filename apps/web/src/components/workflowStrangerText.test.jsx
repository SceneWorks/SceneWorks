import { readFileSync } from "node:fs";
import { join } from "node:path";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { WorkflowMissingInputs } from "./WorkflowMissingInputs.jsx";

// Every surface of the workflow offer that renders a string a STRANGER wrote has to be able to
// break it (sc-15952 review).
//
// The strings come out of a PNG someone else made. `workflow_share`'s sanitizer bounds a label at
// `LABEL_MAX_CHARS` (200) and — this is the part that matters — DROPS an over-length one rather
// than truncating it, so a 200-character slug is not a pathological input that gets cut down on
// the way in: it is a valid envelope value that arrives whole and lands inside the report's
// sentences ("No model matching `…`"). The panel is a 560px modal with `overflow-y: auto` and no
// `overflow-x`, so one unbreakable 200-character word pushes the whole dialog sideways and the
// user cannot scroll to what it pushed off.
//
// vitest never loads styles.css and jsdom does no layout, so `getComputedStyle` sees nothing and
// no component test can observe the overflow. The only assertion that can catch a CSS-only
// regression is one that reads the CSS — the same approach `validation/styles.test.js` takes for
// the readiness pill. The pairing below is what makes it meaningful: first that a 200-character
// slug really does reach the DOM inside these classes, then that those classes wrap.

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

// The `workflow_share` ceiling. A label at exactly this length is ACCEPTED; one character more is
// dropped — so this is the longest unbroken run the reader can be handed.
const LABEL_MAX_CHARS = 200;
const HOSTILE_SLUG = "z".repeat(LABEL_MAX_CHARS);

describe("styles.css: stranger-written workflow text can be broken", () => {
  // Both the report's rows (sc-15951) and the fix panel's rows (sc-15952) render the same
  // server-written `detail` sentences, which interpolate the raw slug; the chips interpolate it
  // directly. Listing them together is the point — `.workflow-fix-head` had the property and its
  // own sibling `.workflow-fix-detail` did not, which is how the hole opened.
  it.each([
    ".workflow-fix-head",
    ".workflow-fix-detail",
    ".workflow-req-head",
    ".workflow-req-detail",
    ".validation-chip",
    ".workflow-drop-field-value",
  ])("%s breaks an unbreakable word rather than widening the dialog", (selector) => {
    expect(declaration(ruleBody(selector), "overflow-wrap")).toBe("anywhere");
  });

  it("keeps the panel itself from being the thing that scrolls sideways", () => {
    // The other half of the same failure: if the modal ever gains `overflow-x: auto` the wrap
    // rules stop being load-bearing and the next long slug silently reintroduces a sideways
    // scrollbar. It must not have one.
    expect(declaration(ruleBody(".workflow-fix"), "overflow-x")).toBeNull();
  });
});

describe("a 200-character slug reaches those classes intact", () => {
  let host = null;
  let root = null;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
  });

  it("renders the full slug inside .workflow-fix-detail", () => {
    // The sanitizer's ceiling is not a truncation: a 200-char slug is a legal envelope value, so
    // the reader really is handed 200 unbroken characters. If this ever starts arriving elided,
    // the CSS assertions above stop mattering and this test says so.
    act(() =>
      root.render(
        <WorkflowMissingInputs
          assets={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo" }]}
          report={{
            runnable: false,
            model: {
              slug: HOSTILE_SLUG,
              state: "missing",
              detail: `No model matching \`${HOSTILE_SLUG}\` is in this install's catalog.`,
            },
            loras: [],
            styles: [],
            inputs: [],
            omitted: [],
            inputImagesRequired: 0,
          }}
        />,
      ),
    );
    const details = [...host.querySelectorAll(".workflow-fix-detail")];
    expect(details.length).toBeGreaterThan(0);
    expect(details.some((node) => node.textContent.includes(HOSTILE_SLUG))).toBe(true);
    // And the substitute picker is present, so this is the real panel rather than a stub that
    // happens to carry the class.
    expect(host.querySelector("#workflow-substitute-model")).not.toBeNull();
  });
});
