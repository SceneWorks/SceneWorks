// Transparency reads as a checkerboard, not as black (sc-24111).
//
// jsdom applies no stylesheet and computes no background, so this cannot be a rendering test. What
// it can be — and what the first version of it was not — is a CASCADE test. Grepping the source
// for the checkerboard block proves the block exists; it does not prove the block wins. A later,
// more specific rule setting `background: var(--bg-2)` on the same element leaves every substring
// assertion green while the browser paints a flat near-black surface, which is the entire defect
// this story is about.
//
// So each surface below is described as a concrete element, every rule in BOTH shells' stylesheets
// that would match it is collected, and the winner for `background-image` is resolved by
// (specificity, document order) — treating a `background:` shorthand as the `background-image`
// reset it actually is. See `testUtils/cssCascade.js`.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  backgroundShorthand,
  parseRules,
  specificity,
  winningDeclaration,
} from "./testUtils/cssCascade.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const ADVANCED = path.join(here, "styles.css");
// The Simple shell is a whole second product surface with its own result view, asset grid and
// fullscreen preview, and it shares none of the advanced shell's class names. It is also imported
// later (SimpleShell mounts it), so it is appended after — which is what lets the resolver catch a
// simple.css rule that re-flattens an advanced-shell surface.
const SIMPLE = path.join(here, "simple", "simple.css");

const advanced = readFileSync(ADVANCED, "utf8");
const simple = readFileSync(SIMPLE, "utf8");

function loadRules(extraSimpleCss = "") {
  const first = parseRules(advanced, { source: "styles.css" });
  const second = parseRules(simple + extraSimpleCss, {
    source: "simple.css",
    startOrder: first.nextOrder,
  });
  return [...first.rules, ...second.rules];
}

const RULES = loadRules();

/** Two 45-degree gradients offset by half a tile — the app's checkerboard, from `.ie-canvas`. */
const isCheckerboard = (value) =>
  (value.match(/linear-gradient\(\s*45deg/g) ?? []).length === 2;

/**
 * Every surface an RGBA asset can be painted on, described as the element it actually is.
 *
 * `ancestorClasses` is what makes the resolver able to decide whether a rule like
 * `.preview-modal-stage img` competes for this element — which is the question a grep cannot ask.
 */
const SURFACES = [
  {
    name: "library / asset grid tile",
    shell: "advanced",
    element: { tag: "img", classes: [], ancestorClasses: ["asset-grid", "asset-tile"] },
  },
  {
    name: "Image Studio result card",
    shell: "advanced",
    element: { tag: "img", classes: [], ancestorClasses: ["review-grid", "review-card"] },
  },
  {
    name: "asset detail preview",
    shell: "advanced",
    element: { tag: "img", classes: [], ancestorClasses: ["asset-detail", "preview-button"] },
  },
  {
    name: "tray item",
    shell: "advanced",
    element: { tag: "img", classes: [], ancestorClasses: ["tray-item"] },
  },
  {
    name: "reference tile",
    shell: "advanced",
    element: { tag: "img", classes: [], ancestorClasses: ["reference-card", "reference-media"] },
  },
  {
    name: "fullscreen preview",
    shell: "advanced",
    element: {
      tag: "img",
      classes: [],
      ancestorClasses: [
        "preview-modal",
        "preview-modal-stage",
        "preview-zoom-viewport",
        "preview-zoom-inner",
      ],
    },
  },
  {
    name: "queue / job-output thumbnail",
    shell: "advanced",
    element: {
      tag: "img",
      classes: ["worker-progress-card__thumb-media"],
      ancestorClasses: ["worker-progress-card", "worker-progress-card__thumb-cell"],
    },
  },
  {
    name: "Image Editor layer thumbnail",
    shell: "advanced",
    element: {
      tag: "img",
      classes: ["ie-layer-thumb"],
      ancestorClasses: ["ie-layers", "ie-layer", "ie-layer-row"],
    },
  },
  {
    name: "Simple shell result",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-results-grid", "su-result"] },
  },
  {
    name: "Simple shell asset grid",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-asset-grid", "su-asset"] },
  },
  {
    name: "Simple shell fullscreen preview",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-preview-stage"] },
  },
  {
    name: "Simple shell reference chip",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-tile-ref"] },
  },
  {
    name: "Simple shell option row",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-option-row"] },
  },
  {
    name: "Simple shell style thumbnail",
    shell: "simple",
    element: { tag: "img", classes: [], ancestorClasses: ["su-style-thumb"] },
  },
];

describe("the transparency backdrop wins the cascade", () => {
  it.each(SURFACES)("$name", ({ element }) => {
    const winner = winningDeclaration(
      RULES,
      element,
      "background-image",
      backgroundShorthand,
    );
    expect(winner, "no rule paints a background on this surface at all").not.toBeNull();
    expect(
      isCheckerboard(winner.value),
      `the winning background for this surface is \`${winner.via}: ${winner.value.replace(/\s+/g, " ").slice(0, 120)}\` from \`${winner.selector}\` (${winner.rule.source}), not the checkerboard`,
    ).toBe(true);
  });

  it("is blind to neither a shell nor a media query", () => {
    // Guards the resolver itself: if `parseRules` silently returned nothing for one of the two
    // sheets, every assertion above would be resolving over the wrong corpus.
    expect(RULES.some((rule) => rule.source === "styles.css")).toBe(true);
    expect(RULES.some((rule) => rule.source === "simple.css")).toBe(true);
    expect(RULES.length).toBeGreaterThan(500);
  });
});

describe("the checkerboard itself", () => {
  it("is a checkerboard and not a stripe, in both shells", () => {
    // One gradient would be diagonal stripes — which is the repo's OTHER 45-degree pattern, the
    // empty-state hazard fill, and means something else entirely.
    for (const [label, css] of [
      ["styles.css", advanced],
      ["simple.css", simple],
    ]) {
      const at = css.indexOf("--checker-tile:");
      expect(at, `${label} has no checkerboard block`).toBeGreaterThan(-1);
      const block = css.slice(at, css.indexOf("}", at));
      expect(block.match(/linear-gradient\(\s*45deg/g), label).toHaveLength(2);
      expect(block, label).toContain(
        "background-position: 0 0, calc(var(--checker-tile) / 2)",
      );
    }
  });

  it("leaves <video> surfaces flat", () => {
    // No video format this app plays carries alpha, so a checkerboard behind one could only ever
    // be decoration. Resolved the same way as the image surfaces, so this is a claim about what
    // the browser paints rather than about what the selector list says.
    for (const ancestorClasses of [["asset-tile"], ["preview-button"], ["su-result"]]) {
      const winner = winningDeclaration(
        RULES,
        { tag: "video", classes: [], ancestorClasses },
        "background-image",
        backgroundShorthand,
      );
      expect(
        winner === null || !isCheckerboard(winner.value),
        `video inside .${ancestorClasses[0]} got a checkerboard`,
      ).toBe(true);
    }
  });

  it("keeps the Image Editor canvas checkerboarded", () => {
    // The third surface the story names, and the one that already worked — the sc-22461 color-key
    // and SAM3 cutout previews read against it.
    const winner = winningDeclaration(
      RULES,
      { tag: "div", classes: ["ie-canvas"], ancestorClasses: ["ie-shell"] },
      "background-image",
      backgroundShorthand,
    );
    expect(winner).not.toBeNull();
    expect(isCheckerboard(winner.value)).toBe(true);
  });
});

describe("the fullscreen preview is sized to its bitmap", () => {
  it("wins `width` against .preview-modal-stage img", () => {
    // If `width` does not resolve to `auto` here, a portrait image letterboxes inside the stage
    // and the bars paint checkerboard — the checkerboard is only safe on this surface BECAUSE the
    // element is sized to its bitmap. Asserting the WINNER rather than the presence of the
    // declaration is the whole point: the competitor is `.preview-modal-stage img,
    // .preview-modal-stage video { width: 100% }`, and whether the override beats it is a cascade
    // question, not a grep question.
    const winner = winningDeclaration(
      RULES,
      {
        tag: "img",
        classes: [],
        ancestorClasses: [
          "preview-modal",
          "preview-modal-stage",
          "preview-zoom-viewport",
          "preview-zoom-inner",
        ],
      },
      "width",
    );
    expect(winner).not.toBeNull();
    expect(
      winner.value.trim(),
      `\`width\` is won by \`${winner.selector}\``,
    ).toBe("auto");
    // And it is not winning by accident of being matched more loosely than the competitor: it is
    // at least as specific as the rule it overrides.
    expect(
      compareSpecificity(
        specificity(winner.selector),
        specificity(".preview-modal-stage img"),
      ),
    ).toBeGreaterThanOrEqual(0);
  });

  it("still lets a video fill the stage", () => {
    // The override is `img`-only on purpose: a <video> has no transparency to reveal and its
    // player chrome expects the full stage width.
    const winner = winningDeclaration(
      RULES,
      {
        tag: "video",
        classes: [],
        ancestorClasses: ["preview-modal", "preview-modal-stage"],
      },
      "width",
    );
    expect(winner?.value.trim()).toBe("100%");
  });
});

function compareSpecificity(a, b) {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}
