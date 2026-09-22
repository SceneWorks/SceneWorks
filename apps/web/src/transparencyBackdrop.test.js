// Transparency reads as a checkerboard, not as black (sc-24111).
//
// jsdom applies no stylesheet and computes no background, so this cannot be a rendering test. What
// it can be is a guard on the stylesheet itself, which is where the whole of this behaviour lives:
// before sc-24111 every image surface declared `background: var(--bg-2)`, which in the dark theme
// is very nearly black, so a natively transparent render was indistinguishable from a render on a
// black background. The assertions below are deliberately about the two things that would silently
// undo that — the checkerboard block disappearing, and a flat `background:` shorthand for one of
// these selectors reappearing *after* it (the shorthand resets `background-image`, so ordering is
// load-bearing, not cosmetic).

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const stylesPath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "./styles.css",
);
const styles = readFileSync(stylesPath, "utf8");

/** The byte offset of the checkerboard rule's `background-image`, for ordering assertions. */
const checkerIndex = styles.indexOf("--checker-tile:");

/** The rule's selector list: everything from its first selector up to its opening brace. */
const selectorList = styles.slice(
  styles.indexOf(".checker-backdrop,"),
  styles.indexOf("{", styles.indexOf(".checker-backdrop,")),
);

/** Every surface an RGBA asset can be painted on, per the sc-24111 hop map. */
const CHECKERED_SURFACES = [
  ".asset-tile img", // library grid + every asset grid
  ".review-card img", // Image Studio result view
  ".preview-button img", // result / asset-detail preview
  ".tray-item img",
  ".reference-media img",
  ".preview-modal img", // fullscreen preview
  ".worker-progress-card__thumb-media", // queue / job-output thumbnails
  ".ie-layer-thumb", // Image Editor layers panel
];

describe("the transparency backdrop", () => {
  it("exists exactly once, as a shared block", () => {
    expect(checkerIndex).toBeGreaterThan(-1);
    expect(styles.indexOf("--checker-tile:", checkerIndex + 1)).toBe(
      styles.lastIndexOf("--checker-tile:"),
    );
  });

  it("is a checkerboard and not a stripe", () => {
    const block = styles.slice(checkerIndex, styles.indexOf("}", checkerIndex));
    // Two 45-degree gradients offset by half a tile is the app's existing convention, taken from
    // `.ie-canvas`. One gradient would be diagonal stripes, which is the repo's *other* 45-degree
    // pattern (the empty-state hazard fill) and means something else entirely.
    expect(block.match(/linear-gradient\(\s*45deg/g)).toHaveLength(2);
    expect(block).toContain("background-position: 0 0, calc(var(--checker-tile) / 2)");
  });

  it.each(CHECKERED_SURFACES)("covers %s", (selector) => {
    expect(selectorList).toContain(selector);
  });

  it.each(CHECKERED_SURFACES)(
    "is not flattened again after the fact for %s",
    (selector) => {
      // A `background: <color>` shorthand resets `background-image`, so a later rule on the same
      // selector would silently put the flat fill back. Earlier ones are fine — they are what this
      // block overrides.
      const pattern = new RegExp(
        `${selector.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&")}[^{]*\\{[^}]*background:\\s*var\\(`,
        "g",
      );
      for (const match of styles.matchAll(pattern)) {
        expect(match.index).toBeLessThan(checkerIndex);
      }
    },
  );

  it("leaves <video> surfaces flat", () => {
    // No video format this app plays carries alpha, so a checkerboard behind one would be pure
    // decoration. The video rules keep their flat fill, and the checkerboard selector list must
    // not name one.
    expect(selectorList).not.toContain("video");
    expect(styles).toContain(".preview-button video");
  });

  it("keeps the Image Editor canvas checkerboarded", () => {
    // The third surface the story names, and the one that already worked — the sc-22461 color-key
    // and SAM3 cutout previews read against it. Asserted so a redesign of `.ie-canvas` cannot
    // quietly remove the thing the tool panel's copy promises the user.
    // There are two `.ie-canvas` rules (a one-line grid-area assignment and the painted one); the
    // checkerboard is in the multi-line block, which is the one with a newline after its brace.
    const canvasIndex = styles.indexOf(".ie-canvas {\n");
    expect(canvasIndex).toBeGreaterThan(-1);
    const block = styles.slice(canvasIndex, styles.indexOf("}", canvasIndex));
    expect(block.match(/linear-gradient\(45deg/g)).toHaveLength(2);
    expect(block).toContain("background-position: 0 0, 11px 11px");
  });

  it("sizes the fullscreen preview to its bitmap so the checker is not letterbox bars", () => {
    // `.preview-modal img` is the one `object-fit: contain` surface in the list. Contain is what
    // would put the checkerboard in the letterbox bars of every opaque image, so the element is
    // sized to the bitmap instead and there are no bars to fill.
    const modal = styles.lastIndexOf(".preview-modal img {");
    expect(modal).toBeGreaterThan(checkerIndex);
    const block = styles.slice(modal, styles.indexOf("}", modal));
    expect(block).toContain("width: auto");
    expect(block).toContain("max-width: 100%");
    expect(block).toContain("height: auto");
  });
});
