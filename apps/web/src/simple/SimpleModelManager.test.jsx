import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleModelManager } from "./SimpleModelManager.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

// sc-15400: the Simple Model Manager row showed `needs ${mlx.minMemoryGb} GB` unconditionally. That
// blanket integer is a SINGLE per-model value, so it has to admit the model's heaviest INSTALLABLE tier
// — z_image declares 48 while its measured q4 tier peaks at 19.42 GiB, and krea_realtime_14b declares 64
// against a 27.90 GiB q4. The advanced Models screen already dodges this by SUPPRESSING the badge for
// variant-matrix models (`hasTierMatrix`); Simple has no per-tier panel to defer to, so it instead shows
// the number for the tier actually in play.
//
// Asserted through the REAL component and the REAL AppContext, so what's checked is the string the user
// reads in the row, not a helper's return value.

// Real catalog numbers from config/manifests/builtin.models.jsonc.
const Z_IMAGE_Q4_BYTES = 20854139600; // 19.42 GiB — z_image's ONLY measured tier
const KREA_RT_Q4_BYTES = 29957689344; // 27.90 GiB
const KREA_RT_Q8_BYTES = 36980832256; // 34.44 GiB
const KREA_RT_BF16_BYTES = 50154536960; // 46.71 GiB

function variant(key, installState, peakMemoryBytes) {
  return {
    variant: key,
    installState,
    footprint: peakMemoryBytes == null ? { diskSizeBytes: 1 } : { diskSizeBytes: 1, peakMemoryBytes },
  };
}

// z_image as shipped: blanket mlx floor 48, a q4/q8/bf16 matrix, and a measured footprint on q4 only.
function zImage({ installState = "missing", variants } = {}) {
  return {
    id: "z_image",
    name: "Z-Image",
    type: "image",
    family: "z-image",
    installState,
    mlx: { minMemoryGb: 48 },
    hasVariantMatrix: true,
    variants:
      variants ??
      [
        variant("q4", installState === "installed" ? "installed" : "missing", Z_IMAGE_Q4_BYTES),
        variant("q8", "missing", null),
        variant("bf16", "missing", null),
      ],
  };
}

function baseContext(overrides = {}) {
  return {
    models: [],
    loras: [],
    jobs: [],
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    ...overrides,
  };
}

const simpleUi = {
  breakpoint: "desktop",
  toast: vi.fn(),
  openInAdvanced: vi.fn(),
};

async function render(root, context) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={baseContext(context)}>
        <SimpleUiContext.Provider value={simpleUi}>
          <SimpleModelManager />
        </SimpleUiContext.Provider>
      </AppContext.Provider>,
    );
  });
}

// The row's "size · needs N GB" meta line.
function metaText(container) {
  return [...container.querySelectorAll(".su-row-meta")].map((node) => node.textContent);
}

describe("SimpleModelManager memory label", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  it("shows the INSTALLED tier's measured peak, not the blanket floor", async () => {
    // q4 installed and measured at 19.42 GiB. The blanket floor is 48 — the number the row used to show.
    await render(root, { models: [zImage({ installState: "installed" })] });
    expect(metaText(container)).toEqual(["needs 20 GB"]);
    expect(container.textContent).not.toContain("48 GB");
  });

  it("shows the cheapest measured tier as a FLOOR when the model is not installed", async () => {
    // Nothing on disk, so the client cannot know which tier Download will fetch (the catalog omits the
    // manifest `default` flag). "from" states the entry cost without claiming a specific tier.
    await render(root, { models: [zImage()] });
    expect(metaText(container)).toEqual(["from 20 GB"]);
    expect(container.textContent).not.toContain("48 GB");
  });

  it("budgets the CEILING over installed tiers when more than one is on disk", async () => {
    // Krea's full measured ladder: q4 27.90 / q8 34.44 / bf16 46.71 GiB. With q4 AND bf16 installed the
    // row must quote the heaviest thing the user can actually run, not the lightest.
    const model = {
      id: "krea_realtime_14b",
      name: "Krea Realtime 14B",
      type: "video",
      installState: "installed",
      mlx: { minMemoryGb: 64 },
      hasVariantMatrix: true,
      variants: [
        variant("q4", "installed", KREA_RT_Q4_BYTES),
        variant("q8", "missing", KREA_RT_Q8_BYTES),
        variant("bf16", "installed", KREA_RT_BF16_BYTES),
      ],
    };
    await render(root, { models: [model] });
    // Video models live on the Video tab; the Image tab is empty.
    expect(metaText(container)).toEqual([]);
    await render(root, { models: [{ ...model, type: "image" }] });
    expect(metaText(container)).toEqual(["needs 47 GB"]);

    // Only q4 installed ⇒ 27.90 GiB governs. Same model, same ladder, different install state.
    await render(root, {
      models: [
        {
          ...model,
          type: "image",
          variants: [
            variant("q4", "installed", KREA_RT_Q4_BYTES),
            variant("q8", "missing", KREA_RT_Q8_BYTES),
            variant("bf16", "missing", KREA_RT_BF16_BYTES),
          ],
        },
      ],
    });
    expect(metaText(container)).toEqual(["needs 28 GB"]);
  });

  it("falls back to the blanket floor when an INSTALLED tier has no measurement", async () => {
    // q4 measured, q8 installed-but-unmeasured ⇒ the installed set has no known bound, so the curated
    // blanket 48 is the honest answer rather than q4's 20.
    await render(root, {
      models: [
        zImage({
          installState: "installed",
          variants: [
            variant("q4", "installed", Z_IMAGE_Q4_BYTES),
            variant("q8", "installed", null),
            variant("bf16", "missing", null),
          ],
        }),
      ],
    });
    expect(metaText(container)).toEqual(["needs 48 GB"]);
  });

  it("keeps the blanket floor for a model with no measured tier at all", async () => {
    // krea_2_raw as shipped: a matrix, a 48 GB floor, and not one measured footprint. Byte-identical to
    // the pre-fix label — this is what makes the assertions above about the FOOTPRINT and not the shape.
    await render(root, {
      models: [
        {
          id: "krea_2_raw",
          name: "Krea 2 Raw",
          type: "image",
          installState: "installed",
          mlx: { minMemoryGb: 48 },
          hasVariantMatrix: true,
          variants: [
            variant("q4", "installed", null),
            variant("q8", "installed", null),
            variant("bf16", "missing", null),
          ],
        },
      ],
    });
    expect(metaText(container)).toEqual(["needs 48 GB"]);
  });

  it("leaves a single-variant model and a floorless model unchanged", async () => {
    await render(root, {
      models: [
        // Single-variant: the "default" pseudo-tier is not a quant tier, so the blanket floor stands.
        {
          id: "sana_sprint",
          name: "SANA Sprint",
          type: "image",
          installState: "installed",
          mlx: { minMemoryGb: 12 },
          hasVariantMatrix: false,
          variants: [variant("default", "installed", 999)],
        },
        // No floor declared at all ⇒ no memory text (SenseNova's shape).
        {
          id: "sensenova_u1_8b",
          name: "SenseNova U1 8B",
          type: "image",
          installState: "missing",
        },
      ],
    });
    // The floorless row renders the em-dash placeholder the component uses for empty meta.
    expect(metaText(container)).toEqual(["needs 12 GB", "—"]);
  });
});
