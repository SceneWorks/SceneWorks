import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleModelManager } from "./SimpleModelManager.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { MEMORY_HEADROOM_FRACTION, tierFits } from "../tierSuggestion.js";

// sc-15400: the Simple Model Manager row showed `needs ${mlx.minMemoryGb} GB` unconditionally. That
// blanket integer is a SINGLE per-model value, so it has to admit the model's heaviest INSTALLABLE tier
// — z_image declares 48 while its measured q4 tier peaks at 19.42 GiB, and krea_realtime_14b declares 64
// against a 27.90 GiB q4. The advanced Models screen already dodges this by SUPPRESSING the badge for
// variant-matrix models (`hasTierMatrix`); Simple has no per-tier panel to defer to, so it instead shows
// the number for the tier actually in play.
//
// Asserted through the REAL component and the REAL AppContext, so what's checked is the string the user
// reads in the row, not a helper's return value.
//
// TWO PROPERTIES THE NUMBER MUST HOLD, both asserted below against their SOURCE rather than a
// transcribed literal, because both failed silently under a green suite once already:
//   * HEADROOM — the displayed host size must satisfy the app's own `tierFits` budget. Quoting the raw
//     measured peak names a host the app itself will not run the tier on.
//   * LANE — the candle rows must never quote MLX-measured evidence, in either direction.

// Real catalog numbers from config/manifests/builtin.models.jsonc.
const Z_IMAGE_Q4_BYTES = 20852069456; // 19.42 GiB — z_image's ONLY measured tier
const LENS_TURBO_Q4_BYTES = 32749818036; // 30.50 GiB
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
// Note it declares NO candle block at all — which is what makes it the candle-lane silence case below.
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
    // The lane the rows quote. Defaults to the MLX lane because every measured `footprint` in the
    // catalog is an MLX measurement; the candle-lane cases set this to false explicitly.
    macCapabilities: { macGatingActive: true },
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

// The integer the row actually displays, so an assertion can be made about the NUMBER rather than the
// string. Returns null when the row quotes no memory figure.
function displayedGb(container) {
  const match = /(?:needs|from) (\d+) GB/.exec(metaText(container).join(" · "));
  return match ? Number(match[1]) : null;
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
    expect(metaText(container)).toEqual(["needs 22 GB"]);
    expect(container.textContent).not.toContain("48 GB");
  });

  it("shows the cheapest measured tier as a FLOOR when the model is not installed", async () => {
    // Nothing on disk, so the client cannot know which tier Download will fetch (the catalog omits the
    // manifest `default` flag). "from" states the entry cost without claiming a specific tier.
    await render(root, { models: [zImage()] });
    expect(metaText(container)).toEqual(["from 22 GB"]);
    expect(container.textContent).not.toContain("48 GB");
  });

  // THE BLOCKER-1 GUARD. Tied to `tierFits` — the app's own fit criterion — rather than to the string
  // "22 GB", so it cannot be satisfied by a number that merely looks plausible. Reverting the headroom
  // division makes the row quote ceil(19.42) = 20, and 20 × 0.9 = 18.0 GiB does NOT fit a 19.42 GiB
  // peak, so `fitsAtDisplayed` goes false.
  it("displays a host size that SATISFIES tierFits, and is tight one GB below", async () => {
    const model = zImage({ installState: "installed" });
    await render(root, { models: [model] });
    const shown = displayedGb(container);
    expect(shown).not.toBeNull();

    const q4 = model.variants.find((entry) => entry.variant === "q4");
    // The number the user reads must be a host the app would actually run this tier on...
    expect(tierFits(q4, shown), `displayed ${shown} GB must satisfy tierFits`).toBe(true);
    // ...and must not be inflated: one GB less must fail, so the row is the MINIMUM honest host size.
    expect(tierFits(q4, shown - 1), `${shown - 1} GB must not satisfy tierFits`).toBe(false);
    // Stated as the arithmetic the two share, so the intent survives a refactor of either side.
    expect(shown).toBe(Math.ceil(Z_IMAGE_Q4_BYTES / 1024 ** 3 / MEMORY_HEADROOM_FRACTION));
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
    expect(metaText(container)).toEqual(["needs 52 GB"]);

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
    expect(metaText(container)).toEqual(["needs 32 GB"]);
  });

  it("falls back to the blanket floor when an INSTALLED tier has no measurement", async () => {
    // q4 measured, q8 installed-but-unmeasured ⇒ the installed set has no known bound, so the curated
    // blanket 48 is the honest answer rather than q4's 22.
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
        // No floor declared on either lane ⇒ no memory text.
        {
          id: "chroma1_hd",
          name: "Chroma1 HD",
          type: "image",
          installState: "missing",
        },
      ],
    });
    // The floorless row renders the em-dash placeholder the component uses for empty meta.
    expect(metaText(container)).toEqual(["needs 12 GB", "—"]);
  });
});

// THE BLOCKER-2 GUARD (sc-15613's acceptance criterion: "a test pins that a CUDA host never consumes an
// MLX-measured footprint"). `footprint.peakMemoryBytes` comes from the MLX harness
// (crates/sceneworks-worker/src/footprint_measure.rs, sampling mlx_rs::memory::get_peak_memory) and
// describes Apple unified memory. A discrete CUDA card does not share system RAM, so that number does not
// describe the VRAM it must hold — the candle lane has its own measured corpus in `candle.vramGbByTier`.
//
// Crossing them UNDER-states, which is the direction that OOMs: lens_turbo measures 30.50 GiB on MLX and
// needs 37.3 GB of VRAM at the SAME q4 tier.
describe("SimpleModelManager memory label: lanes are never crossed", () => {
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

  // lens_turbo as shipped: a q4 download served on all three OSes that carries an MLX footprint, PLUS a
  // full candle VRAM ladder. The one entry where reading the wrong lane is silently plausible.
  function lensTurbo({ installState = "installed" } = {}) {
    return {
      id: "lens_turbo",
      name: "Lens Turbo",
      type: "image",
      installState,
      mlx: { minMemoryGb: 60 },
      candle: { minMemoryGb: 44, vramGbByTier: { q4: 37.3, q8: 42.0, bf16: 52.0 }, measured: true },
      hasVariantMatrix: true,
      variants: [
        variant("q4", installState === "installed" ? "installed" : "missing", LENS_TURBO_Q4_BYTES),
        variant("q8", "missing", null),
        variant("bf16", "missing", null),
      ],
    };
  }

  it("reads candle.vramGbByTier on a CUDA host, NOT the MLX footprint", async () => {
    const model = lensTurbo();
    // MLX lane: the measured 30.50 GiB footprint ⇒ 34.
    await render(root, { models: [model], macCapabilities: { macGatingActive: true } });
    expect(metaText(container)).toEqual(["needs 34 GB"]);

    // candle lane: the measured 37.3 GB VRAM row ⇒ 42. Strictly HIGHER than the MLX answer, which is
    // exactly why borrowing the MLX number would under-warn a Windows/Linux user.
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 42 GB"]);
    expect(container.textContent).not.toContain("34 GB");
  });

  it("uses the candle ladder for the uninstalled 'from' floor too", async () => {
    const model = lensTurbo({ installState: "missing" });
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    // Cheapest candle tier is q4 at 37.3 ⇒ 42; the MLX answer would have been 34.
    expect(metaText(container)).toEqual(["from 42 GB"]);
  });

  it("says NOTHING on a CUDA host with no candle evidence, rather than borrowing the MLX floor", async () => {
    // z_image declares mlx 48 + a measured MLX q4 footprint, and NO candle block. Pre-sc-15400 the row
    // showed "needs 48 GB" on Windows — an MLX unified-memory figure presented as a VRAM requirement.
    // Silence is the only honest answer: `qwen_image` declares mlx 50 against candle 56, so the MLX
    // blanket is not even reliably the conservative one and cannot be reused as a safe default.
    await render(root, {
      models: [zImage({ installState: "installed" })],
      macCapabilities: { macGatingActive: false },
    });
    expect(displayedGb(container)).toBeNull();
    expect(container.textContent).not.toContain("48 GB");
    expect(container.textContent).not.toContain("22 GB");
  });

  it("prefers the candle BLANKET when the lane has no per-tier row", async () => {
    // sensenova_u1_8b's shape: candle.minMemoryGb only, and no mlx floor at all.
    const model = {
      id: "sensenova_u1_8b",
      name: "SenseNova U1 8B",
      type: "image",
      installState: "installed",
      candle: { minMemoryGb: 16 },
      hasVariantMatrix: true,
      variants: [variant("q4", "installed", null)],
    };
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 16 GB"]);
    // ...and nothing on the MLX lane, which declares no floor for it.
    await render(root, { models: [model], macCapabilities: { macGatingActive: true } });
    expect(metaText(container)).toEqual(["—"]);
  });
});
