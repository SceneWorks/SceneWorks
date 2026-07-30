import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleModelManager } from "./SimpleModelManager.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { MEMORY_HEADROOM_FRACTION, tierFits } from "../tierSuggestion.js";

// sc-15400: the Simple Model Manager row showed `needs ${mlx.minMemoryGb} GB` unconditionally. That
// blanket integer OVER-states for a user who only installed a lighter tier — z_image declares 48 while
// its measured q4 tier peaks at 19.42 GiB, and krea_realtime_14b declares 64 against a 27.90 GiB q4. The
// advanced Models screen already dodges this by SUPPRESSING the badge for variant-matrix models
// (`hasTierMatrix`); Simple has no per-tier panel to defer to, so it instead shows the number for the
// tier actually in play.
//
// The blanket is NOT a ceiling over the tiers, though: the schema and `vram_gate.rs:180` both say it is
// the DEFAULT (lightest) tier's peak plus headroom, which heavier tiers may exceed. So it is a FLOOR the
// per-tier figure is maxed against, never a substitute for a heavier installed tier's requirement.
//
// Asserted through the REAL component and the REAL AppContext, so what's checked is the string the user
// reads in the row, not a helper's return value.
//
// THREE PROPERTIES THE NUMBER MUST HOLD, asserted below against their SOURCE rather than a transcribed
// literal, because the first two failed silently under a green suite once already:
//   * HEADROOM, PER LANE — MLX budgets `peak <= host × 0.9` (`tierFits`); candle's own gate admits at
//     `peak + 2` (`vram_gate.rs` HEADROOM_GB). Quoting the raw peak under-states; quoting the MLX form on
//     candle over-states against the gate that actually rejects.
//   * LANE — the candle rows must never quote MLX-measured evidence, in either direction.
//   * COMPOSITION — the fallback must not undercut an installed tier's own evidence.
//
// These are literal-driven by design: they pin the WORDING and the per-case wiring. The composition is
// swept exhaustively against the real manifest in memoryFloorCatalogParity.test.js, which drives this
// component's own `needsLabel` over every (model, OS, installed-subset) the catalog can produce.

// Real catalog numbers from config/manifests/builtin.models.jsonc.
const Z_IMAGE_Q4_BYTES = 20852069456; // 19.42 GiB — z_image's ONLY measured tier
const LENS_TURBO_Q4_BYTES = 32749818036; // 30.50 GiB
const KREA_RT_Q4_BYTES = 29957689344; // 27.90 GiB
const KREA_RT_Q8_BYTES = 36980832256; // 34.44 GiB
const KREA_RT_BF16_BYTES = 50154536960; // 46.71 GiB

// REAL `footprint.diskSizeBytes` values, because sc-15400 round 4 made them LOAD-BEARING: an installed
// tier with no measured peak is now filled from `variantFootprintBytes`, which reads exactly this field.
// These fixtures previously stubbed every disk size as `diskSizeBytes: 1`, which silently modelled every
// unmeasured tier as a 14 GiB (transient-only) requirement — a fixture understating what ships, in the
// one field the fill consumes.
const Z_IMAGE_DISK = { q4: 5907321677, q8: 10996438540, bf16: 20538412852 };
const KREA_RT_DISK = { q4: 20265258499, q8: 27290719452, bf16: 40463370856 };
const LENS_TURBO_DISK = { q4: 21830326407, q8: 32753474289, bf16: 30651441376 };
const FLUX_DEV_DISK = { q4: 9617334683, q8: 18010071290, bf16: 33746402173 };
const SENSENOVA_DISK = { q4: 11824472971, q8: 19927870461, bf16: 35121691427 };

// `diskSizeBytes` is REQUIRED (pass null only for an entry that genuinely ships no footprint at all, as
// krea_2_raw does) so no fixture can silently reintroduce a stub disk size.
function variant(key, installState, peakMemoryBytes, diskSizeBytes) {
  if (diskSizeBytes === undefined) {
    throw new Error(`fixture ${key}: pass the REAL diskSizeBytes, or null when the entry ships none`);
  }
  const footprint = {};
  if (diskSizeBytes !== null) {
    footprint.diskSizeBytes = diskSizeBytes;
  }
  if (peakMemoryBytes != null) {
    footprint.peakMemoryBytes = peakMemoryBytes;
  }
  return {
    variant: key,
    installState,
    footprint: Object.keys(footprint).length > 0 ? footprint : null,
  };
}

// z_image as shipped: blanket mlx floor 48, a q4/q8/bf16 matrix, real disk sizes on all three, and a
// measured footprint on q4 only.
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
        variant(
          "q4",
          installState === "installed" ? "installed" : "missing",
          Z_IMAGE_Q4_BYTES,
          Z_IMAGE_DISK.q4,
        ),
        variant("q8", "missing", null, Z_IMAGE_DISK.q8),
        variant("bf16", "missing", null, Z_IMAGE_DISK.bf16),
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
        variant("q4", "installed", KREA_RT_Q4_BYTES, KREA_RT_DISK.q4),
        variant("q8", "missing", KREA_RT_Q8_BYTES, KREA_RT_DISK.q8),
        variant("bf16", "installed", KREA_RT_BF16_BYTES, KREA_RT_DISK.bf16),
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
            variant("q4", "installed", KREA_RT_Q4_BYTES, KREA_RT_DISK.q4),
            variant("q8", "missing", KREA_RT_Q8_BYTES, KREA_RT_DISK.q8),
            variant("bf16", "missing", KREA_RT_BF16_BYTES, KREA_RT_DISK.bf16),
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
            variant("q4", "installed", Z_IMAGE_Q4_BYTES, Z_IMAGE_DISK.q4),
            variant("q8", "installed", null, Z_IMAGE_DISK.q8),
            variant("bf16", "missing", null, Z_IMAGE_DISK.bf16),
          ],
        }),
      ],
    });
    expect(metaText(container)).toEqual(["needs 48 GB"]);
  });

  // krea_2_raw AS SHIPPED, in full. An earlier version of this fixture claimed "as shipped" while omitting
  // the `candle` block the manifest really carries (`minMemoryGb: 32`, `vramGbByTier { q4: 28.8, q8: 33.4,
  // bf16: 46.4 }`, `measured: false`) — and that omission concealed the candle under-statement for two
  // review rounds, because with no candle block the row could only ever fall silent on that lane.
  //
  // It is also the one shipped matrix entry whose downloads carry NO `footprint` AT ALL — not even a
  // `diskSizeBytes` — which makes it the fixture for `installedFloorHostGb`'s "still unbounded after the
  // fill" path: there is nothing for `mlxEstimatedPeakGb` to read, so the blanket has to carry the answer.
  function krea2Raw() {
    return {
      id: "krea_2_raw",
      name: "Krea 2 Raw",
      type: "image",
      installState: "installed",
      mlx: { minMemoryGb: 48 },
      candle: {
        minMemoryGb: 32,
        vramGbByTier: { q4: 28.8, q8: 33.4, bf16: 46.4 },
        measured: false,
      },
      hasVariantMatrix: true,
      variants: [
        variant("q4", "installed", null, null),
        variant("q8", "installed", null, null),
        variant("bf16", "missing", null, null),
      ],
    };
  }

  it("keeps the blanket floor for a model with no measured tier and no estimable size", async () => {
    // MLX lane: no measured peak, and no `diskSizeBytes` for the fill to fall back on, so the install set
    // stays genuinely unbounded and the curated blanket 48 is the honest answer.
    await render(root, { models: [krea2Raw()] });
    expect(metaText(container)).toEqual(["needs 48 GB"]);
  });

  it("reads krea_2_raw's shipped candle ladder on the candle lane", async () => {
    // The lane the omitted block hid. q4+q8 installed ⇒ the ceiling is the q8 row 33.4, which the candle
    // gate's additive criterion turns into ceil(33.4 + 2) = 36 — ABOVE the blanket 32, so the blanket
    // cannot stand in for it. `measured: false` means neither number is a measurement, hence the max.
    await render(root, {
      models: [krea2Raw()],
      macCapabilities: { macGatingActive: false },
    });
    expect(metaText(container)).toEqual(["needs 36 GB"]);
  });

  it("leaves a single-variant model and a floorless model unchanged", async () => {
    // BOTH fixtures are the SHIPPED shapes. The first previously read `id: "sana_sprint"` with a blanket of
    // 12 — an id that is NOT IN THE MANIFEST at all (the real entries are `sana_sprint_1600m` / `sana_1600m`,
    // and both ship a q4/q8/bf16 matrix on macOS, so neither is a single-variant model). A fixture naming a
    // model that does not exist cannot be checked against anything.
    await render(root, {
      models: [
        // flux2_klein_9b_true_v2 as shipped: ONE download with no `variant` key, so the "default"
        // pseudo-tier is not a quant tier, `installedTiers` is empty, and the blanket floor stands. It also
        // ships no `footprint`, so there is nothing for the fill to read even in principle.
        {
          id: "flux2_klein_9b_true_v2",
          name: "FLUX.2 Klein 9B (true v2)",
          type: "image",
          installState: "installed",
          mlx: { minMemoryGb: 42 },
          hasVariantMatrix: false,
          variants: [variant("default", "installed", null, null)],
        },
        // chroma1_hd as shipped: a q4/q8/bf16 matrix with real disk sizes, NO measured peak on any tier,
        // and no `minMemoryGb` on either lane (`mlx` carries only `quantize`/`standardTierLayout`; there is
        // no `candle` block). Nothing installed, so this is the "from" branch — which reads MEASURED
        // per-tier evidence only and is deliberately NOT filled from disk, because an uninstalled model has
        // no install set to bound. Hence silence.
        {
          id: "chroma1_hd",
          name: "Chroma1 HD",
          type: "image",
          installState: "missing",
          hasVariantMatrix: true,
          variants: [
            variant("q4", "missing", null, 15121207891),
            variant("q8", "missing", null, 19424564251),
            variant("bf16", "missing", null, 27493363692),
          ],
        },
      ],
    });
    // The floorless row renders the em-dash placeholder the component uses for empty meta.
    expect(metaText(container)).toEqual(["needs 42 GB", "—"]);
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
      // `measured: false` is what the manifest ACTUALLY ships for this entry. An earlier version of this
      // fixture said `true` and the PR body called 37.3 "a measured VRAM requirement"; per the schema the
      // flag means "false when ESTIMATED (scaled from weight size + a measured sibling)", and it covers
      // `minMemoryGb` and `vramGbByTier` together. So neither number here is a measurement, and the row
      // reports the higher of the two (see the estimated-lane rule in tierSuggestion.js).
      candle: { minMemoryGb: 44, vramGbByTier: { q4: 37.3, q8: 42.0, bf16: 52.0 }, measured: false },
      hasVariantMatrix: true,
      variants: [
        variant(
          "q4",
          installState === "installed" ? "installed" : "missing",
          LENS_TURBO_Q4_BYTES,
          LENS_TURBO_DISK.q4,
        ),
        variant("q8", "missing", null, LENS_TURBO_DISK.q8),
        variant("bf16", "missing", null, LENS_TURBO_DISK.bf16),
      ],
    };
  }

  it("reads candle.vramGbByTier on a CUDA host, NOT the MLX footprint", async () => {
    const model = lensTurbo();
    // MLX lane: the measured 30.50 GiB footprint ⇒ ceil(30.50 / 0.9) = 34.
    await render(root, { models: [model], macCapabilities: { macGatingActive: true } });
    expect(metaText(container)).toEqual(["needs 34 GB"]);

    // candle lane: the estimated q4 row of 37.3 needs ceil(37.3 + 2) = 40 by the candle gate's own
    // additive criterion, which the curated (also estimated) blanket of 44 floors ⇒ 44. Strictly HIGHER
    // than the MLX answer either way, which is why borrowing the MLX number would under-warn here.
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 44 GB"]);
    expect(container.textContent).not.toContain("34 GB");
  });

  it("uses the candle ladder for the uninstalled 'from' floor too", async () => {
    const model = lensTurbo({ installState: "missing" });
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    // Cheapest candle tier is q4 at 37.3 ⇒ 40, floored at the estimated-lane blanket 44; MLX said 34.
    expect(metaText(container)).toEqual(["from 44 GB"]);
  });

  it("floors the label at the blanket when an installed tier has NO candle row", async () => {
    // BLOCKER 1's shape, as `flux_dev` ships it: measured q4/q8 rows, a bf16 tier with no row at all, and
    // a blanket of 24 that is the LIGHTEST tier's peak plus headroom (schema `candle.minMemoryGb`;
    // vram_gate.rs:180 "minMemoryGb is the WRONG floor"). Falling back to the blanket ALONE reported 24
    // against a measured q8 of 31.8 — an under-statement. The answer is the max of the two bases.
    const model = {
      id: "flux_dev",
      name: "FLUX.1 dev",
      type: "image",
      installState: "installed",
      candle: { minMemoryGb: 24, vramGbByTier: { q4: 21.3, q8: 31.8 }, measured: true },
      hasVariantMatrix: true,
      variants: [
        variant("q4", "installed", null, FLUX_DEV_DISK.q4),
        variant("q8", "installed", null, FLUX_DEV_DISK.q8),
        variant("bf16", "installed", null, FLUX_DEV_DISK.bf16),
      ],
    };
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    // ceil(31.8 + 2) = 34, not the blanket 24.
    expect(metaText(container)).toEqual(["needs 34 GB"]);
    expect(container.textContent).not.toContain("24 GB");
  });

  it("ignores a candle-only tier the host cannot serve", async () => {
    // MINOR 5. `krea_2_turbo`'s shape: an `int8-convrot` download WITH a measured row. The tier is only
    // offered when a live worker advertises `int8_convrot` (sc-9300), so on a host where none does, its
    // 34.7 row must not set this row's number — the user will never be able to select it.
    const model = {
      id: "krea_2_turbo",
      name: "Krea 2 Turbo",
      type: "image",
      installState: "installed",
      candle: {
        minMemoryGb: 32,
        vramGbByTier: { q4: 25.7, q8: 35.2, bf16: 47.2, "int8-convrot": 34.7 },
        measured: true,
      },
      hasVariantMatrix: true,
      variants: [variant("int8-convrot", "installed", null, null)],
    };

    // No worker advertises the capability ⇒ the tier is invisible, so the row falls back to the blanket.
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 32 GB"]);

    // A worker that DOES advertise it ⇒ the tier's own measured row, ceil(34.7 + 2) = 37. This is the
    // number the pre-fix code could not produce at all: keying the candle map off `tierQuantize` dropped
    // the row, so the whole install set collapsed to the blanket 32.
    await render(root, {
      models: [model],
      macCapabilities: { macGatingActive: false },
      visibleWorkers: [{ status: "online", capabilities: ["int8_convrot"] }],
    });
    expect(metaText(container)).toEqual(["needs 37 GB"]);
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

  it("prefers the candle PER-TIER row over the blanket, as sensenova_u1_8b really ships", async () => {
    // sensenova_u1_8b AS SHIPPED. An earlier version of this fixture declared `candle: { minMemoryGb: 16 }`
    // and called that "sensenova_u1_8b's shape: candle.minMemoryGb only" — FALSE. The manifest ships a full
    // `vramGbByTier { q4: 14.8, q8: 22.5, bf16: 36.7 }` with `measured: true`, under which a q4-only install
    // reports 17, not 16. The fixture asserted the number its own invented shape produced.
    const model = {
      id: "sensenova_u1_8b",
      name: "SenseNova U1 8B",
      type: "image",
      installState: "installed",
      // No `mlx.minMemoryGb`: the real `mlx` block carries only `minQualityTier`/`quantize`/
      // `standardTierLayout`.
      mlx: { minQualityTier: "q8", quantize: 4, standardTierLayout: true },
      candle: {
        minMemoryGb: 16,
        vramGbByTier: { q4: 14.8, q8: 22.5, bf16: 36.7 },
        measured: true,
      },
      hasVariantMatrix: true,
      variants: [
        variant("q4", "installed", null, SENSENOVA_DISK.q4),
        variant("q8", "missing", null, SENSENOVA_DISK.q8),
        variant("bf16", "missing", null, SENSENOVA_DISK.bf16),
      ],
    };
    // candle: the q4 row 14.8 by the gate's additive criterion ⇒ ceil(14.8 + 2) = 17. The lane is
    // `measured: true` and coverage over the install set is complete, so the row is used ALONE — it lands
    // above the blanket 16 here, but the point is that the per-tier row decides, not the blanket.
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 17 GB"]);

    // MLX: this entry declares NO mlx floor and NO measured footprint on any tier, so the q4 install is
    // bounded by the disk-based fill alone — 11824472971 B + 14 GiB transient = 25.02 GiB ⇒
    // ceil(25.02 / 0.9) = 28. Before round 4 the row said NOTHING here, while `tierFits`/`suggestTier` were
    // already ranking this tier against exactly that estimate; silence was not the conservative option, it
    // just left the picker free to refuse a tier the row had implied nothing about.
    await render(root, { models: [model], macCapabilities: { macGatingActive: true } });
    expect(metaText(container)).toEqual(["needs 28 GB"]);
    // Pinned to the app's own criterion rather than to the literal 28, so it cannot drift from the picker.
    const q4 = model.variants[0];
    expect(tierFits(q4, displayedGb(container))).toBe(true);
    expect(tierFits(q4, displayedGb(container) - 1)).toBe(false);
  });

  it("uses the candle BLANKET when the lane really has no per-tier row (kokoro_82m)", async () => {
    // The blanket-only shape the previous fixture MISNAMED. It genuinely exists — on eight shipped audio
    // entries, all `candle: { minMemoryGb, measured: false }` with no `vramGbByTier` — `kokoro_82m` being
    // the smallest. Single download, no `variant` key, so `installedTiers` is empty and case 3 of
    // `needsLabel` (the blanket) is the only branch available. Rendered as `type: "image"` purely to land it
    // on the default tab; the memory logic does not read `type`.
    const model = {
      id: "kokoro_82m",
      name: "Kokoro 82M",
      type: "image",
      installState: "installed",
      candle: { minMemoryGb: 2, measured: false },
      hasVariantMatrix: false,
      variants: [variant("default", "installed", null, 327214577)],
    };
    await render(root, { models: [model], macCapabilities: { macGatingActive: false } });
    expect(metaText(container)).toEqual(["needs 2 GB"]);
    // ...and nothing on the MLX lane, which declares no block at all for it. The fill cannot rescue this:
    // `installedTiers` is empty, so there is no installed tier to estimate for.
    await render(root, { models: [model], macCapabilities: { macGatingActive: true } });
    expect(metaText(container)).toEqual(["—"]);
  });
});
