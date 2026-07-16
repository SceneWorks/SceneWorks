import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";

// Manifest ↔ constants parity for VIDEO geometry (sc-12294), a sibling of controlModesParity.test.js.
//
// The Video Studio picker offers `limits.resolutions` and preselects `defaults.resolution`. At runtime
// those come from the live catalog — seeded from `config/manifests/builtin.models.jsonc` — but
// `apps/web/src/constants.js` (`fallbackModels`) carries a HAND-MIRRORED copy that App.jsx serves to the
// real picker whenever the catalog hasn't loaded yet. Nothing stops that mirror from silently drifting.
//
// That drift is not cosmetic here: a stale mirror makes the picker advertise a bucket the engine will
// not render. The manifest-side test added by sc-12294 is manifest-only and structurally cannot see this
// file, so this test is the guard for the mirror — and it earned its keep on sc-12308, catching this
// file when the manifest moved.
//
// The A14B buckets have now moved TWICE, in opposite directions, which is why the mirror needs a guard
// rather than a convention:
//   - sc-12294 retired their `1280x720` for `1280x704`, believing 720 was blocked by a 32-px stride and
//     the 901120 area cap.
//   - sc-12308 restored `1280x720`: the A14B stride is **16** (720 = 45·16 is on-lattice — sc-12294
//     itself corrected that), and the 901120 cap was the TI2V-5B's budget wrongly applied to the 14B
//     family, whose real cap is 921600 (`1280*720`, upstream's own `MAX_AREA_CONFIGS`). So 704 was never
//     an A14B geometry at all; it is the 5B's.
// The ÷32 models — `wan_2_2` (TI2V-5B) and LTX — keep their genuine `1280x704`.
//
// Precedent for the drift being real, not theoretical: sc-4997 set wan_2_2's fast-path default to
// 832x480 in constants.js ONLY and never touched the manifest, so that default never actually shipped —
// the catalog kept serving 720p for a year. This test would have caught it. sc-12319 settled the
// product question the other way round (the manifest adopted 832x480), so both files now say 480p —
// but the guard is what keeps them saying it together.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");

function loadManifestModels() {
  const raw = readFileSync(MANIFEST_PATH, "utf8");
  const parsed = JSON5.parse(raw);
  const models = Array.isArray(parsed) ? parsed : parsed.models;
  expect(Array.isArray(models), "manifest must expose a models array").toBe(true);
  return models;
}

describe("video geometry ↔ manifest parity (sc-12294)", () => {
  const manifestModels = loadManifestModels();
  const manifestById = new Map(manifestModels.map((model) => [model.id, model]));
  const fallbackVideo = fallbackModels.filter((model) => model.type === "video");

  // The dimension stride the ENGINE actually applies to a model: its declared
  // `limits.requiresDimensionsMultipleOf`, else the blanket 32. This mirrors sceneworks-core's
  // `dimension_multiple_of` (video_request.rs) exactly, including its validity filter — a declared
  // multiple is only honored when it is positive and divides 256, because `floor_to_multiple` clamps up
  // to a hard floor of 256 and that rescue is only on-lattice when the multiple divides 256. Anything
  // else falls back to `DEFAULT_DIMENSION_MULTIPLE`, so the test agrees with the worker on typo'd input
  // instead of inventing its own rule.
  const DEFAULT_STRIDE = 32;
  const strideFor = (manifest) => {
    const declared = manifest?.limits?.requiresDimensionsMultipleOf;
    return Number.isInteger(declared) && declared > 0 && 256 % declared === 0 ? declared : DEFAULT_STRIDE;
  };

  // A row's stride declaration is LOAD-BEARING exactly when the row advertises a bucket the default
  // 32-px stride would not explain: that declaration is the only thing that makes `848x480` a legal
  // bucket rather than a typo. Rows whose buckets are all on the ÷32 lattice are already explained by
  // the default, so a stride there is a nicety and stays opt-in.
  const advertisesOffDefaultLattice = (fallback) =>
    (fallback.limits?.resolutions ?? []).some((bucket) => {
      const [w, h] = bucket.split("x").map(Number);
      return w % DEFAULT_STRIDE !== 0 || h % DEFAULT_STRIDE !== 0;
    });

  it("fallbackModels carries the video entries the picker falls back to", () => {
    // App.jsx:823 serves exactly this set to the Video Studio picker before the catalog loads, so an
    // empty/!video-typed list would silently make the rest of this suite vacuous.
    expect(fallbackVideo.length).toBeGreaterThan(0);
  });

  it.each(fallbackVideo.map((model) => [model.id, model]))(
    "constants.js %s mirrors the manifest resolutions + default resolution",
    (id, fallback) => {
      const manifest = manifestById.get(id);
      expect(manifest, `${id} must exist in the manifest`).toBeTruthy();
      // Order matters — the picker renders the buckets in this order.
      expect(fallback.limits?.resolutions, `${id} limits.resolutions must mirror the manifest`).toEqual(
        manifest.limits?.resolutions,
      );
      expect(
        fallback.defaults?.resolution,
        `${id} defaults.resolution must mirror the manifest`,
      ).toEqual(manifest.defaults?.resolution);
      // Where the stride is load-bearing, pin its PRESENCE — not just its value. Pinning only the
      // value is how this check first shipped, gated on `if (requiresDimensionsMultipleOf !== undefined)`,
      // which made it a guard whose trigger was the very field it asserted: DELETING mochi_1's
      // `requiresDimensionsMultipleOf: 16` passed the whole suite. The pair could drift apart by
      // deletion — the same defect class this file fixes in the bucket guard below, where a `maxPixels`
      // trigger was gating an orthogonal stride assertion.
      if (advertisesOffDefaultLattice(fallback)) {
        expect(
          fallback.limits?.requiresDimensionsMultipleOf,
          `${id} advertises a bucket that is not a multiple of ${DEFAULT_STRIDE}px, so its mirror must declare the stride that makes that bucket legal`,
        ).toBeDefined();
      }
      // A mirror that states a stride must state the RIGHT one. The web never reads this field (the
      // worker resolves it from the live catalog), so nothing else would notice it going stale — but a
      // wrong stride sitting next to the buckets is worse than none: it is the note a future reader
      // trusts when deciding whether a new bucket is legal.
      if (fallback.limits?.requiresDimensionsMultipleOf !== undefined) {
        expect(
          fallback.limits.requiresDimensionsMultipleOf,
          `${id} limits.requiresDimensionsMultipleOf must mirror the manifest`,
        ).toEqual(manifest.limits?.requiresDimensionsMultipleOf);
      }
    },
  );

  it("no video entry advertises a bucket its engine would floor (sc-12294, stride fixed in sc-11994)", () => {
    // The direct regression guard: every advertised bucket must be renderable AS ADVERTISED — on the
    // model's own stride, and (where the engine caps area) inside its own maxPixels. A bucket that fails
    // either is a lie the picker tells: the engine floors it and renders something else.
    //
    // sc-11994 fixed two defects in this guard, both found while adding Mochi's ÷16 row:
    //
    //   1. The stride was hardcoded to 32. That is only the DEFAULT (`DEFAULT_DIMENSION_MULTIPLE`);
    //      the real floor is per-model, and sc-12294 itself established the floor "is not one number".
    //      Hardcoding 32 gets ÷16 models BACKWARDS — it would fail `bernini`'s correct native 848x480
    //      (`848 % 32 == 16`, and bernini declares 16 + carries a cap) the moment the mirror carried it,
    //      flagging a truthful row as broken. `mochi_1` advertises the same 848x480 bucket.
    //   2. It `continue`d past every UNCAPPED model, so ltx_2_3 / ltx_2_3_eros / svd / mochi_1 got no
    //      stride check at all — the cap was doing double duty as the trigger for an orthogonal
    //      assertion. Verified before the fix: putting `1280x720` back on ltx_2_3 (stride 64, uncapped)
    //      in BOTH the manifest and the mirror still passed this suite, even though that is precisely
    //      the regression sc-12294 exists to prevent (720 % 64 == 16 → the engine renders 704).
    //
    // Reading the stride per-model fixes both: the check now runs for every video entry, and each is
    // measured against the stride its own engine uses.
    for (const fallback of fallbackVideo) {
      const manifest = manifestById.get(fallback.id);
      const stride = strideFor(manifest);
      const maxPixels = manifest?.limits?.maxPixels;
      for (const bucket of fallback.limits?.resolutions ?? []) {
        const [w, h] = bucket.split("x").map(Number);
        expect(
          w % stride === 0 && h % stride === 0,
          `${fallback.id} advertises ${bucket}, which is not a multiple of its ${stride}px stride — the engine cannot render it as advertised`,
        ).toBe(true);
        if (maxPixels) {
          expect(
            w * h <= maxPixels,
            `${fallback.id} advertises ${bucket} (${w * h}px), which exceeds its ${maxPixels}px cap and would be floored`,
          ).toBe(true);
        }
      }
    }
  });

  it("every manifest video model present in fallbackModels mirrors its geometry", () => {
    // Reverse guard: a manifest video model the mirror DOES carry must not diverge. Models absent from
    // the seed list (bernini/scail2_14b load from the live catalog) are intentionally skipped.
    // `mochi_1` JOINED the seed list in sc-11994, so it is now covered by this guard rather than skipped.
    for (const manifest of manifestModels.filter((model) => model.type === "video")) {
      const fallback = fallbackVideo.find((model) => model.id === manifest.id);
      if (!fallback) {
        continue;
      }
      expect(fallback.limits?.resolutions, `${manifest.id} resolutions must be mirrored`).toEqual(
        manifest.limits?.resolutions,
      );
      expect(fallback.defaults?.resolution, `${manifest.id} default resolution must be mirrored`).toEqual(
        manifest.defaults?.resolution,
      );
    }
  });
});
