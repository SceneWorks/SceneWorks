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
// That drift is not cosmetic here. sc-12294 retired the A14B/LTX `1280x720` buckets in favour of
// `1280x704`, because 720 is not a multiple of the models' 32-px dimension stride and the 901120 area cap
// floors it: the engine advertised 720 and rendered 704. A stale mirror reintroduces exactly that lie —
// the picker advertises a bucket the engine will not render. The manifest-side test added by sc-12294 is
// manifest-only and structurally cannot see this file, so this test is the guard for the mirror.
//
// Precedent for the drift being real, not theoretical: sc-4997 set wan_2_2's fast-path default to
// 832x480 in constants.js ONLY and never touched the manifest, so that default never actually shipped —
// the catalog kept serving 720p. This test would have caught it. (Whether the manifest should adopt
// sc-4997's 832x480 fast path is a separate product call, tracked in sc-12319.)
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
    },
  );

  it("no video entry advertises a bucket the area cap would floor (sc-12294)", () => {
    // The direct regression guard, independent of the manifest: every advertised bucket must survive the
    // model's own maxPixels cap at its stride. This is what makes reintroducing `1280x720` fail loudly
    // even if someone "fixes" the manifest to match a stale mirror rather than the other way round.
    const STRIDE = 32;
    for (const fallback of fallbackVideo) {
      const manifest = manifestById.get(fallback.id);
      const maxPixels = manifest?.limits?.maxPixels;
      if (!maxPixels) {
        continue; // Uncapped model (ltx/svd/mochi) — nothing to floor against.
      }
      for (const bucket of fallback.limits?.resolutions ?? []) {
        const [w, h] = bucket.split("x").map(Number);
        expect(
          w % STRIDE === 0 && h % STRIDE === 0,
          `${fallback.id} advertises ${bucket}, which is not a multiple of the ${STRIDE}px stride — the engine cannot render it as advertised`,
        ).toBe(true);
        expect(
          w * h <= maxPixels,
          `${fallback.id} advertises ${bucket} (${w * h}px), which exceeds its ${maxPixels}px cap and would be floored`,
        ).toBe(true);
      }
    }
  });

  it("every manifest video model present in fallbackModels mirrors its geometry", () => {
    // Reverse guard: a manifest video model the mirror DOES carry must not diverge. Models absent from
    // the seed list (bernini/scail2_14b/mochi_1 load from the live catalog) are intentionally skipped.
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
