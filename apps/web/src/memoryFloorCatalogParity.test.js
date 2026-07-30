import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import {
  MEMORY_HEADROOM_FRACTION,
  blanketFloorGb,
  cheapestDeclaredTierPeakGb,
  hostGbForPeakGb,
  installedTierPeakGb,
  tierFits,
} from "./tierSuggestion.js";
import { tierQuantize } from "./quantTier.js";

// CATALOG-DRIVEN memory-floor guard (sc-15400 review). Sibling of controlModesParity.test.js /
// videoGeometryParity.test.js, which established the pattern of asserting against the REAL
// `config/manifests/builtin.models.jsonc` rather than a synthetic literal.
//
// WHY THIS FILE EXISTS. sc-15400 shipped two defects behind a fully green suite of 14 tests, because
// every one of them used a hand-written model literal:
//
//   1. The row displayed the RAW measured peak (`Math.ceil(peak)`), which does not satisfy the app's own
//      `peak <= host * MEMORY_HEADROOM_FRACTION` fit criterion. lens_turbo's 30.50 GiB q4 read
//      "needs 31 GB", and 31 x 0.9 = 27.9 GiB does not fit it. Four shipped models under-stated.
//   2. The MLX-measured footprint was read on the candle/CUDA lane. lens_turbo's q4 download serves
//      windows/linux and carries an MLX footprint, so a CUDA host read "needs 31 GB" against a measured
//      37.3 GB VRAM requirement — converting a pre-existing over-warn into an UNDER-warn.
//
// A synthetic literal cannot catch either, because the author picks both the input and the expected
// output. So every assertion below derives its expectation from the manifest and from the app's own
// constants — never from a transcribed number — and the suite fails if it ever covers zero models.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const BYTES_PER_GB = 1024 ** 3;

function loadManifestModels() {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  const models = Array.isArray(parsed) ? parsed : parsed.models;
  expect(Array.isArray(models), "manifest must expose a models array").toBe(true);
  return models;
}

// Project a manifest entry into the CATALOG shape the web client actually receives, for the memory
// fields only. Mirrors apps/rust-api/src/models.rs:
//   * `retain_downloads_for_os` — per-OS `platforms` filtering, and its early return when NO entry
//     declares `platforms` (in which case nothing is filtered).
//   * `is_co_requisite_download` — `coRequisite: true` rows are dependencies, never selectable tiers.
//   * `model_variant_downloads` — duplicate `variant` keys collapse FIRST-wins.
//   * `apply_variant_fields` — emits `variants[]` carrying the raw `footprint`, plus `hasVariantMatrix`.
// The rest of the entry passes through unchanged (the catalog decorates the manifest object in place),
// which is what makes `mlx` / `candle` — including `candle.vramGbByTier` — reach the client at all.
//
// `installedTiers` is the set to mark installed, so a caller can ask "what would this host be told if
// exactly these tiers were on disk".
function catalogEntry(manifestModel, os, installedTierKeys) {
  let downloads = manifestModel.downloads ?? [];
  if (downloads.some((entry) => entry.platforms !== undefined)) {
    downloads = downloads.filter(
      (entry) => !Array.isArray(entry.platforms) || entry.platforms.includes(os),
    );
  }
  const selectable = downloads.filter((entry) => entry.coRequisite !== true);
  const seen = new Set();
  const variants = [];
  for (const entry of selectable) {
    const key =
      typeof entry.variant === "string" && entry.variant.trim() ? entry.variant.trim() : "default";
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const installed = installedTierKeys.includes(key);
    variants.push({
      variant: key,
      installed,
      installState: installed ? "installed" : "missing",
      footprint: entry.footprint ?? null,
    });
  }
  return {
    ...manifestModel,
    hasVariantMatrix: selectable.some(
      (entry) => typeof entry.variant === "string" && entry.variant.trim() !== "",
    ),
    variants,
  };
}

// Every real quant tier a manifest entry offers on `os`, after the same filters.
function tiersOn(manifestModel, os) {
  const entry = catalogEntry(manifestModel, os, []);
  return entry.variants.map((v) => v.variant).filter((key) => tierQuantize(key) !== null);
}

// The MLX measured peak (GB) the manifest declares for a tier on `os`, or null.
function mlxPeakGb(manifestModel, os, tier) {
  const entry = catalogEntry(manifestModel, os, [tier]);
  const bytes = entry.variants.find((v) => v.variant === tier)?.footprint?.peakMemoryBytes;
  return typeof bytes === "number" && bytes > 0 ? bytes / BYTES_PER_GB : null;
}

const manifestModels = loadManifestModels();
const OSES = ["macos", "windows"];
// The lane each OS runs, matching every consumer's `macGatingActive ? "mlx" : "candle"`.
const BACKEND_FOR_OS = { macos: "mlx", windows: "candle" };

describe("catalog memory floors: the displayed number satisfies the app's own fit criterion", () => {
  // Every (model, tier) on macOS that carries a real measured MLX footprint. This is the set whose
  // numbers reach a user as "needs N GB", so every one of them must satisfy `tierFits`.
  const measuredMlx = [];
  for (const model of manifestModels) {
    for (const tier of tiersOn(model, "macos")) {
      const peak = mlxPeakGb(model, "macos", tier);
      if (peak !== null) {
        measuredMlx.push({ id: model.id, tier, peak, model });
      }
    }
  }

  it("covers a non-trivial set of really-measured tiers", () => {
    // Guards the whole describe against silently asserting nothing if the manifest shape moves.
    expect(measuredMlx.length).toBeGreaterThanOrEqual(8);
  });

  it.each(measuredMlx.map((row) => [`${row.id} ${row.tier}`, row]))(
    "%s: the derived host size fits the measured peak, and one GB less does not",
    (_label, row) => {
      const shown = hostGbForPeakGb(row.peak);
      expect(shown).toBeGreaterThan(0);
      // The app's OWN criterion, not a restatement of it: the peak must fit under the headroom budget.
      expect(row.peak, `${_label}: ${shown} GB must fit a ${row.peak.toFixed(2)} GiB peak`).toBeLessThanOrEqual(
        shown * MEMORY_HEADROOM_FRACTION,
      );
      // TIGHT: the displayed figure is the MINIMUM honest host, so one GB below must fail. This is the
      // assertion the raw-peak bug fails — ceil(peak) is always below peak/0.9 for a peak above ~0.
      expect(row.peak, `${_label}: ${shown - 1} GB must NOT fit`).toBeGreaterThan(
        (shown - 1) * MEMORY_HEADROOM_FRACTION,
      );
    },
  );

  it.each(measuredMlx.map((row) => [`${row.id} ${row.tier}`, row]))(
    "%s: agrees with tierFits, the gate the download suggestion already uses",
    (_label, row) => {
      const entry = catalogEntry(row.model, "macos", [row.tier]);
      const variant = entry.variants.find((v) => v.variant === row.tier);
      const shown = installedTierPeakGb(entry, { backend: "mlx" });
      expect(shown).not.toBeNull();
      const hostGb = hostGbForPeakGb(shown);
      expect(tierFits(variant, hostGb), `${_label}: tierFits must accept ${hostGb} GB`).toBe(true);
      expect(tierFits(variant, hostGb - 1), `${_label}: tierFits must reject ${hostGb - 1} GB`).toBe(
        false,
      );
    },
  );

  // The reported figure must never sit BELOW the lane's own measured requirement — the one direction
  // that causes the reported failure mode (a user downloads a model that then OOMs).
  it("never reports a host size below the lane's measured peak, on either lane", () => {
    let checked = 0;
    for (const os of OSES) {
      const backend = BACKEND_FOR_OS[os];
      for (const model of manifestModels) {
        const tiers = tiersOn(model, os);
        if (tiers.length === 0) {
          continue;
        }
        // The "everything installed" host: the ceiling over all tiers.
        const entry = catalogEntry(model, os, tiers);
        const peak = installedTierPeakGb(entry, { backend });
        if (peak === null) {
          continue;
        }
        const shown = hostGbForPeakGb(peak);
        checked++;
        expect(shown, `${model.id} on ${os}`).toBeGreaterThanOrEqual(Math.ceil(peak));
        // ...and it is the ceiling, not some tier's number: no single tier may exceed it.
        for (const tier of tiers) {
          const lanePeak =
            backend === "candle" ? (model.candle?.vramGbByTier ?? {})[tier] : mlxPeakGb(model, os, tier);
          if (typeof lanePeak === "number" && lanePeak > 0) {
            expect(lanePeak, `${model.id} ${tier} on ${os} must not exceed the reported ceiling`).toBeLessThanOrEqual(
              shown * MEMORY_HEADROOM_FRACTION,
            );
          }
        }
      }
    }
    expect(checked, "must actually check some models on both lanes").toBeGreaterThanOrEqual(8);
  });
});

describe("catalog memory floors: the two backend lanes are independent", () => {
  // THE BLOCKER-2 GUARD, proved structurally rather than by transcribing expected numbers: strip one
  // lane's evidence out of the catalog entry and the OTHER lane's answer must be unchanged. If any
  // consumer ever reads `footprint.peakMemoryBytes` for candle (or `candle.vramGbByTier` for MLX), one of
  // these goes red for whichever real models carry both — no new fixture needed.
  function stripMlxFootprints(entry) {
    return {
      ...entry,
      variants: entry.variants.map((variant) => ({
        ...variant,
        footprint: variant.footprint
          ? { ...variant.footprint, peakMemoryBytes: undefined, residentMemoryBytes: undefined }
          : variant.footprint,
      })),
    };
  }

  function stripCandle(entry) {
    const { candle: _candle, ...rest } = entry;
    return rest;
  }

  // Models carrying BOTH an MLX footprint and candle per-tier evidence on the candle OS — the entries
  // where crossing the lanes is silently plausible because both numbers exist.
  const bothLanes = manifestModels.filter((model) => {
    const tiers = tiersOn(model, "windows");
    const hasMlx = tiers.some((tier) => mlxPeakGb(model, "windows", tier) !== null);
    const hasCandle = Object.keys(model.candle?.vramGbByTier ?? {}).length > 0;
    return hasMlx && hasCandle;
  });

  it("finds real models that carry both lanes' evidence", () => {
    // z_image_turbo and lens_turbo at minimum. If this ever hits zero the two tests below are vacuous.
    expect(bothLanes.map((model) => model.id)).toContain("lens_turbo");
    expect(bothLanes.length).toBeGreaterThanOrEqual(2);
  });

  it.each(bothLanes.map((model) => [model.id, model]))(
    "%s: the candle answer ignores the MLX footprint entirely",
    (_id, model) => {
      const tiers = tiersOn(model, "windows");
      const entry = catalogEntry(model, "windows", tiers);
      const withMlx = installedTierPeakGb(entry, { backend: "candle" });
      const withoutMlx = installedTierPeakGb(stripMlxFootprints(entry), { backend: "candle" });
      expect(withoutMlx).toEqual(withMlx);
      // Same for the uninstalled "from" floor.
      expect(cheapestDeclaredTierPeakGb(stripMlxFootprints(entry), { backend: "candle" })).toEqual(
        cheapestDeclaredTierPeakGb(entry, { backend: "candle" }),
      );
    },
  );

  it.each(bothLanes.map((model) => [model.id, model]))(
    "%s: the MLX answer ignores the candle block entirely",
    (_id, model) => {
      const tiers = tiersOn(model, "macos");
      const entry = catalogEntry(model, "macos", tiers);
      expect(installedTierPeakGb(stripCandle(entry), { backend: "mlx" })).toEqual(
        installedTierPeakGb(entry, { backend: "mlx" }),
      );
      expect(blanketFloorGb(stripCandle(entry), "mlx")).toEqual(blanketFloorGb(entry, "mlx"));
    },
  );

  it("never answers the candle lane from MLX evidence when candle declares none", () => {
    // The pre-fix failure shape: a windows/linux-served tier with an MLX footprint and NO candle
    // evidence. z_image and sdxl are the shipped examples. Silence is required — NOT the MLX number.
    let checked = 0;
    for (const model of manifestModels) {
      const tiers = tiersOn(model, "windows");
      const hasMlx = tiers.some((tier) => mlxPeakGb(model, "windows", tier) !== null);
      const hasCandleTier = Object.keys(model.candle?.vramGbByTier ?? {}).length > 0;
      const hasCandleBlanket = Number.isFinite(model.candle?.minMemoryGb);
      if (!hasMlx || hasCandleTier || hasCandleBlanket) {
        continue;
      }
      checked++;
      const entry = catalogEntry(model, "windows", tiers);
      expect(installedTierPeakGb(entry, { backend: "candle" }), `${model.id}`).toBeNull();
      expect(cheapestDeclaredTierPeakGb(entry, { backend: "candle" }), `${model.id}`).toBeNull();
      expect(blanketFloorGb(entry, "candle"), `${model.id}`).toBeNull();
    }
    expect(checked, "z_image / sdxl / illustrious_xl_* have this shape").toBeGreaterThanOrEqual(3);
  });

  // The reason the MLX blanket may NOT be reused as a "conservative" candle default. This is a manifest
  // fact, so it belongs here: if it ever became true that mlx >= candle everywhere, the fallback choice
  // could be revisited — until then, cross-lane reuse can under-state.
  it("records that the MLX blanket is NOT always at or above the candle requirement", () => {
    const inversions = manifestModels
      .filter((model) => {
        const mlx = model.mlx?.minMemoryGb;
        const candle = model.candle?.minMemoryGb;
        return Number.isFinite(mlx) && Number.isFinite(candle) && mlx < candle;
      })
      .map((model) => `${model.id} (mlx ${model.mlx.minMemoryGb} < candle ${model.candle.minMemoryGb})`);
    // qwen_image is the shipped counterexample. Its existence is what makes silence, rather than the MLX
    // integer, the correct candle-lane fallback in SimpleModelManager's `needsLabel`.
    expect(inversions.join(", ")).toContain("qwen_image");
  });
});

describe("catalog memory floors: duplicate tier keys cannot reach the client", () => {
  // The invariant MINOR-6's max-over-duplicates change stops depending on silently. Three upstream
  // filters enforce it (coRequisite, per-OS platforms, first-wins de-dupe); this pins that they do, so
  // the client-side hardening is provably belt-and-braces rather than load-bearing.
  it.each(OSES)("emits at most one entry per tier key on %s", (os) => {
    let withDuplicatesUpstream = 0;
    for (const model of manifestModels) {
      const entry = catalogEntry(model, os, []);
      const keys = entry.variants.map((variant) => variant.variant);
      expect(new Set(keys).size, `${model.id} on ${os} emitted duplicate variant keys`).toBe(
        keys.length,
      );
      // ...while the RAW manifest really does declare duplicates, so the filters are doing work.
      const rawTierKeys = (model.downloads ?? [])
        .map((download) => download.variant)
        .filter((key) => tierQuantize(key) !== null);
      if (new Set(rawTierKeys).size !== rawTierKeys.length) {
        withDuplicatesUpstream++;
      }
    }
    // mage_flow x6 (coRequisite component rows) + wan_2_2* x3 (per-platform twins).
    expect(withDuplicatesUpstream).toBeGreaterThanOrEqual(6);
  });
});
