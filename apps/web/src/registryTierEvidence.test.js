import { expect, it } from "vitest";
import old from "./testUtils/catalogBaseline/old-catalog.json";
import candidate from "./testUtils/catalogBaseline/candidate-catalog.json";
import { installedFloorHostGb, suggestTier } from "./registryTierEvidence.js";

const memoryFields = [
  "minMemoryGb",
  "minMemoryGbByTier",
  "vramGbByTier",
  "measured",
  "sequentialPeakGb",
  "vramMeasuredPixels",
];

function memoryFacts(model) {
  const lane = (backend) =>
    Object.fromEntries(memoryFields.map((field) => [field, model[backend]?.[field] ?? null]));
  return {
    mlx: lane("mlx"),
    candle: lane("candle"),
    downloads: (model.downloads ?? []).map(({ repo, variant, platforms, footprint }) => ({
      repo,
      variant: variant ?? null,
      platforms: platforms ?? null,
      footprint: footprint ?? null,
    })),
  };
}

function installedProjection(model, platform) {
  const byTier = new Map();
  for (const download of model.downloads ?? []) {
    if (
      !download.variant ||
      (download.platforms?.length && !download.platforms.includes(platform))
    ) {
      continue;
    }
    byTier.set(download.variant, {
      variant: download.variant,
      footprint: download.footprint,
      installState: "installed",
      cacheState: "complete",
    });
  }
  return {
    ...model,
    hasVariantMatrix: byTier.size > 1,
    variants: [...byTier.values()],
  };
}

it("preserves every frozen catalog memory fact and tier decision in the registry candidate", () => {
  const prior = new Map(old.catalog.models.map((model) => [model.id, model]));
  const current = new Map(candidate.models.map((model) => [model.id, model]));
  expect([...current.keys()].sort()).toEqual([...prior.keys()].sort());

  let tierDecisions = 0;
  for (const [id, oldModel] of prior) {
    const newModel = current.get(id);
    expect(memoryFacts(newModel), id).toEqual(memoryFacts(oldModel));
    for (const [platform, backend] of [["macos", "mlx"], ["windows", "candle"]]) {
      const before = installedProjection(oldModel, platform);
      const after = installedProjection(newModel, platform);
      if (!before.hasVariantMatrix) continue;
      for (const hostGb of [16, 32, 64, 128]) {
        const options = { backend };
        expect(suggestTier(after, hostGb, options), `${id}/${platform}/${hostGb}`).toBe(
          suggestTier(before, hostGb, options),
        );
        expect(installedFloorHostGb(after, options), `${id}/${platform}`).toBe(
          installedFloorHostGb(before, options),
        );
        tierDecisions++;
      }
    }
  }
  expect(tierDecisions).toBeGreaterThan(100);
});
