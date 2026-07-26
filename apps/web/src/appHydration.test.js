import { describe, expect, it } from "vitest";

import {
  hydrationDomainsForRoute,
  hydrationLedgerKey,
  isProjectHydrationDomain,
} from "./appHydration.js";

describe("route-owned hydration contract (sc-14783)", () => {
  it("keeps default Assets isolated from every inactive catalog", () => {
    expect(hydrationDomainsForRoute("Library")).toEqual(["assets"]);
  });

  it.each([
    ["Image", ["assets", "models", "loras", "presets", "promptBatches", "characters"]],
    ["Video", ["assets", "models", "loras", "presets", "characters", "personTracks"]],
    ["Audio", ["assets", "models", "savedVoices"]],
    ["Characters", ["assets", "characters", "models", "loras", "presets", "trainingDatasets"]],
    ["Document", ["assets", "models"]],
    [
      "Train",
      ["assets", "characters", "models", "trainingTargets", "trainingPresets", "trainingDatasets"],
    ],
    [
      "LibraryDataSets",
      ["assets", "characters", "models", "trainingTargets", "trainingPresets", "trainingDatasets"],
    ],
    ["Poses", ["assets", "characters"]],
    ["Keypoints", ["assets", "characters"]],
    ["Presets", ["models", "loras", "presets"]],
    ["Models", ["models", "loras"]],
    ["Editor", ["assets", "characters", "models", "loras", "presets", "timelines"]],
    ["ImageEditor", ["assets", "characters", "models", "loras"]],
    ["Queue", []],
    ["Stats", []],
    ["Logs", []],
    ["Settings", []],
    ["Licenses", []],
  ])("declares every required domain for %s", (route, domains) => {
    expect(hydrationDomainsForRoute(route)).toEqual(domains);
  });

  it.each([
    ["image", ["assets", "models", "loras"]],
    ["video", ["assets", "models"]],
    ["audio", ["assets", "models"]],
    ["assets", ["assets"]],
    ["models", ["models", "loras"]],
    ["queue", []],
    ["settings", []],
    ["licenses", []],
  ])("keeps Simple %s demand-driven", (route, domains) => {
    expect(hydrationDomainsForRoute(route, { simple: true })).toEqual(domains);
  });

  it("keys global data once and project overlays by project", () => {
    expect(isProjectHydrationDomain("models")).toBe(false);
    expect(isProjectHydrationDomain("loras")).toBe(true);
    expect(hydrationLedgerKey("models", "project-a")).toBe("models:global");
    expect(hydrationLedgerKey("loras", "project-a")).toBe("loras:project:project-a");
    expect(hydrationLedgerKey("loras", "project-b")).toBe("loras:project:project-b");
  });
});
