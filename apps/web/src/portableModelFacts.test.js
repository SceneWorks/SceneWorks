// @vitest-environment jsdom
import { beforeEach, expect, it } from "vitest";
import { cachedModelCatalog, rememberModelCatalog } from "./portableModelFacts.js";

beforeEach(() => localStorage.clear());

it("stores only display fields and the last served time", () => {
  const servedAt = Date.now();
  expect(rememberModelCatalog([{ id: "model-a", name: "A", type: "image", installState: "installed",
    capabilities: ["text_to_image"], usable: true, limits: { steps: [1, 50] } }])).toBe(true);
  expect(cachedModelCatalog()).toEqual({ savedAt: expect.any(Number), entries: [{
    id: "model-a", name: "A", type: "image", installState: "installed",
    capabilities: ["text_to_image"], usable: true,
  }] });
  expect(cachedModelCatalog().savedAt).toBeGreaterThanOrEqual(servedAt);
  expect(cachedModelCatalog().entries[0]).not.toHaveProperty("limits");
});
