// @vitest-environment jsdom
import { beforeEach, expect, it } from "vitest";
import { cachedModelFacts, rememberModelFacts } from "./portableModelFacts.js";

beforeEach(() => localStorage.clear());

it("retains last served model facts for offline pure validation", () => {
  const entry = { id: "model-a", family: "sdxl", limits: { steps: [1, 50] } };
  expect(rememberModelFacts("model-a", entry)).toBe(true);
  expect(cachedModelFacts("model-a")).toEqual(entry);
  expect(cachedModelFacts("model-b")).toBeNull();
});

it("does not cache a catalog row under another model identity", () => {
  expect(rememberModelFacts("model-a", { id: "model-b" })).toBe(false);
  expect(cachedModelFacts("model-a")).toBeNull();
});
