import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_GENERATION_QUALITY,
  generationQualityLabel,
  normalizeGenerationQuality,
  readDefaultGenerationQuality,
  writeDefaultGenerationQuality,
} from "./generationQuality.js";

const STORAGE_KEY = "sceneworks-default-generation-quality";

describe("generationQuality store (sc-10728)", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("defaults to q8 when nothing is stored", () => {
    expect(DEFAULT_GENERATION_QUALITY).toBe("q8");
    expect(readDefaultGenerationQuality()).toBe("q8");
  });

  it("reads back a valid stored value", () => {
    window.localStorage.setItem(STORAGE_KEY, "bf16");
    expect(readDefaultGenerationQuality()).toBe("bf16");
    window.localStorage.setItem(STORAGE_KEY, "q4");
    expect(readDefaultGenerationQuality()).toBe("q4");
  });

  it("normalizes an unknown/garbage stored value back to q8", () => {
    window.localStorage.setItem(STORAGE_KEY, "int8-convrot");
    expect(readDefaultGenerationQuality()).toBe("q8");
    window.localStorage.setItem(STORAGE_KEY, "totally-bogus");
    expect(readDefaultGenerationQuality()).toBe("q8");
  });

  it("persists a write and survives a simulated restart (round-trip)", () => {
    // writeDefaultGenerationQuality returns the normalized value it stored…
    expect(writeDefaultGenerationQuality("bf16")).toBe("bf16");
    // …and a fresh read (no in-memory cache) returns it, mimicking a new session.
    expect(readDefaultGenerationQuality()).toBe("bf16");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("bf16");
  });

  it("write normalizes an invalid value to q8 before persisting", () => {
    expect(writeDefaultGenerationQuality("nonsense")).toBe("q8");
    expect(readDefaultGenerationQuality()).toBe("q8");
  });

  it("normalizeGenerationQuality validates against bf16|q8|q4", () => {
    expect(normalizeGenerationQuality("bf16")).toBe("bf16");
    expect(normalizeGenerationQuality("q8")).toBe("q8");
    expect(normalizeGenerationQuality("q4")).toBe("q4");
    expect(normalizeGenerationQuality("int8-convrot")).toBe("q8");
    expect(normalizeGenerationQuality(null)).toBe("q8");
    expect(normalizeGenerationQuality(undefined)).toBe("q8");
  });

  it("labels each tier legibly and falls back to the raw key", () => {
    expect(generationQualityLabel("bf16")).toBe("High fidelity (bf16)");
    expect(generationQualityLabel("q8")).toBe("Balanced (Q8)");
    expect(generationQualityLabel("q4")).toBe("Fast (Q4)");
    expect(generationQualityLabel("mystery")).toBe("mystery");
  });
});
