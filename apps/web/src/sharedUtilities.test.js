import { describe, expect, it } from "vitest";
import { downloadSizeText, formatMiB, formatTierBytes } from "./formatting.js";
import { backoffDelay } from "./retry.js";
import { isModelMissing } from "./modelMissing.js";

describe("shared utility contracts", () => {
  it("keeps download and memory fallbacks and binary display conventions", () => {
    expect(downloadSizeText(null, "Unavailable")).toBe("Unavailable");
    expect(downloadSizeText({ downloadSizeLabel: "2 GB", downloadSizeEstimated: true })).toBe("~2 GB");
    expect(formatTierBytes(1)).toBe("0 MiB");
    expect(formatTierBytes("1024")).toBe("Size unavailable");
    expect(formatMiB(1024)).toBe("1.0 GiB");
    expect(formatMiB(NaN)).toBe("Unknown");
  });
  it("caps retries and honors authoritative model state", () => {
    expect(backoffDelay(8, { baseMs: 1000, maxMs: 30000 })).toBe(30000);
    expect(isModelMissing({ installState: "installed" }, "not cached")).toBe(false);
    expect(isModelMissing(null, "snapshot is not cached")).toBe(true);
  });
});
