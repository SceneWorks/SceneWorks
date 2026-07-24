import { describe, expect, it } from "vitest";
import { licenseBadge, licenseIsNonCommercial, modelLicenseRows } from "./licenseTerms.js";
import { bundledLicenses } from "../data/bundledLicenses.js";

describe("licenseIsNonCommercial", () => {
  it("flags every restricted licence string the shipped corpus actually carries", () => {
    // Pinned against the real strings in apps/desktop/licenses/manifest.json, so a
    // rename upstream that this classifier stops matching fails here rather than
    // silently badging a non-commercial model as "Commercial OK".
    for (const license of [
      "FLUX.1 [dev] Non-Commercial License v1.1.1",
      "FLUX.2 [dev] Non-Commercial License",
      "FLUX Non-Commercial License",
      "Ideogram Non-Commercial Model Agreement",
      "CircleStone Labs Non-Commercial License v1.2",
      "Stable Video Diffusion Non-Commercial Community License",
      "Research / Non-Commercial (CC-BY-NC-4.0 · Apple ML Research · MIT)",
    ]) {
      expect(licenseIsNonCommercial(license), license).toBe(true);
    }
  });

  it("leaves permissive and conditional licences on the permissive badge", () => {
    for (const license of [
      "Apache-2.0",
      "MIT",
      "BSD-3-Clause",
      "CreativeML Open RAIL++-M",
      "Stability AI Community License",
      "Krea 2 Community License",
      "LTX-2 Community License",
      "NVIDIA Open Model License",
    ]) {
      expect(licenseIsNonCommercial(license), license).toBe(false);
    }
  });

  it("treats a missing licence string as unrestricted rather than throwing", () => {
    expect(licenseIsNonCommercial(null)).toBe(false);
    expect(licenseIsNonCommercial(undefined)).toBe(false);
    expect(licenseIsNonCommercial("")).toBe(false);
  });

  it("maps to the design's two badges", () => {
    expect(licenseBadge("MIT")).toEqual({ label: "Commercial OK", tone: "ok" });
    expect(licenseBadge("FLUX Non-Commercial License")).toEqual({
      label: "Non-commercial",
      tone: "danger",
    });
  });
});

describe("modelLicenseRows", () => {
  it("keeps only components that ship model weights", () => {
    const rows = modelLicenseRows(bundledLicenses);
    expect(rows.length).toBeGreaterThan(0);
    // FFmpeg / ONNX Runtime / the CUDA runtime are tooling binaries with `models: []`.
    expect(rows.some((row) => row.id === "ffmpeg")).toBe(false);
    expect(rows.some((row) => row.id === "onnxruntime")).toBe(false);
    // …and a real weights component IS present, with its badge resolved.
    const zImage = rows.find((row) => row.name.includes("Z-Image"));
    expect(zImage).toBeTruthy();
    expect(zImage.badge.label).toBe("Commercial OK");
  });

  it("badges a restricted weights component as non-commercial", () => {
    const rows = modelLicenseRows(bundledLicenses);
    const fluxDev = rows.find((row) => row.name === "FLUX.1 [dev]");
    expect(fluxDev).toBeTruthy();
    expect(fluxDev.badge.label).toBe("Non-commercial");
  });

  it("tolerates an empty or absent corpus", () => {
    expect(modelLicenseRows([])).toEqual([]);
    expect(modelLicenseRows(undefined)).toEqual([]);
  });
});
