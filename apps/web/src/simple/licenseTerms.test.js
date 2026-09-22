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

  it("badges a restricted component whose licence NAME carries no marker (sc-24108)", () => {
    // The Qwen RESEARCH LICENSE AGREEMENT is research/evaluation-only (§1(i), §2(a)) but its
    // TITLE says nothing of the kind, so the name match alone badges it "Commercial OK". The
    // component's explicit `nonCommercial: true` is what makes the badge right, and this pins
    // BOTH halves: the declaration is present on the shipped component, and the badge prefers it.
    const component = bundledLicenses.find((entry) => entry.id === "qwen-image-2-1");
    expect(component).toBeTruthy();
    expect(component.models).toEqual(["qwen_image_2_1"]);
    expect(component.nonCommercial).toBe(true);
    // The title genuinely does NOT match the historical classifier — i.e. the flag is load-bearing
    // here, not redundant belt-and-braces.
    expect(licenseIsNonCommercial(component.license)).toBe(false);
    expect(licenseBadge(component)).toEqual({ label: "Non-commercial", tone: "danger" });

    const row = modelLicenseRows(bundledLicenses).find((entry) => entry.id === "qwen-image-2-1");
    expect(row).toBeTruthy();
    expect(row.license).toBe("Qwen RESEARCH LICENSE AGREEMENT");
    expect(row.badge.label).toBe("Non-commercial");
  });

  it("falls back to the licence name for a component that declares no flag", () => {
    // Every pre-existing component keeps the badge it had: no `nonCommercial` key, name match only.
    const fluxDev = bundledLicenses.find((entry) => entry.id === "flux1-dev");
    expect(fluxDev?.nonCommercial).toBeUndefined();
    expect(licenseBadge(fluxDev).label).toBe("Non-commercial");
    const apache = bundledLicenses.find((entry) => entry.id === "qwen-image");
    expect(apache?.nonCommercial).toBeUndefined();
    expect(licenseBadge(apache).label).toBe("Commercial OK");
  });

  it("surfaces the alternate decoder terms for every compatible catalog product", () => {
    const decoder = bundledLicenses.find((component) => component.id === "wan2_1_t2v_14b_diffusers");
    expect(decoder?.models).toEqual([]);
    expect(decoder?.appliesToModels).toEqual([
      "krea_2_turbo",
      "krea_2_raw",
      "qwen_image",
      "qwen_image_edit_2511",
      "qwen_image_edit_2511_lightning",
    ]);

    const decoderRow = modelLicenseRows(bundledLicenses).find(
      (row) => row.id === "wan2_1_t2v_14b_diffusers",
    );
    expect(decoderRow?.license).toBe("Apache-2.0");
  });

  it("keeps Tile primary while composing OpenPose terms with the exact five accepted backbones", () => {
    const controlnet = bundledLicenses.find(
      (component) => component.id === "controlnet-tile-sdxl",
    );
    expect(controlnet?.models).toEqual(["controlnet_tile_sdxl"]);
    expect(controlnet?.appliesToModels).toEqual([
      "sdxl",
      "realvisxl",
      "realvisxl_lightning",
      "illustrious_xl_v1",
      "illustrious_xl_v2",
    ]);

    const controlnetRow = modelLicenseRows(bundledLicenses).find(
      (row) => row.id === "controlnet-tile-sdxl",
    );
    expect(controlnetRow?.license).toBe("Apache-2.0");
  });

  it("tolerates an empty or absent corpus", () => {
    expect(modelLicenseRows([])).toEqual([]);
    expect(modelLicenseRows(undefined)).toEqual([]);
  });
});
