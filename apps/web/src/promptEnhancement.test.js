import { describe, expect, it } from "vitest";

import { promptEnhancementAvailable } from "./promptEnhancement.js";

const dev = { id: "flux2_dev", ui: { promptEnhance: true } };

describe("promptEnhancementAvailable", () => {
  it("admits every native MLX FLUX.2-dev base and edit route", () => {
    for (const mode of ["text_to_image", "edit_image", "character_image", "style_variations"]) {
      expect(promptEnhancementAvailable({ model: dev, backend: "mlx", mode })).toBe(true);
    }
  });

  it("admits only executable Candle FLUX.2-dev base and canonical edit routes", () => {
    for (const mode of ["text_to_image", "edit_image"]) {
      expect(promptEnhancementAvailable({ model: dev, backend: "candle", mode })).toBe(true);
    }
    for (const mode of [
      "character_image",
      "style_variations",
      "reference",
      "image_to_image",
    ]) {
      expect(promptEnhancementAvailable({ model: dev, backend: "candle", mode })).toBe(false);
    }
  });

  it("fails closed for Klein, unknown backends, missing catalog declarations, and strict control", () => {
    expect(
      promptEnhancementAvailable({
        model: { id: "flux2_klein_9b", ui: { promptEnhance: true } },
        backend: "candle",
        mode: "text_to_image",
      }),
    ).toBe(false);
    expect(promptEnhancementAvailable({ model: dev, backend: "torch", mode: "text_to_image" })).toBe(false);
    expect(
      promptEnhancementAvailable({ model: { id: "flux2_dev", ui: {} }, backend: "mlx", mode: "text_to_image" }),
    ).toBe(false);
    expect(
      promptEnhancementAvailable({
        model: dev,
        backend: "mlx",
        mode: "text_to_image",
        strictControlActive: true,
      }),
    ).toBe(false);
  });
});
