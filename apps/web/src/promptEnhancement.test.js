import { describe, expect, it } from "vitest";

import { promptEnhancementAvailable } from "./promptEnhancement.js";

const dev = { id: "flux2_dev", ui: { promptEnhance: true } };

describe("promptEnhancementAvailable", () => {
  it("admits both FLUX 3 checkpoints on native generation and reference routes", () => {
    for (const id of ["flux3_klein", "flux3_klein_base"]) {
      const model = { id, ui: { promptEnhance: true } };
      for (const backend of ["mlx", "candle"]) {
        for (const mode of ["text_to_image", "edit_image", "character_image", "image_to_image"]) {
          expect(promptEnhancementAvailable({ model, backend, mode })).toBe(true);
        }
        for (const mode of ["style_variations", "reference"]) {
          expect(promptEnhancementAvailable({ model, backend, mode })).toBe(false);
        }
        expect(promptEnhancementAvailable({ model, backend, mode: "text_to_image", strictControlActive: true })).toBe(false);
        expect(promptEnhancementAvailable({ model: { ...model, ui: {} }, backend, mode: "text_to_image" })).toBe(false);
      }
      expect(promptEnhancementAvailable({ model, backend: "torch", mode: "text_to_image" })).toBe(false);
    }
  });

  it("admits every native MLX FLUX.2-dev base and edit route", () => {
    for (const mode of ["text_to_image", "edit_image", "character_image"]) {
      expect(promptEnhancementAvailable({ model: dev, backend: "mlx", mode })).toBe(true);
    }
  });

  it("does not revive the retired style-variations mode on either backend", () => {
    for (const backend of ["mlx", "candle"]) {
      expect(promptEnhancementAvailable({ model: dev, backend, mode: "style_variations" })).toBe(false);
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

  it("fails closed for FLUX.2 Klein, unknown backends, missing catalog declarations, and strict control", () => {
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
