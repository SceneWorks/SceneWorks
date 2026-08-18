import { describe, expect, it } from "vitest";
import { reconcileDecoderSelection } from "./imageDecoderSelection.js";

describe("reconcileDecoderSelection", () => {
  it("preserves a compatible replay selection even while its donor is unavailable", () => {
    expect(
      reconcileDecoderSelection("wan_2_1_vae", [
        { id: "wan_2_1_vae", available: false },
      ]),
    ).toBe("wan_2_1_vae");
  });

  it("resets only selections that are incompatible with the active model/backend", () => {
    expect(reconcileDecoderSelection("wan_2_1_vae", [])).toBe("native");
    expect(
      reconcileDecoderSelection("wan_2_1_vae", [
        { id: "some_other_decoder", available: true },
      ]),
    ).toBe("native");
    expect(reconcileDecoderSelection("native", [])).toBe("native");
  });
});
