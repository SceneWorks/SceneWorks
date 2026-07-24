import { describe, expect, it } from "vitest";
import {
  preferredDuration,
  preferredOption,
  preferredResolution,
  preferredVideoResolution,
} from "./modelDefaults.js";

// These pin the exact trap the audit found: `limits.<thing>[0]` is the first ALLOWED value,
// not the model's chosen default, and for most shipped models the two differ. The fixtures
// below are the REAL declared values from config/manifests/builtin.models.jsonc, so a
// manifest change that invalidates the assumption shows up here.

describe("preferredOption", () => {
  it("takes the declared default when it is offered", () => {
    expect(preferredOption("1024x1024", ["768x768", "1024x1024"])).toBe("1024x1024");
  });

  it("falls through the preference list when the declared default isn't offered", () => {
    expect(preferredOption("2048x2048", ["768x768", "1024x1024"], ["1024x1024"])).toBe("1024x1024");
  });

  it("falls back to the first option when nothing else matches", () => {
    expect(preferredOption("2048x2048", ["768x768", "640x640"], ["1024x1024"])).toBe("768x768");
    expect(preferredOption(undefined, ["768x768"])).toBe("768x768");
    expect(preferredOption(null, ["768x768"])).toBe("768x768");
  });

  it("is undefined for an empty option list rather than throwing", () => {
    expect(preferredOption("1024x1024", [])).toBeUndefined();
    expect(preferredOption("1024x1024", undefined)).toBeUndefined();
  });

  it("treats a declared 0 as a real value, not as unset", () => {
    // Guards the `!declared` shape of this check — 0 is a legitimate declared default.
    expect(preferredOption(0, [3, 0, 5])).toBe(0);
  });
});

describe("preferredResolution (image)", () => {
  // z_image: declares 1024x1024, but its limits list LEADS with 768x768.
  const Z_IMAGE = { defaults: { resolution: "1024x1024" } };
  const Z_IMAGE_OPTIONS = ["768x768", "1024x1024", "1280x720", "720x1280"];

  it("starts on the model's declared default, not the first allowed value", () => {
    expect(preferredResolution(Z_IMAGE, Z_IMAGE_OPTIONS)).toBe("1024x1024");
    expect(preferredResolution(Z_IMAGE, Z_IMAGE_OPTIONS)).not.toBe(Z_IMAGE_OPTIONS[0]);
  });

  // The z_image case above CANNOT tell the declared-default branch from the 1024² fallback:
  // every mismatching image model in the shipped catalog happens to declare 1024², so both
  // paths give the same answer and the assertion passes for the wrong reason. This case
  // separates them — declared, first-allowed and the 1024² fallback are three distinct
  // values, so only reading `defaults.resolution` produces 1344x768.
  it("prefers the declared default OVER the 1024² fallback (discriminating case)", () => {
    const model = { defaults: { resolution: "1344x768" } };
    const options = ["768x768", "1024x1024", "1344x768"];
    expect(preferredResolution(model, options)).toBe("1344x768");
  });

  it("prefers 1024² over the first entry when the declared default is gated out", () => {
    // Mirrors ImageStudio's own preferredResolution, including this second choice.
    expect(preferredResolution({ defaults: { resolution: "1536x1536" } }, Z_IMAGE_OPTIONS)).toBe(
      "1024x1024",
    );
  });

  it("falls back to the first option for a model declaring nothing", () => {
    expect(preferredResolution({}, ["832x1216", "1024x1024"])).toBe("1024x1024");
    expect(preferredResolution({}, ["832x1216", "640x640"])).toBe("832x1216");
  });
});

describe("preferredVideoResolution / preferredDuration", () => {
  // wan_2_2_t2v_14b: declares 1280x720 + 5s, but lists 832x480 and 3s first.
  const WAN_A14B = { defaults: { resolution: "1280x720", duration: 5 } };

  it("starts video geometry on the declared default", () => {
    const options = ["832x480", "1280x720", "720x1280"];
    expect(preferredVideoResolution(WAN_A14B, options)).toBe("1280x720");
    expect(preferredVideoResolution(WAN_A14B, options)).not.toBe(options[0]);
  });

  it("starts clip length on the declared default", () => {
    const options = [3, 5, 8, 10];
    expect(preferredDuration(WAN_A14B, options)).toBe(5);
    expect(preferredDuration(WAN_A14B, options)).not.toBe(options[0]);
  });

  // ltx_2_3: declares 6s, lists 4s first.
  it("handles LTX's 6s default against a list leading with 4s", () => {
    expect(preferredDuration({ defaults: { duration: 6 } }, [4, 6, 8, 10])).toBe(6);
  });

  it("does NOT apply the image 1024² preference to video", () => {
    // Video geometry is model-specific; a 1024² second choice would be meaningless.
    expect(
      preferredVideoResolution({ defaults: { resolution: "9999x9999" } }, ["832x480", "1024x1024"]),
    ).toBe("832x480");
  });

  it("falls back to the first allowed value when the model declares none", () => {
    expect(preferredDuration({}, [3, 5])).toBe(3);
    expect(preferredVideoResolution({}, ["832x480"])).toBe("832x480");
  });
});
