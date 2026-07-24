import { describe, expect, it } from "vitest";
import {
  DESKTOP_MIN_WIDTH,
  PHONE_MAX_WIDTH,
  assetGridColumns,
  breakpointFor,
  contentMaxWidth,
  isPhoneWidth,
  resultGridColumns,
} from "./breakpoint.js";

describe("Simple UI breakpoints", () => {
  it("splits the three bands on the design's boundaries", () => {
    expect(breakpointFor(PHONE_MAX_WIDTH - 1)).toBe("phone");
    expect(breakpointFor(PHONE_MAX_WIDTH)).toBe("tablet");
    expect(breakpointFor(DESKTOP_MIN_WIDTH - 1)).toBe("tablet");
    expect(breakpointFor(DESKTOP_MIN_WIDTH)).toBe("desktop");
  });

  it("resolves an unmeasured width to desktop rather than flashing the phone drawer", () => {
    expect(breakpointFor(null)).toBe("desktop");
    expect(breakpointFor(0)).toBe("desktop");
    expect(breakpointFor(Number.NaN)).toBe("desktop");
  });

  it("agrees with isPhoneWidth, which App uses for the forced-Simple rule", () => {
    expect(isPhoneWidth(PHONE_MAX_WIDTH - 1)).toBe(true);
    expect(isPhoneWidth(PHONE_MAX_WIDTH)).toBe(false);
    // Unknown width must NOT read as a phone — it would force a shell swap on no signal.
    expect(isPhoneWidth(null)).toBe(false);
    expect(isPhoneWidth(0)).toBe(false);
  });

  it("caps the content column per band", () => {
    expect(contentMaxWidth("phone")).toBe("none");
    expect(contentMaxWidth("tablet")).toBe("680px");
    expect(contentMaxWidth("desktop")).toBe("900px");
  });

  it("sets asset-grid columns to 3 / 4 / 6", () => {
    expect(assetGridColumns("phone")).toBe(3);
    expect(assetGridColumns("tablet")).toBe(4);
    expect(assetGridColumns("desktop")).toBe(6);
  });

  it("gives a single result its own column and only 6-up on desktop three", () => {
    expect(resultGridColumns(1, "desktop")).toBe(1);
    expect(resultGridColumns(2, "desktop")).toBe(2);
    expect(resultGridColumns(4, "desktop")).toBe(2);
    expect(resultGridColumns(6, "desktop")).toBe(3);
    // A 6-up run on a narrower band stays 2 columns.
    expect(resultGridColumns(6, "tablet")).toBe(2);
    expect(resultGridColumns(6, "phone")).toBe(2);
  });
});
