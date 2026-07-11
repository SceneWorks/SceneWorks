import { describe, expect, it } from "vitest";

import {
  PID_HEADS_UP_MIN_OUTPUT_LONG_SIDE,
  PID_SR_SCALE,
  pidDecodeHeadsUp,
} from "./pidDecodeNotice.js";

// sc-10144: PiD super-resolves the base render 4×, so a large base at the default 4K tier is a
// multi-minute (or auto-tiled, sc-10087) decode. These tests pin exactly when the Studio surfaces the
// "it's working, not stuck" heads-up so the threshold + copy can't silently drift.

describe("pidDecodeHeadsUp", () => {
  it("returns null when PiD is off (whatever the size)", () => {
    expect(pidDecodeHeadsUp({ usePid: false, pidTarget: "4k", width: 2048, height: 2048 })).toBeNull();
  });

  it("returns null for the 2K tier — the base is capped to ~2048² output, always fast", () => {
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "2k", width: 1536, height: 1536 })).toBeNull();
    // Case-insensitive / padded values still resolve to the fast tier.
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: " 2K ", width: 4096, height: 4096 })).toBeNull();
  });

  it("returns null below the multi-minute threshold (e.g. a 768 base → 3072² output)", () => {
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 768, height: 768 })).toBeNull();
  });

  it("warns (non-tiled) exactly at the 4096² boundary — a 1024 base at 4K", () => {
    const notice = pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 1024, height: 1024 });
    expect(notice).not.toBeNull();
    expect(notice.outputWidth).toBe(4096);
    expect(notice.outputHeight).toBe(4096);
    expect(notice.tiled).toBe(false);
    expect(notice.megapixels).toBeCloseTo(16.78, 1);
    expect(notice.message).toContain("4096×4096");
    expect(notice.message).toContain("not stuck");
    expect(notice.message).not.toContain("tiles");
  });

  it("warns (tiled) above 4096² — the sc-10087 1536 base → 6144² repro", () => {
    const notice = pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 1536, height: 1536 });
    expect(notice.outputWidth).toBe(6144);
    expect(notice.tiled).toBe(true);
    expect(notice.megapixels).toBeCloseTo(37.75, 1);
    expect(notice.message).toContain("6144×6144");
    expect(notice.message).toContain("tiles");
  });

  it("defaults an unrecognized / missing tier to the 4K (warn) path, matching the worker", () => {
    // The worker treats any non-\"2k\" pidTarget as 4K; the helper mirrors that so a stray value warns.
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: undefined, width: 1024, height: 1024 })).not.toBeNull();
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 1024, height: 1024 }).tiled).toBe(false);
  });

  it("keys off the long side for non-square requests", () => {
    // 1280×720 base → 5120×2880 output: long side 5120 > 4096 → tiled warn.
    const notice = pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 1280, height: 720 });
    expect(notice.outputWidth).toBe(5120);
    expect(notice.outputHeight).toBe(2880);
    expect(notice.tiled).toBe(true);
  });

  it("returns null for invalid / non-finite dimensions", () => {
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: 0, height: 1024 })).toBeNull();
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: NaN, height: 1024 })).toBeNull();
    expect(pidDecodeHeadsUp({ usePid: true, pidTarget: "4k", width: "", height: "" })).toBeNull();
  });

  it("exposes the super-res factor + threshold as the single source of truth", () => {
    expect(PID_SR_SCALE).toBe(4);
    expect(PID_HEADS_UP_MIN_OUTPUT_LONG_SIDE).toBe(4096);
  });
});
