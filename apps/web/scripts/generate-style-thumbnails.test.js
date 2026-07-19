// sc-13135 — Unit coverage for the pure logic of the style-thumbnail generator. The live
// submit→poll→download→resize path needs a running rust-api + model/GPU and is not exercised
// here; these tests pin the deterministic pieces (enumeration, client-side prompt composition,
// payload shape, path mapping, arg parsing) that must stay correct across API builds.
import { describe, expect, it } from "vitest";

import { composeStyledPrompt } from "../src/styleComposer.js";
import { styleTextForId } from "../src/data/styleCatalog.js";

import {
  buildJobPayload,
  composeThumbnailPrompt,
  enumerateStyleIds,
  parseArgs,
  shouldSkip,
  STYLE_GROUPS,
  thumbnailPath,
} from "./generate-style-thumbnails.mjs";

const REF = "a red fox in a snowy forest";

describe("enumerateStyleIds", () => {
  it("returns all 286 ids: 8 group ids + 278 sub-style ids, all unique", () => {
    const ids = enumerateStyleIds();
    expect(ids).toHaveLength(286);
    expect(new Set(ids).size).toBe(286);

    const groupIds = STYLE_GROUPS.map((g) => g.id);
    expect(groupIds).toHaveLength(8);
    for (const gid of groupIds) {
      expect(ids).toContain(gid);
    }

    const subIds = STYLE_GROUPS.flatMap((g) => g.styles.map((s) => s.id));
    expect(subIds).toHaveLength(278);
    for (const sid of subIds) {
      expect(ids).toContain(sid);
    }
  });
});

describe("composeThumbnailPrompt", () => {
  it("matches composeStyledPrompt for a SUB-STYLE id and wraps the reference prompt", () => {
    const id = "ghibli-style";
    const expected = composeStyledPrompt({ styleText: styleTextForId(id), userPrompt: REF });
    const actual = composeThumbnailPrompt(id, REF);
    expect(actual).toBe(expected);
    expect(actual.startsWith("Style: ")).toBe(true);
    expect(actual).toContain(`\nDescription: ${REF}`);
  });

  it("matches composeStyledPrompt for a GROUP id (resolves the group description)", () => {
    const id = "anime-style"; // a group-level id
    expect(STYLE_GROUPS.some((g) => g.id === id)).toBe(true);
    const expected = composeStyledPrompt({ styleText: styleTextForId(id), userPrompt: REF });
    const actual = composeThumbnailPrompt(id, REF);
    expect(actual).toBe(expected);
    expect(actual.startsWith("Style: ")).toBe(true);
    expect(actual).toContain(`\nDescription: ${REF}`);
  });
});

describe("buildJobPayload", () => {
  it("includes count:1, the fixed seed, width+height=size, model, projectId, and composed prompt", () => {
    const prompt = composeThumbnailPrompt("ghibli-style", REF);
    const payload = buildJobPayload({
      id: "ghibli-style",
      prompt,
      model: "z_image_turbo",
      seed: 20240719,
      size: 1024,
      projectId: "proj-123",
    });
    expect(payload.count).toBe(1);
    expect(payload.seed).toBe(20240719);
    expect(payload.width).toBe(1024);
    expect(payload.height).toBe(1024);
    expect(payload.model).toBe("z_image_turbo");
    expect(payload.projectId).toBe("proj-123");
    expect(payload.prompt).toBe(prompt);
  });

  // sc-13135 blocker: the prompt is composed CLIENT-SIDE, so the payload must NOT carry a
  // `styleId` (a recognized rust-api DTO field, sc-13134) AND must set
  // presetPromptResolvedClientSide:true. If both are not enforced, the server folds styleId a
  // second time over the already-styled prompt and DOUBLES the style. This test fails if anyone
  // re-adds styleId or drops the flag.
  it("never sends styleId and always sets presetPromptResolvedClientSide:true", () => {
    const payload = buildJobPayload({
      id: "ghibli-style",
      prompt: composeThumbnailPrompt("ghibli-style", REF),
      model: "z_image_turbo",
      seed: 20240719,
      size: 1024,
      projectId: "proj-123",
    });
    expect(Object.prototype.hasOwnProperty.call(payload, "styleId")).toBe(false);
    expect(payload.styleId).toBeUndefined();
    expect(payload.presetPromptResolvedClientSide).toBe(true);
  });
});

describe("thumbnailPath", () => {
  it("maps id → <out>/<id>.png", () => {
    expect(thumbnailPath("/tmp/thumbs", "ghibli-style")).toBe("/tmp/thumbs/ghibli-style.png");
  });
});

describe("shouldSkip", () => {
  it("never skips when force is set", () => {
    expect(shouldSkip("/no/such/file.png", true)).toBe(false);
  });
  it("does not skip a non-existent thumb when force is off", () => {
    expect(shouldSkip("/no/such/file.png", false)).toBe(false);
  });
});

describe("parseArgs", () => {
  it("applies defaults when no flags are passed", () => {
    const opts = parseArgs([]);
    expect(opts.prompt).toBe("a red fox in a snowy forest");
    expect(opts.model).toBe("z_image_turbo");
    expect(opts.seed).toBe(20240719);
    expect(opts.size).toBe(1024);
    expect(opts.thumb).toBe(128);
    expect(opts.base).toBe("http://127.0.0.1:8000");
    expect(opts.concurrency).toBe(1);
    expect(opts.dryRun).toBe(false);
    expect(opts.force).toBe(false);
    expect(opts.project).toBeNull();
    expect(opts.limit).toBeNull();
  });

  it("respects overrides including --prompt", () => {
    const opts = parseArgs([
      "--prompt",
      "a lighthouse at dusk",
      "--model",
      "krea_2_turbo",
      "--seed",
      "42",
      "--size",
      "512",
      "--thumb",
      "64",
      "--project",
      "proj-9",
      "--limit",
      "3",
      "--dry-run",
      "--force",
    ]);
    expect(opts.prompt).toBe("a lighthouse at dusk");
    expect(opts.model).toBe("krea_2_turbo");
    expect(opts.seed).toBe(42);
    expect(opts.size).toBe(512);
    expect(opts.thumb).toBe(64);
    expect(opts.project).toBe("proj-9");
    expect(opts.limit).toBe(3);
    expect(opts.dryRun).toBe(true);
    expect(opts.force).toBe(true);
  });

  it("throws on an unknown flag", () => {
    expect(() => parseArgs(["--bogus"])).toThrow(/Unknown flag/);
  });
});

describe("parseArgs base-URL resolution", () => {
  // Empty env is injected so these assertions never depend on the real process.env.
  const EMPTY = {};

  it("defaults to http://127.0.0.1:8000 with no flags and empty env", () => {
    expect(parseArgs([], EMPTY).base).toBe("http://127.0.0.1:8000");
  });

  it("--port composes onto the default host", () => {
    expect(parseArgs(["--port", "8787"], EMPTY).base).toBe("http://127.0.0.1:8787");
  });

  it("--host + --port compose the full URL", () => {
    expect(parseArgs(["--host", "192.168.1.50", "--port", "8787"], EMPTY).base).toBe(
      "http://192.168.1.50:8787",
    );
  });

  it("uses SCENEWORKS_API_PORT from injected env when no --port flag", () => {
    expect(parseArgs([], { SCENEWORKS_API_PORT: "8787" }).base).toBe("http://127.0.0.1:8787");
  });

  it("uses SCENEWORKS_API_HOST from injected env when no --host flag", () => {
    expect(parseArgs([], { SCENEWORKS_API_HOST: "10.0.0.9" }).base).toBe("http://10.0.0.9:8000");
  });

  it("--port flag overrides SCENEWORKS_API_PORT from env", () => {
    expect(parseArgs(["--port", "9001"], { SCENEWORKS_API_PORT: "8787" }).base).toBe(
      "http://127.0.0.1:9001",
    );
  });

  it("--base overrides host, port, and env entirely", () => {
    expect(
      parseArgs(["--base", "http://lan.example:5555", "--host", "192.168.1.50", "--port", "8787"], {
        SCENEWORKS_API_PORT: "8787",
        SCENEWORKS_API_HOST: "10.0.0.9",
      }).base,
    ).toBe("http://lan.example:5555");
  });

  it("trims a trailing slash from --base so URL joins don't double-slash", () => {
    expect(parseArgs(["--base", "http://127.0.0.1:8000/"], EMPTY).base).toBe("http://127.0.0.1:8000");
  });

  it("throws on a non-numeric --port", () => {
    expect(() => parseArgs(["--port", "abc"], EMPTY)).toThrow(/positive integer port/);
  });

  it("throws on a zero / negative --port", () => {
    expect(() => parseArgs(["--port", "0"], EMPTY)).toThrow(/positive integer port/);
    expect(() => parseArgs(["--port", "-5"], EMPTY)).toThrow(/positive integer port/);
  });
});
