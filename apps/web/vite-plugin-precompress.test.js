import { gunzipSync, brotliDecompressSync } from "node:zlib";
import { Buffer } from "node:buffer";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  MINIMUM_PRECOMPRESS_BYTES,
  compressRepresentations,
  precompressDirectory,
  shouldPrecompress,
} from "./vite-plugin-precompress.js";

describe("production precompression", () => {
  it("round-trips Brotli and gzip representations", () => {
    const source = Buffer.from("SceneWorks production JavaScript\n".repeat(200));
    const encoded = compressRepresentations(source);

    expect(brotliDecompressSync(encoded.br)).toEqual(source);
    expect(gunzipSync(encoded.gzip)).toEqual(source);
    expect(encoded.br.length).toBeLessThan(source.length);
    expect(encoded.gzip.length).toBeLessThan(source.length);
  });

  it("limits work to compressible, non-trivial response types", () => {
    expect(shouldPrecompress("assets/index-ABC12345.js", MINIMUM_PRECOMPRESS_BYTES)).toBe(true);
    expect(shouldPrecompress("index.html", MINIMUM_PRECOMPRESS_BYTES)).toBe(true);
    expect(shouldPrecompress("assets/poster-ABC12345.jpg", 50_000)).toBe(false);
    expect(shouldPrecompress("assets/font-ABC12345.woff2", 50_000)).toBe(false);
    expect(shouldPrecompress("assets/tiny-ABC12345.js", 20)).toBe(false);
  });

  it("writes variants without recompressing media or generated variants", () => {
    const directory = mkdtempSync(join(tmpdir(), "sceneworks-precompress-"));
    try {
      mkdirSync(join(directory, "assets"), { recursive: true });
      const javascript = Buffer.from("export const scene = true;\n".repeat(200));
      const media = Buffer.alloc(2048, 7);
      writeFileSync(join(directory, "assets", "index-ABC12345.js"), javascript);
      writeFileSync(join(directory, "assets", "poster-ABC12345.jpg"), media);

      const results = precompressDirectory(directory);

      expect(results.map((result) => result.path)).toEqual(["assets/index-ABC12345.js"]);
      expect(brotliDecompressSync(
        readFileSync(join(directory, "assets", "index-ABC12345.js.br")),
      )).toEqual(javascript);
      expect(gunzipSync(
        readFileSync(join(directory, "assets", "index-ABC12345.js.gz")),
      )).toEqual(javascript);
      expect(() => readFileSync(join(directory, "assets", "poster-ABC12345.jpg.br"))).toThrow();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
