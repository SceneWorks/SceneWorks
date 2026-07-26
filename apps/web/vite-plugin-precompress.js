import {
  brotliCompressSync,
  constants as zlibConstants,
  gzipSync,
} from "node:zlib";
import { Buffer } from "node:buffer";
import {
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { extname, join, relative, resolve } from "node:path";

const COMPRESSIBLE_EXTENSIONS = new Set([
  ".css",
  ".html",
  ".js",
  ".json",
  ".map",
  ".md",
  ".svg",
  ".txt",
  ".wasm",
  ".xml",
]);

// Tiny responses cost more in headers and decompression setup than they save.
export const MINIMUM_PRECOMPRESS_BYTES = 256;

export function shouldPrecompress(filePath, byteLength) {
  return (
    byteLength >= MINIMUM_PRECOMPRESS_BYTES
    && COMPRESSIBLE_EXTENSIONS.has(extname(filePath).toLowerCase())
  );
}

export function compressRepresentations(source) {
  const input = Buffer.isBuffer(source) ? source : Buffer.from(source);
  return {
    br: brotliCompressSync(input, {
      params: {
        [zlibConstants.BROTLI_PARAM_QUALITY]: 9,
      },
    }),
    gzip: gzipSync(input, { level: 9, mtime: 0 }),
  };
}

function filesUnder(directory) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...filesUnder(path));
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files;
}

export function precompressDirectory(directory) {
  const root = resolve(directory);
  const results = [];
  for (const filePath of filesUnder(root)) {
    if (filePath.endsWith(".br") || filePath.endsWith(".gz")) {
      continue;
    }
    const size = statSync(filePath).size;
    if (!shouldPrecompress(filePath, size)) {
      continue;
    }
    const source = readFileSync(filePath);
    const encoded = compressRepresentations(source);
    writeFileSync(`${filePath}.br`, encoded.br);
    writeFileSync(`${filePath}.gz`, encoded.gzip);
    results.push({
      path: relative(root, filePath).replaceAll("\\", "/"),
      identityBytes: source.length,
      brBytes: encoded.br.length,
      gzipBytes: encoded.gzip.length,
    });
  }
  return results;
}

/**
 * Generate Brotli and gzip siblings after Vite has written the complete bundle,
 * including assets emitted by other plugins such as /theme-init.js.
 *
 * Compression is paid once at build time. The Rust embedded-web handler only
 * selects a representation, so requests never spend CPU recompressing bytes.
 *
 * @returns {import("vite").Plugin}
 */
export default function precompressPlugin() {
  let outputDirectory = "dist";
  return {
    name: "sceneworks-precompress",
    apply: "build",
    configResolved(config) {
      outputDirectory = resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      precompressDirectory(outputDirectory);
    },
  };
}
