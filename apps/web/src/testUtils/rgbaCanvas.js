// A canvas fake that actually holds pixels, plus a real 8-bit PNG codec (sc-24111).
//
// Why this exists: `npm test` runs vitest in jsdom, where `HTMLCanvasElement.getContext` throws.
// The established workaround in this repo (`useMaskTool.test.jsx`) is a fake whose `getImageData`
// hands back a zeroed buffer and whose `toBlob` hands back an empty `Blob` — fine for asserting
// that a call happened, useless for asserting what happened to the alpha channel. Every claim in
// the transparency story ("the exported PNG still has the same transparent pixels") is a claim
// about bytes, so the fake here keeps a real RGBA buffer and the codec below writes and reads a
// real PNG with `node:zlib`.
//
// The alternative was adding `canvas` (node-canvas) as a dev dependency. That is a native build on
// three CI lanes for a handful of assertions, and it would not test the production compositing any
// harder than driving it over a real `Uint8ClampedArray` does.
//
// Scope, stated so nobody mistakes this for a canvas implementation: 8-bit colour, `source-over`
// and `destination-out` composition, `globalAlpha`, axis-aligned `drawImage` with optional
// destination size, `getImageData` / `putImageData`, `fillRect`. No transforms, no paths, no
// interpolation beyond nearest-neighbour. That is what the export paths under test use.

import { Buffer } from "node:buffer";
import { deflateSync, inflateSync } from "node:zlib";

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc ^= bytes[index];
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function readUint32(bytes, offset) {
  return (
    ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0
  );
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

/**
 * Decode an 8-bit RGB or RGBA PNG into `{ width, height, data }`, where `data` is always RGBA.
 * Supports the five standard row filters, which is what any real encoder (PIL, the `image` crate,
 * a browser) emits. Interlaced, paletted and 16-bit files are refused loudly rather than guessed
 * at — no fixture in this repo is one, and a silent wrong answer here would look like a bug in the
 * code under test.
 */
export function decodePng(bytes) {
  const input = Uint8Array.from(bytes);
  for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
    if (input[index] !== PNG_SIGNATURE[index]) {
      throw new Error("not a PNG");
    }
  }
  let offset = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat = [];
  while (offset < input.length) {
    const length = readUint32(input, offset);
    const type = String.fromCharCode(
      input[offset + 4],
      input[offset + 5],
      input[offset + 6],
      input[offset + 7],
    );
    const body = input.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = readUint32(body, 0);
      height = readUint32(body, 4);
      const depth = body[8];
      const colorType = body[9];
      if (depth !== 8) throw new Error(`unsupported PNG bit depth ${depth}`);
      if (body[12] !== 0) throw new Error("interlaced PNGs are not supported here");
      if (colorType === 2) channels = 3;
      else if (colorType === 6) channels = 4;
      else throw new Error(`unsupported PNG colour type ${colorType}`);
    } else if (type === "IDAT") {
      idat.push(Buffer.from(body));
    } else if (type === "IEND") {
      break;
    }
    offset += 12 + length;
  }
  const raw = new Uint8Array(inflateSync(Buffer.concat(idat)));
  const stride = width * channels;
  const out = new Uint8ClampedArray(width * height * 4);
  const previous = new Uint8Array(stride);
  const current = new Uint8Array(stride);
  let cursor = 0;
  for (let row = 0; row < height; row += 1) {
    const filter = raw[cursor];
    cursor += 1;
    current.set(raw.subarray(cursor, cursor + stride));
    cursor += stride;
    for (let index = 0; index < stride; index += 1) {
      const left = index >= channels ? current[index - channels] : 0;
      const up = previous[index];
      const upLeft = index >= channels ? previous[index - channels] : 0;
      let value = current[index];
      if (filter === 1) value += left;
      else if (filter === 2) value += up;
      else if (filter === 3) value += (left + up) >> 1;
      else if (filter === 4) value += paeth(left, up, upLeft);
      else if (filter !== 0) throw new Error(`unknown PNG row filter ${filter}`);
      current[index] = value & 0xff;
    }
    for (let x = 0; x < width; x += 1) {
      const from = x * channels;
      const to = (row * width + x) * 4;
      out[to] = current[from];
      out[to + 1] = current[from + 1];
      out[to + 2] = current[from + 2];
      out[to + 3] = channels === 4 ? current[from + 3] : 255;
    }
    previous.set(current);
  }
  return { width, height, data: out };
}

/** Encode an RGBA buffer as a PNG. Filter 0 throughout — correctness, not file size. */
export function encodePng(width, height, data) {
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let row = 0; row < height; row += 1) {
    raw[row * (stride + 1)] = 0;
    Buffer.from(data.buffer ?? data, data.byteOffset ?? 0, data.length).copy(
      raw,
      row * (stride + 1) + 1,
      row * stride,
      row * stride + stride,
    );
  }
  const chunk = (type, body) => {
    const header = Buffer.alloc(8);
    header.writeUInt32BE(body.length, 0);
    header.write(type, 4, "ascii");
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([header.subarray(4), body])), 0);
    return Buffer.concat([header, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from(PNG_SIGNATURE),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** How many pixels sit at each alpha value. The shape every hop has to preserve. */
export function alphaHistogram(data) {
  const histogram = new Map();
  for (let index = 3; index < data.length; index += 4) {
    histogram.set(data[index], (histogram.get(data[index]) ?? 0) + 1);
  }
  return histogram;
}

export function transparentPixelCount(data) {
  return alphaHistogram(data).get(0) ?? 0;
}

/**
 * A stand-in for the `HTMLImageElement` the editor draws from: carries its own pixels so
 * `drawImage` has something real to copy.
 */
export function fakeImage(width, height, data) {
  return { width, height, naturalWidth: width, naturalHeight: height, __pixels: data };
}

/** The same, built straight from PNG bytes (what `blobToImage` produces in production). */
export function fakeImageFromPng(bytes) {
  const { width, height, data } = decodePng(bytes);
  return fakeImage(width, height, data);
}

function makeContext(canvas, options) {
  // The single most important line in the file: the editor's export paths must never ask for an
  // opaque context, because one composites every transparent pixel onto black. Recorded rather
  // than tolerated so a test can assert on it.
  canvas.contextOptions.push(options ?? null);
  const state = { globalAlpha: 1, globalCompositeOperation: "source-over", fillStyle: "#000000" };
  // Axis-aligned transform only — translate and scale, with a save/restore stack, which is the
  // whole of what `compositeLayersToCanvas` and `drawLayerIntoCrop` use for an unrotated stack.
  // `rotate` throws rather than silently drawing in the wrong place.
  let transform = { tx: 0, ty: 0, sx: 1, sy: 1 };
  const stack = [];
  const blend = (dx, dy, source) => {
    const alpha = state.globalAlpha;
    const erase = state.globalCompositeOperation === "destination-out";
    const index = (dy * canvas.width + dx) * 4;
    if (dx < 0 || dy < 0 || dx >= canvas.width || dy >= canvas.height) return;
    const sa = (source[3] / 255) * alpha;
    if (erase) {
      canvas.data[index + 3] = Math.round(canvas.data[index + 3] * (1 - sa));
      return;
    }
    const da = canvas.data[index + 3] / 255;
    const out = sa + da * (1 - sa);
    for (let channel = 0; channel < 3; channel += 1) {
      const value =
        out === 0
          ? 0
          : (source[channel] * sa + canvas.data[index + channel] * da * (1 - sa)) / out;
      canvas.data[index + channel] = Math.round(value);
    }
    canvas.data[index + 3] = Math.round(out * 255);
  };
  return {
    canvas,
    get globalAlpha() {
      return state.globalAlpha;
    },
    set globalAlpha(value) {
      state.globalAlpha = value;
    },
    get globalCompositeOperation() {
      return state.globalCompositeOperation;
    },
    set globalCompositeOperation(value) {
      state.globalCompositeOperation = value;
    },
    get fillStyle() {
      return state.fillStyle;
    },
    set fillStyle(value) {
      state.fillStyle = value;
      canvas.fills.push(value);
    },
    fillRect(x, y, width, height) {
      canvas.fillRects.push({ x, y, width, height, style: state.fillStyle });
      const match = /^#([0-9a-f]{6})$/i.exec(String(state.fillStyle));
      const rgb = match
        ? [
            parseInt(match[1].slice(0, 2), 16),
            parseInt(match[1].slice(2, 4), 16),
            parseInt(match[1].slice(4, 6), 16),
          ]
        : [0, 0, 0];
      for (let row = 0; row < height; row += 1) {
        for (let column = 0; column < width; column += 1) {
          blend(x + column, y + row, [rgb[0], rgb[1], rgb[2], 255]);
        }
      }
    },
    drawImage(image, ...args) {
      const pixels = image.__pixels ?? image.canvas?.data ?? image.data;
      const sourceWidth = image.width ?? image.naturalWidth;
      const sourceHeight = image.height ?? image.naturalHeight;
      if (!pixels) {
        throw new Error("drawImage source carries no pixels");
      }
      // Only the shapes the editor uses: (image, dx, dy) and (image, dx, dy, dw, dh).
      const [rawX = 0, rawY = 0, rawW = sourceWidth, rawH = sourceHeight] = args;
      const dx = transform.tx + rawX * transform.sx;
      const dy = transform.ty + rawY * transform.sy;
      const dw = Math.round(rawW * transform.sx);
      const dh = Math.round(rawH * transform.sy);
      for (let row = 0; row < dh; row += 1) {
        for (let column = 0; column < dw; column += 1) {
          const sx = Math.min(sourceWidth - 1, Math.floor((column * sourceWidth) / dw));
          const sy = Math.min(sourceHeight - 1, Math.floor((row * sourceHeight) / dh));
          const from = (sy * sourceWidth + sx) * 4;
          blend(
            Math.round(dx) + column,
            Math.round(dy) + row,
            [pixels[from], pixels[from + 1], pixels[from + 2], pixels[from + 3]],
          );
        }
      }
    },
    translate(x, y) {
      transform = {
        ...transform,
        tx: transform.tx + x * transform.sx,
        ty: transform.ty + y * transform.sy,
      };
    },
    scale(x, y) {
      transform = { ...transform, sx: transform.sx * x, sy: transform.sy * y };
    },
    rotate(radians) {
      if (radians !== 0) {
        throw new Error("rgbaCanvas does not model rotation; test an unrotated stack");
      }
    },
    getImageData(x, y, width, height) {
      const out = new Uint8ClampedArray(width * height * 4);
      for (let row = 0; row < height; row += 1) {
        for (let column = 0; column < width; column += 1) {
          const from = ((y + row) * canvas.width + (x + column)) * 4;
          const to = (row * width + column) * 4;
          out.set(canvas.data.subarray(from, from + 4), to);
        }
      }
      return { width, height, data: out };
    },
    putImageData(imageData, x, y) {
      for (let row = 0; row < imageData.height; row += 1) {
        for (let column = 0; column < imageData.width; column += 1) {
          const from = (row * imageData.width + column) * 4;
          const to = ((y + row) * canvas.width + (x + column)) * 4;
          canvas.data.set(imageData.data.subarray(from, from + 4), to);
        }
      }
    },
    save() {
      stack.push({ ...transform, ...state });
    },
    restore() {
      const saved = stack.pop();
      if (!saved) return;
      transform = { tx: saved.tx, ty: saved.ty, sx: saved.sx, sy: saved.sy };
      state.globalAlpha = saved.globalAlpha;
      state.globalCompositeOperation = saved.globalCompositeOperation;
      state.fillStyle = saved.fillStyle;
    },
    clearRect(x, y, width, height) {
      for (let row = 0; row < height; row += 1) {
        for (let column = 0; column < width; column += 1) {
          const index = ((y + row) * canvas.width + (x + column)) * 4;
          canvas.data.fill(0, index, index + 4);
        }
      }
    },
    beginPath() {},
    closePath() {},
    moveTo() {},
    lineTo() {},
    arc() {},
    fill() {},
    stroke() {},
  };
}

/** A canvas whose pixels are real and whose `toBlob` writes a real PNG. */
export function makeCanvas(width = 0, height = 0) {
  const canvas = {
    width,
    height,
    contextOptions: [],
    fills: [],
    fillRects: [],
    toBlobMimes: [],
  };
  let backing = new Uint8ClampedArray(Math.max(0, width * height * 4));
  Object.defineProperty(canvas, "data", {
    get: () => backing,
    set: (value) => {
      backing = value;
    },
  });
  // The editor sets `canvas.width`/`canvas.height` after creating the element, which in a real
  // canvas allocates (and clears) the backing store. Mirror that, or every draw lands out of
  // bounds on a zero-sized buffer.
  for (const dimension of ["width", "height"]) {
    let value = canvas[dimension];
    Object.defineProperty(canvas, dimension, {
      get: () => value,
      set: (next) => {
        value = next;
        backing = new Uint8ClampedArray(Math.max(0, canvas.width * canvas.height * 4));
      },
      configurable: true,
    });
  }
  canvas.getContext = (kind, options) => {
    if (kind !== "2d") return null;
    return makeContext(canvas, options);
  };
  canvas.toBlob = (callback, mime = "image/png") => {
    canvas.toBlobMimes.push(mime);
    const bytes = encodePng(canvas.width, canvas.height, canvas.data);
    const blob = new Blob([bytes], { type: mime });
    // jsdom's `Blob` has no `arrayBuffer()` at the version pinned here, and the point of this fake
    // is the pixels, not re-implementing `Blob`. The bytes ride along so `blobBytes` can read them
    // when the platform method is missing; it still prefers the real one when it exists.
    blob.__bytes = bytes;
    callback(blob);
  };
  canvas.toDataURL = (mime = "image/png") => {
    canvas.toBlobMimes.push(mime);
    return `data:${mime};base64,${encodePng(canvas.width, canvas.height, canvas.data).toString("base64")}`;
  };
  return canvas;
}

/**
 * The bytes of a `Blob` or `File`, whichever way this jsdom will give them up.
 *
 * jsdom at the pinned version has no `Blob.prototype.arrayBuffer`, and the production exporter
 * wraps the canvas blob in `new File([blob], ...)` — which is the point, since that File is what
 * Save uploads and Download hands to the anchor — so a side-channel on the original blob does not
 * survive. `FileReader` does, and it reads the real bytes of the real File.
 */
export async function blobBytes(blob) {
  if (typeof blob.arrayBuffer === "function") {
    return new Uint8Array(await blob.arrayBuffer());
  }
  if (typeof FileReader === "function") {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(new Uint8Array(reader.result));
      reader.onerror = () => reject(reader.error ?? new Error("could not read the blob"));
      reader.readAsArrayBuffer(blob);
    });
  }
  if (blob.__bytes) {
    return new Uint8Array(blob.__bytes);
  }
  throw new Error("blob carries no readable bytes");
}
