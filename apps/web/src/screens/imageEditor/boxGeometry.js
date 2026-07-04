// Pure box-layout geometry / validation / adapters for the Image Editor's Box tool
// (Workstream A: sc-6089..6095). Extracted verbatim from ImageEditor.jsx (sc-9752, F-052
// follow-up) so the stateful `useBoxesTool` hook can share them without importing back
// into ImageEditor.jsx (which would be a cycle). ImageEditor.jsx re-exports every symbol
// here to keep its public surface — and its test imports — byte-for-byte unchanged.
// Nothing here is stateful or React-aware; it is the same pure code, only relocated.
import { makeObjElement, makeTextElement, normalizeHexColor } from "../../ideogramCaption.js";

// Local clamp (the editor keeps its own copy for crop/view math; this keeps the box
// module self-contained and cycle-free). Identical semantics.
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// ── Box layout (Workstream A, sc-6089) ───────────────────────────────────────
// The colored-box layout tool lets the user draw labeled rectangles that drive
// generation two ways: a structured `bbox` for Ideogram 4 (epic 4725) and a
// color-keyed region prompt for any edit model. A box is a pure data record in
// image-pixel coords:
//   { id, rect:{x,y,width,height}, color:"#RRGGBB", type:"obj"|"text",
//     desc, text? /* type==="text" */, colorPalette?:["#RRGGBB",…] /* ≤5 */ }
// The conversion/validation below is pure (no React/Konva) so the box tool, the
// Ideogram elements adapter (sc-6095), and the color-keyed path (sc-6093/6094)
// all share one source of truth.
export const BOX_TYPES = ["obj", "text"];

// Ideogram's structured-caption palette limits (epic 4725 S3): ≤5 colors per
// element, ≤16 across the whole document.
export const MAX_BOX_PALETTE = 5;
export const MAX_DOCUMENT_PALETTE = 16;

// Uppercase `#RRGGBB` only — the Ideogram S3 contract is case-sensitive, so a
// lowercase value is invalid (the per-box metadata editor, sc-6091, normalizes
// user input to uppercase before storing). Pure.
const HEX_COLOR_RE = /^#[0-9A-F]{6}$/;
export function isValidHexColor(color) {
  return typeof color === "string" && HEX_COLOR_RE.test(color);
}

// Normalize one pixel coordinate to Ideogram's 0–1000 grid (origin top-left),
// rounded to an integer and clamped to the canvas. Guards a zero/absent dim.
function normBboxCoord(px, dim) {
  if (!dim) return 0;
  return clamp(Math.round((px / dim) * 1000), 0, 1000);
}

// rect {x,y,width,height} (image-pixel coords) → `[y_min, x_min, y_max, x_max]`,
// integers normalized 0–1000, origin top-left, clamped to the canvas. Component
// order matches epic 4725 S3 exactly. Robust to flipped (negative-size) rects.
export function rectToBbox(rect, imgW, imgH) {
  const x0 = normBboxCoord(rect.x, imgW);
  const x1 = normBboxCoord(rect.x + rect.width, imgW);
  const y0 = normBboxCoord(rect.y, imgH);
  const y1 = normBboxCoord(rect.y + rect.height, imgH);
  return [Math.min(y0, y1), Math.min(x0, x1), Math.max(y0, y1), Math.max(x0, x1)];
}

// Inverse of `rectToBbox` for round-tripping a stored bbox back onto a canvas of
// the given size. Returns image-pixel coords (unrounded, like `centeredCropRect`);
// the 0–1000 quantization means the round-trip is exact only to grid resolution.
export function bboxToRect([yMin, xMin, yMax, xMax], imgW, imgH) {
  return {
    x: (xMin / 1000) * imgW,
    y: (yMin / 1000) * imgH,
    width: ((xMax - xMin) / 1000) * imgW,
    height: ((yMax - yMin) / 1000) * imgH,
  };
}

// A per-element palette is valid when it is ≤5 uppercase `#RRGGBB` colors. An
// absent palette is valid (it's optional). Pure.
export function boxPaletteIsValid(palette) {
  if (palette == null) return true;
  if (!Array.isArray(palette)) return false;
  return palette.length <= MAX_BOX_PALETTE && palette.every(isValidHexColor);
}

// The document-level palette: the de-duplicated union of every box's per-element
// `colorPalette`, order-preserving (Ideogram key order is quality-relevant, S3). Pure.
export function documentPalette(boxes) {
  const seen = [];
  for (const box of boxes ?? []) {
    for (const color of box?.colorPalette ?? []) {
      if (!seen.includes(color)) seen.push(color);
    }
  }
  return seen;
}

// The document palette must stay ≤16 colors overall (epic 4725 S3). Pure.
export function documentPaletteIsValid(boxes) {
  return documentPalette(boxes).length <= MAX_DOCUMENT_PALETTE;
}

// A box is valid for serialization when it has positive geometry, a known type,
// a non-empty description, and — for text elements — a non-empty literal string.
// Color/palette validity is checked separately (`isValidHexColor`/`boxPaletteIsValid`)
// since the color-keyed path needs only color + desc, not a full Ideogram element. Pure.
export function boxIsValid(box) {
  if (!box || !box.rect) return false;
  if (!(box.rect.width > 0) || !(box.rect.height > 0)) return false;
  if (!BOX_TYPES.includes(box.type)) return false;
  if (typeof box.desc !== "string" || box.desc.trim() === "") return false;
  if (box.type === "text" && (typeof box.text !== "string" || box.text.trim() === "")) return false;
  return true;
}

// ── Box drawing tool (Workstream A, sc-6090) ─────────────────────────────────
// A small palette of distinct, nameable colors for the box tool, plus a custom
// `#RRGGBB`. All entries are uppercase #RRGGBB (valid per `isValidHexColor`) so a
// drawn box is well-formed for the color-keyed path and the Ideogram adapter.
export const BOX_PALETTE = [
  { name: "Red", value: "#FF0000" },
  { name: "Green", value: "#00C853" },
  { name: "Blue", value: "#2962FF" },
  { name: "Yellow", value: "#FFD600" },
  { name: "Orange", value: "#FF6D00" },
  { name: "Purple", value: "#AA00FF" },
  { name: "Cyan", value: "#00B8D4" },
  { name: "Pink", value: "#FF4081" },
];

// Smallest box (image pixels) a drag must cover to commit — a click or tiny
// smudge is discarded rather than creating a degenerate box.
export const MIN_BOX_PX = 8;

// Axis-aligned rect spanning two points (image-pixel coords). Pure — the drag
// direction (up-left vs down-right) is normalized to a positive-size rect.
export function rectFromPoints(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

// Clamp a rect to the canvas, keeping width/height ≥ minPx and the rect fully
// inside [0,imgW]×[0,imgH]. Mirrors the crop tool's clamp but pure (takes dims).
export function clampRectToCanvas(rect, imgW, imgH, minPx = MIN_BOX_PX) {
  const width = clamp(rect.width, minPx, imgW);
  const height = clamp(rect.height, minPx, imgH);
  return {
    width,
    height,
    x: clamp(rect.x, 0, imgW - width),
    y: clamp(rect.y, 0, imgH - height),
  };
}

// Build a new box record (the sc-6089 model) from a drawn rect + color. Metadata
// (type/desc/text/colorPalette) starts at safe defaults; the per-box metadata
// editor (sc-6091) fills it in. `id` is supplied by the caller (session-unique).
export function makeBox(id, rect, color) {
  return { id, rect, color, type: "obj", desc: "", text: "", colorPalette: [] };
}

// A semi-transparent CSS rgba() fill from a `#RRGGBB` color for the box overlay.
// Pure; falls back to a neutral fill if the color isn't a valid 6-digit hex.
export function boxFillStyle(hex, alpha) {
  if (!isValidHexColor(hex)) return `rgba(127,127,127,${alpha})`;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Per-box metadata (Workstream A, sc-6091) ─────────────────────────────────
// Append a color to a per-element palette (uppercased), ignoring duplicates,
// invalid hex, and anything past the ≤5 cap. Pure; returns the same array
// reference when nothing changes so callers can no-op cheaply.
export function addPaletteColor(palette, color, max = MAX_BOX_PALETTE) {
  const list = palette ?? [];
  const value = typeof color === "string" ? color.toUpperCase() : color;
  if (!isValidHexColor(value) || list.includes(value) || list.length >= max) return list;
  return [...list, value];
}

// Remove a color from a per-element palette. Pure; returns a new array.
export function removePaletteColor(palette, color) {
  return (palette ?? []).filter((entry) => entry !== color);
}

// What a box still needs to serialize as a valid Ideogram element (S3): a
// description, the literal text for a text element, and a valid ≤5 palette.
// Returns a human list of what's missing ("" when ready). The color-keyed edit
// path only needs color + desc, so this does NOT gate that path. Pure.
export function boxMetadataGaps(box) {
  if (!box) return [];
  const gaps = [];
  if (typeof box.desc !== "string" || box.desc.trim() === "") gaps.push("a description");
  if (box.type === "text" && (typeof box.text !== "string" || box.text.trim() === "")) gaps.push("the literal text");
  if (!boxPaletteIsValid(box.colorPalette)) gaps.push("a valid color palette (≤5)");
  return gaps;
}

// ── Bake → pass-through edit (Workstream A, sc-6093) ─────────────────────────
// Paint each box as a solid colored rectangle onto a 2D context — the color-keyed
// region signal the edit model reads ("replace the {color} region with …"). The
// caller draws the working image first; this overlays the boxes. Pure given the
// context, so the paint order/coords are unit-testable without a real canvas.
export function paintBoxesOnContext(ctx, boxes) {
  for (const box of boxes ?? []) {
    ctx.fillStyle = box.color;
    ctx.fillRect(box.rect.x, box.rect.y, box.rect.width, box.rect.height);
  }
}

// ── Auto color-prompt (Workstream A, sc-6094) ────────────────────────────────
// Friendly color name for a palette/custom hex — palette colors get their name
// lowercased (#FF0000 → "red"); anything else falls back to the hex itself so the
// prompt still references a concrete color. Pure.
export function colorName(hex) {
  const found = BOX_PALETTE.find((entry) => entry.value === hex);
  return found ? found.name.toLowerCase() : hex;
}

// Compose an editable color-keyed edit prompt from the boxes: one clause per
// described box, referencing it by its visible color so the model maps region →
// element. Boxes missing the needed text (obj → desc; text → literal) are skipped.
// Pure; "" when nothing is describable yet. The user can edit the result freely.
export function composeColorPrompt(boxes) {
  const clauses = [];
  for (const box of boxes ?? []) {
    const name = colorName(box.color);
    if (box.type === "text") {
      const text = (box.text ?? "").trim();
      if (!text) continue;
      const desc = (box.desc ?? "").trim();
      clauses.push(`place the text "${text}" in the ${name} region${desc ? ` (${desc})` : ""}`);
    } else {
      const desc = (box.desc ?? "").trim();
      if (!desc) continue;
      clauses.push(`replace the ${name} region with ${desc}`);
    }
  }
  if (!clauses.length) return "";
  return `${clauses.map((clause) => clause.charAt(0).toUpperCase() + clause.slice(1)).join(". ")}.`;
}

// ── Boxes → Ideogram elements[] adapter (Workstream A, sc-6095) ──────────────
// Convert the editor's boxes into Ideogram 4 structured-caption `elements[]`
// (epic 4725 S3 contract), one element per box, via ideogramCaption.js's factories
// so the canonical key order is guaranteed (obj: type,bbox,desc,color_palette;
// text: type,bbox,text,desc,color_palette). bbox is the 0–1000 grid from
// `rectToBbox`; palette entries are normalized to uppercase #RRGGBB and dropped if
// empty/invalid (an empty palette is omitted entirely). Pure — this supplies only
// the spatial elements; the non-spatial caption fields are epic 4725's (S3/S4/S7).
export function boxesToIdeogramElements(boxes, imgW, imgH) {
  return (boxes ?? []).map((box) => {
    const bbox = rectToBbox(box.rect, imgW, imgH);
    const palette = (box.colorPalette ?? []).map(normalizeHexColor).filter(Boolean);
    const color_palette = palette.length ? palette : null;
    if (box.type === "text") {
      return makeTextElement({ bbox, text: box.text ?? "", desc: box.desc ?? "", color_palette });
    }
    return makeObjElement({ bbox, desc: box.desc ?? "", color_palette });
  });
}
