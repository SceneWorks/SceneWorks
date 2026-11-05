import React from "react";

export const UPSCALE_ENGINE_DESC = {
  "real-esrgan": "Fast, faithful general-purpose upscaler. Great default.",
  seedvr2: "Detail-restoring diffusion upscaler for degraded sources.",
  iris: "Iris-3B restoration: any scale (1× restores only), tiled.",
  "aura-sr": "Sharpest output, best for print. Slower.",
};

export const MIN_SCALE = 0.05;

export const MAX_SCALE = 16;

export const ZOOM_STEP = 1.2;

export const MIN_CROP_PX = 8;

// Redesign panel layout (epic 10243): accordion (default) / right / left / bottom,
// persisted across sessions. Invalid/absent → accordion.
export const EDITOR_LAYOUTS = ["accordion", "right", "left", "bottom"];

export const EDITOR_LAYOUT_KEY = "sceneworks-ie-layout";

export function readStoredEditorLayout() {
  try {
    const saved = window.localStorage.getItem(EDITOR_LAYOUT_KEY);
    if (EDITOR_LAYOUTS.includes(saved)) return saved;
  } catch {
    /* ignore (private mode etc.) */
  }
  return "accordion";
}

// Tool identity for the rail / accordion headers / inspector header (epic 10243).
export const EDITOR_TOOL_ORDER = ["move", "transform", "crop", "upscale", "detail", "color", "cutout", "edit", "boxes", "depth"];

export const EDITOR_TOOL_META = {
  move: { label: "Move", desc: "Pan and inspect the canvas" },
  transform: { label: "Transform", desc: "Move, scale & rotate the layer" },
  crop: { label: "Crop", desc: "Trim to a size or ratio" },
  upscale: { label: "Upscale", desc: "Increase resolution with AI" },
  detail: { label: "Detail", desc: "Refine texture with tile ControlNet" },
  color: { label: "Color grade", desc: "Adjust tone, levels & curves" },
  cutout: { label: "Cutout", desc: "Remove a color-keyed background" },
  edit: { label: "AI Edit", desc: "Prompt-driven edit & inpaint" },
  boxes: { label: "Boxes", desc: "Region layout & color-keyed edit" },
  depth: { label: "Depth map", desc: "Estimate relative depth (Iris-3B)" },
};

// Inline stroke icons ported from the design handoff `ICONS` map (epic 10243).
export const strokeSvg = (props, children) => (
  <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.9} viewBox="0 0 24 24" {...props}>
    {children}
  </svg>
);

export const EDITOR_TOOL_ICONS = {
  move: strokeSvg({ key: "move" }, [
    <polyline key="a" points="5 9 2 12 5 15" />, <polyline key="b" points="9 5 12 2 15 5" />,
    <polyline key="c" points="15 19 12 22 9 19" />, <polyline key="d" points="19 9 22 12 19 15" />,
    <line key="e" x1="2" x2="22" y1="12" y2="12" />, <line key="f" x1="12" x2="12" y1="2" y2="22" />,
  ]),
  transform: strokeSvg({ key: "transform" }, [
    <rect key="a" height="16" rx="1" width="16" x="4" y="4" />,
    <circle key="b" cx="4" cy="4" fill="currentColor" r="2" />, <circle key="c" cx="20" cy="4" fill="currentColor" r="2" />,
    <circle key="d" cx="4" cy="20" fill="currentColor" r="2" />, <circle key="e" cx="20" cy="20" fill="currentColor" r="2" />,
  ]),
  crop: strokeSvg({ key: "crop" }, [
    <path key="a" d="M6 2v14a2 2 0 0 0 2 2h14" />, <path key="b" d="M18 22V8a2 2 0 0 0-2-2H2" />,
  ]),
  upscale: strokeSvg({ key: "upscale" }, [
    <polyline key="a" points="15 3 21 3 21 9" />, <polyline key="b" points="9 21 3 21 3 15" />,
    <line key="c" x1="21" x2="14" y1="3" y2="10" />, <line key="d" x1="3" x2="10" y1="21" y2="14" />,
  ]),
  detail: strokeSvg({ key: "detail" }, [
    <path key="a" d="M12 3l1.9 4.8L19 9.5l-4.1 2.9L16 18l-4-2.7L8 18l1.1-5.6L5 9.5l5.1-1.7z" />,
  ]),
  color: strokeSvg({ key: "color" }, [
    <circle key="a" cx="12" cy="12" r="9" />, <path key="b" d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none" />,
  ]),
  cutout: strokeSvg({ key: "cutout" }, [
    <path key="a" d="M4 5h16v14H4z" />, <path key="b" d="M7 15l3-4 2 2 2-3 3 5" />,
    <circle key="c" cx="9" cy="9" r="1.2" />,
  ]),
  edit: strokeSvg({ key: "edit" }, [
    <path key="a" d="M15 4l5 5" />, <path key="b" d="M4 20l4-1 10-10-3-3L5 16z" />,
    <path key="c" d="M14 5l1.5-1.5a2 2 0 0 1 3 3L17 8" />,
  ]),
  boxes: strokeSvg({ key: "boxes" }, [
    <rect key="a" height="7" rx="1" width="8" x="3" y="4" />, <rect key="b" height="12" rx="1" width="8" x="13" y="4" />,
    <rect key="c" height="6" rx="1" width="8" x="3" y="14" />,
  ]),
  depth: strokeSvg({ key: "depth" }, [
    <path key="a" d="M3 20l6-9 4 5 3-4 5 8z" />, <path key="b" d="M3 20h18" />,
  ]),
};

export const IeChevron = () => (
  <svg fill="none" height="15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.2} viewBox="0 0 24 24" width="15">
    <path d="M6 9l6 6 6-6" />
  </svg>
);

export const IeEyeOpen = () => (
  <svg fill="none" height="15" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" width="15">
    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

export const IeEyeOff = () => (
  <svg className="ie-vis-off" fill="none" height="15" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" width="15">
    <path d="M17.9 17.9A10.4 10.4 0 0 1 12 19C5 19 1 12 1 12a19 19 0 0 1 5.1-5.9M9.9 4.2A10.9 10.9 0 0 1 12 4c7 0 11 7 11 7a19 19 0 0 1-2.2 3.2M1 1l22 22" />
  </svg>
);

// Modifier glyph for the shortcut reference (the handler accepts both ⌘ and Ctrl).
export const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.platform || navigator.userAgent || "");

export const MOD_KEY = IS_MAC ? "⌘" : "Ctrl";

// Keyboard shortcut reference (sc-6111). Single source of truth for the in-editor
// quick reference; the keydown handler implements these exact bindings.
export const EDITOR_SHORTCUTS = [
  {
    group: "Tools",
    items: [
      { keys: ["M"], label: "Move / pan" },
      { keys: ["T"], label: "Transform layer" },
      { keys: ["C"], label: "Crop" },
      { keys: ["U"], label: "Upscale" },
      { keys: ["D"], label: "Detail enhance" },
      { keys: ["G"], label: "Color grade" },
      { keys: ["E"], label: "AI edit" },
      { keys: ["B"], label: "Boxes" },
    ],
  },
  {
    group: "View",
    items: [
      { keys: ["+"], label: "Zoom in" },
      { keys: ["−"], label: "Zoom out" },
      { keys: ["0"], label: "Fit to view" },
      { keys: ["1"], label: "Actual size (100%)" },
    ],
  },
  {
    group: "Edit",
    items: [
      { keys: [MOD_KEY, "Z"], label: "Undo" },
      { keys: ["⇧", MOD_KEY, "Z"], label: "Redo" },
      { keys: ["Delete"], label: "Delete selected box" },
      { keys: ["Esc"], label: "Cancel / deselect" },
      { keys: ["?"], label: "Toggle this help" },
    ],
  },
];
