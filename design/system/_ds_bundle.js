/* @ds-bundle: {"format":4,"namespace":"SceneWorksDesignSystem_b6febf","components":[{"name":"Logo","sourcePath":"components/brand/Logo.jsx"},{"name":"Wordmark","sourcePath":"components/brand/Logo.jsx"},{"name":"Markdown","sourcePath":"components/content/Markdown.jsx"},{"name":"AccentPicker","sourcePath":"components/controls/AccentPicker.jsx"},{"name":"CompactSelector","sourcePath":"components/controls/CompactSelector.jsx"},{"name":"Button","sourcePath":"components/core/Button.jsx"},{"name":"Icon","sourcePath":"components/core/Icon.jsx"},{"name":"Modal","sourcePath":"components/feedback/Modal.jsx"},{"name":"StatusDot","sourcePath":"components/feedback/StatusDot.jsx"},{"name":"ACCENTS","sourcePath":"theme/accents.js"},{"name":"DEFAULT_ACCENT","sourcePath":"theme/accents.js"}],"sourceHashes":{"components/brand/Logo.jsx":"9cb786693702","components/content/Markdown.jsx":"31490d8e2bd4","components/controls/AccentPicker.jsx":"7d2025cf5359","components/controls/CompactSelector.jsx":"a0cea8320f8f","components/core/Button.jsx":"6f732f56c9ae","components/core/Icon.jsx":"adcf3425af03","components/feedback/Modal.jsx":"8160841f0aed","components/feedback/StatusDot.jsx":"70211dc67209","theme/accents.js":"5fa93e4c557c","ui_kits/studio/app.jsx":"51f48855459f","ui_kits/studio/imageStudio.jsx":"dfcbcb07d1b0","ui_kits/studio/library.jsx":"facb9a05f34c","ui_kits/studio/modelManager.jsx":"2d56ea5873d3","ui_kits/studio/shell.jsx":"6f8ab776ce78"},"inlinedExternals":[],"unexposedExports":[{"name":"isAccentId","sourcePath":"theme/accents.js"}]} */

(() => {

const __ds_ns = (window.SceneWorksDesignSystem_b6febf = window.SceneWorksDesignSystem_b6febf || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/brand/Logo.jsx
try { (() => {
const {
  useId
} = React;
/**
 * SceneWorks mark — "scene cut". A rounded square split by a diagonal seam: a
 * teal triangle over a solid ground. Colors come from CSS variables
 * (--logo-ground / --logo-seam / --teal) so it tracks the active theme and the
 * user-selectable accent. Verbatim from @sceneworks/ui.
 */
function Logo({
  size = 32,
  title = "SceneWorks",
  className
}) {
  const clipId = useId();
  return /*#__PURE__*/React.createElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 100 100",
    role: "img",
    "aria-label": title,
    className: className
  }, /*#__PURE__*/React.createElement("title", null, title), /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("clipPath", {
    id: clipId
  }, /*#__PURE__*/React.createElement("rect", {
    x: "10",
    y: "10",
    width: "80",
    height: "80",
    rx: "14"
  }))), /*#__PURE__*/React.createElement("g", {
    clipPath: `url(#${clipId})`
  }, /*#__PURE__*/React.createElement("rect", {
    x: "10",
    y: "10",
    width: "80",
    height: "80",
    fill: "var(--logo-ground)"
  }), /*#__PURE__*/React.createElement("polygon", {
    points: "10,10 90,10 90,90",
    fill: "var(--teal)"
  }), /*#__PURE__*/React.createElement("line", {
    x1: "10",
    y1: "90",
    x2: "90",
    y2: "10",
    stroke: "var(--logo-seam)",
    strokeWidth: "2.5",
    strokeLinecap: "square"
  })));
}

/**
 * The full lockup: scene-cut mark + "SceneWorks" wordmark (Works in --mist).
 * Matches the .brand block in the sidebar.
 */
function Wordmark({
  size = 32
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: "brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "brand-mark"
  }, /*#__PURE__*/React.createElement(Logo, {
    size: size
  })), /*#__PURE__*/React.createElement("h1", null, "Scene", /*#__PURE__*/React.createElement("span", {
    className: "light"
  }, "Works")));
}
Object.assign(__ds_scope, { Logo, Wordmark, __ds_default_components_brand_Logo_1q8yv4x: Logo });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/brand/Logo.jsx", error: String((e && e.message) || e) }); }

// components/content/Markdown.jsx
try { (() => {
// Compact, dependency-free Markdown renderer — verbatim from @sceneworks/ui.
// Covers ATX headings, paragraphs, ordered/unordered lists, blockquotes, fenced
// code, and inline bold/italic/code/links. Builds React elements directly (never
// dangerouslySetInnerHTML); link hrefs are restricted to safe schemes.

const HEADING = /^(#{1,6})\s+(.*)$/;
const LIST_ITEM = /^\s*([-*+]|\d+\.)\s+/;
const ORDERED_ITEM = /^\s*\d+\.\s+/;
const FENCE = /^```/;
const QUOTE = /^\s*>\s?/;
function safeHref(url) {
  const trimmed = (url || "").trim();
  if (/^(https?:|mailto:)/i.test(trimmed)) return trimmed;
  if (/^[/#]/.test(trimmed)) return trimmed;
  return undefined;
}
const INLINE_RULES = [{
  re: /`([^`]+)`/,
  node: (m, key) => /*#__PURE__*/React.createElement("code", {
    key: key
  }, m[1])
}, {
  re: /\[([^\]]+)\]\(([^)\s]+)\)/,
  node: (m, key) => {
    const href = safeHref(m[2]);
    if (!href) return /*#__PURE__*/React.createElement("span", {
      key: key
    }, renderInline(m[1], key));
    return /*#__PURE__*/React.createElement("a", {
      key: key,
      href: href,
      target: "_blank",
      rel: "noopener noreferrer"
    }, renderInline(m[1], key));
  }
}, {
  re: /\*\*([^*]+)\*\*/,
  node: (m, key) => /*#__PURE__*/React.createElement("strong", {
    key: key
  }, renderInline(m[1], key))
}, {
  re: /\*([^*]+)\*/,
  node: (m, key) => /*#__PURE__*/React.createElement("em", {
    key: key
  }, renderInline(m[1], key))
}, {
  re: /_([^_]+)_/,
  node: (m, key) => /*#__PURE__*/React.createElement("em", {
    key: key
  }, renderInline(m[1], key))
}];
function renderInline(text, keyPrefix) {
  let earliest = null;
  for (const rule of INLINE_RULES) {
    const match = rule.re.exec(text);
    if (match && (!earliest || match.index < earliest.match.index)) earliest = {
      rule,
      match
    };
  }
  if (!earliest) return text;
  const {
    rule,
    match
  } = earliest;
  const before = text.slice(0, match.index);
  const after = text.slice(match.index + match[0].length);
  const nodes = [];
  if (before) nodes.push(before);
  nodes.push(rule.node(match, `${keyPrefix}-${match.index}`));
  const rest = renderInline(after, `${keyPrefix}-r`);
  if (Array.isArray(rest)) nodes.push(...rest);else if (rest) nodes.push(rest);
  return nodes;
}
function parseBlocks(content) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length,
        text: heading[2].trim()
      });
      i += 1;
      continue;
    }
    if (FENCE.test(line.trim())) {
      i += 1;
      const code = [];
      while (i < lines.length && !FENCE.test(lines[i].trim())) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push({
        type: "code",
        text: code.join("\n")
      });
      continue;
    }
    if (LIST_ITEM.test(line)) {
      const ordered = ORDERED_ITEM.test(line);
      const items = [];
      while (i < lines.length && LIST_ITEM.test(lines[i])) {
        items.push(lines[i].replace(LIST_ITEM, ""));
        i += 1;
      }
      blocks.push({
        type: "list",
        ordered,
        items
      });
      continue;
    }
    if (QUOTE.test(line)) {
      const quote = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quote.push(lines[i].replace(QUOTE, ""));
        i += 1;
      }
      blocks.push({
        type: "quote",
        text: quote.join(" ")
      });
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !HEADING.test(lines[i]) && !LIST_ITEM.test(lines[i]) && !FENCE.test(lines[i].trim()) && !QUOTE.test(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({
      type: "paragraph",
      text: para.join(" ")
    });
  }
  return blocks;
}
function renderBlock(block, key) {
  switch (block.type) {
    case "heading":
      {
        const Tag = `h${Math.min(block.level, 6)}`;
        return /*#__PURE__*/React.createElement(Tag, {
          key: key
        }, renderInline(block.text, key));
      }
    case "list":
      {
        const Tag = block.ordered ? "ol" : "ul";
        return /*#__PURE__*/React.createElement(Tag, {
          key: key
        }, block.items.map((item, index) => /*#__PURE__*/React.createElement("li", {
          key: `${key}-${index}`
        }, renderInline(item, `${key}-${index}`))));
      }
    case "quote":
      return /*#__PURE__*/React.createElement("blockquote", {
        key: key
      }, /*#__PURE__*/React.createElement("p", null, renderInline(block.text, key)));
    case "code":
      return /*#__PURE__*/React.createElement("pre", {
        key: key
      }, /*#__PURE__*/React.createElement("code", null, block.text));
    case "paragraph":
    default:
      return /*#__PURE__*/React.createElement("p", {
        key: key
      }, renderInline(block.text, key));
  }
}
function Markdown({
  content
}) {
  const blocks = parseBlocks(content || "");
  return /*#__PURE__*/React.createElement("div", {
    className: "markdown-body"
  }, blocks.map((block, index) => renderBlock(block, `b${index}`)));
}
Object.assign(__ds_scope, { Markdown, __ds_default_components_content_Markdown_1izshz: Markdown });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/Markdown.jsx", error: String((e && e.message) || e) }); }

// components/core/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Button — a thin wrapper over SceneWorks' real action classes (.primary-action,
// .secondary-action, .danger-action, .icon-btn). These are pure-CSS in the
// SceneWorks source (no React <Button> is exported); this wrapper is an
// INTENTIONAL design-system addition so consumers get a typed, consistent API.
// See readme.md → "Intentional additions".
const VARIANT_CLASS = {
  primary: "primary-action",
  secondary: "secondary-action",
  danger: "danger-action",
  icon: "icon-btn"
};
function Button({
  variant = "secondary",
  icon = null,
  iconRight = null,
  children,
  className = "",
  type = "button",
  ...rest
}) {
  const base = VARIANT_CLASS[variant] ?? VARIANT_CLASS.secondary;
  const cls = className ? `${base} ${className}` : base;
  return /*#__PURE__*/React.createElement("button", _extends({
    type: type,
    className: cls
  }, rest), icon, variant !== "icon" ? children : children ?? icon, iconRight);
}
Object.assign(__ds_scope, { Button, __ds_default_components_core_Button_51d4zy: Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Button.jsx", error: String((e && e.message) || e) }); }

// components/core/Icon.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// SceneWorks icon set — verbatim from @sceneworks/ui. A single 24×24 stroked
// grid, 1.7px strokes, round caps/joins, currentColor. Each glyph is a tiny
// component so callers write <Icon.Video /> etc. Pass size / className / style.
function I({
  d,
  fill = false,
  size = 18,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("svg", _extends({
    "aria-hidden": "true",
    fill: "none",
    height: size,
    stroke: "currentColor",
    strokeLinecap: "round",
    strokeLinejoin: "round",
    strokeWidth: "1.7",
    viewBox: "0 0 24 24",
    width: size
  }, rest), fill ? /*#__PURE__*/React.createElement("path", {
    d: d,
    fill: "currentColor",
    stroke: "none"
  }) : /*#__PURE__*/React.createElement("path", {
    d: d
  }));
}
const Icon = {
  Library: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M4 5h6v14H4zM14 5h6v14h-6zM10 9h4M10 13h4M10 17h4"
  })),
  Image: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4"
  })),
  ImageEditor: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M7 2v15a1 1 0 001 1h15M2 7h15a1 1 0 011 1v15"
  })),
  Video: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M3 6h12v12H3zM15 9l6-3v12l-6-3z"
  })),
  Editor: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M3 8h18M3 12h12M3 16h18M16 10l4 2-4 2z"
  })),
  Train: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M5 19V5M12 19V5M19 19V5M3 9h4M10 15h4M17 11h4"
  })),
  Character: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 20a8 8 0 0116 0"
  })),
  Preset: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M4 7h16M7 12h10M10 17h4"
  })),
  Model: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 3v9M12 12l8-4.5M12 12l-8-4.5"
  })),
  Queue: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M4 7h16M4 12h16M4 17h10"
  })),
  Logs: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7M9 8h3"
  })),
  Search: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3"
  })),
  Sparkle: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 4l1.6 4.4L18 10l-4.4 1.6L12 16l-1.6-4.4L6 10l4.4-1.6zM18 4l.7 1.8L20.5 6.5l-1.8.7L18 9l-.7-1.8L15.5 6.5l1.8-.7z"
  })),
  Plus: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 5v14M5 12h14"
  })),
  Sun: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 4v2M12 18v2M4 12H2M22 12h-2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4l1.4-1.4M17 7l1.4-1.4M12 8a4 4 0 100 8 4 4 0 000-8z"
  })),
  Moon: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M21 13.5A9 9 0 1110.5 3a7 7 0 0010.5 10.5z"
  })),
  Bell: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M6 16V11a6 6 0 0112 0v5l2 2H4zM10 20a2 2 0 004 0"
  })),
  Folder: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"
  })),
  Book: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M5 4h11a2 2 0 012 2v14H7a2 2 0 00-2 2zM5 4v16M9 8h6M9 12h5"
  })),
  Info: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 8h.01M11 12h1v5h1"
  })),
  ChevDown: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M6 9l6 6 6-6"
  })),
  Sliders: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M18 18h2M14 4v4M6 10v4M16 16v4"
  })),
  Play: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    fill: true,
    d: "M7 4l13 8-13 8z"
  })),
  Pause: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    fill: true,
    d: "M7 5h4v14H7zM13 5h4v14h-4z"
  })),
  ArrowLeft: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M19 12H5M11 6l-6 6 6 6"
  })),
  ArrowRight: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M5 12h14M13 6l6 6-6 6"
  })),
  Wand: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    d: "M15 4l5 5L9 20l-5-5zM14 5l5 5M3 14l3 3"
  })),
  Star: ({
    filled = false,
    ...p
  }) => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    fill: filled,
    d: "M12 2l2.4 7.4H22l-6.2 4.5L18 21l-6-4.4L6 21l2.2-7.1L2 9.4h7.6z"
  })),
  Stars: p => /*#__PURE__*/React.createElement(I, _extends({}, p, {
    fill: true,
    d: "M12 2l2.4 7.4H22l-6.2 4.5L18 21l-6-4.4L6 21l2.2-7.1L2 9.4h7.6z"
  }))
};
Object.assign(__ds_scope, { Icon, __ds_default_components_core_Icon_pwifaz: Icon });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Icon.jsx", error: String((e && e.message) || e) }); }

// components/controls/CompactSelector.jsx
try { (() => {
const {
  useEffect,
  useRef,
  useState
} = React;
// Compact Workspace-style switcher — verbatim from @sceneworks/ui. A thumbnail +
// name pill that opens a dropdown to switch the active item. Outside-click +
// Escape close. Thumbnail rendering is injected (renderThumbnail) so the DS stays
// app-agnostic.
function CompactSelector({
  items = [],
  selectedId = "",
  onSelect,
  onCreate,
  createLabel = "New",
  getThumbAsset = () => null,
  renderThumbnail = () => null,
  getSubtitle = () => "",
  busyId = "",
  label = "Select",
  placeholder = "Select…",
  emptyLabel = "Nothing to select yet",
  disabled = false
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const selected = items.find(item => item.id === selectedId) ?? null;
  useEffect(() => {
    if (!open) return undefined;
    function onDocMouseDown(event) {
      if (!containerRef.current?.contains(event.target)) setOpen(false);
    }
    function onDocKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onDocKey);
    };
  }, [open]);
  function renderThumb(item) {
    const asset = item ? getThumbAsset(item) : null;
    return /*#__PURE__*/React.createElement("span", {
      className: "compact-selector-thumb",
      "aria-hidden": "true"
    }, asset ? renderThumbnail(asset) : null);
  }
  return /*#__PURE__*/React.createElement("div", {
    className: "compact-selector",
    ref: containerRef
  }, /*#__PURE__*/React.createElement("button", {
    "aria-expanded": open,
    "aria-haspopup": "listbox",
    "aria-label": label,
    className: "compact-selector-pill",
    disabled: disabled,
    onClick: () => setOpen(v => !v),
    title: selected?.name ?? placeholder,
    type: "button"
  }, renderThumb(selected), /*#__PURE__*/React.createElement("span", {
    className: "compact-selector-meta"
  }, /*#__PURE__*/React.createElement("strong", null, selected?.name ?? placeholder), selected && getSubtitle(selected) ? /*#__PURE__*/React.createElement("span", null, getSubtitle(selected)) : null), /*#__PURE__*/React.createElement(__ds_scope.Icon.ChevDown, {
    className: "chev"
  })), open ? /*#__PURE__*/React.createElement("div", {
    className: "compact-selector-menu",
    role: "listbox"
  }, onCreate ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    className: "compact-selector-item compact-selector-create",
    onClick: () => {
      onCreate();
      setOpen(false);
    },
    type: "button"
  }, /*#__PURE__*/React.createElement("span", {
    className: "compact-selector-thumb compact-selector-create-thumb",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon.Plus, null)), /*#__PURE__*/React.createElement("span", {
    className: "compact-selector-label"
  }, /*#__PURE__*/React.createElement("strong", null, createLabel))), items.length ? /*#__PURE__*/React.createElement("div", {
    className: "compact-selector-divider",
    role: "separator"
  }) : null) : null, items.length === 0 ? onCreate ? null : /*#__PURE__*/React.createElement("p", {
    className: "compact-selector-empty"
  }, emptyLabel) : items.map(item => /*#__PURE__*/React.createElement("button", {
    "aria-selected": item.id === selectedId,
    className: item.id === selectedId ? "compact-selector-item active" : "compact-selector-item",
    disabled: busyId === item.id,
    key: item.id,
    onClick: () => {
      onSelect(item);
      setOpen(false);
    },
    role: "option",
    type: "button"
  }, renderThumb(item), /*#__PURE__*/React.createElement("span", {
    className: "compact-selector-label"
  }, /*#__PURE__*/React.createElement("strong", null, item.name), getSubtitle(item) ? /*#__PURE__*/React.createElement("span", null, busyId === item.id ? "Opening…" : getSubtitle(item)) : null)))) : null);
}
Object.assign(__ds_scope, { CompactSelector, __ds_default_components_controls_CompactSelector_wqk59h: CompactSelector });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/controls/CompactSelector.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Modal.jsx
try { (() => {
const {
  useEffect,
  useRef
} = React; // Shared modal primitive — verbatim from @sceneworks/ui. A backdrop that closes
// on outside mousedown, a role="dialog" that closes on Escape, and focus moved
// into the dialog on mount. Give it a title via `label` (or `labelledBy`).
function Modal({
  children,
  onClose,
  className,
  labelledBy,
  label
}) {
  const dialogRef = useRef(null);
  useEffect(() => {
    dialogRef.current?.focus();
  }, []);
  return /*#__PURE__*/React.createElement("div", {
    className: "modal-backdrop",
    onMouseDown: e => e.target === e.currentTarget && onClose()
  }, /*#__PURE__*/React.createElement("div", {
    "aria-label": label,
    "aria-labelledby": labelledBy,
    "aria-modal": "true",
    className: className ? `modal-card ${className}` : "modal-card",
    onKeyDown: e => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    },
    onMouseDown: e => e.stopPropagation(),
    ref: dialogRef,
    role: "dialog",
    tabIndex: -1
  }, children));
}
Object.assign(__ds_scope, { Modal, __ds_default_components_feedback_Modal_sadu4b: Modal });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Modal.jsx", error: String((e && e.message) || e) }); }

// components/feedback/StatusDot.jsx
try { (() => {
function StatusDot({
  ok
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: ok ? "status-dot ok" : "status-dot",
    "aria-hidden": "true"
  });
}
Object.assign(__ds_scope, { StatusDot, __ds_default_components_feedback_StatusDot_ipwxa1: StatusDot });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/StatusDot.jsx", error: String((e && e.message) || e) }); }

// theme/accents.js
try { (() => {
// SceneWorks — user-selectable accent palettes (verbatim from @sceneworks/ui).
// Each entry maps to a [data-accent="<id>"] block in tokens/colors.css that sets
// the accent + secondary hue; the light/dark ramps consume those hues, so
// flipping the attribute recolors the whole app. `swatch` is the dot shown in the
// topbar picker. "teal" is the default brand accent.
const ACCENTS = [{
  id: "teal",
  name: "Teal",
  swatch: "oklch(0.60 0.13 178)"
}, {
  id: "indigo",
  name: "Indigo",
  swatch: "oklch(0.55 0.16 274)"
}, {
  id: "cobalt",
  name: "Cobalt",
  swatch: "oklch(0.55 0.16 252)"
}, {
  id: "violet",
  name: "Violet",
  swatch: "oklch(0.55 0.18 305)"
}, {
  id: "coral",
  name: "Coral",
  swatch: "oklch(0.64 0.16 28)"
}, {
  id: "amber",
  name: "Amber",
  swatch: "oklch(0.72 0.13 80)"
}, {
  id: "emerald",
  name: "Emerald",
  swatch: "oklch(0.58 0.14 152)"
}];
const DEFAULT_ACCENT = "teal";
const ACCENT_IDS = new Set(ACCENTS.map(accent => accent.id));
function isAccentId(value) {
  return typeof value === "string" && ACCENT_IDS.has(value);
}
Object.assign(__ds_scope, { ACCENTS, DEFAULT_ACCENT, isAccentId });
})(); } catch (e) { __ds_ns.__errors.push({ path: "theme/accents.js", error: String((e && e.message) || e) }); }

// components/controls/AccentPicker.jsx
try { (() => {
const {
  useEffect,
  useRef,
  useState
} = React;
// Topbar accent picker — verbatim from SceneWorks apps/web. A single trigger
// swatch showing the current accent; clicking opens a grid of the remaining
// accents. Picking one calls onChange(id) — the app then sets
// document.documentElement.setAttribute("data-accent", id).
function AccentPicker({
  accent,
  onChange
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const selected = __ds_scope.ACCENTS.find(option => option.id === accent) ?? __ds_scope.ACCENTS[0];
  const others = __ds_scope.ACCENTS.filter(option => option.id !== selected.id);
  useEffect(() => {
    if (!open) return undefined;
    function onDocMouseDown(event) {
      if (!containerRef.current?.contains(event.target)) setOpen(false);
    }
    function onDocKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onDocKey);
    };
  }, [open]);
  return /*#__PURE__*/React.createElement("div", {
    className: "accent-picker",
    ref: containerRef
  }, /*#__PURE__*/React.createElement("button", {
    "aria-expanded": open,
    "aria-haspopup": "listbox",
    "aria-label": `Accent color: ${selected.name}`,
    className: "accent-swatch active",
    onClick: () => setOpen(v => !v),
    style: {
      "--sw": selected.swatch
    },
    title: `Accent color: ${selected.name}`,
    type: "button"
  }), open ? /*#__PURE__*/React.createElement("div", {
    className: "accent-picker-menu",
    role: "listbox",
    "aria-label": "Accent color"
  }, others.map(option => /*#__PURE__*/React.createElement("button", {
    "aria-label": option.name,
    className: "accent-swatch",
    key: option.id,
    onClick: () => {
      onChange(option.id);
      setOpen(false);
    },
    role: "option",
    style: {
      "--sw": option.swatch
    },
    title: option.name,
    type: "button"
  }))) : null);
}
Object.assign(__ds_scope, { AccentPicker, __ds_default_components_controls_AccentPicker_v40xs9: AccentPicker });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/controls/AccentPicker.jsx", error: String((e && e.message) || e) }); }

// ui_kits/studio/app.jsx
try { (() => {
// SceneWorks Studio UI kit — root app. Wires the shell + screens, and drives
// data-theme / data-accent on <html> exactly like the real SceneWorks shell.
const {
  Icon,
  Button
} = window.SceneWorksDesignSystem_b6febf;
const {
  Sidebar,
  Topbar
} = window.SW_Shell;
function Placeholder({
  title,
  body
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "page-frame"
  }, /*#__PURE__*/React.createElement("div", {
    className: "empty-panel"
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      display: "grid",
      placeItems: "center",
      width: 52,
      height: 52,
      borderRadius: "var(--r-lg)",
      background: "var(--accent-soft)",
      color: "var(--accent-strong)"
    }
  }, /*#__PURE__*/React.createElement(Icon.Sparkle, {
    size: 26
  })), /*#__PURE__*/React.createElement("div", {
    className: "section-heading",
    style: {
      justifyItems: "center"
    }
  }, /*#__PURE__*/React.createElement("h2", null, title)), /*#__PURE__*/React.createElement("p", {
    className: "view-copy",
    style: {
      textAlign: "center"
    }
  }, body), /*#__PURE__*/React.createElement(Button, {
    variant: "primary",
    icon: /*#__PURE__*/React.createElement(Icon.Plus, {
      size: 16
    })
  }, "Get started")));
}
const SCREENS = {
  Library: window.SW_Library,
  Image: window.SW_ImageStudio,
  Models: window.SW_Models,
  Video: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Video Studio",
    body: "Text- and image-to-video, clip extend, bridge, and person replacement \u2014 all rendered locally on your GPU."
  }),
  Character: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Character Studio",
    body: "Keep the same face across every shot using identity models and character LoRAs."
  }),
  Training: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Training",
    body: "Build captioned datasets and train image or video LoRAs locally \u2014 no cloud, no Python."
  }),
  Presets: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Presets",
    body: "Save and reuse recurring generation setups across studios."
  }),
  Queue: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Queue",
    body: "Running and recent jobs, with per-job routing and worker activity."
  }),
  Logs: () => /*#__PURE__*/React.createElement(Placeholder, {
    title: "Logs",
    body: "This session's routing decisions and worker activity."
  })
};
function App() {
  const [active, setActive] = React.useState("Image");
  const [theme, setTheme] = React.useState("light");
  const [accent, setAccent] = React.useState("teal");
  React.useEffect(() => {
    const html = document.documentElement;
    html.setAttribute("data-theme", theme);
    html.setAttribute("data-accent", accent);
  }, [theme, accent]);
  const Screen = SCREENS[active] ?? SCREENS.Image;
  return /*#__PURE__*/React.createElement("div", {
    className: "app"
  }, /*#__PURE__*/React.createElement(Sidebar, {
    active: active,
    setActive: setActive
  }), /*#__PURE__*/React.createElement("div", {
    className: "workspace"
  }, /*#__PURE__*/React.createElement(Topbar, {
    active: active,
    theme: theme,
    setTheme: setTheme,
    accent: accent,
    setAccent: setAccent
  }), /*#__PURE__*/React.createElement(Screen, null)));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/studio/app.jsx", error: String((e && e.message) || e) }); }

// ui_kits/studio/imageStudio.jsx
try { (() => {
// SceneWorks Studio UI kit — Image Studio screen (cosmetic recreation).
const {
  Icon,
  Button
} = window.SceneWorksDesignSystem_b6febf;
const {
  AssetTile
} = window.SW_Shell;
const MODES = ["Text to image", "Edit", "With character"];
const MODELS = ["Z-Image-Turbo", "FLUX.2 [klein] 9B", "Qwen-Image", "Krea 2 (Turbo)", "SDXL / RealVisXL"];
function Field({
  label,
  children
}) {
  return /*#__PURE__*/React.createElement("label", {
    className: "settings-field",
    style: {
      flex: "1 1 140px"
    }
  }, label, children);
}
function ImageStudio() {
  const [mode, setMode] = React.useState(0);
  const [prompt, setPrompt] = React.useState("cinematic portrait of a lighthouse keeper, 85mm, soft window light, film grain");
  return /*#__PURE__*/React.createElement("section", {
    className: "page-frame"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel-rule"
  }), /*#__PURE__*/React.createElement("div", {
    className: "mode-tabs"
  }, MODES.map((m, i) => /*#__PURE__*/React.createElement("button", {
    key: m,
    type: "button",
    className: mode === i ? "mode-tab active" : "mode-tab",
    onClick: () => setMode(i)
  }, i === 0 ? /*#__PURE__*/React.createElement(Icon.Image, {
    size: 16
  }) : i === 1 ? /*#__PURE__*/React.createElement(Icon.Editor, {
    size: 16
  }) : /*#__PURE__*/React.createElement(Icon.Character, {
    size: 16
  }), m))), /*#__PURE__*/React.createElement("div", {
    className: "prompt-input-row"
  }, /*#__PURE__*/React.createElement("textarea", {
    className: "prompt-input",
    value: prompt,
    onChange: e => setPrompt(e.target.value),
    placeholder: "Describe the image\u2026"
  }), /*#__PURE__*/React.createElement(Button, {
    variant: "primary",
    icon: /*#__PURE__*/React.createElement(Icon.Sparkle, null),
    style: {
      alignSelf: "stretch",
      minWidth: 140
    }
  }, "Generate")), /*#__PURE__*/React.createElement("div", {
    className: "settings-bar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "settings-bar-row"
  }, /*#__PURE__*/React.createElement(Field, {
    label: "Model"
  }, /*#__PURE__*/React.createElement("select", {
    defaultValue: "Z-Image-Turbo"
  }, MODELS.map(m => /*#__PURE__*/React.createElement("option", {
    key: m
  }, m)))), /*#__PURE__*/React.createElement(Field, {
    label: "Aspect"
  }, /*#__PURE__*/React.createElement("select", {
    defaultValue: "1:1"
  }, /*#__PURE__*/React.createElement("option", null, "1:1"), /*#__PURE__*/React.createElement("option", null, "3:2"), /*#__PURE__*/React.createElement("option", null, "16:9"), /*#__PURE__*/React.createElement("option", null, "2:3"))), /*#__PURE__*/React.createElement(Field, {
    label: "Variations"
  }, /*#__PURE__*/React.createElement("select", {
    defaultValue: "4"
  }, /*#__PURE__*/React.createElement("option", null, "1"), /*#__PURE__*/React.createElement("option", null, "2"), /*#__PURE__*/React.createElement("option", null, "4"), /*#__PURE__*/React.createElement("option", null, "6"))), /*#__PURE__*/React.createElement(Field, {
    label: "Quality"
  }, /*#__PURE__*/React.createElement("select", {
    defaultValue: "Q8"
  }, /*#__PURE__*/React.createElement("option", null, "bf16"), /*#__PURE__*/React.createElement("option", null, "Q8"), /*#__PURE__*/React.createElement("option", null, "Q4")))))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "baseline",
      gap: 10,
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("p", {
    className: "eyebrow"
  }, "Latest batch"), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 12.5,
      color: "var(--text-muted)"
    }
  }, "4 variations \xB7 seed 84213 \xB7 28 steps")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "repeat(4, 1fr)",
      gap: 14
    }
  }, /*#__PURE__*/React.createElement(AssetTile, {
    badge: "new",
    label: "var 1"
  }), /*#__PURE__*/React.createElement(AssetTile, {
    label: "var 2"
  }), /*#__PURE__*/React.createElement(AssetTile, {
    label: "var 3"
  }), /*#__PURE__*/React.createElement(AssetTile, {
    label: "var 4"
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "eyebrow",
    style: {
      marginBottom: 12
    }
  }, "Earlier today"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "repeat(6, 1fr)",
      gap: 12
    }
  }, ["lighthouse dawn", "harbor wide", "keeper close", "storm sky", "lantern detail", "cliffside"].map(l => /*#__PURE__*/React.createElement(AssetTile, {
    key: l,
    label: l
  })))));
}
window.SW_ImageStudio = ImageStudio;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/studio/imageStudio.jsx", error: String((e && e.message) || e) }); }

// ui_kits/studio/library.jsx
try { (() => {
// SceneWorks Studio UI kit — Library screen (cosmetic recreation).
const {
  Icon,
  Button
} = window.SceneWorksDesignSystem_b6febf;
const {
  AssetTile
} = window.SW_Shell;
const TAGS = ["portrait", "product", "storyboard", "reference", "upscaled"];
const GRID = [{
  l: "keeper_close_v2",
  b: "★★★★"
}, {
  l: "harbor_wide"
}, {
  l: "lantern_detail",
  b: "upscaled"
}, {
  l: "storm_sky"
}, {
  l: "cliffside_dawn"
}, {
  l: "boat_moored"
}, {
  l: "rope_texture"
}, {
  l: "window_light"
}, {
  l: "keeper_hands"
}, {
  l: "gull_flight"
}];
function LibraryScreen() {
  const [mode, setMode] = React.useState("assets");
  const [type, setType] = React.useState("all");
  return /*#__PURE__*/React.createElement("section", {
    className: "page-frame"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel-rule"
  }), /*#__PURE__*/React.createElement("div", {
    className: "toolbar"
  }, /*#__PURE__*/React.createElement("label", {
    className: "file-upload-button"
  }, /*#__PURE__*/React.createElement(Icon.Folder, {
    size: 16
  }), " Import"), /*#__PURE__*/React.createElement("input", {
    type: "search",
    placeholder: "Search name, prompt, or tags\u2026"
  }), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Asset type",
    value: type,
    onChange: e => setType(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All media"), /*#__PURE__*/React.createElement("option", {
    value: "image"
  }, "Images"), /*#__PURE__*/React.createElement("option", {
    value: "video"
  }, "Videos"), /*#__PURE__*/React.createElement("option", {
    value: "upload"
  }, "Uploads")), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Asset tag",
    defaultValue: "all"
  }, /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All tags"), TAGS.map(t => /*#__PURE__*/React.createElement("option", {
    key: t
  }, t))), /*#__PURE__*/React.createElement("label", {
    className: "checkline",
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      color: "var(--text-muted)",
      fontSize: 13
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox"
  }), " Rejected"), /*#__PURE__*/React.createElement("div", {
    className: "segmented-control",
    role: "group",
    "aria-label": "Asset collection"
  }, /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: mode === "assets" ? "active" : "",
    onClick: () => setMode("assets")
  }, "Assets"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: mode === "trashcan" ? "active" : "",
    onClick: () => setMode("trashcan")
  }, "Trashcan"))), /*#__PURE__*/React.createElement("div", {
    className: "stat-strip"
  }, /*#__PURE__*/React.createElement("div", {
    className: "stat-chip"
  }, /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Project"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-value"
  }, "Portrait lookbook")), /*#__PURE__*/React.createElement("div", {
    className: "stat-chip"
  }, /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Assets"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-value"
  }, "312 total")), /*#__PURE__*/React.createElement("div", {
    className: "stat-chip"
  }, /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Images"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-value"
  }, "274")), /*#__PURE__*/React.createElement("div", {
    className: "stat-chip"
  }, /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Clips"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-value"
  }, "38")))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "minmax(0, 1fr) 280px",
      gap: 18,
      alignItems: "start"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "repeat(5, 1fr)",
      gap: 12
    }
  }, GRID.map((g, i) => /*#__PURE__*/React.createElement(AssetTile, {
    key: i,
    label: g.l,
    badge: g.b
  }))), /*#__PURE__*/React.createElement("aside", {
    className: "work-panel",
    style: {
      gap: 14
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel-rule"
  }), /*#__PURE__*/React.createElement(AssetTile, {
    label: "keeper_close_v2"
  }), /*#__PURE__*/React.createElement("div", {
    className: "section-heading"
  }, /*#__PURE__*/React.createElement("h2", {
    style: {
      fontSize: 16
    }
  }, "keeper_close_v2")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gap: 6,
      fontSize: 12.5,
      color: "var(--text-muted)"
    }
  }, /*#__PURE__*/React.createElement("div", null, "Z-Image-Turbo \xB7 Q8 \xB7 1024\xD71024"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: "var(--font-mono)",
      fontSize: 11.5
    }
  }, "seed 84213 \xB7 cfg 4.5 \xB7 28 steps")), /*#__PURE__*/React.createElement("div", {
    className: "rating-stars",
    style: {
      alignSelf: "start"
    }
  }, [0, 1, 2, 3, 4].map(i => /*#__PURE__*/React.createElement("button", {
    key: i,
    type: "button",
    className: i < 4 ? "star-rating-button active" : "star-rating-button"
  }, /*#__PURE__*/React.createElement(Icon.Star, {
    filled: i < 4
  })))), /*#__PURE__*/React.createElement("div", {
    className: "detail-actions"
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    icon: /*#__PURE__*/React.createElement(Icon.Image, {
      size: 16
    })
  }, "Send to Image"), /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    icon: /*#__PURE__*/React.createElement(Icon.Editor, {
      size: 16
    })
  }, "Edit"), /*#__PURE__*/React.createElement(Button, {
    variant: "danger",
    icon: /*#__PURE__*/React.createElement(Icon.Logs, {
      size: 16
    })
  }, "Discard")))));
}
window.SW_Library = LibraryScreen;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/studio/library.jsx", error: String((e && e.message) || e) }); }

// ui_kits/studio/modelManager.jsx
try { (() => {
// SceneWorks Studio UI kit — Model Manager screen (cosmetic recreation).
const {
  Icon,
  Button
} = window.SceneWorksDesignSystem_b6febf;
const TABS = ["Image", "Video", "Utility", "LoRAs"];
const MODELS = {
  Image: [{
    name: "Z-Image / Z-Image-Edit",
    note: "Turbo · edit-capable",
    size: "6.6 GB",
    mem: "12 GB",
    state: "installed"
  }, {
    name: "FLUX.2 [klein] 9B",
    note: "Non-commercial license",
    size: "17.2 GB",
    mem: "24 GB",
    state: "download"
  }, {
    name: "Qwen-Image (+ Edit)",
    note: "Text render · edit",
    size: "12.1 GB",
    mem: "18 GB",
    state: "installed"
  }, {
    name: "Krea 2 (Raw / Turbo)",
    note: "Photoreal",
    size: "9.4 GB",
    mem: "16 GB",
    state: "download"
  }, {
    name: "SDXL / RealVisXL",
    note: "SDXL-family kernel",
    size: "6.9 GB",
    mem: "10 GB",
    state: "installed"
  }, {
    name: "InstantID",
    note: "Identity model",
    size: "2.1 GB",
    mem: "8 GB",
    state: "gated"
  }],
  Video: [{
    name: "LTX-2.3",
    note: "Text- & image-to-video",
    size: "14.8 GB",
    mem: "24 GB",
    state: "installed"
  }, {
    name: "Wan 2.2 (TI2V-5B)",
    note: "Single-expert",
    size: "10.2 GB",
    mem: "18 GB",
    state: "download"
  }, {
    name: "Stable Video Diffusion",
    note: "Image-to-video",
    size: "9.6 GB",
    mem: "16 GB",
    state: "download"
  }],
  Utility: [{
    name: "Real-ESRGAN",
    note: "Upscaler",
    size: "0.7 GB",
    mem: "4 GB",
    state: "installed"
  }, {
    name: "JoyCaption",
    note: "Vision captioner",
    size: "8.1 GB",
    mem: "12 GB",
    state: "download"
  }, {
    name: "Anubis-8B",
    note: "Prompt refiner",
    size: "8.0 GB",
    mem: "12 GB",
    state: "download"
  }],
  LoRAs: [{
    name: "lighthouse-look",
    note: "Image LoRA · Z-Image",
    size: "0.2 GB",
    mem: "—",
    state: "installed"
  }, {
    name: "product-studio",
    note: "Image LoRA · SDXL",
    size: "0.3 GB",
    mem: "—",
    state: "installed"
  }]
};
function StateCell({
  state
}) {
  if (state === "installed") return /*#__PURE__*/React.createElement("span", {
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      color: "var(--success)",
      fontSize: 12.5,
      fontWeight: 600
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot"
  }), " Installed");
  if (state === "gated") return /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    icon: /*#__PURE__*/React.createElement(Icon.Info, {
      size: 15
    })
  }, "Add token");
  return /*#__PURE__*/React.createElement(Button, {
    variant: "primary",
    icon: /*#__PURE__*/React.createElement(Icon.Sparkle, {
      size: 15
    }),
    style: {
      minHeight: 36,
      padding: "0 14px",
      fontSize: 13
    }
  }, "Download");
}
function ModelRow({
  m
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "minmax(0,1fr) 100px 100px 150px",
      alignItems: "center",
      gap: 12,
      padding: "12px 14px",
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--r-md)"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 12,
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      display: "grid",
      placeItems: "center",
      width: 38,
      height: 38,
      borderRadius: "var(--r-sm)",
      background: "var(--accent-soft)",
      color: "var(--accent-strong)",
      flex: "0 0 auto"
    }
  }, /*#__PURE__*/React.createElement(Icon.Model, {
    size: 20
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13.5,
      fontWeight: 600,
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap"
    }
  }, m.name), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      color: "var(--text-muted)"
    }
  }, m.note))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: "var(--font-mono)",
      fontSize: 12.5,
      color: "var(--text-muted)"
    }
  }, m.size), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: "var(--font-mono)",
      fontSize: 12.5,
      color: "var(--text-muted)"
    }
  }, m.mem), /*#__PURE__*/React.createElement("div", {
    style: {
      justifySelf: "end"
    }
  }, /*#__PURE__*/React.createElement(StateCell, {
    state: m.state
  })));
}
function ModelManager() {
  const [tab, setTab] = React.useState("Image");
  return /*#__PURE__*/React.createElement("section", {
    className: "page-frame"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel-rule"
  }), /*#__PURE__*/React.createElement("div", {
    className: "work-panel-head"
  }, /*#__PURE__*/React.createElement("div", {
    className: "work-panel-head-text"
  }, /*#__PURE__*/React.createElement("p", {
    className: "eyebrow work-panel-eyebrow"
  }, "Local catalog"), /*#__PURE__*/React.createElement("div", {
    className: "section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Models")), /*#__PURE__*/React.createElement("p", {
    class: "work-panel-hint"
  }, "Weights download on first use \u2014 nothing is bundled. Each model lists size and minimum memory.")), /*#__PURE__*/React.createElement("div", {
    className: "work-panel-actions"
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    icon: /*#__PURE__*/React.createElement(Icon.Folder, {
      size: 16
    })
  }, "Import checkpoint"))), /*#__PURE__*/React.createElement("div", {
    className: "toolbar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "segmented-control"
  }, TABS.map(t => /*#__PURE__*/React.createElement("button", {
    key: t,
    type: "button",
    className: tab === t ? "active" : "",
    onClick: () => setTab(t)
  }, t))), /*#__PURE__*/React.createElement("input", {
    type: "search",
    placeholder: "Search models\u2026"
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "minmax(0,1fr) 100px 100px 150px",
      gap: 12,
      padding: "0 14px"
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Model"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Size"), /*#__PURE__*/React.createElement("span", {
    className: "stat-chip-label"
  }, "Min memory"), /*#__PURE__*/React.createElement("span", null)), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gap: 8
    }
  }, MODELS[tab].map(m => /*#__PURE__*/React.createElement(ModelRow, {
    key: m.name,
    m: m
  }))));
}
window.SW_Models = ModelManager;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/studio/modelManager.jsx", error: String((e && e.message) || e) }); }

// ui_kits/studio/shell.jsx
try { (() => {
// SceneWorks Studio UI kit — app shell (sidebar + topbar). Cosmetic recreation of
// the real SceneWorks web shell, composing design-system primitives + classes.
const {
  Logo,
  Wordmark,
  Icon,
  StatusDot,
  AccentPicker,
  CompactSelector
} = window.SceneWorksDesignSystem_b6febf;
const NAV_STUDIOS = [{
  id: "Library",
  label: "Library",
  icon: Icon.Library
}, {
  id: "Image",
  label: "Image Studio",
  icon: Icon.Image
}, {
  id: "Video",
  label: "Video Studio",
  icon: Icon.Video
}, {
  id: "Character",
  label: "Character Studio",
  icon: Icon.Character
}, {
  id: "Training",
  label: "Training",
  icon: Icon.Train
}];
const NAV_MANAGE = [{
  id: "Models",
  label: "Model Manager",
  icon: Icon.Model,
  badge: "64"
}, {
  id: "Presets",
  label: "Presets",
  icon: Icon.Preset
}, {
  id: "Queue",
  label: "Queue",
  icon: Icon.Queue
}, {
  id: "Logs",
  label: "Logs",
  icon: Icon.Logs
}];
const PROJECTS = [{
  id: "look",
  name: "Portrait lookbook"
}, {
  id: "promo",
  name: "Product promo"
}, {
  id: "story",
  name: "Storyboard — Act I"
}];
function NavItem({
  item,
  active,
  onClick
}) {
  const IconCmp = item.icon;
  return /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: active ? "nav-item active" : "nav-item",
    onClick: onClick
  }, /*#__PURE__*/React.createElement(IconCmp, null), /*#__PURE__*/React.createElement("span", {
    className: "nav-label"
  }, item.label), item.badge ? /*#__PURE__*/React.createElement("span", {
    className: "nav-badge"
  }, item.badge) : null);
}
function Sidebar({
  active,
  setActive
}) {
  const [project, setProject] = React.useState("look");
  return /*#__PURE__*/React.createElement("aside", {
    className: "sidebar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "brand-mark"
  }, /*#__PURE__*/React.createElement(Logo, {
    size: 30
  })), /*#__PURE__*/React.createElement("h1", null, "Scene", /*#__PURE__*/React.createElement("span", {
    className: "light"
  }, "Works"))), /*#__PURE__*/React.createElement("div", {
    className: "sidebar-section"
  }, /*#__PURE__*/React.createElement(CompactSelector, {
    items: PROJECTS,
    selectedId: project,
    onSelect: p => setProject(p.id),
    onCreate: () => {},
    createLabel: "New project",
    getSubtitle: () => "Local project",
    label: "Project"
  })), /*#__PURE__*/React.createElement("div", {
    className: "sidebar-section"
  }, /*#__PURE__*/React.createElement("div", {
    className: "sidebar-section-title"
  }, "Studios"), /*#__PURE__*/React.createElement("div", {
    className: "nav-list"
  }, NAV_STUDIOS.map(item => /*#__PURE__*/React.createElement(NavItem, {
    key: item.id,
    item: item,
    active: active === item.id,
    onClick: () => setActive(item.id)
  })))), /*#__PURE__*/React.createElement("div", {
    className: "sidebar-section"
  }, /*#__PURE__*/React.createElement("div", {
    className: "sidebar-section-title"
  }, "Manage"), /*#__PURE__*/React.createElement("div", {
    className: "nav-list"
  }, NAV_MANAGE.map(item => /*#__PURE__*/React.createElement(NavItem, {
    key: item.id,
    item: item,
    active: active === item.id,
    onClick: () => setActive(item.id)
  })))), /*#__PURE__*/React.createElement("div", {
    className: "sidebar-footer"
  }, /*#__PURE__*/React.createElement("span", {
    className: "app-version"
  }, "SceneWorks 0.7.5 \xB7 MLX")));
}
const TITLES = {
  Library: {
    title: "Library",
    hint: "Every still and clip across this project"
  },
  Image: {
    title: "Image Studio",
    hint: "Text-to-image, edits, and reference-guided generation"
  },
  Video: {
    title: "Video Studio",
    hint: "Text- and image-to-video, extend, and person replace"
  },
  Character: {
    title: "Character Studio",
    hint: "Keep one face across every shot"
  },
  Training: {
    title: "Training",
    hint: "Caption datasets and train LoRAs locally"
  },
  Models: {
    title: "Model Manager",
    hint: "Download and manage local checkpoints"
  },
  Presets: {
    title: "Presets",
    hint: "Saved generation setups"
  },
  Queue: {
    title: "Queue",
    hint: "Running and recent jobs"
  },
  Logs: {
    title: "Logs",
    hint: "This session's routing and worker activity"
  }
};
function Topbar({
  active,
  theme,
  setTheme,
  accent,
  setAccent
}) {
  const meta = TITLES[active] ?? TITLES.Image;
  return /*#__PURE__*/React.createElement("header", {
    className: "topbar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "topbar-title"
  }, /*#__PURE__*/React.createElement("h1", null, meta.title), /*#__PURE__*/React.createElement("p", null, meta.hint)), /*#__PURE__*/React.createElement("div", {
    className: "topbar-spacer"
  }), /*#__PURE__*/React.createElement("span", {
    className: "topbar-status"
  }, /*#__PURE__*/React.createElement("span", {
    className: "status-pill"
  }, /*#__PURE__*/React.createElement(StatusDot, {
    ok: true
  }), " worker online")), /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: "queue-chip"
  }, /*#__PURE__*/React.createElement(Icon.Queue, {
    size: 15
  }), " 2 running"), /*#__PURE__*/React.createElement("span", {
    className: "topbar-divider"
  }), /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: "icon-btn",
    "aria-label": "Notifications"
  }, /*#__PURE__*/React.createElement(Icon.Bell, null)), /*#__PURE__*/React.createElement(AccentPicker, {
    accent: accent,
    onChange: setAccent
  }), /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: "icon-btn",
    "aria-label": "Toggle theme",
    onClick: () => setTheme(theme === "light" ? "dark" : "light")
  }, theme === "light" ? /*#__PURE__*/React.createElement(Icon.Moon, null) : /*#__PURE__*/React.createElement(Icon.Sun, null)));
}

// Checkerboard placeholder tile matching SceneWorks' own thumb aesthetic.
function AssetTile({
  ratio = "1 / 1",
  label,
  badge
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: "relative",
      aspectRatio: ratio,
      borderRadius: "var(--r-md)",
      overflow: "hidden",
      border: "1px solid var(--border)",
      background: "repeating-linear-gradient(45deg, color-mix(in oklch, var(--accent) 20%, transparent) 0 6px, transparent 6px 12px), color-mix(in oklch, var(--warm) 14%, var(--surface))"
    }
  }, badge ? /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      top: 8,
      left: 8,
      padding: "2px 8px",
      fontSize: 10.5,
      fontWeight: 700,
      letterSpacing: "0.04em",
      textTransform: "uppercase",
      color: "var(--accent-fg)",
      background: "var(--accent)",
      borderRadius: "var(--r-pill)"
    }
  }, badge) : null, label ? /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      bottom: 8,
      left: 8,
      right: 8,
      fontSize: 11,
      fontWeight: 500,
      color: "var(--text)",
      background: "color-mix(in oklch, var(--surface) 82%, transparent)",
      padding: "3px 7px",
      borderRadius: "var(--r-xs)",
      backdropFilter: "blur(4px)",
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap"
    }
  }, label) : null);
}
window.SW_Shell = {
  Sidebar,
  Topbar,
  AssetTile
};
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/studio/shell.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Logo = __ds_scope.Logo;

__ds_ns.Wordmark = __ds_scope.Wordmark;

__ds_ns.Markdown = __ds_scope.Markdown;

__ds_ns.AccentPicker = __ds_scope.AccentPicker;

__ds_ns.CompactSelector = __ds_scope.CompactSelector;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Icon = __ds_scope.Icon;

__ds_ns.Modal = __ds_scope.Modal;

__ds_ns.StatusDot = __ds_scope.StatusDot;

__ds_ns.ACCENTS = __ds_scope.ACCENTS;

__ds_ns.DEFAULT_ACCENT = __ds_scope.DEFAULT_ACCENT;

})();
