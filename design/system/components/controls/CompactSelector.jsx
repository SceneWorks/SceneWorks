import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../core/Icon.jsx";

// Compact Workspace-style switcher — verbatim from @sceneworks/ui. A thumbnail +
// name pill that opens a dropdown to switch the active item. Outside-click +
// Escape close. Thumbnail rendering is injected (renderThumbnail) so the DS stays
// app-agnostic.
export function CompactSelector({
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
  disabled = false,
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const selected = items.find((item) => item.id === selectedId) ?? null;

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
    return (
      <span className="compact-selector-thumb" aria-hidden="true">
        {asset ? renderThumbnail(asset) : null}
      </span>
    );
  }

  return (
    <div className="compact-selector" ref={containerRef}>
      <button aria-expanded={open} aria-haspopup="listbox" aria-label={label} className="compact-selector-pill"
        disabled={disabled} onClick={() => setOpen((v) => !v)} title={selected?.name ?? placeholder} type="button">
        {renderThumb(selected)}
        <span className="compact-selector-meta">
          <strong>{selected?.name ?? placeholder}</strong>
          {selected && getSubtitle(selected) ? <span>{getSubtitle(selected)}</span> : null}
        </span>
        <Icon.ChevDown className="chev" />
      </button>

      {open ? (
        <div className="compact-selector-menu" role="listbox">
          {onCreate ? (
            <>
              <button className="compact-selector-item compact-selector-create"
                onClick={() => { onCreate(); setOpen(false); }} type="button">
                <span className="compact-selector-thumb compact-selector-create-thumb" aria-hidden="true"><Icon.Plus /></span>
                <span className="compact-selector-label"><strong>{createLabel}</strong></span>
              </button>
              {items.length ? <div className="compact-selector-divider" role="separator" /> : null}
            </>
          ) : null}
          {items.length === 0 ? (
            onCreate ? null : <p className="compact-selector-empty">{emptyLabel}</p>
          ) : (
            items.map((item) => (
              <button aria-selected={item.id === selectedId}
                className={item.id === selectedId ? "compact-selector-item active" : "compact-selector-item"}
                disabled={busyId === item.id} key={item.id}
                onClick={() => { onSelect(item); setOpen(false); }} role="option" type="button">
                {renderThumb(item)}
                <span className="compact-selector-label">
                  <strong>{item.name}</strong>
                  {getSubtitle(item) ? <span>{busyId === item.id ? "Opening…" : getSubtitle(item)}</span> : null}
                </span>
              </button>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}

export default CompactSelector;
