import React, { useEffect, useRef, useState } from "react";
import { ACCENTS } from "../../theme/accents.js";

// Topbar accent picker — verbatim from SceneWorks apps/web. A single trigger
// swatch showing the current accent; clicking opens a grid of the remaining
// accents. Picking one calls onChange(id) — the app then sets
// document.documentElement.setAttribute("data-accent", id).
export function AccentPicker({ accent, onChange }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const selected = ACCENTS.find((option) => option.id === accent) ?? ACCENTS[0];
  const others = ACCENTS.filter((option) => option.id !== selected.id);

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

  return (
    <div className="accent-picker" ref={containerRef}>
      <button aria-expanded={open} aria-haspopup="listbox" aria-label={`Accent color: ${selected.name}`}
        className="accent-swatch active" onClick={() => setOpen((v) => !v)}
        style={{ "--sw": selected.swatch }} title={`Accent color: ${selected.name}`} type="button" />
      {open ? (
        <div className="accent-picker-menu" role="listbox" aria-label="Accent color">
          {others.map((option) => (
            <button aria-label={option.name} className="accent-swatch" key={option.id}
              onClick={() => { onChange(option.id); setOpen(false); }} role="option"
              style={{ "--sw": option.swatch }} title={option.name} type="button" />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default AccentPicker;
