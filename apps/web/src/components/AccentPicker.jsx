import React, { useEffect, useRef, useState } from "react";
import { ACCENTS } from "../accents.js";

// Topbar accent picker (sc-accent): collapses the old full swatch row down to a
// single trigger showing the current accent. Clicking opens a 2x3 dropdown of the
// remaining accents; picking one recolors the app and closes the menu. Mirrors the
// CompactSelector outside-click + Escape close pattern.
export function AccentPicker({ accent, onChange }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const selected = ACCENTS.find((option) => option.id === accent) ?? ACCENTS[0];
  const others = ACCENTS.filter((option) => option.id !== selected.id);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function onDocMouseDown(event) {
      if (!containerRef.current?.contains(event.target)) {
        setOpen(false);
      }
    }
    function onDocKey(event) {
      if (event.key === "Escape") {
        setOpen(false);
      }
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
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={`Accent color: ${selected.name}`}
        className="accent-swatch active"
        onClick={() => setOpen((value) => !value)}
        style={{ "--sw": selected.swatch }}
        title={`Accent color: ${selected.name}`}
        type="button"
      />
      {open ? (
        <div className="accent-picker-menu" role="listbox" aria-label="Accent color">
          {others.map((option) => (
            <button
              aria-label={option.name}
              className="accent-swatch"
              key={option.id}
              onClick={() => {
                onChange(option.id);
                setOpen(false);
              }}
              role="option"
              style={{ "--sw": option.swatch }}
              title={option.name}
              type="button"
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
