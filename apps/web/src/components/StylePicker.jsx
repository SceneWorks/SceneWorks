import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./Icons.jsx";

// sc-13130 — Style Catalog picker for the Image Studio. A searchable, grouped, single-select
// dropdown over the 278-entry style catalog (styles.json), with an always-available "None"
// option that resets to pass-through (no style applied → prompt is the untouched user prompt).
//
// Copy is "Style" / "Style Catalog" — never "text style" (that label belongs to Krea's numeric
// `textStyleGain` slider elsewhere in ImageStudio, an unrelated control).
//
// Modeled on CompactSelector (the app's pill+menu switcher): outside-click + Escape close, a
// pill trigger with aria-expanded, and role="listbox"/role="option" items — plus a search box
// and group headers this catalog needs. Selection is by style `id`; the parent owns the state.
export function StylePicker({ groups = [], selectedId = null, onSelect, label = "Style", disabled = false }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const containerRef = useRef(null);
  const searchRef = useRef(null);

  const selected = useMemo(() => {
    if (!selectedId) {
      return null;
    }
    for (const group of groups) {
      const match = group.styles.find((style) => style.id === selectedId);
      if (match) {
        return match;
      }
    }
    return null;
  }, [groups, selectedId]);

  // Filter styles by name (case-insensitive substring), dropping groups with no surviving match
  // so the menu only shows headers that have options under them.
  const filteredGroups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return groups;
    }
    return groups
      .map((group) => ({
        ...group,
        styles: group.styles.filter((style) => style.name.toLowerCase().includes(needle)),
      }))
      .filter((group) => group.styles.length > 0);
  }, [groups, query]);

  const hasResults = filteredGroups.some((group) => group.styles.length > 0);

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

  // Move focus into the search box when the menu opens so the whole flow is keyboard-usable:
  // open → type to filter → Tab/Arrow to an option → Enter to select.
  useEffect(() => {
    if (open) {
      searchRef.current?.focus();
    } else {
      setQuery("");
    }
  }, [open]);

  function choose(id) {
    onSelect?.(id);
    setOpen(false);
  }

  return (
    <div className="compact-selector style-picker" ref={containerRef}>
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={label}
        className="compact-selector-pill"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        title={selected?.name ?? "None"}
        type="button"
      >
        <span className="compact-selector-meta">
          <strong>{selected ? selected.name : "None"}</strong>
          <span>{selected ? "Style Catalog" : "No style — pass-through"}</span>
        </span>
        <Icon.ChevDown className="chev" />
      </button>

      {open ? (
        <div className="compact-selector-menu style-picker-menu" role="listbox" aria-label="Style Catalog">
          <input
            aria-label="Search styles"
            className="style-picker-search"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search styles…"
            ref={searchRef}
            type="search"
            value={query}
          />
          <button
            aria-selected={!selectedId}
            className={!selectedId ? "compact-selector-item active" : "compact-selector-item"}
            onClick={() => choose(null)}
            role="option"
            type="button"
          >
            <span className="compact-selector-label">
              <strong>None</strong>
              <span>Pass-through — your prompt is sent unchanged</span>
            </span>
          </button>

          {hasResults ? (
            filteredGroups.map((group) => (
              <div className="style-picker-group" key={group.id}>
                <p className="style-picker-group-label" role="presentation">
                  {group.name}
                </p>
                {group.styles.map((style) => (
                  <button
                    aria-selected={style.id === selectedId}
                    className={style.id === selectedId ? "compact-selector-item active" : "compact-selector-item"}
                    key={style.id}
                    onClick={() => choose(style.id)}
                    role="option"
                    title={style.name}
                    type="button"
                  >
                    <span className="compact-selector-label">
                      <strong>{style.name}</strong>
                    </span>
                  </button>
                ))}
              </div>
            ))
          ) : (
            <p className="compact-selector-empty">No styles match “{query}”.</p>
          )}
        </div>
      ) : null}
    </div>
  );
}
