import React, { useEffect, useRef, useState } from "react";
import { APPEARANCE_AXES, appearanceOptions } from "../data/appearanceCatalog.js";
import { useDismissOnOutsideOrEscape } from "../hooks/useDismissOnOutsideOrEscape.js";
import { Icon } from "./Icons.jsx";

function RecipeSelector({ axis, control }) {
  const { key, label, multiple } = axis;
  const options = appearanceOptions(control.model, axis.axis ?? key);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [position, setPosition] = useState({});
  const container = useRef(null);
  const trigger = useRef(null);
  const search = useRef(null);
  useDismissOnOutsideOrEscape(container, open, () => setOpen(false));
  useEffect(() => {
    if (!open) return undefined;
    function place() {
      const rect = trigger.current.getBoundingClientRect();
      const width = Math.min(360, window.innerWidth - 24);
      const below = window.innerHeight - rect.bottom - 16;
      const above = rect.top - 16;
      const down = below >= 280 || below >= above;
      setPosition({ position: "fixed", width, left: Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)), right: "auto",
        top: down ? rect.bottom + 8 : "auto", bottom: down ? "auto" : window.innerHeight - rect.top + 8,
        maxHeight: Math.max(120, Math.min(440, down ? below : above)) });
    }
    place();
    setQuery("");
    search.current?.focus();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); window.removeEventListener("scroll", place, true); };
  }, [open]);
  const ids = multiple ? control.selection[key] : control.selection[key] ? [control.selection[key]] : [];
  const selected = ids.map((id) => options.find((entry) => entry.id === id)).filter(Boolean);
  const summary = selected.length ? selected.map((entry) => entry.name).join(" + ") : "None";
  function choose(id) {
    const value = multiple ? ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id] : id;
    control.setSelection({ ...control.selection, [key]: value });
    if (!multiple) { setOpen(false); trigger.current?.focus(); }
  }
  function handleKeyDown(event) {
    if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); }
    if (event.key === "Enter" && event.target.tagName === "INPUT") {
      // These controls live inside the generation form. Enter searches or chooses a
      // recipe; it must never implicitly submit an image/video generation request.
      event.preventDefault();
      if (event.target.type !== "search") event.target.click();
    }
  }
  const filtered = options.filter((entry) => `${entry.name} ${entry.group}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="style-axis-field appearance-axis" ref={container}>
    <span className="settings-bar-label">{label}</span>
    <div className="compact-selector">
      <button type="button" className="compact-selector-pill appearance-selector-pill" aria-label={label} aria-expanded={open} aria-haspopup="dialog" onClick={() => setOpen(!open)} ref={trigger} title={summary}>
        <span className="compact-selector-meta"><strong>{summary}</strong><span>{multiple ? selected.length ? `${selected.length} selected` : "Combine treatments" : "Prompt recipe"}</span></span><Icon.ChevDown className="chev" />
      </button>
      {open ? <div className="compact-selector-menu appearance-selector-menu" style={position} role="dialog" aria-label={`Choose ${label}`} onKeyDown={handleKeyDown}>
        <input type="search" aria-label={`Search ${label}`} placeholder={`Search ${label.toLowerCase()}…`} className="style-picker-search" ref={search} value={query} onChange={(event) => setQuery(event.target.value)} />
        <button type="button" className="appearance-clear" onClick={() => { control.setSelection({ ...control.selection, [key]: multiple ? [] : null }); setOpen(false); trigger.current?.focus(); }}>None — clear {label.toLowerCase()}</button>
        <div className="appearance-options">
          {filtered.length ? filtered.map((entry) => <label className="appearance-option" key={entry.id}>
            <input type={multiple ? "checkbox" : "radio"} name={`appearance-${key}`} checked={ids.includes(entry.id)} onChange={() => choose(entry.id)} />
            <span><strong>{entry.name}</strong><small>{entry.group}</small>{entry.note ? <small>{entry.note}</small> : null}</span>
          </label>) : <p className="muted">No matching recipes.</p>}
        </div>
        {multiple ? <button type="button" className="appearance-clear" onClick={() => { setOpen(false); trigger.current?.focus(); }}>Done</button> : null}
      </div> : null}
    </div>
  </div>;
}
export function AppearanceSelectors({ control }) {
  if (!control?.available) return null;
  return <>{APPEARANCE_AXES.map((axis) => <RecipeSelector axis={axis} control={control} key={axis.key} />)}</>;
}
export function AppearanceNotes({ control }) {
  if (!control) return null;
  return <div className="appearance-notes">
    {control.notice ? <p role="status">{control.notice}</p> : null}
    {control.warnings.map((warning) => <p key={warning}>{warning}</p>)}
  </div>;
}
