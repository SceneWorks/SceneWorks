import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./Icons.jsx";

// sc-13130 / sc-13171 — Style Catalog picker for the Image Studio. A TWO-LEVEL cascade over the
// 278-entry style catalog (styles.json): first pick one of the 8 top-level groups, then pick a
// style within that group. Each group also carries its own top-level "overall" style (the group's
// `description`, distinct from any sub-style) surfaced as the FIRST option within the group and
// stored as the GROUP id. An always-available "None" option resets to pass-through (no style
// applied → the prompt is the untouched user prompt). An optional global search jumps across all
// styles at once.
//
// Selection contract is unchanged from the flat picker (sc-13130): `selectedId` is a single style
// id (a GROUP id OR a sub-style id) or null, and `onSelect(id)` reports the same. The parent owns
// the state; styleTextForId(id) resolves either id kind to the composer's free-text `prompt`.
//
// Copy is "Style" / "Style Catalog" — never "text style" (that label belongs to Krea's numeric
// `textStyleGain` slider elsewhere in ImageStudio, an unrelated control).
//
// Modeled on CompactSelector (the app's pill+menu switcher): outside-click + Escape close, a pill
// trigger with aria-expanded, and role="listbox"/role="option" items — plus the group cascade,
// breadcrumb, and search this catalog needs.
export function StylePicker({ groups = [], selectedId = null, onSelect, label = "Style", disabled = false }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  // Which group's styles are shown (level 2). null → the group list (level 1). Search overrides.
  const [activeGroupId, setActiveGroupId] = useState(null);
  const containerRef = useRef(null);
  const searchRef = useRef(null);

  // Resolve the current selection into a breadcrumb ({ kind, groupName, leafLabel, crumb, groupId })
  // so the pill and header can describe "where" the choice lives without the parent knowing the
  // catalog shape. Handles all three id kinds: none, a group's "overall", and a sub-style.
  const selection = useMemo(() => {
    if (!selectedId) {
      return { kind: "none", leafLabel: "None", crumb: null, groupId: null };
    }
    const group = groups.find((g) => g.id === selectedId);
    if (group) {
      return {
        kind: "general",
        groupName: group.name,
        leafLabel: `${group.name} (overall)`,
        crumb: `${group.name} — general`,
        groupId: group.id,
      };
    }
    for (const g of groups) {
      const style = g.styles.find((s) => s.id === selectedId);
      if (style) {
        return {
          kind: "style",
          groupName: g.name,
          leafLabel: style.name,
          crumb: `${g.name} › ${style.name}`,
          groupId: g.id,
        };
      }
    }
    return { kind: "none", leafLabel: "None", crumb: null, groupId: null };
  }, [groups, selectedId]);

  const activeGroup = useMemo(
    () => groups.find((g) => g.id === activeGroupId) ?? null,
    [groups, activeGroupId],
  );

  // Flat "search across all styles" index (sc-13171 optional affordance): every sub-style plus
  // each group's "overall" entry, each carrying its owning group name for the result breadcrumb.
  const searchResults = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return [];
    }
    const results = [];
    for (const group of groups) {
      const overallLabel = `${group.name} (overall)`;
      if (overallLabel.toLowerCase().includes(needle) || group.name.toLowerCase().includes(needle)) {
        results.push({ id: group.id, name: overallLabel, groupName: group.name });
      }
      for (const style of group.styles) {
        if (style.name.toLowerCase().includes(needle)) {
          results.push({ id: style.id, name: style.name, groupName: group.name });
        }
      }
    }
    return results;
  }, [groups, query]);

  const searching = query.trim().length > 0;

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

  // On open, jump straight to the current selection's group (level 2) so "change" is one step, and
  // move focus into the search box so the whole flow is keyboard-usable. On close, reset transient
  // navigation/search state.
  useEffect(() => {
    if (open) {
      setActiveGroupId(selection.groupId ?? null);
      searchRef.current?.focus();
    } else {
      setQuery("");
      setActiveGroupId(null);
    }
    // Only re-run when the menu opens/closes; selection.groupId is read as the initial target.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function choose(id) {
    onSelect?.(id);
    setOpen(false);
  }

  function renderOption({ id, strong, sub, active, key }) {
    return (
      <button
        aria-selected={active}
        className={active ? "compact-selector-item active" : "compact-selector-item"}
        key={key ?? id ?? "none"}
        onClick={() => choose(id)}
        role="option"
        title={strong}
        type="button"
      >
        <span className="compact-selector-label">
          <strong>{strong}</strong>
          {sub ? <span>{sub}</span> : null}
        </span>
      </button>
    );
  }

  const noneOption = renderOption({
    id: null,
    strong: "None",
    sub: "Pass-through — your prompt is sent unchanged",
    active: !selectedId,
    key: "__none__",
  });

  return (
    <div className="compact-selector style-picker" ref={containerRef}>
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={label}
        className="compact-selector-pill"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        title={selection.crumb ?? "None"}
        type="button"
      >
        <span className="compact-selector-meta">
          <strong>{selection.leafLabel}</strong>
          <span>{selection.crumb ?? "No style — pass-through"}</span>
        </span>
        <Icon.ChevDown className="chev" />
      </button>

      {open ? (
        <div className="compact-selector-menu style-picker-menu">
          <input
            aria-label="Search styles"
            className="style-picker-search"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search all styles…"
            ref={searchRef}
            type="search"
            value={query}
          />

          {searching ? (
            searchResults.length > 0 ? (
              <div className="style-picker-level" role="listbox" aria-label="Style search results">
                {searchResults.map((result) =>
                  renderOption({
                    id: result.id,
                    strong: result.name,
                    sub: result.groupName,
                    active: result.id === selectedId,
                    key: result.id,
                  }),
                )}
              </div>
            ) : (
              <p className="compact-selector-empty">No styles match “{query.trim()}”.</p>
            )
          ) : activeGroup ? (
            <>
              <div className="style-picker-crumb-row">
                <button
                  className="style-picker-back"
                  onClick={() => setActiveGroupId(null)}
                  type="button"
                >
                  <Icon.ChevDown className="chev-back" />
                  Groups
                </button>
                <span className="style-picker-crumb" aria-hidden="true">
                  {activeGroup.name}
                </span>
              </div>
              <div className="style-picker-level" role="listbox" aria-label={`${activeGroup.name} styles`}>
                {renderOption({
                  id: activeGroup.id,
                  strong: `${activeGroup.name} (overall)`,
                  sub: "Whole-group style",
                  active: activeGroup.id === selectedId,
                  key: `general-${activeGroup.id}`,
                })}
                {activeGroup.styles.map((style) =>
                  renderOption({
                    id: style.id,
                    strong: style.name,
                    active: style.id === selectedId,
                    key: style.id,
                  }),
                )}
              </div>
            </>
          ) : (
            <>
              <div className="style-picker-level" role="listbox" aria-label="Style Catalog">
                {noneOption}
              </div>
              <div className="style-picker-groups" role="group" aria-label="Style groups">
                {groups.map((group) => {
                  const isCurrent = group.id === selection.groupId;
                  return (
                    <button
                      aria-current={isCurrent ? "true" : undefined}
                      className={
                        isCurrent
                          ? "compact-selector-item style-picker-group-nav active"
                          : "compact-selector-item style-picker-group-nav"
                      }
                      key={group.id}
                      onClick={() => setActiveGroupId(group.id)}
                      title={`Browse ${group.name} styles`}
                      type="button"
                    >
                      <span className="compact-selector-label">
                        <strong>{group.name}</strong>
                        <span>{group.styles.length} styles</span>
                      </span>
                      <Icon.ChevDown className="chev-next" />
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
