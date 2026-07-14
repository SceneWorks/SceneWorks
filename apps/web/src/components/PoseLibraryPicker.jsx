import React, { useState } from "react";
import { usePoseLibrary } from "../poseLibrary.js";

// Multi-select gallery of OpenPose poses, grouped by category. Controlled: the parent
// owns `selectedIds` (array) and gets toggles via `onToggle(id)` / `onClear()`. Shared
// by the Character Studio pose panel and the Image Studio pose sections.
//
// Two layouts:
//   * default — the whole library as one vertical scroll, a labelled grid per category.
//   * `categoryFilter` — one category grid at a time behind a wrapping chip row (design
//     handoff sc-8245 Fix 2). Selections still persist ACROSS categories; only the visible
//     grid swaps. Used by the Image Studio Structure Control panel to cap the panel height.
export function PoseLibraryPicker({ selectedIds = [], onToggle, onClear, loadUserPoses, categoryFilter = false }) {
  const { poses, categories, loading, error } = usePoseLibrary({ loadUserPoses });
  // Which chip's grid is shown (categoryFilter layout only). Default to the design's
  // "standing" when present, else the first available category.
  const [activeCategory, setActiveCategory] = useState("standing");
  const selected = new Set(selectedIds);

  if (loading) {
    return <p className="muted">Loading pose library…</p>;
  }
  if (error) {
    return <p className="inline-warning">Pose library unavailable: {error}</p>;
  }
  if (!poses.length) {
    return <p className="inline-warning">No poses found in the library.</p>;
  }

  const toolbar = (
    <div className="pose-library-toolbar">
      <span className="muted">
        {selected.size ? `${selected.size} pose${selected.size === 1 ? "" : "s"} selected` : "Select one or more poses"}
      </span>
      {selected.size ? (
        <button className="link-button" onClick={onClear} type="button">
          Clear
        </button>
      ) : null}
    </div>
  );

  // `showCheck` gates the top-right ✓ badge to the categoryFilter layout — the default (flat)
  // layout leaves the .pose-thumb markup untouched so the shared call sites are unchanged.
  const renderThumb = (pose, showCheck = false) => {
    const isSelected = selected.has(pose.id);
    return (
      <button
        aria-label={`${isSelected ? "Deselect" : "Select"} pose ${pose.label}`}
        aria-pressed={isSelected}
        className={isSelected ? "pose-thumb selected" : "pose-thumb"}
        key={pose.id}
        onClick={() => onToggle?.(pose.id)}
        title={pose.label}
        type="button"
      >
        <img alt={pose.label} loading="lazy" src={pose.previewUrl ?? `/${pose.preview}`} />
        <span className="pose-thumb-label">{pose.label}</span>
        {showCheck && isSelected ? (
          <span aria-hidden="true" className="pose-thumb-check">
            ✓
          </span>
        ) : null}
      </button>
    );
  };

  if (categoryFilter) {
    // Derive a display label per category from its first pose ("T-Pose 01" → "T-Pose"), so the
    // chip/heading read nicely without a hardcoded map (user categories degrade gracefully).
    const labelFor = (category) => {
      const sample = poses.find((pose) => pose.category === category);
      return sample?.label?.replace(/\s*\d+$/, "").trim() || category;
    };
    const active = categories.includes(activeCategory) ? activeCategory : categories[0];
    const activePoses = poses.filter((pose) => pose.category === active);

    return (
      <div className="pose-library category-filter">
        {toolbar}
        <div className="pose-chip-row">
          {categories.map((category) => {
            const inCategory = poses.filter((pose) => pose.category === category);
            const selCount = inCategory.filter((pose) => selected.has(pose.id)).length;
            const isActive = category === active;
            return (
              <button
                aria-pressed={isActive}
                className={isActive ? "pose-chip active" : "pose-chip"}
                key={category}
                onClick={() => setActiveCategory(category)}
                type="button"
              >
                <span className="pose-chip-label">{labelFor(category)}</span>
                <span className="pose-chip-count">{inCategory.length}</span>
                {selCount ? <span className="pose-chip-badge">{selCount}</span> : null}
              </button>
            );
          })}
        </div>
        <div className="pose-category-head">
          <span className="pose-category-name">{labelFor(active)}</span>
          <span className="pose-category-count">
            {activePoses.length} pose{activePoses.length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="pose-grid">{activePoses.map((pose) => renderThumb(pose, true))}</div>
      </div>
    );
  }

  return (
    <div className="pose-library">
      {toolbar}
      {categories.map((category) => {
        const inCategory = poses.filter((pose) => pose.category === category);
        if (!inCategory.length) {
          return null;
        }
        return (
          <div className="pose-category" key={category}>
            <p className="eyebrow">{category}</p>
            <div className="pose-grid">{inCategory.map((pose) => renderThumb(pose))}</div>
          </div>
        );
      })}
    </div>
  );
}
