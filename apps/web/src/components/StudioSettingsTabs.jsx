import React, { useId } from "react";
import { Icon } from "./Icons.jsx";
import "./StudioSettingsTabs.css";

// StudioSettingsTabs — the settings strip along the bottom of a studio's compose panel
// (studio layout simplification). Each tab names a group of settings and shows a one-line
// summary of its current values; picking a tab opens that group's panel directly underneath,
// picking the open tab closes it. One panel at a time, and only the open panel is mounted —
// the same lazy-render contract the Advanced disclosure it replaces had.
//
// `tabs`: [{ id, label, summary?, changed?, attention?, attentionLabel?, content }]
//   - `changed` draws the accent dot (a value differs from the model default),
//   - `attention` draws the warn dot (e.g. a model update is available).
// Tabs whose `content` is null/false are dropped, so a screen can declare every group and
// let availability decide which ones show.
// `active`: the open tab id, or null for all closed. `onChange(id | null)`.
export function StudioSettingsTabs({ tabs, active, onChange, label = "Settings", className }) {
  const baseId = useId();
  const visible = tabs.filter((tab) => tab && tab.content);
  const open = visible.find((tab) => tab.id === active) ?? null;
  return (
    <div className={className ? `studio-settings-tabs ${className}` : "studio-settings-tabs"}>
      <div aria-label={label} className="studio-settings-tablist" role="tablist">
        {visible.map((tab) => {
          const selected = open?.id === tab.id;
          return (
            <button
              aria-controls={selected ? `${baseId}-${tab.id}` : undefined}
              aria-expanded={selected}
              aria-selected={selected}
              className={selected ? "studio-settings-tab on" : "studio-settings-tab"}
              data-tab={tab.id}
              id={`${baseId}-${tab.id}-tab`}
              key={tab.id}
              onClick={() => onChange(selected ? null : tab.id)}
              role="tab"
              type="button"
            >
              <span className="studio-settings-tab-title">
                <span className="studio-settings-tab-label">{tab.label}</span>
                {tab.attention ? (
                  <span className="studio-settings-dot warn" role="img" aria-label={tab.attentionLabel ?? "Needs attention"} />
                ) : null}
                {tab.changed ? <span className="studio-settings-dot" role="img" aria-label="Changed from default" /> : null}
                <Icon.ChevDown className={selected ? "studio-settings-tab-caret open" : "studio-settings-tab-caret"} />
              </span>
              {tab.summary ? <span className="studio-settings-tab-summary">{tab.summary}</span> : null}
            </button>
          );
        })}
      </div>
      {open ? (
        <div
          aria-labelledby={`${baseId}-${open.id}-tab`}
          className="studio-settings-panel"
          data-panel={open.id}
          id={`${baseId}-${open.id}`}
          role="tabpanel"
        >
          {open.content}
        </div>
      ) : null}
    </div>
  );
}

// A heading that starts a sub-group inside a settings panel ("For this model").
export function StudioSettingsGroup({ label, children }) {
  return (
    <div className="studio-settings-group">
      <span className="eyebrow studio-settings-group-label">{label}</span>
      {children}
    </div>
  );
}
