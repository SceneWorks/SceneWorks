import React from "react";
import { ACCENTS } from "../../accents.js";
import { GENERATION_QUALITY_OPTIONS, generationQualityLabel, shortQualityLabel } from "../../generationQuality.js";
import { Icon } from "../../components/Icons.jsx";

export function AppearanceTab({
  accent,
  changeDefaultQuality,
  changeTheme,
  defaultQuality,
  lockedToSimple,
  onAccentChange,
  onSimpleDefaultChange,
  simpleDefault,
  theme,
}) {
  return (
<div className="settings-tab-body">
            <div className="settings-group-title">Theme</div>
            <div className="settings-row">
              <span className="settings-row-title">Mode</span>
              <div className="settings-segment" role="radiogroup" aria-label="Theme">
                <button
                  aria-checked={theme === "light"}
                  className={theme === "light" ? "active" : ""}
                  onClick={() => changeTheme("light")}
                  role="radio"
                  type="button"
                >
                  <Icon.Sun size={14} />
                  Light
                </button>
                <button
                  aria-checked={theme === "dark"}
                  className={theme === "dark" ? "active" : ""}
                  onClick={() => changeTheme("dark")}
                  role="radio"
                  type="button"
                >
                  <Icon.Moon size={14} />
                  Dark
                </button>
              </div>
            </div>
            <div className="settings-row settings-row--top">
              <div>
                <div className="settings-row-title">Accent</div>
                <div className="settings-row-sub">Recolors the whole app — chips, buttons, the mark.</div>
              </div>
              <div className="settings-accents" role="radiogroup" aria-label="Accent">
                {ACCENTS.map((entry) => (
                  <button
                    aria-checked={accent === entry.id}
                    aria-label={entry.name}
                    className={
                      accent === entry.id ? "settings-accent-swatch active" : "settings-accent-swatch"
                    }
                    key={entry.id}
                    onClick={() => onAccentChange?.(entry.id)}
                    role="radio"
                    style={{ background: entry.swatch }}
                    type="button"
                  />
                ))}
              </div>
            </div>

            <div className="work-panel-divider" />
            <div className="settings-group-title">Simple mode</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Use Simple mode by default</div>
                <div className="settings-row-sub">
                  {lockedToSimple
                    ? "Phones always use it."
                    : "Phones always use it; the sidebar switch overrides it for this session."}
                </div>
              </div>
              <button
                aria-checked={simpleDefault}
                aria-label="Use Simple mode by default"
                className={simpleDefault ? "settings-toggle on" : "settings-toggle"}
                onClick={() => onSimpleDefaultChange?.(!simpleDefault)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>

            <div className="work-panel-divider" />
            <div className="settings-group-title">Generation</div>
            <div>
              <div className="settings-row-title">Default quality</div>
              {/* SHORT labels (the tier keys) — the descriptive wording ("Auto (best that fits this
                  Mac)") doesn't fit a 34px chip, so it rides the title attribute and the line below. */}
              <div className="settings-chips" role="radiogroup" aria-label="Default generation quality">
                {GENERATION_QUALITY_OPTIONS.map((value) => (
                  <button
                    aria-checked={defaultQuality === value}
                    className={defaultQuality === value ? "settings-chip active" : "settings-chip"}
                    key={value}
                    onClick={() => changeDefaultQuality(value)}
                    role="radio"
                    title={generationQualityLabel(value)}
                    type="button"
                  >
                    {shortQualityLabel(value)}
                  </button>
                ))}
              </div>
              <div className="settings-row-sub">{generationQualityLabel(defaultQuality)}</div>
              <p className="settings-note settings-note--spaced">
                The tier new generations use for a model you haven’t picked one for yet.{" "}
                <strong>Auto</strong> takes the highest-fidelity tier that fits this machine for each
                model. Per-model picks in a studio always win, and are remembered.
              </p>
            </div>
          </div>
  );
}
