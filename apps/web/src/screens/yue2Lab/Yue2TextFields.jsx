import { SessionOnlyText, Yue2LyricsField } from "./LabFields.jsx";
import React from "react";
import { toggleYue2StyleTag, yue2StyleTagSet } from "../../yue2Lab.js";
import { YueSuggestedTags } from "../audioYueControls.jsx";

export function Yue2TextFields({
  update,
  restoring,
  planWords,
  settings,
  sessionOnly
}) {
  return (
<div className="yue2-text-fields">
                <label>
                  Style
                  <textarea
                    aria-label="Style"
                    onChange={(event) => update({ style: event.target.value })}
                    placeholder="Genre, instruments, mood, vocal timbre…"
                    readOnly={restoring}
                    rows={2}
                    value={restoring ? (planWords?.style ?? "") : settings.style}
                  />
                </label>
                {!restoring ? (
                  <YueSuggestedTags
                    chosen={yue2StyleTagSet(settings.style)}
                    onToggle={(tag) => update({ style: toggleYue2StyleTag(settings.style, tag) })}
                  />
                ) : null}
                {!restoring && sessionOnly.has("style") ? <SessionOnlyText testId="yue2-style-session-only" /> : null}
                {restoring ? (
                  <label>
                    Lyrics
                    <textarea aria-label="Lyrics" readOnly rows={6} value={planWords?.lyrics ?? ""} />
                  </label>
                ) : (
                  <Yue2LyricsField onChange={(lyrics) => update({ lyrics })} value={settings.lyrics} />
                )}
                {!restoring && sessionOnly.has("lyrics") ? <SessionOnlyText testId="yue2-lyrics-session-only" /> : null}
                {restoring ? (
                  <p className="yue2-muted" data-testid="yue2-plan-words">
                    {planWords
                      ? "The saved plan renders its own style and lyrics. To change them, plan again."
                      : "Choose a saved plan — it renders its own style and lyrics."}
                  </p>
                ) : null}
              </div>
  );
}
