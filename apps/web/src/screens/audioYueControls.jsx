import React, { useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import {
  YUE_SECTION_LABELS,
  YUE_TAG_CATEGORIES,
  addGenreTags,
  allYueTagSuggestions,
  parseGenreTags,
  yueTagSuggestions,
} from "../yueSong.js";

// YuE lyrics-to-song controls for the Audio Studio's Music tab (epic sc-19373, sc-19385). Rendered
// ONLY when the selected music model sings segmented lyrics (audio.supportsSegmentedLyrics); the
// ACE-Step controls they replace are untouched. Presentation reuses the multi-speaker script
// editor's surfaces (.script-segment*) and the shared .preset-chip pill so both themes hold.

/** Section-labelled lyrics editor: one row per song section, each a [label | text | remove] row. */
export function YueLyricsEditor({ sections, onChange }) {
  const update = (index, patch) =>
    onChange(sections.map((section, i) => (i === index ? { ...section, ...patch } : section)));
  const remove = (index) => onChange(sections.length > 1 ? sections.filter((_, i) => i !== index) : sections);
  // A new section alternates verse → chorus, the shape most songs follow.
  const add = () =>
    onChange([
      ...sections,
      { label: sections[sections.length - 1]?.label === "verse" ? "chorus" : "verse", text: "" },
    ]);
  return (
    <div
      aria-label="Lyrics"
      className="prompt-input multi-speaker-script yue-lyrics-editor"
      data-testid="yue-lyrics-editor"
      role="group"
    >
      {sections.map((section, index) => {
        const labels = YUE_SECTION_LABELS.includes(section.label)
          ? YUE_SECTION_LABELS
          : [...YUE_SECTION_LABELS, section.label];
        return (
          <div className="script-segment" key={index}>
            <select
              aria-label={`Section ${index + 1} label`}
              className="script-segment-speaker"
              onChange={(event) => update(index, { label: event.target.value })}
              value={section.label}
            >
              {labels.map((label) => (
                <option key={label} value={label}>
                  [{label}]
                </option>
              ))}
            </select>
            <div className="script-segment-content">
              <textarea
                aria-label={`Section ${index + 1} lyrics`}
                className="script-segment-text"
                onChange={(event) => update(index, { text: event.target.value })}
                placeholder="Four or so lines fill one ~30 s section…"
                rows={3}
                value={section.text ?? ""}
              />
            </div>
            <button
              aria-label={`Remove section ${index + 1}`}
              className="script-segment-remove"
              disabled={sections.length <= 1}
              onClick={() => remove(index)}
              title="Remove this section"
              type="button"
            >
              <Icon.Trash size={14} />
            </button>
          </div>
        );
      })}
      <button className="script-add-segment" data-testid="yue-add-section" onClick={add} type="button">
        <Icon.Plus size={14} />
        Add section
      </button>
    </div>
  );
}

/**
 * Genre tags — YuE's prompt. Chosen tags as removable chips, a free-text input (Enter / comma adds;
 * the datalist offers upstream's top-200 tags), and a default-COLLAPSED "Suggested tags" browser
 * with upstream's five categories.
 */
export function YueGenreTags({ tags, onChange }) {
  const [draft, setDraft] = useState("");
  const [category, setCategory] = useState(YUE_TAG_CATEGORIES[0].id);
  const everySuggestion = useMemo(() => allYueTagSuggestions(), []);
  const suggestions = useMemo(() => yueTagSuggestions(category), [category]);
  const chosen = new Set(tags.map((tag) => tag.toLowerCase()));

  const commitDraft = () => {
    const parsed = parseGenreTags(draft);
    if (parsed.length) {
      onChange(addGenreTags(tags, parsed));
    }
    setDraft("");
  };
  const toggle = (tag) =>
    onChange(
      chosen.has(tag.toLowerCase())
        ? tags.filter((item) => item.toLowerCase() !== tag.toLowerCase())
        : addGenreTags(tags, [tag]),
    );

  return (
    <div className="settings-field yue-genre-tags" data-testid="yue-genre-tags" role="group" aria-label="Genre tags">
      <span className="field-label">Genre tags</span>
      {tags.length ? (
        <div className="preset-chips yue-genre-tags__chosen">
          {tags.map((tag) => (
            <span className="preset-chip saved-voice-chip is-selected" key={tag}>
              <span>{tag}</span>
              <button
                aria-label={`Remove tag ${tag}`}
                className="saved-voice-delete"
                onClick={() => toggle(tag)}
                title="Remove tag"
                type="button"
              >
                <Icon.Close />
              </button>
            </span>
          ))}
        </div>
      ) : null}
      <input
        aria-label="Add genre tags"
        list="yue-genre-tag-suggestions"
        onBlur={commitDraft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === ",") {
            event.preventDefault();
            commitDraft();
          }
        }}
        placeholder="genre, instrument, mood, vocal gender, timbre — e.g. pop, piano, uplifting, female, airy vocal"
        type="text"
        value={draft}
      />
      <datalist id="yue-genre-tag-suggestions">
        {everySuggestion.map((tag) => (
          <option key={tag} value={tag} />
        ))}
      </datalist>
      <details className="yue-genre-tags__suggested">
        <summary>Suggested tags</summary>
        <div className="preset-chips" role="group" aria-label="Tag category">
          {YUE_TAG_CATEGORIES.map((item) => (
            <button
              aria-pressed={item.id === category}
              className={item.id === category ? "preset-chip active" : "preset-chip"}
              key={item.id}
              onClick={() => setCategory(item.id)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="preset-chips yue-genre-tags__suggestions" data-testid="yue-tag-suggestions">
          {suggestions.map((tag) => (
            <button
              aria-pressed={chosen.has(tag.toLowerCase())}
              className={chosen.has(tag.toLowerCase()) ? "preset-chip active" : "preset-chip"}
              key={tag}
              onClick={() => toggle(tag)}
              type="button"
            >
              {tag}
            </button>
          ))}
        </div>
        <span className="field-hint" role="note">
          Suggestions: YuE&apos;s published top-200 tags (M-A-P / HKUST, Apache-2.0).
        </span>
      </details>
    </div>
  );
}

/**
 * The ICL reference band — a reference song for an in-context-learning (`_icl`) checkpoint: one
 * mixed clip, or a vocal + instrumental pair, and the window of it to use. Every control sits in a
 * <fieldset disabled> unless the selected checkpoint is ICL, so a CoT model can never send one.
 */
export function YueReferenceBand({
  enabled,
  showRegion,
  audioAssets,
  iclMode,
  onIclModeChange,
  referenceAssetId,
  onReferenceChange,
  vocalAssetId,
  onVocalChange,
  instrumentalAssetId,
  onInstrumentalChange,
  startSecs,
  onStartChange,
  endSecs,
  onEndChange,
  windowError = "",
}) {
  return (
    <fieldset className="studio-source-band yue-reference-band" data-testid="yue-reference-band" disabled={!enabled}>
      <div className="settings-bar-styles">
        <span className="settings-bar-label">Reference</span>
        <div className="preset-chips">
          {[
            ["single", "Single track"],
            ["dual", "Vocal + instrumental"],
          ].map(([value, label]) => (
            <button
              aria-pressed={iclMode === value}
              className={iclMode === value ? "preset-chip active" : "preset-chip"}
              key={value}
              onClick={() => onIclModeChange(value)}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      {iclMode === "dual" ? (
        <>
          <AssetPickerField
            assets={audioAssets}
            buttonLabel="Select vocal track"
            emptyLabel="No vocal track selected"
            label="Vocal track"
            onChange={onVocalChange}
            showCategories={false}
            value={vocalAssetId}
          />
          <AssetPickerField
            assets={audioAssets}
            buttonLabel="Select instrumental track"
            emptyLabel="No instrumental track selected"
            label="Instrumental track"
            onChange={onInstrumentalChange}
            showCategories={false}
            value={instrumentalAssetId}
          />
        </>
      ) : (
        <AssetPickerField
          assets={audioAssets}
          buttonLabel="Select reference song"
          emptyLabel="No reference song selected"
          label="Reference song"
          onChange={onReferenceChange}
          showCategories={false}
          value={referenceAssetId}
        />
      )}
      {showRegion ? (
        <div className="settings-bar-row">
          <label className="settings-field settings-field-icl-start">
            Reference start (s)
            <input
              min="0"
              onChange={(event) => onStartChange(event.target.value)}
              placeholder="0"
              step="0.1"
              type="number"
              value={startSecs}
            />
          </label>
          <label className="settings-field settings-field-icl-end">
            Reference end (s)
            <input
              min="0"
              onChange={(event) => onEndChange(event.target.value)}
              placeholder="30"
              step="0.1"
              type="number"
              value={endSecs}
            />
          </label>
        </div>
      ) : null}
      {showRegion && windowError ? (
        <p className="inline-warning" data-testid="yue-icl-window-error" role="status">
          {windowError}
        </p>
      ) : null}
      <p className="helper-copy">
        {enabled
          ? "The song follows the reference clip's style. A vocal + instrumental pair usually steers better than one mixed track; a chorus is usually the most useful window."
          : "Reference songs need an ICL checkpoint — pick a YuE … ICL model to use one."}
      </p>
    </fieldset>
  );
}
