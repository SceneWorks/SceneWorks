import React from "react";
import { AssetPickerField } from "../../components/AssetPicker.jsx";
import { TierPickerField } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { YueGenreTags } from "../audioYueControls.jsx";

export function AudioGenerationSettings({
  setModel,
  selectedModel,
  pickerModels,
  showVoice,
  setVoice,
  voice,
  voiceGroups,
  showLanguage,
  setLanguage,
  language,
  languages,
  showDuration,
  maxDurationSecs,
  setTargetDurationSecs,
  targetDurationSecs,
  showMusicFields,
  setBpm,
  bpm,
  setMusicalKey,
  musicalKey,
  songModel,
  setSegments,
  segments,
  songTierItems,
  handleTierChange,
  tierSwitching,
  quantTier,
  setGenreTags,
  genreTags,
  setLyrics,
  lyrics,
  showEditModes,
  audioAssets,
  setSourceAudioAssetId,
  sourceAudioAssetId,
  editModes,
  editMode,
  setEditMode,
  setEditRegionStartSecs,
  editRegionStartSecs,
  setEditRegionEndSecs,
  editRegionEndSecs,
  setEditStrength,
  editStrength
}) {
  return (
<div className="settings-bar audio-settings-bar">
              <div className="settings-bar-row">
                <label className="settings-field settings-field-model">
                  Model
                  <select
                    onChange={(event) => setModel(event.target.value)}
                    value={selectedModel?.id ?? ""}
                  >
                    {/* Models gated on the selected tab: only real generators that serve this mode. */}
                    {pickerModels.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name ?? item.ui?.label ?? item.id}
                      </option>
                    ))}
                  </select>
                </label>

                {/* Speech: the voice bank the selected model ships (audio.voices), grouped into
                    <optgroup>s by accent + gender (sc-13408). The buckets and their headings are
                    capability-driven — straight from voices[].accent / voices[].gender — never a
                    hardcoded taxonomy; a flat bank (no accent/gender) renders as bare options. */}
                {showVoice ? (
                  <label className="settings-field settings-field-voice">
                    Voice
                    <select onChange={(event) => setVoice(event.target.value)} value={voice}>
                      {voiceGroups.map((group) =>
                        group.label ? (
                          <optgroup key={group.key} label={group.label}>
                            {group.voices.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.label ?? item.id}
                              </option>
                            ))}
                          </optgroup>
                        ) : (
                          <React.Fragment key={group.key}>
                            {group.voices.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.label ?? item.id}
                              </option>
                            ))}
                          </React.Fragment>
                        ),
                      )}
                    </select>
                  </label>
                ) : null}

                {/* Language options the model advertises (audio.languages). */}
                {showLanguage ? (
                  <label className="settings-field settings-field-language">
                    Language
                    <select onChange={(event) => setLanguage(event.target.value)} value={language}>
                      {languages.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}

                {/* Length, capped to the model's audio.maxDurationSecs (never a hardcoded ceiling). */}
                {showDuration ? (
                  <label className="settings-field settings-field-duration">
                    Length (s)
                    <input
                      max={maxDurationSecs}
                      min="1"
                      onChange={(event) => setTargetDurationSecs(event.target.value)}
                      step="1"
                      type="number"
                      value={targetDurationSecs}
                    />
                  </label>
                ) : null}

                {/* Music: optional tempo (BPM) + musical key. Both ride the gen-core AudioParams music
                    sub-block ACE-Step reads; cleared ⇒ omitted so the model derives its own. */}
                {showMusicFields ? (
                  <label className="settings-field settings-field-bpm">
                    BPM
                    <input
                      min="1"
                      onChange={(event) => setBpm(event.target.value)}
                      placeholder="Optional"
                      step="1"
                      type="number"
                      value={bpm}
                    />
                  </label>
                ) : null}

                {showMusicFields ? (
                  <label className="settings-field settings-field-key">
                    Key
                    <input
                      onChange={(event) => setMusicalKey(event.target.value)}
                      placeholder="e.g. C minor"
                      type="text"
                      value={musicalKey}
                    />
                  </label>
                ) : null}

                {/* YuE (sc-19385): how many lyric sections to render (song length follows this and
                    the lyrics, not a duration) and the weight tier. */}
                {songModel ? (
                  <label className="settings-field settings-field-segments">
                    Sections
                    <input
                      max="100"
                      min="1"
                      onChange={(event) => setSegments(event.target.value)}
                      placeholder="2 (default)"
                      step="1"
                      type="number"
                      value={segments}
                    />
                  </label>
                ) : null}
                {songModel && songTierItems.length > 0 ? (
                  <TierPickerField
                    className="settings-field settings-field-tier"
                    items={songTierItems}
                    onChange={handleTierChange}
                    tierLabel={tierLabel}
                    tierSwitching={tierSwitching}
                    title="Which installed weight tier renders the song. Higher precision uses more memory."
                    value={quantTier}
                  />
                ) : null}
              </div>

              {songModel ? <YueGenreTags onChange={setGenreTags} tags={genreTags} /> : null}

              {/* Music: optional lyrics (free-form; empty ⇒ instrumental). Rides the AudioParams
                  `lyrics` field, distinct from the describe-the-music prompt. */}
              {showMusicFields ? (
                <label className="settings-field settings-field-lyrics">
                  Lyrics
                  <textarea
                    aria-label="Lyrics"
                    onChange={(event) => setLyrics(event.target.value)}
                    placeholder="Optional — [verse] / [chorus] tags supported; leave empty for instrumental"
                    value={lyrics}
                  />
                </label>
              ) : null}

              {/* Music extend/edit SOURCE band (sc-13410) — revealed ONLY when the selected model
                  advertises audio.editModes (Conditioning::AudioEdit). Mirrors VideoStudio's
                  `studio-source-band`: pick a library audio track + an edit mode the model advertises,
                  then (for a bounded inpaint/repaint) a region window. Extend reuses the Length field as
                  the new total length. Empty source ⇒ plain text-to-music (the band is optional). */}
              {showEditModes ? (
                <div className="studio-source-band">
                  <AssetPickerField
                    assets={audioAssets}
                    buttonLabel="Select audio"
                    changeLabel="Change"
                    emptyLabel="No source track selected"
                    label="Source track (extend / edit)"
                    onChange={setSourceAudioAssetId}
                    showCategories={false}
                    value={sourceAudioAssetId}
                  />
                  <div className="settings-bar-styles">
                    <span className="settings-bar-label">Edit</span>
                    <div className="preset-chips">
                      {editModes.map((value) => (
                        <button
                          className={editMode === value ? "preset-chip active" : "preset-chip"}
                          key={value}
                          onClick={() => setEditMode(value)}
                          type="button"
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </div>
                  {/* Inpaint / repaint bound a region (seconds) inside the source clip. Extend needs no
                      region here — its new total length is the Length field, and the worker begins the
                      appended tail at the source clip's own length. */}
                  {sourceAudioAssetId && (editMode === "inpaint" || editMode === "repaint") ? (
                    <div className="settings-bar-row">
                      <label className="settings-field settings-field-region-start">
                        Region start (s)
                        <input
                          min="0"
                          onChange={(event) => setEditRegionStartSecs(event.target.value)}
                          step="0.1"
                          type="number"
                          value={editRegionStartSecs}
                        />
                      </label>
                      <label className="settings-field settings-field-region-end">
                        Region end (s)
                        <input
                          min="0"
                          onChange={(event) => setEditRegionEndSecs(event.target.value)}
                          placeholder="To clip end"
                          step="0.1"
                          type="number"
                          value={editRegionEndSecs}
                        />
                      </label>
                    </div>
                  ) : null}
                  {/* Edit strength (0..=1) — an optional weight on the whole AudioEdit, part of the
                      Conditioning::AudioEdit contract. Cleared ⇒ the model default; a model that ignores
                      it (the ACE-Step turbo) simply falls back to its own behaviour. */}
                  {sourceAudioAssetId ? (
                    <label className="settings-field settings-field-edit-strength">
                      Edit strength
                      <input
                        max="1"
                        min="0"
                        onChange={(event) => setEditStrength(event.target.value)}
                        placeholder="Model default"
                        step="0.05"
                        type="number"
                        value={editStrength}
                      />
                    </label>
                  ) : null}
                </div>
              ) : null}
            </div>
  );
}
