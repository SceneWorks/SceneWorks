import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { AssetPickerField } from "../../components/AssetPicker.jsx";

export function AudioVoiceReferencePanel({
  audioAssets,
  setReferenceAudioAssetId,
  referenceAudioAssetId,
  savedVoices,
  selectedSavedVoiceId,
  handleSelectSavedVoice,
  deletingVoiceId,
  handleDeleteSavedVoice,
  createSavedVoice,
  setSavedVoiceName,
  savedVoiceName,
  canRegisterVoice,
  handleRegisterVoice,
  registeringVoice,
  savedVoiceNotice,
  showMatchStrength,
  setMatchStrength,
  matchStrength,
  selectedIsNativeClone
}) {
  return (
<div className="studio-source-band">
                <AssetPickerField
                  assets={audioAssets}
                  buttonLabel="Select reference voice"
                  changeLabel="Change"
                  emptyLabel="No reference voice selected"
                  label="Reference voice"
                  onChange={setReferenceAudioAssetId}
                  showCategories={false}
                  value={referenceAudioAssetId}
                />
                {savedVoices.length > 0 ? (
                  <div className="saved-voices" role="group" aria-label="Saved voices">
                    <span className="field-label">Saved voices</span>
                    <div className="preset-chips saved-voices-chips">
                      {savedVoices.map((voice) => (
                        <span
                          key={voice.id}
                          className={`preset-chip saved-voice-chip${
                            selectedSavedVoiceId === voice.id ? " is-selected" : ""
                          }`}
                        >
                          <button
                            type="button"
                            className="saved-voice-select"
                            aria-pressed={selectedSavedVoiceId === voice.id}
                            onClick={() => handleSelectSavedVoice(voice)}
                          >
                            {voice.name}
                          </button>
                          <button
                            type="button"
                            className="saved-voice-delete"
                            aria-label={`Delete saved voice ${voice.name}`}
                            disabled={Boolean(deletingVoiceId)}
                            title="Delete saved voice"
                            onClick={() => handleDeleteSavedVoice(voice)}
                          >
                            <Icon.Trash />
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}
                {createSavedVoice ? (
                  <div className="register-voice">
                    <label className="settings-field">
                      Save this reference as a voice
                      <div className="register-voice-row">
                        <input
                          onChange={(event) => setSavedVoiceName(event.target.value)}
                          placeholder="Voice name (e.g. Narrator)"
                          type="text"
                          value={savedVoiceName}
                        />
                        <button
                          className="secondary-action register-voice-save"
                          disabled={!canRegisterVoice}
                          onClick={handleRegisterVoice}
                          type="button"
                        >
                          <Icon.Save />
                          {registeringVoice ? "Saving…" : "Save voice"}
                        </button>
                      </div>
                      <span className="field-hint" role="note">
                        Registers the selected reference clip as a reusable named voice. We compute its
                        speaker fingerprint and warn if it closely matches a voice you already saved.
                      </span>
                    </label>
                    {savedVoiceNotice ? (
                      <p
                        className={
                          savedVoiceNotice.tone === "warning" ? "inline-warning" : "field-hint"
                        }
                        role="status"
                      >
                        {savedVoiceNotice.message}
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {showMatchStrength ? (
                  <label className="settings-field settings-field-match-strength">
                    Match strength
                    <input
                      max="1"
                      min="0"
                      onChange={(event) => setMatchStrength(event.target.value)}
                      placeholder="0.3 (default)"
                      step="0.05"
                      type="number"
                      value={matchStrength}
                    />
                    <span className="field-hint" role="note">
                      OpenVoice tone-color sampling temperature (τ). Cleared → model default (0.3).
                    </span>
                  </label>
                ) : null}
                <p className="helper-copy">
                  {selectedIsNativeClone
                    ? "Your script is rendered directly in the reference clip’s voice in a single step. Record or import a reference clip into the library to use it here."
                    : "The base voice speaks your script, then OpenVoice V2 converts it to match the reference clip’s voice. Record or import a reference clip into the library to use it here."}
                </p>
              </div>
  );
}
