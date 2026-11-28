import React from "react";
import { showTransparencyToggle, transparencyPromptSuggestion, TRANSPARENCY_PROMPT_HINT } from "../../qwenAlpha.js";
import { upscaleEngineHasSoftness, upscaleEngineIsRestoration, upscaleFactorsForEngine } from "../../upscaleEngines.js";
import { irisRestoreControls } from "../../irisRestore.js";
import { IrisRestoreControls } from "../../components/IrisRestoreControls.jsx";
import { InfoTip } from "../../components/InfoTip.jsx";

// Settings tab › Upscale & finish: what happens after the denoise — upscale / restoration,
// hires.fix, the decoder (alternate decoders, PiD), and RGBA output. Same controls and
// conditions as the Advanced section they came from.
export function ImageFinishSettings({
  selectedModel,
  supportsGuidance,
  transparentBackground,
  setTransparentBackground,
  prompt,
  setPromptFromUser,
  showPidToggle,
  usePid,
  upscaleEnabled,
  hiresFixEnabled,
  setUsePid,
  setDecoder,
  setPidTarget,
  pidTarget,
  pidDecodeNotice,
  showDecoderPicker,
  decoder,
  decoderOptions,
  setUpscaleEnabled,
  setHiresFixEnabled,
  upscaleEngine,
  setUpscaleFactor,
  upscaleFactor,
  handleUpscaleEngineChange,
  availableUpscaleEngines,
  setUpscaleRestore,
  setRestorePlanStatus,
  restorePlanBody,
  token,
  upscaleRestore,
  setUpscaleSoftness,
  upscaleSoftness,
  hiresFixDisabledReason,
  hiresFixDisabled,
  setHiresFixSteps,
  hiresFixSteps,
  setHiresFixDenoisingStrength,
  hiresFixDenoisingStrength,
  setHiresFixUpscaleBy,
  hiresFixUpscaleBy,
  hiresFixTargetInvalid,
  hiresFixTargetWidth,
  hiresFixTargetHeight,
  setHiresFixCfgScale,
  hiresFixCfgScale,
}) {
  return (
    <div className="studio-grid image-finish-settings">
      <label
        className="checkline upscale-toggle"
        title={usePid ? "Disabled while the PiD decoder is on — PiD already super-resolves to 2K/4K." : undefined}
      >
        <input
          checked={upscaleEnabled}
          disabled={usePid}
          onChange={(event) => {
            const enabled = event.target.checked;
            setUpscaleEnabled(enabled);
            if (enabled) {
              setHiresFixEnabled(false);
            }
          }}
          type="checkbox"
        />
        Upscale
      </label>
      {upscaleEngineIsRestoration(upscaleEngine) ? null : (
      <label>
        Scale
        <select disabled={!upscaleEnabled || usePid} onChange={(event) => setUpscaleFactor(Number(event.target.value))} value={upscaleFactor}>
          {upscaleFactorsForEngine(upscaleEngine).map((factor) => (
            <option key={factor} value={factor}>
              {factor}x
            </option>
          ))}
        </select>
      </label>
      )}
      <label>
        Engine
        <select disabled={!upscaleEnabled || usePid} onChange={(event) => handleUpscaleEngineChange(event.target.value)} value={upscaleEngine}>
          {availableUpscaleEngines.map((engine) => (
            <option key={engine.key} value={engine.key}>
              {engine.label}
            </option>
          ))}
        </select>
      </label>
      {upscaleEngineHasSoftness(upscaleEngine) ? (
        <label title="Higher restores more detail from a degraded source; 0 keeps it faithful.">
          Detail
          <input
            aria-label="SeedVR2 detail (softness)"
            disabled={!upscaleEnabled || usePid}
            max="1"
            min="0"
            onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
            step="0.05"
            type="range"
            value={upscaleSoftness}
          />
          <span>{upscaleSoftness.toFixed(2)}</span>
        </label>
      ) : null}
      {upscaleEngineIsRestoration(upscaleEngine) && upscaleEnabled && !usePid ? (
        // Iris restoration (sc-25683): scale / input sizing / colour fix, with the engine
        // planner's processing + output dimensions for THIS generation's size shown before
        // submission (a budgeted result is labelled as such).
        <div className="span-full image-finish-restore">
          <IrisRestoreControls
            idPrefix="studio-iris"
            onChange={(next) => setUpscaleRestore(irisRestoreControls(next))}
            onStatus={setRestorePlanStatus}
            planBody={restorePlanBody}
            token={token}
            value={upscaleRestore}
          />
        </div>
      ) : null}
      <label
        className="checkline hires-fix-toggle"
        title={hiresFixDisabledReason}
      >
        <input
          checked={hiresFixEnabled}
          disabled={hiresFixDisabled}
          onChange={(event) => {
            const enabled = event.target.checked;
            setHiresFixEnabled(enabled);
            if (enabled) {
              setUpscaleEnabled(false);
            }
          }}
          type="checkbox"
        />
        Hires.fix
      </label>
      {hiresFixEnabled ? (
        <>
          <label title="0 inherits the first-pass step count.">
            Hires Steps
            <input
              aria-label="Hires Steps"
              max="150"
              min="0"
              onChange={(event) => setHiresFixSteps(Number(event.target.value))}
              step="1"
              type="number"
              value={hiresFixSteps}
            />
          </label>
          <label title="How strongly the high-resolution pass may redraw the upscaled first pass.">
            Denoising Strength
            <input
              aria-label="Hires Denoising Strength"
              max="1"
              min="0"
              onChange={(event) =>
                setHiresFixDenoisingStrength(Number(event.target.value))
              }
              step="0.05"
              type="range"
              value={hiresFixDenoisingStrength}
            />
            <span>{hiresFixDenoisingStrength.toFixed(2)}</span>
          </label>
          <label>
            Upscale by
            <select
              aria-label="Hires Upscale by"
              onChange={(event) => setHiresFixUpscaleBy(Number(event.target.value))}
              value={hiresFixUpscaleBy}
            >
              {[1.25, 1.5, 1.75, 2, 2.5, 3, 4].map((factor) => (
                <option key={factor} value={factor}>
                  {factor}x
                </option>
              ))}
            </select>
            <span
              className="field-hint"
              role={hiresFixTargetInvalid ? "alert" : undefined}
            >
              Output {hiresFixTargetWidth}×{hiresFixTargetHeight}
              {hiresFixTargetInvalid ? " · exceeds 4096px limit" : ""}
            </span>
          </label>
          <label
            title={
              supportsGuidance
                ? "Blank inherits the first-pass CFG scale."
                : "This model does not use CFG."
            }
          >
            Hires CFG Scale
            <input
              aria-label="Hires CFG Scale"
              disabled={!supportsGuidance}
              max="60"
              min="0"
              onChange={(event) => setHiresFixCfgScale(event.target.value)}
              placeholder="Inherit"
              step="0.1"
              type="number"
              value={supportsGuidance ? hiresFixCfgScale : ""}
            />
          </label>
        </>
      ) : null}
      {showDecoderPicker ? (
        <label className="alternate-decoder-select span-2">
          <span className="studio-field-label">
            Decoder <span className="badge">Experimental</span>
            <InfoTip label="About Decoder">
              Alternate decode only; the model, text encoder, and native VAE remain installed. Output licensing
              appends the selected decoder component.
            </InfoTip>
          </span>
          <select
            onChange={(event) => {
              const next = event.target.value;
              setDecoder(next);
              if (next !== "native") setUsePid(false);
            }}
            value={decoder}
          >
            <option value="native">Native VAE (default)</option>
            {decoderOptions.map((option) => {
              const mib = option.estimatedSizeBytes
                ? Math.round(option.estimatedSizeBytes / (1024 * 1024))
                : null;
              return (
                <option disabled={!option.available} key={option.id} value={option.id}>
                  {option.label}{mib ? ` (+${mib} MiB)` : ""}{option.available ? "" : " — install component"}
                </option>
              );
            })}
          </select>
        </label>
      ) : null}
      {showPidToggle ? (
        <>
          <label
            className="checkline pid-decoder-toggle"
            title="Decode this generation through NVIDIA's PiD pixel-diffusion decoder instead of the model's VAE: it decodes and super-resolves in one pass to 2K or 4K (pick the tier at right — sharper detail, but slower and more memory). Non-commercial use only — PiD output is licensed for research/evaluation, unlike the rest of the pipeline. Off = the model's native VAE at the selected resolution."
          >
            <input
              checked={usePid}
              disabled={upscaleEnabled || hiresFixEnabled}
              onChange={(event) => {
                const enabled = event.target.checked;
                setUsePid(enabled);
                if (enabled) setDecoder("native");
              }}
              type="checkbox"
            />
            PiD decoder <span className="badge badge-nc">Non-Commercial</span>
          </label>
          {usePid ? (
            <label
              className="pid-target-select"
              title="PiD super-resolves the base render 4×, so this sets the output size: 4K (~4096px, max detail) or 2K (~2048px, faster and less GPU memory). Both are super-resolved from the model's latent."
            >
              Output
              <select
                onChange={(event) => setPidTarget(event.target.value)}
                value={pidTarget}
              >
                <option value="4k">4K · max detail</option>
                <option value="2k">2K · faster</option>
              </select>
              {pidDecodeNotice ? (
                <span className="field-hint pid-decode-hint" role="status">
                  {pidDecodeNotice.message}
                </span>
              ) : null}
            </label>
          ) : null}
        </>
      ) : null}
      {/* sc-24113 — native transparency. Rendered only when the selected model advertises
          four-channel decode (`supportsAlphaOutput`), because a cut-out toggle on a model
          that can only emit opaque RGB is a control that always fails. Hidden rather than
          disabled: a control that can never apply here is noise, not information. */}
      {showTransparencyToggle(selectedModel) ? (
        <label
          className="checkline transparency-toggle span-2"
          title="Keep the model's alpha channel instead of compositing the render onto white, and save the result as an RGBA PNG — so cut-outs, logos and stickers keep real transparency through editing, layering and export. The model has no transparency mode: ASK for a transparent background in the prompt, and use this to keep it."
        >
          <input
            checked={transparentBackground}
            onChange={(event) => setTransparentBackground(event.target.checked)}
            type="checkbox"
          />
          Transparent background (RGBA)
        </label>
      ) : null}
      {/* The other half of the control, and the non-obvious one. There is no transparency
          MODE in this model: the toggle only keeps the alpha channel, and whether that
          channel holds a cut-out is decided by the PROMPT. Without this the toggle hands
          back a fully opaque RGBA PNG and reads as broken.

          Offered, never applied: the wording goes into the prompt the user can see and
          edit, on a button they press. `transparencyPromptSuggestion` returns null once the
          prompt already says it, so re-running never accretes the sentence. */}
      {transparencyPromptSuggestion(prompt, transparentBackground) ? (
        <p className="transparency-prompt-hint span-full">
          This model has no transparency mode — ask for it in the prompt.{" "}
          <button
            className="hero-link"
            onClick={() =>
              setPromptFromUser(
                transparencyPromptSuggestion(prompt, transparentBackground),
              )
            }
            type="button"
          >
            Add “{TRANSPARENCY_PROMPT_HINT}”
          </button>
        </p>
      ) : null}
    </div>
  );
}
