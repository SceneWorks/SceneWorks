import React from "react";

export function AudioAdvancedPanel({
  setSeed,
  seed,
  showGuidance,
  setGuidance,
  guidance,
  schedulerShift,
  setSchedulerShift,
  showSchedulerShift,
  selectedModel,
  showSteps,
  setSteps,
  steps,
  songModel,
  guidanceEnabled,
  setGuidanceEnabled,
  setSongGuidance,
  songGuidance,
  setMaxNewTokensPerSegment,
  maxNewTokensPerSegment,
  setRepetitionPenalty,
  repetitionPenalty,
  showOutputLimiter,
  setOutputLimiter,
  outputLimiter,
  showNegative,
  negativePromptCfgOff,
  setNegativePrompt,
  negativePrompt,
  sampleRates
}) {
  return (
<div className="advanced-panel">
                <label>
                  Seed
                  <input
                    onChange={(event) => setSeed(event.target.value)}
                    placeholder="Random"
                    type="number"
                    value={seed}
                  />
                </label>
                {/* Diffusion sampling knobs. Steps (the solver step count) rides the top-level request
                    both the Sound FX (MOSS) and Music (ACE-Step) samplers read. Guidance (the CFG scale)
                    surfaces only when the model advertises guidance support — MOSS does; the
                    guidance-distilled ACE-Step turbo does NOT, so it stays hidden rather than sent (the
                    gen-core floor would reject it). Cleared ⇒ the model's own default; the floor
                    range-checks any value sent, so no hardcoded ceiling is baked into the input. */}
                {showGuidance ? (
                  <label>
                    Guidance (CFG)
                    <input

                      onChange={(event) => setGuidance(event.target.value)}
                      placeholder="Model default"
                      step="any"
                      type="number"
                      value={guidance}
                    />
                  </label>
                ) : null}
                {showSchedulerShift ? (
                  <label>Schedule shift<input type="number" min="0" step="any"
                    value={schedulerShift} onChange={(event) => setSchedulerShift(event.target.value)}
                    placeholder={selectedModel?.defaults?.schedulerShift == null ? "Model default" : String(selectedModel.defaults.schedulerShift)} /></label>
                ) : null}
                {showSteps ? (
                  <label>
                    Steps
                    <input
                      min="1"
                      onChange={(event) => setSteps(event.target.value)}
                      placeholder="Model default"
                      step="1"
                      type="number"
                      value={steps}
                    />
                  </label>
                ) : null}
                {/* YuE song knobs (sc-19385, R5). Defaults are upstream's published values: guidance on
                    (1.5 first section / 1.2 after), 3000 tokens per section, repetition penalty 1.1.
                    Cleared ⇒ omitted ⇒ the model default. The native segmented-song provider reads
                    scales 0–1 as guidance off; the switch also explicitly disables guidance. */}
                {songModel ? (
                  <>
                    <label className="checkline">
                      <input
                        checked={Boolean(guidanceEnabled)}
                        onChange={(event) => setGuidanceEnabled(event.target.checked)}
                        type="checkbox"
                      />
                      Guidance (CFG)
                    </label>
                    <label>
                      Guidance scale
                      <input
                        disabled={!guidanceEnabled}
                        min="0"
                        onChange={(event) => setSongGuidance(event.target.value)}
                        placeholder="1.5 → 1.2 (default)"
                        step="any"
                        type="number"
                        value={songGuidance}
                      />
                    </label>
                    <label>
                      Max tokens per section
                      <input
                        max="16382"
                        min="1"
                        onChange={(event) => setMaxNewTokensPerSegment(event.target.value)}
                        placeholder="3000 (default)"
                        step="100"
                        type="number"
                        value={maxNewTokensPerSegment}
                      />
                    </label>
                    <label>
                      Repetition penalty
                      <input
                        max="10"
                        min="0.05"
                        onChange={(event) => setRepetitionPenalty(event.target.value)}
                        placeholder="1.1 (default)"
                        step="any"
                        type="number"
                        value={repetitionPenalty}
                      />
                    </label>
                    {showOutputLimiter ? (
                      <label>
                        Output limiter
                        <select onChange={(event) => setOutputLimiter(event.target.value)} value={outputLimiter}>
                          <option value="">Model default (clamp)</option>
                          <option value="clamp">Clamp — hard-clip peaks</option>
                          <option value="rescale">Rescale — normalize the whole stem</option>
                        </select>
                      </label>
                    ) : null}
                  </>
                ) : null}
                {/* Negative prompt — the traits to steer away from. Capability-gated: surfaced only when
                    the model advertises negative-prompt support (audio.supportsNegativePrompt). The
                    guidance-distilled ACE-Step turbo advertises none, so it stays hidden; a music model
                    that supports it renders this and rides it through as GenerationRequest.negative_prompt. */}
                {showNegative ? (
                  <label>
                    Negative prompt
                    <textarea disabled={negativePromptCfgOff}
                      aria-label="Negative prompt"
                      onChange={(event) => setNegativePrompt(event.target.value)}
                      placeholder="Traits to avoid"
                      value={negativePrompt}
                    />
                  </label>
                ) : null}
                {/* Sample rate the model emits at (audio.sampleRates). Capability-driven, never
                    hardcoded — and read-only: the output rate is a fixed property of the model's
                    vocoder (Kokoro is 24 kHz mono), not a request knob the audio route accepts, so it
                    is surfaced for transparency rather than sent in the payload. */}
                {sampleRates.length ? (
                  <label>
                    Sample rate
                    <select disabled value={String(sampleRates[0])}>
                      {sampleRates.map((rate) => (
                        <option key={rate} value={String(rate)}>
                          {rate} Hz
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
  );
}
