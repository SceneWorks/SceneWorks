// Video-model `limits` readers the generation FORM needs, not only the enqueue gate (sc-17161).
//
// Three declared facts had no web reader at all before this story, and each one let the studio
// build a request the server then refused:
//
//   * the four sc-17160 reference caps (`maxReferenceAssets` / `maxSourceClipAssets` /
//     `maxReferenceAudioAssets` / `maxCombinedReferenceAssets`). MiniMax-H3 Ref2VA is the first
//     model whose per-list caps do NOT multiply out to its real ceiling — 9 + 3 + 3 is 15 files
//     against a 12-file limit — so the combined cap is the only one that can refuse a selection in
//     which every individual picker is inside its own cap.
//   * `limits.hardMinSteps` (sc-19426). The Steps input hardcoded `min="1"`, so a 1-step
//     MiniMax-H3 request submitted cleanly and came back as a 400 from `steps_limit_error`.
//
// The DEFAULTS here are the same blanket values `reference_caps` applies in
// `crates/sceneworks-core/src/video_request.rs` — images/clips 8, audio 0 — so a model that
// declares nothing behaves exactly as it did before this module existed. Audio defaulting to
// ZERO is the load-bearing one: audio references are surface no shipped engine consumed before
// MiniMax-H3, so a model has to declare that it takes them before the picker appears.

// `limits.maxReferenceAssets` when the model declares none — `DEFAULT_MAX_REFERENCE_ASSETS`.
export const DEFAULT_MAX_REFERENCE_ASSETS = 8;
// `limits.maxSourceClipAssets` when the model declares none.
export const DEFAULT_MAX_SOURCE_CLIP_ASSETS = 8;
// `limits.maxReferenceAudioAssets` when the model declares none. Zero, deliberately — see above.
export const DEFAULT_MAX_REFERENCE_AUDIO_ASSETS = 0;

// A cap is DECLARED only when it is a non-negative integer. Anything else (a string, a float, a
// negative) is not a constraint anyone could satisfy, so it falls back to the blanket rather than
// bricking the model — the same never-block-without-evidence posture `reference_caps` takes.
function declaredCap(limits, key) {
  const value = limits?.[key];
  return Number.isInteger(value) && value >= 0 ? value : null;
}

/**
 * The per-model reference-media caps, plus whether the model DECLARED a source-clip cap.
 *
 * `clipsDeclared` is not the same question as `clips > 0`: every model resolves to a non-zero
 * clip cap through the blanket default, so `clips > 0` would offer a reference-clip picker on
 * Bernini's reference→video — a mode whose engine encodes reference IMAGES and would silently
 * ignore the clips. Only a model that says it takes reference clips gets the picker.
 */
export function referenceCaps(model) {
  const limits = model?.limits;
  const clips = declaredCap(limits, "maxSourceClipAssets");
  return {
    images: declaredCap(limits, "maxReferenceAssets") ?? DEFAULT_MAX_REFERENCE_ASSETS,
    clips: clips ?? DEFAULT_MAX_SOURCE_CLIP_ASSETS,
    audio: declaredCap(limits, "maxReferenceAudioAssets") ?? DEFAULT_MAX_REFERENCE_AUDIO_ASSETS,
    combined: declaredCap(limits, "maxCombinedReferenceAssets"),
    clipsDeclared: clips !== null,
  };
}

/**
 * The refusal for a reference selection over what the model declares, or `null` to admit —
 * the client half of `reference_limit_error`.
 *
 * Per-list caps are checked before the combined one so the message names the picker actually
 * over budget. The lever is the CONTROL, not the wire field the Rust message names: on this side
 * of the seam the user's move is "remove some from that picker", and the field name would be
 * jargon for a sentence rendered next to the picker itself.
 */
export function referenceLimitError({ modelName, caps, images = 0, clips = 0, audio = 0 } = {}) {
  if (!caps) return null;
  const model = modelName || "This model";
  for (const [count, cap, noun] of [
    [images, caps.images, "reference images"],
    [clips, caps.clips, "reference video clips"],
    [audio, caps.audio, "reference audio clips"],
  ]) {
    if (count > cap) {
      return cap === 0
        ? `${model} takes no ${noun}, but ${count} are selected. Remove them, or choose a model that conditions on ${noun}.`
        : `${model} takes up to ${cap} ${noun}, but ${count} are selected. Remove ${count - cap}.`;
    }
  }
  if (caps.combined === null || caps.combined === undefined) return null;
  const total = images + clips + audio;
  if (total <= caps.combined) return null;
  return (
    `${model} takes up to ${caps.combined} reference files in total, but ${total} are selected ` +
    `(${images} reference images + ${clips} video clips + ${audio} audio clips). ` +
    `Remove ${total - caps.combined}.`
  );
}

/**
 * The Steps floor a model declares (`limits.hardMinSteps`, sc-19426), or 1 when it declares none.
 *
 * MiniMax-H3's scheduler grid is `linspace(1, 0, steps)` and the terminal 0 is a grid point, so a
 * 1-step request is a schedule with nothing between its ends — unrenderable, and REFUSED rather
 * than quietly raised to 2. The form has to know that before it lets the value be typed.
 */
export function minStepsForModel(model) {
  const declared = model?.limits?.hardMinSteps;
  return Number.isInteger(declared) && declared > 0 ? declared : 1;
}

/**
 * A duration menu entry as seconds, trimmed to 2 dp with trailing zeros dropped.
 *
 * MiniMax-H3 is the first video model whose durations are not integers — its fourteen `17n + 5`
 * lattice rungs are frame counts divided by 24 — and the raw values render as "5.1667s",
 * "6.5833s", "10.8333s". The stored value is untouched; only the label is rounded, because the
 * enqueue gate matches the exact `limits.durations` number.
 */
export function formatDurationSeconds(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "";
  return String(Number(value.toFixed(2)));
}

/**
 * A duration menu entry with the frame count it resolves to, e.g. `5.17s · 124 frames`.
 *
 * The frame count is what the menu is really enumerating: MiniMax-H3's legal clip lengths are a
 * FRAME lattice (`17n + 5`), and "5.17s" on its own reads like a rounded slider position rather
 * than one of exactly fourteen renderable values. `Math.round` is exact here by construction —
 * each rung is `frames / fps` — and any model with an integral duration is unchanged in meaning.
 */
export function formatDurationOption(seconds, fps) {
  const label = `${formatDurationSeconds(seconds)}s`;
  const rate = Number(fps);
  if (!Number.isFinite(rate) || rate <= 0) return label;
  const frames = Math.round(Number(seconds) * rate);
  return Number.isFinite(frames) && frames > 0 ? `${label} · ${frames} frames` : label;
}
