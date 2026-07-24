// Resolving a model's DECLARED default for a control the Simple UI shows.
//
// The trap this exists to close: `limits.<thing>[0]` is the first ALLOWED value, not the
// model's chosen default, and for most shipped models the two differ:
//
//   z_image        limits.resolutions[0] 768x768   vs  defaults.resolution 1024x1024
//   wan_2_2_t2v    limits.resolutions[0] 832x480   vs  defaults.resolution 1280x720
//   ltx_2_3        limits.durations[0]   4s        vs  defaults.duration   6s
//   wan_2_2        limits.fps[0]         16        vs  defaults.fps        24
//
// Picking `[0]` therefore silently downgrades a Simple run relative to the same model in the
// full workspace — a lower resolution, a shorter clip, a frame rate the model never chose
// (Wan 2.2 at 16 fps instead of 24 plays 33% slow). The advanced studios all read
// `defaults.*` (ImageStudio's `preferredResolution`, VideoStudio's duration/fps/resolution
// seeds); this module is the one place Simple does the same.

/**
 * The value a control should start on: the model's declared default when it is actually
 * offered, else the first entry of `preferences` that is offered, else the first option.
 *
 * @param {*} declared - the model's `defaults.<field>`.
 * @param {Array} options - the values the picker offers (already memory-gated).
 * @param {Array} [preferences] - ordered fallbacks to try before giving up on `options[0]`.
 */
export function preferredOption(declared, options, preferences = []) {
  const list = Array.isArray(options) ? options : [];
  if (list.length === 0) {
    return undefined;
  }
  if (declared !== undefined && declared !== null && list.includes(declared)) {
    return declared;
  }
  for (const preference of preferences) {
    if (list.includes(preference)) {
      return preference;
    }
  }
  return list[0];
}

// The Image Studio's resolution default. Mirrors ImageStudio's `preferredResolution`
// exactly, including its 1024² second choice — so a Simple run and an untouched advanced
// run of the same model start on the same geometry.
export function preferredResolution(model, options) {
  return preferredOption(model?.defaults?.resolution, options, ["1024x1024"]);
}

// The Video Studio's resolution default. No 1024² second choice — video geometry is
// model-specific and the advanced studio seeds straight from `defaults.resolution`.
export function preferredVideoResolution(model, options) {
  return preferredOption(model?.defaults?.resolution, options);
}

// The Video Studio's clip-length default.
export function preferredDuration(model, options) {
  return preferredOption(model?.defaults?.duration, options);
}
