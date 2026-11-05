/**
 * The value a control should start on: the model's declared default when it is actually
 * offered, else the first entry of `preferences` that is offered, else the first option.
 *
 * @param {*} declared - the model's `defaults.<field>`.
 * @param {Array} options - the values the picker offers (already memory-gated).
 * @param {Array} [preferences] - ordered fallbacks to try before giving up on `options[0]`.
 */
export function preferredOption(declared, options, preferences = [], { empty = undefined } = {}) {
  const list = Array.isArray(options) ? options : [];
  if (list.length === 0) {
    return empty;
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

// Image surfaces prefer the declared geometry, then 1024² when offered.
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

export function formatResolutionLabel(value) {
  const [width, height] = String(value).split("x");
  return height ? `${width} × ${height}` : value;
}
