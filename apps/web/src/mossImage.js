// Product wire spelling follows gen-core::moss_image::MossImageOptions.
export const isMossImage = (id) => id === "moss_image_base" || id === "moss_image_sft";
export const mossDefaults = () => ({ acceleration: "off", max_sequence_length: 4096 });
export const mossAccelerated = (options) => Boolean(options?.acceleration && options.acceleration !== "off");
export function mossWireOptions(options) {
  const value = { ...mossDefaults(), ...options };
  if (mossAccelerated(value)) {
    delete value.training_shift;
    delete value.uncond_skip_layers;
    delete value.slg;
  }
  return value;
}
export function mossComponentPurpose(options) {
  return mossAccelerated(options) ? `moss_dmd_${options.acceleration}` : null;
}
export function mossComponentState(model, options) {
  const purpose = mossComponentPurpose(options);
  if (!purpose) return { ready: true, purpose: null, state: "installed" };
  const state = model?.conditionalPurposes?.[purpose]?.installState ?? "missing";
  return { ready: state === "installed", purpose, state };
}

