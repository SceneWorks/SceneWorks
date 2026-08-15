// MiniMax-H3 turbo variant selection for the Video Studio (sc-18727, epic 17137).
//
// The turbo control is NOT a second way to say "accelerate". It selects a catalog LoRA — the exact
// same thing the generic LoRA picker does — so both entry points converge on one payload field
// (`loras`) and one server-side resolver (`sceneworks_core::minimax_h3_turbo::resolve_turbo_recipe`).
// sc-18727 asked for exactly that: "either route both through one resolver or hide it from the
// generic picker". Routing both is the better half of that choice, because hiding an installed
// adapter from the picker would make it un-deselectable from the surface that shows it selected.
//
// Everything here is DERIVED from the LoRA catalog the API already serves. There is deliberately no
// mirrored table of variants, step counts or shifts on this side: `builtin.loras.jsonc` declares the
// `sampling` block, the API passes it through unmodified, and the Rust resolver reads the same
// declaration. A web-side copy would be a fourth place for the numbers to drift and the first place
// nothing would notice.

/// Whether `lora` is a step-distill accelerator carrying a declared sampling recipe — the pair of
/// facts that makes selecting it change the SCHEDULE rather than just stack a residual.
///
/// Both halves are required. `role: accelerator` alone is what the Krea 2 turbo adapter carries and
/// means only "this adapter's intent is a different sampler regime"; without a `sampling` block the
/// worker has nothing to apply and the file loads as a plain residual, so offering it in a control
/// labelled "turbo" would promise a speedup nothing delivers.
export function isTurboVariant(lora) {
  const role = typeof lora?.role === "string" ? lora.role.trim().toLowerCase().replace(/-/g, "_") : null;
  if (role !== "accelerator") return false;
  const sampling = lora?.sampling;
  return (
    Boolean(sampling) &&
    Number.isInteger(sampling.steps) &&
    sampling.steps > 0 &&
    Number.isFinite(sampling.schedulerShift) &&
    sampling.schedulerShift > 0
  );
}

/// The turbo variants offerable for `selectedModel`, given the studio's already-filtered
/// `compatibleLoras` (installed + family-compatible + not preset-managed).
///
/// Built on `compatibleLoras` rather than the raw catalog so the control can never offer a variant
/// the picker would refuse, or one whose weights are not on disk — an uninstalled adapter would
/// produce a selection the enqueue gate rejects ("LoRA is not installed"), which is the shape of
/// unreachable control this story exists to avoid.
///
/// Non-MiniMax-H3 models get an empty list even if some other family ships an accelerator, because
/// the worker's turbo resolver returns "no recipe" off this family: the control must not appear
/// where flipping it would change nothing.
export function turboVariantsForModel(selectedModel, compatibleLoras = []) {
  if (!modelIsMinimaxH3(selectedModel)) return [];
  return compatibleLoras.filter(isTurboVariant);
}

/// Whether `selectedModel` is a MiniMax-H3 catalog entry. Mirrors
/// `sceneworks_core::video_request::is_minimax_h3_model` (an `id` prefix test), which is what every
/// MiniMax-H3 gate in the Rust half already keys on — both catalog partitions (`minimax_h3` and
/// `minimax_h3_ref`) are in scope, since both load the same DiT architecture and both have a
/// published turbo adapter.
export function modelIsMinimaxH3(selectedModel) {
  return typeof selectedModel?.id === "string" && selectedModel.id.startsWith("minimax_h3");
}

/// The variant a fresh MiniMax-H3 studio defaults to, or `null` when none is installed.
///
/// **Default-on**, matching the Wan Lightning precedent (sc-10048) and for a much larger reason:
/// sc-18729 measured a 1344x768 base render at 2.42 h against 12.6 min with the 768p turbo adapter.
/// A default-off control would make the shipped default experience a two-and-a-half-hour render,
/// which is not a default anyone would choose on purpose.
///
/// The preference order is the checkpoint pairing, not the step count: the reference partition takes
/// the ref2v adapter (it distils the reference-conditioned path), and the base partition takes the
/// 768p 4-step file — the one variant sc-18729 validated end to end, and the one trained at the
/// model's own default canvas. Anything else installed is a fallback rather than a default, so a
/// user who has only the 8-step file still gets accelerated instead of silently getting nothing.
///
/// Returns `null` — not a variant — when nothing is installed. Selecting an uninstalled adapter
/// would 400 at enqueue, so "no turbo" is the honest default state on a fresh install, and the
/// control says so rather than pretending.
export function defaultTurboVariant(selectedModel, variants = []) {
  if (!variants.length) return null;
  const preferred =
    selectedModel?.id === "minimax_h3_ref"
      ? ["minimax_h3_ref2v_turbo_4step"]
      : ["minimax_h3_turbo_4step_768p", "minimax_h3_turbo_8step", "minimax_h3_turbo_4step_v01"];
  for (const id of preferred) {
    const match = variants.find((variant) => variant.id === id);
    if (match) return match;
  }
  return variants[0];
}

/// The turbo variant currently selected, given the studio's selected LoRA ids. `null` = the base
/// 50-step regime.
///
/// Reads the SELECTION, not a separate toggle state, so the control and the LoRA picker cannot
/// disagree: deselecting the adapter in the picker turns the control off, and vice versa, because
/// there is only one piece of state.
export function selectedTurboVariant(variants = [], selectedLoraIds = []) {
  return variants.find((variant) => selectedLoraIds.includes(variant.id)) ?? null;
}

/// The one-line schedule summary shown under the control: the recipe the selected variant actually
/// applies, in the same units the worker records on the asset.
///
/// Spells out the audio shift as well as the video one because MiniMax-H3 runs two schedulers, and a
/// control that reported only the video half would be describing half the render.
export function turboRecipeSummary(variant) {
  if (!variant) return null;
  const { steps, schedulerShift, audioSchedulerShift } = variant.sampling ?? {};
  const audio = Number.isFinite(audioSchedulerShift) ? `, audio shift ${audioSchedulerShift}` : "";
  return `${steps} steps, video shift ${schedulerShift}${audio}`;
}
