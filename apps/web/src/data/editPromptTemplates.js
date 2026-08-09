// Built-in one-tap edit instructions. These are model-agnostic: every image-edit engine
// we ship is instruction-tuned on plain imperative English, so the same five recipes read
// the same way to Qwen-Image-Edit, Z-Image-Edit, Bernini, and anything added later. They
// are deliberately terse — the edit models respond better to a single clear instruction
// than to a loaded scene paragraph, which is why the generation-side scene suggestions
// (PROMPT_SUGGESTION_POOL) are the wrong pills for the Edit surface.
//
// `prompt` REPLACES the instruction box rather than appending: each one is a complete
// instruction, and stacking two of them ("deblur this image. correct the colors…") is a
// worse edit than running them as two passes.
export const EDIT_PROMPT_TEMPLATES = [
  {
    id: "restore",
    label: "Restore photo",
    prompt: "colorize and restore this photo, like it was taken in modern day",
    hint: "Colorize and repair a vintage or damaged photo.",
  },
  {
    id: "detail_zoom",
    // The upstream recipe names the subject ("zoom in on the bird"). We can't know the
    // subject, so the pill ships a runnable generic and the user swaps in the noun.
    label: "Detail zoom",
    prompt: "zoom in on the main subject. ultra sharp details of the subject. professional photography",
    hint: "Crop in tight and resolve fine detail — name the subject for a sharper result.",
  },
  {
    id: "color_fix",
    label: "Fix colors",
    prompt: "correct the colors of this photo",
    hint: "One-shot white balance and color-cast correction.",
  },
  {
    id: "character_sheet",
    label: "Character sheet",
    prompt: "make a model sheet of this character on a white background. full body shot. front, back, and side",
    hint: "Turn a character into a front / side / back model sheet.",
  },
  {
    id: "deblur",
    label: "Deblur",
    prompt: "deblur this image",
    hint: "Recover a soft or motion-blurred shot.",
  },
];
