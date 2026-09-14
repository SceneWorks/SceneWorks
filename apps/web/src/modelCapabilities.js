// Capability descriptors shown as chips on a model card. With models grouped by `type`, the chips
// are what tell the user what a card actually does (plain text-to-image vs editing vs character
// reference, etc.). Unknown keys fall back to a humanized form so a new capability still reads
// sensibly without a code change.
//
// Extracted from ModelManagerScreen (epic 20398, sc-20650) so the checkpoint-import panel can show
// the SAME chips for a checkpoint the user is about to select, rather than a second, drifting copy.
export const CAPABILITY_LABELS = {
  text_to_image: "Text to Image",
  image_to_image: "Image to Image",
  edit_image: "Image Edit",
  character_image: "Character",
  vqa: "Visual Q&A",
  interleave: "Interleaved",
  image_to_video: "Image to Video",
  text_to_video: "Text to Video",
  // sc-8445: Krea Realtime advertises text/image/video-to-video, and without this row its third
  // chip fell to the humanized fallback ("video to video") — visibly out of style beside the two
  // title-cased siblings on the same card.
  video_to_video: "Video to Video",
  first_last_frame: "First / Last Frame",
  extend_clip: "Extend Clip",
  video_bridge: "Video Bridge",
  replace_person: "Replace Person",
};

// Modes the app has retired. A stale imported/catalog record may still carry the token; turning it
// back into a visible affordance would advertise something no studio can run.
export const RETIRED_MODEL_CAPABILITIES = new Set(["style_variations"]);

export function capabilityLabel(capability) {
  return CAPABILITY_LABELS[capability] ?? String(capability).replaceAll("_", " ");
}

// The live capability chips for a catalog row, retired tokens dropped.
export function modelCapabilityChips(model) {
  const capabilities = Array.isArray(model?.capabilities) ? model.capabilities : [];
  return capabilities.filter((capability) => !RETIRED_MODEL_CAPABILITIES.has(capability)).map(capabilityLabel);
}
