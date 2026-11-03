import { useCallback } from "react";
import { parseMagicPromptCaption, parseVisionCaption } from "../../ideogramCaption.js";
import { PROMPT_REFINE_MODEL_ID, VISION_CAPTION_MODEL_ID, VISION_CAPTION_MODEL_REPO } from "../../constants.js";

// Extracted from ImageStudio along its existing concern boundary.
export function useStructuredPrompt({
  model,
  magicPrompt,
  gcd,
  width,
  height,
  setMagicPromptBackend,
  imageCaption,
  activeProject,
  selectedModel,
  imageDescribe
}) {
  const onMagicExpand = useCallback(
    async (idea) => {
      if (typeof magicPrompt !== "function") {
        throw new Error("Magic-prompt is unavailable.");
      }
      const divisor = gcd(width, height) || 1;
      const aspectRatio = Number.isFinite(width) && Number.isFinite(height) ? `${width / divisor}:${height / divisor}` : "1:1";
      const raw = await magicPrompt({ prompt: idea, modelId: model, aspectRatio });
      const { caption: expanded, error } = parseMagicPromptCaption(raw);
      if (error || !expanded) {
        throw new Error(error || "Magic-prompt returned an unusable caption.");
      }
      setMagicPromptBackend(PROMPT_REFINE_MODEL_ID);
      return expanded;
    },
    [magicPrompt, gcd, width, height, model, setMagicPromptBackend],
  );

  // Reference-image → JSON caption (epic 8102, sc-8108): run the worker's `image_caption` vision job on
  // the picked reference asset and parse the reply into an editable caption. Uses `parseVisionCaption`
  // (strips the non-schema `aspect_ratio`, KEEPS the grounded bboxes — they are derived from the actual
  // image, unlike magic-prompt's guessed boxes). Throws on a malformed/non-caption reply so the builder
  // surfaces the error and lets the user retry, mirroring the magic-prompt error UX. C1: the image is
  // captioning-only — it is consumed here to produce JSON and never passed to generation.
  const onImageCaption = useCallback(
    // Single id (string) for one reference, or an ARRAY of ids for a mood board (sc-8595): >1 synthesizes
    // ONE Ideogram JSON caption from the shared style, exactly one keeps the scalar single-image path.
    async (source) => {
      if (typeof imageCaption !== "function") {
        throw new Error("Image captioning is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const ids = Array.isArray(source) ? source.filter(Boolean) : [source].filter(Boolean);
      const multi = ids.length > 1;
      const raw = await imageCaption({
        sourceAssetId: multi ? undefined : ids[0],
        sourceAssetIds: multi ? ids : undefined,
        projectId: activeProject.id,
        model: VISION_CAPTION_MODEL_REPO,
      });
      const { caption: parsed, error } = parseVisionCaption(raw);
      if (error || !parsed) {
        throw new Error(error || "The image did not produce a usable caption.");
      }
      setMagicPromptBackend(VISION_CAPTION_MODEL_ID);
      return parsed;
    },
    [imageCaption, activeProject?.id, setMagicPromptBackend],
  );

  // Reference-image → plain-text description (epic 8203, sc-8208): the NON-structured sibling of
  // onImageCaption. Runs the worker's `image_describe` job on the picked reference and resolves to the
  // raw description text — prose by default, or booru tags when the model declares `captionStyle:"tags"`
  // (sc-8205). The shared picker drops the returned text into the prompt textarea. Gated to
  // text-to-image only, like the caption flow. C1: the image is consumed to produce the prompt and is
  // never passed to generation.
  const describeCaptionStyle = selectedModel?.captionStyle;

  const onImageDescribe = useCallback(
    // The shared picker passes a single asset id (string) for one reference, or an ARRAY of ids for a
    // mood board (sc-8595). Normalize: >1 rides `sourceAssetIds` (worker synthesizes one prompt from the
    // shared style), exactly one collapses to the scalar `sourceAssetId` (the unchanged single path).
    async (source) => {
      if (typeof imageDescribe !== "function") {
        throw new Error("Image description is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const ids = Array.isArray(source) ? source.filter(Boolean) : [source].filter(Boolean);
      const multi = ids.length > 1;
      const text = await imageDescribe({
        sourceAssetId: multi ? undefined : ids[0],
        sourceAssetIds: multi ? ids : undefined,
        projectId: activeProject.id,
        model: VISION_CAPTION_MODEL_REPO,
        captionStyle: describeCaptionStyle,
      });
      const trimmed = (text || "").trim();
      if (!trimmed) {
        throw new Error("The image did not produce a usable description.");
      }
      setMagicPromptBackend(VISION_CAPTION_MODEL_ID);
      return trimmed;
    },
    [imageDescribe, activeProject?.id, describeCaptionStyle, setMagicPromptBackend],
  );
  return { onMagicExpand, onImageCaption, onImageDescribe };
}
