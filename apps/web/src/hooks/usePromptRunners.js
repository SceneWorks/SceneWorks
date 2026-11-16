import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { JOB_PATHS } from "../api/jobs.js";
import { useCallback } from "react";
import { pollJobToCompletion } from "../pollJob.js";

// F-189: extracted from App along its existing concern boundary.
export function usePromptRunners({
  token
}) {
  // Refine a prompt via the prompt_refine worker job: POST creates the job, then
  // poll until it reaches a terminal state and return the rewritten prompt. Project-
  // independent (no activeProject gate); throws on failure so the studio can surface
  // the message inline without clobbering the original prompt.
  const refinePrompt = useCallback(
    ({ prompt, modelId, workflow, guide, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        body: { prompt, modelId, workflow, guide },
        // 180 s, matching magicPrompt: a rewrite that runs to the worker's full 1536-token budget
        // (sc-24029) measures ~60-75 s on the dev Mac and can exceed 120 s on a slower host, so the
        // old 120 s deadline could time out a decode that was still going to succeed.
        deadlineMs: 180000,
        resolveResult: (job) => {
          const refined = job.result?.refinedPrompt;
          if (!refined) {
            throw new Error("Refinement returned an empty prompt.");
          }
          return refined;
        },
        signal,
        token,
        startError: "Could not start prompt refinement.",
        failureError: "Prompt refinement failed.",
        timeoutError: "Prompt refinement timed out. Is the refinement runtime running?",
      }),
    [token],
  );

  // Qwen-Image 2.1's OFFICIAL prompt rewriting (sc-24113, epic 24107): the SAME `prompts/refine`
  // endpoint and poll-to-completion contract as the refiner above, with `task:
  // "qwen_image_rewrite"`. No new route and no second LLM runtime — the two PE checkpoints are
  // Qwen3.5/3.6 (`qwen3_5`), the architecture the native TextLlm lane already loads for the
  // optional film planner.
  //
  // Two differences from `refinePrompt`, both of which the caller needs:
  //
  //  * It resolves the WHOLE result object, not just a string. The rewriter returns an aspect-ratio
  //    suggestion alongside the prose, and the user accepts or ignores the two independently.
  //  * `sourceAssetIds` carries the ORDERED references. Their presence is what selects the editing
  //    rewriter over the text-to-image one — server-side, from the request — and their order is
  //    what the rewrite's `<imageN>` numbering refers to.
  //
  // The deadline matches the caption tasks rather than the 120 s rewrite: a 9B checkpoint emitting
  // a reasoning block and a long paragraph is closer to a caption than to a one-line rewrite, and a
  // cold load of ~19 GB of weights happens inside this window.
  const qwenRewritePrompt = useCallback(
    ({ prompt, modelId, projectId, sourceAssetIds, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        body: {
          prompt,
          modelId,
          projectId,
          task: "qwen_image_rewrite",
          workflow: "image",
          sourceAssetIds: Array.isArray(sourceAssetIds) ? sourceAssetIds : [],
        },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const refinedPrompt = job.result?.refinedPrompt;
          if (!refinedPrompt) {
            // An empty rewrite must never reach the review panel: Apply would paste it over the
            // user's own prompt.
            throw new Error("The rewriter returned an empty prompt.");
          }
          return { refinedPrompt, rewriteSuggestion: job.result?.rewriteSuggestion ?? null };
        },
        signal,
        token,
        startError: "Could not start prompt rewriting.",
        failureError: "Prompt rewriting failed.",
        timeoutError: "Prompt rewriting timed out. Is the worker running?",
      }),
    [token],
  );

  // MiniMax-H3-Image prompt expansion (SW-7102): the same endpoint and contract as the Qwen
  // rewrite above, with `task: "minimax_h3_image_expand"`. The API forwards the expander's catalog
  // entry and refuses the run until its licence is accepted; references present select the edit
  // workflow, and their order is the `<Picture N>` numbering. `aspectRatio` is the edit's target
  // selection — `adaptive` lets the expander propose one.
  //
  // The deadline is longer than the rewrite's: a 27B checkpoint decodes up to 4096 tokens of
  // reasoning and answer, after a cold load of ~18-55 GB of weights inside this window.
  const h3ImageExpandPrompt = useCallback(
    ({ prompt, modelId, projectId, sourceAssetIds, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        body: {
          prompt,
          modelId,
          projectId,
          task: "minimax_h3_image_expand",
          workflow: "image",
          aspectRatio: "adaptive",
          sourceAssetIds: Array.isArray(sourceAssetIds) ? sourceAssetIds : [],
        },
        deadlineMs: 600000,
        resolveResult: (job) => {
          const refinedPrompt = job.result?.refinedPrompt;
          if (!refinedPrompt) {
            // An empty expansion must never reach the review panel: Apply would paste it over the
            // user's own prompt.
            throw new Error("The prompt expander returned an empty prompt.");
          }
          return { refinedPrompt, rewriteSuggestion: job.result?.rewriteSuggestion ?? null };
        },
        signal,
        token,
        startError: "Could not start prompt expansion.",
        failureError: "Prompt expansion failed.",
        timeoutError: "Prompt expansion timed out. Is the worker running?",
      }),
    [token],
  );

  // Magic-prompt expansion (epic 4725, sc-5997): same `prompts/refine` endpoint + native
  // utility model, but `task: "magic_prompt"` swaps in Ideogram's caption system prompt and
  // returns a JSON caption string (the caller parses + validates it). Reuses the refine job's
  // poll-to-completion contract; captions can take longer, so the deadline is generous.
  const magicPrompt = useCallback(
    ({ prompt, modelId, aspectRatio, guide, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        body: { prompt, modelId, task: "magic_prompt", aspectRatio, guide },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Magic-prompt returned an empty caption.");
          }
          return caption;
        },
        signal,
        token,
        startError: "Could not start magic-prompt.",
        failureError: "Magic-prompt failed.",
        timeoutError: "Magic-prompt timed out. Is the refinement runtime running?",
      }),
    [token],
  );

  // Reference-image → JSON caption (epic 8102, sc-8108): same `prompts/refine` endpoint + poll-to-
  // completion contract as magic-prompt, but `task: "image_caption"` drives the worker's `core_llm`
  // VISION path (sc-8105). The reference image is supplied as a project `sourceAssetId` (+ `projectId`),
  // which the API resolves to a confined on-disk `imagePath`; the vision model is named by its HF repo
  // string in `model` (the worker resolves it by repo, like the refiner). The caller parses + validates
  // the returned JSON with `parseVisionCaption` (aspect_ratio stripped, bboxes KEPT). C1: the image is
  // consumed only to produce the caption — it is NEVER passed to generation as img2img conditioning.
  const imageCaption = useCallback(
    ({ sourceAssetId, sourceAssetIds, projectId, model, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        // A mood board (epic 8588, sc-8595) sends `sourceAssetIds` (plural); the API synthesizes ONE
        // caption from the shared aesthetic. A single reference keeps the scalar `sourceAssetId`.
        body: { task: "image_caption", sourceAssetId, sourceAssetIds, projectId, model },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Image captioning returned an empty caption.");
          }
          return caption;
        },
        signal,
        token,
        startError: "Could not start image captioning.",
        failureError: "Image captioning failed.",
        timeoutError: "Image captioning timed out. Is the captioning runtime running?",
      }),
    [token],
  );

  // Reference-image → plain-text description (epic 8203, sc-8208): the sibling of `imageCaption` for
  // NON-structured text-to-image models. Same `prompts/refine` endpoint + poll contract, but
  // `task: "image_describe"` drives the worker's prose/tags vision path (sc-8204/8205). `captionStyle`
  // (from the catalog, default prose) selects natural-language prose vs booru tags. Resolves to the raw
  // text the caller drops into the prompt box. C1: the image is consumed only to produce the prompt — it
  // is NEVER passed to generation as img2img conditioning.
  const imageDescribe = useCallback(
    ({ sourceAssetId, sourceAssetIds, projectId, model, captionStyle, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.refine,
        // A mood board (epic 8588, sc-8595) sends `sourceAssetIds` (plural); the API synthesizes ONE
        // prompt from the aesthetic they share. A single reference keeps the scalar `sourceAssetId`.
        body: { task: "image_describe", sourceAssetId, sourceAssetIds, projectId, model, captionStyle },
        deadlineMs: 180000,
        resolveResult: (job) => {
          const description = job.result?.refinedPrompt;
          if (!description) {
            throw new Error("Image description returned empty text.");
          }
          return description;
        },
        signal,
        token,
        startError: "Could not start image description.",
        failureError: "Image description failed.",
        timeoutError: "Image description timed out. Is the captioning runtime running?",
      }),
    [token],
  );

  // On-demand "compare image to another" likeness (epic 4406, sc-4415): score a CANDIDATE asset
  // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer in the worker.
  // Same poll-to-completion contract as the refine/describe runners, but `/face-likeness/compare`
  // enqueues the GPU-routed `face_likeness_compare` job and returns the full result object
  // (`{ score, detected, method, sourceRef, reason? }`) so the caller can render the band + N/A framing
  // via `classifyLikeness` / `LikenessBadge`. Non-fatal end to end: a no-face / non-frontal candidate
  // is an honest detected:false result (NOT an error); only a hard failure throws.
  const compareFaceLikeness = useCallback(
    ({ sourceAssetId, candidateAssetId, projectId, signal }) =>
      pollJobToCompletion({
        createPath: JOB_PATHS.compareFaces,
        body: { sourceAssetId, candidateAssetId, projectId },
        deadlineMs: 180000,
        resolveResult: (job) => {
          // A completed compare always carries a result block (a detected:false N/A is a valid,
          // non-error outcome). Surface the whole block so the UI can band/N-A it.
          if (!job.result) {
            throw new Error("Likeness compare returned no result.");
          }
          return job.result;
        },
        signal,
        token,
        startError: "Could not start the likeness compare.",
        failureError: "Likeness compare failed.",
        timeoutError: "Likeness compare timed out. Is the worker running?",
      }),
    [token],
  );
  const context = useAppDomainValue({
    refinePrompt,
    qwenRewritePrompt,
    h3ImageExpandPrompt,
    magicPrompt,
    imageCaption,
    imageDescribe,
    compareFaceLikeness
  });
  return { ...context, context };
}
