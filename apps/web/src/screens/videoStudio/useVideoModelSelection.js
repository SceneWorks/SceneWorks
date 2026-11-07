import { ltxVideoModelId } from "./videoTierHelpers.js";
import { useCallback, useEffect, useMemo, useState } from "react";
import { downloadOffersFor, videoModelServesMode, videoModelUsable } from "../../modelEligibility.js";
import { macAvailableModels, macGatingActive } from "../../modelEligibility.js";
import { candleAvailableModels, candleGatingActive } from "../../modelEligibility.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoModelSelection({
  videoModels,
  saved,
  setRecipeModelNotice,
  setRecipeTierRequest,
  macCapabilities,
  models,
  jobs
}) {
  // LTX-2.3 remains the default video route. LTX-2.5 is deliberately opt-in even though its
  // catalog entry can arrive first (the manifest groups the shared LTX family together).
  const defaultVideoModelId =
    videoModels.find((item) => item.id === ltxVideoModelId)?.id ??
    videoModels[0]?.id ??
    ltxVideoModelId;

  const [model, setModel] = useState(saved.model ?? defaultVideoModelId);

  const [expectedImportedModelRef, setExpectedImportedModelRef] = useState(saved.expectedImportedModelRef ?? null);

  // sc-26691: the MiniMax-H3 reference-sizing knobs a replayed H3 recipe recorded (the studio has
  // no control for them). Replay-only state, like the imported identity above.
  const [replayReferenceResize, setReplayReferenceResize] = useState(null);

  // Every USER model picker must retire replay-only state. Automatic catalog/mode/recipe snaps use
  // the raw state setter below and preserve the replay while its target model becomes active.
  const setUserModel = useCallback((nextModel) => {
    setRecipeModelNotice("");
    setRecipeTierRequest(null);
    setExpectedImportedModelRef(null);
    setReplayReferenceResize(null);
    setModel(nextModel);
  }, [setRecipeModelNotice, setRecipeTierRequest]);

  // Platform UI gating: hide whole models with no lane on THIS platform and snap off one if
  // selected. Two composed partitions, one per platform, each a no-op on the other's platform:
  //   * `macAvailableModels` (sc-3486) — torch-only models (e.g. SVD) on a gated Mac.
  //   * `candleAvailableModels` (sc-19570) — models with no candle lane at all off-Mac (e.g.
  //     `wan_2_2_vace_fun_14b`, whose only advertised mode is candle-unclaimable). Before this the
  //     export existed and NOTHING imported it, so whole-model hiding was dead in the one screen
  //     that matters and the picker still listed a model no off-Mac worker can claim.
  const macVideoModels = useMemo(
    () => candleAvailableModels(macAvailableModels(videoModels, macCapabilities), macCapabilities),
    [videoModels, macCapabilities],
  );

  useEffect(() => {
    if (macVideoModels.length && !macVideoModels.some((item) => item.id === model)) {
      setModel(macVideoModels[0].id);
    }
  }, [macVideoModels, model]);

  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];

  // Imported catalog entries carry their explicit task partition; an id prefix/family alone
  // cannot prove FL2VA support (Ref2VA shares the same architecture).
  // An IMPORTED H3 partition that serves a last-only request (sc-26687). Stock `minimax_h3` is
  // deliberately NOT in this set: an ambient first frame is part of the shipped stock behaviour
  // for that model, and changing it here would alter an existing model rather than fix the
  // import platform this epic is about.
  const importedLastOnlyModel = (entry) => Boolean(
    entry?.family === "minimax-h3" && entry?.baseModel === "minimax_h3" &&
    entry?.importSourceShape === "transformer_directory" &&
    Boolean(entry?.importPlan?.checkpointId?.trim?.()) &&
    ["fl2va", "hybrid"].includes(entry?.minimaxH3Partition) &&
    entry?.capabilities?.includes("image_to_video")
  );

  const supportsLastOnly = (entry) => entry?.id === "minimax_h3" || importedLastOnlyModel(entry);

  const acceptsLastOnly = supportsLastOnly(selectedModel);

  // Whether an ambient first frame must be withheld from this model's request (sc-26687).
  // IMPORTED H3 partitions only; stock `minimax_h3` keeps its shipped behaviour.
  const importedLastOnly = importedLastOnlyModel(selectedModel);

  // Models gated on the selected tab, not tabs on the selected model (sc-5716). "Serves" is the
  // SHARED `videoModelServesMode` from modelEligibility.js — this screen used to keep a local copy
  // that read the Mac block only, and that is exactly how sc-19570's off-Mac gate got added to the
  // shared predicate (and so to the Simple studio, the screen gate and the download offers) while
  // the Advanced shell kept rendering MLX-only tabs off-Mac. One authority, three layers:
  // declaration + `macVideoModeBlock` + `candleVideoModeBlock`. The mode tabs, the model picker and
  // the snap-on-mode-switch effect all derive from it so the user is never trapped on a mode whose
  // model can't serve the others.
  const macGating = macGatingActive(macCapabilities);

  const candleGating = candleGatingActive(macCapabilities);

  // sc-19570 — NO `macVideoModels.length ? macVideoModels : videoModels` FALLBACK, and its removal
  // is the point rather than a tidy-up. That fallback restored the UNFILTERED catalog precisely when
  // the platform filter had emptied it — i.e. exactly when every installed video model is blocked on
  // this host. `modelReady` read off it, so it stayed `true`, the sc-5947 gate never engaged, and
  // the user got the full Studio with a picker of models that cannot serve any mode: every tab
  // disabled by `modeTabBlocked`, no download offer, and no statement of why. Degraded rather than
  // hung, but it hid the one screen that could fix it.
  //
  // Pre-existing, and it did not fire before this story because `macAvailableModels` alone empties
  // the list only for a Mac user whose every video model is torch-only. `candleAvailableModels`
  // makes it reachable for an ordinary Windows/Linux user with an MLX-only catalog, which is the
  // common case off-Mac.
  //
  // `ImageStudio.jsx:765` is the precedent and settles the semantics: `modelReady =
  // macImageModels.length > 0`, filtered list, no fallback. This is the video twin.
  //
  // Not a behaviour change when no gate is engaged: `macAvailableModels` and `candleAvailableModels`
  // both return the input list unfiltered when their gate is inactive (and before the capabilities
  // endpoint responds), so `macVideoModels === videoModels` and the two expressions are identical.
  // The difference appears only when a gate is active AND has filtered everything out — the case the
  // gate exists for.
  const modelsForMode = (value) =>
    macVideoModels.filter((item) => videoModelServesMode(item, value, macCapabilities));

  // Model-availability gate (sc-5947): when the user has no PLATFORM-available video model at all,
  // show recommended video-model downloads instead of the studio. `ready` matches the picker; offers
  // come from the full catalog via videoModelUsable, recommended-first.
  const modelReady = macVideoModels.length > 0;

  const modelOffers = useMemo(
    () => downloadOffersFor(models, videoModelUsable, macCapabilities),
    [models, macCapabilities],
  );

  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  return { model, setModel, expectedImportedModelRef, setExpectedImportedModelRef, replayReferenceResize, setReplayReferenceResize, setUserModel, macVideoModels, selectedModel, supportsLastOnly, acceptsLastOnly, importedLastOnly, macGating, candleGating, modelsForMode, modelReady, modelOffers, modelDownloadJobs };
}
