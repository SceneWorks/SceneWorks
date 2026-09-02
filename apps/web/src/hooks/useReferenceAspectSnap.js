import { useEffect, useRef } from "react";

import { assetIsHdrSource, assetNativeSize, assetUrl } from "../components/assetMedia.jsx";
import { probeImageDimensions } from "../imageDimensions.js";

// The `snapKey` a resolution-bucket caller should pass: which model's option UNIVERSE we are in.
// Deliberately derived from the model's DECLARED buckets and nothing else — NOT from the
// memory-gated list the picker actually shows. That gated list also varies with the picked quant
// tier (sc-16020), and a tier switch re-snapping over the user's Aspect is the same clobber
// sc-18255 fixes. If the gate trims the selected bucket, the caller's own validity effect re-picks
// it. Returns a VALUE, so a model-catalog refetch that hands back an equal-but-new object is inert.
export function resolutionUniverseKey(model, fallbackResolutions = []) {
  const declared = model?.limits?.resolutions?.length ? model.limits.resolutions : fallbackResolutions;
  return `${model?.id ?? ""}|${declared.join(",")}`;
}

// Shared reference-image aspect auto-preset (sc-8109 / sc-8220 / sc-10195): probe the picked
// reference's natural size and hand it to `onReferenceImageLoaded`, which snaps the Aspect picker to
// the declared bucket closest to that aspect. Keeping the composition's aspect matters because a
// reference grounded for 9:16 but rendered at 1:1 comes out wrong-shaped.
//
// The probe stays keyed on the asset LIST as well as the id, because a freshly imported / dragged
// reference lands in the list a render after its id is set (sc-8220) and the snap must still happen
// once it resolves. But the list gets a NEW ARRAY IDENTITY on every asset refresh — and the worker
// emits a generationSetId on the first denoise step, so clicking Generate refreshes the list within
// a second. Re-running the snap there overwrote whatever Aspect the user had chosen since (sc-18255:
// a 768x1024 Krea job submitted at 1152x2048, the 1080x1920 reference's closest bucket).
//
// So the snap fires at most once per (reference, snapKey) pair. `snapKey` is the caller's option set
// as a VALUE, not an identity — the option list is memoized on the selected model, whose identity
// churns on every model-catalog refetch, while a genuine change (switching models, or the catalog
// resolving for the first time) changes the value and legitimately re-snaps.
// `enabled` is deliberately SEPARATE from an empty `referenceId`: only an empty id means "the user
// cleared the reference", which re-arms the snap. A caller that is merely inoperative right now (the
// img2img panel is hidden because the mode or model changed) must suppress the snap WITHOUT re-arming
// it — collapsing the two would re-snap on the way back and clobber the Aspect all over again.
export function useReferenceAspectSnap({
  referenceId,
  assets = [],
  onReferenceImageLoaded,
  snapKey = "",
  enabled = true,
}) {
  const snapped = useRef("");
  useEffect(() => {
    // Wrapping the callback below defeats probeImageDimensions' own callable check, and a throw
    // from an image onload lands outside React's tree where no boundary catches it.
    if (typeof onReferenceImageLoaded !== "function") return undefined;
    if (!referenceId) {
      // Cleared: re-picking is a fresh user action and snaps again.
      snapped.current = "";
      return undefined;
    }
    if (!enabled) return undefined;
    // "|" separates safely: asset ids and resolution buckets never contain one.
    const pairKey = `${referenceId}|${snapKey}`;
    if (snapped.current === pairKey) return undefined;
    const asset = assets.find((item) => item.id === referenceId);
    // Mark only on a SUCCESSFUL probe: an id whose asset has not resolved yet never reaches this
    // callback, so it stays eligible and snaps on the render where the asset appears.
    // An HDR source cannot be probed: no browser decodes OpenEXR, so an <img> pointed at it never
    // fires `onload` and the snap silently never happens. Use the size the server recorded from
    // the file header instead. Scoped to HDR deliberately — every other asset keeps the existing
    // decode-and-measure path unchanged, including its timing.
    if (assetIsHdrSource(asset)) {
      const recorded = assetNativeSize(asset);
      if (recorded) {
        snapped.current = pairKey;
        onReferenceImageLoaded(recorded.width, recorded.height);
      }
      return undefined;
    }
    return probeImageDimensions(asset && assetUrl(asset), (width, height) => {
      snapped.current = pairKey;
      onReferenceImageLoaded(width, height);
    });
  }, [referenceId, assets, onReferenceImageLoaded, snapKey, enabled]);
}
