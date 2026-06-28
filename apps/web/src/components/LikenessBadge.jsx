// Frontal-identity likeness badge (epic 4406, sc-4413). Surfaces the per-asset ArcFace
// antelopev2 likeness score the backend persists at
// `recipe.rawAdapterSettings.faceLikeness = { score, detected, method, sourceAssetId, reason? }`
// (written by sc-4408) as a small colour-banded badge on generated result thumbnails.
//
// The metric is *frontal-identity confidence*, NOT overall image quality. The band cut-points,
// labels, descriptions, and method string live in ../faceLikeness.js (sc-4414) — this component
// is a pure presenter and MUST NOT duplicate or re-derive any threshold.
//
// Three render states:
//   - scored (detected, finite cosine)  -> "92%" badge in a strong/moderate/weak colour band
//   - N/A   (detected:false / no score) -> neutral "—" chip reading "No frontal face to score",
//                                          never a red low number (metric honesty)
//   - absent (no faceLikeness block)    -> render nothing (legacy/unscored assets) — no layout box
import React from "react";
import { LIKENESS_BAND, LIKENESS_METHOD, classifyLikeness, likenessBand } from "../faceLikeness.js";

// The human-facing name for the recognition method (LIKENESS_METHOD is the machine string the
// backend stamps; this is what the tooltip shows). Kept adjacent so the literal lives in one place.
export const LIKENESS_METHOD_LABEL = "ArcFace antelopev2";

// Pull the persisted faceLikeness block off an asset's recipe. Returns null when the block is
// absent (legacy / unscored asset) so callers can render nothing without poking at the recipe.
export function assetFaceLikeness(asset) {
  const block = asset?.recipe?.rawAdapterSettings?.faceLikeness;
  return block && typeof block === "object" ? block : null;
}

// Cosine in [-1, 1] -> integer percent for the badge label. 0.92 -> "92%". Clamps to [0, 100]
// so a rare negative cosine never renders as "-12%".
function formatPercent(score) {
  const clamped = Math.min(1, Math.max(0, score));
  return `${Math.round(clamped * 100)}%`;
}

// Build the tooltip text: raw cosine + method + which reference it was scored against.
function buildTooltip({ band, descriptor, score, detected, sourceLabel }) {
  const lines = [descriptor?.label ?? "Frontal-identity confidence"];
  if (descriptor?.description) {
    lines.push(descriptor.description);
  }
  if (band !== LIKENESS_BAND.NA && typeof score === "number" && Number.isFinite(score)) {
    lines.push(`Cosine: ${score.toFixed(3)}`);
  }
  lines.push(`Method: ${LIKENESS_METHOD_LABEL}`);
  if (sourceLabel) {
    lines.push(`Scored against: ${sourceLabel}`);
  }
  // Reinforce that this is identity confidence, not overall quality — once, at the end.
  lines.push("Measures frontal-identity confidence, not overall image quality.");
  void detected;
  return lines.join("\n");
}

// `faceLikeness` may be passed directly, or inferred from `asset`. `sourceLabel` is an optional
// pre-resolved human name for the reference (sourceAssetId) — falls back to the raw id.
export function LikenessBadge({ asset = null, faceLikeness = null, sourceLabel = null }) {
  const block = faceLikeness ?? assetFaceLikeness(asset);

  // Absent block (legacy / unscored): render nothing — no badge, no layout box.
  if (!block) {
    return null;
  }

  const band = classifyLikeness(block);
  const descriptor = likenessBand(band);
  const score = typeof block.score === "number" ? block.score : null;
  const resolvedSource = sourceLabel || block.sourceAssetId || null;
  const tooltip = buildTooltip({
    band,
    descriptor,
    score,
    detected: block.detected,
    sourceLabel: resolvedSource,
  });

  // N/A: neutral chip, explanatory copy — never a low number coloured red.
  if (band === LIKENESS_BAND.NA) {
    return (
      <span
        className="likeness-badge likeness-badge--na"
        data-band={LIKENESS_BAND.NA}
        title={tooltip}
        aria-label={`Frontal-identity confidence: no frontal face to score (${LIKENESS_METHOD_LABEL})`}
      >
        <span className="likeness-badge__value" aria-hidden="true">
          —
        </span>
        <span className="likeness-badge__note">No frontal face to score</span>
      </span>
    );
  }

  return (
    <span
      className={`likeness-badge likeness-badge--${band}`}
      data-band={band}
      data-method={LIKENESS_METHOD}
      title={tooltip}
      aria-label={`Frontal-identity confidence ${formatPercent(score ?? 0)} — ${descriptor?.label ?? band} (${LIKENESS_METHOD_LABEL})`}
    >
      <span className="likeness-badge__value">{formatPercent(score ?? 0)}</span>
    </span>
  );
}

export default LikenessBadge;
