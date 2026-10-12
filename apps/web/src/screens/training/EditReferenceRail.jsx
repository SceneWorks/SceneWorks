import React from "react";

import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { moveReference, referenceOrdinalLabel } from "../../imageReferenceLimits.js";

function referenceButton(label, disabled, onClick, glyph) {
  return (
    <button aria-label={label} className="training-edit-ref-move" disabled={disabled} onClick={onClick} type="button">
      {glyph}
    </button>
  );
}

// The ordered reference rail of one instruction-edit training item (sc-24161, epic 24107).
//
// An edit pair is the item image (the edit TARGET the adapter learns to produce) + its caption
// (the edit INSTRUCTION) + 1..N ORDERED reference images. The rail reuses the Image Editor's
// ordered-reference affordances (sc-24113): every reference carries the 1-based ordinal the
// engine's template numbers it by ("Image 1" …), because an instruction names images by position
// ("put the hat from image 2 on image 1"); the arrows reorder in place; and Add is disabled at the
// model's reference cap, so the rail can never hold more than the trainer accepts.
export function EditReferenceRail({
  itemName,
  referenceIds = [],
  assetsById,
  cap = 0,
  onChange,
  onAdd,
}) {
  const atCap = referenceIds.length >= cap;
  return (
    <div className="training-edit-refs" aria-label={`References for ${itemName}`}>
      <div className="training-edit-refs-head">
        <span className="training-edit-refs-title">References</span>
        <span className="training-edit-refs-count">
          {referenceIds.length} / {cap}
        </span>
      </div>
      <ol className="training-edit-refs-list">
        {referenceIds.map((id, index) => {
          const asset = assetsById?.get(id);
          const ordinal = referenceOrdinalLabel(index);
          return (
            <li className="training-edit-ref" key={id}>
              <span className="training-edit-ref-thumb">
                {asset ? <AssetThumbnail asset={asset} /> : <span aria-hidden="true">?</span>}
              </span>
              <span className="training-edit-ref-ordinal">
                {ordinal}
              </span>
              <span className="training-edit-ref-actions">
                {referenceButton(`Move ${ordinal} earlier`, index === 0, () => onChange(moveReference(referenceIds, index, index - 1)), "‹")}
                {referenceButton(`Move ${ordinal} later`, index === referenceIds.length - 1, () => onChange(moveReference(referenceIds, index, index + 1)), "›")}
                {referenceButton(`Remove ${ordinal}`, false, () => onChange(referenceIds.filter((other) => other !== id)), "✕")}
              </span>
            </li>
          );
        })}
      </ol>
      <button
        aria-label={`Add references to ${itemName}`}
        className="secondary-action"
        disabled={atCap}
        onClick={onAdd}
        type="button"
      >
        + Reference
      </button>
    </div>
  );
}
