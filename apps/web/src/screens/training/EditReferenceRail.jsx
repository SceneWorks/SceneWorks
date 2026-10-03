import React from "react";

import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { moveReference, referenceOrdinalLabel } from "../../imageReferenceLimits.js";

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
    <div className="training-edit-refs" aria-label={`Reference images for ${itemName}`}>
      <div className="training-edit-refs-head">
        <span className="training-edit-refs-title">References</span>
        <span className="training-edit-refs-count" title="Ordered: the instruction names them by position">
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
              <span className="training-edit-ref-ordinal" title="References are sent in this order">
                {ordinal}
              </span>
              <span className="training-edit-ref-actions">
                <button
                  aria-label={`Move ${ordinal} of ${itemName} earlier`}
                  className="training-edit-ref-move"
                  disabled={index === 0}
                  onClick={() => onChange(moveReference(referenceIds, index, index - 1))}
                  type="button"
                >
                  ‹
                </button>
                <button
                  aria-label={`Move ${ordinal} of ${itemName} later`}
                  className="training-edit-ref-move"
                  disabled={index === referenceIds.length - 1}
                  onClick={() => onChange(moveReference(referenceIds, index, index + 1))}
                  type="button"
                >
                  ›
                </button>
                <button
                  aria-label={`Remove ${ordinal} of ${itemName}`}
                  className="training-edit-ref-move"
                  onClick={() => onChange(referenceIds.filter((other) => other !== id))}
                  type="button"
                >
                  ✕
                </button>
              </span>
            </li>
          );
        })}
      </ol>
      <button
        aria-label={`Add reference images to ${itemName}`}
        className="secondary-action"
        disabled={atCap}
        onClick={onAdd}
        title={atCap ? `At most ${cap} reference images per edit` : "Add the image(s) this edit starts from"}
        type="button"
      >
        + Reference
      </button>
    </div>
  );
}
