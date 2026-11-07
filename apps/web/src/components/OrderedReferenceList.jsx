import React from "react";
import { moveReference, referenceOrdinalLabel } from "../imageReferenceLimits.js";

// The ordered-reference rail, shared by Image Studio and the Simple shell (sc-24113).
//
// # Why a list beside the picker rather than a change to the picker
//
// `AssetPickerField` is used by many screens for sets whose ORDER means nothing, and it renders a
// selection, not a sequence. Teaching it about order would push an ordering concept onto every
// caller that has none. This renders the order the picker's array already has, and is mounted only
// where order is semantic.
//
// # Why order is semantic here
//
// Qwen-Image 2.1's template NUMBERS the references (`<image1>` …) and its block-causal attention
// makes each one visible only to what follows, so swapping two is a DIFFERENT request. Two things
// follow, and this component exists for both:
//
//  * the ordinal is VISIBLE, because the prompt conventions for this family name images directly
//    ("use the second image as a mask") and a user who cannot see which one is second cannot write
//    that prompt;
//  * the order is EDITABLE in place, because the only alternative was to remove every reference and
//    re-add them in the right sequence.
//
// Ordering-only callers hide empty/single lists. An instance-removal owner keeps its rail mounted
// through 2 -> 1 -> 0 so the last attachment remains removable without changing selection modes.
export function OrderedReferenceList({
  assetIds = [],
  onChange,
  // (id) => label, so each caller can name an asset the way its own screen does.
  labelFor,
  className = "",
  // Layout controls bind attachment instances, including repeated assets, not asset ids.
  itemKeys,
  onMove,
  onRemove,
  firstOrdinal = 1,
}) {
  if (!Array.isArray(assetIds) || (assetIds.length < 2 && !onRemove)) {
    return null;
  }
  const move = (from, to) => onMove ? onMove(from, to) : onChange(moveReference(assetIds, from, to));
  return (
    <ol className={`ordered-references ${className}`.trim()}>
      {assetIds.map((id, index) => (
        <li className="ordered-reference" key={itemKeys?.[index] ?? `${id}:${index}`}>
          <span className="ordered-reference-ordinal">{referenceOrdinalLabel(index + firstOrdinal - 1)}</span>
          <span className="ordered-reference-label">
            {typeof labelFor === "function" ? labelFor(id) : id}
          </span>
          <span className="ordered-reference-actions">
            <button
              aria-label={`Move ${referenceOrdinalLabel(index + firstOrdinal - 1)} earlier`}
              className="secondary-action"
              disabled={index === 0}
              onClick={() => move(index, index - 1)}
              type="button"
            >
              ↑
            </button>
            <button
              aria-label={`Move ${referenceOrdinalLabel(index + firstOrdinal - 1)} later`}
              className="secondary-action"
              disabled={index === assetIds.length - 1}
              onClick={() => move(index, index + 1)}
              type="button"
            >
              ↓
            </button>
            {onRemove ? <button aria-label={`Remove ${referenceOrdinalLabel(index)}`}
              className="secondary-action" type="button" onClick={() => onRemove(index)}>Remove</button> : null}
          </span>
        </li>
      ))}
    </ol>
  );
}
