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
// Renders nothing for an empty or single-item list: with one reference there is no order to show or
// change, and an empty rail is noise next to a picker that already says "none selected".
export function OrderedReferenceList({
  assetIds = [],
  onChange,
  // (id) => label, so each caller can name an asset the way its own screen does.
  labelFor,
  className = "",
}) {
  if (!Array.isArray(assetIds) || assetIds.length < 2) {
    return null;
  }
  const move = (from, to) => onChange(moveReference(assetIds, from, to));
  return (
    <ol className={`ordered-references ${className}`.trim()}>
      {assetIds.map((id, index) => (
        <li className="ordered-reference" key={id}>
          <span className="ordered-reference-ordinal">{referenceOrdinalLabel(index)}</span>
          <span className="ordered-reference-label">
            {typeof labelFor === "function" ? labelFor(id) : id}
          </span>
          <span className="ordered-reference-actions">
            <button
              aria-label={`Move ${referenceOrdinalLabel(index)} earlier`}
              className="secondary-action"
              disabled={index === 0}
              onClick={() => move(index, index - 1)}
              type="button"
            >
              ↑
            </button>
            <button
              aria-label={`Move ${referenceOrdinalLabel(index)} later`}
              className="secondary-action"
              disabled={index === assetIds.length - 1}
              onClick={() => move(index, index + 1)}
              type="button"
            >
              ↓
            </button>
          </span>
        </li>
      ))}
    </ol>
  );
}
