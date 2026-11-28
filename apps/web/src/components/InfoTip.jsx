import React, { useId, useRef, useState } from "react";
import { useDismissOnOutsideOrEscape } from "../hooks/useDismissOnOutsideOrEscape.js";
import "./InfoTip.css";

// InfoTip — a small (i) next to a field label whose explanation opens in a tooltip, so guidance
// text doesn't sit under the control as a block that breaks the settings grid.
//
// The tooltip opens on hover and on keyboard focus, and a click (or tap) pins it open until the
// next click elsewhere or Escape. Its text stays in the DOM while closed (visually hidden) and is
// wired to the button with aria-describedby, so screen readers read it with the icon. Code that
// reads a label's caption from its text should skip `.info-tip` (see `withoutInfoTips`).
// `tone="warn"` tints the icon for a caution (e.g. a tier that may run out of memory).
//
// Only for explanations. Errors, live status and anything with an action the user must see keep
// rendering inline.
// A node's visible caption text without any InfoTip inside it (the (i) and its tooltip text).
export function withoutInfoTips(node) {
  if (!node) return "";
  if (!node.querySelector?.(".info-tip")) return node.textContent ?? "";
  const copy = node.cloneNode(true);
  copy.querySelectorAll(".info-tip").forEach((tip) => tip.remove());
  return copy.textContent ?? "";
}

export function InfoTip({ children, label = "More info", tone, className }) {
  const tipId = useId();
  const [pinned, setPinned] = useState(false);
  const rootRef = useRef(null);
  useDismissOnOutsideOrEscape(rootRef, pinned, setPinned);

  if (children == null || children === false || children === "") return null;
  const classes = ["info-tip", tone === "warn" ? "warn" : "", pinned ? "pinned" : "", className].filter(Boolean).join(" ");
  return (
    <span className={classes} ref={rootRef}>
      <button
        aria-describedby={tipId}
        aria-expanded={pinned}
        aria-label={label}
        className="info-tip-button"
        onClick={(event) => {
          // Inside a <label>, don't let the click fall through to the labelled control.
          event.preventDefault();
          setPinned((open) => !open);
        }}
        type="button"
      >
        <svg aria-hidden="true" viewBox="0 0 16 16">
          <circle cx="8" cy="8" r="6.6" />
          <path d="M8 7.2v4M8 4.9h.01" />
        </svg>
      </button>
      {/* aria-hidden keeps the text out of the surrounding <label>'s accessible name; it is still
          read as the button's description, since aria-describedby includes hidden targets. */}
      <span aria-hidden="true" className="info-tip-bubble" id={tipId} role="tooltip">
        {children}
      </span>
    </span>
  );
}
