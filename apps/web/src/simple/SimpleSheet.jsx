import { useDialogFocus } from "../components/Modal.jsx";
import React, { useRef } from "react";
import { getMediaTicket, subscribeMediaTicket } from "../api.js";
import { assetDisplayUrl } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";

// The Simple UI's one picker/overlay chrome (design handoff): a bottom sheet on phone
// (drag handle, ≤76% height, slide-up) and a centered 460px modal card on tablet/desktop
// (fade+pop). Which one renders is pure CSS off the root's `data-bp`, so this component
// is the same in both bands — only the handle is phone-only.
//
// Backdrop click and Escape both close, and focus moves into the panel on open so a
// keyboard user isn't left behind on the trigger.
export function SimpleSheet({ title, onClose, phone, children, labelId = "su-sheet-title" }) {
  const panelRef = useRef(null);

  const onKeyDown = useDialogFocus(panelRef, onClose);

  return (
    <>
      <button aria-label="Close" className="su-overlay" onClick={onClose} type="button" />
      <div
        aria-labelledby={labelId}
        aria-modal="true"
        className="su-sheet su-scroll"
        onKeyDown={(event) => {
          if (event.key === "Escape") event.stopPropagation();
          onKeyDown(event);
        }}
        ref={panelRef}
        role="dialog"
        tabIndex={-1}
      >
        {phone ? <div aria-hidden="true" className="su-sheet-handle" /> : null}
        {title ? (
          <h2 className="su-sheet-title" id={labelId}>
            {title}
          </h2>
        ) : null}
        {children}
      </div>
    </>
  );
}

// The picker body for a sheet descriptor. Three shapes, matching the design:
//   list  — model / voice / reference pickers (a tick marks the current value)
//   grid  — size & aspect (3-up tiles with a label + sub-label)
//   pills — duration (wrapping pills)
export function SimpleSheetOptions({ kind, options, onSelect }) {
  React.useSyncExternalStore(subscribeMediaTicket, getMediaTicket, getMediaTicket);
  if (kind === "grid") {
    return (
      <div className="su-option-grid">
        {options.map((option) => (
          <button
            aria-pressed={Boolean(option.active)}
            className={option.active ? "su-option-tile active" : "su-option-tile"}
            key={option.value}
            onClick={() => onSelect(option.value)}
            type="button"
          >
            <strong>{option.label}</strong>
            {option.sub ? <small>{option.sub}</small> : null}
          </button>
        ))}
      </div>
    );
  }

  if (kind === "pills") {
    return (
      <div className="su-option-pills">
        {options.map((option) => (
          <button
            aria-pressed={Boolean(option.active)}
            className={option.active ? "su-filter-pill active" : "su-filter-pill"}
            key={option.value}
            onClick={() => onSelect(option.value)}
            type="button"
          >
            {option.label}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="su-option-list">
      {options.map((option) => (
        <button
          aria-pressed={Boolean(option.active)}
          className={option.active ? "su-option-row active" : "su-option-row"}
          key={option.value}
          onClick={() => onSelect(option.value)}
          type="button"
        >
          {option.thumbnailAsset || option.thumbnail ? (
            <img alt="" referrerPolicy="no-referrer" src={option.thumbnailAsset ? assetDisplayUrl(option.thumbnailAsset) : option.thumbnail} />
          ) : null}
          <span>{option.label}</span>
          {option.active ? <Icon.Check size={16} /> : null}
        </button>
      ))}
    </div>
  );
}
