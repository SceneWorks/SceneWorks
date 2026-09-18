import React, { useEffect, useRef } from "react";

// Shared modal primitive — verbatim from @sceneworks/ui. A backdrop that closes
// on outside mousedown, a role="dialog" that closes on Escape, and focus moved
// into the dialog on mount. Give it a title via `label` (or `labelledBy`).
export function Modal({ children, onClose, className, labelledBy, label }) {
  const dialogRef = useRef(null);
  useEffect(() => { dialogRef.current?.focus(); }, []);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div aria-label={label} aria-labelledby={labelledBy} aria-modal="true"
        className={className ? `modal-card ${className}` : "modal-card"}
        onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); onClose(); } }}
        onMouseDown={(e) => e.stopPropagation()} ref={dialogRef} role="dialog" tabIndex={-1}>
        {children}
      </div>
    </div>
  );
}

export default Modal;
