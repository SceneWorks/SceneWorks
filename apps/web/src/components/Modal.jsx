import React, { useEffect, useRef } from "react";

// Shared modal primitive: a backdrop that closes on outside mousedown, a
// role="dialog" container that closes on Escape, and focus moved into the
// dialog on mount. Extracted so every overlay behaves consistently for
// keyboard and pointer users.
export function Modal({ children, onClose, className, labelledBy, label }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        aria-label={label}
        aria-labelledby={labelledBy}
        aria-modal="true"
        className={className}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onClose();
          }
        }}
        onMouseDown={(event) => event.stopPropagation()}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );
}
