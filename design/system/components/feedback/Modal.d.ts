import React from "react";

export interface ModalProps {
  children: React.ReactNode;
  /** Called on Escape or backdrop click. */
  onClose: () => void;
  /** Extra class(es) on the dialog card (added to the base .modal-card). */
  className?: string;
  /** id of the element labelling the dialog. */
  labelledBy?: string;
  /** Accessible label when there is no visible title element. */
  label?: string;
}

/**
 * Accessible overlay dialog. Backdrop click + Escape close it; focus moves into
 * the dialog on mount.
 * @dsCard group="Components"
 */
export function Modal(props: ModalProps): React.ReactElement;
