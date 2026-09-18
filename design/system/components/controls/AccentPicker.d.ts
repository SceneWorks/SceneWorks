import React from "react";

export interface AccentPickerProps {
  /** Active accent id — one of teal|indigo|cobalt|violet|coral|amber|emerald. */
  accent: string;
  /** Called with the chosen accent id; set data-accent on <html> in response. */
  onChange: (id: string) => void;
}

/**
 * Topbar accent-color picker. A trigger swatch that opens the remaining accents.
 * @dsCard group="Components"
 */
export function AccentPicker(props: AccentPickerProps): React.ReactElement;
