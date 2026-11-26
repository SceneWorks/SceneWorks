import React from "react";

export const PROMPT_CONTENT_OPTIONS = [
  ["identifyingCharacteristics", "Do not mention identifying characteristics (hair color or style, skin color, tattoos)"],
  ["clothingAndAccessories", "Do not mention clothing or accessories"],
  ["colorMode", "Do not mention if the image is color or black & white"],
  ["style", "Do not mention the style (illustration, painting, photograph)"],
];

export function PromptContentOptions({ value, onChange, disabled = false }) {
  return (
    <fieldset className="dataset-caption-options" disabled={disabled}>
      <legend>Prompt options (optional)</legend>
      {PROMPT_CONTENT_OPTIONS.map(([key, label]) => (
        <label className="training-toggle-line" key={key}>
          <input type="checkbox" checked={value.includes(key)} onChange={() => {
            onChange(value.includes(key) ? value.filter((item) => item !== key) : [...value, key]);
          }} />
          <span>{label}</span>
        </label>
      ))}
    </fieldset>
  );
}
