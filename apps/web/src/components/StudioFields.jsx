import React from "react";
import { Icon } from "./Icons.jsx";
import { PromptGuideModal } from "./PromptGuideModal.jsx";

export function CharacterLookFields({ characters, characterId, lookId, onCharacterChange, onLookChange,
  characterPlaceholder = "No character", lookPlaceholder = "Default look" }) {
  const looks = characters.find((character) => character.id === characterId)?.looks ?? [];
  return <>
    <label>
      Character
      <select onChange={(event) => onCharacterChange(event.target.value)} value={characterId}>
        <option value="">{characterPlaceholder}</option>
        {characters.map((character) => <option key={character.id} value={character.id}>{character.name}</option>)}
      </select>
    </label>
    <label>
      Look
      <select onChange={(event) => onLookChange(event.target.value)} value={lookId}>
        <option value="">{lookPlaceholder}</option>
        {looks.map((look) => <option key={look.id} value={look.id}>{look.name}</option>)}
      </select>
    </label>
  </>;
}

export function GpuSelect({ options, value, onChange }) {
  return <label>
    GPU
    <select onChange={(event) => onChange(event.target.value)} value={value}>
      {options.map((gpu) => <option key={gpu} value={gpu}>{gpu === "auto" ? "Auto" : gpu}</option>)}
    </select>
  </label>;
}

export function StudioGuideButton({ onOpen, hint = false }) {
  return <button className={hint ? "prompt-hint-link" : "hero-link"} onClick={onOpen} type="button">
    {hint ? "Prompt guide →" : <><Icon.Book size={14} /> Prompt guide</>}
  </button>;
}

export function StudioPromptGuide({ open, guide, modelName, onClose }) {
  return open ? <PromptGuideModal guide={guide} modelName={modelName} onClose={onClose} /> : null;
}
