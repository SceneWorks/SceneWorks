import React, { useState } from "react";
import { StudioUpdateBadge } from "./StudioUpdateNotice.jsx";

// Every LoRA selection surface uses this list; callers own eligibility and caps.
export function LoraPickerList({ loras, onAdd, isDisabled = () => false, emptyMessage, variant = "studio", actionLabel = "Add" }) {
  const [query, setQuery] = useState("");
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filteredLoras = loras.filter((lora) => {
    const text = [lora.name, lora.id, lora.family, lora.scope, ...(lora.triggerWords ?? [])]
      .filter(Boolean).join(" ").toLowerCase();
    return terms.every((term) => text.includes(term));
  });
  const simple = variant === "simple";

  return (
    <div className="lora-search-list">
      <input
        aria-label="Search LoRAs"
        autoFocus
        className="lora-search-input"
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.preventDefault();
        }}
        placeholder="Search LoRAs…"
        type="search"
        value={query}
      />
      {filteredLoras.length ? (
        <div className={simple ? "su-lora-pick-list" : "lora-picker-list"}>
          {filteredLoras.map((lora) => {
            const disabled = isDisabled(lora);
            const scope = lora.scope ?? "global";
            return (
              <button
                className={simple ? "su-lora-pick-row" : "lora-pick-row"}
                disabled={disabled}
                key={lora.id}
                onClick={() => onAdd(lora)}
                type="button"
              >
                <span className={simple ? "su-lora-pick-meta" : "lora-slot-meta"}>
                  <strong>{lora.name ?? lora.id}<StudioUpdateBadge item={lora} /></strong>
                  <small>{lora.family ? `${scope} · ${lora.family}` : scope}</small>
                </span>
                <span className={simple ? "su-lora-pick-add" : "lora-pick-add"}>
                  {disabled ? "Limit reached" : actionLabel}
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className={simple ? "su-lora-pick-empty" : "lora-pick-empty"} role="status">
          {loras.length ? "No LoRAs match your search." : emptyMessage}
        </p>
      )}
    </div>
  );
}
