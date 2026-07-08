import React, { useRef, useState } from "react";

// Reusable trigger-keyword tag editor (epic 10328). Renders the current keywords
// as removable chips plus a type-to-add input (commit on Enter or comma), and an
// optional row of click-to-add suggestion chips — e.g. tags read from a LoRA's
// embedded `ss_tag_frequency` metadata. Serializes to a plain string array so the
// existing `triggerWords` contract is unchanged; the comma delimiter never leaks
// into a stored keyword because tokens are split before they become chips.
export function KeywordTagEditor({
  value,
  onChange,
  disabled = false,
  suggestions = [],
  suggestionsLabel = "From this LoRA:",
  placeholder = "Add a keyword…",
  inputId,
}) {
  const [draft, setDraft] = useState("");
  const inputRef = useRef(null);
  const keywords = Array.isArray(value) ? value : [];

  const hasKeyword = (candidate) =>
    keywords.some((keyword) => keyword.toLowerCase() === candidate.toLowerCase());

  const addKeyword = (raw) => {
    const token = raw.trim();
    setDraft("");
    if (!token || hasKeyword(token)) {
      return;
    }
    onChange([...keywords, token]);
  };

  const removeKeyword = (index) => {
    onChange(keywords.filter((_, position) => position !== index));
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addKeyword(draft);
    } else if (event.key === "Backspace" && draft === "" && keywords.length) {
      event.preventDefault();
      removeKeyword(keywords.length - 1);
    }
  };

  const availableSuggestions = suggestions.filter((tag) => !hasKeyword(tag));

  return (
    <div className="kw-editor">
      <div
        className={`kw-editor-field${disabled ? " disabled" : ""}`}
        onClick={() => inputRef.current?.focus()}
      >
        {keywords.map((keyword, index) => (
          <span className="kw-chip" key={`${keyword}-${index}`}>
            {keyword}
            <button
              aria-label={`Remove ${keyword}`}
              disabled={disabled}
              onClick={(event) => {
                event.stopPropagation();
                removeKeyword(index);
              }}
              type="button"
            >
              ×
            </button>
          </span>
        ))}
        <input
          className="kw-input"
          disabled={disabled}
          id={inputId}
          onBlur={() => addKeyword(draft)}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={keywords.length ? "" : placeholder}
          ref={inputRef}
          value={draft}
        />
      </div>
      {availableSuggestions.length ? (
        <div className="kw-suggestions">
          <span className="kw-suggestions-label">{suggestionsLabel}</span>
          {availableSuggestions.map((tag) => (
            <button
              className="kw-suggestion"
              disabled={disabled}
              key={tag}
              onClick={() => addKeyword(tag)}
              type="button"
            >
              + {tag}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
