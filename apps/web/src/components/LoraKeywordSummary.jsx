import React from "react";

// Read-only display of a LoRA's trigger keywords (chips) + usage notes, shown
// between the family and the weight everywhere a LoRA is added to a generation
// (epic 10328). Renders nothing when the LoRA has neither, so pickers stay compact.
export function LoraKeywordSummary({ lora }) {
  const keywords = Array.isArray(lora?.triggerWords) ? lora.triggerWords : [];
  const notes = typeof lora?.notes === "string" ? lora.notes.trim() : "";
  if (!keywords.length && !notes) {
    return null;
  }
  return (
    <div className="lora-keyword-summary">
      {keywords.length ? (
        <div className="lora-keywords">
          {keywords.map((keyword) => (
            <span className="kw-chip" key={keyword}>
              {keyword}
            </span>
          ))}
        </div>
      ) : null}
      {notes ? (
        <p className="lora-notes" title={notes}>
          {notes}
        </p>
      ) : null}
    </div>
  );
}
