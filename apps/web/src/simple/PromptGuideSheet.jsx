import React from "react";
import { Icon } from "../components/Icons.jsx";
import { SimpleSheet } from "./SimpleSheet.jsx";

// Prompt guide (design handoff): four numbered tips plus a "Try" example, in the same
// sheet/modal chrome as the pickers. Deliberately short and model-agnostic — the full
// per-model Markdown guide (PromptGuideModal) stays in the advanced workspace.
export const PROMPT_GUIDE_TIPS = [
  {
    title: "Lead with the subject",
    body: "Say what it is first, then style, setting, lighting, and framing.",
  },
  {
    title: "Be concrete",
    body: "“Golden hour”, “35mm”, “low angle” beat vague adjectives like “beautiful”.",
  },
  {
    title: "Name the look",
    body:
      "Add camera & lens for photographic; medium (watercolor, cel) for illustration — or pick a Style preset.",
  },
  {
    title: "Describe what you want",
    body: "Focus on the positive; keep negatives short and only for real problems.",
  },
];

export const PROMPT_GUIDE_EXAMPLE =
  "“A lone lighthouse on a rocky cliff at golden hour, dramatic storm clouds, wide cinematic shot, 35mm, warm rim light.”";

export function PromptGuideSheet({ onClose, phone }) {
  return (
    <SimpleSheet labelId="su-guide-title" onClose={onClose} phone={phone}>
      <div className="su-sheet-head">
        <h2 className="su-sheet-title" id="su-guide-title">
          Prompt guide
        </h2>
        <button aria-label="Close" className="su-icon-btn" onClick={onClose} type="button">
          <Icon.Close size={15} />
        </button>
      </div>
      <p className="su-guide-intro">
        A few things that make a prompt land. Or tap <strong>Refine</strong> to auto-expand a short
        idea.
      </p>
      <div className="su-guide-tips">
        {PROMPT_GUIDE_TIPS.map((tip, index) => (
          <div className="su-guide-tip" key={tip.title}>
            <span aria-hidden="true" className="su-guide-num">
              {index + 1}
            </span>
            <div>
              <strong>{tip.title}</strong>
              <p>{tip.body}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="su-guide-try">
        <div className="su-guide-try-label">Try</div>
        <p>{PROMPT_GUIDE_EXAMPLE}</p>
      </div>
    </SimpleSheet>
  );
}
