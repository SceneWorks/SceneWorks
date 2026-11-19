// The longest image prompt a model's request may carry, mirroring
// `sceneworks_options::image_prompt_char_limit` (crates/sceneworks-options/src/lib.rs), which the
// server enforces. A flat 4,000 characters is far below what a long-window encoder reads
// (MiniMax-H3-Image reads 4,096 tokens, about 16,000 characters), so a model that declares
// `image.maxPromptTokens` gets that window in characters, between 4,000 and 16,384.

export const MAX_PROMPT_CHARS = 4000;
export const PROMPT_CHARS_PER_TOKEN = 4;
export const MAX_IMAGE_PROMPT_CHARS = 4096 * PROMPT_CHARS_PER_TOKEN;

export function imagePromptCharLimit(model) {
  const tokens = model?.image?.maxPromptTokens;
  if (!Number.isInteger(tokens) || tokens <= 0) return MAX_PROMPT_CHARS;
  return Math.min(MAX_IMAGE_PROMPT_CHARS, Math.max(MAX_PROMPT_CHARS, tokens * PROMPT_CHARS_PER_TOKEN));
}

// Unicode scalar values, matching Rust's `chars().count()` (`.length` counts UTF-16 units).
export function promptCharCount(text) {
  return [...(typeof text === "string" ? text : "")].length;
}

// The first `limit` characters of `text`, trailing whitespace dropped.
export function clipPromptChars(text, limit) {
  const chars = [...(typeof text === "string" ? text : "")];
  return chars.length <= limit ? text : chars.slice(0, Math.max(0, limit)).join("").trimEnd();
}

// Shorten a prompt so its COMPOSED form (preset fold, style and appearance wrapping added around
// it) fits `limit`. `compose` returns the composed string for a prompt. The wrapping is a fixed
// amount around the user's text, so cutting the user's text by the overflow fits the composition
// and keeps every style/appearance directive whole. Returns the prompt to send and, when it had to
// be shortened, the composed length it would otherwise have had.
export function fitPromptToLimit(prompt, limit, compose = (text) => text) {
  const composedLength = promptCharCount(compose(prompt));
  const overflow = composedLength - limit;
  if (overflow <= 0) return { prompt, clipped: false, composedLength };
  const keep = Math.max(1, promptCharCount(prompt) - overflow);
  return { prompt: clipPromptChars(prompt, keep), clipped: true, composedLength };
}

// The non-blocking batch note for prompts that were shortened to fit.
export function batchClipNotice({ items = [], total = items.length, limit, modelName }) {
  if (!total) return null;
  const shown = items.slice(0, 8).join(", ");
  const more = total > items.slice(0, 8).length ? ` and ${total - items.slice(0, 8).length} more` : "";
  const who = `${modelName || "this model"} accepts`;
  return `Batch item${total === 1 ? "" : "s"} ${shown}${more} ${total === 1 ? "is" : "are"} longer than ${who} (${limit.toLocaleString("en-US")} characters). ${total === 1 ? "It" : "They"} will run with the end cut off.`;
}
