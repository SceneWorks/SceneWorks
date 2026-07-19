// sc-13129 — Prompt composer: Style/Description template + directive-collision splice.
//
// Pure, side-effect-free composition of the outgoing `prompt` from a catalog-selected style
// text and the user's own prompt. This is the LAST wrap applied to a prompt: it runs AFTER
// preset prefix/suffix folding (see composePreset in presetUtils.js). By the time this sees
// `userPrompt`, any active presets have already been folded into it — so the folded preset
// text lives inside what becomes the `Description:` block. This module never applies presets
// itself; it only wraps an already-preset-folded prompt in the Style/Description template.
//
// Rules (R3/R5/R6/R7):
//  - No style selected → return userPrompt unchanged.
//  - Style selected, no directives → `Style: {styleText}\nDescription: {userPrompt}`.
//  - Directives detected → emit our `Style:` block first, keep the user's directive lines as
//    top-level siblings (never demoted under Description), and wrap only the free-text prose
//    remainder as `Description:`.
//  - User already has their own `Style:` line → MERGE: catalog style first, then the user's
//    own style content appended after `, ` in the same block (their words get the refinement
//    position). The `Style:` label is never duplicated.

// The recognized directive set that seeds the parser. A line only counts as a directive when
// its line-anchored key is one of these. Easily extensible: add a Capitalized key here.
export const STYLE_DIRECTIVE_KEYS = [
  "Style",
  "Description",
  "Setting",
  "Environment",
  "Angle",
  "Lighting",
  "Camera",
  "Mood",
  "Composition",
  "Negative",
];

const STYLE_KEY = "Style";
const DESCRIPTION_KEY = "Description";

// Longest recognized key we will treat as a directive (~20 chars, 1–3 short words). Guards the
// line-anchored detector against long "Sentence-case sentence: ..." prose that happens to start
// capitalized. Redundant with set membership today, but keeps the structural rule explicit.
const MAX_DIRECTIVE_KEY_LENGTH = 20;

const DIRECTIVE_KEY_SET = new Set(STYLE_DIRECTIVE_KEYS);

// Line-anchored directive shape: optional leading whitespace, then a Capitalized key of 1–3
// words, then a colon, then an optional remainder. A prose colon mid-line (e.g. "a portrait:
// dramatic") never matches because the key must sit at the line start and begin uppercase.
const DIRECTIVE_LINE_RE = /^\s*([A-Z][A-Za-z]*(?:\s+[A-Za-z]+){0,2}):(?:[ \t]+(.*\S))?[ \t]*$/;

/**
 * Split `text` into classified lines. Each entry is either a recognized directive
 * ({ type: "directive", key, content, raw }) or free prose ({ type: "prose", key: null,
 * content, raw }). Exported for unit testing.
 */
export function parseDirectiveLines(text) {
  const source = String(text ?? "");
  // Normalize newlines before classifying so CRLF (\r\n) and lone-CR (\r) line breaks split the
  // same as LF. Splitting on "\n" alone would keep a trailing "\r" on every non-final CRLF line,
  // which fails the anchored DIRECTIVE_LINE_RE, misclassifies the line as prose, and leaks a
  // literal "\r" into the composed Description.
  return source.split(/\r\n?|\n/).map((raw) => {
    const match = DIRECTIVE_LINE_RE.exec(raw);
    if (match) {
      const key = match[1];
      if (key.length <= MAX_DIRECTIVE_KEY_LENGTH && DIRECTIVE_KEY_SET.has(key)) {
        return { type: "directive", key, content: match[2] ?? "", raw: raw.trim() };
      }
    }
    return { type: "prose", key: null, content: raw, raw };
  });
}

/**
 * Compose the outgoing prompt from a catalog-selected style and the user's (already
 * preset-folded) prompt. Returns the composed string, or the untouched `userPrompt` when
 * `styleText` is empty/undefined.
 */
export function composeStyledPrompt({ styleText, userPrompt } = {}) {
  const style = typeof styleText === "string" ? styleText.trim() : "";
  if (!style) {
    return userPrompt;
  }

  const prompt = String(userPrompt ?? "");
  const parsed = parseDirectiveLines(prompt);

  // The user's own Style content (may be several lines) merges into our Style block; every
  // other directive line is preserved verbatim as a top-level sibling.
  const userStyleParts = [];
  const siblingDirectives = [];
  const proseLines = [];
  for (const line of parsed) {
    if (line.type === "directive") {
      if (line.key === STYLE_KEY) {
        if (line.content.trim()) {
          userStyleParts.push(line.content.trim());
        }
      } else {
        siblingDirectives.push(line.raw);
      }
    } else {
      proseLines.push(line.content);
    }
  }

  const styleContent = [style, ...userStyleParts].join(", ");
  const blocks = [`${STYLE_KEY}: ${styleContent}`];

  for (const sibling of siblingDirectives) {
    blocks.push(sibling);
  }

  const description = proseLines.join("\n").trim();
  if (description) {
    blocks.push(`${DESCRIPTION_KEY}: ${description}`);
  }

  return blocks.join("\n");
}
