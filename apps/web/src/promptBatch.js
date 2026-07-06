// Prompt-batch template engine (sc-9953 / sc-9958, epic 9952 — Batch Prompt Processing).
// Pure, mode-agnostic logic: parse a prompt line into text + {{placeholder}} tokens and
// expand a batch to its concrete prompts. React/DOM/network-free so the fan-out math is
// unit-tested in isolation; the authoring UI and the run fan-out consume these functions.
//
// A `{{...}}` placeholder is one of three kinds (sc-9958):
//  - **named**  `{{name}}` — no `|`. Values come from the external variables list (the
//    per-key chip editors). The same key used twice in a line shares one value.
//  - **inline** `{{a|b|c}}` — has `|`. An anonymous, independent axis whose values are
//    the inline options. Each occurrence is its own axis (`{{a|b}} {{a|b}}` → 4 combos).
//  - **linked** `{{label:a|b|c}}` — a `label:` prefix + `|`. All same-label placeholders
//    in a line advance TOGETHER by index (zip), not cross-producted — so correlated sets
//    like pronouns stay grammatical: `{{p:he|she|they}} … {{p:his|her|their}}` → 3 renders,
//    not 9. Unequal-length same-label groups are clamped to the shortest for expansion and
//    surfaced by linkedGroupIssues() so the caller can block the run.
//
// Design invariants the tests pin:
//  - Expansion is PER LINE: each line fans out only over the axes it contains.
//  - A named key with no usable value is NOT silently dropped: its placeholder stays
//    literal and it is surfaced by missingKeys() (inline/linked are never "missing" —
//    their values are inline).
//  - Ordering is deterministic: prompts slowest (outer), then axes in first-seen order
//    with the LAST axis varying fastest.
//  - cardinality(p, v, count) === expandBatch(p, v).length × count (positive integer count).
//  - Escaping: a literal `|` can't appear in an inline option; use a named variable
//    (separate value box) for values that must contain a pipe.

// Fresh regex per call — a shared /g regex carries lastIndex state across matchAll and
// would desync. `[^{}]` can't span across `}}`, so a lazy inner group isolates one token.
const keyPattern = () => /\{\{([^{}]+?)\}\}/g;

// Normalize the raw prompt input to a list of non-blank template strings.
// Whitespace-only entries never produce a render (so "all blank" → 0 downstream).
function promptList(prompts) {
  if (!Array.isArray(prompts)) return [];
  return prompts
    .map((prompt) => (typeof prompt === "string" ? prompt : ""))
    .filter((prompt) => prompt.trim() !== "");
}

// Split inline-alternation options on `|`, trimming each (empty options are kept as
// valid empty choices, e.g. `{{|red}}` = "" or "red").
function splitOptions(text) {
  return text.split("|").map((option) => option.trim());
}

// Classify one `{{...}}` inner string into a placeholder descriptor.
function parsePlaceholder(inner) {
  const trimmed = inner.trim();
  const labelMatch = /^([A-Za-z0-9_-]+)\s*:\s*([\s\S]*)$/.exec(trimmed);
  if (labelMatch && labelMatch[2].includes("|")) {
    return { kind: "linked", label: labelMatch[1], options: splitOptions(labelMatch[2]) };
  }
  if (trimmed.includes("|")) {
    return { kind: "inline", options: splitOptions(trimmed) };
  }
  return { kind: "named", key: trimmed };
}

// Tokenize a line into literal-text and placeholder tokens. `{{}}` / `{{ }}` (empty
// inner) is left as literal text so a stray empty brace pair renders verbatim.
function tokenizeLine(line) {
  const tokens = [];
  let last = 0;
  for (const match of line.matchAll(keyPattern())) {
    if (match.index > last) tokens.push({ text: line.slice(last, match.index) });
    if (match[1].trim() === "") {
      tokens.push({ text: match[0] });
    } else {
      tokens.push({ ph: parsePlaceholder(match[1]) });
    }
    last = match.index + match[0].length;
  }
  if (last < line.length) tokens.push({ text: line.slice(last) });
  return tokens;
}

// Map of usable named-variable values: trimmed, blanks dropped, first entry wins per key.
function buildVariableMap(variables) {
  const map = new Map();
  for (const variable of Array.isArray(variables) ? variables : []) {
    const key = typeof variable?.key === "string" ? variable.key.trim() : "";
    if (!key || map.has(key)) continue;
    const values = (Array.isArray(variable?.values) ? variable.values : [])
      .map((value) => (typeof value === "string" ? value.trim() : ""))
      .filter((value) => value !== "");
    if (values.length) map.set(key, values);
  }
  return map;
}

// Build the independent axes for one line + the per-token resolution data. Each axis is
// `{ size }`; a combination is an index per axis. Named tokens resolve through a shared
// per-key axis, inline tokens each own an axis, linked tokens share one axis per label.
function linePlan(prompt, variableMap) {
  const tokens = tokenizeLine(prompt);
  const named = new Map(); // key -> { axis, values }
  const linked = new Map(); // label -> { axis, size }
  const axes = [];
  for (const token of tokens) {
    const ph = token.ph;
    if (!ph) continue;
    if (ph.kind === "named") {
      const values = variableMap.get(ph.key);
      if (values && !named.has(ph.key)) {
        named.set(ph.key, { axis: axes.length, values });
        axes.push({ size: values.length });
      }
    } else if (ph.kind === "linked") {
      const existing = linked.get(ph.label);
      if (!existing) {
        linked.set(ph.label, { axis: axes.length, size: ph.options.length });
        axes.push({ size: ph.options.length });
      } else {
        // Clamp a mismatched same-label group to the shortest; linkedGroupIssues() reports it.
        existing.size = Math.min(existing.size, ph.options.length);
        axes[existing.axis].size = existing.size;
      }
    } else {
      token.axis = axes.length; // inline: one independent axis per occurrence
      axes.push({ size: ph.options.length });
    }
  }
  return { tokens, named, linked, axes };
}

// Cartesian product of the axis sizes as index tuples. `[]` axes → `[[]]` (one empty
// combination), so a line with no axes still yields exactly one render. The last axis
// varies fastest, matching the documented ordering.
function axisCombos(axes) {
  return axes.reduce(
    (combos, axis) =>
      combos.flatMap((combo) => Array.from({ length: axis.size }, (_, i) => [...combo, i])),
    [[]],
  );
}

// Resolve one line for a combination (an index per axis). Named tokens with no axis
// (unfilled) stay literal; the returned `values` map carries the chosen named values.
function resolveLine(plan, combo) {
  let out = "";
  const values = {};
  for (const token of plan.tokens) {
    if (token.text !== undefined) {
      out += token.text;
      continue;
    }
    const ph = token.ph;
    if (ph.kind === "named") {
      const axis = plan.named.get(ph.key);
      if (!axis) {
        out += `{{${ph.key}}}`;
        continue;
      }
      const value = axis.values[combo[axis.axis]];
      out += value;
      values[ph.key] = value;
    } else if (ph.kind === "linked") {
      out += ph.options[combo[plan.linked.get(ph.label).axis]] ?? "";
    } else {
      out += ph.options[combo[token.axis]] ?? "";
    }
  }
  return { prompt: out, values };
}

// Turn the authoring textarea into a prompt-template array. Default is one prompt per
// line (blank lines ignored) — the common case of pasting a list. For multi-line prompts,
// a line that is exactly `---` is an explicit delimiter; once any `---` line is present
// the text switches to block mode (each `---`-separated block, trimmed, is one prompt).
export function splitPromptLines(text) {
  if (typeof text !== "string") return [];
  const lines = text.split(/\r?\n/);
  if (!lines.some((line) => line.trim() === "---")) {
    return lines.map((line) => line.trim()).filter((line) => line !== "");
  }
  const blocks = [];
  let current = [];
  for (const line of lines) {
    if (line.trim() === "---") {
      blocks.push(current.join("\n").trim());
      current = [];
    } else {
      current.push(line);
    }
  }
  blocks.push(current.join("\n").trim());
  return blocks.filter((block) => block !== "");
}

// Unique NAMED `{{key}}` names referenced across the prompts, in first-seen order. Inline
// (`{{a|b}}`) and linked (`{{p:a|b}}`) placeholders are NOT keys — they carry their own
// values inline — so they never surface a chip editor.
export function extractKeys(prompts) {
  const seen = new Set();
  const keys = [];
  for (const prompt of promptList(prompts)) {
    for (const token of tokenizeLine(prompt)) {
      if (token.ph?.kind === "named" && !seen.has(token.ph.key)) {
        seen.add(token.ph.key);
        keys.push(token.ph.key);
      }
    }
  }
  return keys;
}

// Substitute a single named-value map into one template (named `{{key}}` only). Keys
// absent from the map are left as their literal placeholder. Retained for callers that
// resolve a single named binding; batch expansion uses the token model above.
export function resolvePrompt(template, valueMap) {
  if (typeof template !== "string") return "";
  const map = valueMap ?? {};
  return template.replace(keyPattern(), (whole, rawKey) => {
    const key = rawKey.trim();
    return Object.prototype.hasOwnProperty.call(map, key) ? String(map[key]) : whole;
  });
}

// Expand a batch to its concrete prompts. Each prompt line fans out over its own axes
// (named cross-product × inline occurrences × linked groups zipped). Each entry carries
// the resolved `prompt` plus the `values` map of the named bindings that produced it.
export function expandBatch(prompts, variables) {
  const variableMap = buildVariableMap(variables);
  const out = [];
  for (const prompt of promptList(prompts)) {
    const plan = linePlan(prompt, variableMap);
    for (const combo of axisCombos(plan.axes)) {
      out.push(resolveLine(plan, combo));
    }
  }
  return out;
}

// The first resolved prompt (line 1, first choice of every axis) — for a live preview
// without materializing the whole expansion, which can be enormous with inline axes.
export function firstResolvedPrompt(prompts, variables) {
  const lines = promptList(prompts);
  if (!lines.length) return "";
  const plan = linePlan(lines[0], buildVariableMap(variables));
  return resolveLine(
    plan,
    plan.axes.map(() => 0),
  ).prompt;
}

// A prompt line may start with a `[WxH]` directive to render that prompt at its own size,
// overriding the studio Aspect (sc-10063): `[832x1216] a full-body portrait`. Only a numeric
// `[W x H]` at the very start counts — `[cinematic] …` stays part of the prompt. Returns the
// stripped `prompt` and the parsed `{width,height}` (or null when absent). Range is NOT checked
// here (the caller validates against the backend 256–4096 bounds).
const RESOLUTION_DIRECTIVE = /^\s*\[\s*(\d{2,5})\s*[x×X]\s*(\d{2,5})\s*\]\s*/;
export function parsePromptResolution(text) {
  if (typeof text !== "string") return { prompt: "", resolution: null };
  const match = RESOLUTION_DIRECTIVE.exec(text);
  if (!match) return { prompt: text, resolution: null };
  return {
    prompt: text.slice(match[0].length),
    resolution: { width: Number(match[1]), height: Number(match[2]) },
  };
}

// Number of images a batch will queue: Σ over lines of Π(axis sizes), times `count`.
// Computed from the factors — never materializes the expansion — so it is safe to call
// live on every keystroke even when the product is enormous.
export function cardinality(prompts, variables, count = 1) {
  const lines = promptList(prompts);
  if (lines.length === 0) return 0;
  const variableMap = buildVariableMap(variables);
  const jobs = lines.reduce((total, prompt) => {
    const plan = linePlan(prompt, variableMap);
    return total + plan.axes.reduce((product, axis) => product * axis.size, 1);
  }, 0);
  const n = Number(count);
  const multiplier = Number.isFinite(n) && n > 0 ? Math.floor(n) : 1;
  return jobs * multiplier;
}

// Named keys referenced with no usable value — the batch cannot run until these are
// filled. Inline/linked placeholders are never missing (their values are inline).
export function missingKeys(prompts, variables) {
  const filled = buildVariableMap(variables);
  const seen = new Set();
  const missing = [];
  for (const prompt of promptList(prompts)) {
    for (const token of tokenizeLine(prompt)) {
      if (token.ph?.kind === "named" && !filled.has(token.ph.key) && !seen.has(token.ph.key)) {
        seen.add(token.ph.key);
        missing.push(token.ph.key);
      }
    }
  }
  return missing;
}

// Linked-group length mismatches: a `label:` used in one line with differing option
// counts (e.g. `{{p:he|she|they}} {{p:his|her}}`) can't zip cleanly. Returns
// `[{ label, lengths }]` (lengths ascending) so the caller can block the run and explain.
export function linkedGroupIssues(prompts) {
  const issues = [];
  const reported = new Set();
  for (const prompt of promptList(prompts)) {
    const lengths = new Map();
    for (const token of tokenizeLine(prompt)) {
      if (token.ph?.kind === "linked") {
        if (!lengths.has(token.ph.label)) lengths.set(token.ph.label, new Set());
        lengths.get(token.ph.label).add(token.ph.options.length);
      }
    }
    for (const [label, set] of lengths) {
      if (set.size > 1 && !reported.has(label)) {
        reported.add(label);
        issues.push({ label, lengths: [...set].sort((a, b) => a - b) });
      }
    }
  }
  return issues;
}
