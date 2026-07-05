// Prompt-batch template engine (sc-9953, epic 9952 — Batch Prompt Processing).
// Pure, mode-agnostic logic: pull {{key}} placeholders out of a list of prompt
// templates, resolve them against a value map, and expand a batch to its concrete
// prompts via cross-product over multi-valued variables. React/DOM/network-free so
// the fan-out math is unit-tested in isolation; the authoring UI (slice 3) and the
// execution fan-out (slice 4) consume these functions.
//
// Design notes that the tests pin down:
//  - A key is `{{name}}`; surrounding whitespace is trimmed so `{{ name }}` ==
//    `{{name}}`. Any name is allowed (it just can't contain braces).
//  - Expansion is PER LINE: each prompt fans out only over the variables it
//    actually references. A variable referenced in some lines but not others (or
//    in none) does not multiply the lines that omit it into identical duplicate
//    renders. So cardinality() is a sum over lines, not one global product.
//  - A referenced key with no usable value is NOT silently dropped: it is skipped
//    for expansion (its placeholder stays literal) and surfaced via missingKeys()
//    so the caller (slice 5) can block the run.
//  - Cross-product ordering is deterministic: prompts vary slowest (outer loop),
//    then variables in array order, with the LAST variable varying fastest.
//  - Invariant the tests pin: cardinality(p, v, count) === expandBatch(p, v).length
//    × count (for a positive integer count). expandBatch's length is the job count;
//    cardinality is the image count once each job's `count` variations are applied.

// Fresh regex per call — a shared /g regex carries lastIndex state across
// matchAll/replace and would desync. `[^{}]` can't span across `}}`, so a lazy
// inner group is enough to isolate one placeholder.
const keyPattern = () => /\{\{([^{}]+?)\}\}/g;

// Normalize the raw prompt input to a list of non-blank template strings.
// Whitespace-only entries never produce a render (so "all blank" → 0 downstream).
function promptList(prompts) {
  if (!Array.isArray(prompts)) return [];
  return prompts
    .map((prompt) => (typeof prompt === "string" ? prompt : ""))
    .filter((prompt) => prompt.trim() !== "");
}

// A variable's usable values: trimmed, blanks dropped. Order is preserved and
// duplicates are kept (a repeat is treated as intent, not silently collapsed).
function variableValues(variable) {
  const values = Array.isArray(variable?.values) ? variable.values : [];
  return values
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .filter((value) => value !== "");
}

// The variables that actually drive expansion: referenced in the prompts, first
// entry wins per key, and carrying at least one usable value.
function effectiveVariables(prompts, variables) {
  const referenced = new Set(extractKeys(prompts));
  const list = Array.isArray(variables) ? variables : [];
  const seen = new Set();
  const result = [];
  for (const variable of list) {
    const key = typeof variable?.key === "string" ? variable.key.trim() : "";
    if (!key || !referenced.has(key) || seen.has(key)) continue;
    const values = variableValues(variable);
    if (values.length === 0) continue; // unfilled — surfaced by missingKeys(), skipped here
    seen.add(key);
    result.push({ key, values });
  }
  return result;
}

// Cartesian product of value lists. Empty input → `[[]]` (one empty combination),
// so a line with no active variables still yields exactly one render. The last
// list varies fastest, matching the documented ordering.
function cartesian(lists) {
  return lists.reduce(
    (combos, list) => combos.flatMap((combo) => list.map((value) => [...combo, value])),
    [[]],
  );
}

// Whether a single prompt line references a given (already-trimmed) key.
function promptReferences(prompt, key) {
  for (const match of prompt.matchAll(keyPattern())) {
    if (match[1].trim() === key) return true;
  }
  return false;
}

// Unique {{key}} names referenced across the prompts, trimmed, in first-seen order.
export function extractKeys(prompts) {
  const seen = new Set();
  const keys = [];
  for (const prompt of promptList(prompts)) {
    for (const match of prompt.matchAll(keyPattern())) {
      const key = match[1].trim();
      if (key && !seen.has(key)) {
        seen.add(key);
        keys.push(key);
      }
    }
  }
  return keys;
}

// Substitute a single value map into one template. Keys absent from the map are
// left as their literal `{{key}}` placeholder (so an unfilled key stays visible in
// a preview rather than becoming an empty string).
export function resolvePrompt(template, valueMap) {
  if (typeof template !== "string") return "";
  const map = valueMap ?? {};
  return template.replace(keyPattern(), (whole, rawKey) => {
    const key = rawKey.trim();
    return Object.prototype.hasOwnProperty.call(map, key) ? String(map[key]) : whole;
  });
}

// Expand a batch to its concrete prompts. Each prompt line fans out over only the
// variables it references (their cross-product); a line with none yields exactly
// one render. Each entry carries the resolved `prompt` plus the `values` map that
// produced it. Ordering is deterministic (prompts slowest, last variable fastest).
export function expandBatch(prompts, variables) {
  const lines = promptList(prompts);
  const vars = effectiveVariables(prompts, variables);
  const out = [];
  for (const prompt of lines) {
    const lineVars = vars.filter((variable) => promptReferences(prompt, variable.key));
    const combos = cartesian(lineVars.map((variable) => variable.values));
    for (const combo of combos) {
      const values = {};
      lineVars.forEach((variable, index) => {
        values[variable.key] = combo[index];
      });
      out.push({ prompt: resolvePrompt(prompt, values), values });
    }
  }
  return out;
}

// Number of images a batch will queue: Σ over prompt lines of Π(value counts of the
// variables that line references), times `count`. Computed directly from the
// factors — never materializes the expansion — so it is safe to call live on every
// keystroke even when the product is enormous.
export function cardinality(prompts, variables, count = 1) {
  const lines = promptList(prompts);
  if (lines.length === 0) return 0;
  const vars = effectiveVariables(prompts, variables);
  const jobs = lines.reduce((total, prompt) => {
    const product = vars.reduce(
      (acc, variable) => (promptReferences(prompt, variable.key) ? acc * variable.values.length : acc),
      1,
    );
    return total + product;
  }, 0);
  const n = Number(count);
  const multiplier = Number.isFinite(n) && n > 0 ? Math.floor(n) : 1;
  return jobs * multiplier;
}

// Referenced keys that have no usable value — the batch cannot run until these are
// filled. Slice 5 uses this to block the run and name the offending key(s).
export function missingKeys(prompts, variables) {
  const filled = new Set(effectiveVariables(prompts, variables).map((variable) => variable.key));
  return extractKeys(prompts).filter((key) => !filled.has(key));
}
