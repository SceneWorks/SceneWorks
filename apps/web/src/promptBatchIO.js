// Prompt-batch export / import (sc-9954, epic 9952 — Batch Prompt Processing).
// Server persistence (see apps/rust-api/src/prompt_batches.rs) keeps batches in the
// user's SceneWorks install; this module is the portable file layer on top so a
// batch can be shared as a plain .json and re-imported elsewhere (an OSS-friendly
// "send someone your character-turnaround recipe"). Pure and DOM-free.
//
// The export carries only the authored content — name, prompt templates, variable
// definitions, and remembered defaults — never server-managed fields (id, scope,
// manifestPath, timestamps, archived), so an import is always a fresh create.

export const PROMPT_BATCH_EXPORT_VERSION = 1;
const EXPORT_MARKER = "sceneworksPromptBatch";

function normalizePrompts(value) {
  if (!Array.isArray(value)) return [];
  return value.filter((entry) => typeof entry === "string");
}

function normalizeVariables(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((variable) => {
      const key = typeof variable?.key === "string" ? variable.key : "";
      const values = Array.isArray(variable?.values)
        ? variable.values.filter((entry) => typeof entry === "string")
        : [];
      return { key, values };
    })
    .filter((variable) => variable.key !== "");
}

function normalizeLastValues(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const out = {};
  for (const [key, values] of Object.entries(value)) {
    if (Array.isArray(values)) {
      out[key] = values.filter((entry) => typeof entry === "string");
    }
  }
  return Object.keys(out).length ? out : undefined;
}

// Build the portable export object for a saved (or in-progress) batch. `lastValues`
// is included only when present so round-trips stay minimal.
export function toPromptBatchExport(batch) {
  const exported = {
    [EXPORT_MARKER]: PROMPT_BATCH_EXPORT_VERSION,
    name: typeof batch?.name === "string" ? batch.name : "",
    prompts: normalizePrompts(batch?.prompts),
    variables: normalizeVariables(batch?.variables),
  };
  const lastValues = normalizeLastValues(batch?.lastValues);
  if (lastValues) {
    exported.lastValues = lastValues;
  }
  return exported;
}

// Serialize a batch to the pretty-printed JSON string written to disk.
export function serializePromptBatchExport(batch) {
  return JSON.stringify(toPromptBatchExport(batch), null, 2);
}

// Parse + validate an imported batch (a parsed object OR a raw JSON string) into a
// clean create payload: { name, prompts, variables, lastValues? }. Throws a
// descriptive Error on malformed input so the caller can surface it to the user.
export function fromPromptBatchImport(input) {
  let data = input;
  if (typeof input === "string") {
    try {
      data = JSON.parse(input);
    } catch {
      throw new Error("Not a valid JSON file.");
    }
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("Prompt batch file must contain a JSON object.");
  }
  if (!("prompts" in data) && !(EXPORT_MARKER in data)) {
    throw new Error("This file is not a SceneWorks prompt batch.");
  }
  if ("prompts" in data && !Array.isArray(data.prompts)) {
    throw new Error("Prompt batch prompts must be an array.");
  }
  if ("variables" in data && !Array.isArray(data.variables)) {
    throw new Error("Prompt batch variables must be an array.");
  }
  const payload = {
    name: typeof data.name === "string" ? data.name.trim() : "",
    prompts: normalizePrompts(data.prompts),
    variables: normalizeVariables(data.variables),
  };
  const lastValues = normalizeLastValues(data.lastValues);
  if (lastValues) {
    payload.lastValues = lastValues;
  }
  return payload;
}
