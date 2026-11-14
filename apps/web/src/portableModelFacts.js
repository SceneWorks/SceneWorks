// A model's last served catalog entry can keep pure option feedback usable offline.
// Cached facts never establish current installation, device or memory availability.
const KEY = "sceneworks:validator-model-facts:v1";
const MAX_ENTRIES = 12;
const MAX_BYTES = 1024 * 1024;

function storage() {
  try { return globalThis.localStorage; } catch { return null; }
}

function read() {
  try {
    const value = JSON.parse(storage()?.getItem(KEY) ?? "null");
    return value?.schemaVersion === 1 && Array.isArray(value.entries) ? value.entries : [];
  } catch { return []; }
}

export function cachedModelFacts(modelId) {
  const row = read().find(([id]) => id === modelId);
  return row?.[1] ?? null;
}

export function rememberModelFacts(modelId, entry) {
  if (!modelId || entry?.id !== modelId) return false;
  try {
    const target = storage();
    if (!target) return false;
    const next = [[modelId, entry], ...read().filter(([id]) => id !== modelId)].slice(0, MAX_ENTRIES);
    const serialized = JSON.stringify({ schemaVersion: 1, entries: next });
    if (new TextEncoder().encode(serialized).byteLength > MAX_BYTES) return false;
    target.setItem(KEY, serialized);
    return true;
  } catch { return false; }
}
