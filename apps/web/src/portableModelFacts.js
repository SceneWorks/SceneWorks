// Last served catalog rows are display-only while offline. Cross-field options,
// current installation, device choice and memory fit still require the server.
const KEY = "sceneworks:served-model-catalog:v1";
const MAX_ENTRIES = 200;
const MAX_BYTES = 1024 * 1024;

function storage() {
  try { return globalThis.localStorage; } catch { return null; }
}

function read() {
  try {
    const value = JSON.parse(storage()?.getItem(KEY) ?? "null");
    return value?.schemaVersion === 1 && Array.isArray(value.entries) ? value : { entries: [], savedAt: null };
  } catch { return { entries: [], savedAt: null }; }
}

export function cachedModelCatalog() {
  const { savedAt, entries } = read();
  return { savedAt, entries: entries.map(([, entry]) => entry) };
}

export function rememberModelCatalog(rows) {
  if (!Array.isArray(rows)) return false;
  const entries = rows.filter((row) => typeof row?.id === "string").slice(0, MAX_ENTRIES)
    .map(({ id, name, type, capabilities, installState, usable }) =>
      [id, { id, name, type, capabilities, installState, usable }]);
  return write(entries);
}

function write(entries) {
  try {
    const target = storage();
    if (!target) return false;
    const serialized = JSON.stringify({ schemaVersion: 1, savedAt: Date.now(), entries });
    if (new TextEncoder().encode(serialized).byteLength > MAX_BYTES) return false;
    target.setItem(KEY, serialized);
    return true;
  } catch { return false; }
}
