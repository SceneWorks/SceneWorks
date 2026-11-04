/** Canonical script JSON: compact JSON, arrays in input order, object keys ordered by
 * UTF-8 bytes (including integer-looking keys), no terminal newline. Inputs are JSON
 * values. This is the StarVector production-closure wire format, not RFC 8785.
 * Existing protocols must use their named compatibility format; changing serialization
 * is a protocol migration, never a reason to regenerate historical evidence hashes.
 */
export const compareUtf8Keys = (left, right) => Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));

function orderedJson(value, compare) {
  if (Array.isArray(value)) return `[${value.map((entry) => orderedJson(entry, compare)).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort(compare).map((key) => `${JSON.stringify(key)}:${orderedJson(value[key], compare)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function canonicalJson(value) {
  return orderedJson(value, compareUtf8Keys);
}

/** Historical StarVector terminal and LTX watchdog format: compact, UTF-16 key order.
 * Manual serialization preserves sorted integer-looking keys. */
export function legacyUtf16Json(value) {
  return orderedJson(value);
}

/** Historical memory/KV/Flux3 format: recursively sort UTF-16 keys, then construct
 * ordinary objects. JSON.stringify subsequently places array-index keys in numeric order.
 * Keep this normalization public for protocols that add domain separation or validation. */
export function legacySortedValue(value) {
  if (Array.isArray(value)) return value.map(legacySortedValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, legacySortedValue(value[key])]));
  }
  return value;
}

export function legacySortedJson(value, { space, newline = false } = {}) {
  const json = JSON.stringify(legacySortedValue(value), null, space);
  return newline ? `${json}\n` : json;
}
